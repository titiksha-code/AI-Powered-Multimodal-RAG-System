"""
src/retrieval/vector_store.py
==============================
Vector indexing + hybrid (dense + keyword) retrieval layer.

Supports two interchangeable backends behind a single interface:
    - Qdrant  (production-grade, used by default)
    - Chroma  (lightweight, zero-infra, good for local dev / demos)

Two collections are maintained per backend: one for text/table chunks
(embedded with a sentence-transformer) and one for image chunks
(embedded with a CLIP-style vision encoder), so cross-modal queries can
be issued against either or both and merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.ingestion.parser import Chunk

BackendType = Literal["qdrant", "chroma"]


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    modality: str
    score: float
    page_number: int
    source_path: str
    metadata: dict


class EmbeddingEngine:
    """Wraps text and vision embedding models behind one clean API."""

    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vision_model_name: str = "ViT-B-32",
        vision_pretrained: str = "laion2b_s34b_b79k",
    ):
        logger.info(f"Loading text embedding model: {text_model_name}")
        self.text_model = SentenceTransformer(text_model_name)

        logger.info(f"Loading vision embedding model: {vision_model_name}")
        import open_clip  # imported lazily - heavy dependency

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            vision_model_name, pretrained=vision_pretrained
        )
        self.clip_model.eval()

    def embed_text(self, texts: list[str]) -> np.ndarray:
        return self.text_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_image(self, images: list["Any"]) -> np.ndarray:
        import torch

        with torch.no_grad():
            batch = torch.stack([self.clip_preprocess(img) for img in images])
            features = self.clip_model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def embed_query(self, query: str) -> np.ndarray:
        """Text query embedded in the text space (used for text/table search)."""
        return self.embed_text([query])[0]

    def embed_query_for_image_search(self, query: str) -> np.ndarray:
        """Text query embedded in CLIP's joint text/image space (used for image search)."""
        import open_clip
        import torch

        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer([query])
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_overlap_score(query: str, content: str) -> float:
    """Cheap, dependency-free sparse/keyword scoring (Jaccard overlap).

    Serves as the "sparse" side of hybrid search when a full BM25 index
    isn't provisioned; swap for `rank_bm25` in a production deployment.
    """
    q_tokens, c_tokens = _tokenize(query), _tokenize(content)
    if not q_tokens or not c_tokens:
        return 0.0
    intersection = q_tokens & c_tokens
    union = q_tokens | c_tokens
    return len(intersection) / len(union)


class VectorStore:
    """Unified hybrid retrieval interface over Qdrant or Chroma."""

    def __init__(
        self,
        backend: BackendType = "qdrant",
        host: str = "localhost",
        port: int = 6333,
        collection_text: str = "kb_text_chunks",
        collection_image: str = "kb_image_chunks",
        embedding_engine: EmbeddingEngine | None = None,
        hybrid_alpha: float = 0.6,
    ):
        self.backend = backend
        self.collection_text = collection_text
        self.collection_image = collection_image
        self.embedder = embedding_engine or EmbeddingEngine()
        self.hybrid_alpha = hybrid_alpha  # weight on dense score; (1-alpha) on keyword score

        # In-memory content cache for keyword/BM25-style re-scoring,
        # since Qdrant/Chroma payload lookups round-trip per point.
        self._content_cache: dict[str, str] = {}

        if backend == "qdrant":
            self._init_qdrant(host, port)
        elif backend == "chroma":
            self._init_chroma()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ------------------------------------------------------------------ #
    # Backend initialization
    # ------------------------------------------------------------------ #
    def _init_qdrant(self, host: str, port: int) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(host=host, port=port)

        for name, dim in (
            (self.collection_text, self.embedder.text_model.get_sentence_embedding_dimension()),
            (self.collection_image, 512),  # ViT-B-32 output dim
        ):
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection '{name}' (dim={dim})")

    def _init_chroma(self) -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.text_collection = self.client.get_or_create_collection(self.collection_text)
        self.image_collection = self.client.get_or_create_collection(self.collection_image)

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index_chunks(self, chunks: list[Chunk]) -> None:
        text_chunks = [c for c in chunks if c.modality in ("text", "table")]
        image_chunks = [c for c in chunks if c.modality == "image"]

        if text_chunks:
            self._index_text_chunks(text_chunks)
        if image_chunks:
            self._index_image_chunks(image_chunks)

        logger.success(
            f"Indexed {len(text_chunks)} text/table chunks and {len(image_chunks)} image chunks"
        )

    def _index_text_chunks(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.embed_text([c.content for c in chunks])
        for c in chunks:
            self._content_cache[c.chunk_id] = c.content

        if self.backend == "qdrant":
            from qdrant_client.models import PointStruct

            points = [
                PointStruct(
                    id=self._to_point_id(c.chunk_id),
                    vector=vectors[i].tolist(),
                    payload={**c.to_dict()},
                )
                for i, c in enumerate(chunks)
            ]
            self.client.upsert(collection_name=self.collection_text, points=points)
        else:  # chroma
            self.text_collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=vectors.tolist(),
                documents=[c.content for c in chunks],
                metadatas=[c.to_dict() for c in chunks],
            )

    def _index_image_chunks(self, chunks: list[Chunk]) -> None:
        from io import BytesIO

        from PIL import Image

        pil_images = []
        valid_chunks = []
        for c in chunks:
            if not c.image_bytes:
                continue
            try:
                pil_images.append(Image.open(BytesIO(c.image_bytes)).convert("RGB"))
                valid_chunks.append(c)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Skipping unreadable image chunk {c.chunk_id}: {exc}")

        if not valid_chunks:
            return

        vectors = self.embedder.embed_image(pil_images)
        for c in valid_chunks:
            self._content_cache[c.chunk_id] = c.content

        if self.backend == "qdrant":
            from qdrant_client.models import PointStruct

            points = [
                PointStruct(
                    id=self._to_point_id(c.chunk_id),
                    vector=vectors[i].tolist(),
                    payload={**c.to_dict()},
                )
                for i, c in enumerate(valid_chunks)
            ]
            self.client.upsert(collection_name=self.collection_image, points=points)
        else:  # chroma
            self.image_collection.upsert(
                ids=[c.chunk_id for c in valid_chunks],
                embeddings=vectors.tolist(),
                documents=[c.content for c in valid_chunks],
                metadatas=[c.to_dict() for c in valid_chunks],
            )

    @staticmethod
    def _to_point_id(chunk_id: str) -> int:
        """Qdrant point IDs must be int or UUID; derive a stable int from our hash id."""
        return int(chunk_id, 16) % (2**63)

    # ------------------------------------------------------------------ #
    # Hybrid retrieval
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int = 5,
        modalities: tuple[str, ...] = ("text", "table", "image"),
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []

        if any(m in modalities for m in ("text", "table")):
            results.extend(self._search_text(query, top_k))
        if "image" in modalities:
            results.extend(self._search_image(query, top_k))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _search_text(self, query: str, top_k: int) -> list[RetrievalResult]:
        dense_vec = self.embedder.embed_query(query)
        raw_hits = self._raw_search(self.collection_text, dense_vec, top_k * 3, is_image=False)
        return self._hybrid_rescore(query, raw_hits, top_k)

    def _search_image(self, query: str, top_k: int) -> list[RetrievalResult]:
        dense_vec = self.embedder.embed_query_for_image_search(query)
        raw_hits = self._raw_search(self.collection_image, dense_vec, top_k * 3, is_image=True)
        return self._hybrid_rescore(query, raw_hits, top_k)

    def _raw_search(
        self, collection: str, vector: np.ndarray, top_k: int, is_image: bool
    ) -> list[dict]:
        if self.backend == "qdrant":
            hits = self.client.search(
                collection_name=collection, query_vector=vector.tolist(), limit=top_k
            )
            return [{"score": h.score, **h.payload} for h in hits]
        else:  # chroma
            coll = self.image_collection if is_image else self.text_collection
            res = coll.query(query_embeddings=[vector.tolist()], n_results=top_k)
            hits = []
            for i in range(len(res["ids"][0])):
                meta = res["metadatas"][0][i]
                # Chroma returns squared L2 distance by default; convert to a similarity-ish score
                dist = res["distances"][0][i]
                hits.append({"score": 1.0 / (1.0 + dist), **meta})
            return hits

    def _hybrid_rescore(
        self, query: str, raw_hits: list[dict], top_k: int
    ) -> list[RetrievalResult]:
        rescored: list[RetrievalResult] = []
        for hit in raw_hits:
            dense_score = float(hit.get("score", 0.0))
            content = hit.get("content", "")
            keyword_score = _keyword_overlap_score(query, content)
            combined = self.hybrid_alpha * dense_score + (1 - self.hybrid_alpha) * keyword_score

            rescored.append(
                RetrievalResult(
                    chunk_id=hit.get("chunk_id", ""),
                    content=content,
                    modality=hit.get("modality", "text"),
                    score=round(combined, 4),
                    page_number=hit.get("page_number", 1),
                    source_path=hit.get("source_path", ""),
                    metadata=hit.get("metadata", {}),
                )
            )

        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
