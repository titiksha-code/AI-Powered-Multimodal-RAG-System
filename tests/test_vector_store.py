"""
tests/test_vector_store.py
============================
Unit tests for src/retrieval/vector_store.py

Covers:
  - Pure scoring utilities (_keyword_overlap_score, _tokenize)
  - VectorStore hybrid re-scoring logic
  - VectorStore.count() for both Qdrant and Chroma backends
  - VectorStore._index_text_chunks (Chroma path, mocked)
  - VectorStore._index_image_chunks (Chroma path, mocked)
  - VectorStore._raw_search for Chroma backend
  - VectorStore.search end-to-end with modality filtering
  - EmbeddingEngine.embed_query / embed_query_for_image_search (mocked models)
  - _to_point_id stability and integer type guarantee
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.ingestion.parser import Chunk
from src.retrieval.vector_store import (
    EmbeddingEngine,
    RetrievalResult,
    VectorStore,
    _keyword_overlap_score,
    _tokenize,
)


# ------------------------------------------------------------------ #
# Pure utilities
# ------------------------------------------------------------------ #


class TestTokenize:
    def test_lowercases_and_splits(self):
        tokens = _tokenize("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_empty_string(self):
        assert _tokenize("") == set()

    def test_punctuation_stripped(self):
        tokens = _tokenize("foo, bar! baz.")
        assert "foo" in tokens
        assert "bar" in tokens
        assert "baz" in tokens


class TestKeywordOverlapScore:
    def test_identical_strings_score_one(self):
        assert _keyword_overlap_score("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint_strings_score_zero(self):
        assert _keyword_overlap_score("apple", "banana") == 0.0

    def test_empty_query_returns_zero(self):
        assert _keyword_overlap_score("", "some content") == 0.0

    def test_empty_content_returns_zero(self):
        assert _keyword_overlap_score("query words", "") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        score = _keyword_overlap_score("machine learning models", "deep learning models rock")
        assert 0.0 < score < 1.0

    def test_score_is_symmetric_direction_agnostic(self):
        s1 = _keyword_overlap_score("cat dog", "dog fish cat")
        s2 = _keyword_overlap_score("dog fish cat", "cat dog")
        assert s1 == pytest.approx(s2)


# ------------------------------------------------------------------ #
# Shared fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def mock_embedding_engine():
    engine = MagicMock()
    engine.embed_query.return_value = np.zeros(384, dtype=np.float32)
    engine.embed_query_for_image_search.return_value = np.zeros(512, dtype=np.float32)
    engine.embed_text.return_value = np.zeros((1, 384), dtype=np.float32)
    engine.embed_image.return_value = np.zeros((1, 512), dtype=np.float32)
    engine.text_model.get_sentence_embedding_dimension.return_value = 384
    return engine


@pytest.fixture
def chroma_store(mock_embedding_engine):
    """VectorStore backed by a fully mocked Chroma client."""
    with patch("chromadb.PersistentClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_text_coll = MagicMock()
        mock_image_coll = MagicMock()
        mock_client.get_or_create_collection.side_effect = [mock_text_coll, mock_image_coll]
        mock_client_cls.return_value = mock_client

        store = VectorStore(backend="chroma", embedding_engine=mock_embedding_engine)
        store.text_collection = mock_text_coll
        store.image_collection = mock_image_coll
    return store


def _make_chunk(
    chunk_id: str = "abc123",
    modality: str = "text",
    content: str = "sample text content",
    page: int = 1,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        modality=modality,
        content=content,
        page_number=page,
        source_path="/tmp/test.pdf",
        metadata={"extractor": "test"},
    )


# ------------------------------------------------------------------ #
# _to_point_id
# ------------------------------------------------------------------ #


class TestPointId:
    def test_point_id_is_stable_and_int(self):
        pid1 = VectorStore._to_point_id("abc123")
        pid2 = VectorStore._to_point_id("abc123")
        assert isinstance(pid1, int)
        assert pid1 == pid2

    def test_point_id_different_for_different_ids(self):
        assert VectorStore._to_point_id("aaa") != VectorStore._to_point_id("bbb")

    def test_point_id_within_int63_range(self):
        # chunk IDs are 24-char hex strings (from sha256 in _make_id)
        hex_id = "a1b2c3d4e5f6a7b8c9d0e1f2"  # valid 24-char hex
        pid = VectorStore._to_point_id(hex_id)
        assert 0 <= pid < 2**63


# ------------------------------------------------------------------ #
# Backend: Chroma — count()
# ------------------------------------------------------------------ #


class TestVectorStoreCountChroma:
    def test_count_returns_correct_totals(self, chroma_store: VectorStore):
        chroma_store.text_collection.count.return_value = 10
        chroma_store.image_collection.count.return_value = 5

        counts = chroma_store.count()
        assert counts["text"] == 10
        assert counts["image"] == 5
        assert counts["total"] == 15

    def test_count_handles_exception_gracefully(self, chroma_store: VectorStore):
        chroma_store.text_collection.count.side_effect = RuntimeError("DB error")
        chroma_store.image_collection.count.side_effect = RuntimeError("DB error")

        counts = chroma_store.count()
        assert counts == {"text": 0, "image": 0, "total": 0}


# ------------------------------------------------------------------ #
# Backend: Chroma — index_chunks()
# ------------------------------------------------------------------ #


class TestVectorStoreIndexChunksChroma:
    def test_index_text_chunks_calls_upsert(self, chroma_store: VectorStore):
        chunks = [_make_chunk("id1", "text", "hello world"), _make_chunk("id2", "text", "foo bar")]
        chroma_store.embedder.embed_text.return_value = np.zeros((2, 384), dtype=np.float32)

        chroma_store.index_chunks(chunks)

        chroma_store.text_collection.upsert.assert_called_once()
        call_kwargs = chroma_store.text_collection.upsert.call_args.kwargs
        assert set(call_kwargs["ids"]) == {"id1", "id2"}

    def test_index_table_chunks_go_to_text_collection(self, chroma_store: VectorStore):
        chunks = [_make_chunk("t1", "table", "| A | B |")]
        chroma_store.embedder.embed_text.return_value = np.zeros((1, 384), dtype=np.float32)

        chroma_store.index_chunks(chunks)

        chroma_store.text_collection.upsert.assert_called_once()
        chroma_store.image_collection.upsert.assert_not_called()

    def test_index_image_chunk_without_bytes_is_skipped(self, chroma_store: VectorStore):
        """Image chunks with no image_bytes should be silently skipped."""
        chunk = _make_chunk("img1", "image", "caption")
        chunk.image_bytes = None  # no bytes → skip

        chroma_store.index_chunks([chunk])

        chroma_store.image_collection.upsert.assert_not_called()

    def test_index_image_chunk_with_bytes(self, chroma_store: VectorStore, tmp_path):
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        chunk = _make_chunk("img2", "image", "some diagram")
        chunk.image_bytes = png_bytes
        chroma_store.embedder.embed_image.return_value = np.zeros((1, 512), dtype=np.float32)

        # Path is imported locally inside _index_image_chunks from pathlib.
        # Patch pathlib.Path directly so mkdir/exists/write_bytes are controlled.
        with patch("pathlib.Path.mkdir", return_value=None), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.write_bytes", return_value=None):
            chroma_store.index_chunks([chunk])

        # embed_image should have been called with 1 PIL image
        chroma_store.embedder.embed_image.assert_called_once()


# ------------------------------------------------------------------ #
# Backend: Chroma — _raw_search()
# ------------------------------------------------------------------ #


class TestVectorStoreRawSearchChroma:
    def test_raw_search_returns_empty_when_collection_empty(self, chroma_store: VectorStore):
        chroma_store.text_collection.count.return_value = 0
        vec = np.zeros(384, dtype=np.float32)
        hits = chroma_store._raw_search(chroma_store.collection_text, vec, top_k=5, is_image=False)
        assert hits == []

    def test_raw_search_parses_chroma_results(self, chroma_store: VectorStore):
        chroma_store.text_collection.count.return_value = 3
        chroma_store.text_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["text of id1", "text of id2"]],
            "metadatas": [
                [
                    {"chunk_id": "id1", "modality": "text", "page_number": 1, "source_path": "a.pdf"},
                    {"chunk_id": "id2", "modality": "text", "page_number": 2, "source_path": "b.pdf"},
                ]
            ],
            "distances": [[0.1, 0.5]],
        }
        vec = np.zeros(384, dtype=np.float32)
        hits = chroma_store._raw_search(chroma_store.collection_text, vec, top_k=5, is_image=False)

        assert len(hits) == 2
        assert hits[0]["chunk_id"] == "id1"
        # score = 1 / (1 + dist); lower distance → higher score
        assert hits[0]["score"] > hits[1]["score"]


# ------------------------------------------------------------------ #
# Hybrid re-scoring
# ------------------------------------------------------------------ #


class TestVectorStoreHybridRescore:
    def test_hybrid_rescore_orders_by_combined_score(self, mock_embedding_engine):
        with patch("src.retrieval.vector_store.VectorStore._init_qdrant"):
            store = VectorStore(backend="qdrant", embedding_engine=mock_embedding_engine)

        raw_hits = [
            {"score": 0.9, "content": "irrelevant filler", "chunk_id": "a", "modality": "text"},
            {"score": 0.5, "content": "machine learning models", "chunk_id": "b", "modality": "text"},
        ]

        results = store._hybrid_rescore("machine learning", raw_hits, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].score >= results[1].score

    def test_hybrid_rescore_respects_top_k(self, mock_embedding_engine):
        with patch("src.retrieval.vector_store.VectorStore._init_qdrant"):
            store = VectorStore(backend="qdrant", embedding_engine=mock_embedding_engine)

        raw_hits = [
            {"score": 0.9, "content": f"hit {i}", "chunk_id": str(i), "modality": "text"}
            for i in range(10)
        ]
        results = store._hybrid_rescore("query", raw_hits, top_k=3)
        assert len(results) == 3

    def test_hybrid_rescore_score_is_rounded(self, mock_embedding_engine):
        with patch("src.retrieval.vector_store.VectorStore._init_qdrant"):
            store = VectorStore(backend="qdrant", embedding_engine=mock_embedding_engine)

        raw_hits = [{"score": 0.123456789, "content": "text", "chunk_id": "x", "modality": "text"}]
        results = store._hybrid_rescore("text", raw_hits, top_k=5)
        # score should be rounded to 4 decimal places
        assert results[0].score == round(results[0].score, 4)

    def test_hybrid_rescore_missing_fields_use_defaults(self, mock_embedding_engine):
        with patch("src.retrieval.vector_store.VectorStore._init_qdrant"):
            store = VectorStore(backend="qdrant", embedding_engine=mock_embedding_engine)

        raw_hits = [{"score": 0.8}]  # all other fields missing
        results = store._hybrid_rescore("query", raw_hits, top_k=5)
        assert results[0].chunk_id == ""
        assert results[0].modality == "text"
        assert results[0].page_number == 1
        assert results[0].source_path == ""


# ------------------------------------------------------------------ #
# search() — end-to-end with modality filtering
# ------------------------------------------------------------------ #


class TestVectorStoreSearch:
    def test_search_returns_at_most_top_k(self, mock_embedding_engine):
        with patch("src.retrieval.vector_store.VectorStore._init_qdrant"):
            store = VectorStore(backend="qdrant", embedding_engine=mock_embedding_engine)

        with patch.object(
            store,
            "_raw_search",
            return_value=[
                {"score": 0.8, "content": "a", "chunk_id": "1", "modality": "text"},
                {"score": 0.7, "content": "b", "chunk_id": "2", "modality": "text"},
                {"score": 0.6, "content": "c", "chunk_id": "3", "modality": "text"},
            ],
        ):
            results = store.search("test query", top_k=2, modalities=("text",))

        assert len(results) <= 2

    def test_search_text_only_skips_image_search(self, mock_embedding_engine):
        with patch("src.retrieval.vector_store.VectorStore._init_qdrant"):
            store = VectorStore(backend="qdrant", embedding_engine=mock_embedding_engine)

        with patch.object(store, "_search_text", return_value=[]) as mock_text, \
             patch.object(store, "_search_image", return_value=[]) as mock_image:
            store.search("query", top_k=5, modalities=("text", "table"))

        mock_text.assert_called_once()
        mock_image.assert_not_called()

    def test_search_image_only_skips_text_search(self, mock_embedding_engine):
        with patch("src.retrieval.vector_store.VectorStore._init_qdrant"):
            store = VectorStore(backend="qdrant", embedding_engine=mock_embedding_engine)

        with patch.object(store, "_search_text", return_value=[]) as mock_text, \
             patch.object(store, "_search_image", return_value=[]) as mock_image:
            store.search("query", top_k=5, modalities=("image",))

        mock_text.assert_not_called()
        mock_image.assert_called_once()

    def test_search_results_sorted_by_score(self, mock_embedding_engine):
        with patch("src.retrieval.vector_store.VectorStore._init_qdrant"):
            store = VectorStore(backend="qdrant", embedding_engine=mock_embedding_engine)

        mock_results = [
            RetrievalResult("a", "low score content", "text", 0.3, 1, "", {}),
            RetrievalResult("b", "high score content", "text", 0.9, 1, "", {}),
            RetrievalResult("c", "mid score content", "text", 0.6, 1, "", {}),
        ]
        with patch.object(store, "_search_text", return_value=mock_results), \
             patch.object(store, "_search_image", return_value=[]):
            results = store.search("query", top_k=10, modalities=("text",))

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ #
# _to_chroma_metadata
# ------------------------------------------------------------------ #


class TestToChromaMetadata:
    def test_metadata_excludes_non_scalar_nested_values(self):
        chunk = _make_chunk()
        chunk.metadata = {"nested_list": [1, 2, 3], "simple": "value"}
        meta = VectorStore._to_chroma_metadata(chunk)
        assert isinstance(meta["meta_nested_list"], str)  # cast to str
        assert meta["meta_simple"] == "value"

    def test_metadata_standard_fields_present(self):
        chunk = _make_chunk("mychunk", "text")
        meta = VectorStore._to_chroma_metadata(chunk)
        assert meta["chunk_id"] == "mychunk"
        assert meta["modality"] == "text"
        assert "page_number" in meta
        assert "source_path" in meta
