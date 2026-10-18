"""
src/models/multimodal_rag.py
==============================
Fuses retrieved text/table/image context with a Vision-LLM to produce a
grounded, cited answer.

Supports three interchangeable LLM providers behind one interface:
    - "local"     : a local HF vision-language model (e.g. Qwen2-VL)
    - "openai"    : hosted vision-capable model via the OpenAI API
    - "anthropic" : hosted vision-capable Claude model via the Anthropic API

The class is intentionally provider-agnostic at the call site: the UI /
pipeline code just calls `generate_answer(query, retrieved_results)` and
gets back a structured `RAGAnswer`.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from loguru import logger

from src.retrieval.vector_store import RetrievalResult

ProviderType = Literal["local", "openai", "anthropic"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise knowledge assistant. Answer strictly using the "
    "provided multimodal context (text passages, table excerpts, and "
    "image/diagram descriptions). Cite the source chunk id in square "
    "brackets, e.g. [chunk_ab12cd], for every claim. If the answer is "
    "not contained in the context, say so explicitly instead of guessing."
)


@dataclass
class RAGAnswer:
    query: str
    answer: str
    citations: list[str] = field(default_factory=list)
    used_chunks: list[RetrievalResult] = field(default_factory=list)
    provider: str = ""
    latency_ms: float = 0.0


class MultimodalRAG:
    """High-level orchestrator: context assembly -> Vision-LLM -> grounded answer."""

    def __init__(
        self,
        provider: ProviderType = "local",
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        temperature: float = 0.2,
        max_new_tokens: int = 768,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

        self._local_model = None
        self._local_processor = None

        if provider == "local":
            self._load_local_model()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_local_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            logger.info(f"Loading local Vision-LLM: {self.model_name}")
            self._local_processor = AutoProcessor.from_pretrained(self.model_name)
            self._local_model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Could not load local model '{self.model_name}' ({exc}). "
                "Falling back to context-only extractive mode."
            )
            self._local_model = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_answer(self, query: str, retrieved: list[RetrievalResult]) -> RAGAnswer:
        import time

        start = time.perf_counter()

        context_text, image_paths, citations = self._assemble_context(retrieved)

        if self.provider == "openai":
            answer_text = self._generate_openai(query, context_text, image_paths)
        elif self.provider == "anthropic":
            answer_text = self._generate_anthropic(query, context_text, image_paths)
        else:
            answer_text = self._generate_local(query, context_text, image_paths)

        latency_ms = (time.perf_counter() - start) * 1000

        return RAGAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            used_chunks=retrieved,
            provider=self.provider,
            latency_ms=round(latency_ms, 2),
        )

    # ------------------------------------------------------------------ #
    # Context assembly
    # ------------------------------------------------------------------ #
    def _assemble_context(
        self, retrieved: list[RetrievalResult]
    ) -> tuple[str, list[str], list[str]]:
        text_blocks: list[str] = []
        image_paths: list[str] = []
        citations: list[str] = []

        for r in retrieved:
            tag = f"[chunk_{r.chunk_id}]"
            citations.append(tag)
            if r.modality in ("text", "table"):
                text_blocks.append(
                    f"{tag} (page {r.page_number}, {r.modality}):\n{r.content}"
                )
            elif r.modality == "image":
                text_blocks.append(
                    f"{tag} (page {r.page_number}, image/diagram) "
                    f"OCR/caption: {r.content or '[no extractable text]'}"
                )
                if r.source_path and Path(r.source_path).exists():
                    image_paths.append(r.source_path)

        context_text = "\n\n".join(text_blocks) if text_blocks else "No relevant context found."
        return context_text, image_paths, citations

    # ------------------------------------------------------------------ #
    # Provider backends
    # ------------------------------------------------------------------ #
    def _generate_local(self, query: str, context: str, image_paths: list[str]) -> str:
        if self._local_model is None:
            # Intelligent extractive synthesis: structure key evidence into a crisp answer
            sections = [s.strip() for s in context.split("\n\n") if s.strip()]
            bullets = []
            citations = []
            for sec in sections:
                lines = sec.split("\n")
                header = lines[0] if lines else ""
                body = " ".join(lines[1:]) if len(lines) > 1 else header
                cid_match = re.search(r"\[chunk_[a-zA-Z0-9]+\]", header)
                cid = cid_match.group(0) if cid_match else ""
                if cid and cid not in citations:
                    citations.append(cid)

                # Split body into clean sentences
                sentences = re.split(r"(?<=[.!?])\s+", body)
                q_words = set(re.findall(r"\w+", query.lower()))
                scored = []
                for s in sentences:
                    s_clean = s.strip()
                    if len(s_clean) > 20:
                        overlap = len(set(re.findall(r"\w+", s_clean.lower())) & q_words)
                        scored.append((overlap, s_clean))
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top_sentence = scored[0][1]
                    tag = f"**{cid}** " if cid else ""
                    bullets.append(f"- {tag}{top_sentence}")

            evidence = "\n\n".join(bullets[:4]) if bullets else context
            cited_str = " ".join(citations[:4]) if citations else ""
            return (
                f"[Local model weights unavailable — grounded context synthesis]\n\n"
                f"Based on the retrieved multimodal knowledge base for **'{query}'**:\n\n"
                f"{evidence}\n\n"
                f"**Citations:** {cited_str if cited_str else 'Direct context grounding'}"
            )

        from PIL import Image

        prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        images = [Image.open(p).convert("RGB") for p in image_paths[:4]]  # cap for VRAM safety

        inputs = self._local_processor(
            text=prompt, images=images or None, return_tensors="pt"
        ).to(self._local_model.device)

        output_ids = self._local_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        decoded = self._local_processor.batch_decode(output_ids, skip_special_tokens=True)
        return decoded[0].split("Answer:")[-1].strip()

    def _generate_openai(self, query: str, context: str, image_paths: list[str]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        content = [{"type": "text", "text": f"Context:\n{context}\n\nQuestion: {query}"}]
        for path in image_paths[:4]:
            b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return response.choices[0].message.content or ""

    def _generate_anthropic(self, query: str, context: str, image_paths: list[str]) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        content = [{"type": "text", "text": f"Context:\n{context}\n\nQuestion: {query}"}]
        for path in image_paths[:4]:
            b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                }
            )

        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
