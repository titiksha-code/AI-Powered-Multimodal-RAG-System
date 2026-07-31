"""
tests/test_multimodal_rag.py
==============================
Unit tests for src/models/multimodal_rag.py
"""

from unittest.mock import patch

import pytest

from src.models.multimodal_rag import MultimodalRAG, RAGAnswer
from src.retrieval.vector_store import RetrievalResult


@pytest.fixture
def sample_retrieved():
    return [
        RetrievalResult(
            chunk_id="abc123",
            content="The system achieves 91% retrieval accuracy on the benchmark set.",
            modality="text",
            score=0.87,
            page_number=3,
            source_path="/tmp/report.pdf",
            metadata={},
        ),
        RetrievalResult(
            chunk_id="def456",
            content="| Metric | Value |\n| --- | --- |\n| MRR | 0.82 |",
            modality="table",
            score=0.74,
            page_number=4,
            source_path="/tmp/report.pdf",
            metadata={},
        ),
    ]


class TestContextAssembly:
    def test_assemble_context_includes_all_citations(self, sample_retrieved):
        rag = MultimodalRAG.__new__(MultimodalRAG)  # skip __init__ (avoids model loading)
        rag.system_prompt = "system"

        context_text, image_paths, citations = rag._assemble_context(sample_retrieved)

        assert "[chunk_abc123]" in citations
        assert "[chunk_def456]" in citations
        assert "91% retrieval accuracy" in context_text
        assert image_paths == []  # no image-modality chunks in the fixture

    def test_assemble_context_empty_input(self):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        context_text, image_paths, citations = rag._assemble_context([])
        assert context_text == "No relevant context found."
        assert citations == []


class TestGenerateAnswerLocalFallback:
    def test_generate_local_without_loaded_model_returns_context(self, sample_retrieved):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "local"
        rag.system_prompt = "system"
        rag._local_model = None  # simulate model unavailable in this environment

        result = rag._generate_local("What is the retrieval accuracy?", "some context", [])
        assert "some context" in result
        assert "unavailable" in result.lower()


class TestGenerateAnswerEndToEnd:
    def test_generate_answer_returns_rag_answer(self, sample_retrieved):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "local"
        rag.system_prompt = "system"
        rag._local_model = None

        with patch.object(rag, "_generate_local", return_value="Mocked answer with citation."):
            answer = rag.generate_answer("What is the MRR?", sample_retrieved)

        assert isinstance(answer, RAGAnswer)
        assert answer.answer == "Mocked answer with citation."
        assert answer.provider == "local"
        assert answer.latency_ms >= 0
        assert len(answer.used_chunks) == 2
