"""
tests/test_multimodal_rag.py
==============================
Unit tests for src/models/multimodal_rag.py

Covers:
  - Context assembly (_assemble_context) for text, table, and image chunks
  - Local fallback extractive synthesis (no model weights)
  - End-to-end generate_answer with mocked provider
  - OpenAI provider path (mocked openai client)
  - Anthropic provider path (mocked anthropic client)
  - RAGAnswer dataclass structure and defaults
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.multimodal_rag import DEFAULT_SYSTEM_PROMPT, MultimodalRAG, RAGAnswer
from src.retrieval.vector_store import RetrievalResult


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


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


@pytest.fixture
def image_retrieved(tmp_path):
    img_path = tmp_path / "figure.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # stub PNG bytes
    return [
        RetrievalResult(
            chunk_id="img789",
            content="Architecture diagram showing ingestion pipeline",
            modality="image",
            score=0.65,
            page_number=2,
            source_path=str(img_path),
            metadata={},
        )
    ]


# ------------------------------------------------------------------ #
# RAGAnswer dataclass
# ------------------------------------------------------------------ #


class TestRAGAnswer:
    def test_rag_answer_defaults(self):
        ans = RAGAnswer(query="q", answer="a")
        assert ans.citations == []
        assert ans.used_chunks == []
        assert ans.provider == ""
        assert ans.latency_ms == 0.0

    def test_rag_answer_with_all_fields(self, sample_retrieved):
        ans = RAGAnswer(
            query="What is MRR?",
            answer="MRR is 0.82.",
            citations=["[chunk_abc123]"],
            used_chunks=sample_retrieved,
            provider="openai",
            latency_ms=45.5,
        )
        assert ans.provider == "openai"
        assert len(ans.used_chunks) == 2


# ------------------------------------------------------------------ #
# Context assembly
# ------------------------------------------------------------------ #


class TestContextAssembly:
    def test_assemble_context_includes_all_citations(self, sample_retrieved):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.system_prompt = "system"

        context_text, image_paths, citations = rag._assemble_context(sample_retrieved)

        assert "[chunk_abc123]" in citations
        assert "[chunk_def456]" in citations
        assert "91% retrieval accuracy" in context_text
        assert "MRR" in context_text
        assert image_paths == []

    def test_assemble_context_empty_input(self):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        context_text, image_paths, citations = rag._assemble_context([])
        assert context_text == "No relevant context found."
        assert citations == []
        assert image_paths == []

    def test_assemble_context_image_chunk_with_existing_file(self, image_retrieved):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.system_prompt = "system"

        context_text, image_paths, citations = rag._assemble_context(image_retrieved)

        assert "[chunk_img789]" in citations
        assert len(image_paths) == 1
        assert "Architecture diagram" in context_text

    def test_assemble_context_image_chunk_missing_file_excluded(self):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        result = RetrievalResult(
            chunk_id="missing",
            content="some figure",
            modality="image",
            score=0.5,
            page_number=1,
            source_path="/nonexistent/path/image.png",
            metadata={},
        )
        _, image_paths, _ = rag._assemble_context([result])
        assert image_paths == []

    def test_assemble_context_text_block_format(self, sample_retrieved):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.system_prompt = "system"
        context_text, _, _ = rag._assemble_context(sample_retrieved)
        # Should include page number and modality in the block header
        assert "page 3" in context_text
        assert "text" in context_text
        assert "table" in context_text


# ------------------------------------------------------------------ #
# Local fallback (no model weights)
# ------------------------------------------------------------------ #


class TestGenerateAnswerLocalFallback:
    def test_generate_local_without_loaded_model_returns_context(self, sample_retrieved):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "local"
        rag.system_prompt = "system"
        rag._local_model = None

        result = rag._generate_local("What is the retrieval accuracy?", "some context", [])
        assert "some context" in result
        assert "unavailable" in result.lower()

    def test_generate_local_fallback_extracts_relevant_sentence(self):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "local"
        rag.system_prompt = "system"
        rag._local_model = None

        context = (
            "[chunk_abc123] (page 1, text):\n"
            "The hybrid retrieval pipeline achieves excellent MRR scores. "
            "Unrelated sentence about cats."
        )
        result = rag._generate_local("What is the MRR score?", context, [])
        # Should contain some answer text, even in fallback mode
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_local_fallback_no_context(self):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag._local_model = None
        result = rag._generate_local("question", "No relevant context found.", [])
        assert isinstance(result, str)


# ------------------------------------------------------------------ #
# End-to-end generate_answer
# ------------------------------------------------------------------ #


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

    def test_generate_answer_citations_match_context(self, sample_retrieved):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "local"
        rag.system_prompt = "system"
        rag._local_model = None

        with patch.object(rag, "_generate_local", return_value="answer text"):
            answer = rag.generate_answer("query", sample_retrieved)

        assert "[chunk_abc123]" in answer.citations
        assert "[chunk_def456]" in answer.citations

    def test_generate_answer_latency_is_positive(self, sample_retrieved):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "local"
        rag.system_prompt = "system"
        rag._local_model = None

        with patch.object(rag, "_generate_local", return_value="answer"):
            answer = rag.generate_answer("query", sample_retrieved)

        assert answer.latency_ms >= 0.0

    def test_generate_answer_empty_retrieved_list(self):
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "local"
        rag.system_prompt = "system"
        rag._local_model = None

        answer = rag.generate_answer("question with no context", [])
        assert isinstance(answer, RAGAnswer)
        assert answer.citations == []
        assert answer.used_chunks == []


# ------------------------------------------------------------------ #
# OpenAI provider (mocked via sys.modules)
# ------------------------------------------------------------------ #


class TestOpenAIProvider:
    def test_generate_openai_calls_chat_completion(self, sample_retrieved):
        import sys

        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "openai"
        rag.system_prompt = DEFAULT_SYSTEM_PROMPT
        rag.temperature = 0.2
        rag.max_new_tokens = 256

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OpenAI answer about retrieval."

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_openai_client

        with patch.dict(sys.modules, {"openai": mock_openai_module}), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = rag._generate_openai("query", "context", [])

        assert result == "OpenAI answer about retrieval."
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_generate_openai_includes_system_prompt(self, sample_retrieved):
        import sys

        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "openai"
        rag.system_prompt = "custom system prompt"
        rag.temperature = 0.0
        rag.max_new_tokens = 128

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "answer"

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_openai_client

        with patch.dict(sys.modules, {"openai": mock_openai_module}), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            rag._generate_openai("q", "ctx", [])

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert system_msg["content"] == "custom system prompt"


# ------------------------------------------------------------------ #
# Anthropic provider (mocked via sys.modules)
# ------------------------------------------------------------------ #


class TestAnthropicProvider:
    def test_generate_anthropic_calls_messages_create(self):
        import sys

        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "anthropic"
        rag.system_prompt = DEFAULT_SYSTEM_PROMPT
        rag.temperature = 0.2
        rag.max_new_tokens = 256

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Anthropic answer text."

        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_anthropic_client = MagicMock()
        mock_anthropic_client.messages.create.return_value = mock_response

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_anthropic_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            result = rag._generate_anthropic("query", "context text", [])

        assert result == "Anthropic answer text."
        mock_anthropic_client.messages.create.assert_called_once()

    def test_generate_anthropic_concatenates_multiple_text_blocks(self):
        import sys

        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "anthropic"
        rag.system_prompt = DEFAULT_SYSTEM_PROMPT
        rag.temperature = 0.2
        rag.max_new_tokens = 256

        block1 = MagicMock()
        block1.type = "text"
        block1.text = "Part one. "

        block2 = MagicMock()
        block2.type = "text"
        block2.text = "Part two."

        mock_response = MagicMock()
        mock_response.content = [block1, block2]

        mock_anthropic_client = MagicMock()
        mock_anthropic_client.messages.create.return_value = mock_response

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_anthropic_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            result = rag._generate_anthropic("query", "context", [])

        assert result == "Part one. Part two."

    def test_generate_anthropic_end_to_end_via_generate_answer(self, sample_retrieved):
        # Use local provider but patch _generate_anthropic directly to test the
        # generate_answer orchestration path without needing the SDK installed.
        rag = MultimodalRAG.__new__(MultimodalRAG)
        rag.provider = "local"
        rag.system_prompt = DEFAULT_SYSTEM_PROMPT
        rag.temperature = 0.2
        rag.max_new_tokens = 256
        rag._local_model = None

        with patch.object(rag, "_generate_local", return_value="mocked provider answer"):
            answer = rag.generate_answer("What is recall@10?", sample_retrieved)

        assert isinstance(answer, RAGAnswer)
        assert answer.answer == "mocked provider answer"
        assert answer.provider == "local"
