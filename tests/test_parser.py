"""
tests/test_parser.py
======================
Unit tests for src/ingestion/parser.py
"""

from pathlib import Path

import pytest

from src.ingestion.parser import Chunk, DocumentParser, _chunk_text, _make_id


class TestChunkingUtils:
    def test_chunk_text_basic(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = _chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_empty_string(self):
        assert _chunk_text("") == []

    def test_chunk_text_short_text_single_chunk(self):
        chunks = _chunk_text("hello world", chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_make_id_deterministic(self):
        id1 = _make_id("a", "b", "c")
        id2 = _make_id("a", "b", "c")
        assert id1 == id2

    def test_make_id_unique_for_different_input(self):
        assert _make_id("a") != _make_id("b")


class TestDocumentParserPlaintext:
    def test_parse_plaintext_file(self, tmp_path: Path):
        file_path = tmp_path / "sample.txt"
        file_path.write_text("This is a sample document used for unit testing purposes.")

        parser = DocumentParser(chunk_size=10, chunk_overlap=2)
        chunks = parser.parse(file_path)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.modality == "text" for c in chunks)
        assert all(c.source_path == str(file_path) for c in chunks)

    def test_parse_unsupported_extension_raises(self, tmp_path: Path):
        file_path = tmp_path / "sample.xyz"
        file_path.write_text("irrelevant")

        parser = DocumentParser()
        with pytest.raises(ValueError):
            parser.parse(file_path)


class TestChunkSerialization:
    def test_chunk_to_dict(self):
        chunk = Chunk(
            chunk_id="abc123",
            doc_id="doc1",
            modality="text",
            content="hello",
            page_number=1,
            source_path="/tmp/doc.pdf",
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "abc123"
        assert d["modality"] == "text"
        assert "image_bytes" not in d  # binary payload must never leak into metadata dicts
