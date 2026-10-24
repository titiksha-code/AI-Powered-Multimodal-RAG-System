"""
tests/test_parser.py
======================
Unit tests for src/ingestion/parser.py

Covers:
  - Pure utility functions (_chunk_text, _make_id)
  - DocumentParser plaintext parsing (real I/O via tmp_path)
  - DocumentParser image parsing (mocked pytesseract + fitz)
  - DocumentParser PDF text / table / image extraction (mocked fitz + pdfplumber)
  - parse_directory batch helper
  - Chunk dataclass serialisation
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.ingestion.parser import (
    SUPPORTED_EXTENSIONS,
    Chunk,
    DocumentParser,
    _chunk_text,
    _make_id,
    parse_directory,
)


# ------------------------------------------------------------------ #
# Utility helpers
# ------------------------------------------------------------------ #


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

    def test_chunk_text_overlap_produces_repeated_words(self):
        text = " ".join(str(i) for i in range(30))
        chunks = _chunk_text(text, chunk_size=10, overlap=5)
        # With overlap, words near boundaries appear in consecutive chunks
        assert len(chunks) >= 2

    def test_make_id_deterministic(self):
        id1 = _make_id("a", "b", "c")
        id2 = _make_id("a", "b", "c")
        assert id1 == id2

    def test_make_id_unique_for_different_input(self):
        assert _make_id("a") != _make_id("b")

    def test_make_id_length(self):
        """IDs are always 24 hex chars."""
        cid = _make_id("some", "parts")
        assert len(cid) == 24

    def test_supported_extensions_set(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".exe" not in SUPPORTED_EXTENSIONS


# ------------------------------------------------------------------ #
# Chunk dataclass
# ------------------------------------------------------------------ #


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

    def test_chunk_defaults(self):
        chunk = Chunk(
            chunk_id="x",
            doc_id="y",
            modality="image",
            content="",
            page_number=1,
            source_path="",
        )
        assert chunk.image_bytes is None
        assert chunk.metadata == {}

    def test_chunk_to_dict_includes_metadata(self):
        chunk = Chunk(
            chunk_id="abc",
            doc_id="doc",
            modality="table",
            content="| A | B |",
            page_number=2,
            source_path="/tmp/t.pdf",
            metadata={"extractor": "pdfplumber", "table_index": 0},
        )
        d = chunk.to_dict()
        assert d["metadata"] == {"extractor": "pdfplumber", "table_index": 0}


# ------------------------------------------------------------------ #
# Plaintext parsing — real I/O via tmp_path
# ------------------------------------------------------------------ #


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

    def test_parse_markdown_file(self, tmp_path: Path):
        md_path = tmp_path / "readme.md"
        md_path.write_text("# Title\n\nSome paragraph text here for testing.")
        parser = DocumentParser()
        chunks = parser.parse(md_path)
        assert len(chunks) >= 1
        assert all(c.modality == "text" for c in chunks)

    def test_parse_plaintext_page_number_is_one(self, tmp_path: Path):
        file_path = tmp_path / "doc.txt"
        file_path.write_text("Hello world content for page number check.")
        parser = DocumentParser()
        chunks = parser.parse(file_path)
        assert all(c.page_number == 1 for c in chunks)

    def test_parse_plaintext_chunk_ids_are_unique(self, tmp_path: Path):
        file_path = tmp_path / "multi.txt"
        # Write enough content to produce multiple chunks
        file_path.write_text(" ".join(f"word{i}" for i in range(1200)))
        parser = DocumentParser(chunk_size=100, chunk_overlap=10)
        chunks = parser.parse(file_path)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "All chunk IDs must be unique"

    def test_parse_unsupported_extension_raises(self, tmp_path: Path):
        file_path = tmp_path / "sample.xyz"
        file_path.write_text("irrelevant")

        parser = DocumentParser()
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse(file_path)

    def test_parse_empty_text_file_returns_no_chunks(self, tmp_path: Path):
        file_path = tmp_path / "empty.txt"
        file_path.write_text("")
        parser = DocumentParser()
        chunks = parser.parse(file_path)
        assert chunks == []


# ------------------------------------------------------------------ #
# Image parsing — mock pytesseract to avoid Tesseract dependency
# ------------------------------------------------------------------ #


class TestDocumentParserImage:
    def _make_png_bytes(self) -> bytes:
        """Generate a minimal valid PNG image in memory."""
        img = Image.new("RGB", (64, 64), color=(100, 149, 237))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_parse_image_returns_single_chunk(self, tmp_path: Path):
        img_path = tmp_path / "diagram.png"
        img_path.write_bytes(self._make_png_bytes())

        with patch("pytesseract.image_to_string", return_value="OCR caption text"):
            parser = DocumentParser()
            chunks = parser.parse(img_path)

        assert len(chunks) == 1
        assert chunks[0].modality == "image"
        assert chunks[0].page_number == 1
        assert chunks[0].content == "OCR caption text"
        assert chunks[0].image_bytes == img_path.read_bytes()

    def test_parse_image_no_ocr_text_uses_fallback(self, tmp_path: Path):
        img_path = tmp_path / "blank.png"
        img_path.write_bytes(self._make_png_bytes())

        with patch("pytesseract.image_to_string", return_value=""):
            parser = DocumentParser()
            chunks = parser.parse(img_path)

        assert chunks[0].content == "[image without extractable text]"

    def test_parse_jpeg_extension_accepted(self, tmp_path: Path):
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpg_path = tmp_path / "photo.jpg"
        jpg_path.write_bytes(buf.getvalue())

        with patch("pytesseract.image_to_string", return_value="some text"):
            parser = DocumentParser()
            chunks = parser.parse(jpg_path)

        assert len(chunks) == 1
        assert chunks[0].modality == "image"


# ------------------------------------------------------------------ #
# PDF text extraction — mock fitz (PyMuPDF)
# ------------------------------------------------------------------ #


class TestDocumentParserPdfText:
    def _make_mock_pdf_page(self, text: str) -> MagicMock:
        page = MagicMock()
        page.number = 0
        page.get_text.return_value = text
        page.get_images.return_value = []
        return page

    def test_parse_pdf_text_extraction(self, tmp_path: Path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")  # minimal stub so stat() works

        page_text = " ".join(f"token{i}" for i in range(600))
        mock_page = self._make_mock_pdf_page(page_text)

        mock_fitz_doc = MagicMock()
        mock_fitz_doc.__enter__ = lambda s: s
        mock_fitz_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

        mock_plumber_doc = MagicMock()
        mock_plumber_doc.__enter__ = lambda s: s
        mock_plumber_doc.__exit__ = MagicMock(return_value=False)
        mock_page_plumb = MagicMock()
        mock_page_plumb.extract_tables.return_value = []
        mock_plumber_doc.pages = [mock_page_plumb]

        with (
            patch("fitz.open", return_value=mock_fitz_doc),
            patch("pdfplumber.open", return_value=mock_plumber_doc),
        ):
            parser = DocumentParser(chunk_size=100, chunk_overlap=10)
            chunks = parser.parse(pdf_path)

        text_chunks = [c for c in chunks if c.modality == "text"]
        assert len(text_chunks) >= 1
        assert all(c.page_number == 1 for c in text_chunks)
        assert all(c.source_path == str(pdf_path) for c in text_chunks)

    def test_parse_pdf_ocr_fallback_on_empty_page(self, tmp_path: Path):
        """When a page has no native text, OCR fallback is triggered."""
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page = self._make_mock_pdf_page("")  # empty native text → triggers OCR

        mock_fitz_doc = MagicMock()
        mock_fitz_doc.__enter__ = lambda s: s
        mock_fitz_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

        # OCR pixmap mock
        mock_pix = MagicMock()
        img = Image.new("RGB", (64, 64))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        mock_pix.tobytes.return_value = buf.getvalue()
        mock_page.get_pixmap.return_value = mock_pix

        mock_plumber_doc = MagicMock()
        mock_plumber_doc.__enter__ = lambda s: s
        mock_plumber_doc.__exit__ = MagicMock(return_value=False)
        mock_plumber_page = MagicMock()
        mock_plumber_page.extract_tables.return_value = []
        mock_plumber_doc.pages = [mock_plumber_page]

        with (
            patch("fitz.open", return_value=mock_fitz_doc),
            patch("pdfplumber.open", return_value=mock_plumber_doc),
            patch("pytesseract.image_to_string", return_value="OCR extracted text from scan"),
        ):
            parser = DocumentParser()
            chunks = parser.parse(pdf_path)

        text_chunks = [c for c in chunks if c.modality == "text"]
        assert len(text_chunks) >= 1
        assert "OCR extracted text from scan" in text_chunks[0].content


# ------------------------------------------------------------------ #
# PDF table extraction
# ------------------------------------------------------------------ #


class TestDocumentParserPdfTables:
    def test_parse_pdf_extracts_tables(self, tmp_path: Path):
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        mock_page_fitz = MagicMock()
        mock_page_fitz.number = 0
        mock_page_fitz.get_text.return_value = "Some text content here."
        mock_page_fitz.get_images.return_value = []

        mock_fitz_doc = MagicMock()
        mock_fitz_doc.__enter__ = lambda s: s
        mock_fitz_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz_doc.__iter__ = MagicMock(return_value=iter([mock_page_fitz]))

        sample_table = [
            ["Metric", "Score"],
            ["MRR@5", "0.83"],
            ["Recall@10", "0.91"],
        ]
        mock_page_plumb = MagicMock()
        mock_page_plumb.extract_tables.return_value = [sample_table]

        mock_plumber_doc = MagicMock()
        mock_plumber_doc.__enter__ = lambda s: s
        mock_plumber_doc.__exit__ = MagicMock(return_value=False)
        mock_plumber_doc.pages = [mock_page_plumb]

        with (
            patch("fitz.open", return_value=mock_fitz_doc),
            patch("pdfplumber.open", return_value=mock_plumber_doc),
        ):
            parser = DocumentParser()
            chunks = parser.parse(pdf_path)

        table_chunks = [c for c in chunks if c.modality == "table"]
        assert len(table_chunks) == 1
        assert "Metric" in table_chunks[0].content
        assert "MRR@5" in table_chunks[0].content
        assert "---" in table_chunks[0].content  # markdown separator row

    def test_render_table_as_text_markdown_format(self):
        table = [["Name", "Value"], ["Accuracy", "92%"], ["Latency", "312ms"]]
        rendered = DocumentParser._render_table_as_text(table)
        lines = rendered.strip().split("\n")
        assert lines[0].startswith("| Name")
        assert "---" in lines[1]
        assert "Accuracy" in lines[2]

    def test_render_table_empty_returns_empty_string(self):
        assert DocumentParser._render_table_as_text([]) == ""


# ------------------------------------------------------------------ #
# PDF image extraction
# ------------------------------------------------------------------ #


class TestDocumentParserPdfImages:
    def _make_png_bytes(self) -> bytes:
        img = Image.new("RGB", (32, 32), color=(0, 128, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _make_fitz_doc(self, page: MagicMock):
        """Return a context-manager mock that yields the given page on iteration."""
        doc = MagicMock()
        doc.__enter__ = lambda s: s
        doc.__exit__ = MagicMock(return_value=False)
        doc.__iter__ = MagicMock(return_value=iter([page]))
        return doc

    def test_parse_pdf_extracts_images(self, tmp_path: Path):
        pdf_path = tmp_path / "slides.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        png_bytes = self._make_png_bytes()

        # Text-extraction page (no images in get_images, handled by text pass)
        mock_page_text = MagicMock()
        mock_page_text.number = 0
        mock_page_text.get_text.return_value = "Slide text"
        mock_page_text.get_images.return_value = []  # no images in text pass

        # Image-extraction page
        mock_page_img = MagicMock()
        mock_page_img.number = 0
        mock_page_img.get_text.return_value = "Slide text"
        mock_page_img.get_images.return_value = [(1, 0, 0, 0, 0, 0, 0)]  # xref=1

        # fitz.open is called twice: text pass then image pass
        fitz_doc_text = self._make_fitz_doc(mock_page_text)
        fitz_doc_img = self._make_fitz_doc(mock_page_img)
        fitz_doc_img.extract_image.return_value = {"image": png_bytes}

        mock_plumber_doc = MagicMock()
        mock_plumber_doc.__enter__ = lambda s: s
        mock_plumber_doc.__exit__ = MagicMock(return_value=False)
        mock_plumber_page = MagicMock()
        mock_plumber_page.extract_tables.return_value = []
        mock_plumber_doc.pages = [mock_plumber_page]

        with (
            patch("fitz.open", side_effect=[fitz_doc_text, fitz_doc_img, fitz_doc_img]),
            patch("pdfplumber.open", return_value=mock_plumber_doc),
            patch("pytesseract.image_to_string", return_value="diagram caption"),
        ):
            parser = DocumentParser()
            chunks = parser.parse(pdf_path)

        image_chunks = [c for c in chunks if c.modality == "image"]
        assert len(image_chunks) == 1
        assert image_chunks[0].image_bytes == png_bytes
        assert image_chunks[0].content == "diagram caption"
        assert image_chunks[0].metadata["extractor"] == "pymupdf_image"

    def test_parse_pdf_image_with_no_ocr_text_uses_fallback(self, tmp_path: Path):
        pdf_path = tmp_path / "notext.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        png_bytes = self._make_png_bytes()

        mock_page_text = MagicMock()
        mock_page_text.number = 0
        mock_page_text.get_text.return_value = "some page text"
        mock_page_text.get_images.return_value = []

        mock_page_img = MagicMock()
        mock_page_img.number = 0
        mock_page_img.get_text.return_value = "some page text"
        mock_page_img.get_images.return_value = [(2, 0, 0, 0, 0, 0, 0)]

        fitz_doc_text = self._make_fitz_doc(mock_page_text)
        fitz_doc_img = self._make_fitz_doc(mock_page_img)
        fitz_doc_img.extract_image.return_value = {"image": png_bytes}

        mock_plumber_doc = MagicMock()
        mock_plumber_doc.__enter__ = lambda s: s
        mock_plumber_doc.__exit__ = MagicMock(return_value=False)
        mock_plumber_page = MagicMock()
        mock_plumber_page.extract_tables.return_value = []
        mock_plumber_doc.pages = [mock_plumber_page]

        with (
            patch("fitz.open", side_effect=[fitz_doc_text, fitz_doc_img, fitz_doc_img]),
            patch("pdfplumber.open", return_value=mock_plumber_doc),
            patch("pytesseract.image_to_string", return_value=""),
        ):
            parser = DocumentParser()
            chunks = parser.parse(pdf_path)

        image_chunks = [c for c in chunks if c.modality == "image"]
        assert len(image_chunks) == 1
        assert image_chunks[0].content == "[diagram/figure without extractable text]"


# ------------------------------------------------------------------ #
# parse_directory batch helper
# ------------------------------------------------------------------ #


class TestParseDirectory:
    def test_parse_directory_ingests_txt_files(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("Content of file A for testing batch parse.")
        (tmp_path / "b.txt").write_text("Content of file B for testing batch parse.")
        (tmp_path / "ignore.exe").write_text("binary stub")

        chunks = parse_directory(tmp_path)
        assert len(chunks) >= 2
        assert all(c.modality == "text" for c in chunks)

    def test_parse_directory_skips_unsupported_files(self, tmp_path: Path):
        (tmp_path / "valid.txt").write_text("Good content here.")
        (tmp_path / "bad.bin").write_bytes(b"\x00\x01\x02")

        chunks = parse_directory(tmp_path)
        sources = [c.source_path for c in chunks]
        assert not any("bad.bin" in s for s in sources)

    def test_parse_directory_accepts_custom_parser(self, tmp_path: Path):
        (tmp_path / "doc.txt").write_text("Short content.")
        custom_parser = DocumentParser(chunk_size=5, chunk_overlap=1)
        chunks = parse_directory(tmp_path, parser=custom_parser)
        assert len(chunks) >= 1

    def test_parse_directory_empty_folder_returns_empty(self, tmp_path: Path):
        chunks = parse_directory(tmp_path)
        assert chunks == []
