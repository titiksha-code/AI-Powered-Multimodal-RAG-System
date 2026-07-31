"""
src/ingestion/parser.py
========================
Multi-modal document parser for the Knowledge Intelligence System.

Responsibilities
-----------------
1. Ingest PDFs, plain text, and standalone images.
2. Extract, per PDF page:
     - Raw text (native PDF text layer)
     - Tables (via pdfplumber)
     - Embedded images / diagrams (via PyMuPDF)
     - OCR fallback text for scanned/rasterized pages (via pytesseract)
3. Normalize everything into a list of `Chunk` objects that downstream
   embedding + retrieval modules can consume uniformly, regardless of
   original modality.

Design notes
------------
- Kept dependency-light and defensive: every extractor is wrapped so a
  single malformed page/image never crashes the whole ingestion run.
- Chunk IDs are deterministic (content hash) so re-ingesting the same
  document is idempotent and safe to re-run.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from loguru import logger
from PIL import Image

ModalityType = Literal["text", "table", "image"]

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"}


@dataclass
class Chunk:
    """A single unit of retrievable knowledge, regardless of modality."""

    chunk_id: str
    doc_id: str
    modality: ModalityType
    content: str                       # text content OR OCR/caption text for images
    page_number: int
    source_path: str
    image_bytes: bytes | None = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "modality": self.modality,
            "content": self.content,
            "page_number": self.page_number,
            "source_path": self.source_path,
            "metadata": self.metadata,
        }
        return d


def _make_id(*parts: str) -> str:
    """Deterministic content-based ID so re-ingestion stays idempotent."""
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Simple sliding-window word-based chunker."""
    words = text.split()
    if not words:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), step)
        if words[i : i + chunk_size]
    ]
    return chunks


class DocumentParser:
    """Entry point used by the ingestion pipeline / UI upload handler."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, ocr_lang: str = "eng"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_lang = ocr_lang

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse(self, file_path: str | Path) -> list[Chunk]:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"Parsing {file_path.name} ({ext})")

        if ext == ".pdf":
            return self._parse_pdf(file_path)
        if ext in {".png", ".jpg", ".jpeg"}:
            return self._parse_image(file_path)
        if ext in {".txt", ".md"}:
            return self._parse_plaintext(file_path)

        return []

    # ------------------------------------------------------------------ #
    # PDF handling
    # ------------------------------------------------------------------ #
    def _parse_pdf(self, file_path: Path) -> list[Chunk]:
        doc_id = _make_id(str(file_path), str(file_path.stat().st_size))
        chunks: list[Chunk] = []

        chunks.extend(self._extract_pdf_text(file_path, doc_id))
        chunks.extend(self._extract_pdf_tables(file_path, doc_id))
        chunks.extend(self._extract_pdf_images(file_path, doc_id))

        logger.success(f"Extracted {len(chunks)} chunks from {file_path.name}")
        return chunks

    def _extract_pdf_text(self, file_path: Path, doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        try:
            with fitz.open(file_path) as pdf:
                for page_num, page in enumerate(pdf, start=1):
                    raw_text = page.get_text("text").strip()

                    # Fallback to OCR for scanned / image-only pages
                    if not raw_text:
                        raw_text = self._ocr_page(page)

                    for piece in _chunk_text(raw_text, self.chunk_size, self.chunk_overlap):
                        cid = _make_id(doc_id, str(page_num), piece[:64])
                        chunks.append(
                            Chunk(
                                chunk_id=cid,
                                doc_id=doc_id,
                                modality="text",
                                content=piece,
                                page_number=page_num,
                                source_path=str(file_path),
                                metadata={"extractor": "pymupdf_text_or_ocr"},
                            )
                        )
        except Exception as exc:  # noqa: BLE001 - defensive ingestion boundary
            logger.error(f"Text extraction failed for {file_path.name}: {exc}")
        return chunks

    def _ocr_page(self, page: "fitz.Page") -> str:
        try:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img, lang=self.ocr_lang).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"OCR fallback failed on page {page.number}: {exc}")
            return ""

    def _extract_pdf_tables(self, file_path: Path, doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    for t_idx, table in enumerate(page.extract_tables()):
                        if not table:
                            continue
                        rendered = self._render_table_as_text(table)
                        cid = _make_id(doc_id, str(page_num), f"table{t_idx}")
                        chunks.append(
                            Chunk(
                                chunk_id=cid,
                                doc_id=doc_id,
                                modality="table",
                                content=rendered,
                                page_number=page_num,
                                source_path=str(file_path),
                                metadata={"extractor": "pdfplumber", "table_index": t_idx},
                            )
                        )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Table extraction failed for {file_path.name}: {exc}")
        return chunks

    @staticmethod
    def _render_table_as_text(table: list[list[str | None]]) -> str:
        """Markdown-style rendering so the LLM can reason over table structure."""
        rows = [[cell or "" for cell in row] for row in table]
        if not rows:
            return ""
        header, *body = rows
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _extract_pdf_images(self, file_path: Path, doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        try:
            with fitz.open(file_path) as pdf:
                for page_num, page in enumerate(pdf, start=1):
                    for img_idx, img_info in enumerate(page.get_images(full=True)):
                        xref = img_info[0]
                        try:
                            base_image = pdf.extract_image(xref)
                            img_bytes = base_image["image"]
                            caption = self._ocr_image_bytes(img_bytes)
                            cid = _make_id(doc_id, str(page_num), f"img{img_idx}")
                            chunks.append(
                                Chunk(
                                    chunk_id=cid,
                                    doc_id=doc_id,
                                    modality="image",
                                    content=caption or "[diagram/figure without extractable text]",
                                    page_number=page_num,
                                    source_path=str(file_path),
                                    image_bytes=img_bytes,
                                    metadata={"extractor": "pymupdf_image", "image_index": img_idx},
                                )
                            )
                        except Exception as inner_exc:  # noqa: BLE001
                            logger.warning(
                                f"Skipping image {img_idx} on page {page_num}: {inner_exc}"
                            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Image extraction failed for {file_path.name}: {exc}")
        return chunks

    def _ocr_image_bytes(self, img_bytes: bytes) -> str:
        try:
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img, lang=self.ocr_lang).strip()
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------------ #
    # Standalone image handling
    # ------------------------------------------------------------------ #
    def _parse_image(self, file_path: Path) -> list[Chunk]:
        doc_id = _make_id(str(file_path), str(file_path.stat().st_size))
        img_bytes = file_path.read_bytes()
        caption = self._ocr_image_bytes(img_bytes)
        cid = _make_id(doc_id, "img0")
        chunk = Chunk(
            chunk_id=cid,
            doc_id=doc_id,
            modality="image",
            content=caption or "[image without extractable text]",
            page_number=1,
            source_path=str(file_path),
            image_bytes=img_bytes,
            metadata={"extractor": "standalone_image"},
        )
        return [chunk]

    # ------------------------------------------------------------------ #
    # Plain text handling
    # ------------------------------------------------------------------ #
    def _parse_plaintext(self, file_path: Path) -> list[Chunk]:
        doc_id = _make_id(str(file_path), str(file_path.stat().st_size))
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks: list[Chunk] = []
        for piece in _chunk_text(text, self.chunk_size, self.chunk_overlap):
            cid = _make_id(doc_id, piece[:64])
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    modality="text",
                    content=piece,
                    page_number=1,
                    source_path=str(file_path),
                    metadata={"extractor": "plaintext"},
                )
            )
        return chunks


def parse_directory(directory: str | Path, parser: DocumentParser | None = None) -> list[Chunk]:
    """Convenience batch-ingestion helper for a folder of source documents."""
    parser = parser or DocumentParser()
    directory = Path(directory)
    all_chunks: list[Chunk] = []

    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS and file_path.is_file():
            try:
                all_chunks.extend(parser.parse(file_path))
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to ingest {file_path}: {exc}")

    return all_chunks
