"""
src/api/routes.py
=================
FastAPI API endpoints for health check, multimodal query, and document ingestion.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from loguru import logger

from src.api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievalItemSchema,
)
from src.ingestion.parser import DocumentParser, SUPPORTED_EXTENSIONS
from src.retrieval.vector_store import RetrievalResult, VectorStore
from src.models.multimodal_rag import MultimodalRAG

router = APIRouter(prefix="/api/v1", tags=["Multimodal Knowledge Intelligence"])


def get_vector_store(request: Request) -> VectorStore:
    vs = getattr(request.app.state, "vector_store", None)
    if vs is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store backend is not initialized",
        )
    return vs


def get_rag(request: Request) -> MultimodalRAG:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multimodal RAG model is not initialized",
        )
    return rag


@router.get("/health", response_model=HealthResponse, summary="Pipeline Health Check")
def health_check(request: Request) -> HealthResponse:
    """Returns the operational status of vector storage, embedding models, and LLM backends."""
    vs = get_vector_store(request)
    rag = get_rag(request)
    counts = vs.count()

    return HealthResponse(
        status="healthy",
        vector_store_backend=vs.backend,
        indexed_chunks=counts,
        llm_provider=rag.provider,
        version="1.0.0",
    )


@router.post("/query", response_model=QueryResponse, summary="Multimodal Semantic Query")
def query_knowledge_base(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:
    """
    Performs hybrid dense+sparse retrieval across multimodal collections
    and synthesizes a grounded answer citing source chunk IDs.
    """
    vs = get_vector_store(request)
    rag = get_rag(request)

    # 1. Retrieve candidates across selected modalities
    results = vs.search(
        query=payload.query,
        top_k=payload.top_k,
        modalities=tuple(payload.modalities),
    )

    # 2. Build domain retrieval results for RAG generator
    domain_results = [
        RetrievalResult(
            chunk_id=r.chunk_id,
            content=r.content,
            modality=r.modality,
            score=r.score,
            page_number=r.page_number,
            source_path=r.source_path,
            metadata=r.metadata,
        )
        for r in results
    ]

    # 3. Generate grounded answer
    rag_answer = rag.generate_answer(payload.query, domain_results)

    # 4. Serialize response
    item_schemas = [
        RetrievalItemSchema(
            chunk_id=r.chunk_id,
            modality=r.modality,
            content=r.content,
            score=r.score,
            page_number=r.page_number,
            source_path=Path(r.source_path).name if r.source_path else "unknown",
        )
        for r in results
    ]

    return QueryResponse(
        query=payload.query,
        answer=rag_answer.answer,
        citations=rag_answer.citations,
        retrieved_results=item_schemas,
        provider=rag.provider,
        latency_ms=rag_answer.latency_ms,
    )


@router.post("/ingest", response_model=IngestResponse, summary="Upload & Index Document")
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="Document file (.pdf, .png, .jpg, .txt, .md)"),
) -> IngestResponse:
    """
    Extracts text, tables, and images from an uploaded document, generates embeddings,
    and indexes them into the vector database.
    """
    vs = get_vector_store(request)
    filename = file.filename or "uploaded_file"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported: {sorted(list(SUPPORTED_EXTENSIONS))}",
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / filename

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    target_path.write_bytes(content)

    try:
        parser = DocumentParser()
        chunks = parser.parse(target_path)
        if not chunks:
            return IngestResponse(
                filename=filename,
                chunks_indexed=0,
                modalities={},
                message="No extractable text, table, or image content found in file",
            )

        vs.index_chunks(chunks)

        modality_breakdown: dict[str, int] = {}
        for c in chunks:
            modality_breakdown[c.modality] = modality_breakdown.get(c.modality, 0) + 1

        logger.info(f"API Ingest: Successfully indexed {len(chunks)} chunks from {filename}")

        return IngestResponse(
            filename=filename,
            chunks_indexed=len(chunks),
            modalities=modality_breakdown,
            message=f"Successfully indexed {len(chunks)} chunks",
        )
    except Exception as exc:
        logger.error(f"Failed to ingest file {filename}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )
