"""
main.py
=======
FastAPI server entry point for the AI-Powered Multimodal Retrieval &
Knowledge Intelligence System.

Run directly with:
    python main.py
or with uvicorn:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from src.api.routes import router as api_router
from src.ingestion.parser import DocumentParser
from src.models.multimodal_rag import MultimodalRAG
from src.retrieval.vector_store import VectorStore


def initialize_backend() -> tuple[VectorStore, MultimodalRAG]:
    """Load configuration and initialize the VectorStore and MultimodalRAG pipeline."""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text())

    logger.info("Initializing VectorStore backend...")
    vector_store = VectorStore(
        backend=cfg["vector_store"]["backend"],
        host=cfg["vector_store"]["host"],
        port=cfg["vector_store"]["port"],
        collection_text=cfg["vector_store"]["collection_text"],
        collection_image=cfg["vector_store"]["collection_image"],
        hybrid_alpha=cfg["vector_store"]["hybrid_alpha"],
    )

    logger.info("Initializing MultimodalRAG generator...")
    rag = MultimodalRAG(
        provider=cfg["llm"]["provider"],
        model_name=cfg["llm"]["model_name"],
        temperature=cfg["llm"]["temperature"],
        max_new_tokens=cfg["llm"]["max_new_tokens"],
        system_prompt=cfg["llm"]["system_prompt"],
    )

    # Pre-seed vector store with Paper2 sample if empty
    counts = vector_store.count()
    sample_doc = Path(__file__).parent / "Paper2_Multimodal_RAG.pdf"
    if counts["total"] == 0 and sample_doc.exists():
        logger.info(f"Vector store is empty. Auto-seeding with {sample_doc.name}...")
        parser = DocumentParser()
        chunks = parser.parse(sample_doc)
        vector_store.index_chunks(chunks)

    return vector_store, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle application startup and shutdown."""
    logger.info("Starting up Multimodal RAG API Service...")
    vector_store, rag = initialize_backend()
    app.state.vector_store = vector_store
    app.state.rag = rag
    yield
    logger.info("Shutting down Multimodal RAG API Service...")


def create_app() -> FastAPI:
    """Application factory for the FastAPI service."""
    app = FastAPI(
        title="AI-Powered Multimodal Retrieval & Knowledge Intelligence API",
        description=(
            "Production-grade RESTful API for multimodal document ingestion, "
            "hybrid dense+sparse vector search, and cited cross-modal answer generation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Allow cross-origin requests from frontend dashboards or external microservices
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the API router
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def root_redirect():
        """Redirect root requests to interactive Swagger documentation."""
        return RedirectResponse(url="/docs")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
