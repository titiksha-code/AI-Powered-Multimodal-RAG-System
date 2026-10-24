"""
tests/test_api.py
=================
Integration and unit tests for the FastAPI application endpoints.
Uses FastAPI's TestClient with mocked/lightweight backend models for fast execution.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from main import create_app
from src.retrieval.vector_store import RetrievalResult


@pytest.fixture
def test_app():
    """Create test application with mocked backend services."""
    app = create_app()

    # Mock VectorStore
    mock_vs = MagicMock()
    mock_vs.backend = "chroma"
    mock_vs.count.return_value = {"text": 6, "image": 4, "total": 10}
    mock_vs.search.return_value = [
        RetrievalResult(
            chunk_id="chunk_test_123",
            content="Hybrid retrieval combines dense and keyword similarity.",
            modality="text",
            score=0.89,
            page_number=1,
            source_path="report.pdf",
            metadata={},
        )
    ]

    # Mock MultimodalRAG
    mock_rag = MagicMock()
    mock_rag.provider = "local"
    mock_answer = MagicMock()
    mock_answer.answer = "The system utilizes hybrid retrieval for enhanced recall."
    mock_answer.citations = ["[chunk_test_123]"]
    mock_answer.latency_ms = 45.2
    mock_rag.generate_answer.return_value = mock_answer

    # Attach to app state
    app.state.vector_store = mock_vs
    app.state.rag = mock_rag

    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


class TestApiEndpoints:
    def test_root_redirects_to_docs(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (307, 302)
        assert response.headers["location"] == "/docs"

    def test_health_check_returns_healthy(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["vector_store_backend"] == "chroma"
        assert data["indexed_chunks"]["total"] == 10
        assert data["llm_provider"] == "local"

    def test_query_endpoint_success(self, client):
        payload = {
            "query": "How does hybrid retrieval work?",
            "top_k": 3,
            "modalities": ["text", "table"],
        }
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == payload["query"]
        assert "hybrid retrieval" in data["answer"].lower()
        assert len(data["citations"]) > 0
        assert len(data["retrieved_results"]) == 1
        assert data["retrieved_results"][0]["chunk_id"] == "chunk_test_123"

    def test_query_endpoint_validation_error_on_empty_query(self, client):
        payload = {
            "query": "",  # min_length is 1
            "top_k": 5,
        }
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 422

    def test_query_endpoint_validation_error_on_invalid_top_k(self, client):
        payload = {
            "query": "valid query",
            "top_k": 100,  # max is 20
        }
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 422

    def test_ingest_rejects_unsupported_extension(self, client):
        files = {"file": ("unsupported.exe", b"binarycontent", "application/octet-stream")}
        response = client.post("/api/v1/ingest", files=files)
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_ingest_rejects_empty_file(self, client):
        files = {"file": ("empty.txt", b"", "text/plain")}
        response = client.post("/api/v1/ingest", files=files)
        assert response.status_code == 400
        assert "Uploaded file is empty" in response.json()["detail"]

    def test_ingest_valid_text_file(self, client):
        content = b"This is a test document explaining multimodal RAG architecture."
        files = {"file": ("sample.txt", content, "text/plain")}
        response = client.post("/api/v1/ingest", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "sample.txt"
        assert data["chunks_indexed"] >= 1
