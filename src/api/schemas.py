"""
src/api/schemas.py
==================
Pydantic schemas for request validation and response serialization.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class RetrievalItemSchema(BaseModel):
    chunk_id: str = Field(..., description="Unique deterministic identifier of the chunk")
    modality: Literal["text", "table", "image"] = Field(..., description="Modality of the chunk")
    content: str = Field(..., description="Text content or image OCR/caption")
    score: float = Field(..., description="Hybrid retrieval score (0.0 to 1.0)")
    page_number: int = Field(default=1, description="Source page number")
    source_path: str = Field(default="", description="Path or filename of the source document")


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question or search query",
        examples=["What is the retrieval accuracy and MRR of the hybrid pipeline?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top multimodal context items to retrieve",
    )
    modalities: list[Literal["text", "table", "image"]] = Field(
        default=["text", "table", "image"],
        description="List of modalities to include in retrieval",
    )


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    retrieved_results: list[RetrievalItemSchema] = Field(default_factory=list)
    provider: str = Field(default="local")
    latency_ms: float = Field(default=0.0)


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    modalities: dict[str, int] = Field(default_factory=dict)
    message: str = "Document successfully ingested and indexed"


class HealthResponse(BaseModel):
    status: str = "healthy"
    vector_store_backend: str
    indexed_chunks: dict[str, int]
    llm_provider: str
    version: str = "1.0.0"
