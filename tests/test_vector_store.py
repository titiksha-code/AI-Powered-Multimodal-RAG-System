"""
tests/test_vector_store.py
============================
Unit tests for src/retrieval/vector_store.py

Heavy model/backend calls are mocked out so these tests run fast and
without GPU / network access, per standard CI practice.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval.vector_store import RetrievalResult, VectorStore, _keyword_overlap_score


class TestKeywordOverlapScore:
    def test_identical_strings_score_one(self):
        assert _keyword_overlap_score("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint_strings_score_zero(self):
        assert _keyword_overlap_score("apple", "banana") == 0.0

    def test_empty_query_returns_zero(self):
        assert _keyword_overlap_score("", "some content") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        score = _keyword_overlap_score("machine learning models", "deep learning models rock")
        assert 0.0 < score < 1.0


@pytest.fixture
def mock_embedding_engine():
    engine = MagicMock()
    engine.embed_query.return_value = np.array([0.1, 0.2, 0.3])
    engine.embed_query_for_image_search.return_value = np.array([0.4, 0.5, 0.6])
    engine.text_model.get_sentence_embedding_dimension.return_value = 384
    return engine


class TestVectorStoreHybridRescore:
    def test_hybrid_rescore_orders_by_combined_score(self, mock_embedding_engine):
        with patch("src.retrieval.vector_store.VectorStore._init_qdrant"):
            store = VectorStore(backend="qdrant", embedding_engine=mock_embedding_engine)

        raw_hits = [
            {"score": 0.9, "content": "irrelevant filler", "chunk_id": "a", "modality": "text"},
            {"score": 0.5, "content": "machine learning models", "chunk_id": "b", "modality": "text"},
        ]

        results = store._hybrid_rescore("machine learning", raw_hits, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Result "b" should benefit from keyword overlap and be competitive despite lower dense score
        assert results[0].score >= results[1].score

    def test_search_returns_at_most_top_k(self, mock_embedding_engine):
        with patch("src.retrieval.vector_store.VectorStore._init_qdrant"):
            store = VectorStore(backend="qdrant", embedding_engine=mock_embedding_engine)

        with patch.object(store, "_raw_search", return_value=[
            {"score": 0.8, "content": "a", "chunk_id": "1", "modality": "text"},
            {"score": 0.7, "content": "b", "chunk_id": "2", "modality": "text"},
            {"score": 0.6, "content": "c", "chunk_id": "3", "modality": "text"},
        ]):
            results = store.search("test query", top_k=2, modalities=("text",))

        assert len(results) <= 2

    def test_point_id_is_stable_and_int(self):
        pid1 = VectorStore._to_point_id("abc123")
        pid2 = VectorStore._to_point_id("abc123")
        assert isinstance(pid1, int)
        assert pid1 == pid2
