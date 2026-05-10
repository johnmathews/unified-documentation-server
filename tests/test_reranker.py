"""Tests for the ONNX cross-encoder reranker (L2 of the hybrid search pipeline)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from docserver.reranker import OnnxCrossEncoder, get_reranker, rerank

if TYPE_CHECKING:
    from docserver.knowledge_base import SearchResult


@pytest.fixture(scope="module")
def reranker_model() -> OnnxCrossEncoder:
    """Module-scoped to avoid re-downloading/loading the model per test."""
    r = get_reranker()
    r._ensure_model()
    return r


def _candidate(doc_id: str, content: str, score: float = 0.0) -> SearchResult:
    return cast(
        "SearchResult",
        {"doc_id": doc_id, "content": content, "metadata": {}, "score": score},
    )


class TestRerank:
    def test_reorders_when_l1_top_is_wrong(self, reranker_model: OnnxCrossEncoder) -> None:
        """Cross-encoder bumps the true match above a misleading L1-top."""
        candidates = [
            _candidate("a", "Python packaging tutorial about wheels and sdists."),
            _candidate("b", "How Docker containers handle networking and DNS resolution."),
            _candidate("c", "Setting up a SQL database with stored procedures."),
        ]
        result = rerank("docker compose network", candidates, top_k=3)
        assert result[0]["doc_id"] == "b", (
            f"Cross-encoder should put the docker chunk first, got {result[0]['doc_id']}"
        )

    def test_top_k_truncation(self, reranker_model: OnnxCrossEncoder) -> None:
        candidates = [_candidate(f"doc{i}", f"content {i}") for i in range(10)]
        result = rerank("query", candidates, top_k=3)
        assert len(result) == 3

    def test_score_field_replaced_with_logit(
        self, reranker_model: OnnxCrossEncoder,
    ) -> None:
        """Each returned candidate's score is a cross-encoder logit (float)."""
        candidates = [
            _candidate("a", "irrelevant content", score=0.123),
            _candidate("b", "different content", score=0.456),
        ]
        result = rerank("query", candidates, top_k=2)
        for c in result:
            assert isinstance(c["score"], float)
            # Must be the logit, not the placeholder L1 score.
            assert c["score"] not in (0.123, 0.456)

    def test_empty_candidates(self, reranker_model: OnnxCrossEncoder) -> None:
        assert rerank("anything", [], top_k=10) == []

    def test_failure_returns_l1_unchanged(self) -> None:
        """If the reranker raises, rerank() falls back to L1[:top_k]."""
        candidates = [_candidate("a", "x"), _candidate("b", "y"), _candidate("c", "z")]
        with patch("docserver.reranker.OnnxCrossEncoder._rerank", side_effect=RuntimeError("boom")):
            result = rerank("query", candidates, top_k=2)
        assert [c["doc_id"] for c in result] == ["a", "b"]

    def test_singleton_loaded_once(self) -> None:
        """get_reranker() returns the same instance across calls."""
        a = get_reranker()
        b = get_reranker()
        assert a is b
