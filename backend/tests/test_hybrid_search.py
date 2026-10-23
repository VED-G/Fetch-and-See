"""
Tests for hybrid search RRF fusion logic.
No model or network calls — pure algorithmic tests.
"""

from __future__ import annotations

import pytest
from app.services.retrieval.hybrid_search import reciprocal_rank_fusion


def _make_hits(ids: list[str], scores: list[float] | None = None) -> list[dict]:
    return [
        {"chunk_id": cid, "score": (scores[i] if scores else 1.0), "text": f"text {cid}"}
        for i, cid in enumerate(ids)
    ]


def test_rrf_combines_two_lists() -> None:
    bm25  = _make_hits(["a", "b", "c"])
    dense = _make_hits(["c", "a", "d"])
    result = reciprocal_rank_fusion([bm25, dense], k=60, top_k=10)
    ids = [r["chunk_id"] for r in result]
    # "a" and "c" appear in both lists → should rank highest
    assert ids[0] in ("a", "c")
    assert ids[1] in ("a", "c")


def test_rrf_higher_k_compresses_scores() -> None:
    hits = _make_hits(["x", "y", "z"])
    r_low  = reciprocal_rank_fusion([hits], k=1,  top_k=3)
    r_high = reciprocal_rank_fusion([hits], k=100, top_k=3)
    # Higher k → lower scores but same ordering
    assert r_low[0]["rrf_score"] > r_high[0]["rrf_score"]
    assert [r["chunk_id"] for r in r_low] == [r["chunk_id"] for r in r_high]


def test_rrf_top_k_limits_results() -> None:
    hits = _make_hits([str(i) for i in range(20)])
    result = reciprocal_rank_fusion([hits], k=60, top_k=5)
    assert len(result) == 5


def test_rrf_empty_lists_returns_empty() -> None:
    result = reciprocal_rank_fusion([[], []], k=60, top_k=10)
    assert result == []


def test_rrf_single_list_preserves_order() -> None:
    hits = _make_hits(["a", "b", "c", "d"])
    result = reciprocal_rank_fusion([hits], k=60, top_k=4)
    ids = [r["chunk_id"] for r in result]
    assert ids == ["a", "b", "c", "d"]


def test_rrf_scores_are_positive() -> None:
    hits = _make_hits(["x", "y"])
    result = reciprocal_rank_fusion([hits], k=60, top_k=2)
    for r in result:
        assert r["rrf_score"] > 0
