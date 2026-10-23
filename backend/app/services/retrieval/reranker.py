"""
Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Scores (query, chunk) pairs jointly for higher accuracy than bi-encoder search.
Only runs on the top candidates from RRF (not the full corpus) due to O(n) cost.
Set ENABLE_RERANKER=false to skip on low-RAM machines.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import structlog

from app.config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()


class Reranker:
    """Async cross-encoder reranker wrapper."""

    def __init__(self) -> None:
        self._model_name = settings.RERANKER_MODEL
        self._model = None   # Loaded lazily in warm_up()

    async def warm_up(self) -> None:
        """Load reranker model weights.  Called once at startup."""
        log.info("reranker.loading", model=self._model_name)
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        log.info("reranker.loaded", model=self._model_name)

    def _load_model(self) -> object:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(
            self._model_name,
            max_length=512,
            device=settings.EMBEDDING_DEVICE,
        )

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank a list of candidates by cross-encoder relevance score.

        Args:
            query:       The user's (possibly rewritten) query.
            candidates:  List of chunk dicts from hybrid search.
            top_k:       How many to keep.  Default from settings.

        Returns:
            Reranked list, highest relevance first.  Each item gets a
            "reranker_score" key (sigmoid-normalised, 0-1 range).
        """
        if not candidates or self._model is None:
            return candidates[:top_k or settings.RERANKER_TOP_K]

        top_k = top_k or settings.RERANKER_TOP_K
        # Only rerank as many candidates as configured (balance quality/speed)
        candidates_to_rank = candidates[:settings.RERANKER_CANDIDATE_K]

        pairs = [[query, c.get("text", "")] for c in candidates_to_rank]

        loop = asyncio.get_event_loop()
        scores: list[float] = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, convert_to_numpy=True).tolist(),  # type: ignore[union-attr]
        )

        # Attach scores and sort
        scored = sorted(
            zip(candidates_to_rank, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk, score in scored[:top_k]:
            import math
            normalised = 1.0 / (1.0 + math.exp(-score))   # sigmoid to [0, 1]
            results.append({**chunk, "reranker_score": normalised})

        log.info(
            "reranker.done",
            input_count=len(candidates_to_rank),
            output_count=len(results),
            top_score=results[0]["reranker_score"] if results else 0,
        )
        return results


def get_confidence_score(results: list[dict], reranker_enabled: bool) -> float:
    """Return the confidence score for the top result.

    With reranker:  uses reranker_score (sigmoid-normalised cross-encoder logit)
    Without:        uses rrf_score or vector cosine similarity score

    This is what gets compared against NO_ANSWER_THRESHOLD and LOW_CONFIDENCE_THRESHOLD.
    """
    if not results:
        return 0.0
    top = results[0]
    if reranker_enabled and "reranker_score" in top:
        return float(top["reranker_score"])
    # Fallback: use normalised RRF score proxy (higher RRF → higher confidence)
    rrf = top.get("rrf_score", 0.0)
    vec = top.get("score", 0.0)    # Qdrant cosine similarity
    # Take max of available signals, cap at 1.0
    return min(max(rrf * 10, vec), 1.0)   # RRF scores are tiny (0.008-0.033), scale up
