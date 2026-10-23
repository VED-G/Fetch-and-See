"""
DocMind — Hybrid Search with Reciprocal Rank Fusion (RRF)
===========================================================
Combines BM25 sparse retrieval and dense vector retrieval into a single
ranked list using Reciprocal Rank Fusion.

Why RRF instead of score normalisation?
  BM25 scores and cosine similarity scores live on completely different
  mathematical scales.  Score normalisation (e.g., min-max) makes both
  sit in [0, 1] but doesn't fix the distribution mismatch — a BM25 score
  of 0.8 and a cosine score of 0.8 mean very different things.

  RRF uses only RANKS (positions in each list), not raw scores.  This
  makes it robust to any score distribution and requires zero tuning.

  RRF formula:
      RRF_score(d) = Σ  1 / (k + rank_m(d))
      over all retrievers m, where k=60 (standard constant)

  Intuition: a document ranked #1 by both BM25 and vector gets
      1/(60+1) + 1/(60+1) ≈ 0.033, which beats anything ranked lower
      in either list.

Interview talking point:
  "RRF is O(n log n) — just sorting two lists and merging.  It's the same
   algorithm used in multi-stage retrieval systems at places like Google and
   LinkedIn.  Qdrant also exposes it natively, but I implemented it here so
   the logic is visible and testable."
"""

from __future__ import annotations

import structlog

from app.config import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int | None = None,
    top_k: int = 50,
) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        result_lists:  Each inner list is a ranked list of result dicts.
                       Each dict must have a "chunk_id" key.
        k:             RRF smoothing constant.  Default from settings (60).
        top_k:         Number of results to return after fusion.

    Returns:
        Merged, de-duplicated, re-ranked list of result dicts, highest RRF
        score first.  Each result also carries an "rrf_score" key.
    """
    k = k or settings.RRF_K

    # Accumulate RRF scores: chunk_id → cumulative score
    rrf_scores: dict[str, float] = {}
    # Store the best metadata we've seen for each chunk_id
    chunk_data: dict[str, dict] = {}

    for ranked_list in result_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            # Keep the version with the highest individual score as metadata
            if chunk_id not in chunk_data or item.get("score", 0) > chunk_data[chunk_id].get("score", 0):
                chunk_data[chunk_id] = item

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    results = []
    for chunk_id in sorted_ids:
        item = {**chunk_data[chunk_id], "rrf_score": rrf_scores[chunk_id]}
        results.append(item)

    log.debug(
        "rrf.fused",
        input_lists=len(result_lists),
        total_candidates=len(rrf_scores),
        output=len(results),
    )
    return results


class HybridSearcher:
    """Runs BM25 and vector search in parallel, fuses results with RRF.

    Usage:
        searcher = HybridSearcher(bm25_index, vector_store, embedder)
        results = await searcher.search(query, top_k=50)
        # results → fused list, ready for reranking
    """

    def __init__(self, bm25_index: object, vector_store: object, embedder: object) -> None:
        from app.services.retrieval.bm25_index import BM25Index
        from app.services.retrieval.vector_store import VectorStore
        from app.services.retrieval.embedder import Embedder

        self._bm25: BM25Index = bm25_index  # type: ignore[assignment]
        self._qdrant: VectorStore = vector_store  # type: ignore[assignment]
        self._embedder: Embedder = embedder  # type: ignore[assignment]

    async def search(
        self,
        query: str,
        top_k: int = 50,
        collection_filter: list[str] | None = None,
    ) -> list[dict]:
        """Run hybrid search.

        1. BM25 search  (sync, fast, CPU)
        2. Vector search (async, Qdrant)
        3. RRF fusion
        Returns fused top_k results.
        """
        import asyncio

        # BM25 is synchronous and fast — run directly
        bm25_results = self._bm25.search(query, top_k=settings.BM25_TOP_K)

        # Vector search needs the query embedding first
        query_vector = await self._embedder.embed_query(query)
        vector_results = await self._qdrant.search(
            query_vector=query_vector,
            top_k=settings.VECTOR_TOP_K,
            filter_doc_ids=collection_filter,
        )

        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=settings.RRF_K,
            top_k=top_k,
        )

        log.info(
            "hybrid.search.done",
            query_len=len(query),
            bm25_count=len(bm25_results),
            vector_count=len(vector_results),
            fused_count=len(fused),
        )
        return fused
