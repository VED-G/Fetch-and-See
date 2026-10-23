"""
Search debug router.

Exposes raw retrieval results (BM25, vector, hybrid, reranked) without
calling the LLM. Useful for evaluating retrieval quality and tuning
chunk sizes, BM25 weight, and confidence thresholds.
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_auth
from app.config import get_settings
from app.models.db import APIKey
from app.models.schemas import SearchDebugRequest, SearchDebugResponse
from app.services.retrieval.hybrid_search import HybridSearcher, reciprocal_rank_fusion

log = structlog.get_logger(__name__)
router = APIRouter()
settings = get_settings()


@router.post(
    "/search/debug",
    response_model=SearchDebugResponse,
    summary="Raw retrieval results without LLM (for debugging and eval)",
)
async def debug_search(
    request: Request,
    body: SearchDebugRequest,
    db: AsyncSession = Depends(get_db),
    _key: APIKey = Depends(require_auth),
) -> SearchDebugResponse:
    """Return raw BM25, vector, and hybrid results for a query.

    Also returns per-stage latencies so you can see where time is spent.
    Use this during evaluation to understand retrieval quality before
    worrying about generation quality.
    """
    bm25_index = getattr(request.app.state, "bm25_index", None)
    vector_store = request.app.state.vector_store
    embedder = request.app.state.embedder
    reranker = request.app.state.reranker

    # Embed query
    t0 = time.monotonic()
    query_vec = await embedder.embed_query(body.query)
    embed_ms = int((time.monotonic() - t0) * 1000)

    # BM25 retrieval
    t1 = time.monotonic()
    bm25_results: list[dict] = []
    if bm25_index is not None:
        bm25_results = bm25_index.search(body.query, top_k=body.top_k or settings.BM25_TOP_K)
    bm25_ms = int((time.monotonic() - t1) * 1000)

    # Vector retrieval
    t2 = time.monotonic()
    vector_results = await vector_store.search(
        query_vector=query_vec,
        top_k=body.top_k or settings.VECTOR_TOP_K,
        collection_name=settings.QDRANT_COLLECTION,
    )
    vector_ms = int((time.monotonic() - t2) * 1000)

    # RRF fusion
    t3 = time.monotonic()
    hybrid_results = reciprocal_rank_fusion(
        [bm25_results, vector_results],
        k=settings.RRF_K,
        top_k=body.top_k or settings.RERANKER_CANDIDATE_K,
    )
    rrf_ms = int((time.monotonic() - t3) * 1000)

    # Optional reranking
    reranked_results: list[dict] = []
    rerank_ms = 0
    if reranker is not None and hybrid_results:
        t4 = time.monotonic()
        reranked_results = await reranker.rerank(
            body.query, hybrid_results, top_k=body.top_k or settings.RERANKER_TOP_K
        )
        rerank_ms = int((time.monotonic() - t4) * 1000)

    return SearchDebugResponse(
        query=body.query,
        bm25_results=bm25_results[:10],
        vector_results=vector_results[:10],
        hybrid_results=hybrid_results[:10],
        reranked_results=reranked_results,
        latency={
            "embed_ms": embed_ms,
            "bm25_ms": bm25_ms,
            "vector_ms": vector_ms,
            "rrf_ms": rrf_ms,
            "rerank_ms": rerank_ms,
            "total_ms": embed_ms + bm25_ms + vector_ms + rrf_ms + rerank_ms,
        },
    )
