"""
Query latency benchmark.

Sends N queries through the full RAG pipeline (no streaming, direct service
calls) and reports p50/p95/p99 latency broken down by stage:
  embed → retrieval → rerank → generate → total

Usage:
    cd backend
    python scripts/bench_query.py --queries 50
    python scripts/bench_query.py --queries 100 --no-rerank
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Sample queries for the benchmark. Add domain-specific queries after ingesting
# your own documents.
BENCH_QUERIES = [
    "What is the refund policy?",
    "How do I contact support?",
    "What are the payment options?",
    "Is there a free trial available?",
    "How long does delivery take?",
    "What documents are required?",
    "Can I cancel my subscription?",
    "What is the warranty period?",
    "How do I reset my password?",
    "Are there any hidden fees?",
]


async def run_bench(n_queries: int, use_rerank: bool) -> None:
    from app.config import get_settings
    from app.services.retrieval.embedder import Embedder
    from app.services.retrieval.vector_store import VectorStore
    from app.services.retrieval.bm25_index import BM25Index
    from app.services.retrieval.hybrid_search import HybridSearcher, reciprocal_rank_fusion
    from app.services.generation.providers.base import get_provider
    from app.services.generation.prompt import build_messages

    settings = get_settings()
    embedder = Embedder()
    await embedder.warm_up()
    vector_store = VectorStore()
    await vector_store.ensure_collection()
    bm25 = BM25Index()
    llm = get_provider()

    reranker = None
    if use_rerank and settings.ENABLE_RERANKER:
        from app.services.retrieval.reranker import Reranker
        reranker = Reranker()
        await reranker.warm_up()

    queries = (BENCH_QUERIES * ((n_queries // len(BENCH_QUERIES)) + 1))[:n_queries]

    embed_times: list[float] = []
    retrieval_times: list[float] = []
    rerank_times: list[float] = []
    gen_times: list[float] = []
    total_times: list[float] = []

    print(f"\nBenchmarking {n_queries} queries | reranker={use_rerank}")
    print("─" * 50)

    for i, query in enumerate(queries):
        t_start = time.monotonic()

        # Embed
        t0 = time.monotonic()
        qvec = await embedder.embed_query(query)
        embed_ms = (time.monotonic() - t0) * 1000

        # Retrieval
        t1 = time.monotonic()
        bm25_hits = bm25.search(query, top_k=settings.BM25_TOP_K)
        vec_hits = await vector_store.search(qvec, top_k=settings.VECTOR_TOP_K,
                                             collection_name=settings.QDRANT_COLLECTION)
        candidates = reciprocal_rank_fusion([bm25_hits, vec_hits], k=settings.RRF_K,
                                            top_k=settings.RERANKER_CANDIDATE_K)
        retrieval_ms = (time.monotonic() - t1) * 1000

        # Rerank
        t2 = time.monotonic()
        if reranker and candidates:
            candidates = await reranker.rerank(query, candidates, top_k=settings.RERANKER_TOP_K)
        else:
            candidates = candidates[:settings.RERANKER_TOP_K]
        rerank_ms = (time.monotonic() - t2) * 1000

        # Generate
        t3 = time.monotonic()
        msgs = build_messages(query, candidates)
        await llm.complete(msgs, max_tokens=256, temperature=0.0)
        gen_ms = (time.monotonic() - t3) * 1000

        total_ms = (time.monotonic() - t_start) * 1000

        embed_times.append(embed_ms)
        retrieval_times.append(retrieval_ms)
        rerank_times.append(rerank_ms)
        gen_times.append(gen_ms)
        total_times.append(total_ms)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_queries} done | last total={total_ms:.0f}ms")

    def pct(data: list[float], p: int) -> float:
        return round(statistics.quantiles(data, n=100)[p - 1], 1)

    print(f"\n{'='*55}")
    print(f"{'Stage':<18} {'p50':>8} {'p95':>8} {'p99':>8}  (ms)")
    print(f"{'─'*55}")
    for label, times in [
        ("embed", embed_times),
        ("retrieval (BM25+vec)", retrieval_times),
        ("rerank", rerank_times),
        ("generation", gen_times),
        ("TOTAL", total_times),
    ]:
        print(f"  {label:<18} {pct(times,50):>6.0f}   {pct(times,95):>6.0f}   {pct(times,99):>6.0f}")
    print(f"{'='*55}")
    print(f"\nN={n_queries} | reranker={'on' if use_rerank else 'off'}")
    print("\nCopy p50/p95/p99 totals into PROGRESS.md → Phase 5 metrics.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and See query latency benchmark")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries to run")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranker")
    args = parser.parse_args()
    asyncio.run(run_bench(args.queries, not args.no_rerank))


if __name__ == "__main__":
    main()
