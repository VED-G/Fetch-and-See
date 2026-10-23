"""
RAGAS evaluation script.

Measures RAG quality across four dimensions on a ground-truth QA dataset:
  - Faithfulness:      are claims in the answer supported by the retrieved context?
  - Answer Relevancy:  does the answer actually address the question?
  - Context Precision: are the most relevant chunks ranked highest?
  - Context Recall:    does the retrieved context contain all needed information?

Usage:
    cd backend
    python scripts/eval_ragas.py --dataset tests/fixtures/eval_dataset.json
    python scripts/eval_ragas.py --dataset tests/fixtures/eval_dataset.json --mode vector_only
    python scripts/eval_ragas.py --dataset tests/fixtures/eval_dataset.json --mode hybrid
    python scripts/eval_ragas.py --dataset tests/fixtures/eval_dataset.json --mode hybrid_rerank

Modes run the same query through different retrieval configurations so you
can produce the ablation table for your resume and PROGRESS.md.

Output: JSON results + a markdown table printed to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

RetrievalMode = Literal["vector_only", "hybrid", "hybrid_rerank"]


async def run_eval(dataset_path: Path, mode: RetrievalMode, output_path: Path | None) -> None:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError:
        print("ERROR: Install eval deps first: pip install ragas datasets")
        sys.exit(1)

    from app.config import get_settings
    from app.services.retrieval.embedder import Embedder
    from app.services.retrieval.vector_store import VectorStore
    from app.services.retrieval.bm25_index import BM25Index
    from app.services.retrieval.hybrid_search import HybridSearcher, reciprocal_rank_fusion
    from app.services.generation.providers.base import get_provider

    settings = get_settings()

    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"\nEvaluating {len(dataset)} QA pairs | mode={mode}")
    print("─" * 50)

    # Initialise services
    embedder = Embedder()
    await embedder.warm_up()

    vector_store = VectorStore()
    await vector_store.ensure_collection()

    bm25_index = BM25Index()
    llm = get_provider()

    reranker = None
    if mode == "hybrid_rerank" and settings.ENABLE_RERANKER:
        from app.services.retrieval.reranker import Reranker
        reranker = Reranker()
        await reranker.warm_up()
    elif mode == "hybrid_rerank" and not settings.ENABLE_RERANKER:
        print("WARNING: ENABLE_RERANKER=false in .env — running as hybrid (no reranker)")
        mode = "hybrid"

    # Build RAGAS rows
    rows: list[dict] = []
    total_retrieval_ms = 0
    total_generation_ms = 0

    for i, item in enumerate(dataset):
        question: str = item["question"]
        ground_truth: str = item["ground_truth"]
        print(f"  [{i+1:02d}/{len(dataset)}] {question[:60]}...", end=" ", flush=True)

        # Retrieval
        t0 = time.monotonic()
        query_vec = await embedder.embed_query(question)

        if mode == "vector_only":
            candidates = await vector_store.search(
                query_vector=query_vec,
                top_k=settings.VECTOR_TOP_K,
                collection_name=settings.QDRANT_COLLECTION,
            )
        else:
            bm25_hits = bm25_index.search(question, top_k=settings.BM25_TOP_K)
            vec_hits = await vector_store.search(
                query_vector=query_vec,
                top_k=settings.VECTOR_TOP_K,
                collection_name=settings.QDRANT_COLLECTION,
            )
            candidates = reciprocal_rank_fusion(
                [bm25_hits, vec_hits],
                k=settings.RRF_K,
                top_k=settings.RERANKER_CANDIDATE_K,
            )

        if reranker is not None and candidates:
            candidates = await reranker.rerank(
                question, candidates, top_k=settings.RERANKER_TOP_K
            )
        else:
            candidates = candidates[:settings.RERANKER_TOP_K]

        retrieval_ms = int((time.monotonic() - t0) * 1000)
        total_retrieval_ms += retrieval_ms

        # Extract context texts
        contexts = [c.get("text", "") for c in candidates]

        # Generation
        from app.services.generation.prompt import build_messages
        t1 = time.monotonic()
        messages = build_messages(question, candidates, conversation_history=None)
        answer = await llm.complete(messages, max_tokens=512, temperature=0.0)
        generation_ms = int((time.monotonic() - t1) * 1000)
        total_generation_ms += generation_ms

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

        print(f"retrieval={retrieval_ms}ms gen={generation_ms}ms")

    # Run RAGAS
    print("\nRunning RAGAS metrics...")
    hf_dataset = Dataset.from_list(rows)
    result = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    # Summary
    n = len(dataset)
    avg_retrieval = total_retrieval_ms / n
    avg_generation = total_generation_ms / n

    print(f"\n{'='*50}")
    print(f"Mode: {mode}  |  N={n}")
    print(f"{'='*50}")
    print(f"  Faithfulness:      {result['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {result['answer_relevancy']:.3f}")
    print(f"  Context Precision: {result['context_precision']:.3f}")
    print(f"  Context Recall:    {result['context_recall']:.3f}")
    print(f"  Avg retrieval:     {avg_retrieval:.0f} ms")
    print(f"  Avg generation:    {avg_generation:.0f} ms")
    print(f"{'='*50}")

    # Markdown table row (copy into PROGRESS.md)
    print("\nMarkdown row for PROGRESS.md:")
    print(
        f"| {mode} | {result['faithfulness']:.3f} | {result['answer_relevancy']:.3f} "
        f"| {result['context_precision']:.3f} | {result['context_recall']:.3f} "
        f"| {avg_retrieval:.0f}ms |"
    )

    # Save JSON
    if output_path is None:
        output_path = Path("docs") / f"eval_{mode}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "mode": mode,
                "n": n,
                "faithfulness": result["faithfulness"],
                "answer_relevancy": result["answer_relevancy"],
                "context_precision": result["context_precision"],
                "context_recall": result["context_recall"],
                "avg_retrieval_ms": avg_retrieval,
                "avg_generation_ms": avg_generation,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for DocMind")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/fixtures/eval_dataset.json"),
        help="Path to QA dataset JSON",
    )
    parser.add_argument(
        "--mode",
        choices=["vector_only", "hybrid", "hybrid_rerank"],
        default="hybrid_rerank",
        help="Retrieval mode to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: docs/eval_{mode}.json)",
    )
    args = parser.parse_args()
    asyncio.run(run_eval(args.dataset, args.mode, args.output))


if __name__ == "__main__":
    main()
