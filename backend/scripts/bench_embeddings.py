"""
Embedding throughput benchmark.

Measures how many chunks per second the embedder can process at different
batch sizes. Helps you find the optimal EMBEDDING_BATCH_SIZE for your PC.

Usage:
    cd backend
    python scripts/bench_embeddings.py
    python scripts/bench_embeddings.py --chunks 500 --batches 8,16,32
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_TEXT = (
    "This is a sample document chunk for benchmarking the embedding throughput. "
    "It contains a realistic amount of text similar to what would be extracted "
    "from a PDF document during the ingestion pipeline. The embedding model "
    "encodes this into a fixed-size dense vector for similarity search."
)


async def run_bench(n_chunks: int, batch_sizes: list[int]) -> None:
    from app.services.retrieval.embedder import Embedder

    embedder = Embedder()
    print("Warming up embedding model...")
    await embedder.warm_up()

    texts = [SAMPLE_TEXT] * n_chunks

    print(f"\nEmbedding {n_chunks} chunks at different batch sizes")
    print("─" * 45)
    print(f"{'Batch size':>12} {'Time (s)':>10} {'Chunks/sec':>12}")
    print("─" * 45)

    best: tuple[int, float] = (0, 0.0)

    for batch_size in batch_sizes:
        # Temporarily patch batch size
        import sentence_transformers
        t0 = time.monotonic()
        # Embed in manual batches to test different sizes
        for start in range(0, n_chunks, batch_size):
            batch = texts[start: start + batch_size]
            await embedder.embed_documents(batch)
        elapsed = time.monotonic() - t0
        cps = n_chunks / elapsed
        print(f"  {batch_size:>10}   {elapsed:>9.2f}s   {cps:>10.1f}")
        if cps > best[1]:
            best = (batch_size, cps)

    print("─" * 45)
    print(f"  Best batch size: {best[0]} ({best[1]:.1f} chunks/sec)")
    print(f"\n  Set EMBEDDING_BATCH_SIZE={best[0]} in .env for optimal throughput.")
    print("\nCopy 'chunks/sec' into PROGRESS.md → Phase 5 metrics.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding throughput benchmark")
    parser.add_argument("--chunks", type=int, default=200, help="Total chunks to embed")
    parser.add_argument(
        "--batches",
        type=str,
        default="8,16,32",
        help="Comma-separated batch sizes to test",
    )
    args = parser.parse_args()
    batch_sizes = [int(b) for b in args.batches.split(",")]
    asyncio.run(run_bench(args.chunks, batch_sizes))


if __name__ == "__main__":
    main()
