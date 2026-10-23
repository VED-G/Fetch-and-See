"""
Ablation runner: evaluate all three retrieval modes in sequence and
print a combined comparison table.

Usage:
    cd backend
    python scripts/run_ablation.py --dataset tests/fixtures/eval_dataset.json

Runs eval_ragas.py internally for each mode, collects results, and
prints a markdown table you can paste directly into PROGRESS.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_all(dataset_path: Path) -> None:
    # Import inline to avoid circular deps at module level
    from scripts.eval_ragas import run_eval

    modes = ["vector_only", "hybrid", "hybrid_rerank"]
    results: list[dict] = []

    for mode in modes:
        output = Path("docs") / f"eval_{mode}.json"
        await run_eval(dataset_path, mode, output)  # type: ignore[arg-type]
        with open(output) as f:
            results.append(json.load(f))

    # Print combined markdown table
    print("\n\n## Ablation Results\n")
    print("| Mode | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall | Avg Retrieval |")
    print("|------|-------------|-------------|---------------|------------|---------------|")
    for r in results:
        print(
            f"| {r['mode']} "
            f"| {r['faithfulness']:.3f} "
            f"| {r['answer_relevancy']:.3f} "
            f"| {r['context_precision']:.3f} "
            f"| {r['context_recall']:.3f} "
            f"| {r['avg_retrieval_ms']:.0f}ms |"
        )
    print()
    print("Copy this table into PROGRESS.md under Phase 5.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/fixtures/eval_dataset.json"),
    )
    args = parser.parse_args()
    asyncio.run(run_all(args.dataset))


if __name__ == "__main__":
    main()
