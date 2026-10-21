"""Evaluate on LitQA2.

Usage: uv run python scripts/evaluate_litqa2.py --split tune [--limit N] [--k 5]
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evidence_rag_assistant.evaluation.litqa2 import MODES, run_question, summarize
from evidence_rag_assistant.generation.answer import MIN_RETRIEVAL_SCORE, OLLAMA_MODEL
from evidence_rag_assistant.retrieval.retriever import RETRIEVAL_MODES, Retriever

BENCHMARK_PATH = Path("data/processed/litqa2_benchmark.jsonl")
RESULTS_DIR = Path("experiments/results")


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "uncommitted_changes": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["tune", "test"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--retrieval", choices=RETRIEVAL_MODES, default="dense")
    args = parser.parse_args()

    with BENCHMARK_PATH.open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    items = [item for item in items if item["split"] == args.split][: args.limit]

    retriever = Retriever(mode=args.retrieval)
    if retriever.stale:
        sys.exit("The index is stale (chunks changed since it was built). Rebuild it first.")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"_{args.split}_{args.retrieval}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True)
    config = {
        "split": args.split,
        "n_questions": len(items),
        "modes": args.modes,
        "k": args.k,
        "min_retrieval_score": MIN_RETRIEVAL_SCORE,
        "llm": OLLAMA_MODEL,
        "embedding_model": retriever.meta["model"],
        "chunks_sha256": retriever.meta["chunks_sha256"],
        "git": git_state(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    total = len(items) * len(args.modes)
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as out:
        for item in items:
            for mode in args.modes:
                result = run_question(item, retriever, mode, k=args.k)
                results.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{len(results)}/{total}] {mode:<15} {result['outcome']}")

    summary = summarize(results)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for mode, stats in summary.items():
        print(f"\n{mode}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
