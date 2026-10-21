"""Build the LitQA2 evaluation set from questions whose source papers are in the corpus.

Each kept question gets shuffled, lettered options (seeded by its ID, so the order
is stable across runs) and a permanent tune/test split assigned by hashing its ID.
Output is a local, derived copy of LitQA2 (CC BY-SA 4.0) and stays out of Git.
"""

import hashlib
import json
import random
import string
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from evidence_rag_assistant.ingestion.europe_pmc import normalize_doi

LITQA2_DIR = Path("data/raw/lab-bench/LitQA2")
CANDIDATES_PATH = Path("data/raw/litqa2_candidates.jsonl")
MANIFEST_PATH = Path("data/raw/fulltext_manifest.jsonl")
OUTPUT_PATH = Path("data/processed/litqa2_benchmark.jsonl")

INSUFFICIENT = "Insufficient information to answer this question"
TUNE_PERCENT = 40


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_for(question_id: str) -> str:
    """Assign a permanent split from a hash of the question ID."""
    bucket = int(hashlib.sha256(question_id.encode("utf-8")).hexdigest(), 16) % 100
    return "tune" if bucket < TUNE_PERCENT else "test"


def main() -> None:
    rows = pq.read_table(LITQA2_DIR).to_pylist()
    pmcid_by_doi = {c["doi"]: c["pmcid"] for c in read_jsonl(CANDIDATES_PATH) if c["pmcid"]}
    in_corpus = {
        e["pmcid"] for e in read_jsonl(MANIFEST_PATH) if e["status"] in {"stored", "cached"}
    }

    splits: Counter[str] = Counter()
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for row in rows:
            dois = [normalize_doi(source) for source in row["sources"] or []]
            pmcids = [pmcid_by_doi.get(doi) for doi in dois]
            if not pmcids or not all(pmcid in in_corpus for pmcid in pmcids):
                continue

            options = [row["ideal"], *row["distractors"], INSUFFICIENT]
            random.Random(row["id"]).shuffle(options)
            lettered = dict(zip(string.ascii_uppercase, options, strict=False))
            letter_of = {text: letter for letter, text in lettered.items()}

            item = {
                "id": row["id"],
                "split": split_for(row["id"]),
                "question": row["question"],
                "options": lettered,
                "correct": letter_of[row["ideal"]],
                "insufficient": letter_of[INSUFFICIENT],
                "source_pmcids": sorted({p for p in pmcids if p is not None}),
                "key_passage": row["key-passage"],
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            splits[item["split"]] += 1

    print(f"LitQA2 questions: {len(rows)}")
    print(f"Kept (all source papers in corpus): {sum(splits.values())}")
    print(f"Split: {dict(splits)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
