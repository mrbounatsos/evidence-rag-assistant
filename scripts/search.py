"""Search the dense index: uv run python scripts/search.py "your question" [k]"""

import hashlib
import json
import sys
from pathlib import Path

from evidence_rag_assistant.dense_index import (
    embed_query,
    load_index,
    load_model,
    search,
)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("data/index")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python scripts/search.py "your question" [k]')
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    meta = json.loads((INDEX_DIR / "dense_meta.json").read_text(encoding="utf-8"))
    raw = CHUNKS_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["chunks_sha256"]:
        print("Warning: chunks.jsonl changed since the index was built. Rebuild it.\n")
    chunks = {
        c["chunk_id"]: c
        for c in (json.loads(line) for line in raw.decode("utf-8").splitlines() if line)
    }
    chunk_ids = json.loads((INDEX_DIR / "chunk_ids.json").read_text(encoding="utf-8"))

    model = load_model(meta["model"])
    index = load_index(INDEX_DIR / "dense.faiss")
    results = search(index, embed_query(model, query), k)

    for rank, (row, score) in enumerate(results, start=1):
        chunk = chunks[chunk_ids[row]]
        section = " > ".join(chunk["section_path"]) or "Main text"
        print(f"{rank}. [{score:.3f}] {chunk['chunk_id']}  {chunk['title'][:70]}")
        print(f"   {section}")
        print(f"   {chunk['text'][:250]}...\n")


if __name__ == "__main__":
    main()
