"""Embed all chunks and build the dense FAISS index."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from evidence_rag_assistant.dense_index import (
    MODEL_NAME,
    build_index,
    embed_passages,
    load_model,
    save_index,
)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("data/index")


def main() -> None:
    raw = CHUNKS_PATH.read_bytes()
    chunks = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]

    model = load_model()
    print(f"Model: {MODEL_NAME} on {model.device}")
    start = time.perf_counter()
    vectors = embed_passages(model, [c["embedding_text"] for c in chunks])
    elapsed = time.perf_counter() - start

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    save_index(build_index(vectors), INDEX_DIR / "dense.faiss")
    chunk_ids = [c["chunk_id"] for c in chunks]
    (INDEX_DIR / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")
    meta = {
        "model": MODEL_NAME,
        "dimension": int(vectors.shape[1]),
        "n_chunks": len(chunks),
        "chunks_sha256": hashlib.sha256(raw).hexdigest(),
        "built_at": datetime.now(UTC).isoformat(),
    }
    (INDEX_DIR / "dense_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Embedded {len(chunks)} chunks into {vectors.shape[1]}-dim vectors in {elapsed:.1f}s")
    print(f"Index: {INDEX_DIR / 'dense.faiss'}")


if __name__ == "__main__":
    main()
