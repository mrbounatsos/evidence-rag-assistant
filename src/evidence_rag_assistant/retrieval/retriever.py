"""Load the dense index and chunks once, then retrieve chunks for queries."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_rag_assistant.indexing.dense_index import embed_query, load_index, load_model, search

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("data/index")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: dict[str, Any]
    score: float


class Retriever:
    def __init__(self, chunks_path: Path = CHUNKS_PATH, index_dir: Path = INDEX_DIR):
        meta = json.loads((index_dir / "dense_meta.json").read_text(encoding="utf-8"))
        raw = chunks_path.read_bytes()
        # True if the chunks changed after the index was built.
        self.stale = hashlib.sha256(raw).hexdigest() != meta["chunks_sha256"]
        self.chunks = {
            c["chunk_id"]: c
            for c in (json.loads(line) for line in raw.decode("utf-8").splitlines() if line)
        }
        self.chunk_ids: list[str] = json.loads(
            (index_dir / "chunk_ids.json").read_text(encoding="utf-8")
        )
        self.model = load_model(meta["model"])
        self.index = load_index(index_dir / "dense.faiss")

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        results = search(self.index, embed_query(self.model, query), k)
        return [RetrievedChunk(self.chunks[self.chunk_ids[row]], score) for row, score in results]
