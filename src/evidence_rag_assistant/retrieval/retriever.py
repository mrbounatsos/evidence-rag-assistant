"""Load the dense index and chunks once, then retrieve chunks for queries."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_rag_assistant.indexing.dense_index import embed_query, load_index, load_model, search
from evidence_rag_assistant.retrieval.bm25 import BM25Index
from evidence_rag_assistant.retrieval.fusion import reciprocal_rank_fusion

RETRIEVAL_MODES = ("dense", "hybrid")

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("data/index")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: dict[str, Any]
    score: float


class Retriever:
    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        index_dir: Path = INDEX_DIR,
        mode: str = "hybrid",
    ):
        meta = json.loads((index_dir / "dense_meta.json").read_text(encoding="utf-8"))
        self.meta: dict[str, Any] = meta
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
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")
        self.mode = mode
        self.bm25 = (
            BM25Index([self.chunks[chunk_id]["embedding_text"] for chunk_id in self.chunk_ids])
            if mode == "hybrid"
            else None
        )

    def retrieve(
        self, query: str, k: int = 5, exclude_pmcids: frozenset[str] = frozenset()
    ) -> list[RetrievedChunk]:
        """Return the k best chunks, optionally skipping chunks from some papers.

        Each chunk carries its embedding similarity as `score`. In hybrid mode,
        the order comes from fusing the embedding and BM25 rankings.
        """
        dense = search(self.index, embed_query(self.model, query), self.index.ntotal)
        dense_score = dict(dense)
        dense_ranking = [row for row, _ in dense]
        if self.bm25 is None:
            ranking = dense_ranking
        else:
            fused = reciprocal_rank_fusion([dense_ranking, self.bm25.ranking(query)])
            ranking = [row for row, _ in fused]

        retrieved: list[RetrievedChunk] = []
        for row in ranking:
            chunk = self.chunks[self.chunk_ids[row]]
            if chunk["pmcid"] in exclude_pmcids:
                continue
            retrieved.append(RetrievedChunk(chunk, dense_score[row]))
            if len(retrieved) == k:
                break
        return retrieved
