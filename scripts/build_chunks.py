"""Build retrieval chunks from extracted blocks."""

import json
import statistics
from dataclasses import asdict
from pathlib import Path

from evidence_rag_assistant.chunking import DEFAULT_MAX_WORDS, chunk_article

BLOCKS_PATH = Path("data/processed/blocks.jsonl")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def main() -> None:
    articles: dict[str, list[dict]] = {}
    with BLOCKS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            block = json.loads(line)
            articles.setdefault(block["pmcid"], []).append(block)

    sizes: list[int] = []
    with CHUNKS_PATH.open("w", encoding="utf-8") as out:
        for blocks in articles.values():
            licence = blocks[0]["licence"]
            source_sha256 = blocks[0]["source_sha256"]
            for chunk in chunk_article(blocks, DEFAULT_MAX_WORDS):
                record = {
                    "chunk_id": chunk.chunk_id,
                    **asdict(chunk),
                    "embedding_text": chunk.embedding_text(),
                    "licence": licence,
                    "source_sha256": source_sha256,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                sizes.append(len(chunk.text.split()))

    print(f"Articles: {len(articles)}")
    print(f"Chunks: {len(sizes)}")
    print(
        f"Words per chunk: min {min(sizes)}, median {statistics.median(sizes):.0f}, "
        f"max {max(sizes)}"
    )
    print(f"Output: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
