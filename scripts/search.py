"""Search the dense index: uv run python scripts/search.py "your question" [k]"""

import sys

from evidence_rag_assistant.retrieval.retriever import Retriever
from evidence_rag_assistant.settings import get_settings


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python scripts/search.py "your question" [k]')
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    settings = get_settings()
    retriever = Retriever(settings.chunks_path, settings.index_dir, mode=settings.retrieval_mode)
    if retriever.stale:
        print("Warning: chunks.jsonl changed since the index was built. Rebuild it.\n")

    for rank, item in enumerate(retriever.retrieve(sys.argv[1], k), start=1):
        chunk = item.chunk
        section = " > ".join(chunk["section_path"]) or "Main text"
        print(f"{rank}. [{item.score:.3f}] {chunk['chunk_id']}  {chunk['title'][:70]}")
        print(f"   {section}")
        print(f"   {chunk['text'][:250]}...\n")


if __name__ == "__main__":
    main()
