"""Ask a question: uv run python scripts/ask.py "your question" """

import sys

from evidence_rag_assistant.generation.answer import answer_question
from evidence_rag_assistant.generation.llm import create_llm
from evidence_rag_assistant.retrieval.retriever import Retriever
from evidence_rag_assistant.settings import get_settings


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python scripts/ask.py "your question"')

    settings = get_settings()
    retriever = Retriever(settings.chunks_path, settings.index_dir, mode=settings.retrieval_mode)
    if retriever.stale:
        print("Warning: chunks.jsonl changed since the index was built. Rebuild it.\n")

    answer = answer_question(
        sys.argv[1],
        retriever,
        create_llm(settings),
        k=settings.k,
        min_score=settings.min_retrieval_score,
    )
    print(f"Status: {answer.status} (top retrieval score {answer.top_score:.3f})\n")
    print(answer.text)
    if answer.invalid_citations:
        print(f"\nWarning: citations to non-existent sources: {answer.invalid_citations}")

    print("\nSources (* = cited in the answer):")
    for number, item in enumerate(answer.sources, start=1):
        chunk = item.chunk
        section = " > ".join(chunk["section_path"]) or "Main text"
        marker = "*" if number in answer.cited else " "
        print(f" {marker}[{number}] {chunk['chunk_id']} ({item.score:.3f})")
        print(f"      {chunk['title'][:70]} — {section}")


if __name__ == "__main__":
    main()
