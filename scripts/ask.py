"""Ask a question: uv run python scripts/ask.py "your question" """

import sys

from evidence_rag_assistant.generation.answer import answer_question
from evidence_rag_assistant.retrieval.retriever import Retriever


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python scripts/ask.py "your question"')

    retriever = Retriever()
    if retriever.stale:
        print("Warning: chunks.jsonl changed since the index was built. Rebuild it.\n")

    answer = answer_question(sys.argv[1], retriever)
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
