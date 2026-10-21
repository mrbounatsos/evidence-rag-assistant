"""Tests for prompt building, citation checking and the low-retrieval gate."""

from evidence_rag_assistant.generation.answer import (
    ABSTAIN_MESSAGE,
    answer_question,
    build_prompt,
    is_abstention,
    parse_citations,
)
from evidence_rag_assistant.retrieval.retriever import RetrievedChunk


def make_source(n: int, score: float = 0.9) -> RetrievedChunk:
    chunk = {
        "chunk_id": f"PMC1:{n:04d}",
        "title": "A paper",
        "section_path": ["Results"],
        "text": f"Finding number {n}.",
    }
    return RetrievedChunk(chunk, score)


def test_build_prompt_numbers_sources_from_one() -> None:
    prompt = build_prompt("What was found?", [make_source(1), make_source(2)])
    assert "[1] A paper — Results\nFinding number 1." in prompt
    assert "[2] A paper — Results\nFinding number 2." in prompt
    assert prompt.endswith("Question: What was found?")


def test_parse_citations_separates_valid_and_invalid() -> None:
    text = "A holds [1]. B holds [2][3]. C holds [1, 4]. D holds [9]."
    assert parse_citations(text, n_sources=4) == ([1, 2, 3, 4], [9])


def test_is_abstention_detects_fixed_sentence() -> None:
    assert is_abstention(ABSTAIN_MESSAGE)
    assert is_abstention(f"Sorry. {ABSTAIN_MESSAGE.upper()}")
    assert not is_abstention("DyAb works well [1].")


class LowScoreRetriever:
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return [make_source(1, score=0.4)]


def test_low_retrieval_score_abstains_without_calling_model() -> None:
    answer = answer_question("What is the capital of France?", LowScoreRetriever())  # type: ignore[arg-type]
    assert answer.status == "abstained_low_retrieval"
    assert answer.text == ABSTAIN_MESSAGE
