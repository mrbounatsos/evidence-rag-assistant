"""Tests for prompt building, citation checking and every answer path."""

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


class FixedRetriever:
    """Returns three sources with a fixed score."""

    def __init__(self, score: float) -> None:
        self.score = score

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return [make_source(n, self.score) for n in range(1, 4)]


class ScriptedLLM:
    """A stand-in model that always gives the same reply and counts calls."""

    model = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def ask(score: float, reply: str) -> tuple:
    llm = ScriptedLLM(reply)
    answer = answer_question("A question?", FixedRetriever(score), llm, k=3, min_score=0.6)
    return answer, llm


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


def test_low_retrieval_score_abstains_without_calling_model() -> None:
    answer, llm = ask(score=0.4, reply="should never be used")
    assert answer.status == "abstained_low_retrieval"
    assert answer.text == ABSTAIN_MESSAGE
    assert llm.calls == 0


def test_model_abstention_is_detected() -> None:
    answer, llm = ask(score=0.9, reply=ABSTAIN_MESSAGE)
    assert answer.status == "abstained_by_model"
    assert llm.calls == 1


def test_citations_are_validated() -> None:
    answer, _ = ask(score=0.9, reply="Finding one holds [1]. Another holds [7].")
    assert answer.status == "answered"
    assert answer.cited == [1]
    assert answer.invalid_citations == [7]


def test_answer_without_valid_citation_is_ungrounded() -> None:
    answer, _ = ask(score=0.9, reply="A confident claim without any sources.")
    assert answer.status == "ungrounded"
