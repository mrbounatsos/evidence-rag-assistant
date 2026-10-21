"""Tests for LitQA2 answer parsing and key-passage matching."""

import pytest

from evidence_rag_assistant.evaluation.litqa2 import key_passage_hit, parse_choice
from evidence_rag_assistant.retrieval.retriever import RetrievedChunk

LETTERS = {"A", "B", "C", "D", "E"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Option B is ruled out by [1].\nAnswer: C", "C"),
        ("Answer: (d)", "D"),
        ("**Answer: A**", "A"),
        ("Answer: B\nOn reflection...\nAnswer: E", "E"),
        ("Answer: Insufficient information", None),
        ("Answer: F", None),
        ("I am not sure.", None),
    ],
)
def test_parse_choice(text: str, expected: str | None) -> None:
    assert parse_choice(text, LETTERS) == expected


def chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk({"chunk_id": "PMC1:0000", "pmcid": "PMC1", "text": text}, 0.9)


def test_key_passage_hit_tolerates_small_differences() -> None:
    passage = "A. lwoffii only evolved resistance to ciprofloxacin"
    assert key_passage_hit(passage, [chunk("Unrelated text."), chunk(f"In contrast, {passage}.")])
    assert not key_passage_hit(passage, [chunk("A. lwoffii was grown overnight.")])


def test_key_passage_hit_is_none_without_passage() -> None:
    assert key_passage_hit(None, [chunk("Some text.")]) is None
    assert key_passage_hit("", [chunk("Some text.")]) is None
