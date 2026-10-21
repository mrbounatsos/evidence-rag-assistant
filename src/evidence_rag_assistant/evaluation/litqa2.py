"""LitQA2 multiple-choice evaluation in three modes.

- rag: the model answers from the top retrieved chunks.
- source_removed: the source paper is excluded from retrieval, so the correct
  behaviour is choosing the "insufficient information" option.
- closed_book: no sources; correct answers here come from the model's memory.
"""

import re
from collections import Counter
from statistics import mean
from typing import Any

from evidence_rag_assistant.generation.answer import (
    MIN_RETRIEVAL_SCORE,
    build_prompt,
    generate,
    parse_citations,
)
from evidence_rag_assistant.retrieval.retriever import RetrievedChunk, Retriever

MODES = ("rag", "source_removed", "closed_book")

RAG_SYSTEM_PROMPT = "\n".join(
    [
        "You are a biomedical research assistant. Answer the multiple-choice question "
        "using ONLY the numbered sources provided.",
        "",
        "Rules:",
        "- Use only information stated in the sources, not outside knowledge.",
        "- If the sources do not contain the answer, choose the option saying there is "
        "insufficient information.",
        "- Cite the sources that support your choice in square brackets, e.g. [2].",
        "- End your reply with a final line of the form: Answer: <letter>",
    ]
)

CLOSED_BOOK_SYSTEM_PROMPT = "\n".join(
    [
        "You are a biomedical research assistant. Answer the multiple-choice question "
        "from your own knowledge.",
        "",
        "Rules:",
        "- If you do not know the answer, choose the option saying there is "
        "insufficient information.",
        "- End your reply with a final line of the form: Answer: <letter>",
    ]
)

# "Answer: C", "Answer: (c)", "**Answer: C**" — but not the "I" of "Answer: Insufficient".
_ANSWER = re.compile(r"Answer:\s*\**\s*\(?([A-Z])\)?(?![A-Z])", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}) {text}" for letter, text in options.items())


def build_mc_prompt(
    question: str, options: dict[str, str], sources: list[RetrievedChunk] | None
) -> str:
    question_block = f"{question}\n\nOptions:\n{format_options(options)}"
    if sources is None:
        return f"Question: {question_block}"
    return build_prompt(question_block, sources)


def parse_choice(text: str, valid_letters: set[str]) -> str | None:
    """Return the letter on the last 'Answer:' line, or None if absent or invalid."""
    matches = _ANSWER.findall(text)
    if not matches:
        return None
    letter = str(matches[-1]).upper()
    return letter if letter in valid_letters else None


def key_passage_hit(
    key_passage: str, sources: list[RetrievedChunk], threshold: float = 0.8
) -> bool:
    """True if one chunk contains at least `threshold` of the key passage's words."""
    words = set(_WORD.findall(key_passage.lower()))
    if not words:
        return False
    for source in sources:
        chunk_words = set(_WORD.findall(source.chunk["text"].lower()))
        if len(words & chunk_words) / len(words) >= threshold:
            return True
    return False


def _grade(result: dict[str, Any]) -> dict[str, Any]:
    choice = result["choice"]
    if choice is None:
        outcome = "invalid"
    elif choice == result["insufficient_option"]:
        outcome = "abstained"
    elif choice == result["correct_option"]:
        outcome = "correct"
    else:
        outcome = "wrong"
    result["outcome"] = outcome
    result["is_expected"] = choice == result["expected"]
    return result


def run_question(
    item: dict[str, Any],
    retriever: Retriever,
    mode: str,
    k: int = 5,
    min_score: float = MIN_RETRIEVAL_SCORE,
) -> dict[str, Any]:
    """Run one benchmark question in one mode and grade it."""
    result: dict[str, Any] = {
        "id": item["id"],
        "split": item["split"],
        "mode": mode,
        "expected": item["insufficient"] if mode == "source_removed" else item["correct"],
        "correct_option": item["correct"],
        "insufficient_option": item["insufficient"],
        "gated": False,
        "citations": [],
        "invalid_citations": [],
    }

    sources: list[RetrievedChunk] | None = None
    system = CLOSED_BOOK_SYSTEM_PROMPT
    if mode != "closed_book":
        exclude = frozenset(item["source_pmcids"]) if mode == "source_removed" else frozenset()
        sources = retriever.retrieve(item["question"], k, exclude_pmcids=exclude)
        pmcids = [s.chunk["pmcid"] for s in sources]
        result["top_score"] = sources[0].score if sources else 0.0
        result["retrieved_chunk_ids"] = [s.chunk["chunk_id"] for s in sources]
        result["distinct_papers"] = len(set(pmcids))
        if mode == "rag":
            result["source_hit"] = any(p in item["source_pmcids"] for p in pmcids)
            result["key_passage_hit"] = key_passage_hit(item["key_passage"], sources)
        if result["top_score"] < min_score:
            result.update(gated=True, choice=item["insufficient"], raw_answer="")
            return _grade(result)
        system = RAG_SYSTEM_PROMPT

    text = generate(system, build_mc_prompt(item["question"], item["options"], sources))
    result["raw_answer"] = text
    result["choice"] = parse_choice(text, set(item["options"]))
    if sources is not None:
        result["citations"], result["invalid_citations"] = parse_citations(text, len(sources))
    return _grade(result)


def summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-question results into metrics per mode."""
    summary: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        rows = [r for r in results if r["mode"] == mode]
        if not rows:
            continue
        outcomes = Counter(r["outcome"] for r in rows)
        answered = outcomes["correct"] + outcomes["wrong"]
        stats: dict[str, Any] = {
            "n": len(rows),
            **{name: outcomes[name] for name in ("correct", "wrong", "abstained", "invalid")},
            "accuracy": outcomes["correct"] / len(rows),
            "precision": outcomes["correct"] / answered if answered else None,
            "expected_behaviour_rate": sum(r["is_expected"] for r in rows) / len(rows),
        }
        if mode != "closed_book":
            stats["gated"] = sum(r["gated"] for r in rows)
            stats["mean_distinct_papers"] = mean(r["distinct_papers"] for r in rows)
        if mode == "rag":
            stats["source_hit_rate"] = mean(r["source_hit"] for r in rows)
            stats["key_passage_hit_rate"] = mean(r["key_passage_hit"] for r in rows)
        summary[mode] = stats
    return summary
