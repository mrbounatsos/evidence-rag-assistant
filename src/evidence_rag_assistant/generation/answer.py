"""Grounded answer generation.

The model answers only from numbered sources, cites them as [n], and replies with
a fixed sentence when the sources are insufficient. Questions whose best retrieved
chunk scores below a threshold are abstained on without calling the model.
Citations in the answer are validated against the sources provided.
"""

import re
from dataclasses import dataclass, field

from evidence_rag_assistant.generation.llm import LLM
from evidence_rag_assistant.retrieval.retriever import RetrievedChunk, Retriever

ABSTAIN_MESSAGE = "I cannot answer this from the provided sources."

SYSTEM_PROMPT = "\n".join(
    [
        "You are a biomedical research assistant. "
        "Answer the question using ONLY the numbered sources provided.",
        "",
        "Rules:",
        "- End every factual sentence with citations to the sources that support it, "
        "in square brackets, e.g. [1] or [2][3].",
        "- Use only information stated in the sources. "
        "Do not use outside knowledge, even if you know the answer.",
        "- If the sources do not contain enough information to answer, "
        f"reply with exactly: {ABSTAIN_MESSAGE}",
        "- If the sources answer only part of the question, "
        "answer that part and say what is not covered.",
        "- Be concise: at most 5 sentences.",
    ]
)

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    status: str  # "answered" | "ungrounded" | "abstained_by_model" | "abstained_low_retrieval"
    top_score: float
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)


def build_prompt(question: str, sources: list[RetrievedChunk]) -> str:
    """Return the user message: numbered sources followed by the question."""
    parts = ["Sources:\n"]
    for number, item in enumerate(sources, start=1):
        chunk = item.chunk
        section = " > ".join(chunk["section_path"]) or "Main text"
        parts.append(f"[{number}] {chunk['title']} — {section}\n{chunk['text']}\n")
    parts.append(f"Question: {question}")
    return "\n".join(parts)


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) cited source numbers, each sorted and unique."""
    numbers = {int(n) for group in _CITATION.findall(text) for n in group.split(",")}
    valid = sorted(n for n in numbers if 1 <= n <= n_sources)
    invalid = sorted(n for n in numbers if not 1 <= n <= n_sources)
    return valid, invalid


def is_abstention(text: str) -> bool:
    return ABSTAIN_MESSAGE.lower() in text.lower()


def answer_question(
    question: str, retriever: Retriever, llm: LLM, k: int, min_score: float
) -> Answer:
    """Retrieve, gate on the best retrieval score, generate, and validate citations."""
    sources = retriever.retrieve(question, k)
    top_score = max((s.score for s in sources), default=0.0)
    if top_score < min_score:
        return Answer(question, ABSTAIN_MESSAGE, "abstained_low_retrieval", top_score, sources)

    text = llm.generate(SYSTEM_PROMPT, build_prompt(question, sources))
    if is_abstention(text):
        return Answer(question, text, "abstained_by_model", top_score, sources)

    cited, invalid = parse_citations(text, len(sources))
    status = "answered" if cited else "ungrounded"
    return Answer(question, text, status, top_score, sources, cited, invalid)
