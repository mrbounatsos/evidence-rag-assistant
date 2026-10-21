"""Split extracted blocks into chunks sized for retrieval.

Consecutive paragraphs from the same section are merged up to a word limit.
Captions are self-contained and become their own chunks. Each chunk keeps
the indices of the blocks it came from, so provenance is preserved.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_WORDS = 300


@dataclass(frozen=True)
class Chunk:
    pmcid: str
    chunk_index: int
    title: str
    section_path: tuple[str, ...]
    text: str
    block_indices: tuple[int, ...]

    @property
    def chunk_id(self) -> str:
        return f"{self.pmcid}:{self.chunk_index:04d}"

    def embedding_text(self) -> str:
        """Text to embed: the paper title and section path, then the chunk."""
        section = " > ".join(self.section_path) or "Main text"
        return f"{self.title}\n{section}\n\n{self.text}"


@dataclass(frozen=True)
class _Unit:
    text: str
    n_words: int
    path: tuple[str, ...]
    block_index: int
    standalone: bool


def _units(blocks: list[dict[str, Any]], max_words: int) -> Iterator[_Unit]:
    """Turn blocks into units no longer than max_words (long blocks are split)."""
    for block in blocks:
        if block["kind"] == "title":
            continue
        words = block["text"].split()
        for start in range(0, len(words), max_words):
            piece = words[start : start + max_words]
            yield _Unit(
                text=" ".join(piece),
                n_words=len(piece),
                path=tuple(block["section_path"]),
                block_index=block["block_index"],
                standalone=block["kind"] == "caption",
            )


def chunk_article(blocks: list[dict[str, Any]], max_words: int = DEFAULT_MAX_WORDS) -> list[Chunk]:
    """Chunk one article's blocks, given in article order."""
    if not blocks:
        return []
    pmcid = blocks[0]["pmcid"]
    title = next((b["text"] for b in blocks if b["kind"] == "title"), "")
    chunks: list[Chunk] = []
    group: list[_Unit] = []

    def close_group() -> None:
        if group:
            chunks.append(
                Chunk(
                    pmcid=pmcid,
                    chunk_index=len(chunks),
                    title=title,
                    section_path=group[0].path,
                    text="\n\n".join(unit.text for unit in group),
                    block_indices=tuple(unit.block_index for unit in group),
                )
            )
            group.clear()

    for unit in _units(blocks, max_words):
        if group and (
            unit.standalone
            or group[0].standalone
            or unit.path != group[0].path
            or sum(u.n_words for u in group) + unit.n_words > max_words
        ):
            close_group()
        group.append(unit)
    close_group()
    return chunks
