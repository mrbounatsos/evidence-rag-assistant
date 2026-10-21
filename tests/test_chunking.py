"""Tests for chunking extracted blocks."""

from evidence_rag_assistant.indexing.chunking import chunk_article


def block(index: int, kind: str, path: tuple[str, ...], text: str) -> dict[str, object]:
    return {
        "pmcid": "PMC1",
        "block_index": index,
        "kind": kind,
        "section_path": list(path),
        "text": text,
    }


BLOCKS = [
    block(0, "title", (), "A paper"),
    block(1, "abstract", ("Abstract",), "one two three"),
    block(2, "paragraph", ("Results",), "four five"),
    block(3, "paragraph", ("Results",), "six seven"),
    block(4, "paragraph", ("Results",), "eight nine ten"),
    block(5, "caption", ("Results",), "Figure 1: a caption"),
    block(6, "paragraph", ("Discussion",), "w " * 7),
]


def test_chunks_respect_sections_size_and_captions() -> None:
    chunks = chunk_article(BLOCKS, max_words=5)
    assert [(c.section_path, c.block_indices) for c in chunks] == [
        (("Abstract",), (1,)),  # section change closes the abstract
        (("Results",), (2, 3)),  # 4 words; adding 3 more would exceed 5
        (("Results",), (4,)),
        (("Results",), (5,)),  # caption is its own chunk
        (("Discussion",), (6,)),  # 7-word paragraph split into 5 + 2
        (("Discussion",), (6,)),
    ]


def test_embedding_text_has_title_and_section_header() -> None:
    chunk = chunk_article(BLOCKS, max_words=5)[1]
    assert chunk.embedding_text() == "A paper\nResults\n\nfour five\n\nsix seven"
