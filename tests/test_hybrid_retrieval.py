"""Tests for BM25 tokenization and ranking, and Reciprocal Rank Fusion."""

from evidence_rag_assistant.retrieval.bm25 import BM25Index, tokenize
from evidence_rag_assistant.retrieval.fusion import reciprocal_rank_fusion


def test_tokenize_keeps_codes_and_drops_stopwords() -> None:
    assert tokenize("What does the A456V mutation do?") == ["a456v", "mutation"]


def test_bm25_ranks_text_with_rare_term_first() -> None:
    index = BM25Index(
        [
            "The A456V mutation stabilizes the closed state.",
            "Protein folding in general.",
            "Mutation effects on protein stability.",
        ]
    )
    assert index.ranking("What does A456V do?")[0] == 0


def test_bm25_ranking_excludes_texts_without_shared_terms() -> None:
    index = BM25Index(["antibody binding", "protein folding", "cell growth"])
    assert index.ranking("antibody") == [0]


def test_rrf_rewards_items_ranked_high_in_both() -> None:
    fused = reciprocal_rank_fusion([[0, 1], [1, 2]])
    assert [item for item, _ in fused] == [1, 0, 2]
