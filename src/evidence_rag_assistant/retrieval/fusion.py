"""Combine several rankings with Reciprocal Rank Fusion (RRF).

RRF uses only positions, not scores, so rankings on incompatible scales
(cosine similarity, BM25) can be combined without tuning. Each item receives
1 / (k + rank) from every ranking it appears in.
"""

from collections import defaultdict

RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Return (item, fused score) pairs, best first."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
