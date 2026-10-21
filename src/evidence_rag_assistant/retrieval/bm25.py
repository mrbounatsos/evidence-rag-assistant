"""Keyword retrieval with BM25.

BM25 scores a text by the query words it contains, weighting rare words much
more than common ones and correcting for text length. It complements embedding
search, which captures meaning but blurs rare exact terms such as mutation
codes or species names.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")

# Common words that carry no topical information.
STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from had has have how in into "
    "is it its of on or that the their these this those to was were what when where "
    "which while who why with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase letter-and-digit tokens without stopwords: 'A456V' -> 'a456v'."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


class BM25Index:
    def __init__(self, texts: list[str]) -> None:
        self._bm25 = BM25Okapi([tokenize(text) for text in texts])

    def ranking(self, query: str) -> list[int]:
        """Row numbers of texts sharing at least one term with the query, best first."""
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        order = np.argsort(-scores, kind="stable")
        return [int(row) for row in order if scores[row] > 0]
