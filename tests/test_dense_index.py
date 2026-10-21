"""Tests for the FAISS index wrapper."""

import numpy as np
import pytest

from evidence_rag_assistant.indexing.dense_index import build_index, search


def test_search_ranks_by_cosine_similarity() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    index = build_index(vectors)
    results = search(index, np.array([[0.0, 1.0]], dtype=np.float32), k=2)
    assert [row for row, _ in results] == [1, 2]
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(0.8)
