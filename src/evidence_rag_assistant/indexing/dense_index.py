"""Dense retrieval: embed chunks with a sentence-transformer, search with FAISS.

Embeddings are L2-normalized, so the inner product FAISS computes equals
cosine similarity. BGE models expect a short instruction before queries
(not passages), which places questions near the passages that answer them.
"""

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_model(name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed_passages(
    model: SentenceTransformer, texts: list[str], batch_size: int = 64
) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    vector = model.encode(
        [QUERY_INSTRUCTION + query], normalize_embeddings=True, convert_to_numpy=True
    )
    return np.asarray(vector, dtype=np.float32)


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search(index: faiss.Index, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return (row, score) pairs for the k most similar vectors, best first."""
    scores, rows = index.search(query_vector, k)
    return [
        (int(row), float(score)) for row, score in zip(rows[0], scores[0], strict=True) if row != -1
    ]


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))
