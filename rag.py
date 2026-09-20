"""
Embedding + retrieval.

The VectorStore interface (add / search) is the seam you'd cut along to
swap in a real vector database (pgvector, Chroma, Pinecone) for a customer
deployment that needs persistence or scale beyond a few thousand chunks.
Everything above this layer (main.py) doesn't need to know the difference.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingest import Chunk

_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, model_name: str = _MODEL_NAME):
        self._model = SentenceTransformer(model_name)
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None  # shape: (n_chunks, dim)

    def __len__(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        new_vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        self._chunks.extend(chunks)
        if self._embeddings is None:
            self._embeddings = new_vectors
        else:
            self._embeddings = np.vstack([self._embeddings, new_vectors])

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Returns up to k (chunk, similarity_score) pairs, best first."""
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        query_vec = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        # Embeddings are normalized, so dot product == cosine similarity.
        scores = self._embeddings @ query_vec
        k = min(k, len(self._chunks))
        top_indices = np.argsort(-scores)[:k]

        return [(self._chunks[i], float(scores[i])) for i in top_indices]
