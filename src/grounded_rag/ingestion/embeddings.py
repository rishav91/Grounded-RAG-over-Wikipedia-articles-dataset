"""Dense + sparse embedding for chunks.

ADR-002: OpenAI text-embedding-3-small for the dense leg.
DATA-MODEL.md: fastembed's Qdrant/bm25 sparse model for the sparse leg —
both run over the same chunk text.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from fastembed import SparseTextEmbedding
from openai import OpenAI

from grounded_rag.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


def embed_dense(client: OpenAI, texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> Iterator[list[float]]:
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in response.data:
            yield item.embedding


class SparseEmbedder:
    """Wraps fastembed's sparse model so the (CPU-bound) model load happens once."""

    def __init__(self) -> None:
        self._model = SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL)

    def embed(self, texts: list[str]) -> Iterator[SparseVector]:
        for embedding in self._model.embed(texts):
            yield SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
