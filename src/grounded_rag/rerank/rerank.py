"""`rerank`: Cohere cross-encoder rerank over the fused candidate set.

ADR-003: hosted Cohere Rerank API (`rerank-v3.5`), not a self-hosted
cross-encoder. FR-3.2: a Cohere failure must never fail the request — it
degrades to fusion-only ranking (the order `retrieve` already produced), so
the broad `except` below is deliberate, not sloppy error handling. A plain
function for M2, not a LangGraph node yet — see `retrieve.py`'s docstring
for why.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cohere

from grounded_rag.config import RERANK_MODEL, RERANK_TOP_K
from grounded_rag.retrieval.records import RetrievedChunk


@dataclass(frozen=True)
class RerankResult:
    chunks: list[RetrievedChunk]
    reranked: bool  # False: the Cohere call failed; chunks is the fusion-only fallback (FR-3.2).


def rerank(
    cohere_client: cohere.ClientV2,
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = RERANK_TOP_K,
) -> RerankResult:
    if not chunks:
        return RerankResult(chunks=[], reranked=False)

    try:
        response = cohere_client.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=[chunk.text for chunk in chunks],
            top_n=min(top_k, len(chunks)),
        )
    except Exception:
        return RerankResult(chunks=chunks[:top_k], reranked=False)

    reranked_chunks = [replace(chunks[result.index], score=result.relevance_score) for result in response.results]
    return RerankResult(chunks=reranked_chunks, reranked=True)
