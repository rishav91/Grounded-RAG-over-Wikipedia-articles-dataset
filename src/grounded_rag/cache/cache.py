"""`lookup_cache` / `write_cache`: the ACL-aware semantic cache (FR9; ADR-005).

Plain functions, not LangGraph nodes — mirrors `retrieve()`/`rerank()`.
Caching is a latency/cost optimization, never a correctness dependency
(ARCHITECTURE.md's failure-modes table): both functions fail open on any
Qdrant/OpenAI error, degrading to "treat as a cache miss" rather than
failing the request.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from grounded_rag.cache.records import CacheLookupResult
from grounded_rag.config import CACHE_SIMILARITY_THRESHOLD, DENSE_VECTOR_NAME, QUERY_CACHE_COLLECTION
from grounded_rag.ingestion.embeddings import embed_dense


def acl_signature(access_context_groups: list[str]) -> str:
    # Sorted before hashing so group order in the request never changes the
    # signature (DATA-MODEL.md).
    joined = ",".join(sorted(access_context_groups))
    return hashlib.sha256(joined.encode()).hexdigest()


def lookup_cache(
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
    query: str,
    access_context_groups: list[str],
) -> CacheLookupResult:
    if not access_context_groups:
        return CacheLookupResult(hit=False)

    try:
        signature = acl_signature(access_context_groups)
        vector = next(embed_dense(openai_client, [query]))
        result = qdrant_client.query_points(
            collection_name=QUERY_CACHE_COLLECTION,
            query=vector,
            using=DENSE_VECTOR_NAME,
            query_filter=Filter(must=[FieldCondition(key="acl_signature", match=MatchValue(value=signature))]),
            limit=1,
            with_payload=True,
        )
    except Exception:
        return CacheLookupResult(hit=False)

    if not result.points or result.points[0].score < CACHE_SIMILARITY_THRESHOLD:
        return CacheLookupResult(hit=False)

    point = result.points[0]
    return CacheLookupResult(
        hit=True,
        answer=point.payload["answer"],
        citations=point.payload["citations"],
        confidence=point.payload["confidence"],
    )


def write_cache(
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
    query: str,
    access_context_groups: list[str],
    answer: str,
    citations: list[dict],
    confidence: float,
) -> None:
    try:
        signature = acl_signature(access_context_groups)
        vector = next(embed_dense(openai_client, [query]))
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={DENSE_VECTOR_NAME: vector},
            payload={
                "acl_signature": signature,
                "query_text": query,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        qdrant_client.upsert(collection_name=QUERY_CACHE_COLLECTION, points=[point])
    except Exception:
        # A write-through failure is a lost cache opportunity, not a request
        # failure — the response was already built successfully.
        pass
