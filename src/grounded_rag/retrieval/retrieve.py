"""`retrieve`: Qdrant hybrid (dense + sparse) query with ACL/metadata pre-filter.

ADR-004: single-engine Qdrant hybrid query, fusion pinned to RRF. ADR-008:
the ACL/metadata filter is attached to each `Prefetch` leg, so filtering
happens before fusion — a chunk outside the caller's groups never enters the
candidate set. A plain function for M1/M2, not a LangGraph node yet (see
ARCHITECTURE.md) — wrapped into one at M3 when the tool-call cycle needs it.

`use_sparse=False` runs the exact same code path with only the dense leg, so
the M1 eval harness measures hybrid-vs-dense-only through the same
production code, not a parallel reimplementation.
"""

from __future__ import annotations

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch
from qdrant_client.models import SparseVector as QdrantSparseVector

from grounded_rag.config import ARTICLES_COLLECTION, DENSE_VECTOR_NAME, RETRIEVE_CANDIDATE_K, SPARSE_VECTOR_NAME
from grounded_rag.ingestion.embeddings import SparseEmbedder, embed_dense
from grounded_rag.retrieval.filters import build_filter
from grounded_rag.retrieval.records import RetrievedChunk


def retrieve(
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
    sparse_embedder: SparseEmbedder,
    query: str,
    access_context_groups: list[str],
    doc_type: str | None = None,
    date_range: dict[str, str] | None = None,
    candidate_k: int = RETRIEVE_CANDIDATE_K,
    use_sparse: bool = True,
) -> list[RetrievedChunk]:
    if not access_context_groups:
        return []

    payload_filter = build_filter(access_context_groups, doc_type=doc_type, date_range=date_range)
    dense_vector = next(embed_dense(openai_client, [query]))

    if use_sparse:
        sparse_vector = next(sparse_embedder.query_embed([query]))
        prefetch = [
            Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, filter=payload_filter, limit=candidate_k),
            Prefetch(
                query=QdrantSparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
                using=SPARSE_VECTOR_NAME,
                filter=payload_filter,
                limit=candidate_k,
            ),
        ]
        result = qdrant_client.query_points(
            collection_name=ARTICLES_COLLECTION,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=candidate_k,
            with_payload=True,
        )
    else:
        # A single leg has nothing to fuse — Qdrant rejects Prefetch without
        # an outer query ("A query is needed to merge the prefetches"), so
        # query the dense vector directly instead.
        result = qdrant_client.query_points(
            collection_name=ARTICLES_COLLECTION,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=payload_filter,
            limit=candidate_k,
            with_payload=True,
        )

    return [
        RetrievedChunk(
            chunk_id=str(point.id),
            doc_id=point.payload["doc_id"],
            title=point.payload["title"],
            url=point.payload["url"],
            text=point.payload["text"],
            doc_type=point.payload["doc_type"],
            acl_tags=point.payload["acl_tags"],
            score=point.score,
        )
        for point in result.points
    ]
