"""Qdrant `articles` collection: schema, payload indexes, upsert.

DATA-MODEL.md#collection-articles: one point per chunk, named dense + sparse
vectors, payload-indexed doc_type/acl_tags/created_at/updated_at so
ADR-008's pre-filter applies before fusion.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from grounded_rag.config import (
    ARTICLES_COLLECTION,
    DENSE_VECTOR_NAME,
    EMBEDDING_DIM,
    SPARSE_VECTOR_NAME,
    Settings,
    UPSERT_BATCH_SIZE,
)
from grounded_rag.ingestion.embeddings import SparseVector as LocalSparseVector
from grounded_rag.ingestion.records import ChunkRecord

_INDEXED_KEYWORD_FIELDS = ("doc_type", "acl_tags")
_INDEXED_DATETIME_FIELDS = ("created_at", "updated_at")


def get_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection(client: QdrantClient, collection_name: str = ARTICLES_COLLECTION) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={DENSE_VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        )

    existing_indexes = client.get_collection(collection_name).payload_schema
    for field_name in _INDEXED_KEYWORD_FIELDS:
        if field_name not in existing_indexes:
            client.create_payload_index(collection_name, field_name, field_schema=PayloadSchemaType.KEYWORD)
    for field_name in _INDEXED_DATETIME_FIELDS:
        if field_name not in existing_indexes:
            client.create_payload_index(collection_name, field_name, field_schema=PayloadSchemaType.DATETIME)


def build_point(chunk_id: str, record: ChunkRecord, dense: list[float], sparse: LocalSparseVector) -> PointStruct:
    return PointStruct(
        id=chunk_id,
        vector={
            DENSE_VECTOR_NAME: dense,
            SPARSE_VECTOR_NAME: SparseVector(indices=sparse.indices, values=sparse.values),
        },
        payload={
            "doc_id": record.doc_id,
            "chunk_index": record.chunk_index,
            "title": record.title,
            "url": record.url,
            "text": record.text,
            "doc_type": record.doc_type,
            "acl_tags": record.acl_tags,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        },
    )


def upsert_points(
    client: QdrantClient,
    points: list[PointStruct],
    collection_name: str = ARTICLES_COLLECTION,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[start : start + batch_size])
