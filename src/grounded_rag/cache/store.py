"""Qdrant `query_cache` collection: schema + payload index.

DATA-MODEL.md#collection-query_cache: one point per distinct cached answer,
a single dense vector (the query embedding), payload-indexed
`acl_signature` so a lookup/write-through never crosses an ACL boundary
(FR9; ADR-005). Mirrors `ingestion/qdrant_store.py`'s `ensure_collection`
shape for the `articles` collection.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from grounded_rag.config import DENSE_VECTOR_NAME, EMBEDDING_DIM, QUERY_CACHE_COLLECTION


def ensure_collection(client: QdrantClient, collection_name: str = QUERY_CACHE_COLLECTION) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={DENSE_VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)},
        )

    existing_indexes = client.get_collection(collection_name).payload_schema
    if "acl_signature" not in existing_indexes:
        client.create_payload_index(collection_name, "acl_signature", field_schema=PayloadSchemaType.KEYWORD)
