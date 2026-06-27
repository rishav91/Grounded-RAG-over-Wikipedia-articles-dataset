"""End-to-end M0 ingestion: load -> chunk -> derive metadata -> embed -> upsert.

ARCHITECTURE.md#ingestion-flow. Chunk IDs are deterministic
(ingestion/ids.py), so re-running this pipeline on the same slice upserts
rather than duplicates (FR-1.2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI
from tqdm import tqdm

from grounded_rag.config import MVP_SLICE_SIZE, get_settings
from grounded_rag.ingestion.chunker import chunk_text
from grounded_rag.ingestion.embeddings import SparseEmbedder, embed_dense
from grounded_rag.ingestion.ids import chunk_id
from grounded_rag.ingestion.loader import load_articles
from grounded_rag.ingestion.metadata import derive_metadata
from grounded_rag.ingestion.qdrant_store import build_point, ensure_collection, get_client, upsert_points
from grounded_rag.ingestion.records import ChunkRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionStats:
    num_docs: int
    num_chunks: int


def _build_chunk_records(limit: int) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for article in tqdm(load_articles(limit), total=limit, desc="chunking"):
        meta = derive_metadata(article.doc_id, article.text)
        for index, text in enumerate(chunk_text(article.text)):
            records.append(
                ChunkRecord(
                    doc_id=article.doc_id,
                    chunk_index=index,
                    text=text,
                    title=article.title,
                    url=article.url,
                    doc_type=meta.doc_type,
                    acl_tags=meta.acl_tags,
                    created_at=meta.created_at,
                    updated_at=meta.updated_at,
                )
            )
    return records


def run_ingestion(limit: int = MVP_SLICE_SIZE) -> IngestionStats:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set — required for dense embedding")

    records = _build_chunk_records(limit)
    logger.info("chunked %d documents into %d chunks", limit, len(records))

    texts = [r.text for r in records]

    openai_client = OpenAI(api_key=settings.openai_api_key)
    logger.info("embedding %d chunks (dense, OpenAI)", len(texts))
    dense_vectors = list(tqdm(embed_dense(openai_client, texts), total=len(texts), desc="dense embed"))

    logger.info("embedding %d chunks (sparse, fastembed BM25)", len(texts))
    sparse_embedder = SparseEmbedder()
    sparse_vectors = list(tqdm(sparse_embedder.embed(texts), total=len(texts), desc="sparse embed"))

    points = [
        build_point(chunk_id(r.doc_id, r.chunk_index), r, dense, sparse)
        for r, dense, sparse in zip(records, dense_vectors, sparse_vectors)
    ]

    qdrant_client = get_client(settings)
    ensure_collection(qdrant_client)
    logger.info("upserting %d points into Qdrant", len(points))
    upsert_points(qdrant_client, points)

    return IngestionStats(num_docs=limit, num_chunks=len(records))
