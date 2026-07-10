"""Central configuration: env vars and the pinned constants from the design docs.

Values here are pinned by the design-doc suite (DATA-MODEL.md, ADRs.md), not
arbitrary defaults — see the referenced doc/ADR on each constant before
changing it.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# DATA-MODEL.md: chunk_id = uuid5(NAMESPACE, f"{doc_id}:{chunk_index}").
# Generated once and frozen — changing this value changes every chunk_id and
# is equivalent to a full re-ingestion (loses upsert idempotency for any
# existing collection).
CHUNK_ID_NAMESPACE = uuid.UUID("c1b1d8b4-3b1a-4b6a-9b1a-5b1c0a8e2f3d")

# ADR-002: pinned embedding model and dimension.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# DATA-MODEL.md: BM25-style sparse vectors via fastembed's Qdrant/bm25 model.
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"

# DATA-MODEL.md §Source -> canonical mapping: chunking defaults.
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# Batch sizes for the embedding/upsert calls during ingestion. Not pinned by
# any doc — tuned for throughput against API/Qdrant limits, not correctness.
EMBEDDING_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 100

# PRD.md §3.1 / §3.2: dataset and the deterministic MVP slice size.
HF_DATASET_NAME = "wikimedia/wikipedia"
HF_DATASET_CONFIG = "20231101.en"
MVP_SLICE_SIZE = 1000

# DATA-MODEL.md: Qdrant collection + named-vector identifiers.
ARTICLES_COLLECTION = "articles"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# DATA-MODEL.md §Source -> canonical mapping: doc_type length bands (chars).
DOC_TYPE_SHORT_MAX_CHARS = 2000
DOC_TYPE_MEDIUM_MAX_CHARS = 8000

# DATA-MODEL.md §ACL tag derivation: closed set of four synthetic groups.
ACL_GROUPS = ["eng", "finance", "legal"]

# ADR-003: pinned candidate-set size for the fusion-stage retrieve query,
# within ADR-003's qualitative "~20-50 chunks per query" range. Feeds both
# M1's recall@k measurement and M2's rerank input size.
RETRIEVE_CANDIDATE_K = 30


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY") or None)
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.environ.get("QDRANT_API_KEY") or None)
    cohere_api_key: str | None = field(default_factory=lambda: os.environ.get("COHERE_API_KEY") or None)


def get_settings() -> Settings:
    return Settings()
