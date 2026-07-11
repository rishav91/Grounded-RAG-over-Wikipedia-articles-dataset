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

# ADR-003: pinned Cohere rerank model.
RERANK_MODEL = "rerank-v3.5"

# API-CONTRACTS.md `options.top_k` default: final chunk count passed to
# generation after rerank.
RERANK_TOP_K = 5

# FR-5.1: confidence threshold gating the abstain decision — an arbitrary
# starting cut, flagged as a placeholder in REQUIREMENTS.md's Open
# assumptions pending the real score distribution observed in M3.
FAITHFULNESS_CONFIDENCE_THRESHOLD = 0.7

# FR-4.2 / UC-4: the retrieval tool may fire at most this many additional
# rounds per request — "exactly one additional tool call fires," not an
# open-ended agentic loop.
TOOL_CALL_MAX_ROUNDS = 1

# DATA-MODEL.md: second Qdrant collection for the ACL-aware semantic cache
# (FR9; ADR-005). Reuses DENSE_VECTOR_NAME/EMBEDDING_DIM — the cache embeds
# query text with the same model as article chunks, no separate config.
QUERY_CACHE_COLLECTION = "query_cache"

# DATA-MODEL.md: a cache lookup is a hit only if the top result's cosine
# score clears this bar. Placeholder pending real paraphrase-pattern
# measurement once M4 is live — see REQUIREMENTS.md Open assumptions.
CACHE_SIMILARITY_THRESHOLD = 0.92

# FR15 / ADR-010: check_sufficiency's tier-1 score gate, on Cohere's 0-1
# relevance_score scale (only trusted when reranking actually ran — see
# ADR-010). Below LOW: obviously hopeless, skip generation entirely. At or
# above HIGH: obviously fine, skip the tier-2 LLM judge. Between the two:
# genuinely ambiguous, spend the LLM call. Both are placeholders pending real
# measurement — see REQUIREMENTS.md Open assumptions.
SUFFICIENCY_LOW_SCORE_THRESHOLD = 0.2
SUFFICIENCY_HIGH_SCORE_THRESHOLD = 0.6

# FR11 / ADR-011: caps how many independent sub_queries rewrite_query may
# produce for one request — bounded, not an open-ended decomposition.
QUERY_REWRITE_MAX_SUB_QUERIES = 3

# FR12 / ADR-012: caps how many retrieve_chunks calls execute_tool_node will
# run concurrently in a single round — bounded, mirrors TOOL_CALL_MAX_ROUNDS'
# "not an open-ended agentic loop" spirit, applied to within-round fan-out.
MAX_PARALLEL_RETRIEVE_CALLS = 3


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY") or None)
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.environ.get("QDRANT_API_KEY") or None)
    cohere_api_key: str | None = field(default_factory=lambda: os.environ.get("COHERE_API_KEY") or None)
    # ADR-007: provider-agnostic via init_chat_model, no default pinned here —
    # the operator sets the provider/model string via env config (see
    # .env.example). faithfulness_model falls back to generation_model so the
    # two are independently swappable only when the operator opts in.
    generation_model: str | None = field(default_factory=lambda: os.environ.get("GENERATION_MODEL") or None)
    faithfulness_model: str | None = field(
        default_factory=lambda: os.environ.get("FAITHFULNESS_MODEL") or os.environ.get("GENERATION_MODEL") or None
    )


def get_settings() -> Settings:
    return Settings()
