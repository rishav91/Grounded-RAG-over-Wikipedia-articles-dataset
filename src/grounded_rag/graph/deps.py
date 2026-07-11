"""`GraphDeps`: the clients/LLMs every node needs, bundled once per graph.

Mirrors the explicit-params style `retrieve()`/`rerank()` already use — nodes
stay plain functions over `(deps, state)`, not a hidden DI container.
"""

from __future__ import annotations

from dataclasses import dataclass

import cohere
from langchain_core.language_models.chat_models import BaseChatModel
from openai import OpenAI
from qdrant_client import QdrantClient

from grounded_rag.ingestion.embeddings import SparseEmbedder


@dataclass(frozen=True)
class GraphDeps:
    qdrant_client: QdrantClient
    openai_client: OpenAI
    cohere_client: cohere.ClientV2
    sparse_embedder: SparseEmbedder
    generation_llm: BaseChatModel
    faithfulness_llm: BaseChatModel
