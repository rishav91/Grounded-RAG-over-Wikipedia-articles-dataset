"""The retrieval tool (FR8) — API-CONTRACTS.md's `retrieve_chunks` schema.

FR-4.3: the tool's schema exposes only `query`/`top_k`. `access_context_groups`,
`doc_type`, and `date_range` are closed over from the request's `state`, never
accepted as tool arguments — the model can refine what it searches for but
can never widen what it's permitted to see, structurally, not by convention.
Same underlying `retrieve()`/`rerank()` calls the deterministic first pass
uses (ARCHITECTURE.md) — the tool schema exists so the *generator* decides
if/when to call it, not a different retrieval mechanism.
"""

from __future__ import annotations

from langchain_core.tools import tool

from grounded_rag.config import RERANK_TOP_K
from grounded_rag.generation.generate import RETRIEVE_TOOL_NAME
from grounded_rag.graph.deps import GraphDeps
from grounded_rag.graph.state import GraphState
from grounded_rag.rerank.rerank import rerank
from grounded_rag.retrieval.records import RetrievedChunk
from grounded_rag.retrieval.retrieve import retrieve


def build_retrieve_tool(deps: GraphDeps, state: GraphState):
    @tool(RETRIEVE_TOOL_NAME)
    def retrieve_chunks(query: str, top_k: int = RERANK_TOP_K) -> list[RetrievedChunk]:
        """Search the document corpus for chunks relevant to a query, under the caller's existing access context and any active filters. Use this if the chunks already provided are insufficient to answer the question."""
        candidates = retrieve(
            deps.qdrant_client,
            deps.openai_client,
            deps.sparse_embedder,
            query,
            state["access_context_groups"],
            doc_type=state["doc_type"],
            date_range=state["date_range"],
        )
        result = rerank(deps.cohere_client, query, candidates, top_k=top_k)
        return result.chunks

    return retrieve_chunks
