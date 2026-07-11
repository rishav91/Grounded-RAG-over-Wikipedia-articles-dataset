"""`GraphState`: the state object threaded through every M3 node (ADR-001).

A consistent state object is also what FR13's future per-request trace would
hang off directly (see ARCHITECTURE.md's `+` consequence for `ADR-001`).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from grounded_rag.cache.records import CacheLookupResult
from grounded_rag.faithfulness.records import FaithfulnessResult
from grounded_rag.generation.records import Citation
from grounded_rag.retrieval.records import RetrievedChunk
from grounded_rag.rewrite.records import RewriteResult
from grounded_rag.sufficiency.records import SufficiencyResult


class GraphState(TypedDict):
    # Request inputs (API-CONTRACTS.md POST /query), fixed for the request.
    query: str
    access_context_groups: list[str]
    doc_type: str | None
    date_range: dict[str, str] | None
    top_k: int
    allow_generation: bool
    bypass_cache: bool

    # cache_lookup's verdict (FR9; ADR-005). Node id is "cache_lookup", not
    # "cache_result" — avoids the node-id/state-key collision langgraph
    # rejects at add_node time (see M3's judge_faithfulness/faithfulness).
    cache_result: CacheLookupResult | None

    # rewrite_query's output (FR11; ADR-011) — None if cache_lookup hit and
    # this node never ran. Node id "rewrite_query" matches the state key
    # here (no collision — "rewrite" isn't a node id).
    rewrite: RewriteResult | None

    # Retrieval/rerank output — grows if the retrieval tool fires (FR8).
    chunks: list[RetrievedChunk]
    reranked: bool  # False if the Cohere call failed and rerank fell back to fusion order (FR-3.2)

    # check_sufficiency's verdict on the first-pass chunks (FR15; ADR-010).
    sufficiency: SufficiencyResult | None

    # The generate node's tool-call conversation (FR8's ReAct-style loop).
    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_count: int

    # generate's terminal output (FR5).
    draft_answer: str | None
    citations: list[Citation]

    # faithfulness's verdict (FR6, FR7, FR-5.4). Stays None if check_sufficiency
    # short-circuited the request before generate/faithfulness ever ran.
    faithfulness: FaithfulnessResult | None

    # response's final API-CONTRACTS.md-shaped dict.
    response: dict
