"""`run_query`: the single entry point into the compiled M3 graph.

The seam `scripts/eval_m3.py` calls today, and a future `POST /query` HTTP
layer would call without changing shape (ARCHITECTURE.md's API layer
component just runs the compiled graph and returns this response dict).
"""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from grounded_rag.config import RERANK_TOP_K


def run_query(
    graph: CompiledStateGraph,
    query: str,
    access_context_groups: list[str],
    doc_type: str | None = None,
    date_range: dict[str, str] | None = None,
    top_k: int = RERANK_TOP_K,
    allow_generation: bool = True,
) -> dict:
    initial_state = {
        "query": query,
        "access_context_groups": access_context_groups,
        "doc_type": doc_type,
        "date_range": date_range,
        "top_k": top_k,
        "allow_generation": allow_generation,
        "chunks": [],
        "reranked": False,
        "sufficiency": None,
        "messages": [],
        "tool_call_count": 0,
        "draft_answer": None,
        "citations": [],
        "faithfulness": None,
        "response": {},
    }
    final_state = graph.invoke(initial_state)
    return final_state["response"]
