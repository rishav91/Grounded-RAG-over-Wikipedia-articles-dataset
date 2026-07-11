"""`build_graph`: wires the M3+M4 read path (ADR-001).

Matches ARCHITECTURE.md's query-flow diagram:
cache_lookup -> (hit? response : retrieve); retrieve -> rerank ->
(allow_generation? check_sufficiency : response);
check_sufficiency -> (sufficient? generate : response) (FR15; ADR-010);
generate -> (tool call? execute_tool : faithfulness); execute_tool -> generate;
faithfulness -> response -> END. A passing response writes through to the
cache (FR9; ADR-005) before returning.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from grounded_rag.generation.generate import RETRIEVE_TOOL_NAME
from grounded_rag.graph import nodes
from grounded_rag.graph.deps import GraphDeps
from grounded_rag.graph.state import GraphState


# Node ids deliberately differ from GraphState's field names (e.g.
# "judge_faithfulness" not "faithfulness") — langgraph rejects a node id that
# collides with a state key.
def _route_after_cache_lookup(state: GraphState) -> str:
    return "build_response" if state["cache_result"].hit else "retrieve"


def _route_after_rerank(state: GraphState) -> str:
    return "check_sufficiency" if state["allow_generation"] else "build_response"


def _route_after_sufficiency(state: GraphState) -> str:
    return "generate" if state["sufficiency"].sufficient else "build_response"


def _route_after_generate(state: GraphState) -> str:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if tool_calls and tool_calls[0]["name"] == RETRIEVE_TOOL_NAME:
        return "execute_tool"
    return "judge_faithfulness"


def build_graph(deps: GraphDeps) -> CompiledStateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("cache_lookup", partial(nodes.cache_lookup_node, deps))
    graph.add_node("retrieve", partial(nodes.retrieve_node, deps))
    graph.add_node("rerank", partial(nodes.rerank_node, deps))
    graph.add_node("check_sufficiency", partial(nodes.check_sufficiency_node, deps))
    graph.add_node("generate", partial(nodes.generate_node, deps))
    graph.add_node("execute_tool", partial(nodes.execute_tool_node, deps))
    graph.add_node("judge_faithfulness", partial(nodes.faithfulness_node, deps))
    graph.add_node("build_response", partial(nodes.response_node, deps))

    graph.add_edge(START, "cache_lookup")
    graph.add_conditional_edges(
        "cache_lookup", _route_after_cache_lookup, {"retrieve": "retrieve", "build_response": "build_response"}
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank", _route_after_rerank, {"check_sufficiency": "check_sufficiency", "build_response": "build_response"}
    )
    graph.add_conditional_edges(
        "check_sufficiency", _route_after_sufficiency, {"generate": "generate", "build_response": "build_response"}
    )
    graph.add_conditional_edges(
        "generate", _route_after_generate, {"execute_tool": "execute_tool", "judge_faithfulness": "judge_faithfulness"}
    )
    graph.add_edge("execute_tool", "generate")
    graph.add_edge("judge_faithfulness", "build_response")
    graph.add_edge("build_response", END)

    return graph.compile()
