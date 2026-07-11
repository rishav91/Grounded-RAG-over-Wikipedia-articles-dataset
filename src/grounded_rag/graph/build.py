"""`build_graph`: wires the M3 read path (ADR-001).

Matches ARCHITECTURE.md's query-flow diagram minus `cache_lookup` (M4):
retrieve -> rerank -> (allow_generation? generate : response);
generate -> (tool call? execute_tool : faithfulness); execute_tool -> generate;
faithfulness -> response -> END.
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
def _route_after_rerank(state: GraphState) -> str:
    return "generate" if state["allow_generation"] else "build_response"


def _route_after_generate(state: GraphState) -> str:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if tool_calls and tool_calls[0]["name"] == RETRIEVE_TOOL_NAME:
        return "execute_tool"
    return "judge_faithfulness"


def build_graph(deps: GraphDeps) -> CompiledStateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", partial(nodes.retrieve_node, deps))
    graph.add_node("rerank", partial(nodes.rerank_node, deps))
    graph.add_node("generate", partial(nodes.generate_node, deps))
    graph.add_node("execute_tool", partial(nodes.execute_tool_node, deps))
    graph.add_node("judge_faithfulness", partial(nodes.faithfulness_node, deps))
    graph.add_node("build_response", partial(nodes.response_node, deps))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank", _route_after_rerank, {"generate": "generate", "build_response": "build_response"}
    )
    graph.add_conditional_edges(
        "generate", _route_after_generate, {"execute_tool": "execute_tool", "judge_faithfulness": "judge_faithfulness"}
    )
    graph.add_edge("execute_tool", "generate")
    graph.add_edge("judge_faithfulness", "build_response")
    graph.add_edge("build_response", END)

    return graph.compile()
