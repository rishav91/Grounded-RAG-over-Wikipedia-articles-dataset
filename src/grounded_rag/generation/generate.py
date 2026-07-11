"""`generate`: one citation-constrained LLM round, with the retrieval tool
optionally bound (FR5, FR8, FR12).

ADR-007: the LLM itself is provider-agnostic (`init_chat_model`), passed in
already configured — this module never imports a vendor SDK directly.

"Answer as a tool call" (`SubmitAnswer`) is the mechanism for
citation-constrained structured output: binding `SubmitAnswer` alongside
`retrieve_chunks` and forcing `tool_choice="required"` means every round
ends in exactly one recognized decision — cite and answer, or re-retrieve —
never free-form prose that would need a fragile parsing step to extract
citations from.

ADR-012: `parallel_tool_calls=True` lets a round request more than one
`retrieve_chunks` call at once (executed concurrently by `graph/nodes.py`'s
`execute_tool_node`). If the round mixes `SubmitAnswer` with other calls,
`SubmitAnswer` wins and finishes the round immediately — the other calls are
simply never invoked, which is safe because a finished round never sends
another message to this LLM (no dangling tool_call_id to answer).

A plain function for one LLM round, not a LangGraph node — the tool-call
loop (how many rounds, when to stop offering `retrieve_chunks`) is graph
wiring that lives in `graph/nodes.py`, mirroring how `retrieve.py`/`rerank.py`
stay plain functions the graph wraps.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from grounded_rag.config import RERANK_TOP_K
from grounded_rag.generation.records import Citation, GenerationResult, ToolCallRequest

RETRIEVE_TOOL_NAME = "retrieve_chunks"
# Pydantic-model-as-tool: langchain names the tool after the class itself.
SUBMIT_ANSWER_TOOL_NAME = "SubmitAnswer"


class SubmitAnswerCitation(BaseModel):
    chunk_id: str
    claim: str


class SubmitAnswer(BaseModel):
    """Call this once you can answer the question using only the given chunks, citing every claim."""

    answer: str
    citations: list[SubmitAnswerCitation]


def generate(llm: BaseChatModel, messages: list[BaseMessage], tools: list) -> tuple[BaseMessage, GenerationResult]:
    # ADR-012: parallel_tool_calls=True allows more than one retrieve_chunks
    # call in a single round; execute_tool_node answers every tool_call_id
    # it's handed, so no dangling call is left unanswered on the next turn.
    bound = llm.bind_tools(tools, tool_choice="required", parallel_tool_calls=True)
    response = bound.invoke(messages)

    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return response, GenerationResult(answer=None, citations=[], tool_calls=[], finished=False)

    submit_call = next((c for c in tool_calls if c["name"] == SUBMIT_ANSWER_TOOL_NAME), None)
    if submit_call is not None:
        args = submit_call["args"]
        citations = [Citation(chunk_id=c["chunk_id"], claim=c["claim"]) for c in args.get("citations", [])]
        return response, GenerationResult(answer=args["answer"], citations=citations, tool_calls=[], finished=True)

    retrieve_calls = [c for c in tool_calls if c["name"] == RETRIEVE_TOOL_NAME]
    if retrieve_calls:
        requests = [
            ToolCallRequest(call_id=c["id"], query=c["args"]["query"], top_k=c["args"].get("top_k", RERANK_TOP_K))
            for c in retrieve_calls
        ]
        return response, GenerationResult(answer=None, citations=[], tool_calls=requests, finished=False)

    return response, GenerationResult(answer=None, citations=[], tool_calls=[], finished=False)
