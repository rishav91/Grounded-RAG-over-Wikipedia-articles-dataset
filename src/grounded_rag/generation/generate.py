"""`generate`: one citation-constrained LLM round, with the retrieval tool
optionally bound (FR5, FR8).

ADR-007: the LLM itself is provider-agnostic (`init_chat_model`), passed in
already configured — this module never imports a vendor SDK directly.

"Answer as a tool call" (`SubmitAnswer`) is the mechanism for
citation-constrained structured output: binding `SubmitAnswer` alongside
`retrieve_chunks` and forcing `tool_choice="required"` means every round
ends in exactly one recognized decision — cite and answer, or re-retrieve —
never free-form prose that would need a fragile parsing step to extract
citations from.

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
from grounded_rag.generation.records import Citation, GenerationResult

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
    # parallel_tool_calls=False: the graph handles exactly one tool_call per
    # round (execute_tool_node only answers tool_calls[0]) — a provider that
    # returns two calls in one AIMessage would leave the second tool_call_id
    # without a response message, which OpenAI's API rejects on the next turn.
    bound = llm.bind_tools(tools, tool_choice="required", parallel_tool_calls=False)
    response = bound.invoke(messages)

    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return response, GenerationResult(answer=None, citations=[], tool_query=None, tool_top_k=None, finished=False)

    call = tool_calls[0]
    if call["name"] == SUBMIT_ANSWER_TOOL_NAME:
        args = call["args"]
        citations = [Citation(chunk_id=c["chunk_id"], claim=c["claim"]) for c in args.get("citations", [])]
        return response, GenerationResult(
            answer=args["answer"], citations=citations, tool_query=None, tool_top_k=None, finished=True
        )
    if call["name"] == RETRIEVE_TOOL_NAME:
        args = call["args"]
        return response, GenerationResult(
            answer=None,
            citations=[],
            tool_query=args["query"],
            tool_top_k=args.get("top_k", RERANK_TOP_K),
            finished=False,
        )
    return response, GenerationResult(answer=None, citations=[], tool_query=None, tool_top_k=None, finished=False)
