"""LangGraph node wrappers around the plain retrieve/rerank/check_sufficiency/
generate/faithfulness functions (ADR-001). Each node is `(deps, state) -> dict`,
bound to a `GraphDeps` via `functools.partial` in `build.py`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from grounded_rag.cache.cache import lookup_cache, write_cache
from grounded_rag.cache.records import CacheLookupResult
from grounded_rag.config import MAX_PARALLEL_RETRIEVE_CALLS, TOOL_CALL_MAX_ROUNDS
from grounded_rag.faithfulness.faithfulness import judge_faithfulness
from grounded_rag.generation.generate import SubmitAnswer, generate
from grounded_rag.generation.prompts import SYSTEM_PROMPT, build_question_prompt, format_context
from grounded_rag.graph.deps import GraphDeps
from grounded_rag.graph.state import GraphState
from grounded_rag.graph.tool import build_retrieve_tool
from grounded_rag.rerank.rerank import rerank
from grounded_rag.retrieval.records import RetrievedChunk
from grounded_rag.retrieval.retrieve import retrieve
from grounded_rag.rewrite.rewrite import rewrite_query
from grounded_rag.sufficiency.sufficiency import check_sufficiency


def _merge_dedup_chunks(chunk_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Merges several chunk lists (e.g. concurrent retrieve() calls) into
    one, deduped by chunk_id — the highest score for a given chunk wins on a
    collision, first-seen order otherwise (ADR-011, ADR-012)."""
    best: dict[str, RetrievedChunk] = {}
    order: list[str] = []
    for chunks in chunk_lists:
        for chunk in chunks:
            existing = best.get(chunk.chunk_id)
            if existing is None:
                order.append(chunk.chunk_id)
                best[chunk.chunk_id] = chunk
            elif chunk.score > existing.score:
                best[chunk.chunk_id] = chunk
    return [best[chunk_id] for chunk_id in order]


def cache_lookup_node(deps: GraphDeps, state: GraphState) -> dict:
    # Retrieval-only requests (allow_generation=False) have no cached answer
    # to serve — the cache only ever stores finished, faithfulness-passed
    # answers (FR-5.3), so there's nothing for that mode to hit.
    if state["bypass_cache"] or not state["allow_generation"]:
        return {"cache_result": CacheLookupResult(hit=False)}

    result = lookup_cache(deps.qdrant_client, deps.openai_client, state["query"], state["access_context_groups"])
    return {"cache_result": result}


def rewrite_query_node(deps: GraphDeps, state: GraphState) -> dict:
    result = rewrite_query(deps.generation_llm, state["query"])
    return {"rewrite": result}


def retrieve_node(deps: GraphDeps, state: GraphState) -> dict:
    rewrite = state["rewrite"]

    def run(query: str) -> list[RetrievedChunk]:
        return retrieve(
            deps.qdrant_client,
            deps.openai_client,
            deps.sparse_embedder,
            query,
            state["access_context_groups"],
            doc_type=state["doc_type"],
            date_range=state["date_range"],
        )

    primary_chunks = run(rewrite.rewritten_query)
    if not rewrite.sub_queries:
        return {"chunks": primary_chunks}

    def run_sub(query: str) -> list[RetrievedChunk]:
        try:
            return run(query)
        except Exception:
            # FR12/ADR-011: one sub-query's retrieval failing degrades that
            # leg only — the primary retrieval above is the hard dependency.
            return []

    with ThreadPoolExecutor(max_workers=len(rewrite.sub_queries)) as pool:
        sub_results = list(pool.map(run_sub, rewrite.sub_queries))

    return {"chunks": _merge_dedup_chunks([primary_chunks, *sub_results])}


def rerank_node(deps: GraphDeps, state: GraphState) -> dict:
    result = rerank(deps.cohere_client, state["query"], state["chunks"], top_k=state["top_k"])
    return {"chunks": result.chunks, "reranked": result.reranked}


def check_sufficiency_node(deps: GraphDeps, state: GraphState) -> dict:
    # Reuses the faithfulness judge model — this is the same "judge" role,
    # not a new independently-configurable knob (ADR-007's spirit: no new
    # config surface unless there's a stated need for one).
    result = check_sufficiency(deps.faithfulness_llm, state["query"], state["chunks"], state["reranked"])
    return {"sufficiency": result}


def generate_node(deps: GraphDeps, state: GraphState) -> dict:
    messages = state["messages"]
    if not messages:
        rewrite = state["rewrite"]
        sub_questions = rewrite.sub_queries if rewrite is not None else None
        prompt = build_question_prompt(state["query"], state["chunks"], sub_questions=sub_questions)
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]

    allow_retrieve_tool = state["tool_call_count"] < TOOL_CALL_MAX_ROUNDS
    tools = [SubmitAnswer] + ([build_retrieve_tool(deps, state)] if allow_retrieve_tool else [])
    response, result = generate(deps.generation_llm, messages, tools)

    updates: dict = {"messages": [response]}
    if result.finished:
        updates["draft_answer"] = result.answer
        updates["citations"] = result.citations
    return updates


def execute_tool_node(deps: GraphDeps, state: GraphState) -> dict:
    # FR12/ADR-012: a round may request more than one retrieve_chunks call —
    # they run concurrently, capped at MAX_PARALLEL_RETRIEVE_CALLS. Every
    # tool_call_id gets exactly one ToolMessage back, win or lose, since
    # OpenAI's API requires a response for each one on the next turn.
    last_message = state["messages"][-1]
    calls = last_message.tool_calls
    runnable_calls, skipped_calls = calls[:MAX_PARALLEL_RETRIEVE_CALLS], calls[MAX_PARALLEL_RETRIEVE_CALLS:]

    retrieve_tool = build_retrieve_tool(deps, state)

    def run(call: dict) -> tuple[str, list, Exception | None]:
        try:
            return call["id"], retrieve_tool.invoke(call["args"]), None
        except Exception as exc:  # noqa: BLE001 - a failed call degrades that call, not the round
            return call["id"], [], exc

    if len(runnable_calls) <= 1:
        results = [run(call) for call in runnable_calls]
    else:
        with ThreadPoolExecutor(max_workers=len(runnable_calls)) as pool:
            results = list(pool.map(run, runnable_calls))

    tool_messages = []
    new_chunk_lists = []
    for call_id, new_chunks, error in results:
        if error is None:
            new_chunk_lists.append(new_chunks)
            # The model can only cite what it can actually see — the labeled
            # chunk content must round-trip back into the conversation, not
            # just a count.
            tool_messages.append(ToolMessage(content=format_context(new_chunks), tool_call_id=call_id))
        else:
            tool_messages.append(
                ToolMessage(
                    content=(
                        f"This search failed: {error}. Use the results from any other search "
                        "calls, or answer using the context already available."
                    ),
                    tool_call_id=call_id,
                )
            )
    for call in skipped_calls:
        tool_messages.append(
            ToolMessage(
                content=f"Skipped: at most {MAX_PARALLEL_RETRIEVE_CALLS} searches are allowed per round.",
                tool_call_id=call["id"],
            )
        )

    known_ids = {chunk.chunk_id for chunk in state["chunks"]}
    new_chunks = [c for c in _merge_dedup_chunks(new_chunk_lists) if c.chunk_id not in known_ids]

    return {
        "chunks": list(state["chunks"]) + new_chunks,
        "messages": tool_messages,
        "tool_call_count": state["tool_call_count"] + 1,
    }


def faithfulness_node(deps: GraphDeps, state: GraphState) -> dict:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in state["chunks"]}
    result = judge_faithfulness(
        deps.faithfulness_llm, state["query"], state["draft_answer"], state["citations"], chunks_by_id
    )
    return {"faithfulness": result}


def response_node(deps: GraphDeps, state: GraphState) -> dict:
    cache_result = state["cache_result"]
    if cache_result is not None and cache_result.hit:
        # A cache hit never ran retrieve/rerank — retrieved_chunks stays
        # empty, per API-CONTRACTS.md ("always populated when retrieval
        # ran... not on a cache hit").
        return {
            "response": {
                "answer": cache_result.answer,
                "abstained": False,
                "confidence": cache_result.confidence,
                "cache_hit": True,
                "citations": cache_result.citations,
                "retrieved_chunks": [],
            }
        }

    retrieved_chunks = [
        {"chunk_id": chunk.chunk_id, "text": chunk.text, "score": chunk.score} for chunk in state["chunks"]
    ]

    if not state["allow_generation"]:
        return {
            "response": {
                "answer": None,
                "abstained": False,
                "confidence": None,
                "cache_hit": False,
                "citations": [],
                "retrieved_chunks": retrieved_chunks,
            }
        }

    # faithfulness is None either because check_sufficiency short-circuited
    # before generate/faithfulness ever ran (FR15; ADR-010), or — should the
    # graph reach here in some other unexpected way — because it just wasn't
    # set. Either way, no verified pass exists, so abstain.
    faithfulness = state["faithfulness"]
    if faithfulness is None or not faithfulness.passed:
        return {
            "response": {
                "answer": None,
                "abstained": True,
                "confidence": faithfulness.confidence if faithfulness is not None else 0.0,
                "cache_hit": False,
                "citations": [],
                "retrieved_chunks": retrieved_chunks,
            }
        }

    # NFR-9 / API-CONTRACTS.md: enforced structurally — a citation is only
    # ever surfaced if its chunk_id resolves to a chunk actually retrieved
    # for this request, never trusted blindly from the model's output.
    known_ids = {chunk.chunk_id for chunk in state["chunks"]}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in state["chunks"]}
    seen_chunk_ids: set[str] = set()
    valid_citations = []
    for citation in state["citations"]:
        if citation.chunk_id not in known_ids or citation.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(citation.chunk_id)
        chunk = chunks_by_id[citation.chunk_id]
        valid_citations.append(
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "title": chunk.title, "url": chunk.url}
        )

    # FR-6.3: write-through only after a faithfulness pass (this line is
    # only reached once `faithfulness.passed` is confirmed above); FR-5.3's
    # "abstained answers are never cached" is therefore structural, not a
    # separate check. Unconditional on `bypass_cache` — API-CONTRACTS.md:
    # bypass_cache skips the *lookup*, never the write.
    write_cache(
        deps.qdrant_client,
        deps.openai_client,
        state["query"],
        state["access_context_groups"],
        state["draft_answer"],
        valid_citations,
        faithfulness.confidence,
    )

    return {
        "response": {
            "answer": state["draft_answer"],
            "abstained": False,
            "confidence": faithfulness.confidence,
            "cache_hit": False,
            "citations": valid_citations,
            "retrieved_chunks": retrieved_chunks,
        }
    }
