#!/usr/bin/env python3
"""M5 verification — ROADMAP.md's M5 "Verify" step.

Runs the eval set's UC-9 case (new this milestone) plus a re-check of M3's
UC-4 case, against the real corpus, the real Qdrant+Cohere pipeline, and the
real configured LLM (`GENERATION_MODEL`/`FAITHFULNESS_MODEL`), through the
same compiled graph `eval_m3.py`/`eval_m4.py` use — M5 only adds
`rewrite_query` in front of `retrieve` and lets `generate` request more than
one `retrieve_chunks` call per round, nothing else in the pipeline changes:

  1. UC-9 (FR-7.1; ADR-011): a bundled, genuinely independent two-part
     question, run `UC9_REPEATS` times. Two distinct things are measured,
     mirroring how `eval_m3.py` splits UC-8 into a hard gate (citation
     validity) and a measured rate (faithfulness/relevance): the *decompose
     mechanism* (`rewrite_query` produces `sub_queries`; `retrieve` recalls
     both halves' source articles in the first pass; the reactive tool-call
     loop never has to fire) is a hard gate, expected to hold every run —
     it's deterministic wiring, not a judgment call. Whether the generator's
     answer actually *covers both halves* is reported as a rate, not a hard
     gate — the same treatment `REQUIREMENTS.md`'s NFR-7/NFR-10 give M3's
     UC-8, since this is generation-quality variance (`gpt-4o-mini`
     sometimes answers only the more prominent half despite both docs being
     in context), not a defect in decompose's own mechanism.
  2. UC-4 regression check (FR-7.1's other half, ADR-011's whole reason for
     existing): M3's sequential multi-hop case must be unaffected by
     `rewrite_query` being wired in — `sub_queries` must stay empty (the
     independence judgment correctly declines to decompose a question whose
     second half depends on the first half's answer) and the reactive
     tool-call loop must still fire exactly once, recalling the second hop,
     exactly as `eval_m3.py` already verifies.

FR-7.2 (parallel tool calls, partial-failure handling; ADR-012) is
deliberately not exercised here: forcing one of several concurrent live API
calls to fail on demand isn't something this eval harness can do
deterministically. That property is verified by a unit test instead
(`tests/test_graph_nodes.py::test_execute_tool_node_runs_concurrent_calls_and_degrades_on_partial_failure`),
which injects the failure directly — see `ADR-012`.

This is a black-box check against the already-ingested `articles`
collection — it does not re-run ingestion, retrieval, or rerank logic.
"""

from __future__ import annotations

import cohere
from langchain.chat_models import init_chat_model
from openai import OpenAI
from qdrant_client import QdrantClient

from grounded_rag.config import get_settings
from grounded_rag.eval.eval_set import UC4Case, UC9Case, load_eval_set
from grounded_rag.graph.build import build_graph
from grounded_rag.graph.deps import GraphDeps
from grounded_rag.ingestion.embeddings import SparseEmbedder

# UC-9's "answered both halves" is generation-quality variance, not
# deterministic wiring — repeated like M3's UC-8 to measure a rate rather
# than assert a single run.
UC9_REPEATS = 3


def _initial_state(query: str, access_context_groups: list[str]) -> dict:
    return {
        "query": query,
        "access_context_groups": access_context_groups,
        "doc_type": None,
        "date_range": None,
        "top_k": 5,
        "allow_generation": True,
        # Bypass the M4 cache so repeated eval runs measure rewrite/retrieval
        # quality fresh, not a stale cache hit (same reasoning as eval_m3.py).
        "bypass_cache": True,
        "cache_result": None,
        "rewrite": None,
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


def run_uc9(graph, cases: list[UC9Case]) -> bool:
    print(f"=== UC-9: bundled independent multi-part query (FR-7.1) ({len(cases)} case(s), {UC9_REPEATS} runs each) ===")
    mechanism_ok = True
    answered_count = 0
    total_runs = 0
    for case in cases:
        for run_idx in range(1, UC9_REPEATS + 1):
            final_state = graph.invoke(_initial_state(case.query, case.access_context_groups))
            response = final_state["response"]
            rewrite = final_state["rewrite"]
            decomposed = bool(rewrite.sub_queries)
            recalled_doc_ids = {chunk.doc_id for chunk in final_state["chunks"]}
            recalled_both = case.doc_id_a in recalled_doc_ids and case.doc_id_b in recalled_doc_ids
            never_needed_tool_call = final_state["tool_call_count"] == 0
            cited_doc_ids = {c["doc_id"] for c in response["citations"]}
            answered_both = (
                not response["abstained"] and case.doc_id_a in cited_doc_ids and case.doc_id_b in cited_doc_ids
            )

            mechanism = decomposed and recalled_both and never_needed_tool_call
            mechanism_ok = mechanism_ok and mechanism
            total_runs += 1
            answered_count += int(answered_both)
            print(
                f"  [{'PASS' if mechanism else 'FAIL'}] {case.id} run {run_idx}: decomposed={decomposed} "
                f"sub_queries={rewrite.sub_queries!r} recalled_both_docs={recalled_both} "
                f"tool_call_count={final_state['tool_call_count']} answered_both={answered_both} "
                f"abstained={response['abstained']}"
            )

    answered_rate = 100 * answered_count / total_runs if total_runs else 0.0
    print(f"  Decompose mechanism (structural, hard gate): {'PASS' if mechanism_ok else 'FAIL'} across all runs")
    print(
        f"  Answered-both-parts rate: {answered_rate:.1f}% ({answered_count}/{total_runs}) — measured, not a "
        "hard gate (same treatment as M3's UC-8 faithfulness/relevance rate; see REQUIREMENTS.md Open assumptions)"
    )
    return mechanism_ok


def run_uc4_regression(graph, cases: list[UC4Case]) -> bool:
    print(
        f"\n=== UC-4 regression: sequential multi-hop must NOT decompose, "
        f"still uses FR8's reactive tool call ({len(cases)} cases) ==="
    )
    all_ok = True
    for case in cases:
        final_state = graph.invoke(_initial_state(case.query, case.access_context_groups))
        rewrite = final_state["rewrite"]
        not_decomposed = rewrite.sub_queries == []
        fired_once = final_state["tool_call_count"] == 1
        recalled_doc_ids = {chunk.doc_id for chunk in final_state["chunks"]}
        recalled_second_hop = case.second_hop_doc_id in recalled_doc_ids
        ok = not_decomposed and fired_once and recalled_second_hop
        all_ok = all_ok and ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {case.id}: sub_queries={rewrite.sub_queries!r} "
            f"tool_call_count={final_state['tool_call_count']} second_hop_recalled={recalled_second_hop}"
        )
    return all_ok


def main() -> None:
    settings = get_settings()
    if not settings.generation_model:
        raise SystemExit("GENERATION_MODEL is not set — see .env.example (ADR-007).")

    deps = GraphDeps(
        qdrant_client=QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        openai_client=OpenAI(api_key=settings.openai_api_key),
        cohere_client=cohere.ClientV2(api_key=settings.cohere_api_key),
        sparse_embedder=SparseEmbedder(),
        generation_llm=init_chat_model(settings.generation_model),
        faithfulness_llm=init_chat_model(settings.faithfulness_model),
    )
    graph = build_graph(deps)
    eval_set = load_eval_set()

    uc9_ok = run_uc9(graph, eval_set.uc9_cases)
    uc4_ok = run_uc4_regression(graph, eval_set.uc4_cases)

    print()
    if uc9_ok and uc4_ok:
        print(
            "M5 verification passed (FR-7.1 decompose-and-answer, UC-4 sequential multi-hop unaffected; "
            "FR-7.2 partial-failure handling verified by unit test, see this script's docstring)."
        )
    else:
        print("M5 verification FAILED: see FAIL lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
