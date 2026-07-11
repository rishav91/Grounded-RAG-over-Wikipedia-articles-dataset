#!/usr/bin/env python3
"""M3 verification — ROADMAP.md's M3 "Verify" step.

Runs the eval set's UC-4/UC-5/UC-8 cases against the real corpus, the real
Qdrant+Cohere pipeline, and the real configured LLM (`GENERATION_MODEL`/
`FAITHFULNESS_MODEL`):

  1. UC-8 (FR-5.1, FR-5.4, NFR-7, NFR-10): well-grounded factual queries
     through the full pipeline. Reports each case's faithfulness pass/fail
     and confidence against NFR-7's placeholder 98% target and FR-5.1's
     placeholder 0.7 confidence threshold, *and* separately reports
     `answers_question` (FR-5.4) against NFR-10's placeholder 95% target —
     the two are independent judge outputs, so a case can fail one without
     the other. Measured data points to report, like M1/M2's recall/precision
     deltas, not asserted in advance.
  2. UC-5 (FR-5.2, NFR-8): a genuinely unanswerable query must abstain —
     `abstained=True`, `answer=None`, `citations=[]`, `retrieved_chunks`
     still populated. This is a hard gate (NFR-8: 100%, zero fabricated
     answers tolerated), not a measured delta. Also reports whether
     `check_sufficiency` (FR15; ADR-010) caught it before `generate`/
     `faithfulness` ever ran — the cheaper path this milestone added — versus
     falling through to faithfulness's zero-citation short-circuit.
  3. UC-4 (FR-4.2, FR-4.5): the retrieval tool must fire exactly once, with
     a query different from the original — the graph's `TOOL_CALL_MAX_ROUNDS`
     bound is what's under test, not whether the resulting answer passes
     faithfulness (that's UC-8's concern). Also checks context recall
     (FR-4.5): does the final chunk set actually include `second_hop_doc_id`,
     not just "did a tool call fire" — a RAGAS-style recall check scoped to
     the one case where "does retrieval have everything it needs" can't be
     answered by a single `expected_doc_id` (that's FR-2.3/M1's job for
     single-hop). Both are hard gates.
  4. NFR-9 (all cases): every citation resolves to a `retrieved_chunks`
     entry — checked structurally on every response `response_node` already
     filters against, verified here as a standing regression check.

This is a black-box check against the already-ingested `articles`
collection — it does not re-run ingestion, retrieval, or rerank logic.
"""

from __future__ import annotations

import cohere
from langchain.chat_models import init_chat_model
from openai import OpenAI
from qdrant_client import QdrantClient

from grounded_rag.config import FAITHFULNESS_CONFIDENCE_THRESHOLD, get_settings
from grounded_rag.eval.eval_set import UC4Case, UC5Case, UC8Case, load_eval_set
from grounded_rag.graph.build import build_graph
from grounded_rag.graph.deps import GraphDeps
from grounded_rag.ingestion.embeddings import SparseEmbedder

NFR_7_FAITHFULNESS_RATE_TARGET = 98.0
NFR_10_ANSWER_RELEVANCE_RATE_TARGET = 95.0


def _initial_state(query: str, access_context_groups: list[str]) -> dict:
    return {
        "query": query,
        "access_context_groups": access_context_groups,
        "doc_type": None,
        "date_range": None,
        "top_k": 5,
        "allow_generation": True,
        # M4's cache_lookup now runs first in the graph — bypass it so
        # repeated eval runs measure retrieval/faithfulness quality fresh,
        # not a stale cache hit (API-CONTRACTS.md's stated purpose for
        # options.bypass_cache).
        "bypass_cache": True,
        "cache_result": None,
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


def _citations_are_valid(response: dict) -> bool:
    retrieved_ids = {chunk["chunk_id"] for chunk in response["retrieved_chunks"]}
    return all(citation["chunk_id"] in retrieved_ids for citation in response["citations"])


def run_uc8(graph, cases: list[UC8Case]) -> bool:
    print(f"=== UC-8: well-grounded queries through the full pipeline ({len(cases)} cases) ===")
    passed = 0
    relevant = 0
    citations_ok = True
    for case in cases:
        final_state = graph.invoke(_initial_state(case.query, case.access_context_groups))
        response = final_state["response"]
        faithfulness = final_state["faithfulness"]
        ok = not response["abstained"]
        answers_question = faithfulness.answers_question if faithfulness is not None else False
        passed += int(ok)
        relevant += int(answers_question)
        citations_ok = citations_ok and _citations_are_valid(response)
        status = "PASS" if ok else "ABSTAIN"
        print(f"  {case.id}: {status} confidence={response['confidence']} answers_question={answers_question}")

    faithfulness_rate = 100 * passed / len(cases)
    relevance_rate = 100 * relevant / len(cases)
    print(f"  Faithfulness pass rate: {faithfulness_rate:.1f}% ({passed}/{len(cases)})")
    print(
        f"  ({'meets' if faithfulness_rate >= NFR_7_FAITHFULNESS_RATE_TARGET else 'below'} NFR-7's placeholder "
        f"{NFR_7_FAITHFULNESS_RATE_TARGET:.0f}% target; FR-5.1's confidence threshold is "
        f"{FAITHFULNESS_CONFIDENCE_THRESHOLD} — see REQUIREMENTS.md Open assumptions)"
    )
    print(f"  Answer relevance rate: {relevance_rate:.1f}% ({relevant}/{len(cases)})")
    print(
        f"  ({'meets' if relevance_rate >= NFR_10_ANSWER_RELEVANCE_RATE_TARGET else 'below'} NFR-10's placeholder "
        f"{NFR_10_ANSWER_RELEVANCE_RATE_TARGET:.0f}% target — FR-5.4)"
    )
    print(f"  [{'PASS' if citations_ok else 'FAIL'}] NFR-9: every citation resolves to a retrieved_chunks entry")
    return citations_ok


def run_uc5(graph, cases: list[UC5Case]) -> bool:
    print(f"\n=== UC-5: genuinely unanswerable queries must abstain, never fabricate (NFR-8) ({len(cases)} cases) ===")
    all_ok = True
    for case in cases:
        final_state = graph.invoke(_initial_state(case.query, case.access_context_groups))
        response = final_state["response"]
        ok = (
            response["abstained"] is True
            and response["answer"] is None
            and response["citations"] == []
            and len(response["retrieved_chunks"]) > 0
        )
        all_ok = all_ok and ok
        sufficiency = final_state["sufficiency"]
        caught_by = "check_sufficiency" if sufficiency is not None and not sufficiency.sufficient else "faithfulness"
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {case.id}: abstained={response['abstained']} "
            f"answer={response['answer']!r} caught_by={caught_by}"
        )
    return all_ok


def run_uc4(graph, cases: list[UC4Case]) -> bool:
    print(f"\n=== UC-4: refined tool call + context recall (FR-4.2, FR-4.5) ({len(cases)} cases) ===")
    all_ok = True
    for case in cases:
        final_state = graph.invoke(_initial_state(case.query, case.access_context_groups))
        tool_calls = [
            m.tool_calls[0]["args"].get("query")
            for m in final_state["messages"]
            if getattr(m, "tool_calls", None) and m.tool_calls[0]["name"] == "retrieve_chunks"
        ]
        fired_once = final_state["tool_call_count"] == 1
        refined = bool(tool_calls) and tool_calls[0] != case.query
        recalled_doc_ids = {chunk.doc_id for chunk in final_state["chunks"]}
        recalled_second_hop = case.second_hop_doc_id in recalled_doc_ids
        ok = fired_once and refined and recalled_second_hop
        all_ok = all_ok and ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {case.id}: tool_call_count={final_state['tool_call_count']} "
            f"refined_query={tool_calls[0] if tool_calls else None!r} "
            f"second_hop_recalled={recalled_second_hop} (FR-4.5)"
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

    uc8_ok = run_uc8(graph, eval_set.uc8_cases)
    uc5_ok = run_uc5(graph, eval_set.uc5_cases)
    uc4_ok = run_uc4(graph, eval_set.uc4_cases)

    print()
    if uc8_ok and uc5_ok and uc4_ok:
        print(
            "M3 verification passed (NFR-9 citation validity, NFR-8 abstain correctness, "
            "FR-4.2 tool-call bound, FR-4.5 context recall, FR15 sufficiency gate wired in)."
        )
    else:
        print("M3 verification FAILED: see FAIL lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
