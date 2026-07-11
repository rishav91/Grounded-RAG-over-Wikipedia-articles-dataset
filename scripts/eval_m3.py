#!/usr/bin/env python3
"""M3 verification — ROADMAP.md's M3 "Verify" step.

Runs the eval set's UC-4/UC-5/UC-8 cases against the real corpus, the real
Qdrant+Cohere pipeline, and the real configured LLM (`GENERATION_MODEL`/
`FAITHFULNESS_MODEL`):

  1. UC-8 (FR-5.1, NFR-7): well-grounded factual queries through the full
     pipeline. Reports each case's faithfulness pass/fail and confidence, and
     the aggregate pass rate against NFR-7's placeholder 98% target and
     FR-5.1's placeholder 0.7 confidence threshold — a measured data point to
     report, like M1/M2's recall/precision deltas, not asserted in advance.
  2. UC-5 (FR-5.2, NFR-8): a genuinely unanswerable query must abstain —
     `abstained=True`, `answer=None`, `citations=[]`, `retrieved_chunks`
     still populated. This is a hard gate (NFR-8: 100%, zero fabricated
     answers tolerated), not a measured delta.
  3. UC-4 (FR-4.2): the retrieval tool must fire exactly once, with a query
     different from the original — the graph's `TOOL_CALL_MAX_ROUNDS` bound
     is what's under test here, not whether the resulting answer passes
     faithfulness (that's UC-8's concern). Also a hard gate.
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
from grounded_rag.graph.run import run_query
from grounded_rag.ingestion.embeddings import SparseEmbedder

NFR_7_FAITHFULNESS_RATE_TARGET = 98.0


def _initial_state(query: str, access_context_groups: list[str]) -> dict:
    return {
        "query": query,
        "access_context_groups": access_context_groups,
        "doc_type": None,
        "date_range": None,
        "top_k": 5,
        "allow_generation": True,
        "chunks": [],
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
    citations_ok = True
    for case in cases:
        response = run_query(graph, case.query, case.access_context_groups)
        ok = not response["abstained"]
        passed += int(ok)
        citations_ok = citations_ok and _citations_are_valid(response)
        status = "PASS" if ok else "ABSTAIN"
        print(f"  {case.id}: {status} confidence={response['confidence']}")

    rate = 100 * passed / len(cases)
    print(f"  Faithfulness pass rate: {rate:.1f}% ({passed}/{len(cases)})")
    print(
        f"  ({'meets' if rate >= NFR_7_FAITHFULNESS_RATE_TARGET else 'below'} NFR-7's placeholder "
        f"{NFR_7_FAITHFULNESS_RATE_TARGET:.0f}% target; FR-5.1's confidence threshold is "
        f"{FAITHFULNESS_CONFIDENCE_THRESHOLD} — see REQUIREMENTS.md Open assumptions)"
    )
    print(f"  [{'PASS' if citations_ok else 'FAIL'}] NFR-9: every citation resolves to a retrieved_chunks entry")
    return citations_ok


def run_uc5(graph, cases: list[UC5Case]) -> bool:
    print(f"\n=== UC-5: genuinely unanswerable queries must abstain, never fabricate (NFR-8) ({len(cases)} cases) ===")
    all_ok = True
    for case in cases:
        response = run_query(graph, case.query, case.access_context_groups)
        ok = (
            response["abstained"] is True
            and response["answer"] is None
            and response["citations"] == []
            and len(response["retrieved_chunks"]) > 0
        )
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {case.id}: abstained={response['abstained']} answer={response['answer']!r}")
    return all_ok


def run_uc4(graph, cases: list[UC4Case]) -> bool:
    print(f"\n=== UC-4: exactly one refined tool call fires (FR-4.2) ({len(cases)} cases) ===")
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
        ok = fired_once and refined
        all_ok = all_ok and ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {case.id}: tool_call_count={final_state['tool_call_count']} "
            f"refined_query={tool_calls[0] if tool_calls else None!r}"
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
        print("M3 verification passed (NFR-9 citation validity, NFR-8 abstain correctness, FR-4.2 tool-call bound).")
    else:
        print("M3 verification FAILED: see FAIL lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
