#!/usr/bin/env python3
"""M4 verification — ROADMAP.md's M4 "Verify" step.

Runs the eval set's UC-6/UC-7 cases against the real corpus, the real
Qdrant+Cohere pipeline, and the real configured LLM (`GENERATION_MODEL`/
`FAITHFULNESS_MODEL`), through the same compiled graph `eval_m3.py` uses —
M4 only adds `cache_lookup` in front of it and a write-through in
`build_response`, nothing else in the pipeline changes:

  1. UC-6 (FR-6.1): a query answered once, then repeated verbatim under the
     same `access_context` — the second call must be `cache_hit: true` with
     the same answer, without `retrieve`/`rerank`/`generate`/`faithfulness`
     running again.
  2. UC-7 (FR-6.2): the same query text under two different `access_context`
     values, where context B lacks a group context A has — B must never
     receive a cache hit warmed by A's answer. A hard gate (the governing
     principle: no answer crosses an access boundary), not a measured
     delta — this is a standing regression test from M4 onward.

`query_cache` is reset (deleted + recreated empty) at the start of the run
so results are deterministic across repeated invocations — DATA-MODEL.md
flags TTL/pruning as an unbuilt gap for production, but a clean slate is
what a repeatable eval run needs. This does not touch the `articles`
collection.
"""

from __future__ import annotations

import cohere
from langchain.chat_models import init_chat_model
from openai import OpenAI
from qdrant_client import QdrantClient

from grounded_rag.cache.store import ensure_collection
from grounded_rag.config import QUERY_CACHE_COLLECTION, get_settings
from grounded_rag.eval.eval_set import UC6Case, UC7Case, load_eval_set
from grounded_rag.graph.build import build_graph
from grounded_rag.graph.deps import GraphDeps
from grounded_rag.ingestion.embeddings import SparseEmbedder


def _initial_state(query: str, access_context_groups: list[str]) -> dict:
    return {
        "query": query,
        "access_context_groups": access_context_groups,
        "doc_type": None,
        "date_range": None,
        "top_k": 5,
        "allow_generation": True,
        "bypass_cache": False,
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


def _reset_cache_collection(qdrant_client: QdrantClient) -> None:
    if qdrant_client.collection_exists(QUERY_CACHE_COLLECTION):
        qdrant_client.delete_collection(QUERY_CACHE_COLLECTION)
    ensure_collection(qdrant_client)


def run_uc6(graph, cases: list[UC6Case]) -> bool:
    print(f"=== UC-6: repeat query, same access context, must cache hit (FR-6.1) ({len(cases)} cases) ===")
    all_ok = True
    for case in cases:
        first = graph.invoke(_initial_state(case.query, case.access_context_groups))["response"]
        second = graph.invoke(_initial_state(case.query, case.access_context_groups))["response"]
        ok = (
            first["cache_hit"] is False
            and first["abstained"] is False
            and second["cache_hit"] is True
            and second["answer"] == first["answer"]
        )
        all_ok = all_ok and ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {case.id}: first_cache_hit={first['cache_hit']} "
            f"first_abstained={first['abstained']} second_cache_hit={second['cache_hit']}"
        )
    return all_ok


def run_uc7(graph, cases: list[UC7Case]) -> bool:
    print(f"\n=== UC-7: cross-context cache safety, never shares a hit (FR-6.2) ({len(cases)} cases) ===")
    all_ok = True
    for case in cases:
        context_a = graph.invoke(_initial_state(case.query, case.access_context_groups_a))["response"]
        context_b = graph.invoke(_initial_state(case.query, case.access_context_groups_b))["response"]
        ok = context_b["cache_hit"] is False
        all_ok = all_ok and ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {case.id}: context_a_cache_hit={context_a['cache_hit']} "
            f"context_b_cache_hit={context_b['cache_hit']} (must be False)"
        )
    return all_ok


def main() -> None:
    settings = get_settings()
    if not settings.generation_model:
        raise SystemExit("GENERATION_MODEL is not set — see .env.example (ADR-007).")

    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    deps = GraphDeps(
        qdrant_client=qdrant_client,
        openai_client=OpenAI(api_key=settings.openai_api_key),
        cohere_client=cohere.ClientV2(api_key=settings.cohere_api_key),
        sparse_embedder=SparseEmbedder(),
        generation_llm=init_chat_model(settings.generation_model),
        faithfulness_llm=init_chat_model(settings.faithfulness_model),
    )
    _reset_cache_collection(qdrant_client)
    graph = build_graph(deps)
    eval_set = load_eval_set()

    uc6_ok = run_uc6(graph, eval_set.uc6_cases)
    uc7_ok = run_uc7(graph, eval_set.uc7_cases)

    print()
    if uc6_ok and uc7_ok:
        print("M4 verification passed (FR-6.1 cache hit, FR-6.2 cross-context cache safety).")
    else:
        print("M4 verification FAILED: see FAIL lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
