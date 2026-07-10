#!/usr/bin/env python3
"""M2 verification — ROADMAP.md's M2 "Verify" step.

Runs the eval set's UC-2 cases against the real corpus:
  1. UC-2: for each ambiguous-candidate query, fetches the fused candidate
     set (M1's `retrieve`, unchanged) and computes precision@3 — the
     fraction of the top-3 chunks that belong to the query's genuinely
     correct expected_doc_id — for fusion-only ordering vs. Cohere-reranked
     ordering (FR-3.1). This is a measured data point to report, not a
     pass/fail gate: REQUIREMENTS.md's "Open assumptions" section already
     flags the +15-point margin as a placeholder pending this exact real
     measurement (mirrors M1's treatment of the FR-2.3 recall margin).
  2. FR-3.2: a simulated Cohere failure (an invalid API key) must still
     degrade to fusion-only ranking rather than raising — this one is a
     correctness assertion and does fail the run if wrong.

This is a black-box check against the already-ingested `articles`
collection — it does not re-run ingestion or retrieval logic.
"""

from __future__ import annotations

import cohere
from openai import OpenAI

from grounded_rag.config import RETRIEVE_CANDIDATE_K, get_settings
from grounded_rag.eval.eval_set import load_eval_set
from grounded_rag.ingestion.embeddings import SparseEmbedder
from grounded_rag.ingestion.qdrant_store import get_client
from grounded_rag.rerank.rerank import rerank
from grounded_rag.retrieval.retrieve import retrieve

PRECISION_K = 3
FR_3_1_MARGIN_POINTS = 15


def precision_at_k(chunks, expected_doc_id: str, k: int) -> float:
    top = chunks[:k]
    if not top:
        return 0.0
    return 100 * sum(1 for c in top if c.doc_id == expected_doc_id) / len(top)


def main() -> None:
    settings = get_settings()
    qdrant_client = get_client(settings)
    openai_client = OpenAI(api_key=settings.openai_api_key)
    cohere_client = cohere.ClientV2(api_key=settings.cohere_api_key)
    sparse_embedder = SparseEmbedder()
    eval_set = load_eval_set()

    print(f"=== UC-2: ambiguous-candidate precision@{PRECISION_K} (fusion-only vs. rerank) ===")
    fusion_precisions = []
    rerank_precisions = []
    for case in eval_set.uc2_cases:
        fused_chunks = retrieve(
            qdrant_client,
            openai_client,
            sparse_embedder,
            case.query,
            case.access_context_groups,
            candidate_k=RETRIEVE_CANDIDATE_K,
        )
        rerank_result = rerank(cohere_client, case.query, fused_chunks, top_k=PRECISION_K)
        if not rerank_result.reranked:
            print(f"  {case.id}: WARNING — Cohere rerank call failed, precision numbers below are fusion-only twice")

        fusion_p = precision_at_k(fused_chunks, case.expected_doc_id, PRECISION_K)
        rerank_p = precision_at_k(rerank_result.chunks, case.expected_doc_id, PRECISION_K)
        fusion_precisions.append(fusion_p)
        rerank_precisions.append(rerank_p)
        print(f"  {case.id}: fusion-only={fusion_p:.1f}% rerank={rerank_p:.1f}%")

    n = len(eval_set.uc2_cases)
    avg_fusion = sum(fusion_precisions) / n
    avg_rerank = sum(rerank_precisions) / n
    delta = avg_rerank - avg_fusion
    print(f"  Precision@{PRECISION_K} fusion-only: {avg_fusion:.1f}%")
    print(f"  Precision@{PRECISION_K} rerank:      {avg_rerank:.1f}%")
    print(
        f"  delta: {delta:+.1f} points "
        f"({'meets' if delta >= FR_3_1_MARGIN_POINTS else 'below'} the FR-3.1 placeholder margin of "
        f"+{FR_3_1_MARGIN_POINTS} points — see REQUIREMENTS.md Open assumptions)"
    )

    print("\n=== FR-3.2: simulated Cohere failure degrades to fusion-only, never raises ===")
    broken_cohere_client = cohere.ClientV2(api_key="invalid-simulated-key")
    probe_chunks = retrieve(
        qdrant_client,
        openai_client,
        sparse_embedder,
        eval_set.uc2_cases[0].query,
        eval_set.uc2_cases[0].access_context_groups,
        candidate_k=RETRIEVE_CANDIDATE_K,
    )
    fallback_result = rerank(broken_cohere_client, eval_set.uc2_cases[0].query, probe_chunks, top_k=PRECISION_K)
    fallback_ok = (not fallback_result.reranked) and fallback_result.chunks == probe_chunks[:PRECISION_K]
    print(f"  [{'PASS' if fallback_ok else 'FAIL'}] invalid API key still returned fusion-ranked chunks, no exception")

    print()
    if not fallback_ok:
        print("M2 verification FAILED: FR-3.2's fallback path did not degrade as expected.")
        raise SystemExit(1)
    print("M2 verification passed (FR-3.2 fallback gate). See UC-2 precision numbers above.")


if __name__ == "__main__":
    main()
