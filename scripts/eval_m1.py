#!/usr/bin/env python3
"""M1 verification — ROADMAP.md's M1 "Verify" step.

Runs the eval set's UC-1/UC-3 cases against the real corpus:
  1. UC-1: for each single-hop query, checks whether the expected doc_id
     appears in the top-5 candidates, in both hybrid and dense-only mode —
     reports Recall@5 for each and the delta (FR-2.3). This is a measured
     data point to report, not a pass/fail gate: REQUIREMENTS.md's "Open
     assumptions" section already flags the +10-point margin as a
     placeholder pending this exact real measurement.
  2. UC-3: for each ACL case, checks whether the restricted doc_id appears
     anywhere in the *full candidate set* (not just top-5) — this one is a
     correctness assertion (FR-2.2) and does fail the run if wrong.

This is a black-box check against the already-ingested `articles`
collection — it does not re-run ingestion.
"""

from __future__ import annotations

from openai import OpenAI

from grounded_rag.config import RETRIEVE_CANDIDATE_K, get_settings
from grounded_rag.eval.eval_set import load_eval_set
from grounded_rag.ingestion.embeddings import SparseEmbedder
from grounded_rag.ingestion.qdrant_store import get_client
from grounded_rag.retrieval.retrieve import retrieve

RECALL_K = 5
FR_2_3_MARGIN_POINTS = 10


def main() -> None:
    settings = get_settings()
    qdrant_client = get_client(settings)
    openai_client = OpenAI(api_key=settings.openai_api_key)
    sparse_embedder = SparseEmbedder()
    eval_set = load_eval_set()

    print(f"=== UC-1: single-hop recall@{RECALL_K} (hybrid vs. dense-only) ===")
    hybrid_hits = 0
    dense_hits = 0
    for case in eval_set.uc1_cases:
        hybrid_chunks = retrieve(
            qdrant_client, openai_client, sparse_embedder, case.query, case.access_context_groups, candidate_k=RECALL_K
        )
        dense_chunks = retrieve(
            qdrant_client,
            openai_client,
            sparse_embedder,
            case.query,
            case.access_context_groups,
            candidate_k=RECALL_K,
            use_sparse=False,
        )
        hybrid_hit = any(c.doc_id == case.expected_doc_id for c in hybrid_chunks)
        dense_hit = any(c.doc_id == case.expected_doc_id for c in dense_chunks)
        hybrid_hits += hybrid_hit
        dense_hits += dense_hit
        print(f"  {case.id}: hybrid={'HIT' if hybrid_hit else 'MISS'} dense-only={'HIT' if dense_hit else 'MISS'}")

    n = len(eval_set.uc1_cases)
    hybrid_recall = 100 * hybrid_hits / n
    dense_recall = 100 * dense_hits / n
    delta = hybrid_recall - dense_recall
    print(f"  Recall@{RECALL_K} hybrid:     {hybrid_recall:.1f}% ({hybrid_hits}/{n})")
    print(f"  Recall@{RECALL_K} dense-only: {dense_recall:.1f}% ({dense_hits}/{n})")
    print(
        f"  delta: {delta:+.1f} points "
        f"({'meets' if delta >= FR_2_3_MARGIN_POINTS else 'below'} the FR-2.3 placeholder margin of "
        f"+{FR_2_3_MARGIN_POINTS} points — see REQUIREMENTS.md Open assumptions)"
    )

    print(f"\n=== UC-3: ACL/metadata filter, checked against the full candidate set (k={RETRIEVE_CANDIDATE_K}) ===")
    uc3_all_pass = True
    for case in eval_set.uc3_cases:
        chunks = retrieve(
            qdrant_client,
            openai_client,
            sparse_embedder,
            case.query,
            case.access_context_groups,
            candidate_k=RETRIEVE_CANDIDATE_K,
        )
        present = any(c.doc_id == case.restricted_doc_id for c in chunks)
        excluded = not present
        passed = excluded == case.expect_excluded
        uc3_all_pass &= passed
        expectation = "excluded" if case.expect_excluded else "included"
        print(f"  [{'PASS' if passed else 'FAIL'}] {case.id}: expected {expectation}, doc present={present}")

    print()
    if not uc3_all_pass:
        print("M1 verification FAILED: at least one UC-3 case did not behave as expected.")
        raise SystemExit(1)
    print("M1 verification passed (UC-3 correctness gate). See UC-1 recall numbers above.")


if __name__ == "__main__":
    main()
