#!/usr/bin/env python3
"""M0 verification — ROADMAP.md's M0 "Verify" step and FR-1.1's acceptance criterion.

Queries the `articles` collection directly with a payload filter
(doc_type=long AND acl_tags contains "public") and confirms:
  1. total chunk count falls in the 5,000-15,000 range (5-15 chunks/doc at 1K docs)
  2. every point the filtered query returns actually matches the filter
  3. the filter is non-trivial — it excludes at least one point that exists
     in the unfiltered collection (otherwise the "filter" proves nothing)
  4. a sample of points actually carry a full-length dense vector and a
     non-empty sparse vector — PRD.md's M0 exit criterion is that every
     document is "searchable via both the dense and sparse representation",
     which the payload-only checks above don't verify

This is a black-box check against whatever the ingestion script already
wrote to Qdrant — it does not re-run ingestion.
"""

from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchValue

from grounded_rag.config import ARTICLES_COLLECTION, DENSE_VECTOR_NAME, EMBEDDING_DIM, SPARSE_VECTOR_NAME, get_settings
from grounded_rag.ingestion.qdrant_store import get_client

CHUNK_COUNT_MIN = 5_000
CHUNK_COUNT_MAX = 15_000
VECTOR_SAMPLE_SIZE = 20


def main() -> None:
    client = get_client(get_settings())

    total = client.count(ARTICLES_COLLECTION, exact=True).count
    print(f"total chunks: {total}")
    in_range = CHUNK_COUNT_MIN <= total <= CHUNK_COUNT_MAX
    print(f"  [{'PASS' if in_range else 'FAIL'}] in expected range [{CHUNK_COUNT_MIN}, {CHUNK_COUNT_MAX}]")

    payload_filter = Filter(
        must=[
            FieldCondition(key="doc_type", match=MatchValue(value="long")),
            FieldCondition(key="acl_tags", match=MatchValue(value="public")),
        ]
    )
    filtered_count = client.count(ARTICLES_COLLECTION, count_filter=payload_filter, exact=True).count
    print(f'filtered chunks (doc_type=long AND acl_tags contains "public"): {filtered_count}')

    sample, _ = client.scroll(ARTICLES_COLLECTION, scroll_filter=payload_filter, limit=100, with_payload=True)
    all_match = all(
        point.payload["doc_type"] == "long" and "public" in point.payload["acl_tags"] for point in sample
    )
    print(f"  [{'PASS' if all_match else 'FAIL'}] every sampled point matches the filter ({len(sample)} sampled)")

    non_trivial = 0 < filtered_count < total
    print(f"  [{'PASS' if non_trivial else 'FAIL'}] filter is non-trivial ({filtered_count} of {total} match)")

    vector_sample, _ = client.scroll(ARTICLES_COLLECTION, limit=VECTOR_SAMPLE_SIZE, with_vectors=True)
    vectors_ok = all(
        len(point.vector.get(DENSE_VECTOR_NAME, [])) == EMBEDDING_DIM
        and len(point.vector.get(SPARSE_VECTOR_NAME).indices) > 0
        for point in vector_sample
    )
    print(
        f"  [{'PASS' if vectors_ok else 'FAIL'}] sampled points carry a "
        f"{EMBEDDING_DIM}-dim dense vector and a non-empty sparse vector ({len(vector_sample)} sampled)"
    )

    if not (in_range and all_match and non_trivial and vectors_ok):
        raise SystemExit(1)
    print("M0 verification passed.")


if __name__ == "__main__":
    main()
