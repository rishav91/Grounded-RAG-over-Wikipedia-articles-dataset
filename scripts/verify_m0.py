#!/usr/bin/env python3
"""M0 verification — ROADMAP.md's M0 "Verify" step and FR-1.1's acceptance criterion.

Queries the `articles` collection directly with a payload filter
(doc_type=long AND acl_tags contains "public") and confirms:
  1. total chunk count falls in the 5,000-15,000 range (5-15 chunks/doc at 1K docs)
  2. every point the filtered query returns actually matches the filter
  3. the filter is non-trivial — it excludes at least one point that exists
     in the unfiltered collection (otherwise the "filter" proves nothing)

This is a black-box check against whatever the ingestion script already
wrote to Qdrant — it does not re-run ingestion.
"""

from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchValue

from grounded_rag.config import ARTICLES_COLLECTION, get_settings
from grounded_rag.ingestion.qdrant_store import get_client

CHUNK_COUNT_MIN = 5_000
CHUNK_COUNT_MAX = 15_000


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

    if not (in_range and all_match and non_trivial):
        raise SystemExit(1)
    print("M0 verification passed.")


if __name__ == "__main__":
    main()
