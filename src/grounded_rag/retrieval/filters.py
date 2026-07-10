"""ACL/metadata payload filter, applied before fusion per ADR-008.

API-CONTRACTS.md's `access_context.groups` semantics: a chunk is visible if
its `acl_tags` intersects the caller's groups at all (`MatchAny`); an empty
group list must resolve to zero visible documents — handled by `retrieve()`
short-circuiting before ever calling Qdrant, not by relying on
`MatchAny(any=[])`'s (undocumented) empty-list behavior.

`date_range` matches if *either* `created_at` or `updated_at` falls in the
range (OR, via Qdrant's `should` clause) — the literal reading of
API-CONTRACTS.md's "inclusive range over `created_at`/`updated_at`".
"""

from __future__ import annotations

from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchAny, MatchValue


def build_filter(
    access_context_groups: list[str],
    doc_type: str | None = None,
    date_range: dict[str, str] | None = None,
) -> Filter:
    must: list[FieldCondition] = [FieldCondition(key="acl_tags", match=MatchAny(any=access_context_groups))]
    if doc_type is not None:
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))

    should: list[FieldCondition] | None = None
    if date_range and (date_range.get("from") or date_range.get("to")):
        date_bounds = DatetimeRange(gte=date_range.get("from"), lte=date_range.get("to"))
        should = [
            FieldCondition(key="created_at", range=date_bounds),
            FieldCondition(key="updated_at", range=date_bounds),
        ]

    return Filter(must=must, should=should)
