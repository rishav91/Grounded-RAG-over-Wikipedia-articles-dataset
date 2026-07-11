"""`RewriteResult`: `rewrite_query`'s output — the query used for the
first-pass `retrieve` call, plus any independently-retrievable sub-queries
(FR11; ADR-011)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewriteResult:
    rewritten_query: str
    sub_queries: list[str] = field(default_factory=list)
    # False when the LLM call failed and this defaulted to a pass-through
    # (rewritten_query = the raw query, sub_queries = []) — kept for
    # eval/observability, mirrors SufficiencyResult.checked_by_llm.
    rewritten_by_llm: bool = False
