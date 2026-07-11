"""Loader for the eval set — PRD.md's UC-1/UC-2/UC-3 cases, grounded in real
doc_ids from the ingested corpus (not placeholders)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class UC1Case:
    """UC-1: single-hop factual query — the expected doc should appear in the top-k."""

    id: str
    query: str
    access_context_groups: list[str]
    expected_doc_id: str


@dataclass(frozen=True)
class UC2Case:
    """UC-2: ambiguous-candidate query — several similarly-titled docs compete for top-k,
    but only expected_doc_id genuinely answers the query. Reranking should promote its
    chunks over the competing_doc_ids' chunks; measured as precision@k, not by inspection."""

    id: str
    query: str
    access_context_groups: list[str]
    expected_doc_id: str
    competing_doc_ids: list[str]


@dataclass(frozen=True)
class UC3Case:
    """UC-3: ACL/metadata filter — the restricted doc must (not) appear in the candidate set."""

    id: str
    query: str
    access_context_groups: list[str]
    restricted_doc_id: str
    expect_excluded: bool


@dataclass(frozen=True)
class UC4Case:
    """UC-4: multi-hop tool use — the first-pass doc names an entity (second_hop_doc_id's
    subject) without itself containing the fact the query actually asks for, so answering
    requires the generator to call the retrieval tool a second time (FR-4.2)."""

    id: str
    query: str
    access_context_groups: list[str]
    first_pass_doc_id: str
    second_hop_doc_id: str


@dataclass(frozen=True)
class UC5Case:
    """UC-5: genuinely unanswerable from the 1K slice — no supporting article exists."""

    id: str
    query: str
    access_context_groups: list[str]


@dataclass(frozen=True)
class UC8Case:
    """UC-8: straightforward answerable factual query, run through the full pipeline —
    the faithfulness positive case."""

    id: str
    query: str
    access_context_groups: list[str]
    expected_doc_id: str


@dataclass(frozen=True)
class UC6Case:
    """UC-6: a query answered once, then repeated verbatim under the same
    access_context — the second call must be a cache hit (FR-6.1)."""

    id: str
    query: str
    access_context_groups: list[str]


@dataclass(frozen=True)
class UC7Case:
    """UC-7: the same query text under two different access_context values,
    where context B lacks a group context A has — B must never receive a
    cache hit warmed by A's answer (FR-6.2), a standing regression test
    from M4 onward."""

    id: str
    query: str
    access_context_groups_a: list[str]
    access_context_groups_b: list[str]


@dataclass(frozen=True)
class UC9Case:
    """UC-9: a bundled, genuinely independent multi-part query — each half
    names its own subject already, so rewrite_query should decompose it into
    sub_queries and retrieve both articles' chunks in the first pass, without
    the reactive tool call ever firing (FR11; ADR-011)."""

    id: str
    query: str
    access_context_groups: list[str]
    doc_id_a: str
    doc_id_b: str


@dataclass(frozen=True)
class EvalSet:
    uc1_cases: list[UC1Case]
    uc2_cases: list[UC2Case]
    uc3_cases: list[UC3Case]
    uc4_cases: list[UC4Case]
    uc5_cases: list[UC5Case]
    uc8_cases: list[UC8Case]
    uc6_cases: list[UC6Case]
    uc7_cases: list[UC7Case]
    uc9_cases: list[UC9Case]


def load_eval_set() -> EvalSet:
    raw = json.loads(resources.files("grounded_rag.eval").joinpath("eval_set.json").read_text())
    return EvalSet(
        uc1_cases=[UC1Case(**case) for case in raw["uc1_cases"]],
        uc2_cases=[UC2Case(**case) for case in raw["uc2_cases"]],
        uc3_cases=[UC3Case(**case) for case in raw["uc3_cases"]],
        uc4_cases=[UC4Case(**case) for case in raw["uc4_cases"]],
        uc5_cases=[UC5Case(**case) for case in raw["uc5_cases"]],
        uc8_cases=[UC8Case(**case) for case in raw["uc8_cases"]],
        uc6_cases=[UC6Case(**case) for case in raw["uc6_cases"]],
        uc7_cases=[UC7Case(**case) for case in raw["uc7_cases"]],
        uc9_cases=[UC9Case(**case) for case in raw["uc9_cases"]],
    )
