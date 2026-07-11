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
class EvalSet:
    uc1_cases: list[UC1Case]
    uc2_cases: list[UC2Case]
    uc3_cases: list[UC3Case]
    uc4_cases: list[UC4Case]
    uc5_cases: list[UC5Case]
    uc8_cases: list[UC8Case]


def load_eval_set() -> EvalSet:
    raw = json.loads(resources.files("grounded_rag.eval").joinpath("eval_set.json").read_text())
    return EvalSet(
        uc1_cases=[UC1Case(**case) for case in raw["uc1_cases"]],
        uc2_cases=[UC2Case(**case) for case in raw["uc2_cases"]],
        uc3_cases=[UC3Case(**case) for case in raw["uc3_cases"]],
        uc4_cases=[UC4Case(**case) for case in raw["uc4_cases"]],
        uc5_cases=[UC5Case(**case) for case in raw["uc5_cases"]],
        uc8_cases=[UC8Case(**case) for case in raw["uc8_cases"]],
    )
