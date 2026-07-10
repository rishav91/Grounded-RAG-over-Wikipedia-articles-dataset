"""Loader for the M1 eval set — PRD.md's UC-1/UC-3 cases, grounded in real
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
class UC3Case:
    """UC-3: ACL/metadata filter — the restricted doc must (not) appear in the candidate set."""

    id: str
    query: str
    access_context_groups: list[str]
    restricted_doc_id: str
    expect_excluded: bool


@dataclass(frozen=True)
class EvalSet:
    uc1_cases: list[UC1Case]
    uc3_cases: list[UC3Case]


def load_eval_set() -> EvalSet:
    raw = json.loads(resources.files("grounded_rag.eval").joinpath("eval_set.json").read_text())
    return EvalSet(
        uc1_cases=[UC1Case(**case) for case in raw["uc1_cases"]],
        uc3_cases=[UC3Case(**case) for case in raw["uc3_cases"]],
    )
