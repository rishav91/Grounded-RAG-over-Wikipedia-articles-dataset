"""Records produced by `judge_faithfulness` — the graded support judgment
gating the abstain decision (FR6, FR7; ADR-006)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    chunk_id: str
    supported: bool


@dataclass(frozen=True)
class FaithfulnessResult:
    passed: bool  # judge's verdict AND confidence >= FAITHFULNESS_CONFIDENCE_THRESHOLD (FR-5.1)
    confidence: float
    reasoning: str
    claim_checks: list[ClaimCheck] = field(default_factory=list)
