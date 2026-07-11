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
    # passed: judge's groundedness verdict AND answers_question AND
    # confidence >= FAITHFULNESS_CONFIDENCE_THRESHOLD (FR-5.1, FR-5.4)
    passed: bool
    confidence: float
    reasoning: str
    answers_question: bool = False  # FR-5.4: does the answer actually address the question, not just cite real chunks
    claim_checks: list[ClaimCheck] = field(default_factory=list)
