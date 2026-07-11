"""Records produced by `check_sufficiency` — a context-only, reference-free
judgment of whether retrieval found enough to attempt an answer (FR15;
ADR-010). Distinct from `FaithfulnessResult`: this never looks at a drafted
answer, only at the question and the retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SufficiencyResult:
    sufficient: bool
    confidence: float
    reasoning: str
    missing_aspects: list[str] = field(default_factory=list)
    # False when the tier-1 score gate decided without spending an LLM call
    # (ADR-010) — kept for eval/observability, not part of the decision itself.
    checked_by_llm: bool = False
