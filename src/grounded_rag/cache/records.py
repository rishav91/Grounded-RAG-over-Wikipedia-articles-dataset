"""Records produced by `lookup_cache` (FR9; ADR-005)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CacheLookupResult:
    hit: bool
    answer: str | None = None
    citations: list[dict] = field(default_factory=list)
    confidence: float | None = None
