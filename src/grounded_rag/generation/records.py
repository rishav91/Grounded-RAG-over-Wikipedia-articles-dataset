"""Records produced by `generate` — one LLM round's outcome (FR5, FR8)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    claim: str


@dataclass(frozen=True)
class GenerationResult:
    """One round's outcome: either a finished answer, or a request to re-retrieve.

    `finished=False` with `tool_query=None` means the model didn't call a
    recognized tool at all (shouldn't happen under `tool_choice="required"`,
    but the graph must still degrade to abstain rather than crash on it).
    """

    answer: str | None
    citations: list[Citation]
    tool_query: str | None
    tool_top_k: int | None
    finished: bool
