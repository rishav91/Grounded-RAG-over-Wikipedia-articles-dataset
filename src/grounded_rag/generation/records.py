"""Records produced by `generate` — one LLM round's outcome (FR5, FR8)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    claim: str


@dataclass(frozen=True)
class ToolCallRequest:
    """One `retrieve_chunks` call requested in a generation round (FR12;
    ADR-012) — `call_id` is the tool_call_id the resulting ToolMessage must
    answer, since a round may request more than one call at once."""

    call_id: str
    query: str
    top_k: int


@dataclass(frozen=True)
class GenerationResult:
    """One round's outcome: either a finished answer, or one or more requests
    to re-retrieve (FR12: possibly executed concurrently).

    `finished=False` with `tool_calls=[]` means the model didn't call a
    recognized tool at all (shouldn't happen under `tool_choice="required"`,
    but the graph must still degrade to abstain rather than crash on it).
    """

    answer: str | None
    citations: list[Citation]
    tool_calls: list[ToolCallRequest]
    finished: bool
