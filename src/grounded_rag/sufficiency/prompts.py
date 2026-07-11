"""Prompt for `check_sufficiency`'s tier-2 LLM judge (FR15; ADR-010).

Context-only: the judge never sees a drafted answer, only the question and
the retrieved chunks — it's answering "could a well-informed answer be built
from only this," not "is some specific answer supported."
"""

from __future__ import annotations

from grounded_rag.generation.prompts import format_context
from grounded_rag.retrieval.records import RetrievedChunk

SYSTEM_PROMPT = """You judge whether a set of retrieved document chunks \
contains enough information to answer a question — nothing else. You are \
not answering the question and no drafted answer is provided; you are only \
judging the chunks' adequacy as evidence.

The question may have multiple parts. Sufficiency requires that EVERY part \
be addressable from the chunks, not just the most prominent one — a chunk \
that fully covers one part of a two-part question is still insufficient \
overall if the other part isn't covered anywhere in the given chunks.

Give:
- `sufficient`: true only if a well-informed answer to the full question \
could be constructed using only the given chunks.
- `confidence`: your confidence (0.0-1.0) in that judgment.
- `missing_aspects`: a list of the specific sub-questions or facts the \
question asks for that the given chunks do not cover. Empty if sufficient.
- `reasoning`: a brief explanation."""


def build_sufficiency_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    return f"{format_context(chunks)}\n\nQuestion: {query}"
