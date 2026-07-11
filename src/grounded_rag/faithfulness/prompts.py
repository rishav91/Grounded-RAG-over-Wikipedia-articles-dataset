"""Structured-rubric judge prompt for `judge_faithfulness` (ADR-006).

Scores each cited claim against the specific chunk it's attributed to, not
the answer as a whole — a claim citing the wrong (even if topically related)
chunk must fail, since citation *support* (not just citation *presence*) is
the judge's job.
"""

from __future__ import annotations

from grounded_rag.generation.records import Citation
from grounded_rag.retrieval.records import RetrievedChunk

SYSTEM_PROMPT = """You are a strict faithfulness judge for a grounded \
question-answering system. You will be given a question, a drafted answer, \
and the specific claim/chunk citation pairs the answer relies on.

For each citation, decide whether the cited chunk's text actually supports \
the claim it's attached to — not whether the chunk is topically related, but \
whether it specifically entails or states the claim. Mark `supported: false` \
for any claim the chunk doesn't actually back, including claims that go \
beyond what the chunk says.

Then give an overall verdict:
- `passed`: true only if every claim is supported by its citation.
- `confidence`: your overall confidence (0.0-1.0) that the answer as a whole \
is fully grounded in the cited chunks. This is a qualitative judgment, not a \
calibrated statistic.
- `reasoning`: a brief explanation, especially for any unsupported claim."""


def format_claims(citations: list[Citation], chunks_by_id: dict[str, RetrievedChunk]) -> str:
    blocks = []
    for citation in citations:
        chunk = chunks_by_id.get(citation.chunk_id)
        chunk_text = chunk.text if chunk is not None else "(chunk not found)"
        blocks.append(f'Claim: "{citation.claim}"\nCited chunk [{citation.chunk_id}]: {chunk_text}')
    return "\n\n".join(blocks)


def build_judge_prompt(query: str, answer: str, citations: list[Citation], chunks_by_id: dict[str, RetrievedChunk]) -> str:
    return f"Question: {query}\nAnswer: {answer}\n\n{format_claims(citations, chunks_by_id)}"
