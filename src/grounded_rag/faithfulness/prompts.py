"""Structured-rubric judge prompt for `judge_faithfulness` (ADR-006).

Scores each cited claim against the specific chunk it's attributed to, not
the answer as a whole — a claim citing the wrong (even if topically related)
chunk must fail, since citation *support* (not just citation *presence*) is
the judge's job.
"""

from __future__ import annotations

from grounded_rag.generation.records import Citation
from grounded_rag.retrieval.records import RetrievedChunk

SYSTEM_PROMPT = """You are a strict judge for a grounded question-answering \
system, scoring two independent things: whether the answer is faithful to \
its citations, and whether it actually answers the question. An answer can \
fail either one on its own — a fully-cited answer to the wrong question is \
still a failure, and a directly-responsive answer with an unsupported claim \
is still a failure.

You will be given a question, a drafted answer, and the specific \
claim/chunk citation pairs the answer relies on.

Faithfulness — for each citation, decide whether the cited chunk's text \
actually supports the claim it's attached to — not whether the chunk is \
topically related, but whether it specifically entails or states the claim. \
Mark `supported: false` for any claim the chunk doesn't actually back, \
including claims that go beyond what the chunk says.

Relevance — separately, decide whether the answer as a whole actually \
addresses what the question asked. An answer that is fully grounded but \
responds to a narrower, tangential, or different question than the one \
asked is not relevant, even if every claim in it is supported.

Then give an overall verdict:
- `passed`: true only if every claim is supported by its citation.
- `answers_question`: true only if the answer actually addresses the \
question asked, independent of `passed`.
- `confidence`: your overall confidence (0.0-1.0) that the answer as a whole \
is fully grounded in the cited chunks. This is a qualitative judgment, not a \
calibrated statistic.
- `reasoning`: a brief explanation, covering both faithfulness and \
relevance, especially for any unsupported claim or off-target answer."""


def format_claims(citations: list[Citation], chunks_by_id: dict[str, RetrievedChunk]) -> str:
    blocks = []
    for citation in citations:
        chunk = chunks_by_id.get(citation.chunk_id)
        chunk_text = chunk.text if chunk is not None else "(chunk not found)"
        blocks.append(f'Claim: "{citation.claim}"\nCited chunk [{citation.chunk_id}]: {chunk_text}')
    return "\n\n".join(blocks)


def build_judge_prompt(query: str, answer: str, citations: list[Citation], chunks_by_id: dict[str, RetrievedChunk]) -> str:
    return f"Question: {query}\nAnswer: {answer}\n\n{format_claims(citations, chunks_by_id)}"
