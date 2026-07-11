"""`check_sufficiency`: gates whether retrieved context is adequate to
attempt an answer, independent of the generator's own self-assessment
(FR15; ADR-010).

Three-tier design, cheapest first — mirrors this codebase's existing
"deterministic where possible, LLM only where genuinely ambiguous" pattern
(`rerank.py`'s FR-3.2 fallback, `judge_faithfulness`'s zero-citation
short-circuit):

  1. No chunks at all -> insufficient, no LLM call.
  2. Reranked, and the top chunk's relevance score is below
     `SUFFICIENCY_LOW_SCORE_THRESHOLD` -> insufficient, no LLM call
     (obviously hopeless). At or above `SUFFICIENCY_HIGH_SCORE_THRESHOLD` ->
     sufficient, no LLM call (obviously fine).
  3. Otherwise (genuinely ambiguous, or reranking degraded to the fusion-only
     fallback whose scores aren't on Cohere's 0-1 scale) -> one LLM call.

A plain function, not a LangGraph node — mirrors `generate.py`/`rerank.py`/
`faithfulness.py`.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from grounded_rag.config import SUFFICIENCY_HIGH_SCORE_THRESHOLD, SUFFICIENCY_LOW_SCORE_THRESHOLD
from grounded_rag.retrieval.records import RetrievedChunk
from grounded_rag.sufficiency.prompts import SYSTEM_PROMPT, build_sufficiency_prompt
from grounded_rag.sufficiency.records import SufficiencyResult


class SufficiencyJudgment(BaseModel):
    sufficient: bool
    confidence: float
    missing_aspects: list[str]
    reasoning: str


def check_sufficiency(
    llm: BaseChatModel,
    query: str,
    chunks: list[RetrievedChunk],
    reranked: bool,
) -> SufficiencyResult:
    if not chunks:
        return SufficiencyResult(
            sufficient=False,
            confidence=1.0,
            reasoning="No chunks were retrieved.",
            missing_aspects=[query],
        )

    if reranked:
        max_score = max(chunk.score for chunk in chunks)
        if max_score < SUFFICIENCY_LOW_SCORE_THRESHOLD:
            return SufficiencyResult(
                sufficient=False,
                confidence=1.0 - max_score,
                reasoning=f"Highest retrieved relevance ({max_score:.2f}) is below the low-score threshold.",
                missing_aspects=[query],
            )
        if max_score >= SUFFICIENCY_HIGH_SCORE_THRESHOLD:
            return SufficiencyResult(
                sufficient=True,
                confidence=max_score,
                reasoning=f"Highest retrieved relevance ({max_score:.2f}) is above the high-score threshold.",
            )

    try:
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_sufficiency_prompt(query, chunks))]
        judgment = llm.with_structured_output(SufficiencyJudgment).invoke(messages)
    except Exception:
        # Fail open, not closed: a transient judge failure shouldn't block a
        # request that might otherwise succeed — faithfulness remains the
        # safety-critical gate downstream. Mirrors FR-3.2's Cohere-failure
        # fallback (degrade, don't fail the request).
        return SufficiencyResult(
            sufficient=True,
            confidence=0.0,
            reasoning="Sufficiency check failed; defaulting to proceed to generation.",
        )

    return SufficiencyResult(
        sufficient=judgment.sufficient,
        confidence=judgment.confidence,
        reasoning=judgment.reasoning,
        missing_aspects=judgment.missing_aspects,
        checked_by_llm=True,
    )
