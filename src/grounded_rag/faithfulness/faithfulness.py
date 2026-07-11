"""`judge_faithfulness`: LLM-as-judge scoring each cited claim against its
cited chunk, gating the abstain decision (FR6, FR7; ADR-006). Also scores
answer relevance (FR-5.4): whether the answer actually addresses the
question, independent of whether its citations check out — a fully-cited
answer to the wrong question must still fail.

Zero citations short-circuits without an LLM call — there is nothing to
verify, so the answer cannot be faithful by construction. This is the
trivial case of the deterministic+LLM hybrid ADR-006 defers in full (a
citation-*presence* check before spending an LLM call on citation-*support*),
not a reimplementation of that deferred design.

A plain function, not a LangGraph node — mirrors `generate.py`/`rerank.py`.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from grounded_rag.config import FAITHFULNESS_CONFIDENCE_THRESHOLD
from grounded_rag.faithfulness.prompts import SYSTEM_PROMPT, build_judge_prompt
from grounded_rag.faithfulness.records import ClaimCheck, FaithfulnessResult
from grounded_rag.generation.records import Citation
from grounded_rag.retrieval.records import RetrievedChunk


class ClaimCheckSchema(BaseModel):
    claim: str
    chunk_id: str
    supported: bool


class FaithfulnessJudgment(BaseModel):
    claim_checks: list[ClaimCheckSchema]
    passed: bool
    answers_question: bool
    confidence: float
    reasoning: str


def judge_faithfulness(
    llm: BaseChatModel,
    query: str,
    answer: str | None,
    citations: list[Citation],
    chunks_by_id: dict[str, RetrievedChunk],
) -> FaithfulnessResult:
    if not citations or not answer:
        return FaithfulnessResult(
            passed=False,
            confidence=0.0,
            reasoning="No citations to verify.",
            answers_question=False,
            claim_checks=[],
        )

    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_judge_prompt(query, answer, citations, chunks_by_id))]
    judgment = llm.with_structured_output(FaithfulnessJudgment).invoke(messages)

    # FR-5.4: relevance and faithfulness are independent gates — either one
    # failing converts the response into an abstention (FR-5.2).
    passed = judgment.passed and judgment.answers_question and judgment.confidence >= FAITHFULNESS_CONFIDENCE_THRESHOLD
    claim_checks = [
        ClaimCheck(claim=c.claim, chunk_id=c.chunk_id, supported=c.supported) for c in judgment.claim_checks
    ]
    return FaithfulnessResult(
        passed=passed,
        confidence=judgment.confidence,
        reasoning=judgment.reasoning,
        answers_question=judgment.answers_question,
        claim_checks=claim_checks,
    )
