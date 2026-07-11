from grounded_rag.faithfulness.faithfulness import ClaimCheckSchema, FaithfulnessJudgment, judge_faithfulness
from grounded_rag.generation.records import Citation


class FakeJudgeLLM:
    def __init__(self, judgment: FaithfulnessJudgment | None = None):
        self._judgment = judgment
        self.invoked = False

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.invoked = True
        return self._judgment


def test_judge_faithfulness_short_circuits_on_no_citations():
    llm = FakeJudgeLLM()  # would raise/return None if actually invoked

    result = judge_faithfulness(llm, "query", "answer", citations=[], chunks_by_id={})

    assert result.passed is False
    assert result.confidence == 0.0
    assert llm.invoked is False


def test_judge_faithfulness_short_circuits_on_no_answer():
    llm = FakeJudgeLLM()

    result = judge_faithfulness(llm, "query", None, citations=[Citation(chunk_id="c1", claim="x")], chunks_by_id={})

    assert result.passed is False
    assert llm.invoked is False


def test_judge_faithfulness_fails_below_confidence_threshold():
    judgment = FaithfulnessJudgment(
        claim_checks=[], passed=True, answers_question=True, confidence=0.6, reasoning="mostly supported"
    )
    llm = FakeJudgeLLM(judgment)

    result = judge_faithfulness(llm, "query", "answer", citations=[Citation(chunk_id="c1", claim="x")], chunks_by_id={})

    assert result.passed is False  # judge said passed, but confidence is below FAITHFULNESS_CONFIDENCE_THRESHOLD
    assert result.confidence == 0.6


def test_judge_faithfulness_passes_above_threshold():
    judgment = FaithfulnessJudgment(
        claim_checks=[], passed=True, answers_question=True, confidence=0.9, reasoning="fully supported"
    )
    llm = FakeJudgeLLM(judgment)

    result = judge_faithfulness(llm, "query", "answer", citations=[Citation(chunk_id="c1", claim="x")], chunks_by_id={})

    assert result.passed is True
    assert result.confidence == 0.9


def test_judge_faithfulness_fails_when_judge_says_unsupported_even_with_high_confidence():
    judgment = FaithfulnessJudgment(
        claim_checks=[], passed=False, answers_question=True, confidence=0.9, reasoning="claim not supported"
    )
    llm = FakeJudgeLLM(judgment)

    result = judge_faithfulness(llm, "query", "answer", citations=[Citation(chunk_id="c1", claim="x")], chunks_by_id={})

    assert result.passed is False


def test_judge_faithfulness_fails_when_answer_is_faithful_but_off_topic():
    # FR-5.4: every citation is genuinely supported, but the answer doesn't
    # address the question that was asked — passed must still be False.
    judgment = FaithfulnessJudgment(
        claim_checks=[ClaimCheckSchema(claim="x", chunk_id="c1", supported=True)],
        passed=True,
        answers_question=False,
        confidence=0.95,
        reasoning="claim is supported but answers a different question than the one asked",
    )
    llm = FakeJudgeLLM(judgment)

    result = judge_faithfulness(llm, "query", "answer", citations=[Citation(chunk_id="c1", claim="x")], chunks_by_id={})

    assert result.passed is False
    assert result.answers_question is False
