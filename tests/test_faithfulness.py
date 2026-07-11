from grounded_rag.faithfulness.faithfulness import FaithfulnessJudgment, judge_faithfulness
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
    judgment = FaithfulnessJudgment(claim_checks=[], passed=True, confidence=0.6, reasoning="mostly supported")
    llm = FakeJudgeLLM(judgment)

    result = judge_faithfulness(llm, "query", "answer", citations=[Citation(chunk_id="c1", claim="x")], chunks_by_id={})

    assert result.passed is False  # judge said passed, but confidence is below FAITHFULNESS_CONFIDENCE_THRESHOLD
    assert result.confidence == 0.6


def test_judge_faithfulness_passes_above_threshold():
    judgment = FaithfulnessJudgment(claim_checks=[], passed=True, confidence=0.9, reasoning="fully supported")
    llm = FakeJudgeLLM(judgment)

    result = judge_faithfulness(llm, "query", "answer", citations=[Citation(chunk_id="c1", claim="x")], chunks_by_id={})

    assert result.passed is True
    assert result.confidence == 0.9


def test_judge_faithfulness_fails_when_judge_says_unsupported_even_with_high_confidence():
    judgment = FaithfulnessJudgment(claim_checks=[], passed=False, confidence=0.9, reasoning="claim not supported")
    llm = FakeJudgeLLM(judgment)

    result = judge_faithfulness(llm, "query", "answer", citations=[Citation(chunk_id="c1", claim="x")], chunks_by_id={})

    assert result.passed is False
