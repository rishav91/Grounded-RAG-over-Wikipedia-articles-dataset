from grounded_rag.retrieval.records import RetrievedChunk
from grounded_rag.sufficiency.sufficiency import SufficiencyJudgment, check_sufficiency


def make_chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="doc-1",
        title="title",
        url="https://example.org",
        text="text",
        doc_type="short",
        acl_tags=["public"],
        score=score,
    )


class FakeJudgeLLM:
    def __init__(self, judgment: SufficiencyJudgment | None = None, error: Exception | None = None):
        self._judgment = judgment
        self._error = error
        self.invoked = False

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.invoked = True
        if self._error is not None:
            raise self._error
        return self._judgment


def test_check_sufficiency_no_chunks_short_circuits():
    llm = FakeJudgeLLM()

    result = check_sufficiency(llm, "query", chunks=[], reranked=True)

    assert result.sufficient is False
    assert llm.invoked is False


def test_check_sufficiency_low_score_short_circuits_when_reranked():
    llm = FakeJudgeLLM()  # would raise if actually invoked

    result = check_sufficiency(llm, "query", chunks=[make_chunk(0.05)], reranked=True)

    assert result.sufficient is False
    assert result.checked_by_llm is False
    assert llm.invoked is False


def test_check_sufficiency_high_score_short_circuits_when_reranked():
    llm = FakeJudgeLLM()

    result = check_sufficiency(llm, "query", chunks=[make_chunk(0.95)], reranked=True)

    assert result.sufficient is True
    assert result.checked_by_llm is False
    assert llm.invoked is False


def test_check_sufficiency_ambiguous_score_calls_llm_judge():
    judgment = SufficiencyJudgment(
        sufficient=True, confidence=0.8, missing_aspects=[], reasoning="covers the question"
    )
    llm = FakeJudgeLLM(judgment)

    result = check_sufficiency(llm, "query", chunks=[make_chunk(0.4)], reranked=True)

    assert llm.invoked is True
    assert result.sufficient is True
    assert result.checked_by_llm is True


def test_check_sufficiency_skips_score_gate_when_not_reranked():
    # Fusion-only fallback scores (FR-3.2) aren't on Cohere's 0-1 scale, so
    # even a "high" fusion score must still go to the LLM judge.
    judgment = SufficiencyJudgment(sufficient=False, confidence=0.6, missing_aspects=["x"], reasoning="incomplete")
    llm = FakeJudgeLLM(judgment)

    result = check_sufficiency(llm, "query", chunks=[make_chunk(0.99)], reranked=False)

    assert llm.invoked is True
    assert result.sufficient is False


def test_check_sufficiency_fails_open_on_llm_error():
    llm = FakeJudgeLLM(error=RuntimeError("simulated LLM outage"))

    result = check_sufficiency(llm, "query", chunks=[make_chunk(0.4)], reranked=True)

    assert result.sufficient is True  # fail open — faithfulness is still the safety gate downstream
    assert result.checked_by_llm is False
