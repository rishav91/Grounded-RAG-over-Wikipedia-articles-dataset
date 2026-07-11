from grounded_rag.rewrite.rewrite import RewriteJudgment, rewrite_query


class FakeRewriteLLM:
    def __init__(self, judgment: RewriteJudgment | None = None, error: Exception | None = None):
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


def test_rewrite_query_returns_rewritten_query_and_sub_queries():
    judgment = RewriteJudgment(
        rewritten_query="What is Article-A known for, and what is Article-B known for?",
        sub_queries=["What is Article-A known for?", "What is Article-B known for?"],
    )
    llm = FakeRewriteLLM(judgment)

    result = rewrite_query(llm, "What is A and B known for?")

    assert llm.invoked is True
    assert result.rewritten_query == judgment.rewritten_query
    assert result.sub_queries == judgment.sub_queries
    assert result.rewritten_by_llm is True


def test_rewrite_query_caps_sub_queries_at_the_configured_max():
    judgment = RewriteJudgment(rewritten_query="q", sub_queries=["a", "b", "c", "d", "e"])
    llm = FakeRewriteLLM(judgment)

    result = rewrite_query(llm, "q")

    assert len(result.sub_queries) == 3  # QUERY_REWRITE_MAX_SUB_QUERIES


def test_rewrite_query_fails_open_on_llm_error():
    llm = FakeRewriteLLM(error=RuntimeError("simulated rewrite-LLM outage"))

    result = rewrite_query(llm, "original query")

    assert result.rewritten_query == "original query"
    assert result.sub_queries == []
    assert result.rewritten_by_llm is False
