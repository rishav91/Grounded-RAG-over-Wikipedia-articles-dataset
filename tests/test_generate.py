from langchain_core.messages import AIMessage

from grounded_rag.generation.generate import RETRIEVE_TOOL_NAME, SUBMIT_ANSWER_TOOL_NAME, generate


class FakeToolCallingLLM:
    def __init__(self, response: AIMessage):
        self._response = response
        self.bound_tool_names: list[str] | None = None

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        self.bound_tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
        return self

    def invoke(self, messages):
        return self._response


def test_generate_parses_submit_answer_tool_call():
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": SUBMIT_ANSWER_TOOL_NAME,
                "args": {
                    "answer": "Adam Smith is known for The Wealth of Nations.",
                    "citations": [{"chunk_id": "c1", "claim": "Adam Smith wrote The Wealth of Nations."}],
                },
                "id": "call_1",
            }
        ],
    )
    llm = FakeToolCallingLLM(response)

    _, result = generate(llm, messages=[], tools=[])

    assert result.finished is True
    assert result.answer == "Adam Smith is known for The Wealth of Nations."
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"


def test_generate_parses_retrieve_chunks_tool_call():
    response = AIMessage(
        content="",
        tool_calls=[{"name": RETRIEVE_TOOL_NAME, "args": {"query": "who succeeded them", "top_k": 5}, "id": "call_2"}],
    )
    llm = FakeToolCallingLLM(response)

    _, result = generate(llm, messages=[], tools=[])

    assert result.finished is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].query == "who succeeded them"
    assert result.tool_calls[0].top_k == 5
    assert result.tool_calls[0].call_id == "call_2"


def test_generate_parses_multiple_parallel_retrieve_chunks_calls():
    response = AIMessage(
        content="",
        tool_calls=[
            {"name": RETRIEVE_TOOL_NAME, "args": {"query": "fact one"}, "id": "call_a"},
            {"name": RETRIEVE_TOOL_NAME, "args": {"query": "fact two"}, "id": "call_b"},
        ],
    )
    llm = FakeToolCallingLLM(response)

    _, result = generate(llm, messages=[], tools=[])

    assert result.finished is False
    assert [c.query for c in result.tool_calls] == ["fact one", "fact two"]
    assert [c.call_id for c in result.tool_calls] == ["call_a", "call_b"]


def test_generate_submit_answer_wins_over_concurrent_retrieve_calls():
    response = AIMessage(
        content="",
        tool_calls=[
            {"name": RETRIEVE_TOOL_NAME, "args": {"query": "fact one"}, "id": "call_a"},
            {
                "name": SUBMIT_ANSWER_TOOL_NAME,
                "args": {"answer": "the answer", "citations": []},
                "id": "call_b",
            },
        ],
    )
    llm = FakeToolCallingLLM(response)

    _, result = generate(llm, messages=[], tools=[])

    assert result.finished is True
    assert result.answer == "the answer"
    assert result.tool_calls == []


def test_generate_with_no_tool_calls_degrades_safely():
    response = AIMessage(content="some stray prose", tool_calls=[])
    llm = FakeToolCallingLLM(response)

    _, result = generate(llm, messages=[], tools=[])

    assert result.finished is False
    assert result.answer is None
    assert result.tool_calls == []
