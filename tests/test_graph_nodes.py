from langchain_core.messages import AIMessage

from grounded_rag.faithfulness.records import FaithfulnessResult
from grounded_rag.generation.generate import RETRIEVE_TOOL_NAME, SUBMIT_ANSWER_TOOL_NAME
from grounded_rag.generation.records import Citation
from grounded_rag.graph.build import _route_after_generate, _route_after_rerank
from grounded_rag.graph.nodes import response_node
from grounded_rag.retrieval.records import RetrievedChunk


def make_chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        title="title",
        url="https://example.org",
        text="text",
        doc_type="short",
        acl_tags=["public"],
        score=0.5,
    )


def base_state(**overrides) -> dict:
    state = {
        "query": "q",
        "access_context_groups": ["public"],
        "doc_type": None,
        "date_range": None,
        "top_k": 5,
        "allow_generation": True,
        "chunks": [make_chunk("c1")],
        "messages": [],
        "tool_call_count": 0,
        "draft_answer": "the answer",
        "citations": [Citation(chunk_id="c1", claim="a claim")],
        "faithfulness": FaithfulnessResult(passed=True, confidence=0.9, reasoning="ok"),
        "response": {},
    }
    state.update(overrides)
    return state


def test_response_node_allow_generation_false_skips_generation():
    result = response_node(None, base_state(allow_generation=False))["response"]

    assert result["answer"] is None
    assert result["abstained"] is False
    assert result["confidence"] is None
    assert result["citations"] == []
    assert len(result["retrieved_chunks"]) == 1


def test_response_node_abstains_on_faithfulness_fail():
    failing = FaithfulnessResult(passed=False, confidence=0.3, reasoning="weak")
    result = response_node(None, base_state(faithfulness=failing))["response"]

    assert result["answer"] is None
    assert result["abstained"] is True
    assert result["confidence"] == 0.3
    assert result["citations"] == []
    assert len(result["retrieved_chunks"]) == 1  # still populated per API-CONTRACTS.md


def test_response_node_grounded_answer_filters_invalid_and_duplicate_citations():
    state = base_state(
        citations=[
            Citation(chunk_id="c1", claim="claim one"),
            Citation(chunk_id="c1", claim="claim one repeated"),  # duplicate chunk_id
            Citation(chunk_id="unknown-chunk", claim="dangling citation"),  # never retrieved
        ]
    )
    result = response_node(None, state)["response"]

    assert result["abstained"] is False
    assert result["answer"] == "the answer"
    assert result["confidence"] == 0.9
    assert [c["chunk_id"] for c in result["citations"]] == ["c1"]


def test_route_after_rerank():
    assert _route_after_rerank(base_state(allow_generation=True)) == "generate"
    assert _route_after_rerank(base_state(allow_generation=False)) == "build_response"


def test_route_after_generate_retrieve_call_goes_to_execute_tool():
    state = base_state(
        messages=[AIMessage(content="", tool_calls=[{"name": RETRIEVE_TOOL_NAME, "args": {"query": "q2"}, "id": "1"}])]
    )
    assert _route_after_generate(state) == "execute_tool"


def test_route_after_generate_submit_answer_goes_to_faithfulness():
    state = base_state(
        messages=[
            AIMessage(content="", tool_calls=[{"name": SUBMIT_ANSWER_TOOL_NAME, "args": {"answer": "x", "citations": []}, "id": "1"}])
        ]
    )
    assert _route_after_generate(state) == "judge_faithfulness"
