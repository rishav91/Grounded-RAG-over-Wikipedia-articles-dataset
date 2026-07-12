from langchain_core.messages import AIMessage

from grounded_rag.cache.records import CacheLookupResult
from grounded_rag.faithfulness.records import FaithfulnessResult
from grounded_rag.generation.generate import RETRIEVE_TOOL_NAME, SUBMIT_ANSWER_TOOL_NAME
from grounded_rag.generation.records import Citation
from grounded_rag.graph import nodes as nodes_module
from grounded_rag.graph.build import (
    _route_after_cache_lookup,
    _route_after_generate,
    _route_after_rerank,
    _route_after_sufficiency,
)
from grounded_rag.graph.deps import GraphDeps
from grounded_rag.graph.nodes import cache_lookup_node, check_sufficiency_node, response_node, retrieve_node
from grounded_rag.retrieval.records import RetrievedChunk
from grounded_rag.rewrite.records import RewriteResult
from grounded_rag.rewrite.rewrite import RewriteJudgment
from grounded_rag.sufficiency.records import SufficiencyResult
from grounded_rag.sufficiency.sufficiency import SufficiencyJudgment


def make_chunk(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        title="title",
        url="https://example.org",
        text="text",
        doc_type="short",
        acl_tags=["public"],
        score=score,
    )


def base_state(**overrides) -> dict:
    state = {
        "query": "q",
        "access_context_groups": ["public"],
        "doc_type": None,
        "date_range": None,
        "top_k": 5,
        "allow_generation": True,
        "bypass_cache": False,
        "cache_result": CacheLookupResult(hit=False),
        "rewrite": RewriteResult(rewritten_query="q", sub_queries=[]),
        "chunks": [make_chunk("c1")],
        "reranked": True,
        "sufficiency": SufficiencyResult(sufficient=True, confidence=0.9, reasoning="ok"),
        "messages": [],
        "tool_call_count": 0,
        "draft_answer": "the answer",
        "citations": [Citation(chunk_id="c1", claim="a claim")],
        "faithfulness": FaithfulnessResult(passed=True, confidence=0.9, reasoning="ok", answers_question=True),
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


class FakeEmbeddingsResponse:
    def __init__(self, vectors: list[list[float]]):
        self.data = [type("Item", (), {"embedding": v})() for v in vectors]


class FakeOpenAIClient:
    """Stands in for the OpenAI SDK client `embed_dense` calls (`.embeddings.create`)."""

    class _Embeddings:
        def create(self, model, input):  # noqa: A002 - matches OpenAI SDK's kwarg name
            return FakeEmbeddingsResponse([[0.1, 0.2, 0.3] for _ in input])

    def __init__(self):
        self.embeddings = self._Embeddings()


class FakeQdrantPoint:
    def __init__(self, score: float, payload: dict):
        self.score = score
        self.payload = payload


class FakeQdrantClient:
    """Records `upsert` calls (write-through) and returns a canned `query_points` result (lookup)."""

    def __init__(self, query_points: list[FakeQdrantPoint] | None = None):
        self.upserted_points = []
        self._query_points = query_points or []

    def upsert(self, collection_name, points):
        self.upserted_points.append((collection_name, points))

    def query_points(self, **kwargs):
        return type("Result", (), {"points": self._query_points})()


def make_deps(qdrant_client=None, openai_client=None) -> GraphDeps:
    return GraphDeps(
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        cohere_client=None,
        sparse_embedder=None,
        generation_llm=None,
        faithfulness_llm=None,
    )


def test_response_node_grounded_answer_filters_invalid_and_duplicate_citations():
    state = base_state(
        citations=[
            Citation(chunk_id="c1", claim="claim one"),
            Citation(chunk_id="c1", claim="claim one repeated"),  # duplicate chunk_id
            Citation(chunk_id="unknown-chunk", claim="dangling citation"),  # never retrieved
        ]
    )
    qdrant_client = FakeQdrantClient()
    result = response_node(make_deps(qdrant_client, FakeOpenAIClient()), state)["response"]

    assert result["abstained"] is False
    assert result["answer"] == "the answer"
    assert result["confidence"] == 0.9
    assert [c["chunk_id"] for c in result["citations"]] == ["c1"]
    assert result["cache_hit"] is False
    # FR-6.3: a passing response writes through to query_cache.
    assert len(qdrant_client.upserted_points) == 1


def test_response_node_serves_cache_hit_without_retrieval():
    cached = CacheLookupResult(
        hit=True,
        answer="cached answer",
        citations=[{"chunk_id": "c1", "doc_id": "d1", "title": "t", "url": "u"}],
        confidence=0.95,
    )
    result = response_node(None, base_state(cache_result=cached))["response"]

    assert result == {
        "answer": "cached answer",
        "abstained": False,
        "confidence": 0.95,
        "cache_hit": True,
        "citations": cached.citations,
        "retrieved_chunks": [],  # cache hit skips retrieve entirely
    }


def test_route_after_rerank():
    assert _route_after_rerank(base_state(allow_generation=True)) == "check_sufficiency"
    assert _route_after_rerank(base_state(allow_generation=False)) == "build_response"


def test_route_after_sufficiency():
    sufficient = SufficiencyResult(sufficient=True, confidence=0.9, reasoning="ok")
    insufficient = SufficiencyResult(sufficient=False, confidence=0.9, reasoning="not enough")
    assert _route_after_sufficiency(base_state(sufficiency=sufficient)) == "generate"
    assert _route_after_sufficiency(base_state(sufficiency=insufficient)) == "build_response"


def test_route_after_cache_lookup():
    assert _route_after_cache_lookup(base_state(cache_result=CacheLookupResult(hit=True))) == "build_response"
    assert _route_after_cache_lookup(base_state(cache_result=CacheLookupResult(hit=False))) == "rewrite_query"


def test_cache_lookup_node_skips_lookup_when_bypass_cache():
    result = cache_lookup_node(None, base_state(bypass_cache=True))
    assert result["cache_result"].hit is False


def test_cache_lookup_node_skips_lookup_when_generation_disallowed():
    result = cache_lookup_node(None, base_state(allow_generation=False))
    assert result["cache_result"].hit is False


def test_cache_lookup_node_hit_above_similarity_threshold():
    point = FakeQdrantPoint(score=0.95, payload={"answer": "cached answer", "citations": [], "confidence": 0.9})
    deps = make_deps(FakeQdrantClient(query_points=[point]), FakeOpenAIClient())

    result = cache_lookup_node(deps, base_state())

    assert result["cache_result"].hit is True
    assert result["cache_result"].answer == "cached answer"
    assert result["cache_result"].confidence == 0.9


def test_cache_lookup_node_miss_below_similarity_threshold():
    point = FakeQdrantPoint(score=0.5, payload={"answer": "cached answer", "citations": [], "confidence": 0.9})
    deps = make_deps(FakeQdrantClient(query_points=[point]), FakeOpenAIClient())

    result = cache_lookup_node(deps, base_state())

    assert result["cache_result"].hit is False


def test_cache_lookup_node_miss_when_no_points():
    deps = make_deps(FakeQdrantClient(query_points=[]), FakeOpenAIClient())

    result = cache_lookup_node(deps, base_state())

    assert result["cache_result"].hit is False


class FakeJudgeLLM:
    def __init__(self, judgment):
        self._judgment = judgment

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self._judgment


def test_check_sufficiency_node_reuses_faithfulness_llm():
    judgment = SufficiencyJudgment(sufficient=False, confidence=0.8, missing_aspects=["x"], reasoning="thin context")
    deps = GraphDeps(
        qdrant_client=None,
        openai_client=None,
        cohere_client=None,
        sparse_embedder=None,
        generation_llm=None,
        faithfulness_llm=FakeJudgeLLM(judgment),
    )
    # reranked=False skips the score-gate tiers entirely, forcing the LLM
    # judge — proves the node wires deps.faithfulness_llm through correctly.
    result = check_sufficiency_node(deps, base_state(reranked=False))

    assert result["sufficiency"].sufficient is False
    assert result["sufficiency"].checked_by_llm is True


def test_rewrite_query_node_uses_generation_llm():
    judgment = RewriteJudgment(rewritten_query="rewritten q", sub_queries=["sub1"])
    deps = GraphDeps(
        qdrant_client=None,
        openai_client=None,
        cohere_client=None,
        sparse_embedder=None,
        generation_llm=FakeJudgeLLM(judgment),
        faithfulness_llm=None,
    )

    result = nodes_module.rewrite_query_node(deps, base_state())

    assert result["rewrite"].rewritten_query == "rewritten q"
    assert result["rewrite"].sub_queries == ["sub1"]
    assert result["rewrite"].rewritten_by_llm is True


def test_retrieve_node_merges_primary_and_sub_query_chunks(monkeypatch):
    def fake_retrieve(qdrant_client, openai_client, sparse_embedder, query, access_context_groups, **kwargs):
        return {"primary": [make_chunk("a", 0.9)], "sub1": [make_chunk("b", 0.8)]}[query]

    monkeypatch.setattr(nodes_module, "retrieve", fake_retrieve)
    state = base_state(rewrite=RewriteResult(rewritten_query="primary", sub_queries=["sub1"]))

    result = retrieve_node(make_deps(), state)

    assert {c.chunk_id for c in result["chunks"]} == {"a", "b"}


def test_retrieve_node_dedups_overlapping_sub_query_chunks_keeping_higher_score(monkeypatch):
    def fake_retrieve(qdrant_client, openai_client, sparse_embedder, query, access_context_groups, **kwargs):
        return {"primary": [make_chunk("a", 0.5)], "sub1": [make_chunk("a", 0.9)]}[query]

    monkeypatch.setattr(nodes_module, "retrieve", fake_retrieve)
    state = base_state(rewrite=RewriteResult(rewritten_query="primary", sub_queries=["sub1"]))

    result = retrieve_node(make_deps(), state)

    assert len(result["chunks"]) == 1
    assert result["chunks"][0].score == 0.9


def test_retrieve_node_degrades_when_a_sub_query_retrieval_fails(monkeypatch):
    def fake_retrieve(qdrant_client, openai_client, sparse_embedder, query, access_context_groups, **kwargs):
        if query == "sub-fail":
            raise RuntimeError("simulated retrieval outage")
        return {"primary": [make_chunk("a", 0.9)], "sub-ok": [make_chunk("b", 0.8)]}[query]

    monkeypatch.setattr(nodes_module, "retrieve", fake_retrieve)
    state = base_state(rewrite=RewriteResult(rewritten_query="primary", sub_queries=["sub-ok", "sub-fail"]))

    result = retrieve_node(make_deps(), state)

    assert {c.chunk_id for c in result["chunks"]} == {"a", "b"}


def test_retrieve_node_skips_fan_out_when_no_sub_queries(monkeypatch):
    def fake_retrieve(qdrant_client, openai_client, sparse_embedder, query, access_context_groups, **kwargs):
        return [make_chunk("a", 0.9)]

    monkeypatch.setattr(nodes_module, "retrieve", fake_retrieve)
    state = base_state(rewrite=RewriteResult(rewritten_query="primary", sub_queries=[]))

    result = retrieve_node(make_deps(), state)

    assert [c.chunk_id for c in result["chunks"]] == ["a"]


class FakeRetrieveTool:
    """Stands in for `build_retrieve_tool`'s return value — `invoke(args)`
    either returns chunks or raises, keyed on the query text."""

    def invoke(self, args):
        if args["query"] == "fail":
            raise RuntimeError("simulated retrieval outage")
        return [make_chunk(f"c-{args['query']}", 0.9)]


def test_execute_tool_node_runs_concurrent_calls_and_degrades_on_partial_failure(monkeypatch):
    monkeypatch.setattr(nodes_module, "build_retrieve_tool", lambda deps, state: FakeRetrieveTool())
    state = base_state(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": RETRIEVE_TOOL_NAME, "args": {"query": "ok"}, "id": "call_a"},
                    {"name": RETRIEVE_TOOL_NAME, "args": {"query": "fail"}, "id": "call_b"},
                ],
            )
        ],
        tool_call_count=0,
    )

    result = nodes_module.execute_tool_node(make_deps(), state)

    assert result["tool_call_count"] == 1
    # "c1" was already in state["chunks"] (base_state); "c-ok" is the new one
    # the surviving call fetched.
    assert {c.chunk_id for c in result["chunks"]} == {"c1", "c-ok"}
    tool_messages = {m.tool_call_id: m.content for m in result["messages"]}
    assert set(tool_messages) == {"call_a", "call_b"}
    assert "simulated retrieval outage" in tool_messages["call_b"]


def test_execute_tool_node_skips_calls_beyond_the_parallel_cap(monkeypatch):
    monkeypatch.setattr(nodes_module, "build_retrieve_tool", lambda deps, state: FakeRetrieveTool())
    calls = [{"name": RETRIEVE_TOOL_NAME, "args": {"query": f"q{i}"}, "id": f"call_{i}"} for i in range(5)]
    state = base_state(messages=[AIMessage(content="", tool_calls=calls)], tool_call_count=0)

    result = nodes_module.execute_tool_node(make_deps(), state)

    tool_messages = {m.tool_call_id: m.content for m in result["messages"]}
    assert len(tool_messages) == 5  # every call still gets a ToolMessage (MAX_PARALLEL_RETRIEVE_CALLS=3)
    assert "Skipped" in tool_messages["call_3"]
    assert "Skipped" in tool_messages["call_4"]


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
