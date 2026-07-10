from dataclasses import dataclass

from grounded_rag.rerank.rerank import rerank
from grounded_rag.retrieval.records import RetrievedChunk


def make_chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        title="title",
        url="https://example.org",
        text=f"text {chunk_id}",
        doc_type="short",
        acl_tags=["public"],
        score=score,
    )


@dataclass
class FakeRerankResultItem:
    index: int
    relevance_score: float


@dataclass
class FakeRerankResponse:
    results: list[FakeRerankResultItem]


class FakeCohereClient:
    def __init__(self, response: FakeRerankResponse | None = None, error: Exception | None = None):
        self._response = response
        self._error = error

    def rerank(self, **kwargs):
        if self._error is not None:
            raise self._error
        return self._response


def test_rerank_reorders_chunks_by_cohere_relevance_score():
    chunks = [make_chunk("a", 0.9), make_chunk("b", 0.1), make_chunk("c", 0.5)]
    # Cohere puts "c" (index 2) first, then "a" (index 0) — the inverse of fusion order.
    fake_response = FakeRerankResponse(
        results=[FakeRerankResultItem(index=2, relevance_score=0.99), FakeRerankResultItem(index=0, relevance_score=0.4)]
    )
    result = rerank(FakeCohereClient(response=fake_response), "query", chunks, top_k=2)

    assert result.reranked is True
    assert [c.chunk_id for c in result.chunks] == ["c", "a"]
    assert result.chunks[0].score == 0.99


def test_rerank_falls_back_to_fusion_order_on_cohere_failure():
    chunks = [make_chunk("a", 0.9), make_chunk("b", 0.1), make_chunk("c", 0.5)]
    result = rerank(FakeCohereClient(error=RuntimeError("simulated Cohere outage")), "query", chunks, top_k=2)

    assert result.reranked is False
    assert result.chunks == chunks[:2]


def test_rerank_on_empty_candidate_set_short_circuits():
    result = rerank(FakeCohereClient(error=RuntimeError("should never be called")), "query", [], top_k=5)

    assert result.reranked is False
    assert result.chunks == []
