import tiktoken

from grounded_rag.config import EMBEDDING_MODEL
from grounded_rag.ingestion.chunker import chunk_text

_encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)


def _tokens(text: str) -> int:
    return len(_encoding.encode(text))


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_is_a_single_chunk():
    text = "This is one sentence. This is another."
    chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
    assert len(chunks) == 1
    assert "This is one sentence." in chunks[0]
    assert "This is another." in chunks[0]


def test_packing_respects_sentence_boundaries_and_overlaps():
    # Each sentence is 11 tokens; max_tokens=30 fits two per chunk, and
    # overlap_tokens=15 is enough to carry exactly one trailing sentence
    # (but not two) into the next chunk.
    sentences = [f"Sentence number {i} has a few words in it." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunk_text(text, max_tokens=30, overlap_tokens=15)

    assert len(chunks) > 1
    # No chunk exceeds the budget (sentences are short enough that the
    # oversized-sentence fallback never kicks in here).
    for chunk in chunks:
        assert _tokens(chunk) <= 30
    # No sentence got split mid-way -- every original sentence appears
    # intact in at least one chunk.
    for sentence in sentences:
        assert any(sentence in chunk for chunk in chunks)
    # Consecutive chunks overlap: the last sentence of chunk i reappears
    # at the start of chunk i+1.
    for i in range(len(chunks) - 1):
        last_sentence_of_chunk = chunks[i].split(".")[-2].strip() + "."
        assert chunks[i + 1].startswith(last_sentence_of_chunk)


def test_oversized_single_sentence_is_hard_split():
    # No punctuation, so pysbd treats this as one sentence far longer than
    # the token budget.
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert _tokens(chunk) <= 50
