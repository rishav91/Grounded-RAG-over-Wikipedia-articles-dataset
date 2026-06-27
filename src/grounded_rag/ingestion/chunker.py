"""Sentence-aware chunker.

DATA-MODEL.md#source-canonical-mapping: ~500 tokens per chunk, 50 tokens of
overlap, sentence-boundary aware. A starting default, not a tuned value —
see PRD.md §12.
"""

from __future__ import annotations

import pysbd
import tiktoken

from grounded_rag.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, EMBEDDING_MODEL

_segmenter = pysbd.Segmenter(language="en", clean=False)
_encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _segmenter.segment(text) if s.strip()]


def _token_count(text: str) -> int:
    return len(_encoding.encode(text))


def _split_oversized_sentence(sentence: str, max_tokens: int) -> list[str]:
    """Hard-split a single sentence longer than max_tokens, by token boundary."""
    tokens = _encoding.encode(sentence)
    pieces = []
    for start in range(0, len(tokens), max_tokens):
        pieces.append(_encoding.decode(tokens[start : start + max_tokens]))
    return pieces


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Pack sentences into ~max_tokens chunks, sentence-boundary aware,
    with overlap_tokens of trailing context carried into the next chunk.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens: list[int] = []

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))

    for sentence in sentences:
        tok_count = _token_count(sentence)

        if tok_count > max_tokens:
            flush()
            current, current_tokens = [], []
            chunks.extend(_split_oversized_sentence(sentence, max_tokens))
            continue

        if current and sum(current_tokens) + tok_count > max_tokens:
            flush()
            # Carry trailing sentences worth ~overlap_tokens into the next chunk.
            overlap_sentences: list[str] = []
            overlap_token_counts: list[int] = []
            overlap_total = 0
            for s, st in zip(reversed(current), reversed(current_tokens)):
                if overlap_total + st > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_token_counts.insert(0, st)
                overlap_total += st
            current, current_tokens = overlap_sentences, overlap_token_counts

        current.append(sentence)
        current_tokens.append(tok_count)

    flush()
    return chunks
