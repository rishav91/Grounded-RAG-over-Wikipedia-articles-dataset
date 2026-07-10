"""The chunk record returned by `retrieve`, carrying its fused/dense score."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str
    doc_type: str
    acl_tags: list[str]
    score: float
