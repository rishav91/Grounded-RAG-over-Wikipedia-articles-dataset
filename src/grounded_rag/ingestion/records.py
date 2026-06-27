"""The flat chunk record that flows from chunking through embedding to upsert."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    chunk_index: int
    text: str
    title: str
    url: str
    doc_type: str
    acl_tags: list[str]
    created_at: datetime
    updated_at: datetime
