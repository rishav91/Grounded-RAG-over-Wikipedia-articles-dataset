"""Deterministic chunk identity.

See DATA-MODEL.md#canonical-schema: chunk_id = uuid5(NAMESPACE, "doc_id:chunk_index"),
deterministic so re-ingestion upserts rather than duplicates (FR-1.2).
"""

from __future__ import annotations

import uuid

from grounded_rag.config import CHUNK_ID_NAMESPACE


def chunk_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{doc_id}:{chunk_index}"))
