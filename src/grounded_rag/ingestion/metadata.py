"""Synthetic doc_type / acl_tags / created_at / updated_at derivation.

Formulas are pinned in DATA-MODEL.md#source-canonical-mapping and
DATA-MODEL.md#acl-tag-derivation — deterministic functions of doc_id (or
article length) so re-ingestion is stable across runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from grounded_rag.config import (
    ACL_GROUPS,
    DOC_TYPE_MEDIUM_MAX_CHARS,
    DOC_TYPE_SHORT_MAX_CHARS,
)


@dataclass(frozen=True)
class DocMetadata:
    doc_type: str
    acl_tags: list[str]
    created_at: datetime
    updated_at: datetime


def _doc_hash(doc_id: str) -> int:
    return int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16)


def derive_doc_type(text: str) -> str:
    length = len(text)
    if length < DOC_TYPE_SHORT_MAX_CHARS:
        return "short"
    if length <= DOC_TYPE_MEDIUM_MAX_CHARS:
        return "medium"
    return "long"


def derive_acl_tags(doc_id: str) -> list[str]:
    h = _doc_hash(doc_id)
    if h % 10 < 7:
        return ["public"]
    return [ACL_GROUPS[h % 3]]


def derive_dates(doc_id: str) -> tuple[datetime, datetime]:
    h = _doc_hash(doc_id)
    created_at = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=h % 1000)
    updated_at = created_at + timedelta(days=(h // 1000) % 200)
    return created_at, updated_at


def derive_metadata(doc_id: str, text: str) -> DocMetadata:
    created_at, updated_at = derive_dates(doc_id)
    return DocMetadata(
        doc_type=derive_doc_type(text),
        acl_tags=derive_acl_tags(doc_id),
        created_at=created_at,
        updated_at=updated_at,
    )
