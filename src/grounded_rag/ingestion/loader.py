"""Streaming loader for the deterministic Wikipedia slice.

PRD.md §3.2: stream wikimedia/wikipedia (20231101.en) and take the first
`limit` rows — the same loader scales to millions later by changing one
number, per PRD.md §3.1.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset

from grounded_rag.config import HF_DATASET_CONFIG, HF_DATASET_NAME, MVP_SLICE_SIZE


@dataclass(frozen=True)
class Article:
    doc_id: str
    url: str
    title: str
    text: str


def load_articles(limit: int = MVP_SLICE_SIZE) -> Iterator[Article]:
    ds = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= limit:
            break
        yield Article(doc_id=row["id"], url=row["url"], title=row["title"], text=row["text"])
