from datetime import UTC, datetime

from grounded_rag.ingestion.metadata import derive_acl_tags, derive_dates, derive_doc_type, derive_metadata


def test_doc_type_bands():
    assert derive_doc_type("x" * 100) == "short"
    assert derive_doc_type("x" * 1999) == "short"
    assert derive_doc_type("x" * 2000) == "medium"
    assert derive_doc_type("x" * 8000) == "medium"
    assert derive_doc_type("x" * 8001) == "long"


def test_acl_tags_are_deterministic_and_closed_set():
    for doc_id in ["a", "b", "c", "doc-123", "another-doc"]:
        tags = derive_acl_tags(doc_id)
        assert tags == derive_acl_tags(doc_id)
        assert len(tags) == 1
        assert tags[0] in {"public", "eng", "finance", "legal"}


def test_acl_tags_distribution_roughly_70_30():
    # Not a tuned statistical test -- just a sanity check that the corpus
    # isn't 100% one group, per DATA-MODEL.md's reasoning for the split.
    doc_ids = [f"doc-{i}" for i in range(2000)]
    public_count = sum(1 for d in doc_ids if derive_acl_tags(d) == ["public"])
    fraction = public_count / len(doc_ids)
    assert 0.6 < fraction < 0.8


def test_dates_are_deterministic_and_ordered():
    created_at, updated_at = derive_dates("doc-1")
    assert (created_at, updated_at) == derive_dates("doc-1")
    assert updated_at >= created_at
    assert created_at >= datetime(2020, 1, 1, tzinfo=UTC)


def test_derive_metadata_bundles_all_fields():
    meta = derive_metadata("doc-1", "short text")
    assert meta.doc_type == "short"
    assert meta.acl_tags == derive_acl_tags("doc-1")
    assert (meta.created_at, meta.updated_at) == derive_dates("doc-1")
