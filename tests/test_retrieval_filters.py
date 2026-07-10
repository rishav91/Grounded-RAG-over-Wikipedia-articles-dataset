from datetime import datetime

from grounded_rag.retrieval.filters import build_filter
from grounded_rag.retrieval.retrieve import retrieve


def test_groups_only_produces_must_acl_condition():
    f = build_filter(["public"])
    assert len(f.must) == 1
    assert f.must[0].key == "acl_tags"
    assert f.must[0].match.any == ["public"]
    assert f.should is None


def test_doc_type_adds_a_second_must_condition():
    f = build_filter(["public"], doc_type="long")
    assert len(f.must) == 2
    assert f.must[1].key == "doc_type"
    assert f.must[1].match.value == "long"


def test_date_range_produces_ord_should_clause_over_both_fields():
    f = build_filter(["public"], date_range={"from": "2020-01-01", "to": "2020-06-01"})
    assert len(f.should) == 2
    keys = {c.key for c in f.should}
    assert keys == {"created_at", "updated_at"}
    for condition in f.should:
        assert condition.range.gte == datetime(2020, 1, 1)
        assert condition.range.lte == datetime(2020, 6, 1)


def test_empty_date_range_dict_produces_no_should_clause():
    f = build_filter(["public"], date_range={})
    assert f.should is None


def test_retrieve_short_circuits_on_empty_groups_without_touching_clients():
    # None clients would blow up immediately if retrieve() tried to use them —
    # proves the empty-groups check happens before any embedding/Qdrant call.
    assert retrieve(None, None, None, "any query", []) == []
