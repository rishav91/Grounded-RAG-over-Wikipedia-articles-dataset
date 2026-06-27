from grounded_rag.ingestion.ids import chunk_id


def test_deterministic_across_calls():
    assert chunk_id("doc-1", 0) == chunk_id("doc-1", 0)


def test_distinct_for_different_chunk_index():
    assert chunk_id("doc-1", 0) != chunk_id("doc-1", 1)


def test_distinct_for_different_doc_id():
    assert chunk_id("doc-1", 0) != chunk_id("doc-2", 0)


def test_is_a_valid_uuid_string():
    import uuid

    uuid.UUID(chunk_id("doc-1", 0))
