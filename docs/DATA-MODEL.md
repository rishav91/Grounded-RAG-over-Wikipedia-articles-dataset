# Data Model — Grounded RAG over Wikipedia

See [ADR-004](ADRs.md#adr-004) for why Qdrant is the single store for both
collections below, and [ADR-008](ADRs.md#adr-008) for why ACL/metadata are
denormalized onto every chunk rather than joined at query time.

## ER overview

```mermaid
erDiagram
    ARTICLE ||--o{ CHUNK : "split into"
    CHUNK }o--|| ACL_TAGS : "carries (denormalized)"
    QUERY ..o{ CACHE_ENTRY : "may hit"

    ARTICLE {
        string doc_id "stable HF dataset id"
        string url
        string title
        string text "full article body, not stored past chunking"
    }
    CHUNK {
        string chunk_id "uuid5(doc_id, chunk_index)"
        string doc_id "FK to ARTICLE"
        int chunk_index
        string text
        vector dense "1536-dim, OpenAI text-embedding-3-small"
        sparse_vector sparse "BM25-style, fastembed Qdrant/bm25"
        string doc_type "derived: length band"
        array acl_tags "derived: synthetic group(s)"
        datetime created_at "synthesized"
        datetime updated_at "synthesized"
    }
    CACHE_ENTRY {
        string cache_id "generated per write, not deterministic"
        vector dense "query embedding"
        string acl_signature "hash of access_context.groups"
        string query_text
        string answer
        array citations
        float confidence
        datetime created_at "for future TTL/pruning"
    }
```

Two Qdrant collections hold everything; there is no separate relational
store. This is deliberate — see [ARCHITECTURE.md design tenet
3](ARCHITECTURE.md#design-tenets).

## Canonical schema

### Collection: `articles`

One point per chunk.

| Field | Type | Notes |
|---|---|---|
| `id` (point ID) | UUID | `uuid5(NAMESPACE, f"{doc_id}:{chunk_index}")` — deterministic, so re-ingestion upserts rather than duplicates |
| `vector["dense"]` | float32[1536] | OpenAI `text-embedding-3-small` ([ADR-002](ADRs.md#adr-002)) |
| `vector["sparse"]` | sparse vector | BM25-style, generated via `fastembed`'s `Qdrant/bm25` sparse embedding model — the standard pairing for Qdrant hybrid queries |
| `payload.doc_id` | keyword, indexed | The HF dataset row's `id` field |
| `payload.chunk_index` | integer | Position within the document |
| `payload.title` | keyword | |
| `payload.url` | keyword | Wikipedia source URL, used in citations |
| `payload.text` | text | The chunk's raw text, returned in `retrieved_chunks` |
| `payload.doc_type` | keyword, indexed | Derived length band — see [below](#source--canonical-mapping) |
| `payload.acl_tags` | keyword[], indexed | Derived synthetic group(s) — see [ACL tag derivation](#acl-tag-derivation) |
| `payload.created_at` / `payload.updated_at` | datetime, indexed | Synthesized — see below |

`doc_type`, `acl_tags`, `created_at`, `updated_at` are payload-indexed so
Qdrant's filter can apply them before fusion, per
[ADR-008](ADRs.md#adr-008).

### Collection: `query_cache`

One point per distinct cached answer (not one per request — a hit reuses an
existing point, a miss-then-pass writes a new one).

| Field | Type | Notes |
|---|---|---|
| `id` (point ID) | UUID | Generated at write time; no deterministic relationship to the query (semantic matching, not exact-key lookup) |
| `vector["dense"]` | float32[1536] | The original query's embedding — reused so cache lookups and article lookups share one embedding model |
| `payload.acl_signature` | keyword, indexed | Stable hash of `access_context.groups`, sorted before hashing so group order doesn't change the signature — [ADR-005](ADRs.md#adr-005) |
| `payload.query_text` | text | Original query, for debugging/inspection only — not used for matching |
| `payload.answer` | text | |
| `payload.citations` | array\<object\> | `{chunk_id, doc_id, title, url}` per citation |
| `payload.confidence` | float | The faithfulness judge's score at write time |
| `payload.created_at` | datetime | For future TTL/pruning — see [Indexing & partitioning](#indexing--partitioning) |

A cache **lookup** is a vector search against `query_cache`, filtered to
`payload.acl_signature == <caller's signature>`, treated as a **hit** only
if the top result's score clears a similarity threshold (starting value:
cosine ≥ 0.92, *Assumption: tune against false-hit rate once M4 is built and
real query paraphrase patterns are observed*) — otherwise it's a miss. A
cache **write** only happens after a faithfulness pass
([ARCHITECTURE.md query flow](ARCHITECTURE.md#query-flow)); an abstained
answer is never written.

## Source → canonical mapping

Each Hugging Face dataset row (`id`, `url`, `title`, `text`) becomes one
`ARTICLE` and N `CHUNK`s:

1. **Chunking.** Split `text` into chunks of roughly 500 tokens with 50
   tokens of overlap, sentence-boundary aware. *Assumption: this is a
   starting default, not a tuned value — [PRD.md
   §12](PRD.md#12-risks-and-open-questions) flags chunking strategy as a
   tunable parameter to revisit if retrieval quality stalls in M1/M2.* This
   yields the 5–15 chunks/doc range assumed throughout the capacity math in
   [REQUIREMENTS.md](REQUIREMENTS.md).
2. **`doc_type` derivation** — by article length (`len(text)`), three bands:
   `short` (< 2,000 chars), `medium` (2,000–8,000 chars), `long` (> 8,000
   chars). A real facet for FR3's metadata filter to exercise.
3. **ACL tag derivation** — see below.
4. **`created_at` / `updated_at` derivation** — both deterministic functions
   of a hash of `doc_id`, so identical across re-ingestion runs:
   `created_at = 2020-01-01 + (h % 1000) days`,
   `updated_at = created_at + ((h // 1000) % 200) days`, where
   `h = int(sha256(doc_id).hexdigest(), 16)`. Gives every document a stable,
   spread-out synthetic date range for FR3's date-range filter to exercise.

### ACL tag derivation

Closed set of four synthetic groups for the MVP: `public`, `eng`,
`finance`, `legal` — arbitrary stand-ins for internal departments, with no
real-world meaning attached to which articles land in which group.

```
h = int(sha256(doc_id).hexdigest(), 16)
if h % 10 < 7:
    acl_tags = ["public"]              # ~70% of docs — broadly visible
else:
    acl_tags = [["eng", "finance", "legal"][h % 3]]   # ~30% — restricted to exactly one group
```

The 70/30 split is deliberate, not arbitrary decoration: it guarantees a
meaningful fraction of the corpus is excluded for *some* `access_context`
values, so FR3's pre-filter and UC-3/UC-7 in
[PRD.md §4.2](PRD.md#42-core-use-cases--illustrative-eval-set) have real
cases to exercise — a corpus that's 100% `public` would make the ACL
machinery untestable. `acl_tags` is computed once per document and copied
onto every one of its chunks (denormalization, [ADR-008](ADRs.md#adr-008)).

## Identity / dedup / resolution

- `doc_id` is the HF dataset row's `id` field — Wikipedia's own stable page
  ID, already unique and stable across dataset versions for a given
  snapshot.
- `chunk_id` (the Qdrant point ID) is `uuid5(NAMESPACE, f"{doc_id}:{chunk_index}")`
  — deterministic, so re-running ingestion with the same chunking parameters
  upserts existing points rather than creating duplicates (the idempotency
  property [ARCHITECTURE.md](ARCHITECTURE.md#cross-cutting) relies on).
  Changing the chunking parameters (size/overlap) changes every downstream
  `chunk_index` boundary and is therefore a full re-ingestion, not an
  incremental update — consistent with the "alias-based deployment" roadmap
  mechanism in [PRD.md §10](PRD.md#10-roadmap-from-1k-to-10m).
- No cross-source entity resolution is needed — every chunk traces to
  exactly one source document, and there's only one source (the Wikipedia
  slice).

## Storage mapping

A single Qdrant instance holds both collections. No relational database, no
separate cache store (e.g. Redis), and no separate keyword-search engine —
deliberate, per [ADR-004](ADRs.md#adr-004) and the "no synced systems"
design tenet in [ARCHITECTURE.md](ARCHITECTURE.md#design-tenets). The two
collections are isolated from each other (different vector dimensions and
purposes) but share infrastructure, credentials, and the deployment
lifecycle in [ADR-009](ADRs.md#adr-009).

## Indexing & partitioning

| Concern | MVP (1K docs, ~5–15K chunks) | Scale roadmap |
|---|---|---|
| Shards | Single shard (Qdrant default) | Stage 3: sharded across nodes, per [PRD.md §10](PRD.md#10-roadmap-from-1k-to-10m) |
| Quantization | None — full float32 dense vectors | Stage 2: int8 quantization (~4x size reduction), recall loss recovered by Cohere rerank downstream |
| Payload indexes | `doc_type`, `acl_tags`, `created_at`, `updated_at` indexed for filter performance | Same fields, same indexes — payload indexing strategy doesn't change with scale |
| `query_cache` growth | Unbounded for the MVP — no TTL/expiry mechanism is built; acceptable at MVP traffic volume | A pruning job (delete points past a `created_at` threshold) becomes necessary once cache volume is large enough to matter; not designed yet — flagged here as a known gap, not solved |
