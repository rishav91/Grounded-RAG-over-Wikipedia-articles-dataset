# API Contracts — Grounded RAG over Wikipedia

Expands [PRD.md §8](PRD.md#8-api-contract-mvp)'s illustrative shape into the
full contract. See [ARCHITECTURE.md](ARCHITECTURE.md#tiers--components) for
which component produces each field, and [ADR-007](ADRs.md#adr-007) for why
no LLM provider is named anywhere in this contract.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | The primary endpoint — query in, grounded answer or abstention out |
| `GET` | `/health` | Liveness/readiness — checks the process is up and Qdrant is reachable |

No other endpoints exist in the MVP. There is no ingestion endpoint — M0's
ingestion is a batch script run out-of-band, not an API surface
([PRD.md §9](PRD.md#9-milestones)).

## Versioning

Unversioned (no `/v1` prefix) for the MVP — there is exactly one caller
(the builder's own eval harness), so there's no compatibility surface to
protect yet. *Decision, not an oversight:* if this service ever gets a real
external caller, a `/v1/` prefix should be introduced **before** the first
breaking change, not retrofitted after one. Until then, this doc is the
single source of truth for the current (implicitly v0) shape.

## `POST /query`

### Request

```json
{
  "query": "string, required",
  "access_context": {
    "groups": ["string", "..."]
  },
  "filters": {
    "doc_type": "short | medium | long, optional",
    "date_range": { "from": "ISO-8601 date, optional", "to": "ISO-8601 date, optional" }
  },
  "options": {
    "top_k": 5,
    "allow_generation": true,
    "bypass_cache": false
  }
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `query` | string | yes | Natural-language question. No length limit enforced in the MVP. |
| `access_context.groups` | string[] | yes | The caller's asserted ACL groups, matched against chunk `acl_tags`. **Not authentication** — see [Auth model](#auth-model). Must include at least one group; an empty list resolves to no documents being visible (every chunk requires `acl_tags` to intersect this list). |
| `filters.doc_type` | enum | no | One of the three bands defined in [DATA-MODEL.md](DATA-MODEL.md#source--canonical-mapping). |
| `filters.date_range` | object | no | Inclusive range, matched if **either** `created_at` or `updated_at` falls within it (OR across the two fields, not AND) — a chunk created before the range but updated within it still matches. Omitted fields mean unbounded on that side. |
| `options.top_k` | integer | no, default 5 | Final number of chunks passed to generation after rerank. |
| `options.allow_generation` | boolean | no, default true | If `false`, the request short-circuits after `rerank` and returns `retrieved_chunks` with `answer: null`, `abstained: false` — a retrieval-only mode, useful for evaluating FR2/FR4 (hybrid retrieval, reranking) in isolation from generation and faithfulness. |
| `options.bypass_cache` | boolean | no, default false | Skips `cache_lookup` and forces a fresh run through `retrieve`→`rerank`→`generate`→`faithfulness`. Needed by the eval harness (`REQUIREMENTS.md`'s acceptance criteria) to measure retrieval/rerank/faithfulness quality without a stale cache hit masking the result. Does not skip the *write* — a passing answer is still cached afterward. |

### Response — grounded

```json
{
  "answer": "Albert Einstein developed the theory of relativity...",
  "abstained": false,
  "confidence": 0.92,
  "cache_hit": false,
  "citations": [
    { "chunk_id": "uuid", "doc_id": "string", "title": "string", "url": "string" }
  ],
  "retrieved_chunks": [
    { "chunk_id": "uuid", "text": "string", "score": 0.87 }
  ]
}
```

### Response — abstained

```json
{
  "answer": null,
  "abstained": true,
  "confidence": 0.31,
  "cache_hit": false,
  "citations": [],
  "retrieved_chunks": [
    { "chunk_id": "uuid", "text": "string", "score": 0.41 }
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `answer` | string \| null | `null` whenever `abstained` is `true`, or when `options.allow_generation` was `false`. |
| `abstained` | boolean | `true` only when the faithfulness judge fails the draft answer — see [ADR-006](ADRs.md#adr-006). Never `true` just because retrieval returned few/weak candidates *before* generation was attempted — that case still attempts generation and lets the judge decide. |
| `confidence` | float, 0–1 | The faithfulness judge's score. **Not a calibrated probability** — see [AI-ARCHITECTURE.md](AI-ARCHITECTURE.md#where-ai-earns-its-place). When `allow_generation` was `false`, this field is `null` (no judgment was made). |
| `cache_hit` | boolean | `true` if this response came from `query_cache` without running `retrieve`/`rerank`/`generate`/`faithfulness`. Exists specifically so UC-6 (cache hit) and UC-7 (cross-context cache safety) in [PRD.md §4.2](PRD.md#42-core-use-cases--illustrative-eval-set) can be verified directly from the response, not inferred from latency. |
| `citations[]` | array | Empty when `abstained` is `true` or generation was skipped. Every entry's `chunk_id` must resolve to an entry in this same response's `retrieved_chunks` — a citation pointing outside the retrieved set is a defect, not a valid response shape. |
| `retrieved_chunks[]` | array | Always populated when retrieval ran (i.e. not on a cache hit), even on abstain — so the caller gets the closest matches regardless of whether generation succeeded. |

## Errors

A system *failure* (something broke) is distinct from an *abstention*
(the system worked and decided the evidence was too weak). Abstention is
always a `200` with the shape above. Failures use a standard envelope:

```json
{
  "error": {
    "code": "string",
    "message": "string"
  }
}
```

| HTTP status | `error.code` | When | Maps to |
|---|---|---|---|
| `400` | `invalid_request` | Missing `query`, empty `access_context.groups`, malformed filter | Request validation, before any node runs |
| `502` | `retrieval_unavailable` | Qdrant `articles` collection unreachable | [ARCHITECTURE.md failure modes](ARCHITECTURE.md#failure-modes--degradation) |
| `502` | `embedding_unavailable` | OpenAI embeddings API down — no query embedding means no retrieval is possible | Same |
| `502` | `generation_failed` | Configured LLM provider failed after bounded retry | Same |
| `504` | `timeout` | Any node exceeded its budget | Not yet allocated per-node — see [REQUIREMENTS.md](REQUIREMENTS.md) for the end-to-end latency NFR this would enforce |

A Cohere Rerank failure or a `query_cache` outage never surfaces as an
error to the caller — both degrade silently per
[ARCHITECTURE.md](ARCHITECTURE.md#failure-modes--degradation) (fusion-only
ranking; treat as a cache miss, respectively).

## `GET /health`

```json
{ "status": "ok", "qdrant": "ok" }
```

Returns `503` with `"qdrant": "unreachable"` if the Qdrant connection check
fails. Does not check the embedding/rerank/LLM providers — those failures
surface per-request, not as a standing health signal, since the MVP makes no
uptime claim against third-party APIs ([ADR-009](ADRs.md#adr-009)).

## The retrieval tool (FR8)

The typed tool `generate` can call mid-turn, separate from the deterministic
first-pass `retrieve` call. Same underlying hybrid query and ACL filter as
`retrieve` ([ARCHITECTURE.md](ARCHITECTURE.md#tiers--components)) — the tool
schema exists so the *generator* decides if/when/how to call it, which is
the FR8 technique being demonstrated, not a different retrieval mechanism.
The model may issue more than one call in the same round (FR12;
[ADR-012](ADRs.md#adr-012)) — they execute concurrently, and one call
failing degrades that one call's result rather than the request.

```json
{
  "name": "retrieve_chunks",
  "description": "Search the document corpus for chunks relevant to a query, under the caller's existing access context and any active filters. Use this if the chunks already provided are insufficient to answer the question.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "A focused query, which may differ from the user's original question (e.g. naming an entity discovered in the first retrieval pass)" },
      "top_k": { "type": "integer", "default": 5 }
    },
    "required": ["query"]
  }
}
```

The tool's `access_context` and `filters` are bound from the original
request, not exposed as model-controllable parameters — the model can
refine *what* it searches for, but can never widen *what it's permitted to
see*. This is the same governing-principle constraint as
[ADR-008](ADRs.md#adr-008), applied to the agentic path specifically.

## Auth model

`access_context.groups` is **caller-asserted, not authenticated**. There is
no API key, token, or identity verification in the MVP — see
[ARCHITECTURE.md §Cross-cutting](ARCHITECTURE.md#cross-cutting) and
[PRD.md §2.3](PRD.md#23-non-goals-mvp)'s non-goals. A caller can claim any
`groups` value; the ACL machinery (pre-filter, cache key) is real and
correctly enforced *given* whatever `access_context` arrives, but nothing in
this contract verifies that the caller is honest about it. A real
authentication layer, if ever added, would sit in front of this endpoint and
populate `access_context` from a verified identity — the contract itself
wouldn't change shape.
