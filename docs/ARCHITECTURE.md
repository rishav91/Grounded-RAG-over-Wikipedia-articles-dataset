# Architecture — Grounded RAG over Wikipedia

See [README.md](../README.md) for the governing principle and locked stack.
See [ADRs.md](ADRs.md) for why each contested choice below won over its
alternatives.

## Design tenets

1. **Verifiable-or-abstain** (the governing principle) — every node that can
   produce or influence the final answer is downstream of, or routes into,
   the faithfulness gate (`faithfulness` node). Nothing — not the cache, not
   the retrieval tool, not the query-rewrite or parallel-tool-call path
   (`ADR-011`, `ADR-012`) — returns an answer that skipped it.
2. **ACL is structural, not a convention.** Permission filtering happens
   inside the retrieval query (`ADR-008`) and inside the cache key
   (`ADR-005`) — never as an afterthought filter applied to output after the
   fact.
3. **One hybrid engine, not two synced systems.** Qdrant alone serves dense +
   sparse retrieval *and* the semantic cache (as a second collection), so
   there's no cross-system consistency problem between a vector store and a
   separate keyword index, or between a search index and a separate cache
   store.
4. **No model hosting unless avoiding it costs more than running it.** Every
   model in the stack — embeddings, reranker, generation, faithfulness judge
   — is a hosted API call by default (`ADR-002`, `ADR-003`). Self-hosting is
   the documented fallback in each ADR, not the starting point.
5. **No LLM vendor lock-in.** The generation and faithfulness-judge models
   are selected by config (`ADR-007`), never imported as a hardcoded vendor
   SDK call inside node code.

## High-level architecture

```mermaid
flowchart TB
    subgraph WritePath["Write path — ingestion (M0)"]
        HF["Hugging Face\nwikimedia/wikipedia"]
        Loader["Streaming loader\ndeterministic slice"]
        Chunker["Chunker"]
        Meta["Synthetic metadata\ndoc_type, acl_tags, dates"]
        Embed["OpenAI text-embedding-3-small\ndense vector"]
        Sparse["Sparse vectorizer\nBM25-style"]

        HF --> Loader --> Chunker --> Meta
        Chunker --> Embed
        Chunker --> Sparse
    end

    Articles[("Qdrant: articles\ndense + sparse + payload")]
    Cache[("Qdrant: query_cache\nACL-aware semantic cache")]

    Meta --> Articles
    Embed --> Articles
    Sparse --> Articles

    subgraph ReadPath["Read path — query (LangGraph)"]
        Client["API caller"]
        API["POST /query"]
        CacheLookup["cache_lookup node"]
        Rewrite["rewrite_query node\nLLM: decontextualize/expand/decompose"]
        Retrieve["retrieve node\nhybrid + ACL/metadata filter,\nparallel across rewrite's sub_queries"]
        Rerank["rerank node\nCohere Rerank API"]
        Sufficiency["check_sufficiency node\ntiered score-gate + LLM judge"]
        Generate["generate node\nLLM + bound retrieval tool,\nparallel tool calls"]
        Faith["faithfulness node\nLLM-as-judge"]
        Response["response node"]

        Client --> API --> CacheLookup
        CacheLookup -->|hit| Response
        CacheLookup -->|miss| Rewrite
        Rewrite --> Retrieve
        Retrieve --> Rerank --> Sufficiency
        Sufficiency -->|sufficient| Generate
        Sufficiency -->|insufficient: abstain| Response
        Generate -->|tool call| Retrieve
        Generate --> Faith
        Faith -->|pass| Response
        Faith -->|weak support| Response
        Response -->|pass: write-through| Cache
        Response --> Client
    end

    CacheLookup -.-> Cache
    Retrieve -.-> Articles
```

## Tiers / components

| Component | Responsibility | Tech | Backs |
|---|---|---|---|
| Ingestion pipeline | Load HF dataset slice, chunk, derive synthetic metadata/ACL tags, embed (dense + sparse), upsert to Qdrant | Python batch script | FR1; [ADR-002](ADRs.md#adr-002), [ADR-004](ADRs.md#adr-004), [ADR-008](ADRs.md#adr-008) |
| Qdrant `articles` collection | Dense + sparse vector per chunk; payload carries `doc_type`/`acl_tags`/dates; one hybrid query does fusion + payload filter | Qdrant | FR2, FR3; [ADR-004](ADRs.md#adr-004), [ADR-008](ADRs.md#adr-008) |
| Qdrant `query_cache` collection | Semantic cache: query embedding → cached answer + citations; payload carries `acl_signature` and a write timestamp for TTL | Qdrant, second collection | FR9; [ADR-005](ADRs.md#adr-005) |
| `cache_lookup` node | Embeds the incoming query, searches `query_cache` filtered by `acl_signature`, returns hit/miss | LangGraph node | FR9 |
| `rewrite_query` node | One LLM call: decontextualize/expand the query, decompose into independently-retrievable `sub_queries` when genuinely parallel (never a sequential hop) | LangGraph node + configured LLM | FR11; [ADR-011](ADRs.md#adr-011) |
| `retrieve` node | Hybrid query against `articles`, filtered by ACL/metadata before fusion; fires concurrently across the rewritten query and any `sub_queries`, merging by `chunk_id` | Plain Python function through M1/M2 (`src/grounded_rag/retrieval/retrieve.py`); wrapped as a LangGraph node at M3, when the retrieval-tool cycle first needs the graph | FR2, FR3, FR11 |
| Retrieval tool | The same hybrid+filter query as `retrieve`, exposed as a typed LangGraph tool the `generate` node can call mid-turn — possibly more than once per round, executed concurrently | LangGraph tool | FR8, FR12; [ADR-012](ADRs.md#adr-012) |
| `rerank` node | Calls the Cohere Rerank API over the fused candidate set | LangGraph node + Cohere API | FR4; [ADR-003](ADRs.md#adr-003) |
| `check_sufficiency` node | Tiered score-gate + LLM judge deciding whether the reranked chunks are adequate to attempt an answer; short-circuits to abstain, skipping `generate`/`faithfulness`, on obviously-hopeless context | LangGraph node + (sometimes) configured LLM | FR15; [ADR-010](ADRs.md#adr-010) |
| `generate` node | Calls the configured LLM with the top-k context, a citation-constrained prompt, and the retrieval tool bound with `parallel_tool_calls=True`; a failed concurrent call degrades to a failure `ToolMessage` on its own `tool_call_id` rather than aborting the round | LangGraph node + configured LLM | FR5, FR8, FR12; [ADR-001](ADRs.md#adr-001), [ADR-007](ADRs.md#adr-007), [ADR-012](ADRs.md#adr-012) |
| `faithfulness` node | LLM-as-judge scores each cited claim against its cited chunk and whether the answer addresses the question; decides abstain | LangGraph node + configured LLM | FR6, FR7, FR-5.4; [ADR-006](ADRs.md#adr-006) |
| `response` node | Formats the structured API response; writes through to `query_cache` only after a faithfulness pass | LangGraph node | FR7, FR9 |
| API layer | Accepts `POST /query`, runs the compiled graph, returns the structured response | HTTP service (Python) | [API-CONTRACTS.md](API-CONTRACTS.md) |

There is no intent router in this graph (unlike a multi-path agent design) —
every query takes the same cache → rewrite_query → retrieve → rerank →
check_sufficiency → generate → faithfulness shape. The only branching is
cache hit/miss, the sufficiency sufficient/insufficient split, the
`generate` node's optional tool call(s) back into `retrieve`, and the
faithfulness pass/fail split.

## Key flows

### Query flow

```mermaid
sequenceDiagram
    actor Caller
    participant API as Query API
    participant CL as cache_lookup
    participant RW as rewrite_query
    participant Ret as retrieve
    participant Rerank as rerank (Cohere)
    participant Suff as check_sufficiency
    participant Gen as generate (LLM)
    participant Faith as faithfulness (LLM judge)
    participant Cache as Qdrant query_cache

    Caller->>API: POST /query (query + access_context + filters)
    API->>CL: embed query, lookup by (cluster, acl_signature)
    CL->>Cache: filtered search
    alt Cache hit
        Cache-->>CL: cached answer + citations
        CL-->>Caller: cached response
    else Cache miss
        CL->>RW: proceed
        RW->>RW: decontextualize/expand; decompose only if\nsub-questions are independently retrievable
        RW->>Ret: rewritten_query + sub_queries
        Ret->>Ret: hybrid search per query, concurrently;\nACL/metadata payload filter; merge by chunk_id
        Ret->>Rerank: fused candidate set
        Rerank-->>Suff: precise top-k
        Suff->>Suff: score-gate, LLM judge only if ambiguous
        alt Context insufficient
            Suff-->>API: abstain — generate/faithfulness never run
            API-->>Caller: abstained=true, retrieved_chunks only
        else Context sufficient
            Suff-->>Gen: proceed
            opt First pass insufficient
                Gen->>Ret: call retrieval tool (possibly >1 call, concurrently;\na failed call degrades, doesn't abort the round)
                Ret-->>Gen: additional chunks, same ACL filter
            end
            Gen-->>Faith: draft answer + citations
            Faith->>Faith: score each claim vs. its cited chunk, and answer relevance
            alt Grounded and relevant
                Faith-->>API: pass, confidence
                API->>Cache: write-through (acl_signature key)
                API-->>Caller: answer, citations, confidence
            else Weak support or off-topic
                Faith-->>API: fail
                API-->>Caller: abstained=true, retrieved_chunks only
            end
        end
    end
```

**Cache hit:** `cache_lookup` embeds the query, searches `query_cache`
filtered to the caller's `acl_signature`, and returns the cached answer
directly — `rewrite_query`, `retrieve`, `rerank`, `check_sufficiency`,
`generate`, and `faithfulness` never run.

**Cache miss, context insufficient:** `rewrite_query` → `retrieve` →
`rerank` → `check_sufficiency` determines the retrieved context is
inadequate (FR15) → `response` returns `abstained: true` directly, without
ever running `generate` or `faithfulness` — the cheaper of the two abstain
paths.

**Cache miss, grounded:** `rewrite_query` (decontextualize/expand, decompose
into `sub_queries` only if independently retrievable, FR11) → `retrieve`
(concurrently across the rewritten query and any `sub_queries`) → `rerank`
→ `check_sufficiency` passes → `generate` (optionally calling the retrieval
tool mid-turn, possibly with more than one concurrent call, FR12) →
`faithfulness` passes (both groundedness and relevance, FR-5.4) → `response`
writes through to `query_cache` and returns the answer.

**Cache miss, weak support or off-topic:** same path as grounded, but
`faithfulness` fails on either axis → `response` returns `abstained: true`
with the closest `retrieved_chunks` and **does not** write through to the
cache — an abstention is not a verified answer and must not be served as a
cache hit to a later, identical query.

### Ingestion flow

```mermaid
flowchart LR
    A["Load dataset\nstreaming, deterministic slice"] --> B["Chunk documents\n5-15 chunks/doc"]
    B --> C["Derive synthetic metadata\ndoc_type, acl_tags, dates"]
    C --> D["Embed chunks\nOpenAI text-embedding-3-small"]
    C --> E["Sparse-vectorize chunks\nBM25-style"]
    D --> F["Upsert to Qdrant\narticles collection"]
    E --> F
    C --> F
    F --> G["Searchable corpus"]
```

Chunk IDs are derived deterministically from `(doc_id, chunk_index)`, so
re-running ingestion **upserts**, never duplicates — see
[Cross-cutting](#cross-cutting).

## Multi-tenancy & isolation

There is no real tenant system. `access_context.groups` simulates one: every
chunk carries synthetic `acl_tags` derived deterministically at ingestion
(see [DATA-MODEL.md](DATA-MODEL.md#acl-tag-derivation)), and every read-path
request carries an `access_context` the caller asserts (not authenticates) —
this is a simulation of an access boundary for exercising the pre-filter and
cache-key machinery, not a real authorization system. Two structural
enforcement points, both pre-decided in ADRs:

- **Retrieval pre-filter** ([ADR-008](ADRs.md#adr-008)) — `acl_tags` are
  checked inside the Qdrant payload filter before fusion; a chunk outside
  the caller's groups never enters the candidate set.
- **Cache key** ([ADR-005](ADRs.md#adr-005)) — `acl_signature` is part of
  the `query_cache` key; a cached answer never crosses an `acl_signature`
  boundary.

The standing backstop test is UC-7
([PRD.md §4.2](PRD.md#42-core-use-cases--illustrative-eval-set)): the same
query text under two different `access_context` values must never collide on
a cache hit. This runs from M4 onward and must always pass.

A real permission system, if ever integrated, would replace how
`access_context` is populated (authenticated identity → real group
membership) without changing either enforcement point — both are already
written against an opaque list of group strings.

## Scale & capacity model

Full target numbers and the capacity math behind them are in
[REQUIREMENTS.md](REQUIREMENTS.md#capacity-sizing). How the design gets
there, component by component:

| Lever | Mechanism |
|---|---|
| Vector index size at 10M docs | Qdrant's built-in int8 quantization (~4x reduction), applied at Stage 2 of the roadmap |
| Query throughput | Qdrant shards across nodes (Stage 3); the LangGraph read path is stateless per request and scales horizontally behind a load balancer |
| Rerank/embedding/LLM call volume | Each is a hosted API call with its own provider-side scaling — not infrastructure this project operates, but a cost line that grows linearly with uncached query volume |
| The biggest lever: cache hit rate | A hit skips retrieval, rerank, `check_sufficiency`, *and* the generation + faithfulness LLM calls entirely. The ACL-aware key narrows hit rate ([ADR-005](ADRs.md#adr-005)) — the 40–60% planning assumption ([PRD.md §6.2](PRD.md#62-scale-targets-10m-documents-roadmap--design-exercise-not-a-build-target)) is unvalidated and the top operational risk |
| Index freshness at scale | Deferred to FR10 (P1) — a change-driven pipeline splitting cheap metadata-only updates from expensive re-embeds; not built in the MVP |

This is a design exercise proving the component boundaries hold across three
orders of magnitude ([PRD.md §10](PRD.md#10-roadmap-from-1k-to-10m)), not a
load-tested guarantee.

## Failure modes & degradation

| Tier | What breaks | What the system does |
|---|---|---|
| Qdrant unreachable (`articles`) | Retrieval query fails or times out | No candidate set means generation must not run with zero grounding context — short-circuit to an explicit error response, distinct from an `abstained` answer (abstain means "evidence was weak," not "the system failed") |
| Qdrant unreachable (`query_cache`) | Cache lookup or write-through fails | Treat as a cache miss and proceed through the full read path. Caching is a latency/cost optimization, never a correctness dependency — the read path must never block on cache availability |
| Cohere Rerank API down or rate-limited | `rerank` node call fails | Degrade to fusion-only ranking — pass the fused top-k straight to `check_sufficiency`/`generate` rather than failing the request (documented in [ADR-003](ADRs.md#adr-003)); `check_sufficiency`'s tier-1 score gate is skipped in this case since fusion-only scores aren't on Cohere's 0-1 scale ([ADR-010](ADRs.md#adr-010)) |
| OpenAI embeddings API down | Can't embed the incoming query (or, at ingestion time, a document) | Query-time: fail the request — there is no retrieval without a query embedding, a hard dependency, not a degradable one. Ingestion-time: blocks that ingestion run only, not a live request |
| Configured LLM down/erroring — `rewrite_query` | Rewrite LLM call fails | Fail **open** — default `rewritten_query` to the raw query and `sub_queries` to `[]`, proceed to `retrieve` unchanged. A rewrite failure must never block retrieval ([ADR-011](ADRs.md#adr-011)) |
| A `sub_queries` entry's (or a concurrent `retrieve_chunks` tool call's) retrieval fails | One leg of a concurrent fan-out raises | Degrades gracefully — drop that one leg's results, proceed with whatever the other concurrent calls returned ([ADR-011](ADRs.md#adr-011), [ADR-012](ADRs.md#adr-012)). Scoped narrowly to the parallel fan-out: the single primary `retrieve` call above has no sibling to fall back on and is still a hard failure |
| Configured LLM down/erroring — `check_sufficiency` | Tier-2 judge call fails | Fail **open** — default to `sufficient=True` and proceed to `generate`, never block a request that might otherwise succeed. Distinct from `faithfulness`'s fail-closed behavior below: sufficiency is a cost-saving pre-filter, not the safety-critical gate — that's still faithfulness ([ADR-010](ADRs.md#adr-010)) |
| Configured LLM down/erroring — `generate` | Generation call fails | Bounded retry with backoff; on exhaustion, return an explicit error distinct from `abstained` |
| Configured LLM down/erroring — `faithfulness` | Judge call fails | Same bounded retry; on exhaustion, default to **abstain**, never to an unverified pass — consistent with the governing principle: never skip the gate, never assume a pass |
| Faithfulness judge is too lenient (passes an ungrounded answer) | An unverified answer is returned as if grounded | No runtime mitigation exists for this; it's why the eval set ([PRD.md §4.2](PRD.md#42-core-use-cases--illustrative-eval-set)) exists, and why [ADR-006](ADRs.md#adr-006) documents a deterministic+LLM hybrid as the next step if this proves common |

## Cross-cutting

- **Security:** corpus content is public, low-sensitivity (Wikipedia);
  `access_context` is a synthetic, self-asserted boundary, **not** a real
  authentication/authorization system — calling code must never treat
  passing ACL groups as proof of identity. API keys (OpenAI, Cohere,
  whichever LLM provider is configured per [ADR-007](ADRs.md#adr-007)) live
  in environment variables, never committed.
- **Idempotency:** ingestion upserts into Qdrant by a chunk ID deterministically
  derived from `(doc_id, chunk_index)`, so re-running ingestion never
  duplicates chunks.
- **Consistency:** single writer (the ingestion batch job) for `articles` at
  MVP scale — the read path is read-only against it. `query_cache` has many
  concurrent writers (every request that passes faithfulness), but each
  write targets a distinct `(semantic cluster, acl_signature)` point, so
  last-write-wins is sufficient; no transactional guarantee is needed.
- **Config/secrets:** environment variables hold `OPENAI_API_KEY`,
  `COHERE_API_KEY`, whichever provider key the configured generation/judge
  LLM needs, and `QDRANT_URL`/`QDRANT_API_KEY` (local instance for MVP,
  managed cluster at scale — [ADR-009](ADRs.md#adr-009)). Nothing
  provider-specific is hardcoded in node code.
