# PRD: Grounded RAG Service over a Document Corpus

**Status:** Draft v2
**Owner:** Rishav
**Scope:** MVP (1K documents) with a defined roadmap to 10M documents

See [README.md](../README.md) for the governing principle, locked stack, and the full document map — this doc doesn't re-litigate either.

---

## 1. Summary

This is a personal build to deeply learn and demonstrate six production RAG techniques — hybrid search, reranking, tool use, semantic caching, query rewriting, and parallel tool calls — end to end, in a domain where grounding can be checked against a real, public corpus rather than synthetic text. It is framed as an internal API service (other systems call it, get a grounded answer back) because that framing forces the same boundary decisions a real service would need: access control, abstention, cache safety. There is no actual team or company behind it; the "internal teams" and "service developer" callers below describe the role the service is designed to fill, not an existing customer.

The MVP runs over 1,000 documents and proves the full read path end to end: hybrid retrieval, cross encoder reranking, grounded generation with citations, an abstain path, and a faithfulness check. The architecture is deliberately built so the same code scales toward 10M documents by swapping infrastructure tiers, not by rewriting components — but that 10M roadmap is a design exercise that proves the architecture holds up on paper, not a target this build will actually reach. Depth on the MVP's read path is what this project is actually measured on.

This document defines what the MVP must do, what it explicitly does not do, how success is measured, and the staged path to full scale.

---

## 2. Problem and goals

### 2.1 Problem

Internal teams need a reliable way to ask questions over a large shared document corpus and get answers they can trust. A raw LLM hallucinates. A plain vector search returns passages but no synthesized answer and no guarantee of relevance. Neither enforces who is allowed to see what. We need a service that retrieves the right passages, generates an answer strictly grounded in them, cites every claim, respects access boundaries, and declines when the evidence is thin.

### 2.2 Goals

1. Return grounded, cited answers over the corpus through a single API.
2. Make abstention a first class outcome, so the failure mode is "I do not have enough to answer," never a fabrication.
3. Demonstrate retrieval quality through hybrid search plus reranking, measurably better than vector search alone.
4. Build every component with a scale path, so the jump from 1K to 10M is a configuration and infrastructure change, not a rewrite.

### 2.3 Non-goals (MVP)

| Item | Status | Why |
|------|--------|-----|
| Multi-tenant at scale | Excluded for this build | ACL handling is present in the data model and query path but exercised with synthetic tags, not a real permission system — see [DATA-MODEL.md](DATA-MODEL.md#acl-tag-derivation) |
| Real time ingestion | Deferred — roadmap | MVP indexes in a batch job; near real time updates are a designed-for, not built, capability (FR10) |
| 1000+ QPS target | Deferred — roadmap | MVP targets correctness and quality at low QPS; throughput is a Stage 3 concern |
| A user facing UI | Excluded permanently | This is an API only service; callers are scripts or other services, never a human via a browser |
| Query rewriting, parallel tool calls | Deferred — Phase 2 | Designed for and sequenced into later milestones (FR11, FR12), not the MVP |
| Real multi-tenant permission integration | Excluded permanently | Synthetic ACLs only; a real permission model is a future system's problem to integrate against this one's interfaces |
| Multilingual retrieval | Excluded permanently | English split only |
| Fine-tuning any model | Excluded permanently | Every model in the stack is used off the shelf |
| A production on-call posture | Excluded permanently | Single learner, no users to page anyone for — see [README.md](../README.md#governing-principle-verifiable-or-abstain) |

### 2.4 Reframing "no hallucinations"

"No hallucinations" is not an achievable guarantee for any LLM system, and the PRD does not promise it. It is operationalized as **grounded or abstain**: every assertion in an answer is cited to a retrieved chunk, a faithfulness check scores whether the answer is supported by those chunks before it is returned, and unsupported answers are converted into an explicit abstention. This is a measurable bar, not a slogan.

### 2.5 Governing principle: Verifiable-or-Abstain

> An answer must never leave the system unless every claim in it can be traced to a retrieved chunk the caller is permitted to see. If that traceability cannot be established, the system must abstain — never fabricate, and never ground an answer in evidence the caller's access context doesn't permit.

This is the rule every later "should we build X?" question resolves against, including the contested ones in [ADRs.md](ADRs.md):

- **Why the faithfulness check gates every response (FR6), with no bypass.** Generation alone cannot prove its own output is grounded — a separate check has to exist downstream of every code path that can produce text, including the agentic tool-use path (FR8) and, later, query rewriting and parallel tool calls (FR11, FR12).
- **Why ACL pre-filtering happens before retrieval, not as a post-filter on the answer.** Permission has to bound what evidence ever reaches the generator, not just what gets shown — see [§7.1](#71-key-design-decisions-carried-from-architecture-work).
- **Why the semantic cache key includes the ACL signature (FR9), not just query similarity.** A cached answer is itself an unverified claim until it's re-checked against the *current* caller's permissions — caching can't be allowed to silently widen who an answer's evidence is shown to.
- **Why future phases (query rewriting, parallel tool calls) don't get a faster path that skips faithfulness.** Any new way of producing an answer inherits the same gate; "it's a new mechanism" is never a reason to exempt it.
- **Why generation is gated on an upfront sufficiency check (FR15), not left to the faithfulness check alone.** Faithfulness only catches an ungrounded answer *after* generation already ran; a sufficiency check that judges the retrieved context by itself, before generation, means "there was nothing to work with" is never mistaken for — or allowed to consume the same call budget as — an actual generation failure. See [ADR-010](ADRs.md#adr-010).

---

## 3. Dataset

### 3.1 Choice

The corpus is sourced from the `wikimedia/wikipedia` dataset on Hugging Face (English split). It is a large, clean, well structured corpus of articles, each with a stable `id`, a `url`, a `title`, and a `text` body. It is freely licensed (CC BY SA 4.0; attribution required) and accessible through the Hugging Face `datasets` library with no scraping or special access.

The decisive property for this project: the same loader slices exactly 1,000 articles for the MVP and scales toward millions later by changing one number, so the ingestion code never changes shape as scale grows. The English split alone contains on the order of millions of articles, which comfortably covers the full 10M roadmap target.

### 3.2 Access pattern

Load a fixed slice for the MVP, and stream for large scale so the corpus is never fully resident in memory:

```python
from datasets import load_dataset

# MVP: take a deterministic 1K slice
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
mvp_docs = []
for i, row in enumerate(ds):
    if i >= 1000:
        break
    mvp_docs.append(row)   # row has: id, url, title, text
```

For scale phases, keep `streaming=True` and remove the cap, processing in batches. A pre embedded alternative (`Upstash/wikipedia-2024-06-bge-m3`, Wikipedia articles with BGE-M3 vectors already computed) was considered as a fallback for scale load testing; with the embedding model now pinned to OpenAI `text-embedding-3-small` (see [ADR-002](ADRs.md#adr-002)), that fallback no longer shares a vector space with the MVP's index and is dropped — load testing at scale will re-embed a larger slice with the same pinned model instead.

### 3.3 Synthetic metadata and ACLs

Wikipedia has no real access controls, which is fine. To exercise metadata filtering and the ACL aware paths, derive deterministic synthetic fields at ingestion:

- `doc_type` assigned from article characteristics (for example length bands, or first category) to give a real facet to filter on.
- `acl_tags` assigned deterministically (for example by hashing the doc id into one of N synthetic groups) so the retrieval pre filter and the ACL aware cache key have something concrete to enforce.
- `created_at` / `updated_at` synthesized for date range filtering and for the freshness logic in later phases.

These are assigned once at ingestion and stored on every chunk, so the filtering and ACL machinery is real even though the permission semantics are simulated. The exact derivation rules are pinned in [DATA-MODEL.md §ACL tag derivation](DATA-MODEL.md#acl-tag-derivation).

---

## 4. Personas

| Persona | Scope | Primary need | Success looks like |
|---|---|---|---|
| Internal service developer (hypothetical) | Calls the query API to embed Q&A over the corpus into their own product | A single API call that returns a trustworthy answer or an honest abstention, never a confident wrong answer | Integrates against [API-CONTRACTS.md](API-CONTRACTS.md) without needing to know the retrieval/rerank/generation internals |
| The builder (primary, real) | Designs, builds, and evaluates every component end to end | Hands-on implementation of all six target techniques, each a genuinely distinct, observable mechanism (not one technique standing in for another) | The labeled eval set in [§4.2](#42-core-use-cases--illustrative-eval-set) passes; the MVP milestones in [ROADMAP.md](ROADMAP.md) each end in something runnable |
| Future reader of a write-up (secondary, real) | Read-only, after the fact | See how each technique's textbook description maps to one concrete, checkable decision in this build | The write-up (out of scope for this suite) can point at real ADRs and real eval results instead of hand-waving |
| Tenant-aware caller (future, not built) | Sends an access context; receives only answers grounded in documents that context is allowed to see | The ACL and cache-key machinery already in the MVP accepts a real permission model later without a rewrite | Not exercised in the MVP beyond synthetic `acl_tags` — see [§2.3](#23-non-goals-mvp) |

No real RBAC/visibility boundary exists yet — the synthetic ACL groups simulate one so the pre-filter and cache key have something concrete to enforce, per the governing principle in [§2.5](#25-governing-principle-verifiable-or-abstain).

Primary MVP interaction is a single API call: query plus access context in, structured answer out.

### 4.1 Use case ↔ technique map

The build is organized so all six target techniques are demonstrated, sequenced by when they are most defensible to add.

| Technique | Where it lives | Phase |
|-----------|----------------|-------|
| Hybrid search and metadata filtering | FR2, FR3 | MVP |
| Re ranking with cross encoders and bi encoders | FR4 | MVP |
| Tool use and tool schema design | FR8 | MVP |
| Semantic caching | FR9 | MVP |
| Query rewriting | FR11 | Phase 2 |
| Parallel tool calls and partial failures | FR12 | Phase 2 |

### 4.2 Core use cases / illustrative eval set

These double as the project's labeled eval set; full acceptance criteria live in [REQUIREMENTS.md](REQUIREMENTS.md) under the matching `FR-x.y`. Entity names are placeholders (`[Article-A]`, `[Article-B]`, ...) — *Assumption: the actual 1,000-article slice depends on Hugging Face's shard ordering for `20231101.en` at stream positions 0–999, which isn't known until ingestion (M0) actually runs. These illustrate the query shape and required answer behavior per technique; concrete entity names get substituted once the real slice is loaded, without changing the structure below.*

| ID | Use case | Exercises | Expected behavior |
|---|---|---|---|
| UC-1 | "What is `[Article-A]` known for?" — a single-hop factual question with one clearly best-matching article | Hybrid retrieval (FR2), grounded generation + citations (FR5) | Dense + sparse fusion both surface the same top chunk; answer cites it directly |
| UC-2 | A question whose phrasing is generic enough that several superficially similar articles compete for top-k (e.g. "When did `[Article-B]` happen?" where several similarly-titled events exist in the slice) | Reranking (FR4) | Cross-encoder rerank promotes the genuinely relevant chunk above fusion-only ranking; measured as a precision-at-k delta, not asserted by inspection |
| UC-3 | A query scoped with an explicit `doc_type` or `acl_tags` filter that excludes an otherwise-relevant chunk | Metadata/ACL pre-filter (FR3) | The excluded chunk never reaches reranking or generation — verified by checking retrieval's candidate set directly, not just the final answer |
| UC-4 | A multi-hop question where answering requires first identifying an entity, then retrieving a second time using that entity (e.g. "Who succeeded the subject of `[Article-A]`?") | Tool use / agentic retrieval (FR8) | Generator's first pass is insufficient; it calls the retrieval tool a second time with a query derived from the first pass's result |
| UC-5 | A question with no supporting article anywhere in the 1,000-doc slice | Faithfulness + abstain (FR6, FR7) | `abstained: true`, `answer: null`, `retrieved_chunks` still populated with the closest (insufficient) matches |
| UC-6 | The same query text submitted twice in a row, same access context | Semantic cache (FR9) | Second call is a cache hit; verified via a response-time or trace signal, not just identical output |
| UC-7 | The same query text submitted under two different `access_context` values, where context B lacks a group that context A has | Cross-context cache safety (FR9, the governing principle) | Context B never receives a cached answer warmed by context A; this is a standing test, not a one-time check — see [§7.3](#73-cross-context-safety-correctness-not-optional) |
| UC-8 | A factual question that is answerable, run through the full pipeline | Faithfulness, positive case | Answer is grounded with valid citations; faithfulness check passes; `abstained: false` |

---

## 5. Functional requirements

Priorities: **P0** ships in the MVP. **P1** is designed for and sequenced into later milestones. **P2** is future.

| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Batch ingest the corpus: load, chunk, embed, and index documents | P0 |
| FR2 | Hybrid retrieval: dense vector search plus sparse keyword (BM25) search, with score fusion | P0 |
| FR3 | Metadata filtering applied as a pre filter during retrieval (doc_type, date, synthetic acl_tags) | P0 |
| FR4 | Two stage ranking: bi encoder recall to a candidate set, then cross encoder rerank to a precise top k | P0 |
| FR5 | Grounded generation with inline citations to the specific source chunks used | P0 |
| FR6 | Faithfulness check that scores answer support against retrieved chunks, with an abstain path when support is weak | P0 |
| FR7 | Structured API response: answer, citations, confidence, retrieved chunks, and an explicit abstained flag | P0 |
| FR8 | Tool use: the generator can call retrieval as a typed tool, separate from the deterministic first pass retrieval | P0 |
| FR9 | Semantic caching of query to answer, keyed to be ACL aware so a cached answer is never served across an access boundary | P0 |
| FR10 | Near real time ingestion: document changes become queryable within minutes via a change driven pipeline | P1 |
| FR11 | Query rewriting: decontextualize, expand, and decompose multi hop queries before retrieval | P1 |
| FR12 | Parallel tool calls with partial failure handling, so multiple retrievals or tools run concurrently and a single failure degrades gracefully | P1 |
| FR13 | Per request observability: a trace spanning retrieve, rerank, generate, and faithfulness | P1 |
| FR14 | Feedback ingestion (thumbs up/down) to drive offline evaluation | P2 |
| FR15 | Context sufficiency check: gate generation on whether the retrieved context is adequate to answer, independent of the generator's own self-assessment | P0 |

Full acceptance criteria, stable sub-IDs (`FR-x.y`), and non-functional requirements are in [REQUIREMENTS.md](REQUIREMENTS.md).

---

## 6. Non-functional requirements

Full quantified targets and capacity math are in [REQUIREMENTS.md](REQUIREMENTS.md). Summary:

### 6.1 MVP targets (1K documents)

| NFR | MVP target | Notes |
|-----|-----------|-------|
| Retrieval latency | p95 under 500 ms | At 1K docs the index fits in memory; latency budget is generous |
| End to end latency (cold) | under 5 s including generation | Generation streamed; correctness over speed at this stage |
| Throughput | low, single digit QPS | MVP optimizes for correctness, not load |
| Faithfulness | answers either grounded with citations or abstained, on every response | The core quality bar; measured on the eval set in [§4.2](#42-core-use-cases--illustrative-eval-set) |
| Retrieval quality | hybrid plus rerank beats vector only on the eval set | Measured, not assumed |
| Availability | best effort, single node | No HA requirement in MVP |
| Index freshness | batch; minutes to hours acceptable | Real time is roadmap |

### 6.2 Scale targets (10M documents, roadmap — design exercise, not a build target)

| NFR | Scale target | Driving constraint |
|-----|-------------|--------------------|
| Corpus | 10M docs to roughly 50 to 150M chunks | Mixed length, 5 to 15 chunks per doc |
| Retrieval latency | p99 under 1 s, p50 under 200 ms | Hard SLA; cross encoder rerank is the tight budget |
| Generation latency | time to first token under 800 ms, soft total | Streamed |
| Throughput | 1000+ QPS sustained, design for roughly 3000 peak | Peak around 3x sustained for bursty internal traffic |
| Cache hit rate | planning assumption 40 to 60% | Top risk; ACL aware keying pulls this down; validate against real traffic |
| Vector index size | roughly 307–922 GB raw float32 (1536 dim, OpenAI `text-embedding-3-small`) across 50–150M chunks, roughly 77–230 GB at int8 | Chunk-text payload turns out to dominate total storage at this scale — see the full capacity math in [REQUIREMENTS.md](REQUIREMENTS.md#capacity-sizing) |
| Index freshness | under 5 minutes doc to queryable | Near real time pipeline |
| Availability | 99.9% retrieval, 99.5% generation | Generation has more failure surface; degrade to retrieval only |

---

## 7. System overview

Full component breakdown, diagrams, and failure modes are in [ARCHITECTURE.md](ARCHITECTURE.md). Summary of the two paths:

**Write path (ingestion).** Documents are loaded, split into chunks, embedded, and written into Qdrant — a single engine holding both the dense vector and sparse (BM25-style) representation per chunk, so hybrid retrieval is one query against one store rather than a fan-out and fusion across two systems. Synthetic metadata and ACL tags are attached to every chunk. In the MVP this is a batch job. At scale it becomes a change driven pipeline that distinguishes cheap metadata only updates from expensive content re embeddings.

**Read path (query).** A caller sends a query and an access context. The orchestrator checks the semantic cache first. On a miss it retrieves candidates with hybrid search under a metadata and ACL pre filter, reranks them with a cross encoder to a precise top k, and passes those to the generator. The generator produces an answer with citations and may call retrieval again as a tool when the first pass is insufficient. A faithfulness check scores the answer against the retrieved chunks. If grounded, the answer is cached and returned; if not, the service abstains and returns the closest chunks anyway so the caller still gets something usable.

### 7.1 Key design decisions carried from architecture work

- **Chunk level metadata denormalization.** ACL tags and doc_type are copied from the document down onto every chunk so filters apply inside the retrieval query with no join. Pre filtering is a correctness decision (no cross boundary leakage) and a quality decision (the rerank and generation budget is spent only on permitted chunks). See [DATA-MODEL.md](DATA-MODEL.md).
- **ACL aware semantic cache.** The cache key is (semantic cluster, tenant, acl signature), not query similarity alone. Without the acl signature, a cached answer warmed by one caller could leak to another with weaker permissions. This narrows reuse and is the single biggest risk to cache hit rate, which is itself the biggest lever on cost and latency at scale. See [ADR-005](ADRs.md#adr-005).
- **Generator hosts a retrieval tool.** Deterministic first pass retrieval guarantees a baseline for every query; the agentic tool path lets the model fetch more for multi hop or follow up needs. The faithfulness check gates everything regardless of path, per the governing principle in [§2.5](#25-governing-principle-verifiable-or-abstain).
- **Quantize and shard the vector index at scale.** Int8 quantization (a built-in Qdrant feature) cuts the index roughly 4x so it shards across a handful of nodes; the small recall loss is recovered by the full precision cross encoder rerank downstream.

### 7.2 Locked stack reference

The concrete tool/model choices behind the decisions above are pinned in [ADRs.md](ADRs.md): orchestration (LangGraph), LLM access (provider-agnostic via config), embeddings (OpenAI `text-embedding-3-small`), search engine (Qdrant, hybrid dense+sparse), reranker (Cohere Rerank API), faithfulness mechanism (LLM-as-judge). This PRD doesn't restate the alternatives considered — see the ADRs for that.

### 7.3 Cross context safety (correctness, not optional)

A standing test, not a one-time check: a query answered for access context A must never have its cached answer served to access context B when B lacks A's permissions. This is UC-7 in [§4.2](#42-core-use-cases--illustrative-eval-set), runs from M4 onward, and must always pass.

---

## 8. API contract (MVP)

A single primary endpoint. Full request/response schema, error semantics, and the retrieval tool's schema are in [API-CONTRACTS.md](API-CONTRACTS.md). Illustrative shape:

**Request**

```json
POST /query
{
  "query": "string",
  "access_context": { "groups": ["group_a", "group_b"] },
  "filters": { "doc_type": "optional", "date_range": "optional" },
  "options": { "top_k": 5, "allow_generation": true }
}
```

**Response**

```json
{
  "answer": "string or null",
  "abstained": false,
  "confidence": 0.0,
  "citations": [
    { "chunk_id": "string", "doc_id": "string", "title": "string", "url": "string" }
  ],
  "retrieved_chunks": [
    { "chunk_id": "string", "text": "string", "score": 0.0 }
  ]
}
```

When `abstained` is true, `answer` is null, `confidence` reflects the weak support, and `retrieved_chunks` still holds the closest matches so the caller can decide what to do.

---

## 9. Milestones

Full phase-by-phase scope, what each unlocks, and sequencing rationale are in [ROADMAP.md](ROADMAP.md). Summary:

### M0 — Ingestion and indexing (write path)
Load the 1K slice, chunk, embed, attach synthetic metadata and ACL tags, and populate Qdrant. **Exit:** every document is searchable via both the dense and sparse representation, with metadata attached.

### M1 — Hybrid retrieval plus filtering (FR2, FR3)
Dense plus sparse retrieval with score fusion, under a metadata and ACL pre filter. **Exit:** a query returns a fused, filtered candidate set; the eval set shows hybrid recall beating vector only.

### M2 — Reranking (FR4)
Add Cohere reranking over the candidate set to a precise top k. **Exit:** rerank measurably improves top k precision on the eval set versus fusion alone.

### M3 — Grounded generation, citations, faithfulness, abstain (FR5, FR6, FR7, FR8)
Generate answers strictly from the top k, cite every claim, score faithfulness via LLM-as-judge, and abstain when support is weak. Wire retrieval as a typed LangGraph tool the generator can call. **Exit:** every response is either grounded with citations or an explicit abstention; the API returns the full structured shape.

### M4 — Semantic caching, ACL aware (FR9)
Cache query to answer with the ACL aware key, write through only after the faithfulness check passes. **Exit:** semantically similar repeat queries from the same access context are served from cache; UC-7's cross context test passes.

### M5 — Phase 2 techniques (FR11, FR12)
Add query rewriting ahead of retrieval, and parallel tool calls with partial failure handling in the generator. **Exit:** multi hop queries are decomposed and answered; a deliberately failed tool call degrades gracefully instead of failing the request.

### M6 — Observability and feedback (FR13, FR14)
Per request tracing across all stages; feedback capture for offline eval. **Exit:** any request can be traced stage by stage; feedback is recorded.

---

## 10. Roadmap from 1K to 10M

A design exercise proving the architecture's component boundaries hold across three orders of magnitude — not a build commitment. Full table and the two scale-specific mechanisms are in [ROADMAP.md](ROADMAP.md#scale-stages).

---

## 11. Success metrics

### 11.1 Quality (the bar that matters most)

- **Faithfulness rate:** of non abstained answers, the fraction fully supported by cited chunks. Target: very high, measured on the eval set; any unsupported answer is a defect to drive to zero.
- **Abstention correctness:** when the corpus genuinely lacks the answer, the service abstains rather than fabricating (UC-5).
- **Retrieval quality:** recall at k and a precision measure on the eval set, comparing vector only, hybrid, and hybrid plus rerank (UC-1, UC-2). Target: each stage improves on the last.
- **Citation validity:** every citation points to a chunk that actually supports the claim. Target: no dangling or irrelevant citations.

### 11.2 Operational (gain weight at scale)

- Retrieval and end to end latency against the per phase targets in [§6](#6-non-functional-requirements).
- Semantic cache hit rate, measured under a realistic query mix once the cache is live. This is the flagged top risk and must be validated against real traffic, not assumed.
- Ingestion freshness once the near real time pipeline lands.

### 11.3 Cross context safety

See [§7.3](#73-cross-context-safety-correctness-not-optional) and UC-7.

---

## 12. Risks and open questions

| Risk / question | Impact | Current stance |
|-----------------|--------|----------------|
| Cache hit rate may be far below 40 to 60% | Doubles the cold path fleet at scale; drives cost and latency | Treat as top risk; measure early under real query mix; do not assume |
| Synthetic ACLs do not capture real permission complexity | MVP ACL logic may need rework when a real permission model arrives | Acceptable for MVP; design the key and pre filter to accept a real model later |
| Faithfulness scoring (LLM-as-judge) adds latency on the hot path | May pressure the sub second SLA at scale | Tune the check; decide the latency versus safety tradeoff explicitly in M3; see [ADR-006](ADRs.md#adr-006) for why the judge is LLM-based rather than a faster deterministic check |
| Chunking strategy strongly affects retrieval quality | Poor chunking caps the quality ceiling regardless of rerank | Treat chunking as a tunable parameter; revisit if quality stalls |
| Embedding model choice and dimension affect index size and recall | Wrong choice is expensive to undo at scale | Pinned: OpenAI `text-embedding-3-small`, 1536 dims — see [ADR-002](ADRs.md#adr-002); recorded per chunk so re-embedding is a planned migration |
| Cohere Rerank API introduces an external dependency and per-call cost on the hot path | Vendor outage or pricing change affects M2 onward | Accepted for MVP scale (low call volume); see [ADR-003](ADRs.md#adr-003) for the cost math and the self-hosted fallback considered |
| Resolved: which search engine, embedding model, reranker | Was foundational and previously open | Qdrant (hybrid dense+sparse), OpenAI `text-embedding-3-small`, Cohere Rerank API — see [ADRs.md](ADRs.md) |

---

## 13. Out of scope (explicit)

See [§2.3](#23-non-goals-mvp) for the full table with reasons. In short: no UI, no real multi-tenant permissions, English only, no fine-tuning, no production on-call posture, no 1000+ QPS load.
