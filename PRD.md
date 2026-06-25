# PRD: Grounded RAG Service over a Document Corpus

**Status:** Draft v1
**Owner:** Rishav
**Scope:** MVP (1K documents) with a defined roadmap to 10M documents

---

## 1. Summary

We are building an internal Retrieval Augmented Generation (RAG) service. Other systems call it via an API, send a natural language query plus an access context, and receive a grounded answer with citations, a confidence score, and the supporting source chunks. When retrieval does not surface enough support, the service abstains instead of fabricating.

The MVP runs over 1,000 documents and proves the full read path end to end: hybrid retrieval, cross encoder reranking, grounded generation with citations, an abstain path, and a faithfulness check. The architecture is deliberately built so the same code scales toward 10M documents by swapping infrastructure tiers, not by rewriting components.

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

1. Not multi-tenant at scale. ACL handling is present in the data model and query path but exercised with synthetic tags, not a real permission system.
2. Not real time ingestion. The MVP indexes the corpus in a batch job. Near real time updates are roadmap.
3. Not the full 1000+ QPS target. The MVP targets correctness and quality at low QPS.
4. Not a UI. This is an API only service. Callers are scripts or other services.
5. Not query rewriting or parallel tool calls in the first build. Both are designed for and sequenced into later milestones.

### 2.4 Reframing "no hallucinations"

"No hallucinations" is not an achievable guarantee for any LLM system, and the PRD does not promise it. It is operationalized as **grounded or abstain**: every assertion in an answer is cited to a retrieved chunk, a faithfulness check scores whether the answer is supported by those chunks before it is returned, and unsupported answers are converted into an explicit abstention. This is a measurable bar, not a slogan.

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

For scale phases, keep `streaming=True` and remove the cap, processing in batches. A pre embedded alternative (`Upstash/wikipedia-2024-06-bge-m3`, Wikipedia articles with BGE-M3 vectors already computed) is kept as a fallback for scale load testing, to skip the embedding step when the goal is to stress the index rather than the pipeline.

### 3.3 Synthetic metadata and ACLs

Wikipedia has no real access controls, which is fine. To exercise metadata filtering and the ACL aware paths, derive deterministic synthetic fields at ingestion:

- `doc_type` assigned from article characteristics (for example length bands, or first category) to give a real facet to filter on.
- `acl_tags` assigned deterministically (for example by hashing the doc id into one of N synthetic groups) so the retrieval pre filter and the ACL aware cache key have something concrete to enforce.
- `created_at` / `updated_at` synthesized for date range filtering and for the freshness logic in later phases.

These are assigned once at ingestion and stored on every chunk, so the filtering and ACL machinery is real even though the permission semantics are simulated.

---

## 4. Users and use cases

| User | Use case |
|------|----------|
| Internal service developer | Calls the query API to embed Q&A over the corpus into their own product |
| The builder (you) | Validates retrieval quality, faithfulness, and abstain behavior against a labeled question set |
| Future: tenant aware caller | Sends an access context; receives only answers grounded in documents that context is allowed to see |

Primary MVP interaction is a single API call: query plus access context in, structured answer out.

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

### 5.1 The six required techniques, mapped

The build is organized so all six target techniques are demonstrated, sequenced by when they are most defensible to add.

| Technique | Where it lives | Phase |
|-----------|----------------|-------|
| Hybrid search and metadata filtering | FR2, FR3 | MVP |
| Re ranking with cross encoders and bi encoders | FR4 | MVP |
| Tool use and tool schema design | FR8 | MVP |
| Semantic caching | FR9 | MVP |
| Query rewriting | FR11 | Phase 2 |
| Parallel tool calls and partial failures | FR12 | Phase 2 |

---

## 6. Non-functional requirements

### 6.1 MVP targets (1K documents)

| NFR | MVP target | Notes |
|-----|-----------|-------|
| Retrieval latency | p95 under 500 ms | At 1K docs the index fits in memory; latency budget is generous |
| End to end latency (cold) | under 5 s including generation | Generation streamed; correctness over speed at this stage |
| Throughput | low, single digit QPS | MVP optimizes for correctness, not load |
| Faithfulness | answers either grounded with citations or abstained, on every response | The core quality bar; measured on a labeled set |
| Retrieval quality | hybrid plus rerank beats vector only on the labeled set | Measured, not assumed; see metrics |
| Availability | best effort, single node | No HA requirement in MVP |
| Index freshness | batch; minutes to hours acceptable | Real time is roadmap |

### 6.2 Scale targets (10M documents, roadmap)

| NFR | Scale target | Driving constraint |
|-----|-------------|--------------------|
| Corpus | 10M docs to roughly 50 to 150M chunks | Mixed length, 5 to 15 chunks per doc |
| Retrieval latency | p99 under 1 s, p50 under 200 ms | Hard SLA; cross encoder rerank is the tight budget |
| Generation latency | time to first token under 800 ms, soft total | Streamed |
| Throughput | 1000+ QPS sustained, design for roughly 3000 peak | Peak around 3x sustained for bursty internal traffic |
| Cache hit rate | planning assumption 40 to 60% | Top risk; ACL aware keying pulls this down; validate against real traffic |
| Vector index size | roughly 460 GB raw at 768 dim float32, roughly 115 GB at int8 | Forces sharding plus quantization; no single node |
| Index freshness | under 5 minutes doc to queryable | Near real time pipeline |
| Availability | 99.9% retrieval, 99.5% generation | Generation has more failure surface; degrade to retrieval only |

---

## 7. System overview

The service has two paths.

**Write path (ingestion).** Documents are loaded, split into chunks, embedded, and written to two indexes: a vector index for dense search and a keyword index for sparse search. Synthetic metadata and ACL tags are attached to every chunk. In the MVP this is a batch job. At scale it becomes a change driven pipeline that distinguishes cheap metadata only updates from expensive content re embeddings.

**Read path (query).** A caller sends a query and an access context. The orchestrator checks the semantic cache first. On a miss it retrieves candidates with hybrid search under a metadata and ACL pre filter, reranks them with a cross encoder to a precise top k, and passes those to the generator. The generator produces an answer with citations and may call retrieval again as a tool when the first pass is insufficient. A faithfulness check scores the answer against the retrieved chunks. If grounded, the answer is cached and returned; if not, the service abstains and returns the closest chunks anyway so the caller still gets something usable.

### 7.1 Key design decisions carried from architecture work

- **Chunk level metadata denormalization.** ACL tags and doc_type are copied from the document down onto every chunk so filters apply inside the retrieval query with no join. Pre filtering is a correctness decision (no cross boundary leakage) and a quality decision (the rerank and generation budget is spent only on permitted chunks).
- **ACL aware semantic cache.** The cache key is (semantic cluster, tenant, acl signature), not query similarity alone. Without the acl signature, a cached answer warmed by one caller could leak to another with weaker permissions. This narrows reuse and is the single biggest risk to cache hit rate, which is itself the biggest lever on cost and latency at scale.
- **Generator hosts a retrieval tool.** Deterministic first pass retrieval guarantees a baseline for every query; the agentic tool path lets the model fetch more for multi hop or follow up needs. The faithfulness check gates everything regardless of path.
- **Quantize and shard the vector index at scale.** Int8 quantization cuts the index roughly 4x so it shards across a handful of nodes; the small recall loss is recovered by the full precision cross encoder rerank downstream.

---

## 8. API contract (MVP)

A single primary endpoint. Illustrative shape, to be refined in implementation.

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

### M0 — Ingestion and indexing (write path)
Load the 1K slice, chunk, embed, attach synthetic metadata and ACL tags, and populate the vector and keyword indexes. **Exit:** every document is searchable in both indexes with metadata attached.

### M1 — Hybrid retrieval plus filtering (FR2, FR3)
Dense plus sparse retrieval with score fusion, under a metadata and ACL pre filter. **Exit:** a query returns a fused, filtered candidate set; a labeled query set shows hybrid recall beating vector only.

### M2 — Reranking (FR4)
Add cross encoder reranking over the candidate set to a precise top k. **Exit:** rerank measurably improves top k precision on the labeled set versus fusion alone.

### M3 — Grounded generation, citations, faithfulness, abstain (FR5, FR6, FR7, FR8)
Generate answers strictly from the top k, cite every claim, score faithfulness, and abstain when support is weak. Wire retrieval as a typed tool the generator can call. **Exit:** every response is either grounded with citations or an explicit abstention; the API returns the full structured shape.

### M4 — Semantic caching, ACL aware (FR9)
Cache query to answer with the ACL aware key, write through only after the faithfulness check passes. **Exit:** semantically similar repeat queries from the same access context are served from cache; a different access context never receives another context's cached answer (verified by a deliberate cross context test).

### M5 — Phase 2 techniques (FR11, FR12)
Add query rewriting ahead of retrieval, and parallel tool calls with partial failure handling in the generator. **Exit:** multi hop queries are decomposed and answered; a deliberately failed tool call degrades gracefully instead of failing the request.

### M6 — Observability and feedback (FR13, FR14)
Per request tracing across all stages; feedback capture for offline eval. **Exit:** any request can be traced stage by stage; feedback is recorded.

---

## 10. Roadmap from 1K to 10M

The MVP and the 10M target share the same component boundaries. Scaling is staged so each jump validates one dimension before adding the next.

| Stage | Corpus | What changes | What stays the same |
|-------|--------|--------------|---------------------|
| MVP | 1K | In memory or single node indexes; batch ingest; low QPS | All component interfaces |
| Stage 1 | 100K | Move vector index to a real vector store; verify hybrid fusion and rerank latency hold | Read path logic, API contract |
| Stage 2 | 1M | Introduce int8 quantization; tune candidate set sizes; begin load testing toward higher QPS; turn on the semantic cache under realistic query mix to measure true hit rate | Component boundaries, faithfulness and abstain logic |
| Stage 3 | 10M | Shard the quantized vector index across nodes; convert batch ingest to the change driven near real time pipeline (FR10) with the cheap metadata only versus expensive re embed split; add alias based blue/green for embedding model upgrades; meet the full latency, throughput, and availability SLAs | The same retrieval, rerank, generate, faithfulness, and cache components |

Two scale specific mechanisms are introduced only when justified:

- **Near real time ingestion (FR10)** replaces batch at Stage 3, driven by a change event stream, separating ACL or metadata only updates (rewrite chunk metadata, skip re embedding) from content changes (re chunk and re embed).
- **Alias based deployment** is for whole index rebuilds, specifically embedding model or chunking strategy upgrades, where every vector must be recomputed. Build a new index in the background, warm it, then atomically repoint the alias for zero downtime. It is explicitly not used for routine incremental updates, which the live upsert path already handles.

---

## 11. Success metrics

### 11.1 Quality (the bar that matters most)

- **Faithfulness rate:** of non abstained answers, the fraction fully supported by cited chunks. Target: very high, measured on a labeled set; any unsupported answer is a defect to drive to zero.
- **Abstention correctness:** when the corpus genuinely lacks the answer, the service abstains rather than fabricating. Measured with deliberately unanswerable queries.
- **Retrieval quality:** recall at k and a precision measure on a labeled query set, comparing vector only, hybrid, and hybrid plus rerank. Target: each stage improves on the last.
- **Citation validity:** every citation points to a chunk that actually supports the claim. Target: no dangling or irrelevant citations.

### 11.2 Operational (gain weight at scale)

- Retrieval and end to end latency against the per phase targets in section 6.
- Semantic cache hit rate, measured under a realistic query mix once the cache is live. This is the flagged top risk and must be validated against real traffic, not assumed.
- Ingestion freshness once the near real time pipeline lands.

### 11.3 Cross context safety (correctness, not optional)

A standing test: a query answered for access context A must never have its cached answer served to access context B when B lacks A's permissions. This test runs from M4 onward and must always pass.

---

## 12. Risks and open questions

| Risk / question | Impact | Current stance |
|-----------------|--------|----------------|
| Cache hit rate may be far below 40 to 60% | Doubles the cold path fleet at scale; drives cost and latency | Treat as top risk; measure early under real query mix; do not assume |
| Synthetic ACLs do not capture real permission complexity | MVP ACL logic may need rework when a real permission model arrives | Acceptable for MVP; design the key and pre filter to accept a real model later |
| Faithfulness scoring adds latency on the hot path | May pressure the sub second SLA at scale | Tune the check; decide the latency versus safety tradeoff explicitly in M3 |
| Chunking strategy strongly affects retrieval quality | Poor chunking caps the quality ceiling regardless of rerank | Treat chunking as a tunable parameter; revisit if quality stalls |
| Embedding model choice and dimension affect index size and recall | Wrong choice is expensive to undo at scale | Pin the model and dimension early; record it per chunk so re embedding is a planned migration |
| Open: which vector store and which embedding and cross encoder models | Foundational choices | Decide during M0 and M2; keep interfaces model agnostic |

---

## 13. Out of scope (explicit)

- A user facing UI.
- Real multi tenant permission integration (synthetic only in MVP).
- Multilingual retrieval (English split only).
- Fine tuning any model.
- The full 1000+ QPS load (a roadmap target, not an MVP one).
```
