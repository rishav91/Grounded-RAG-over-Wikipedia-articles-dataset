# Requirements — Grounded RAG over Wikipedia

`FR-x.y` = functional, `NFR-x.y` = non-functional. Acceptance criteria draw
on the labeled eval set (`UC-1`..`UC-8`) in
[PRD.md §4.2](PRD.md#42-core-use-cases--illustrative-eval-set).

## Functional requirements

### FR-1.x — Ingestion (FR1)

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-1.1 | Batch-ingest the 1,000-document slice: load, chunk, derive synthetic metadata/ACL, embed dense + sparse, upsert to Qdrant `articles` | P0 | Total chunk count falls in the 5,000–15,000 range (5–15 chunks/doc); querying `articles` directly with a payload filter (e.g. `doc_type=long AND acl_tags contains "public"`) returns only matching chunks — see [DATA-MODEL.md](DATA-MODEL.md) |
| FR-1.2 | Re-running ingestion upserts rather than duplicates | P0 | Running the ingestion script twice on the same slice leaves the chunk count unchanged ([ARCHITECTURE.md idempotency](ARCHITECTURE.md#cross-cutting)) |

### FR-2.x — Hybrid retrieval & filtering (FR2, FR3)

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-2.1 | `retrieve` executes a Qdrant hybrid query (dense + sparse fusion) for a query string | P0 | UC-1: the correct article's chunk appears in the top-k candidate set |
| FR-2.2 | Metadata/ACL filter (`doc_type`, `date_range`, `acl_tags`) applies inside the hybrid query, before fusion | P0 | UC-3: a chunk outside the filter never appears in the candidate set — verified against retrieval's raw output, not the final answer |
| FR-2.3 | Hybrid retrieval beats dense-only retrieval on the eval set | P0 | Recall@5 (hybrid) ≥ Recall@5 (dense-only) + 10 points across the eval set (*Assumption: placeholder margin pending the real measurement — see [Open assumptions](#open-assumptions)*) |

### FR-3.x — Reranking (FR4)

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-3.1 | `rerank` calls Cohere Rerank API (`rerank-v3.5`) over the candidate set, returns a precise top-k | P0 | UC-2: precision@3 (rerank) ≥ precision@3 (fusion-only) + 15 points on the ambiguous-candidate case (*Assumption — see [Open assumptions](#open-assumptions)*) |
| FR-3.2 | A Cohere API failure degrades to fusion-only ranking, never a failed request | P0 | Simulated Cohere failure (e.g. invalid key) still returns a `200` with fusion-ranked chunks — [ADR-003](ADRs.md#adr-003) |

### FR-4.x — Grounded generation & tool use (FR5, FR8, FR15)

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-4.1 | `generate` answers using only the provided top-k context, with inline citations | P0 | UC-1, UC-8: every claim has ≥1 citation; every citation's `chunk_id` is in this response's `retrieved_chunks` |
| FR-4.2 | `generate` can call the retrieval tool when the first pass is insufficient, with a query derived from the first pass's result | P0 | UC-4: exactly one additional tool call fires, with a refined (not repeated) query |
| FR-4.3 | The retrieval tool inherits the request's `access_context`/`filters`; the model cannot widen them | P0 | The tool schema doesn't expose `access_context`/`filters` as model-settable parameters — [API-CONTRACTS.md](API-CONTRACTS.md#the-retrieval-tool-fr8) |
| FR-4.4 | `check_sufficiency` gates generation on whether the retrieved context is adequate, independent of the generator's own self-assessment (FR15; `ADR-010`) | P0 | UC-5: a query with no relevant chunks (score below `SUFFICIENCY_LOW_SCORE_THRESHOLD`) short-circuits to an abstained response without `generate`/`faithfulness` ever running; a sufficiency-judge failure fails open (proceeds to `generate`), never blocks the request |
| FR-4.5 | Context recall: for a multi-hop query, the final retrieved set (first pass + tool call) contains every document actually needed to answer, not just the document the first pass happened to find | P0 | UC-4: the tool call's refined query resolves back to the case's labeled `second_hop_doc_id`, verified against the graph's final chunk set directly — a RAGAS-style context-recall check, scoped to the one case where recall genuinely can't be assessed by a single `expected_doc_id` (M1's FR-2.3 already covers single-hop recall) |

### FR-5.x — Faithfulness & abstain (FR6, FR7)

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-5.1 | `faithfulness` scores each cited claim against its cited chunk, produces pass/fail + confidence | P0 | UC-8: a well-grounded answer passes with `confidence ≥ 0.7` (*Assumption — threshold pending real measurement*) |
| FR-5.2 | A faithfulness fail converts the response into an explicit abstention | P0 | UC-5: a genuinely unanswerable query returns `abstained: true`, `answer: null`, `retrieved_chunks` still populated |
| FR-5.3 | An abstained response is never written to `query_cache` | P0 | Re-submitting UC-5's query+context twice returns `cache_hit: false` both times |
| FR-5.4 | `faithfulness` also scores answer relevance — whether the answer actually addresses the question — independent of citation support; either gate failing converts the response into an abstention | P0 | UC-8: `FaithfulnessJudgment.answers_question` is scored in the same judge call as claim support (no added LLM call); a faithful-but-off-topic answer (`passed=True, answers_question=False`) still abstains — a RAGAS-style "answer relevance" check, folded into the existing `ADR-006` judge rather than a separate mechanism |

### FR-6.x — ACL-aware semantic caching (FR9)

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-6.1 | `cache_lookup` checks `query_cache` filtered by `acl_signature` before running the full pipeline | P0 | UC-6: a second identical/paraphrased query under the same `access_context` returns `cache_hit: true` |
| FR-6.2 | Cache lookup never crosses an `acl_signature` boundary | P0 | UC-7: the same query text under two different `access_context` values never shares a cache hit — standing regression test from M4 onward |
| FR-6.3 | `response` writes through to `query_cache` only after a faithfulness pass | P0 | A passing response, repeated with `bypass_cache: false`, returns `cache_hit: true` on the second call |

### FR-7.x — Phase 2: query rewriting & parallel tool calls (FR11, FR12)

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-7.1 | Query rewriting decontextualizes/expands/decomposes multi-hop queries before retrieval | P1 | One LLM call (`rewrite_query` node, `ADR-011`), run on every cache miss before `retrieve`, produces a decontextualized/expanded `rewritten_query` plus zero or more `sub_queries` — populated only when the question decomposes into genuinely independently-retrievable parts, never a sequential hop whose subject is only known after an earlier hop resolves (that stays on FR8's reactive tool-call path). All sub-queries retrieve concurrently with `rewritten_query` and merge (score-deduped) into the first-pass candidate set. Verified live: a bundled independent-facts case (UC-9) recalls both parts' source chunks in the first pass, without the reactive tool call firing; M3's UC-4 (sequential multi-hop) re-verified unchanged (`sub_queries: []`, tool call still fires exactly once). |
| FR-7.2 | Parallel tool/retrieval calls with partial-failure handling — one failure degrades gracefully | P1 | `generate` binds tools with `parallel_tool_calls=True` (`ADR-012`); multiple `retrieve_chunks` calls in one round (capped at `MAX_PARALLEL_RETRIEVE_CALLS`) execute concurrently, still counted as one round against `TOOL_CALL_MAX_ROUNDS`. A call whose retrieval raises gets a failure `ToolMessage` on its own `tool_call_id` instead of aborting the round; the surviving calls' chunks still reach the model. Verified by a unit test injecting a failure into one of two concurrent calls: the surviving call's chunks are merged into state, no exception propagates, and the request completes. |

### FR-8.x — Observability & feedback (FR13, FR14)

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-8.1 | Per-request trace spans retrieve/rerank/generate/faithfulness | P1 | Deferred to M6 |
| FR-8.2 | Feedback ingestion (thumbs up/down) for offline eval | P2 | Deferred, no system requirement yet |

### FR-9.x — Near-real-time ingestion (FR10)

| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-9.1 | Document changes queryable within minutes, via a change-driven pipeline splitting metadata-only updates from expensive re-embeds | P1 | Deferred to Stage 3 of the scale roadmap ([ROADMAP.md](ROADMAP.md#scale-stages)) |

## Non-functional requirements

| ID | Requirement | MVP target | Scale target (10M docs, design exercise) |
|---|---|---|---|
| NFR-1 | Retrieval latency | p95 < 500 ms | p99 < 1 s, p50 < 200 ms |
| NFR-2 | End-to-end latency (cold, includes generation) | < 5 s | Time to first token < 800 ms (streamed) |
| NFR-3 | Throughput | Low, single-digit QPS | 1,000+ QPS sustained, ~3,000 peak |
| NFR-4 | Cache hit rate | N/A (low volume) | Planning assumption 40–60% — **unvalidated, top risk** ([PRD.md §12](PRD.md#12-risks-and-open-questions)) |
| NFR-5 | Availability | Best effort, single node | 99.9% retrieval, 99.5% generation |
| NFR-6 | Index freshness | Batch; minutes–hours acceptable | < 5 minutes, doc to queryable (FR10) |
| NFR-7 | Faithfulness rate (non-abstained answers fully supported) | ≥ 98% on the eval set (*Assumption — placeholder pending real measurement*) | Same bar; doesn't relax with scale |
| NFR-8 | Abstention correctness on deliberately unanswerable queries (UC-5 class) | 100% — zero fabricated answers tolerated | Same bar |
| NFR-9 | Citation validity | 100% — every citation resolves to a retrieved chunk | Same bar; enforced structurally per [API-CONTRACTS.md](API-CONTRACTS.md#response--grounded) |
| NFR-10 | Answer relevance rate (non-abstained answers that actually address the question asked) | ≥ 95% on the eval set (*Assumption — placeholder pending real measurement, same treatment as NFR-7*) | Same bar; doesn't relax with scale |

## Capacity sizing

Showing the math behind the scale-target row in [PRD.md
§6.2](PRD.md#62-scale-targets-10m-documents-roadmap--design-exercise-not-a-build-target),
using the actual pinned embedding model (OpenAI `text-embedding-3-small`,
1536 dims, [ADR-002](ADRs.md#adr-002)) rather than a generic placeholder.

### Storage at 10M documents (50M–150M chunks, 5–15 chunks/doc)

| Component | Per chunk | At 50M chunks (low) | At 100M chunks (mid) | At 150M chunks (high) |
|---|---|---|---|---|
| Dense vector, raw float32 | 1536 × 4 B = 6,144 B | ~307 GB | ~614 GB | ~922 GB |
| Dense vector, int8 quantized (~4x) | ~1,536 B | ~77 GB | ~154 GB | ~230 GB |
| Sparse vector (rough estimate, ~100 nonzero terms × 8 B) | ~800 B | ~40 GB | ~80 GB | ~120 GB |
| Payload — chunk text + metadata (~2.2 KB/chunk) | ~2,200 B | ~110 GB | ~220 GB | ~330 GB |
| **Total (quantized dense + sparse + payload)** | — | **~227 GB** | **~454 GB** | **~680 GB** |

*Assumption: sparse-vector and payload-text sizes are rough estimates pending
real corpus statistics from M0 — the actual figures depend on vocabulary
overlap and average chunk length, which aren't known until the real slice is
chunked.*

**The non-obvious result:** at this scale, **chunk text payload is the
single largest storage component** — larger than the quantized dense
vectors. A future optimization (not built, not needed at MVP scale) would
store chunk text in cheaper blob storage keyed by `chunk_id`, fetching it
only for the final top-k rather than keeping all 50–150M chunks' full text
resident in Qdrant's payload. Flagged here as a real implication of the
math, not a roadmap commitment.

With a replication factor of 2 (for the 99.9% availability target), double
the total above — roughly **450 GB to 1.4 TB**, comfortably shardable across
"a handful of nodes" per [PRD.md §10](PRD.md#10-roadmap-from-1k-to-10m).

### LLM call volume at 1,000+ QPS sustained

This is the number behind "cache hit rate is the single biggest lever"
([ARCHITECTURE.md §Scale](ARCHITECTURE.md#scale--capacity-model)), made
concrete:

- At the 40–60% planning assumption (`NFR-4`), 400–600 of every 1,000 QPS
  are cache hits (cheap: one embedding call + one Qdrant point lookup, **no
  LLM call**).
- The remaining 400–600 QPS run the full pipeline: one embedding call, one
  Qdrant hybrid query, one Cohere rerank call, and **at least two** LLM
  calls (`generate`, `faithfulness`) — three if a tool call fires.
- Sustained LLM call volume: **roughly 800–1,200 calls/sec** at 1,000 QPS
  sustained, **2,400–3,600 calls/sec** at the 3,000 QPS peak target.
- Every 10-percentage-point drop in cache hit rate adds ~100 QPS of
  full-pipeline load — ~200 additional LLM calls/sec. This is a real,
  currently-unbudgeted cost line at scale, not just a latency concern.

## P0 summary — the MVP

Every `FR-1.x` through `FR-6.x` requirement above is P0. `FR-7.x` (query
rewriting, parallel tool calls), `FR-8.x` (observability, feedback), and
`FR-9.x` (near-real-time ingestion) are P1/P2 — designed for, not built, in
the MVP. There is no P2 functional gap inside the MVP itself: M0–M4 in
[ROADMAP.md](ROADMAP.md) cover every P0 item end to end.

## Open assumptions

Numeric targets pinned here to keep the suite specific (per house
convention: no adjectival NFRs), flagged as placeholders to retire once
real measurement exists:

- **FR-2.3, FR-3.1 margins (10 and 15 points):** retired by real measurement.
  M1 (`scripts/eval_m1.py`): UC-1 recall@5, hybrid vs. dense-only, delta =
  0.0 points — below the +10-point margin (saturates at 1K-doc scale on
  distinctive named-entity queries). M2 (`scripts/eval_m2.py`): UC-2
  precision@3, rerank vs. fusion-only, delta = +11.1 points — below the
  +15-point margin but directionally positive, and one case (Apollo 13)
  showed a genuine +33-point gain, confirming rerank helps when the fused
  candidate set actually contains the right chunks in the wrong order; it
  cannot fix cases where the correct doc never entered the top-30 (that's
  `retrieve`'s job, not `rerank`'s). Neither number was massaged to hit the
  original placeholder — see [ROADMAP.md](ROADMAP.md#m1--hybrid-retrieval-plus-filtering-fr2-fr3)
  and [ROADMAP.md](ROADMAP.md#m2--reranking-fr4).
- **FR-5.1 confidence threshold (0.7):** M3 (`scripts/eval_m3.py`, UC-8, `gpt-4o-mini`
  for both `generate` and `faithfulness`): the judge's scores cluster at 0.9–1.0 on
  clearly-supported answers and drop to 0.5–0.6 on borderline cases (a citation that's
  topically relevant but only loosely entails the specific claim), so 0.7 sits in a
  real gap in the observed distribution rather than being retired — kept as-is, not
  retuned, pending a larger eval set than 3 cases.
- **NFR-7 faithfulness rate (98%):** M3 (`scripts/eval_m3.py`, UC-8, 3 cases, `gpt-4o-mini`):
  observed 66.7%–100% across repeated runs (2/3–3/3 passing), below the 98%
  placeholder — the one recurring abstain is a genuinely borderline case (a
  citation the judge sometimes scores as only loosely supporting its claim),
  not a fabrication (NFR-8, the zero-tolerance bar, held at 100% every run).
  Not massaged to hit the placeholder — see
  [ROADMAP.md](ROADMAP.md#m3--grounded-generation-citations-faithfulness-abstain-tool-use-fr5-fr6-fr7-fr8).
  A 3-case sample is too small to retire this number; revisit with a larger
  eval set before trusting it as a real ceiling.
- **NFR-10 answer relevance rate (95%):** M3 (`scripts/eval_m3.py`, UC-8, 3
  cases, `gpt-4o-mini`): 100% (3/3) across repeated runs, meeting the
  placeholder. In every run so far, the recurring faithfulness borderline
  case (Analytical Engine, see above) fails on citation *support*, never on
  relevance — the two axes are behaving as genuinely independent signals,
  not duplicating one failure mode. Still a 3-case sample; not retired as a
  real ceiling.
- **FR-4.4 sufficiency score thresholds (`SUFFICIENCY_LOW_SCORE_THRESHOLD = 0.2`,
  `SUFFICIENCY_HIGH_SCORE_THRESHOLD = 0.6`, `ADR-010`):** picked from a
  handful of observed cases (M3's UC-5/UC-8 chunk scores), not a calibrated
  distribution — same treatment as `FR-5.1`'s confidence threshold. M3
  (`scripts/eval_m3.py`, UC-5, 1 case): the pho query's chunk scores
  (0.05–0.10) land well below the 0.2 low threshold, consistently caught by
  the tier-1 score gate across repeated runs — `generate`/`faithfulness`
  never ran for this case, the cost win `ADR-010` is designed for. A single
  case can't validate the threshold itself, only confirm the mechanism
  fires; pending a larger eval set before trusting the exact cutoff.
- **Cache-hit similarity threshold (cosine ≥ 0.92, [DATA-MODEL.md](DATA-MODEL.md)):**
  M4 (`scripts/eval_m4.py`, UC-6, 1 case): an exact-text repeat query scored
  well above 0.92 (a verbatim repeat, not yet a paraphrase), so this run
  only confirms the mechanism fires correctly, not where the real
  false-hit/miss boundary sits — still pending real paraphrase data from a
  larger eval set before the exact cutoff is trusted. FR-6.2/UC-7 (cross-
  context safety) is a hard gate, not threshold-dependent, and held at
  100% (no shared hit across `access_context` values) — see
  [ROADMAP.md](ROADMAP.md#m4--semantic-caching-acl-aware-fr9).
- **NFR-4 cache hit rate (40–60% planning assumption):** still unmeasured —
  M4's eval only exercises the two structural UC-6/UC-7 cases (a verbatim
  repeat and a cross-context probe), not a realistic query mix. Remains the
  top risk flagged in [PRD.md §12](PRD.md#12-risks-and-open-questions);
  real measurement is a Stage 2 scale-roadmap item
  ([ROADMAP.md](ROADMAP.md#scale-stages)), not something the 1K-doc MVP's
  low query volume can produce.
- **FR-7.1 decontextualization is effectively unexercised by this MVP:**
  `POST /query`'s contract ([API-CONTRACTS.md](API-CONTRACTS.md)) carries no
  conversation history — every request is a single, self-contained turn —
  so `rewrite_query`'s decontextualization job (resolving a reference to an
  earlier turn) has nothing to resolve against. Kept in the rewrite prompt
  for forward compatibility with a future multi-turn contract (`ADR-011`),
  not something this MVP's eval set can meaningfully measure.
- **Capacity sizing's sparse-vector and payload-text estimates:** rough
  approximations pending real chunk statistics from M0 (see
  [Capacity sizing](#capacity-sizing) above).
