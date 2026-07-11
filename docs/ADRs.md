# ADRs — Grounded RAG over Wikipedia

Each ADR: context → decision → alternatives → consequences. Reference by ID
(e.g. "enforces `ADR-008`"). IDs are stable once assigned — append, don't
renumber. See [README.md](../README.md) for the governing principle every
contested decision below resolves against.

## Index

| ID | Title | Status |
|---|---|---|
| [ADR-001](#adr-001) | LangGraph for read-path orchestration | Accepted |
| [ADR-002](#adr-002) | OpenAI `text-embedding-3-small` for dense retrieval | Accepted |
| [ADR-003](#adr-003) | Cohere Rerank API for cross-encoder reranking | Accepted |
| [ADR-004](#adr-004) | Qdrant as the single hybrid search engine | Accepted |
| [ADR-005](#adr-005) | ACL-aware semantic cache key | Accepted |
| [ADR-006](#adr-006) | LLM-as-judge for the faithfulness check | Accepted |
| [ADR-007](#adr-007) | Provider-agnostic LLM access, no hard-coded default | Accepted |
| [ADR-008](#adr-008) | ACL/metadata pre-filter happens before retrieval, not after | Accepted |
| [ADR-009](#adr-009) | Local-first deployment, cloud-ready scale path | Accepted |
| [ADR-010](#adr-010) | Tiered (deterministic + LLM) context-sufficiency gate before generation | Accepted |
| [ADR-011](#adr-011) | Query rewriting: single LLM call, decompose only genuinely independent sub-questions | Accepted |
| [ADR-012](#adr-012) | Parallel tool calls with per-call partial-failure handling | Accepted |

---

<a id="adr-001"></a>
## ADR-001 — LangGraph for read-path orchestration

**Context:** The read path needs cycles — the generator calling retrieval a
second time as a tool (FR8), and later query decomposition (FR11) and
parallel tool calls with partial-failure handling (FR12) — plus explicit,
inspectable conditional edges so retrieval, rerank, generation, and the
faithfulness/abstain decision stay distinct steps rather than collapsing into
one large prompt.

**Decision:** Use LangGraph for the read-path orchestration: a graph with
explicit nodes for retrieval, rerank, generation (with a bound retrieval
tool), and faithfulness/abstain routing.

**Alternatives:**
- *Hand-rolled Python control flow, no framework* — full control, but the
  retry/rewrite loops FR8, FR11, and FR12 call for would need bespoke state
  machines built and tested from scratch, duplicating what LangGraph already
  provides.
- *LlamaIndex* — RAG-native, strong batteries for retrievers and query
  engines, but weaker native support for the agentic tool-loop and
  parallel-tool-call-with-partial-failure shape FR12 needs, and abstracts
  away exactly the retrieval/rerank/fusion decisions this project exists to
  make explicitly.

**Consequences:**
- `+` Native cycles and conditional edges make the retrieval-tool loop, and
  later the query-rewrite and parallel-tool-call loops, explicit nodes
  instead of hidden control flow inside one generation call.
- `+` A consistent state object threads through every node, which FR13's
  per-request trace can hang off directly.
- `−` Another framework's abstraction on top of the techniques themselves —
  accepted because the cycles (especially FR12's partial-failure handling)
  are unavoidable regardless of framework, and LangGraph's are explicit
  rather than implicit.

---

<a id="adr-002"></a>
## ADR-002 — OpenAI `text-embedding-3-small` for dense retrieval

**Context:** FR2's dense leg needs a bi-encoder embedding model. The choice
locks in the vector index's dimension and format for the project's life —
re-embedding a 10M-document corpus later is exactly the kind of migration
the architecture should avoid forcing on a whim.

**Decision:** Pin OpenAI `text-embedding-3-small`, 1536 dimensions, for every
embedding call — ingestion and query-time alike.

**Alternatives:**
- *Open-weight, self-hosted (e.g. `BAAI/bge-base-en-v1.5` via
  sentence-transformers)* — no per-call API cost, runs fully locally,
  matches the local-first deployment posture (`ADR-009`). Rejected for the
  MVP: it requires hosting and serving an embedding model (sizing, batching)
  on top of everything else being built, when a hosted call removes that
  surface entirely at a 1K-document MVP's call volume.
- *OpenAI `text-embedding-3-large`* — 3072 dims, higher recall ceiling.
  Rejected for the MVP: the 1K-document corpus doesn't need the extra
  headroom, and committing to it roughly doubles the vector index footprint
  at every later scale stage for a quality gain this project can't yet show
  is necessary.

**Consequences:**
- `+` Zero embedding-model hosting/serving infrastructure — ingestion and
  query-time embedding are both a single API call.
- `+` 1536 dims is a known, moderate footprint, sized concretely in
  [REQUIREMENTS.md](REQUIREMENTS.md)'s capacity math.
- `−` Every embedding call now depends on OpenAI's API availability and
  pricing; an outage degrades the entire read path, not just generation.
  Accepted because the deployment is local-first/low-volume at MVP scale,
  and generation already carries an equivalent dependency (`ADR-007`).
- `−` Drops the previously-considered `Upstash/wikipedia-2024-06-bge-m3`
  pre-embedded fallback for scale load testing — it doesn't share this
  vector space. A future load test has to actually pay to re-embed a larger
  slice with `text-embedding-3-small`.

---

<a id="adr-003"></a>
## ADR-003 — Cohere Rerank API for cross-encoder reranking

**Context:** FR4 needs a cross-encoder rerank stage over the fused candidate
set. OpenAI has no dedicated rerank endpoint, so the real choice is between a
third-party hosted rerank API and self-hosting a small open-weight
cross-encoder.

**Decision:** Use Cohere's Rerank API (`rerank-v3.5`).

**Alternatives:**
- *Self-hosted `BAAI/bge-reranker-v2-m3`* (or the lighter
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) via sentence-transformers — free
  beyond compute, runs fine on CPU at MVP candidate-set sizes (~20-50 chunks
  per query). Rejected for the MVP to avoid standing up and maintaining a
  model-serving process for one narrow function when a hosted call removes
  that surface; revisit if call volume makes per-call cost the dominant
  operating expense.
- *Voyage AI `rerank-2-lite`* — token-priced, often cheaper than Cohere for
  small payloads, but newer and less documented in production RAG writeups.
  Rejected in favor of the more established option for a first build.

**Consequences:**
- `+` No reranker model hosting — one REST call per query, the same
  hosting-avoidance reasoning as `ADR-002`.
- `+` Cohere's free tier and ~$2/1,000-search-unit pricing comfortably covers
  MVP-scale eval and demo traffic.
- `−` A third external dependency (alongside the embedding provider and
  whichever LLM is configured) on the hot path; an outage here specifically
  degrades FR4, the project's biggest differentiator over plain vector
  search. Mitigation: fusion-only (pre-rerank) results are still a valid,
  lower-precision fallback — see [ARCHITECTURE.md §Failure
  modes](ARCHITECTURE.md#failure-modes--degradation).
- `−` At real scale (10M docs / 1000+ QPS) per-call rerank pricing becomes a
  real budget line, not a rounding error; the self-hosted alternative above
  is the documented escape hatch if that math stops working.

**Addendum (M1):** the qualitative "~20-50 chunks per query" candidate-set
range above is pinned to `RETRIEVE_CANDIDATE_K = 30` in `config.py` — the
size `retrieve` returns and `rerank` will consume.

---

<a id="adr-004"></a>
## ADR-004 — Qdrant as the single hybrid search engine

**Context:** FR2 needs both dense vector search and sparse keyword
(BM25-style) search with score fusion. This was the PRD's original open
question ("which vector store"). The real choice is between one engine that
natively supports both representations, and two specialized stores kept in
sync.

**Decision:** Use Qdrant as the single store for both the dense embedding
and a sparse vector per chunk, queried via Qdrant's native hybrid query API
(fusion, e.g. RRF) in one round trip.

**Alternatives:**
- *Elasticsearch / OpenSearch* — also native hybrid (dense + BM25) in one
  engine, and the most battle-tested option at the 10M-doc / 1000+ QPS scale
  target. Rejected for the MVP: JVM-based and meaningfully heavier to stand
  up locally than Qdrant for a first milestone; Qdrant's scale story (managed
  cloud or self-hosted cluster, built-in int8 quantization) is sufficient
  for this project's roadmap.
- *Separate stores* — FAISS or Chroma for dense, `rank_bm25` or Whoosh for
  sparse, fused in application code. Lightest-weight, pure in-process Python
  for an MVP. Rejected: two indexes to keep consistent on every
  ingest/update, and a harder migration story to a "real" store at Stage 1 of
  the roadmap than Qdrant already being that store from day one.

**Consequences:**
- `+` One client, one collection, one query for hybrid retrieval — FR2's
  fusion logic is a single API call's parameter, not application-level merge
  code across two systems.
- `+` Built-in int8 quantization and a real managed-cloud/self-hosted-cluster
  path mean the MVP's storage layer doesn't need replacing at Stage 1 of the
  scale roadmap, only reconfiguring.
- `−` Qdrant's sparse-vector hybrid search is a newer feature than
  Elasticsearch's decades-old BM25 implementation; if sparse-retrieval
  quality proves weaker than a dedicated BM25 engine on this corpus, that's a
  real risk to validate against the eval set in M1, not assume away.
- `−` Locks ACL/metadata pre-filtering (FR3) to Qdrant's payload-filtering
  semantics; [DATA-MODEL.md](DATA-MODEL.md) is written against those
  semantics specifically.

**Addendum (M1):** fusion is pinned to **RRF** (`Fusion.RRF`), not DBSF —
Qdrant's Query API requires an explicit choice and RRF is the standard,
well-documented default for combining dense + sparse legs.

---

<a id="adr-005"></a>
## ADR-005 — ACL-aware semantic cache key

**Context:** FR9 needs a semantic cache so repeated/similar queries skip the
full retrieve-rerank-generate-faithfulness path. A naive cache keyed only on
query-embedding similarity would let an answer cached for one caller's access
context be served to a different caller whose permissions don't cover the
documents that answer was grounded in — a direct violation of the governing
principle ([README.md](../README.md#governing-principle-verifiable-or-abstain)).

**Decision:** Key the semantic cache on `(semantic cluster, tenant,
acl_signature)`, where `acl_signature` is a stable hash of the caller's
`access_context.groups`. A lookup or write-through only ever matches within
the same `acl_signature`.

**Alternatives:**
- *Key on query/answer similarity alone, ACL checked as a post-filter before
  returning the cached answer* — simpler key, but a cache *write* still
  stores an answer reachable across access boundaries internally, relying
  entirely on the read-side check never having a bug. Rejected: the
  governing principle calls for the boundary to be structural, not a single
  conditional that has to stay correct on every read path forever.
- *No semantic cache, exact-string query caching only* — avoids the
  ACL-signature design problem entirely, but loses most of the cache hit
  rate FR9 exists to capture (paraphrased repeat queries are common and
  exact-match misses them).

**Consequences:**
- `+` Cache leakage across access boundaries is structurally prevented by the
  key shape, not by a downstream filter that could be skipped or buggy.
- `+` Directly testable: UC-7 ([PRD.md
  §4.2](PRD.md#42-core-use-cases--illustrative-eval-set)) is a standing
  regression test that two different `access_context` values against the
  same query text never collide on a cache hit.
- `−` Narrows cache reuse versus a query-similarity-only key — two callers
  asking the identical question under different ACL groups never share a
  cache entry, even when both are individually permitted to see the
  answer's underlying chunks. This is the single biggest lever on cache hit
  rate at scale ([PRD.md §12](PRD.md#12-risks-and-open-questions)) and is
  accepted as the cost of the safety guarantee.

---

<a id="adr-006"></a>
## ADR-006 — LLM-as-judge for the faithfulness check

**Context:** FR6 needs a check that scores whether a generated answer is
actually supported by the retrieved chunks, gating the abstain decision
(FR7). The check has to produce a graded "how much of this answer is
supported" judgment, not just binary entailment, and has to evaluate each
citation in context of the specific claim it's attached to.

**Decision:** Implement the faithfulness check as a structured-rubric prompt
to the same provider-agnostic LLM used for generation (`ADR-007`), scoring
each cited claim against its cited chunk and producing a pass/fail plus a
confidence value.

**Alternatives:**
- *A fine-tuned NLI/entailment model* — faster, cheaper, more deterministic
  per call, removes a second LLM call from the hot path. Rejected for the
  MVP: entailment models give a binary-ish signal per claim and don't
  naturally produce the graded `confidence` value FR7's response shape
  needs, and adding a model-serving dependency for one narrow check
  duplicates the hosting-avoidance reasoning already applied in `ADR-002`/
  `ADR-003`.
- *Hybrid: deterministic citation-presence check, then an LLM judge only for
  whether the citation actually supports the claim* — strictly more robust,
  catches "claim with no citation at all" cheaply before spending an LLM
  call. Deferred, not rejected: worth adding if the pure LLM-judge version
  proves unreliable in practice — flagged in
  [AI-ARCHITECTURE.md](AI-ARCHITECTURE.md) as an open assumption.

**Consequences:**
- `+` Single mechanism covers both the binary abstain decision and the
  graded `confidence` field in the API response, with no second model to
  host.
- `+` Reuses the same provider-agnostic LLM access path as generation — no
  new infrastructure, just a second call with a different prompt.
- `−` A second LLM call on every non-cached request adds real latency to the
  hot path ([PRD.md §12](PRD.md#12-risks-and-open-questions)) and is the
  first thing to tune if a real sub-second SLA is ever pursued.
- `−` An LLM judging its own (or a sibling call's) output is a weaker setup
  than an independent check; the deferred hybrid approach above is the
  documented mitigation if this proves too lenient or inconsistent on the
  eval set.

---

<a id="adr-007"></a>
## ADR-007 — Provider-agnostic LLM access, no hard-coded default

**Context:** Generation and the faithfulness judge (`ADR-006`) both need an
LLM. Hard-coding a single vendor SDK call into node code would lock the
entire read path to one provider's availability, pricing, and tool-calling
format.

**Decision:** Access the LLM through a provider-agnostic layer (LangChain's
`init_chat_model`, selected by config/env var — consistent with already
using LangGraph per `ADR-001`), with no specific provider or model pinned as
a default. The operator sets the provider/model via environment
configuration; example snippets elsewhere in this suite use an illustrative
placeholder, never an implied default.

**Alternatives:**
- *Call a specific vendor's SDK directly* (e.g. the Anthropic or OpenAI SDK)
  — simplest, and a well-supported model's tool-use format is reliable.
  Rejected: hard-codes a single vendor into every node that calls the model,
  the exact coupling this decision exists to avoid for a project explicitly
  built to not assume one vendor's continued pricing or availability.
- *Route through a proxy* (e.g. OpenRouter) — single API key, swap models via
  a model-string config change, no need to hold a key per provider. Rejected
  in favor of LangChain's native abstraction, to avoid adding a third-party
  proxy's uptime and per-token markup on top of the model provider itself,
  given LangGraph (`ADR-001`) is already in the stack and its
  `init_chat_model` achieves the same "swap via config" outcome without it.

**Consequences:**
- `+` Swapping the underlying model for generation or for the
  faithfulness judge (independently, if ever desired) is a config change,
  not a code change.
- `+` No dependency on a proxy's uptime, pricing, or model catalog beyond
  whichever provider is actually configured.
- `−` Requires holding a separate API key per provider actually used, instead
  of one proxy key.
- `−` Tool-calling reliability (FR8) and faithfulness-judge consistency
  (`ADR-006`) aren't guaranteed identical across providers/models; since
  there's no pinned default, the eval set
  ([PRD.md §4.2](PRD.md#42-core-use-cases--illustrative-eval-set)) has to be
  re-run against whichever model is actually configured before trusting
  results, every time the configuration changes.

---

<a id="adr-008"></a>
## ADR-008 — ACL/metadata pre-filter happens before retrieval, not after

**Context:** Permission and metadata filtering (FR3) could apply at two
different points: as a filter inside the retrieval query itself, or as a
post-filter stripping disallowed chunks out of an unfiltered candidate set
after the fact.

**Decision:** Denormalize `doc_type` and `acl_tags` onto every chunk's
Qdrant payload at ingestion time, and apply them as a payload filter inside
the retrieval query — disallowed or out-of-scope chunks never enter the
candidate set in the first place.

**Alternatives:**
- *Post-filter after an unfiltered top-k retrieval* — simpler retrieval
  query, filtering logic lives in one place downstream. Rejected: spends
  rerank and generation budget on chunks that get discarded anyway, and more
  importantly violates the governing principle's "permission has to bound
  what evidence ever reaches the generator" stance
  ([README.md](../README.md#governing-principle-verifiable-or-abstain)) — a bug
  in the post-filter step would let disallowed content reach the generator
  even if it never reaches the final answer.
- *Join against a separate permissions table at query time* instead of
  denormalizing onto each chunk — avoids duplicating ACL data across every
  chunk of a document. Rejected for the MVP: a join adds a second round trip
  (or a second system) to every retrieval call, against a corpus and
  permission model simple enough that denormalization's storage cost is
  negligible.

**Consequences:**
- `+` Pre-filtering is structural — the candidate set is correct by
  construction, not by a downstream step remembering to filter.
- `+` Rerank (`ADR-003`, paid per call) and generation only ever spend budget
  on chunks the caller is actually permitted to see.
- `−` Any ACL change for a document requires rewriting the `acl_tags` payload
  field on every one of its chunks, not a single row in a permissions table
  — accepted as cheap at this corpus size; revisit if a real permission
  model ([PRD.md §4](PRD.md#4-personas), tenant-aware caller persona) makes
  ACL changes frequent enough for this to matter.

---

<a id="adr-009"></a>
## ADR-009 — Local-first deployment, cloud-ready scale path

**Context:** The MVP needs to actually run somewhere to be built and
evaluated. The 10M roadmap ([PRD.md §10](PRD.md#10-roadmap-from-1k-to-10m))
needs the same components to plausibly run as a real deployed service later,
without implying the MVP itself gets deployed that way.

**Decision:** Run the MVP as a local single-process service (the API
process, talking to a local or hosted-free-tier Qdrant instance) for all of
M0–M6. Every component in the locked stack (`ADR-002` through `ADR-004`)
already has a managed-cloud or containerized deployment path, so "deploy
this for real" later is a configuration and infrastructure change, not a
rewrite.

**Alternatives:**
- *Containerize and deploy from day one* (e.g. to a cloud VM or managed
  container service) — proves the deployment path early. Rejected for the
  MVP: this project's actual goal is depth on the read-path techniques
  ([PRD.md §1](PRD.md#1-summary)), and standing up real cloud infrastructure
  for a single-learner, low-volume build adds operational surface that
  doesn't serve that goal.

**Consequences:**
- `+` Zero cloud spend and zero deployment-pipeline work for the MVP; all
  effort goes toward the six techniques the project exists to practice.
- `+` Every component was chosen specifically because it has a credible
  managed/cloud path, so this decision doesn't paint the project into a
  corner if a deployed version is ever wanted.
- `−` Never actually validates the deployed/containerized path for real —
  "cloud-ready" stays a design claim until M0–M6 are done and someone
  actually does it. Accepted because deployment validation isn't this
  project's measured outcome.

---

<a id="adr-010"></a>
## ADR-010 — Tiered (deterministic + LLM) context-sufficiency gate before generation

**Context:** M3 (`ADR-006`) catches ungrounded answers *after* generation
already ran — the judge scores whatever the model drafted against the
chunks it was given. That leaves a gap the faithfulness check structurally
cannot close: whether the retrieved context was ever adequate to answer the
question in the first place is left entirely to the generator's own
self-assessment (its system prompt asks it to call the retrieval tool or
admit insufficiency, but nothing independently checks that judgment). A
model that's overconfident about thin context still gets a full generation
pass — wasted cost when the context was obviously hopeless (FR15).

**Decision:** Add `check_sufficiency`, a node between `rerank` and
`generate` that judges the retrieved chunks against the question alone (no
answer involved) and short-circuits straight to an abstained response when
they're clearly inadequate — skipping `generate` and `faithfulness`
entirely for that request. Implemented as three tiers, cheapest first: (1)
zero chunks retrieved, (2) a deterministic score-gate on Cohere's
`relevance_score` — comfortably below `SUFFICIENCY_LOW_SCORE_THRESHOLD` is
insufficient, comfortably above `SUFFICIENCY_HIGH_SCORE_THRESHOLD` is
sufficient, both skip the LLM — and only (3) genuinely ambiguous cases (or
a degraded rerank whose fusion-only scores aren't on the same 0-1 scale,
`FR-3.2`) spend one LLM call, structured-output scored, on a dedicated
judge prompt.

**Alternatives:**
- *LLM judge on every request, no score-gate tier* — simpler, one code
  path. Rejected: spends an LLM call on requests a cheap score check
  already resolves unambiguously (the obviously-hopeless and
  obviously-fine cases), working against `AI-ARCHITECTURE.md`'s
  cost-consciousness and the "at most two LLM calls" framing more than
  necessary — the same reasoning `ADR-006`'s zero-citation short-circuit
  and this project's other deterministic-first patterns already establish.
- *No independent check — trust the generator's self-assessment plus the
  downstream faithfulness gate* — zero new mechanism, faithfulness already
  catches the resulting bad answers. Rejected: it catches them *after*
  paying for a full generation call every time, and conflates two distinct
  failure populations (context was hopeless vs. context was fine but
  generation still erred) into one abstain reason, which weakens
  faithfulness as a diagnostic signal for M3's own quality bar (NFR-7).
- *Fold sufficiency into the faithfulness judge's own call, post-generation*
  — one fewer node, reuses the existing call. Rejected: faithfulness scores
  an answer that's already been drafted; a sufficiency judgment folded in
  there can no longer prevent the wasted generation call, defeating the
  point of gating *before* generation.

**Consequences:**
- `+` Requests with obviously-hopeless context (e.g. UC-5-style,
  off-corpus queries) skip `generate` and `faithfulness` entirely — a real
  latency/cost win, not just a correctness one, verified in M3's own eval
  set once a case's chunk scores fall below the low threshold.
- `+` Keeps faithfulness's failure signal more diagnostically pure from
  here on: a faithfulness abstain now more reliably means "context looked
  adequate but generation still erred," not "there was nothing to work
  with in the first place" — the latter is caught upstream instead.
- `+` Fails open on its own LLM call failing (mirrors `FR-3.2`'s Cohere
  fallback) — a transient sufficiency-judge outage degrades to "proceed to
  generation," never to blocking a request that might otherwise succeed;
  faithfulness remains the actual safety-critical gate regardless of
  whether sufficiency ran cleanly.
- `−` A third potential LLM call on the hot path for ambiguous-context
  requests, on top of `generate` and `faithfulness` — real added latency
  for the tier this call actually fires on, accepted because it's bounded
  to the ambiguous band by the score-gate tiers rather than firing on
  every request.
- `−` The score-gate thresholds (`SUFFICIENCY_LOW_SCORE_THRESHOLD = 0.2`,
  `SUFFICIENCY_HIGH_SCORE_THRESHOLD = 0.6`) are placeholders picked from a
  handful of observed cases, not a calibrated distribution — flagged in
  [REQUIREMENTS.md](REQUIREMENTS.md#open-assumptions) pending real
  measurement, same treatment as `FR-5.1`'s confidence threshold.
- `−` Only gates the *first* retrieval pass, not the state after a tool
  call fires (`FR8`) — a second insufficiency after the one extra tool
  round (`TOOL_CALL_MAX_ROUNDS`) is left to faithfulness to catch, since a
  second sufficiency gate there would just be a second abstain point
  competing with the tool-call bound's own termination logic.

---

<a id="adr-011"></a>
## ADR-011 — Query rewriting: single LLM call, decompose only genuinely independent sub-questions

**Context:** FR11 asks for three distinct techniques ahead of retrieval —
decontextualize, expand, decompose — over a request contract
([API-CONTRACTS.md](API-CONTRACTS.md)) that carries no conversation history
(each `POST /query` is a single, self-contained turn). Decontextualization's
usual job — resolving a reference to an earlier turn — has nothing to
resolve against in the MVP; it's kept as part of the rewrite prompt's job
description for forward compatibility with a future multi-turn contract, but
isn't meaningfully exercised by anything this MVP can test. That gap is
recorded honestly in [REQUIREMENTS.md](REQUIREMENTS.md#open-assumptions)
rather than built around.

Decompose is the technique with real teeth, and also the one that can go
wrong silently: `UC-4` ("Alexander II of Scotland had a son who later
became king. How did that son die?") is multi-hop, but its second hop's
subject is only known *after* the first hop's chunk is read — there is no
sub-query for "how did that son die" that doesn't already require the
answer. Naively decomposing every multi-part-looking question into parallel
sub-queries would silently break M3's `UC-4` regression test (`FR-4.2`
expects the reactive retrieval-tool call to still fire exactly once) by
routing that case's second hop through a fabricated, entity-less sub-query
that the retrieval index simply won't match.

**Decision:** One `rewrite_query` node, inserted after `cache_lookup` on a
cache **miss** (not before — keeping the cache key on the literal query text
means a verbatim-repeat lookup, `UC-6`, isn't perturbed by a non-deterministic
LLM rewrite of it) and before `retrieve`. One structured-output LLM call
(reusing `deps.generation_llm` — a generative rewrite, not a judge, so it
takes the generation role rather than `check_sufficiency`'s judge role)
produces:
- `rewritten_query`: a decontextualized/expanded standalone version of the
  question, used in place of the raw query for the first-pass retrieval.
- `sub_queries`: zero or more **independently retrievable** sub-questions,
  capped at `QUERY_REWRITE_MAX_SUB_QUERIES`. The prompt explicitly instructs
  the model to leave this empty whenever a later part depends on an entity
  or fact that can only be discovered by first retrieving an earlier part —
  that class of multi-hop keeps going through `FR8`'s reactive tool-call
  loop unchanged, not through this node.

`retrieve` then fires `rewritten_query` and every `sub_queries` entry
concurrently (`ADR-012`'s partial-failure handling applies to the
sub-queries specifically — see that ADR), merging the results by
`chunk_id` (max score wins on a collision) into one candidate set before
rerank, so a genuinely independent bundled question (e.g. "what is
`[Article-A]` known for, and what is `[Article-B]` known for") gets both
articles' chunks in the first pass instead of one fused, potentially
diluted embedding search.

**Alternatives:**
- *Always decompose multi-part-looking questions into sub-queries* —
  simpler prompt, no independence judgment call. Rejected: breaks `UC-4` as
  described above, and more generally conflates "the question has multiple
  clauses" with "the clauses are independently searchable," which they are
  not for genuinely sequential multi-hop.
- *Three separate LLM calls (decontextualize, expand, decompose)* — cleaner
  single-responsibility prompts. Rejected: three round-trips on every cache
  miss for a P1 technique the MVP eval set exercises lightly is a latency
  and cost multiplier without an accuracy case to justify it here; revisit
  if any one sub-task's quality needs isolated tuning.
- *Run rewrite before `cache_lookup`, cache on the rewritten query* — could
  raise the semantic cache's hit rate (paraphrases collapse to the same
  rewritten form) — the more ambitious framing of "rewriting helps caching
  too." Rejected for the MVP: makes `UC-6`/`UC-7`'s cache-safety tests
  depend on rewrite-LLM determinism, which isn't guaranteed run to run;
  revisit once the cache's real hit rate is measured against real traffic
  (`REQUIREMENTS.md` Open assumptions already flags that measurement as
  outstanding).

**Consequences:**
- `+` Genuinely independent bundled questions get both/all parts' evidence
  in the first-pass candidate set, without needing `FR8`'s reactive loop to
  discover it — cheaper (one round trip, not two) for the class of
  multi-part question decompose is actually suited to.
- `+` `UC-4`-style sequential multi-hop is unaffected: the rewrite prompt's
  independence judgment routes it to `sub_queries: []`, so `FR8`'s tool-call
  path still carries that case exactly as M3 built it — verified by
  re-running `eval_m3.py` unchanged after this node was wired in.
- `+` Fails open on its own LLM call failing (mirrors `ADR-010`'s tier-2
  fallback): `rewritten_query` defaults to the raw query and `sub_queries`
  to `[]`, never blocking retrieval on a transient rewrite-LLM outage.
- `−` A fourth potential LLM call on the hot path (after `check_sufficiency`,
  `generate`, `faithfulness`), on every cache miss unconditionally (not
  gated to an ambiguous tier the way `ADR-010`'s is) — accepted because
  rewriting has to run before retrieval even happens, so there's no
  retrieval-quality signal yet to gate it on the way `ADR-010` gates its own
  LLM call.
- `−` Decontextualization is effectively unexercised by this MVP's
  single-turn contract — flagged, not hidden, in
  [REQUIREMENTS.md](REQUIREMENTS.md#open-assumptions).

---

<a id="adr-012"></a>
## ADR-012 — Parallel tool calls with per-call partial-failure handling

**Context:** `generate` (`ADR-001`, `ADR-007`) has bound `retrieve_chunks`
and `SubmitAnswer` with `parallel_tool_calls=False` since M3, deliberately —
the M3 code comment on that line spells out why: `execute_tool_node` only
ever answered `tool_calls[0]`, so a provider returning two calls in one
`AIMessage` would leave the second `tool_call_id` without a response
message, which OpenAI's API rejects on the next turn. FR12 asks for the
opposite: the generator should be able to fire more than one retrieval in a
single round (e.g. two independent follow-up lookups it decides it needs
mid-turn) and have a single one of them failing degrade gracefully rather
than take down the request.

**Decision:** Flip `parallel_tool_calls=True` on the `generate` binding.
`GenerationResult.tool_calls` becomes a list (`ToolCallRequest`, one per
`retrieve_chunks` call in the round; capped at
`MAX_PARALLEL_RETRIEVE_CALLS`), replacing the old single `tool_query`/
`tool_top_k` fields — a round is still exactly one increment of
`tool_call_count` against `TOOL_CALL_MAX_ROUNDS` regardless of how many
calls it contains, since the round-count bound (`FR-4.2`) and the
within-round call count are orthogonal knobs. If any call in the round is
`SubmitAnswer`, that one wins and the round finishes immediately — the
generator is never supposed to mix "I'm answering" with "I need more
evidence" in one turn, so the other calls (if any) are simply never invoked
and never get a `ToolMessage`, which is safe precisely because a finished
round means no further `llm.invoke` happens against that message history.

`execute_tool_node` runs every `retrieve_chunks` call in the round
concurrently (`ThreadPoolExecutor`) and always emits exactly one
`ToolMessage` per `tool_call_id`, win or lose: a call whose `retrieve()`
raises gets a `ToolMessage` reporting the failure in its content instead of
chunks, and the round proceeds with whatever the surviving calls returned —
this is the same "one call's failure doesn't obligate failing the whole
request" property `ADR-011`'s sub-query fan-out gets, applied to the
model-driven fan-out instead of the rewrite-driven one. This is scoped
narrowly: it does **not** change the hard-fail behavior of the single
first-pass `retrieve` call in `ARCHITECTURE.md`'s failure-mode table — that
call has no sibling to fall back on, so a real Qdrant/embedding outage there
is still a request failure, not a degradation.

**Alternatives:**
- *Keep `parallel_tool_calls=False`, let the model ask for one lookup per
  round and rely on `TOOL_CALL_MAX_ROUNDS` for multi-round follow-ups* — the
  status quo. Rejected as the whole point of FR12: it demonstrates no
  parallelism and has nothing for partial-failure handling to apply to.
- *Fail the whole round if any one call raises* — simplest error handling.
  Rejected: directly contradicts FR12's explicit ask ("a single failure
  degrades gracefully"), and throws away perfectly good chunks the
  surviving calls already fetched.
- *Retry a failed call once before giving up* — could paper over a
  transient blip. Rejected for the MVP: adds latency to the common case to
  handle a failure mode this project already treats as gracefully
  degradable rather than worth masking; revisit if real traffic shows
  transient single-call failures are common enough to matter.

**Consequences:**
- `+` A deliberately-failed call (verified by a unit test injecting an
  exception into one of two concurrent `retrieve_chunks` calls) never
  aborts the request — the surviving call's chunks still reach `generate`,
  and the model can still finish with `SubmitAnswer` on partial evidence,
  matching `PRD.md §9`'s M5 exit criterion verbatim.
- `+` Two independent follow-up lookups in one round cost one round-trip's
  wall-clock latency instead of two sequential ones (bounded by
  `MAX_PARALLEL_RETRIEVE_CALLS`, not open-ended).
- `−` `GenerationResult`'s shape change (`tool_query`/`tool_top_k` ->
  `tool_calls: list[ToolCallRequest]`) is a breaking change to every caller
  of `generate()` — contained to `graph/nodes.py` and the test suite, no
  external contract (`API-CONTRACTS.md`'s response shape is unaffected;
  this is internal to the agentic loop).
- `−` `execute_tool_node`'s dedup logic (a chunk already known to `state`
  should not be re-added) now also has to dedup *across* the concurrently
  merged calls of the same round, not just against prior rounds — handled
  by the same score-keeping merge helper `ADR-011`'s sub-query fan-out
  uses, rather than two divergent dedup implementations.
