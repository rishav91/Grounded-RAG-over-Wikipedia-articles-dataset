# Grounded RAG over Wikipedia

A personal build to deeply learn and demonstrate six production RAG
techniques — hybrid search, reranking, tool use, semantic caching, query
rewriting, and parallel tool calls — end to end, framed as an internal API
service over a real, public, checkable corpus (1,000 English Wikipedia
articles for the MVP) rather than synthetic text.

The single most important scope decision: **depth on the MVP's read path —
hybrid retrieval, cross-encoder reranking, grounded generation, faithfulness
checking, ACL-aware caching — over literally reaching the 10M-document scale
target.** The scale roadmap is a design exercise proving the architecture
holds up on paper; it is not a build commitment. See
[PRD.md §1](docs/PRD.md#1-summary).

## Governing principle: Verifiable-or-Abstain

> An answer must never leave the system unless every claim in it can be
> traced to a retrieved chunk the caller is permitted to see. If that
> traceability cannot be established, the system must abstain — never
> fabricate, and never ground an answer in evidence the caller's access
> context doesn't permit.

Every contested decision in [ADRs.md](docs/ADRs.md) resolves against this rule —
why the faithfulness check has no bypass ([ADR-006](docs/ADRs.md#adr-006)), why
ACL filtering happens before retrieval rather than after
([ADR-008](docs/ADRs.md#adr-008)), and why the semantic cache key includes the
caller's ACL signature ([ADR-005](docs/ADRs.md#adr-005)). Full reasoning in
[PRD.md §2.5](docs/PRD.md#25-governing-principle-verifiable-or-abstain).

| In scope (MVP) | Deferred (designed for, Phase 2+) | Excluded permanently |
|---|---|---|
| Hybrid search + metadata/ACL filtering, reranking, tool use, ACL-aware semantic caching, grounded generation, faithfulness + abstain — all over a 1,000-doc corpus | Query rewriting, parallel tool calls (FR11, FR12); near-real-time ingestion (FR10); observability + feedback (FR13, FR14) | A user-facing UI; real multi-tenant permissions (synthetic ACLs only); multilingual retrieval; fine-tuning any model; a production on-call posture |

Full table with reasons: [PRD.md §2.3](docs/PRD.md#23-non-goals-mvp).

## Locked stack / key constraints

| Layer | Choice | ADR |
|---|---|---|
| Orchestration | LangGraph | [ADR-001](docs/ADRs.md#adr-001) |
| Embeddings | OpenAI `text-embedding-3-small`, 1536 dims | [ADR-002](docs/ADRs.md#adr-002) |
| Reranker | Cohere Rerank API (`rerank-v3.5`) | [ADR-003](docs/ADRs.md#adr-003) |
| Search engine | Qdrant — single engine, native hybrid dense + sparse | [ADR-004](docs/ADRs.md#adr-004) |
| Semantic cache | A second Qdrant collection (`query_cache`), ACL-signature keyed | [ADR-005](docs/ADRs.md#adr-005) |
| Faithfulness check | LLM-as-judge, structured rubric prompt | [ADR-006](docs/ADRs.md#adr-006) |
| Generation LLM | Provider-agnostic via config (`init_chat_model`); no hard-coded default | [ADR-007](docs/ADRs.md#adr-007) |
| Deployment | Local single-process for the MVP; every component has a credible cloud/managed path | [ADR-009](docs/ADRs.md#adr-009) |

**Rejected, with reasons:** see [ADRs.md](docs/ADRs.md) — hand-rolled control
flow and LlamaIndex (`ADR-001`), self-hosted embeddings and the larger
OpenAI embedding model (`ADR-002`), self-hosted rerankers and Voyage AI
(`ADR-003`), Elasticsearch/OpenSearch and separate FAISS+BM25 stores
(`ADR-004`), a query-similarity-only cache key (`ADR-005`), an NLI
entailment model for faithfulness (`ADR-006`), direct vendor SDK calls and
an LLM proxy (`ADR-007`), ACL post-filtering (`ADR-008`), day-one cloud
deployment (`ADR-009`).

## Document map

| Doc | Purpose |
|---|---|
| [README.md](README.md) | This file — orientation, spine, reading order |
| [PRD.md](docs/PRD.md) | Why this exists, goals/non-goals, personas, the eval set, success criteria, risks |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | The component diagram, query/ingestion flows, failure modes, scale model |
| [ADRs.md](docs/ADRs.md) | The contested decisions: alternatives considered, why they lost |
| [AI-ARCHITECTURE.md](docs/AI-ARCHITECTURE.md) | Where each LLM/ML call earns its place, the deterministic/ML/LLM split, safety posture |
| [DATA-MODEL.md](docs/DATA-MODEL.md) | Chunk schema, Qdrant collection design, synthetic ACL-tag derivation |
| [API-CONTRACTS.md](docs/API-CONTRACTS.md) | The full `/query` request/response, abstain contract, retrieval tool schema, errors |
| [REQUIREMENTS.md](docs/REQUIREMENTS.md) | `FR-x.y`/`NFR-x.y` with acceptance criteria, capacity sizing math |
| [ROADMAP.md](docs/ROADMAP.md) | The phased build plan (M0–M6) and the 1K→10M scale-stage sequencing |

## Reading order

1. This file
2. [PRD.md](docs/PRD.md) — the why
3. [ARCHITECTURE.md](docs/ARCHITECTURE.md) — the how
4. [ADRs.md](docs/ADRs.md) — why not the alternatives
5. [AI-ARCHITECTURE.md](docs/AI-ARCHITECTURE.md) — the AI-specific cross-cutting concerns
6. [DATA-MODEL.md](docs/DATA-MODEL.md) — the schema everything else assumes
7. [API-CONTRACTS.md](docs/API-CONTRACTS.md) — the testable external contract
8. [REQUIREMENTS.md](docs/REQUIREMENTS.md) — the testable internal contract
9. [ROADMAP.md](docs/ROADMAP.md) — the build sequence

## Conventions

- `FR-x.y` functional requirements, `NFR-x.y` non-functional — both in
  [REQUIREMENTS.md](docs/REQUIREMENTS.md), grouped by area (`FR-1.x` ingestion,
  `FR-2.x` hybrid retrieval, `FR-3.x` reranking, `FR-4.x` generation/tools,
  `FR-5.x` faithfulness/abstain, `FR-6.x` caching, `FR-7.x` Phase 2,
  `FR-8.x` observability/feedback, `FR-9.x` near-real-time ingestion).
- `UC-1`..`UC-8` — the labeled eval set in
  [PRD.md §4.2](docs/PRD.md#42-core-use-cases--illustrative-eval-set); every
  acceptance criterion in REQUIREMENTS.md traces to one.
- `ADR-00N` — decisions, in [ADRs.md](docs/ADRs.md). Stable once assigned.
- **Non-negotiable:** nothing returns an answer without going through the
  faithfulness gate, and the semantic cache never crosses an
  `acl_signature` boundary (UC-7) — these two are standing regression tests,
  not one-time checks.
- Assumptions (numeric targets pinned ahead of real measurement) are marked
  inline as *Assumption* and listed in each doc's own assumptions section —
  treat them as debts to retire during the build, not settled facts.

## Project status

| Area | Status |
|---|---|
| Design docs | Complete — this suite |
| Implementation | Not started |
| Corpus | Wikipedia English split (`20231101.en`) via Hugging Face, not yet ingested |

## License and attribution

Wikipedia content is licensed under [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). When using or displaying Wikipedia-derived content, provide appropriate attribution to Wikimedia contributors.

Service code licensing TBD.
