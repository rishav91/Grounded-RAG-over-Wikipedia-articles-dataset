# Grounded RAG over Wikipedia

An internal **Retrieval Augmented Generation (RAG)** service that answers natural-language questions over a document corpus with **grounded citations**, **confidence scores**, and an explicit **abstain** path when evidence is insufficient. Callers interact through a single API — no UI.

The MVP indexes **1,000 English Wikipedia articles** and proves the full read path: hybrid retrieval, cross-encoder reranking, grounded generation, faithfulness checking, and ACL-aware semantic caching. The architecture is designed to scale toward **10M documents** by swapping infrastructure tiers, not rewriting components.

> See [PRD.md](./PRD.md) for the full product requirements, milestones, and scale roadmap.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Query flow](#query-flow)
- [Ingestion flow](#ingestion-flow)
- [Dataset](#dataset)
- [API](#api)
- [Milestones](#milestones)
- [Scale roadmap](#scale-roadmap)
- [Success metrics](#success-metrics)
- [License and attribution](#license-and-attribution)

---

## Why this exists

Internal teams need trustworthy Q&A over a shared document corpus. Raw LLMs hallucinate. Plain vector search returns passages but no synthesized answer. Neither enforces access boundaries.

This service:

- Retrieves the right passages with **hybrid search** (dense + sparse) and **cross-encoder reranking**
- Generates answers **strictly grounded** in retrieved chunks, with inline citations
- **Abstains** when support is weak — the failure mode is *"I do not have enough to answer,"* not fabrication
- Respects **access context** via metadata pre-filtering and ACL-aware caching

---

## Architecture

High-level component view of the two main paths: **write** (ingestion) and **read** (query).

```mermaid
flowchart TB
    VecIdx[("Vector index<br/>dense search")]
    KwIdx[("Keyword index<br/>BM25 sparse search")]

    subgraph WritePath["Write path - ingestion"]
        HF["Hugging Face<br/>wikimedia/wikipedia"]
        Loader["Document loader<br/>(streaming slice)"]
        Chunker["Chunker"]
        Embedder["Bi-encoder<br/>embeddings"]
        Meta["Synthetic metadata<br/>doc_type, acl_tags, dates"]

        HF --> Loader --> Chunker --> Embedder
        Chunker --> Meta
        Embedder --> VecIdx
        Meta --> VecIdx
        Meta --> KwIdx
        Chunker --> KwIdx
    end

    subgraph ReadPath["Read path - query"]
        Client["API caller"]
        API["Query API<br/>POST /query"]
        Cache[("Semantic cache<br/>ACL-aware key")]
        Orch["Orchestrator"]
        Hybrid["Hybrid retrieval<br/>and metadata pre-filter"]
        Rerank["Cross-encoder<br/>rerank"]
        Gen["Grounded generator<br/>and retrieval tool"]
        Faith["Faithfulness check"]

        Client --> API --> Orch
        Orch --> Cache
        Cache -->|hit| Client
        Cache -->|miss| Hybrid
        Hybrid --> VecIdx
        Hybrid --> KwIdx
        Hybrid --> Rerank --> Gen
        Gen -->|tool call| Hybrid
        Gen --> Faith
        Faith -->|grounded| Cache
        Faith -->|weak support| Client
        Cache --> Client
    end
```

---

## Query flow

End-to-end path for a single query, including cache, retrieval, generation, and abstention.

```mermaid
sequenceDiagram
    actor Caller
    participant API as Query API
    participant Cache as Semantic Cache
    participant Ret as Hybrid Retrieval
    participant Rerank as Cross-Encoder
    participant Gen as Generator
    participant Faith as Faithfulness Check

    Caller->>API: POST /query<br/>query + access_context + filters
    API->>Cache: Lookup (semantic cluster, tenant, ACL signature)

    alt Cache hit
        Cache-->>Caller: Cached answer + citations
    else Cache miss
        API->>Ret: Hybrid search under ACL/metadata pre-filter
        Ret-->>API: Fused candidate set
        API->>Rerank: Rerank to top-k
        Rerank-->>API: Precise top-k chunks
        API->>Gen: Generate with citations
        opt Insufficient first pass
            Gen->>Ret: Call retrieval tool
            Ret-->>Gen: Additional chunks
        end
        Gen-->>API: Draft answer + citations
        API->>Faith: Score answer vs. retrieved chunks

        alt Grounded
            Faith-->>API: Pass
            API->>Cache: Write-through (ACL-aware key)
            API-->>Caller: answer, citations, confidence
        else Weak support
            Faith-->>API: Fail
            API-->>Caller: abstained=true, retrieved_chunks only
        end
    end
```

---

## Ingestion flow

Batch ingestion for the MVP. At scale this becomes a change-driven pipeline with cheap metadata-only updates vs. expensive re-embed paths.

```mermaid
flowchart LR
    A["Load dataset<br/>(streaming)"] --> B["Take 1K slice<br/>(MVP)"]
    B --> C["Chunk documents<br/>5-15 chunks/doc"]
    C --> D["Attach synthetic metadata<br/>doc_type, acl_tags, dates"]
    D --> E["Embed chunks<br/>(bi-encoder)"]
    E --> F["Write vector index"]
    C --> G["Write keyword index<br/>(BM25)"]
    D --> F
    D --> G
    F --> H["Searchable corpus"]
    G --> H
```

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) on Hugging Face (English split) |
| MVP size | 1,000 articles (deterministic slice) |
| Scale path | Same loader, remove cap, stream in batches |
| License | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — attribution required |
| Fields | `id`, `url`, `title`, `text` |

**Synthetic metadata** (deterministic, assigned at ingestion):

- `doc_type` — derived from article characteristics (e.g. length bands)
- `acl_tags` — hashed doc ID into synthetic access groups
- `created_at` / `updated_at` — for date-range filtering and freshness logic

```python
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
mvp_docs = []
for i, row in enumerate(ds):
    if i >= 1000:
        break
    mvp_docs.append(row)  # id, url, title, text
```

---

## API

Single primary endpoint. Full contract details are in [PRD.md §8](./PRD.md#8-api-contract-mvp).

### Request

```http
POST /query
```

```json
{
  "query": "Who developed the theory of relativity?",
  "access_context": { "groups": ["group_a", "group_b"] },
  "filters": { "doc_type": "optional", "date_range": "optional" },
  "options": { "top_k": 5, "allow_generation": true }
}
```

### Response

```json
{
  "answer": "Albert Einstein developed the theory of relativity...",
  "abstained": false,
  "confidence": 0.92,
  "citations": [
    { "chunk_id": "...", "doc_id": "...", "title": "Albert Einstein", "url": "..." }
  ],
  "retrieved_chunks": [
    { "chunk_id": "...", "text": "...", "score": 0.87 }
  ]
}
```

When `abstained` is `true`, `answer` is `null`, `confidence` reflects weak support, and `retrieved_chunks` still returns the closest matches.

---

## Milestones

Build is organized into six MVP milestones plus Phase 2 extensions.

```mermaid
flowchart TB
    M0["M0: Ingestion and indexing"] --> M1["M1: Hybrid retrieval and filtering"]
    M1 --> M2["M2: Cross-encoder reranking"]
    M2 --> M3["M3: Generation, faithfulness, tool use"]
    M3 --> M4["M4: ACL-aware semantic caching"]
    M4 --> M5["M5: Query rewriting and parallel tools"]
    M5 --> M6["M6: Observability and feedback"]
```

| Milestone | Scope | Exit criteria |
|-----------|-------|---------------|
| **M0** | Batch ingest 1K docs, chunk, embed, index | Every doc searchable in vector + keyword indexes |
| **M1** | Hybrid retrieval + metadata/ACL pre-filter | Hybrid recall beats vector-only on labeled set |
| **M2** | Cross-encoder reranking | Rerank improves top-k precision vs. fusion alone |
| **M3** | Grounded generation, citations, faithfulness, abstain, retrieval tool | Every response grounded with citations or explicit abstention |
| **M4** | ACL-aware semantic cache | Repeat queries hit cache; cross-context leakage test passes |
| **M5** | Query rewriting, parallel tool calls | Multi-hop queries work; partial tool failure degrades gracefully |
| **M6** | Tracing, feedback capture | Any request traceable end-to-end |

---

## Scale roadmap

Same component boundaries from MVP to 10M — scaling is staged infrastructure changes.

```mermaid
flowchart LR
    MVP["MVP<br/>1K docs<br/>In-memory indexes<br/>Batch ingest"]
    S1["Stage 1<br/>100K docs<br/>Vector store<br/>Verify hybrid + rerank latency"]
    S2["Stage 2<br/>1M docs<br/>Int8 quantization<br/>Load testing + cache tuning"]
    S3["Stage 3<br/>10M docs<br/>Sharded index<br/>Near-real-time ingest<br/>Full SLAs"]

    MVP --> S1 --> S2 --> S3
```

| Stage | Corpus | What changes | What stays the same |
|-------|--------|--------------|---------------------|
| MVP | 1K | In-memory / single-node indexes; batch ingest | All component interfaces |
| Stage 1 | 100K | Move to a real vector store | Read path logic, API contract |
| Stage 2 | 1M | Int8 quantization; tune candidate sizes; measure cache hit rate | Faithfulness, abstain logic |
| Stage 3 | 10M | Shard index; change-driven ingest; alias-based blue/green deploys | Retrieval, rerank, generate, cache components |

---

## Success metrics

```mermaid
flowchart TB
    root((Quality bar))
    root --> faith[Faithfulness]
    root --> retr[Retrieval]
    root --> safe[Safety]
    root --> ops[Operations]
    faith --> f1[Every claim cited]
    faith --> f2[Unsupported answers abstained]
    retr --> r1[Hybrid beats vector-only]
    retr --> r2[Rerank improves top-k]
    safe --> s1[ACL cache isolation]
    safe --> s2[No cross-context leakage]
    ops --> o1[Latency targets per phase]
    ops --> o2[Cache hit rate validated]
```

| Category | Metric | Target |
|----------|--------|--------|
| Quality | Faithfulness rate (non-abstained answers) | Very high; unsupported answers are defects |
| Quality | Abstention correctness | Abstain on genuinely unanswerable queries |
| Quality | Retrieval quality | Each stage (vector → hybrid → rerank) improves on labeled set |
| Safety | Cross-context cache isolation | Cached answer for context A never served to context B |
| Ops | Retrieval latency (MVP) | p95 < 500 ms |
| Ops | End-to-end latency (MVP, cold) | < 5 s including generation |

---

## Key design decisions

- **Chunk-level metadata denormalization** — ACL tags and `doc_type` copied onto every chunk so filters apply inside the retrieval query with no join.
- **ACL-aware semantic cache** — Cache key is `(semantic cluster, tenant, acl signature)`, not query similarity alone.
- **Generator hosts a retrieval tool** — Deterministic first-pass retrieval guarantees a baseline; the model can fetch more for multi-hop needs.
- **Quantize and shard at scale** — Int8 quantization cuts index size ~4×; recall loss recovered by full-precision cross-encoder rerank.

---

## Project status

| Area | Status |
|------|--------|
| PRD | Draft v1 — [PRD.md](./PRD.md) |
| Implementation | Not started |
| Corpus | Wikipedia English split via Hugging Face |

---

## License and attribution

Wikipedia content is licensed under [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). When using or displaying Wikipedia-derived content, provide appropriate attribution to Wikimedia contributors.

Service code licensing TBD.
