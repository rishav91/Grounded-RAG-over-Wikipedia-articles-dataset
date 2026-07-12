# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A grounded RAG service over 1,000 English Wikipedia articles (`20231101.en` split, 11,768 chunks in Qdrant), built around a "Verifiable-or-Abstain" principle: no answer leaves the system unless every claim traces to a permitted, retrieved chunk.

The `docs/` directory is the canonical spine of this project (PRD, ARCHITECTURE, ADRs, AI-ARCHITECTURE, DATA-MODEL, API-CONTRACTS, REQUIREMENTS, ROADMAP). Read `docs/ROADMAP.md` for milestone status and `docs/ADRs.md` before making architectural changes - locked-stack decisions (LangGraph, OpenAI `text-embedding-3-small`, Cohere `rerank-v3.5`, Qdrant, provider-agnostic generation via `init_chat_model`) are already made and justified there.

## Environment setup

- Python venv is built from **pyenv 3.11.11**
- Claude Code's Bash tool auto-activates this project's `.venv`, so bare `python`/`pytest`/`ruff` resolve correctly there. In a plain terminal, `source .venv/bin/activate` also works, or use `.venv/bin/python` / `.venv/bin/pytest` explicitly.
- Hooks and other subprocesses spawned outside an activated shell (e.g. `.claude/settings.json`'s format-on-edit hook) always need the explicit `.venv/bin/...` path - nothing auto-activates for them.
- Qdrant must be running before almost anything works - start it with `docker compose up -d` (single service, binds to `localhost:6333`/`6334`, data persists to `qdrant_storage/`). There is no embedded/mock fallback.
- Required env vars (see `.env.example`): `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `COHERE_API_KEY`, `GENERATION_MODEL`, `FAITHFULNESS_MODEL`. `GENERATION_MODEL` has **no code-level default** (ADR-007: provider-agnostic, no hard-coded default) - it must be set explicitly, e.g. `openai:gpt-4o-mini`. `FAITHFULNESS_MODEL` falls back to `GENERATION_MODEL` if unset.
- Install with `.venv/bin/pip install -e ".[dev]"`.

## Testing

- Unit tests: `.venv/bin/pytest -q` (65 tests in `tests/`, fast, mocked, no external calls).
- `scripts/eval_m1.py` through `scripts/eval_m5.py` are **not** part of the pytest suite - they are milestone-specific integration harnesses that hit real Qdrant + Cohere + the configured LLM. They cost money and take real time; don't run them casually or as a substitute for unit tests. Only run a specific `eval_mN.py` when validating that milestone's behavior.
- Various thresholds (`FAITHFULNESS_CONFIDENCE_THRESHOLD`, `CACHE_SIMILARITY_THRESHOLD`, `SUFFICIENCY_LOW/HIGH_SCORE_THRESHOLD`, recall/precision margins) are explicitly flagged in code and `docs/REQUIREMENTS.md` as placeholders pending real measurement, not settled constants.

## Repo conventions

- From M1 onward, branch off `main` per milestone (e.g. `m5-phase2`); don't commit milestone work straight to `main`.
- Update `docs/ROADMAP.md`'s milestone/Ships checkboxes every time work ships, without being asked.
- `qdrant_storage/` already contains the real ingested corpus (11,768 chunks) that milestones were verified against - treat deleting/resetting it as destructive.
