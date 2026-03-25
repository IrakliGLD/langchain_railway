# Developer Guide

## Purpose

This guide documents the current engineering structure for `langchain_railway` and how to work on it safely.

## Current Architecture

The runtime is organized around a staged query pipeline with typed tool support and SQL fallback:

1. Context preparation and fast routing
2. Agent/tool path for high-frequency intents
3. SQL planning/execution fallback path
4. Analysis and enrichment stage
5. Summarization and chart output stage

## Key Modules

- `main.py`: API entrypoint and request handling
- `agent/`: routing, tool execution, orchestration pieces
- `core/`: LLM, SQL generation, database execution
- `analysis/`: statistical and domain transformations
- `visualization/`: chart selection and chart data construction
- `knowledge/`: runtime knowledge markdown + selector logic
- `tests/`: automated tests (unit/integration/regression)

## Data Contract

Both typed tools and SQL fallback are expected to produce the same tabular contract for downstream stages:

- `df`: pandas DataFrame
- `cols`: column names
- `rows`: row tuples
- `provenance_cols`: exact source columns used for summary grounding
- `provenance_rows`: exact source rows used for summary grounding
- `provenance_source`: `sql` or `tool`
- `provenance_query_hash`: stable identity for the exact SQL/tool invocation

This keeps analyzer, summarizer, and chart pipeline source-agnostic.

## Domain Conventions

- For trade segment filters, use canonical normalized segment values.
- User phrasing like "balancing electricity" maps to the `balancing` segment in `trade_derived_entities`.
- Use `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'` for balancing-segment trade queries.

## Local Workflow

1. Run focused tests while developing:

```bash
pytest -q tests/test_main.py
```

2. Run full test suite before commit:

```bash
pytest -q
```

3. If behavior changed in routing/tooling, run related tests first:

```bash
pytest -q tests/test_router.py tests/test_tool_adapter.py tests/test_pipeline_agent_mode.py
```

## Deployment Modes

Runtime auth and admin-surface enablement should be explicit in deployed environments.

- `ENAI_AUTH_MODE=gateway_only`
  - Intended for trusted proxy or edge-function callers that send `X-App-Key`.
  - `SUPABASE_JWT_SECRET` may be present, but bearer auth stays disabled.
- `ENAI_AUTH_MODE=gateway_and_bearer`
  - Intended when direct `Authorization: Bearer <token>` calls are part of the boundary.
  - `SUPABASE_JWT_SECRET` is mandatory.
- `ENAI_AUTH_MODE=auto`
  - Temporary compatibility mode during rollout.
  - Bearer auth turns on only when `SUPABASE_JWT_SECRET` is present.
  - Migrate to an explicit auth mode instead of leaving production on `auto`.

Use an explicit deployment environment value:

- `ENAI_DEPLOYMENT_ENV=development`
- `ENAI_DEPLOYMENT_ENV=staging`
- `ENAI_DEPLOYMENT_ENV=production`
- `ENAI_DEPLOYMENT_ENV=test`

`/evaluate` is fail-safe by default:

- It is only allowed when `ENAI_DEPLOYMENT_ENV` is `development` or `test`.
- It also requires `ALLOW_EVALUATE_ENDPOINT=true`.
- Keep it disabled in `staging` and `production`.

## Production Env Baseline

Gateway-only production baseline:

```bash
ENAI_DEPLOYMENT_ENV=production
ENAI_AUTH_MODE=gateway_only
ENAI_GATEWAY_SECRET=...
ENAI_SESSION_SIGNING_SECRET=...
ENAI_EVALUATE_SECRET=...
ENABLE_METRICS_ENDPOINT=false
ENABLE_EVALUATE_ENDPOINT=false
ALLOW_EVALUATE_ENDPOINT=false
ASK_RATE_LIMIT_GATEWAY_PER_MINUTE=300
ASK_RATE_LIMIT_PUBLIC_PER_MINUTE=10
ASK_RATE_LIMIT_PREAUTH_PER_MINUTE=300
```

Hybrid gateway + bearer production baseline:

```bash
ENAI_DEPLOYMENT_ENV=production
ENAI_AUTH_MODE=gateway_and_bearer
SUPABASE_JWT_SECRET=...
ENAI_GATEWAY_SECRET=...
ENAI_SESSION_SIGNING_SECRET=...
ENAI_EVALUATE_SECRET=...
ENABLE_METRICS_ENDPOINT=false
ENABLE_EVALUATE_ENDPOINT=false
ALLOW_EVALUATE_ENDPOINT=false
ASK_RATE_LIMIT_GATEWAY_PER_MINUTE=300
ASK_RATE_LIMIT_PUBLIC_PER_MINUTE=10
ASK_RATE_LIMIT_PREAUTH_PER_MINUTE=300
```

## Documentation Policy

- Keep active docs in `docs/active/`.
- Keep runtime knowledge in `knowledge/`.
- Avoid creating new top-level audit/migration markdown files unless explicitly requested.
- If a doc is historical-only, archive or remove it instead of keeping it in active docs.
