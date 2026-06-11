# Developer Guide

Operational reference for working on `langchain_railway`: local workflow, deployment env modes, auth, and manual endpoint validation.

For the runtime architecture, see [`query_pipeline_architecture.md`](query_pipeline_architecture.md). For phased-implementation discipline (planning, audit, targeted-suite-green requirement), see [`skills/developer-phased-audit/`](../../skills/developer-phased-audit/SKILL.md).

## Local Workflow

```bash
# Quick smoke test while iterating
pytest -q tests/test_main.py

# Targeted suite (required green before any phased-audit commit)
pytest tests/ --ignore=tests/security -q
```

The targeted suite is defined and maintained in [`skills/developer-phased-audit/references/targeted-suite.md`](../../skills/developer-phased-audit/references/targeted-suite.md). When you add a new test file under `tests/` it is automatically in scope (directory-scan, fail-closed).

## Domain Conventions

- For trade-segment filters use canonical normalized segment values.
- User phrasing like "balancing electricity" maps to the `balancing` segment in `trade_derived_entities`.
- Use `LOWER(REPLACE(segment, ' ', '_')) = 'balancing'` for balancing-segment trade queries.

## Auth Modes

Set `ENAI_AUTH_MODE` explicitly in deployed environments:

- **`gateway_only`** — trusted proxy / edge function callers that send `X-App-Key`. `SUPABASE_JWT_SECRET` may be present but bearer auth stays disabled.
- **`gateway_and_bearer`** — direct `Authorization: Bearer <token>` calls are part of the boundary. `SUPABASE_JWT_SECRET` is mandatory.
- **`auto`** — temporary compatibility mode during rollout. Bearer auth turns on only when `SUPABASE_JWT_SECRET` is present. **Migrate to an explicit mode**; do not leave production on `auto`.

## Deployment Constraint: Single Replica

Run **exactly one worker process / one replica** (`uvicorn` default single worker; do not add
`--workers N` or scale Railway replicas). Rate limits, session memory, the LLM cache, and
circuit breakers are all in-process — multiple replicas multiply rate limits and fragment
sessions. See `query_pipeline_architecture.md` §4.1 for the rationale and the declined
shared-store alternative.

## Deployment Environment Values

```bash
ENAI_DEPLOYMENT_ENV=development   # local
ENAI_DEPLOYMENT_ENV=staging
ENAI_DEPLOYMENT_ENV=production
ENAI_DEPLOYMENT_ENV=test          # CI
```

`/evaluate` is fail-safe: allowed only when `ENAI_DEPLOYMENT_ENV` ∈ {`development`, `test`} AND `ALLOW_EVALUATE_ENDPOINT=true`. Keep disabled in staging and production.

## Production Env Baselines

### Gateway-only

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

### Gateway + bearer

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

## Prompt-Budget Env Vars (Phase 2.b, 2026-05-13)

```bash
PROMPT_BUDGET_MAX_CHARS=45000                # legacy default; used by llm_summarize / llm_generate_plan_and_sql
ANALYZER_PROMPT_BUDGET_MAX_CHARS=...         # analyzer-only override (defaults to PROMPT_BUDGET_MAX_CHARS)
SUMMARIZER_PROMPT_BUDGET_MAX_CHARS=...       # structured-summarizer-only override (defaults to PROMPT_BUDGET_MAX_CHARS)
```

See [`query_pipeline_architecture.md`](query_pipeline_architecture.md) §3.2 / §3.9. Summarizer prompts routinely hit 90–110k chars in deep mode because `DOMAIN_KNOWLEDGE` + `EXTERNAL_SOURCE_PASSAGES` expand; analyzer prompts do not. Raising `SUMMARIZER_PROMPT_BUDGET_MAX_CHARS` independently is the right knob for that.

## Manual Endpoint Validation

```bash
# Local server
uvicorn main:app --reload --port 8000
```

### Gateway-only request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-App-Key: <ENAI_GATEWAY_SECRET>" \
  -d '{"query":"Compare tariffs for 2024","mode":"light"}'
```

### Bearer-mode request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <SUPABASE_JWT>" \
  -d '{"query":"Compare tariffs for 2024","mode":"light"}'
```

### Local env baseline for manual testing

```bash
ENAI_DEPLOYMENT_ENV=development
ENAI_AUTH_MODE=gateway_only
ENABLE_METRICS_ENDPOINT=false
ENABLE_EVALUATE_ENDPOINT=false
ALLOW_EVALUATE_ENDPOINT=false
ASK_RATE_LIMIT_GATEWAY_PER_MINUTE=300
ASK_RATE_LIMIT_PUBLIC_PER_MINUTE=10
ASK_RATE_LIMIT_PREAUTH_PER_MINUTE=300
```

For hybrid bearer mode also set:

```bash
ENAI_AUTH_MODE=gateway_and_bearer
SUPABASE_JWT_SECRET=<supabase-jwt-secret>
```

`/metrics` is disabled by default. If enabled locally it requires `X-App-Key: <ENAI_EVALUATE_SECRET>`. `/evaluate` should stay disabled in staging and production; if testing it locally set `ENABLE_EVALUATE_ENDPOINT=true` + `ALLOW_EVALUATE_ENDPOINT=true` and use `X-App-Key: <ENAI_EVALUATE_SECRET>`.

## Debugging Order

1. Reproduce with a single focused test.
2. Check router/analyzer decision in trace logs (`stage_0_2_question_analyzer`).
3. Check tool/SQL fallback branch.
4. Check analyzer enrichment, summarizer, chart stages.

For systematic Q&A failure triage (latency spikes, grounding failures, schema validation crashes, routing misclassification) consult [`skills/pipeline-failure-diagnostics/`](../../skills/pipeline-failure-diagnostics/SKILL.md) — it is the source of truth for failure patterns and fix-layer selection.

## Documentation Policy

- Keep active docs in `docs/active/`.
- Keep runtime knowledge in `knowledge/`.
- Keep ingestion-ready sources in `docs_to_ingest/`.
- Avoid creating new audit / migration / handoff markdown files. Phased-implementation discipline (see the `developer-phased-audit` skill) captures rationale in commit bodies instead.
