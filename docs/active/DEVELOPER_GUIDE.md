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
- **`gateway_and_bearer`** — direct `Authorization: Bearer <token>` calls are available only when `ENAI_DEPLOYMENT_ENV=test` and require `SUPABASE_JWT_SECRET`. Every non-test environment fails startup until direct callers share the server-owned entitlement path planned for P3.

`auto` is invalid. The safe default is `gateway_only`, and merely configuring `SUPABASE_JWT_SECRET` never enables bearer authentication.

### P3 gateway actor assertion

The P3.B edge function signs the contract version, request ID, authenticated actor ID, Supabase session ID, and Unix issue time with the same secret used for `X-App-Key`. The backend verifies the assertion before model/database work and selects an opaque actor-bound session.

- `ENAI_GATEWAY_ACTOR_ASSERTION_MODE=optional` is the independent-deployment default. A request with no actor assertion is temporarily accepted, but any partial, malformed, stale, future, replayed, or tampered assertion is rejected.
- `ENAI_GATEWAY_ACTOR_ASSERTION_MODE=required` rejects legacy gateway requests that send no assertion. Enable it only after the tracked P3.B `chat-with-enerbot` function is deployed and backend logs show `Gateway actor assertion: verified=True` for real traffic.
- `ENAI_GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS=120` controls freshness and is constrained to 30–900 seconds.

Keep `ENAI_AUTH_MODE=gateway_only`. Direct bearer remains test-only because local JWT verification alone cannot enforce the deployed active-user, entitlement, idempotency, and persistence authority.

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
ENAI_GATEWAY_ACTOR_ASSERTION_MODE=required  # use optional only during the coordinated P3 rollout
ENAI_GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS=120
ENAI_GATEWAY_SECRET=...
ENAI_SESSION_SIGNING_SECRET=...
ENAI_EVALUATE_SECRET=...
ENABLE_METRICS_ENDPOINT=false
ENABLE_EVALUATE_ENDPOINT=false
ALLOW_EVALUATE_ENDPOINT=false
ASK_RATE_LIMIT_GATEWAY_PER_MINUTE=300
ASK_RATE_LIMIT_PUBLIC_PER_MINUTE=10
ASK_RATE_LIMIT_PREAUTH_PER_MINUTE=300
MAX_REQUEST_BODY_BYTES=262144
```

### Test-only gateway + bearer

```bash
ENAI_DEPLOYMENT_ENV=test
ENAI_AUTH_MODE=gateway_and_bearer
ENAI_GATEWAY_ACTOR_ASSERTION_MODE=optional
ENAI_GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS=120
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
MAX_REQUEST_BODY_BYTES=262144
```

## Prompt-Budget Env Vars (Phase 2.b, 2026-05-13)

```bash
PROMPT_BUDGET_MAX_CHARS=45000                # legacy default; used by llm_summarize / llm_generate_plan_and_sql
ANALYZER_PROMPT_BUDGET_MAX_CHARS=...         # analyzer-only override (defaults to PROMPT_BUDGET_MAX_CHARS)
SUMMARIZER_PROMPT_BUDGET_MAX_CHARS=...       # structured-summarizer-only override (defaults to PROMPT_BUDGET_MAX_CHARS)
```

See [`query_pipeline_architecture.md`](query_pipeline_architecture.md) §3.2 / §3.9. Summarizer prompts routinely hit 90–110k chars in deep mode because `DOMAIN_KNOWLEDGE` + `EXTERNAL_SOURCE_PASSAGES` expand; analyzer prompts do not. Raising `SUMMARIZER_PROMPT_BUDGET_MAX_CHARS` independently is the right knob for that.

## Dedicated Report Model

Set the dedicated profile on the `enai-report-worker` Railway service. The web
service does not need these values. Report mode is authoritative once selected
by the client. In the adaptive v2 pipeline, the research planner and
whole-document writer/repair calls use this profile; the legacy question
analyzer and per-section writers are not invoked. Report mode is not
reclassified from the question text.

The worker still needs the normal provider settings for non-LLM evidence
operations. In particular, vector embeddings keep their separately configured
provider and do not use `REPORT_*`.

```bash
REPORT_MODEL_TYPE=openai
REPORT_MODEL=gpt-5.6-terra
REPORT_MAX_OUTPUT_TOKENS=8192
REPORT_TIMEOUT_SECONDS=240
REPORT_REASONING_EFFORT=medium
REPORT_STRUCTURED_OUTPUT_METHOD=auto
REPORT_PIPELINE_V2_MODE=shadow
REPORT_TRACK_ANALYSIS_MODE=shadow
REPORT_MAX_GENERATIVE_CALLS=3
REPORT_RESEARCH_MAX_TRACKS=4
REPORT_RESEARCH_MAX_WORKERS=3
OPENAI_API_KEY=...
```

Supported `REPORT_MODEL_TYPE` values are `openai`, `gemini`, and `nvidia`.
Use `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `NVIDIA_API_KEY` respectively.
When `REPORT_MODEL_TYPE` is absent, Report inherits the existing primary/stage
model behavior. Report calls never use the implicit OpenAI fallback.

`REPORT_STRUCTURED_OUTPUT_METHOD=auto` uses native JSON Schema for OpenAI and
portable prompt-plus-validation output for Gemini and NVIDIA, preserving their
existing behavior. When changing to a model with different capabilities, set
`json_schema`, `function_calling`, or `prompt` explicitly. This changes only
the output mechanism for the configured Report model; it never switches
providers or enables provider fallback.

`ENABLE_OPENAI_FALLBACK` defaults to `false` globally. Leave it unset to keep
Standard and other primary-provider calls from falling back merely because an
OpenAI key is configured for Report.

`REPORT_PIPELINE_V2_MODE` defaults to `disabled`. Use `shadow` first: it runs
the research planner, deterministic collectors, evidence gate, exhibits, and
document planner for content-free comparison telemetry, but still publishes
the legacy result. After validating coverage and cost, apply the
`2026-07-28_report_result_v2.sql` database patch, deploy the `report-jobs` Edge
function and frontend, and then set the worker to `enabled`. Enabled mode uses
at most one research-planner call, one whole-document writer call, and one
targeted repair call. Set `REPORT_MAX_GENERATIVE_CALLS=2` to disable repair.

When `REPORT_PIPELINE_V2_MODE=enabled`, track-scoped standard-quality analysis
also defaults to `enabled`. Set `REPORT_TRACK_ANALYSIS_MODE=shadow` or
`disabled` explicitly only for observation or rollback. An explicit override
always takes precedence over the derived default.

Every blocking section error must reach the repairer as a named offender, not
just a code. `DERIVED_CLAIM_INVALID` carries `VERIFIED_DERIVED_REPAIR_HINTS`
and `UNGROUNDED_NUMERIC_CLAIM` carries `UNGROUNDED_VALUE_REPAIR_HINTS`, which
lists the exact values a paragraph asserts that its evidence cannot support.
Without them the repairer guesses which of its numbers offended, and a wrong
guess ends the job: `REPORT_DOCUMENT_INVALID` is not retryable. Add a hint
alongside any new blocking code.

The section word floor scales with the rows a section can actually cite
(`report_section_word_floor_ratio`), mirroring the document-level bounds. A
two-row section that was asked for a full-length essay could only reach the
floor by inventing numbers the grounding gate then rejected. The ceiling never
scales.

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
ENAI_GATEWAY_ACTOR_ASSERTION_MODE=optional
ENAI_GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS=120
ENABLE_METRICS_ENDPOINT=false
ENABLE_EVALUATE_ENDPOINT=false
ALLOW_EVALUATE_ENDPOINT=false
ASK_RATE_LIMIT_GATEWAY_PER_MINUTE=300
ASK_RATE_LIMIT_PUBLIC_PER_MINUTE=10
ASK_RATE_LIMIT_PREAUTH_PER_MINUTE=300
MAX_REQUEST_BODY_BYTES=262144
```

For test-only hybrid bearer mode also set (never use it in a deployed environment before P3):

```bash
ENAI_DEPLOYMENT_ENV=test
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
