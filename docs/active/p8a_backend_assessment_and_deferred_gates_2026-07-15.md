# P8.A Backend Assessment and Deferred Gates

**Date:** 2026-07-15
**Repository:** `D:/Enaiapp/langchain_railway`
**Scope:** current backend source and local tests; frontend/Supabase and unobserved live infrastructure are not graded as backend evidence

## Implemented P8.A slice

- Extracted the sliding-window rate-limit state from `main.py` behind `InMemoryRateLimitRepository` while retaining request authentication/key derivation and configured limits at the API boundary.
- Kept the repository explicitly process-local and exposed only aggregate topology/counts; this preserves and clarifies the one-worker/one-replica restriction.
- Chose `pyproject.toml` as the single pytest configuration authority and removed `pytest.ini`.
- Added focused coverage gates for the extracted rate-limit repository and database query/runtime-identity boundary.
- Raised the global production-code CI floor from 70% to 80% only after a clean measured result of 82.11%.

Local evidence:

- all `1,549` tests passed with the consolidated pytest configuration;
- production-code coverage: `82.11%` (`16,051` statements, `2,871` missed);
- `utils/rate_limits.py`: `97.50%` focused coverage;
- `core/query_executor.py`: `98.46%` focused coverage after adding runtime-identity, memory-limit, batching, read-only, and row-cap tests;
- Ruff and `git diff --check` passed for the P8.A working diff;
- gateway schema, routing golden cases, provenance, trace, main API, and deadline tests passed around the extraction.

This is an incremental P8.A slice, not a declaration that every oversized module or debt item is complete. `main.py`, `core/llm.py`, `agent/pipeline.py`, `agent/planner.py`, and `agent/summarizer.py` remain large and must be extracted one stable interface at a time.

## Deferred evidence gates

### Stage 0.7 production counter

Do not delete analyzer-route strategies from local reasoning alone.

1. Deploy an exact attested backend SHA.
2. Call the protected `/metrics` endpoint with the existing gateway/app key at the beginning and end of a representative production window.
3. Record at least:
   - `metrics.request_count`;
   - `metrics.stage_0_7_entered`;
   - `metrics.stage_0_7_invocation_built`;
   - `metrics.stage_0_7_used_result`;
   - relevant `metrics.tool_calls_by_source` and `metrics.tool_time_by_source`.
4. Calculate deltas and the ratios `used_result / entered` and `used_result / request_count`. Do not mix counters across a restart without recording the restart/deployment boundary.
5. Sample successful and failed Stage 0.7 traces, then run routing golden/adversarial regressions before proposing deletion.
6. The backend/AI owner must approve the decision and rollback. Until then the P8.2 counter item is **Manual verification pending**.

No numerical removal threshold is invented here: it must be chosen from observed product value, latency/cost, and correctness impact.

### LIGHT-tier retrieval overlap

Keep current behavior. Before changing it, capture per-tier request counts, vector/provider latency percentiles, total request latency, token/cost deltas, retrieval relevance, grounding/provenance failures, and answer-quality comparison over representative traffic. Run the same routing/vector suites with the candidate behavior. This item remains blocked on production latency/correctness evidence.

### Temporary rollout flags

Do not remove or hard-code rollout flags until each candidate behavior has run for two stable production releases with retained telemetry and a tested rollback. This includes at least:

- `ENAI_EVIDENCE_FINALIZATION_MODE` (default `shadow`);
- `ENAI_PLAN_VALIDATION_MODE` (default `warn`);
- `ENAI_ENABLE_HONEST_TERMINAL_OUTCOMES` (default `false`);
- `ENABLE_EVIDENCE_REANALYSIS` (default `false`).

For each release record deployed SHA, effective value, observation window, counters, incidents, rollback owner, and go/no-go decision. The two-release condition has not been attested here.

## Current A-F assessment

Grades distinguish local implementation quality from production assurance. `A` means strong current evidence with only minor residual risk; `F` means absent or fundamentally unsafe.

| Perspective | Grade | Evidence and limiting factor |
|---|---:|---|
| Functional/analytics correctness | A- | Broad golden, evidence, unit, provenance, chart, and terminal-outcome coverage is green. Several P4 behaviors remain deliberately shadow/warn/off pending rollout evidence. |
| Architecture/design | B | Stable evidence, DB-gateway, provider, metadata, and rate-limit interfaces now exist, but multiple god modules remain and the state repository has no shared implementation. |
| Code quality/technical debt | B+ | Ruff is clean, pytest has one authority, tests are extensive, and changes are phase-separated. Large modules and deferred flag/counter cleanup remain. |
| Security/privacy | B+ locally / B- production assurance | Auth/content/privacy boundaries and a least-privilege role package exist. Live DB identity, PUBLIC/network grants, vendor retention, secret rotation, and log-canary evidence are still manual. |
| Reliability/concurrency | B | Database/provider breakers, deadlines, thread-safe metrics, and locked rate-limit state are tested. Cooperative cancellation, duplicate-charge evidence, and shared multi-replica state remain open. |
| Performance/efficiency | B- | Result limits and bounded state are present, but there is no current production load/latency evidence for LIGHT overlap, provider/DB cancellation, or two replicas. |
| Testing/release confidence | A- locally / B production assurance | `1,549` tests pass at `82.11%` coverage with focused 95% floors. Live DB, container, exact-artifact promotion, rollback, and soak evidence remain unverified. |
| Deployment/operations | B- | Pinned non-root Docker and exact-SHA evidence workflow are implemented. Actual image build, dependency audit, Railway topology, Supabase role, smoke, and rollback attestations are pending. |

**Overall backend grade: B.** The source is materially safer and better tested than the historical assessment, but an A-range production grade would be misleading until live least privilege, immutable artifact promotion, one-replica topology, failure injection/cancellation, privacy retention, latency, and rollout evidence are recorded.
