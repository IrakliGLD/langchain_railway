# Targeted Test Suite

The targeted suite is the fast-feedback test set run **at every sub-phase** of phased work
(`developer-phased-audit`). It is what should catch a regression *before* a long
full-suite run does. The "wide targeted suite of 459 tests" referenced in
[`docs/active/query_pipeline_architecture.md`](../../../docs/active/query_pipeline_architecture.md)
§5.1 was a remembered set; this file pins it.

## Scope

**Every test file directly under `tests/` is in scope.**

The only exclusion is `tests/security/`, which is its own gate (adversarial /
red-team suite, run separately at release time, not at every sub-phase).

This deliberately broad definition is the post-2026-05-13 fix: a previous
narrower informal set let
`tests/test_vector_prompt_integration.py::test_conceptual_summary_uses_vector_as_primary_evidence`
slip past targeted runs for ~3 weeks until the 8h35m full-suite caught it.
"Targeted" now means "everything fast", not "the modules I happened to think
about during the change."

## Command

From repo root:

```bash
python -m pytest tests/ --ignore=tests/security -q
```

Add `-v` for per-test output, `--tb=short` for compact tracebacks, or
`-k <pattern>` when investigating a single failure.

To run only the files known to be relevant to a specific change, list them
explicitly — but a clean **green run of the full targeted suite** is required
before declaring a sub-phase done.

## Files in scope (snapshot)

Run `ls tests/*.py | sort` for the live list. Snapshot at 2026-07-07 (`test_orchestrator.py` deleted with the agent loop; `test_tool_adapter.py` deleted with the dead adapter):

| Group | Files |
|---|---|
| Pipeline orchestration | `test_main.py`, `test_pipeline_agent_mode.py`, `test_pipeline_analyzer_fallback.py`, `test_pipeline_balancing_enrichment.py`, `test_phase5_cleanup.py` |
| Analyzer + routing | `test_question_analysis_catalogs.py`, `test_question_analysis_contract.py`, `test_question_analyzer_phase_c.py`, `test_question_analyzer_shadow.py`, `test_router.py`, `test_routing_regressions.py`, `test_resolved_query_and_fallback.py`, `test_semantic_lock.py`, `test_session_memory.py` |
| Evidence + planning | `test_evidence_joins.py`, `test_evidence_planner.py`, `test_frame_adapters_filter.py`, `test_guardrails.py` |
| Vector / RAG | `test_vector_embeddings.py`, `test_vector_ingestion.py`, `test_vector_knowledge_contract.py`, `test_vector_pipeline.py`, `test_vector_prompt_integration.py`, `test_vector_retrieval.py`, `test_vector_retrieval_tier.py`, `test_vector_store.py` |
| Summarizer + rendering | `test_balancing_prompt_guidance.py`, `test_chart_frame_builder.py`, `test_derived_chart_builder.py`, `test_forecast_direct_answers.py`, `test_regulated_tariff_list.py`, `test_residual_weighted_price_direct.py`, `test_summarizer_absence_guardrail.py`, `test_tariff_snapshot_rendering.py` |
| Tools + SQL | `test_price_tools_sql.py`, `test_shares_sql.py`, `test_sql_executor_pivot.py`, `test_tariff_tools.py` |
| Domain analytics | `test_combined_share_resolution.py`, `test_metric_registry.py`, `test_seasonal_stats.py` |
| Infra + observability | `test_config.py`, `test_context.py`, `test_metrics_observability.py`, `test_query_validation.py`, `test_trace_observability.py` |

## Maintenance rule

When you add a new test file under `tests/`, **it is automatically in the
targeted suite** — the `pytest tests/ --ignore=tests/security` invocation
sweeps the whole directory. There is no allowlist to update.

If a test becomes prohibitively slow (>30 s wall time by itself, without a
real reason), do **not** silently quarantine it. Either:

1. Optimise it (mock more, narrow the assertion surface), or
2. Add it to `tests/security/` if it really is a slow integration check, or
3. Flag the slow case in the relevant phase's audit log and discuss before excluding.

A test that hits real LLMs or real Supabase belongs in a separate manual /
nightly run, **not** in `tests/`. Tests in `tests/` must mock external
services so they remain fast and deterministic.

## Why this list isn't pinned by name

Earlier drafts of this doc proposed listing each test file as an explicit
allowlist. That pattern fails open: a new file added to `tests/` for a new
behaviour silently drops out of the targeted run until someone remembers to
add it. The directory-scan approach fails closed instead — a new file is
covered until someone explicitly excludes it.

This file remains the source of truth for **what counts as the targeted suite**
and **how to invoke it**; the *list of test files* is whatever currently
lives in `tests/*.py`.
