# Architecture Audit — `langchain_railway/`

**Date:** 2026-06-30
**Scope:** Backend-wide architectural review of `langchain_railway/` (Python 3.11 / FastAPI, ~56K LOC, 142 source files).
**Method:** Performed with the `architect-reviewer` agent (installed from `claude-code-templates`, `expert-advisors/architect-review`). Four focused passes — layering/dependencies, the core LLM seam, orchestration/planning, and enrichment/summarization — each grounding findings in `file:line` references. Key claims were spot-checked against the source (see [Grounding & verification](#grounding--verification)).
**Reviewer principle:** *Good architecture enables change. Flag anything that makes future changes harder.*

> Companion to the existing `docs/active/audit_2026-06-10.md` and `docs/active/architecture_quality_improvement_plan_2026-06-10.md`. This pass is purely structural (SOLID, layering, dependency direction, coupling) and does not assess correctness or test coverage.

---

## Executive summary

The **spine of the architecture is sound**: the intended high→low layering (`main → agent → core/analysis → contracts/utils`) holds for the large majority of edges, `contracts/` is a clean dependency sink with zero outbound layer imports, and `core/`/`analysis/` contain no imports back into `agent/`. Several modules are exemplary (`sql_generator.py`, `query_executor.py`, `evidence_validator.py`, `llm_payloads.py`, `prompt_budget.py`).

The debt is **concentrated, not diffuse**, and it has one dominant shape: **five god-modules that grew by absorbing each new case as an inline branch or guardrail rather than as an extension of a contract.**

| Module | Lines | Role | Distinct responsibilities |
|---|---:|---|---|
| `core/llm.py` | 3,387 | LLM seam | ≥7 (provider resolution, cost, cache, resilience, classification, prompts, envelopes) |
| `agent/analyzer.py` | 3,407 | Enrichment | ≥13 (shares, correlation, forecast, causal, trendlines, chart overrides, answer formatting…) |
| `agent/pipeline.py` | 2,573 | Orchestration | 6-stage driver inlined into a ~597-line `process_query` |
| `agent/summarizer.py` | 2,531 | Summarization | narrative + grounding engine + 3 deterministic renderers |
| `agent/planner.py` | 2,384 | Planning | language/mode/intent detection + 7 LLM-output guardrails + legacy plan/SQL gen |

**Overall architectural impact: High** — driven by the orchestration, LLM, and enrichment layers. Layering is the one **Medium** area. None of the findings are emergencies; all are structural separations of *existing, working behavior* and require **no new infrastructure or services**.

### Scorecard

| Area | Impact | Headline |
|---|:--:|---|
| Layering & dependencies | **Medium** | Spine sound; 2 upward edges into `core/llm`, 2 lazy-import cycles, DB access leaking into `main.py`/`pipeline.py` |
| Core LLM seam | **High** | 3,387-line god-module; provider selection by string-sniffing; primary→fallback→cache block copy-pasted at 4 sites |
| Orchestration & planning | **High** | ~597-line `process_query`; tool routing duplicated across 4 modules; implicit stage contracts via mutable `QueryContext`; likely-dead legacy loop |
| Enrichment & summarization | **High** | `analysis/` layer bypassed (dead `seasonal.py` vs. inline CAGR ×4); grounding engine wedged into summarizer; per-request DB fan-out |

---

## Cross-cutting findings (the highest-value synthesis)

These themes recur across multiple independent passes. They are the root causes; the per-area violations below are their symptoms.

### 1. `core/llm.py` is a cross-layer hub, and its public *types* are the reason
`core/llm.py` has fan-in from **7 runtime modules across 4 layers** (`main`, `agent/{planner,summarizer,orchestrator,pipeline}`, `guardrails`, `visualization`). Two of those are **upward dependency-direction violations** (`visualization/` and `guardrails/` importing the LLM layer). The deeper cause: public result types like `SummaryEnvelope` and the helper `classify_query_type` *live inside* the 3,387-line module, so any consumer must import the whole hub.
**Fix (converged from two passes):** move `SummaryEnvelope` and peer result types to `contracts/`, and lift `classify_query_type` into a small `core/query_classifier.py`. This breaks both upward edges **and** de-risks the planned `llm.py` split — it is the single highest-leverage, lowest-risk move in this report.

### 2. "Add a case" is a multi-site shotgun edit everywhere
The same anti-pattern appears in four guises:
- **Providers:** identity is recovered by string-sniffing model names (`_provider_from_model_name`), then re-branched in the factory, model-name resolver, cost table, and breaker. Adding a 4th provider touches **6–7 sites**.
- **Tool routing:** selection logic is duplicated across **4 modules** (`router.match_tool`, `planner.build_tool_invocation_from_analysis`, `evidence_planner._add_steps_from_rules`, `orchestrator._score_dataset_for_query`).
- **Seasonality/CAGR:** the summer/winter split is reimplemented inline ≥3× and `SUMMER_MONTHS` is defined in **7 places** (twice in config with *different types*).
- **Misroute fixes:** each production misroute adds another `_apply_*_guardrail` pair in `planner.py` (now 7, chained unconditionally).
**Fix:** replace each with a single source of truth — a provider **registry/strategy**, one tool **resolver**, one `SeasonConfig`, and a declarative `(predicate, patch)` guardrail table.

### 3. Duplicated control-flow blocks that must change in lockstep
The primary-invoke → `except` → `_should_fallback_to_openai()` → OpenAI-retry → cache set/cancel choreography is **copy-pasted verbatim at 4 call sites** in `core/llm.py` (verified: `_should_fallback_to_openai` has 4 call sites). Any change to resilience policy must be made identically four times.
**Fix:** extract one `invoke_llm_with_policy(...)` orchestrator; collapse ~120 duplicated lines to a single tested path.

### 4. The grounding gate has become an architectural attractor
The provenance/grounding gate now *shapes compute*: `analyzer.py`'s forecast engine emits LLM-prompt strings (`"--- FORECAST SERIES (per year) ---"`) specifically because the gate rejects values the LLM derives itself, and `_append_column_aggregates` exists to feed the grounding corpus. A ~600-line grounding engine lives *inside* `summarizer.py`. Compute decisions are driven by "will the gate accept this number," not by the domain. (This is the structural face of the [[grounding-serialize-computed-series]] and [[forecast-fixed-fx]] issues already in the team's memory.)
**Fix:** extract `agent/grounding.py` as a cohesive cross-cutting module with a clear contract, so the analysis layer stops leaking prompt-shaped strings.

### 5. Stages communicate through an implicitly-contracted, widely-mutated `QueryContext`
~40 `ctx` fields are set in one stage and read in another far away (`effective_answer_kind` set in `pipeline.py` and consumed inside `analyzer.py`; `question_analysis` rewritten in place by 7 guardrails; `evidence_collected` step dicts mutated across 5 functions in 3 modules). Correctness depends on call order and prior in-place edits, with no per-stage schema saying what a stage may assume is populated.
**Fix:** introduce a `Stage` protocol (`run(ctx) -> StageResult`) with typed, read-only-downstream hand-offs.

### 6. DB connection/transaction management is leaking out of the data layer
`main.py` runs raw schema-reflection SQL and owns the table allow-list; `agent/pipeline.py` opens its own `ENGINE.connect()` with `SET TRANSACTION READ ONLY`; `analysis/shares.py` builds and executes SQL inline; `analyzer._build_why_context` fires up to 4 DB round-trips per request. Multiple uncoordinated connection sites are exactly the PgBouncer-exhaustion risk the `knowledge/vector_store.py` lazy-import comment warns about.
**Fix:** centralize the engine in a `core/db.py` (also cuts the `core ⇄ knowledge` cycle), route all SQL through the data layer, and memoize per-request panels.

---

## Prioritized recommendations

Ordered by **(impact × convergence × leverage) ÷ risk**. Every item is a separation of existing behavior; none adds a service or dependency.

### P0 — High leverage, de-risks the planned `llm.py`/`analyzer.py` splits
1. **Move public LLM types + `classify_query_type` out of `core/llm.py`** → `contracts/` (types) and `core/query_classifier.py` (classifier). Breaks both upward edges and unblocks the split. *(Themes 1, 2)*
2. **Introduce a `Provider` strategy + `PROVIDERS` registry.** Collapses `get_primary_llm` / `get_primary_model_name` / `_estimate_cost_usd` / `_provider_from_model_name` to dict lookups; a new provider becomes one registry entry + config. *(Theme 2)*
3. **Extract `invoke_llm_with_policy(...)`** and replace the 4 duplicated fallback blocks. *(Theme 3)*
4. **Introduce a `Stage` protocol + `StageResult`**, and decompose the ~597-line `process_query` into an ordered list of named, testable stages. *(Theme 5)*

### P1 — Structural separations, medium blast radius
5. **Unify the 4 tool-routing tables** behind a single resolver built on the already-shared `agent/router` extractors. *(Theme 2)*
6. **Delete or adopt the dead `analysis/seasonal.py`** (`compute_seasonal_cagr`/`compute_seasonal_comparison` have no external callers) and route the inline forecast CAGR through it; collapse the three CAGR implementations to one. *(Theme 2)* — also a latent **correctness** trap: a fix to `analysis/seasonal.py` changes nothing on the live path.
7. **Extract `agent/grounding.py`** from `summarizer.py`. *(Theme 4)*
8. **Single `SeasonConfig`/`SUMMER_MONTHS`** constant; delete the 6 duplicate/hardcoded definitions. *(Theme 2)*
9. **Centralize DB access** in `core/db.py`; move `refresh_schema_map` out of `main.py` and the `ENGINE.connect()` sites out of `pipeline.py`. Cuts the `core ⇄ knowledge` import cycle structurally. *(Theme 6)*
10. **Convert the 7 `planner.py` guardrails into a declarative `(predicate, patch)` table** evaluated by one applier. *(Theme 2)*

### P2 — Cleanups that compound once P0/P1 land
11. **Delete/quarantine the legacy agent loop** (`orchestrator.py`, `ENABLE_AGENT_LOOP` branch) if Stage 0.2 is permanently active — it is effectively unreachable yet maintained as a second 459-line orchestration engine.
12. **Split `analyzer.py` along its seams** → `forecast_cagr.py`, `why_context.py`, `share_resolution.py`, and move chart-building helpers to the visualization layer.
13. **De-duplicate** `_metric_aliases` (keep `metric_registry.py`), the threshold-rule regex tables, and the scenario-signal tuples.
14. **Re-seam `pipeline` ↔ `evidence_planner`** so the secondary-evidence loop has one owner (removes the lazy back-import).
15. **Type transient/timeout errors** instead of substring-matching `str(exc)`; inject `cache`/`breaker` instead of monkeypatching module globals.

---

## Detailed findings by area

### A. Layering & dependencies — *Medium*

**Pattern compliance:** Layered architecture *partial* · Dependency direction *partial* · Acyclic dependencies *partial* · Separation of concerns *partial*. `contracts/` is **clean (✓, zero outbound)**; no `agent/` imports inside `core/`/`analysis/`.

**Violations**
- `visualization/chart_selector.py:15` — `from core.llm import classify_query_type` — presentation layer reaches **up** into the LLM layer; binds chart selection to the LLM module's lifecycle and config.
- `guardrails/redteam_gate.py:22-23` — imports `agent.summarizer` **and** `core.llm.SummaryEnvelope` (lower→higher). Mitigated: it's a standalone manually-run gate (`python -m guardrails.redteam_gate`), not in the request hot path.
- `core/llm.py:31,85` ⇄ `knowledge/vector_store.py:154` — real **`core ⇄ knowledge` package cycle**, defused only by a lazy in-function import of `core.query_executor.ENGINE`. A routine "hoist imports" cleanup reintroduces an import-time crash.
- `agent/pipeline.py:27` ⇄ `agent/evidence_planner.py:907` — intra-layer cycle masked by a lazy back-import of `pipeline._run_secondary_evidence_loop`.
- `main.py:235-258` (`refresh_schema_map`) — the HTTP layer executes raw `pg_matviews`/`pg_attribute` reflection SQL and owns the schema map / `ALLOWED_TABLES` allow-list; unusable by non-HTTP callers without importing `main`.
- `agent/pipeline.py:~1012, ~1689` — orchestration opens raw `ENGINE.connect()` + `SET TRANSACTION READ ONLY`, duplicating the pattern that correctly lives in `agent/tools/common.py:76-91`.
- `analysis/shares.py:17,117` — builds SQL and executes it inline rather than via the SQL layer (downward import, so least severe).

**Long-term implications:** the hub coupling on `core/llm.py` is the dominant risk for the planned split; the two lazy-import cycles are latent regressions (a constraint enforced only by convention). Centralizing DB access reduces both layering debt and a concrete connection-exhaustion risk under load.

---

### B. Core LLM seam — *High*

**Pattern compliance:** Single Responsibility *✗* · Open/Closed *✗* · Strategy/provider abstraction *✗* · Separation of cross-cutting concerns *partial* (helpers exist but each entry point hand-wires the same 5-step dance).

**Responsibilities bundled in `core/llm.py` (3,387 lines):** provider resolution · cost estimation + usage logging · resilience/circuit-breaking + per-function fallback · cache lifecycle · query-classification heuristics · prompt construction + ~100 static prompt literals · prompt-budget glue + re-export surface.

**Violations**
- `core/llm.py:133` `_provider_from_model_name` — provider identity by `str.startswith` on model names; a new NVIDIA NIM id or renamed OpenAI `o*` model silently misclassifies → wrong cost bucket and wrong breaker. Root cause forcing every other branch.
- `core/llm.py:~1251-1281` **duplicated verbatim at ~1779, ~2809, ~3338** — the primary→`except`→`_should_fallback_to_openai()`→OpenAI-retry→cache set/cancel block across all 4 `llm_*` functions. *(Verified: `_should_fallback_to_openai` = 4 call sites.)*
- **Open/Closed — adding a 4th provider = 6–7 edits:** `llm_runtime.py:~220,266` (factory+singleton) · `llm.py:~250-254` (`get_primary_llm`) · `llm.py:~263-267` (`get_primary_model_name`) · `llm.py:~145-157` (`_provider_from_model_name`) · `llm.py:~171-177` (`_estimate_cost_usd`) · `config.py:~70-74,185-190` (model/key/url + cost constants) · `llm.py:~236-238` (`make_*` alias).
- `core/llm.py:216` `llm_cache = LLMResponseCache(...)` — module-global singleton justified in-comment by "tests monkeypatch `core.llm.llm_cache`"; cache/breaker are ambient, not injected.
- `core/llm.py` — ~100 triple-quoted prompt/policy blocks (`FEW_SHOT_SQL:~558`, `_ANALYZER_*_RULES:~1844-1941`, truncation/ordering tables `~2332-2704`) interleaved with provider dispatch and cost math in one git-blame surface.
- `core/llm.py:~3356` — timeout/transient classification by substring (`"deadline"`, `"504"`, `"timed out"` in `str(exc).lower()`).

**Good (do not regress):** `core/sql_generator.py` (clean 3-phase sanitize→whitelist→validate, no LLM imports), `core/query_executor.py` (pure execution + breaker), `core/llm_payloads.py` & `core/prompt_budget.py` (provider-agnostic), `core/llm_runtime.py` (correct implementation layer). The split is real — just incomplete.

**Long-term implications:** every provider/resilience/cache change is a guarded multi-site edit; the duplicated fallback guarantees the 4 entry points eventually drift. A registry + strategy turns "add a provider" into an additive, testable one-liner.

---

### C. Orchestration & planning — *High*

**Pattern compliance:** Single Responsibility *✗* · Pipeline/state-machine clarity *partial* · No duplicated routing *✗* · Explicit stage contracts *✗*.

**Violations**
- `agent/pipeline.py:1976-2573` — `process_query` (~597 lines) inlines clarify-rewrite, Stages 0/0.2/0.3 prep, answer-kind cross-check, vector-tier + adjacency traces, evidence-plan build/execute, the legacy agent-loop branch, the SQL-fallback branch, provenance re-stamps, and Stages 3/4/5. No seam to test or replace a single stage.
- `agent/pipeline.py:1694-1973` — `_execute_evidence_plan` (~280 lines) with `try/except` up to 4 levels deep; a two-axis state (`primary_is_analyzer_source` × `primary_is_plan_or_router_source`) gates ~12 metric/trace/recovery combinations — a hand-rolled state machine discoverable only by reading every branch.
- **Four overlapping tool-routing implementations:** `router.py:507-598` (`match_tool`) · `planner.py:2136-2188` (`build_tool_invocation_from_analysis`) + `resolve_tool_params:2191-2384` · `evidence_planner.py:194-496` (`_expand_evidence_steps`/`_add_steps_from_rules`) · `orchestrator.py:209-223` (`_score_dataset_for_query`). Adding a tool/alias/keyword touches up to 4 unsynchronized tables.
- `agent/pipeline.py:1321-1400` (`_pick_primary_invocation`) — the `plan→keyword_router→analyzer→router_fallback` chain re-invokes `match_tool` after the analyzer was already authoritative; duplication is now structural.
- **Legacy dual-path partially dead:** `agent/pipeline.py:2379-2416` runs `orchestrator.run_agent_loop` only under a 5-condition guard including `not has_authoritative_question_analysis`; with Stage 0.2 active this branch (and its inline relevance re-validation) is effectively unreachable, yet `orchestrator.py` (459 lines) is maintained and re-tested as live code with its own `_attach_dataset`/provenance copies.
- **Shared mutable `QueryContext`:** `effective_answer_kind` set at `pipeline.py:2113`, consumed in `analyzer.py`; `question_analysis` rewritten in place (`pipeline.py:601,1238` + planner guardrails); `evidence_collected`/`evidence_plan` step dicts mutated across `_execute_evidence_step`, `_run_secondary_evidence_loop`, `merge_evidence_into_context`, and read again in `orchestrator.run_agent_loop`; provenance re-stamped in 3 places.
- **`planner.py` SRP:** language/mode/conceptual detection + structured-analysis invocation + **7 in-place `_apply_*_guardrail` rewriters chained unconditionally** (`analyze_question:1866-1901`) + plan projection + legacy plan/SQL gen + tool-invocation bridge. The guardrail pattern is an open-ended sink that grows with every misroute.
- **Per-site divergent error handling:** analyzer failure nulls `question_analysis` and continues; tool failure branches on source; vector-store failure substring-matches the error to trip a breaker; enrichment failure logs and returns `ctx` unchanged. No uniform `ok | degraded | failed` result type.

**Good (do not regress):** consistent stage-tracing/metrics discipline; shared `agent/router` extractors (the right instinct — make them the *only* routing authority); `evidence_validator.py` is a clean single-responsibility module — the model to refactor toward.

**Long-term implications:** the unit of change (a query capability) is not the unit of code — one new `answer_kind`/tool touches ~5 modules. New contributors cannot trace a query's path without reading ~5,000 lines.

---

### D. Enrichment & summarization — *High*

**Pattern compliance:** Single Responsibility *✗* · Reuse of `analysis/` layer *✗* · Separation of compute vs. narrative *partial* · DRY *✗*.

**Violations**
- **`analysis/` layer bypassed (central defect):** `analysis/seasonal.py:85,145` (`compute_seasonal_cagr`/`compute_seasonal_comparison`) have **no external importers** *(verified: only `compute_seasonal_average` is imported, by `main.py:57`)*, while CAGR is reimplemented inline in `analyzer.py:1988-2307` and again in `analysis/stats.py:206` — three implementations, the canonical one dead. Correlation likewise has 3 paths (`analyzer.py:448`, `analyzer.py:2446-2488`, `analysis/shares.py:22`).
- **`analyzer.py` SRP (3,407 lines, ~13 concerns):** share-shift narrative (`:97`), causal pressure (`:268`), correlation compute (`:448`), regex scenario extraction (`:511`), **chart-building (`:879-1162`, belongs in visualization)**, deterministic answer-string formatting (`:1496`), 450-line forecast engine (`:1856`), 430-line `_build_why_context` (`:2858`), trendline pre-calc (`:3290`) — sharing one 300-line `enrich()` orchestrator (`:2317`).
- **DRY:** `_metric_aliases` duplicated (`analyzer.py:376` + `metric_registry.py:48`); `SUMMER_MONTHS` defined twice with different types (`config_metrics/metric_config.py:91` tuple vs `config.py:353` list) + hardcoded `[4,5,6,7]` at `seasonal.py:67,118,191` and `stats.py:199`; threshold-regex tables duplicated (`analyzer.py:1458` vs `summarizer.py:1657`); scenario-signal tuples duplicated across the module boundary (`analyzer.py:67` vs `summarizer.py:121`).
- **Grounding engine wedged into the summarizer:** `summarizer.py:332-635,856` (`_tokenize_cell_value`, `_build_grounding_tokens`, `_is_summary_grounded`, `_enforce_provenance_gate`) — ~600 lines of numeric-matching with no narrative responsibility. Compute is shaped by it: `analyzer.py:1971-2306` emits prompt strings because the gate rejects LLM-derived values (comments cite prod traces at `:1955,:2284`); `_append_column_aggregates:2677` exists to feed the grounding corpus.
- **Summarizer coupling:** `summarizer.py:18` imports the **private** `_extract_forecast_horizon` from `analyzer`; `:2015` re-imports pandas locally "to avoid import-order issues"; three render paths coexist (`generic_render`, `llm_summarize*`, and hand-rolled `_build_*_direct_answer` at `:1450,1569,1750`) behind a 6-branch `summarize_data` dispatcher (`:2319`), with `_build_scenario_fallback_answer` re-formatting the same ScenarioFrame `generic_render` already handles.
- **Scaling hotspots:** `_build_why_context` (`:2858`) issues up to 4 DB round-trips/request and `pd.concat`s a growing frame 6× (`:3174-3221`) with no caching; `summarizer.py:340-343` materializes/stringifies `ctx.df.head(200)` multiple times per request; repeated `df.copy()` (`analyzer.py:1867,2649,2866,3302`); correlation can run in-memory compute **and** a fresh SQL panel in the same request (worst-case double compute).

**Long-term implications:** the dead-but-duplicated `analysis/` layer is a correctness trap (a fix there changes nothing live). Dated patch archaeology (Fix D/F, Phase G/H; Phase H landed 2026-06-28) shows changes are already hard to land. Throughput is bounded by per-request DB fan-out + repeated full-frame pandas work; the layering fix (one analysis layer + cached panels) is also the throughput fix.

---

## What's working well (preserve through refactors)

- **`contracts/` is a clean dependency sink** — zero outbound layer imports. Keep it that way; it's the natural home for `SummaryEnvelope` & friends.
- **The layering spine holds** — no `agent/` imports in `core/`/`analysis/`; `core/sql_generator.py` and `core/query_executor.py` are well-bounded with no LLM coupling.
- **Real, if incomplete, extractions** — `llm_runtime.py`, `llm_payloads.py`, `prompt_budget.py` are provider-agnostic and correctly separated.
- **`evidence_validator.py`** — clean single-responsibility; the template for the larger files.
- **Stage tracing/metrics discipline** is consistent and genuinely valuable.
- **Shared `agent/router` deterministic extractors** — the right instinct; promote them to the single routing authority.

---

## Grounding & verification

Findings were produced with `file:line` citations and key claims were re-checked against source on 2026-06-30:
- `core/llm.py` — `_provider_from_model_name` @133, `llm_cache = LLMResponseCache(...)` @216, `_should_fallback_to_openai` @270 with **4 call sites** confirmed (5 total occurrences). ✓
- **Dead `analysis/seasonal.py`** — `compute_seasonal_cagr`/`compute_seasonal_comparison` referenced only inside `seasonal.py`; only `compute_seasonal_average` imported externally (`main.py:57`). ✓
- **Upward edges** — `visualization/chart_selector.py:15` and `guardrails/redteam_gate.py:23` both import from `core.llm`. ✓
- **Line counts** — `wc -l`: `llm.py` 3,387 · `analyzer.py` 3,407 · `pipeline.py` 2,573 · `summarizer.py` 2,531 · `planner.py` 2,384. ✓

Line numbers in the more granular citations are accurate to the reviewed revision and may drift by a few lines as the files change; the cited **symbols** are the durable anchors. A handful of deep citations were not individually re-verified — treat any single line number as a pointer, not a contract.

---

## Suggested sequencing

`P0-1` (move types/classifier to `contracts/` + `core/query_classifier.py`) unblocks both the `core/llm.py` decomposition and the upward-edge fixes, so do it first. `P0-4` (Stage protocol) unblocks most of area C. `P1-6` (dead `seasonal.py`) is small, high-signal, and removes a correctness trap — good early win. The legacy-loop removal (`P2-11`) should wait until the Stage protocol lands so its absence is provably safe.
