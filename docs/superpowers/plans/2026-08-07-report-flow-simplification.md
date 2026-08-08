# Report Flow Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse report generation from 8 LLM calls across two parallel code paths into 2 LLM calls on one path — plan once, enrich deterministically, write once — deleting the machinery that becomes redundant.

**Architecture:** The research planner already produces structured per-topic intent. Today that intent is serialized back into a natural-language question (`build_report_track_analysis_query`) and re-derived by a second LLM inside a nested `process_query` — once per track. This plan injects the planner's intent directly into the pipeline so Stage 0.2 is skipped, dedupes tracks that resolve to the same tool and window, then replaces the analysis/synthesis writer pair with a single document writer. The deterministic analytical engine (Stage 0.3 vector knowledge, Stage 0.4–0.8 evidence planning and tool execution, Stage 3 enrichment: seasonal statistics, correlations, why-context, derived chart specs) is untouched throughout — it is the value, and it needs no LLM.

**Tech Stack:** Python 3.11 (container) / 3.14 (local), pydantic v2 contracts, pytest, ruff, OpenAI `gpt-5.6-terra` via LangChain.

## Global Constraints

- Full gate before every commit: `python -m pytest tests/ -q` (2501 tests at plan time), `ruff check .` (must print "All checks passed!"), `python -m guardrails.redteam_gate` (score ≥ 0.92).
- Run pytest from `D:\Enaiapp\langchain_railway`. Do **not** pre-set `SUPABASE_DB_URL`/`ENAI_*`/`OPENAI_API_KEY` in the environment — test modules set their own via `os.environ.setdefault` and exported values break 16 tests in `test_main.py`.
- Local Python is 3.14, the container is 3.11. Keep `from __future__ import annotations` at the top of every module touched; `tests/test_runtime_annotations.py` guards this.
- Every behaviour-changing phase ships behind an `ENABLE_*` flag defaulting to `false`, using the existing convention in `config.py`: `os.getenv("NAME", "false").lower() in ("1", "true", "yes", "on")`. Tests that pin either side must `monkeypatch.setattr(module, "FLAG", bool)` — never rely on the process environment.
- The suite must pass with each new flag both off and on before that phase is committed.
- Never commit to `main` (auto-deploys). Work on `agent/track-scoped-report-analysis` or a branch off it.
- Diagnostic logging may contain schema field names, counts, and enum values — never evidence values, prose, or claim payloads. Use `_diagnostic_identifier` for any interpolated string.

---

## File Structure

**Phase 1 deletes (legacy v1 report path, unreachable in production where `REPORT_PIPELINE_V2_MODE=enabled`):**
- Delete: `agent/report_planner.py` (374 lines)
- Delete: `agent/report_assembly.py` (172 lines)
- Delete: `agent/report_evaluation.py` (74 lines)
- Modify: `agent/report_charts.py` — remove `build_report_charts` and the `ReportPlan`-shaped half; keep `build_report_research_exhibits` and every `_omitted`/`_built` helper
- Modify: `core/report_job_processor.py` — remove `_run_legacy_bound_attempt`, `_run_shadow_research_planner`, and the v1 constructor seams
- Modify: `contracts/report_generation.py` — drop checkpoint contract versions v1 and v2
- Modify: `core/llm.py` — remove `llm_plan_report`
- Modify: `config.py` — remove `REPORT_PIPELINE_V2_MODE`

**Phase 2–3 modify (evidence collection):**
- Modify: `agent/pipeline.py` — add the analysis-injection seam at Stage 0.2
- Modify: `agent/report_research_execution.py` — build an analysis spec instead of a prose query; delete `build_report_track_analysis_query`
- Create: `agent/report_track_specs.py` — maps a `ReportResearchTrack` to a `QuestionAnalysis`, and dedupes tracks by resolved tool + window. New file because this is the one genuinely new responsibility, and `report_research_execution.py` is already 1,140 lines.
- Modify: `core/report_job_processor.py` — `_run_track_analysis` runs deduped work

**Phase 4 deletes (two-batch writer):**
- Modify: `agent/report_document_generation.py` — delete `_materialize_section_batch`, `_repair_section_batch`, `_repair_section_batch_until_valid`, and the batch branch of `generate_report_document` (~400 of 1,243 lines)
- Modify: `core/llm.py` — delete `_llm_write_report_section_batch`, `llm_write_report_analysis_sections`, `llm_write_report_synthesis_sections`

**Phase 5 modifies (citation obligation):**
- Modify: `contracts/report_document.py` — add `optional_evidence_refs`
- Modify: `agent/report_sections.py` — split the allowed set from the required set
- Modify: `agent/report_document_planner.py` — shared narrative refs become optional

---

## Phase 1 — Delete the legacy report path

Production runs `pipeline_v2_mode=enabled` (confirmed in the worker startup line). The v1 path survives only as a config fallback and to finish jobs checkpointed on `report-generation-checkpoint-v1`/`v2`. Deleting it first means every later phase touches one code path instead of two.

### Task 1.1: Confirm the legacy path is drained

**Files:** none (operational precondition)

**Interfaces:**
- Produces: a go/no-go for Tasks 1.2–1.6.

- [ ] **Step 1: Confirm production is on v2**

Check the most recent worker boot line in Railway logs:

```
Report worker started. ... pipeline_v2_mode=enabled ...
```

If it does not say `enabled`, **stop** — the rest of Phase 1 deletes the path currently serving users.

- [ ] **Step 2: Confirm no queued job carries a legacy checkpoint**

`public.report_jobs` is reached only through Postgres RPCs
(`lease_report_job_v1`, `heartbeat_report_job_v1`), so its shape appears
nowhere in the Python source. Verified column names, 2026-08-07 — do not guess
these:

| column | type | note |
|---|---|---|
| `state` | text | **not** `status`; values from `ReportJobState` — `queued`, `running`, `completed`, `failed`, `cancelled` |
| `checkpoint_payload` | jsonb | **not** `checkpoint`; holds the `contract_version` this task cares about |
| `contract_version` | text | the *job* contract, a different thing from the checkpoint's — do not read this one |
| `phase`, `attempt_count`, `error_code`, `result_payload` | | |

```sql
SELECT state,
       checkpoint_payload->>'contract_version' AS checkpoint_version,
       count(*)
FROM report_jobs
WHERE state IN ('queued', 'running')
GROUP BY 1, 2
ORDER BY 1, 2;
```

Expected: no row whose `checkpoint_version` is
`report-generation-checkpoint-v1` or `-v2`. If one exists, wait for that job to
finish or fail out, then re-check. Deleting the legacy path while one is in
flight fails it permanently — `REPORT_CHECKPOINT_INVALID` is not retryable.

- [ ] **Step 3: Record the baseline**

Run the May 2026 report and save the full log to `docs/evidence/`. Every later phase is judged against it: LLM call count, `REPORT_TRACK_ANALYSIS_ENABLED` payload, section word counts, chart decisions, and the report text itself.

### Task 1.2: Delete the legacy planner, assembler, and evaluator

**Files:**
- Delete: `agent/report_planner.py`
- Delete: `agent/report_assembly.py`
- Delete: `agent/report_evaluation.py`
- Delete: `tests/test_report_planner.py`, `tests/test_report_assembly.py` (there is no `test_report_evaluation.py`; `evaluate_report_plan` is covered indirectly by `tests/test_report_charts.py`)
- Modify: `core/report_job_processor.py` (imports at lines 17–54, constructor defaults at lines 270–274)

**Interfaces:**
- Consumes: nothing.
- Produces: `ReportJobProcessor.__init__` no longer accepts `planner`, `evaluator`, `assembler`, or `chart_builder`.

- [x] **Step 1: Find every importer** — DONE 2026-08-07

```bash
grep -rn "report_planner\|report_assembly\|report_evaluation" --include=*.py .
```

**Actual finding, wider than estimated:** `tests/test_report_planner.py` is a
fixture hub for **nine** modules, not two — `test_report_assembly`,
`test_report_charts`, `test_report_document_contract`,
`test_report_job_processor`, `test_report_plan_llm`, `test_report_projection`,
`test_report_result`, `test_report_sections`, `test_report_section_llm`. The
exported surface is `_manifest`, `_plan_payload`, **and `TABLE_REF`** (which
this plan originally missed); `_manifest` and `_plan_payload` also close over
`STATS_REF` and `LIMIT_REF`, so all five names move together.

- [x] **Step 2: Move the surviving fixtures before deleting their home** — DONE (commit 1b79c8d)

Create `tests/fixtures_report_manifest.py` containing `_manifest()` and `_plan_payload()` copied verbatim from `tests/test_report_planner.py`, then update the importers:

```python
from tests.fixtures_report_manifest import _manifest, _plan_payload
```

- [x] **Step 3: Run the suite to confirm the move is clean** — DONE, 2501 passed

Run: `python -m pytest tests/ -q`
Expected: PASS, same count as before.

- [x] **Step 4: Commit the fixture move on its own** — DONE (commit 1b79c8d)

```bash
git add tests/
git commit -m "Move report manifest fixtures out of the legacy planner test module"
```

- [ ] **Step 5: Delete the three modules and their tests**

```bash
git rm agent/report_planner.py agent/report_assembly.py agent/report_evaluation.py
git rm tests/test_report_planner.py tests/test_report_assembly.py
```

- [ ] **Step 6: Remove the processor's v1 seams**

In `core/report_job_processor.py`, delete the imports of `assemble_report`, `ReportAssemblyError`, `evaluate_report_plan`, `plan_report`, `build_report_charts`, and the constructor parameters `planner`, `evaluator`, `assembler`, `chart_builder` along with the `self._planner` / `self._evaluator` / `self._assembler` / `self._chart_builder` assignments.

- [ ] **Step 7: Run the suite and fix the fallout**

Run: `python -m pytest tests/ -q`
Expected: failures only in tests that construct `ReportJobProcessor` with the removed kwargs. Delete those arguments; do not reintroduce the parameters.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Delete the legacy report planner, assembler, and evaluator"
```

### Task 1.3: Delete the legacy attempt path and its shadow

**Files:**
- Modify: `core/report_job_processor.py:1548-1571` (`_run_bound_attempt`), `_run_legacy_bound_attempt`, `_run_shadow_research_planner` (starts line 845)

**Interfaces:**
- Consumes: Task 1.2's processor.
- Produces: `_run_bound_attempt` delegates unconditionally to `_run_v2_bound_attempt`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_report_job_processor.py`:

```python
def test_every_attempt_runs_the_document_pipeline():
    """One path. A checkpoint from the retired pipeline is rejected, not run."""
    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: pytest.fail(
            "no legacy enrichment path remains"
        ),
    )
    lease = _lease(query=_V2_QUERY)
    lease.checkpoint = {"contract_version": "report-generation-checkpoint-v2"}

    with pytest.raises(ReportJobFailure) as failure:
        processor(lease, _Control())

    assert failure.value.error_code == "REPORT_CHECKPOINT_INVALID"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_report_job_processor.py::test_every_attempt_runs_the_document_pipeline -q`
Expected: FAIL — the legacy branch runs instead of raising.

- [ ] **Step 3: Collapse `_run_bound_attempt`**

Replace the body at `core/report_job_processor.py:1553-1571` with:

```python
    def _run_bound_attempt(
        self,
        lease: ReportJobLease,
        control: ReportJobExecutionControl,
    ) -> dict[str, Any]:
        return self._run_v2_bound_attempt(lease, control)
```

Then delete `_run_legacy_bound_attempt` and `_run_shadow_research_planner` entirely.

- [ ] **Step 4: Reject retired checkpoint versions**

In `contracts/report_generation.py`, remove `"report-generation-checkpoint-v1"` and `"report-generation-checkpoint-v2"` from the `contract_version` Literal (lines 34–36) and delete the two migration branches at lines 63 and 78 and the guard at line 143. A lease carrying a retired version now fails `model_validate`, which `_run_v2_bound_attempt` already converts to `REPORT_CHECKPOINT_INVALID`.

- [ ] **Step 5: Run the new test, then the suite**

Run: `python -m pytest tests/test_report_job_processor.py::test_every_attempt_runs_the_document_pipeline -q`
Expected: PASS

Run: `python -m pytest tests/ -q`
Expected: PASS after deleting tests that exercised the legacy attempt path. Delete them — do not adapt them; they test code that no longer exists.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Delete the legacy report attempt path and its shadow planner"
```

### Task 1.4: Delete the v1 chart builder and the v1 plan contract

> **RE-PLANNED 2026-08-07 after Task 1.2 Step 1 — DO NOT EXECUTE AS WRITTEN.**
>
> Step 1 of Task 1.2 found coupling this task did not anticipate. `ReportPlan`
> is not confined to the v1 orchestration path: it is the **test vehicle for
> `agent/report_sections.py`**, whose `validate_report_section` is the live v2
> grounding gate. `tests/test_report_sections.py` (600 lines) and
> `tests/test_report_section_llm.py` both build `ReportPlan.model_validate(
> _plan_payload())` to exercise it, and `tests/test_report_charts.py` does the
> same for rendering behaviour that `build_report_research_exhibits` shares.
>
> Deleting the contract therefore means rewriting the safety net of the most
> failure-prone code in the subsystem — the word bounds, grounding, and claim
> validation that have broken four ways this month. That is not a deletion; it
> is a test migration, and it must not ride along inside a phase whose stated
> value is "zero behaviour change".
>
> **Split as follows:**
>
> - **Task 1.4a (in scope, safe):** delete the legacy *orchestration* only —
>   `_run_legacy_bound_attempt`, `_run_shadow_research_planner`,
>   `agent/report_planner.py`, `agent/report_assembly.py`,
>   `agent/report_evaluation.py`, `llm_plan_report`, checkpoint versions v1/v2,
>   `REPORT_PIPELINE_V2_MODE`. **Keep** `contracts/report.py::ReportPlan`,
>   `agent/report_sections.py::generate_report_sections`, and
>   `agent/report_charts.py::build_report_charts`. Delete only
>   `tests/test_report_planner.py`, `tests/test_report_assembly.py`,
>   `tests/test_report_plan_llm.py`, `tests/test_report_result.py`.
>   Expected: about −900 lines, still no production behaviour change.
>
> - **Task 1.4b (deferred, own plan):** migrate `tests/test_report_sections.py`
>   and `tests/test_report_charts.py` from `ReportPlan` to
>   `ReportDocumentPlan`/`ReportDocumentSectionSpec`, then delete `ReportPlan`,
>   `generate_report_sections`, and `build_report_charts`. Test-only risk, but
>   it is the coverage that guards the live gate, so it deserves its own
>   review rather than a footnote in a deletion phase.
>
> Cosmetically deleting a contract is worth far less than the coverage it would
> put at risk. 1.4b is optional; 1.4a is not.

**Files:**
- Modify: `agent/report_charts.py:865` (`build_report_charts`)
- Modify: `contracts/report.py` — remove `ReportPlan`, `ReportSectionSpec`, `ReportPlanningContext`, `required_report_section_sequence`, `normalize_report_plan_semantics`, `normalize_report_plan_word_budget`, `report_aggregate_word_bounds`
- Modify: `core/llm.py:3846` (`llm_plan_report`)

**Interfaces:**
- Consumes: Task 1.3's processor.
- Produces: `contracts/report.py` retains only what the document pipeline uses — `REPORT_MAX_EXHIBITS`, `REPORT_SECTION_MIN_WORDS`, `REPORT_SECTION_MAX_WORDS`, `report_section_word_floor_ratio`, `report_section_prompt_word_bounds`, `report_section_validation_word_bounds`, `ReportChartRequest`, `ReportChartPurpose`, `ReportIntent`, `ReportSectionKind`.

- [ ] **Step 1: Verify each symbol is v1-only before removing it**

For every name listed above:

```bash
grep -rn "<symbol>" --include=*.py . | grep -v tests
```

If the only hits are in the files this task deletes, remove it. If `agent/report_document_planner.py`, `agent/report_document_generation.py`, `agent/report_charts.py::build_report_research_exhibits`, or `core/llm.py`'s document functions reference it, **keep it** and note why in the commit message. `report_section_validation_word_bounds` in particular is used by the live gate — do not remove it.

- [ ] **Step 2: Delete `build_report_charts` and keep `build_report_research_exhibits`**

Both live in `agent/report_charts.py`. `build_report_charts` takes a `ReportPlan`; `build_report_research_exhibits` takes packets and a manifest and is the v2 entry point. The `_built`, `_omitted`, `_axis_metadata`, `_comparison_projection` helpers are shared — keep all of them.

- [ ] **Step 3: Run the suite**

Run: `python -m pytest tests/ -q`
Expected: PASS after deleting `tests/test_report_charts.py` cases that call `build_report_charts` directly. Any case testing chart *rendering* behaviour must be rewritten against `build_report_research_exhibits` rather than deleted — those cover real regressions (`test_composition_never_pies_columns_that_share_no_unit`, `test_a_two_dimensional_frame_is_pivoted_not_drawn_under_repeated_labels`).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Delete the v1 report plan contract and its chart builder"
```

### Task 1.5: Remove the pipeline-mode flag

**Files:**
- Modify: `config.py:128-137` (`REPORT_PIPELINE_V2_MODE`, `REPORT_TRACK_ANALYSIS_MODE` default derivation)
- Modify: `core/report_job_processor.py:277`, `:311-313`, `:341`, `:582`

**Interfaces:**
- Consumes: Task 1.4.
- Produces: `ReportJobProcessor.__init__` no longer accepts `pipeline_v2_mode`.

- [ ] **Step 1: Delete the flag and its validation**

Remove `REPORT_PIPELINE_V2_MODE` from `config.py` and the `report_pipeline_v2_mode` argument and check from `validate_runtime_settings`. `REPORT_TRACK_ANALYSIS_MODE` currently defaults from it — change that default to the literal `"enabled"`.

- [ ] **Step 2: Remove it from the processor and its telemetry**

Delete the constructor parameter, the `disabled/shadow/enabled` validation, `self._pipeline_v2_mode`, and the `"pipeline_v2_mode"` key from the attempt telemetry payload at line 582.

- [ ] **Step 3: Remove it from the worker startup line**

In `report_worker.py`, drop `pipeline_v2_mode=%s` and its argument. Keep `partial_track_evidence=%s`.

- [ ] **Step 4: Full gate**

Run: `python -m pytest tests/ -q`
Run: `ruff check .`
Run: `python -m guardrails.redteam_gate`
Expected: all pass; redteam score ≥ 0.92.

- [ ] **Step 5: Commit and measure**

```bash
git add -A
git commit -m "Remove the report pipeline mode flag now that one path remains"
```

Record the line-count delta: `git diff --stat <phase-1-base>..HEAD`. Expected: roughly −1,200 lines with no production behaviour change.

---

## Phase 2 — Inject the analysis spec, delete the prose round-trip

The planner's structured intent currently becomes prose and is re-parsed by an LLM once per track. This phase passes the intent directly.

### Task 2.1: Add the analysis-injection seam to the pipeline

**Files:**
- Modify: `agent/pipeline.py:3144-3153` (Stage 0.2), `agent/pipeline.py:3396` (`process_query` signature), `agent/pipeline.py` (`_process_query_impl` signature)
- Test: `tests/test_pipeline_analysis_injection.py` (create)

**Interfaces:**
- Produces: `process_query(..., question_analysis: QuestionAnalysis | None = None)`. When supplied, Stage 0.2 sets `ctx.question_analysis` from it, sets `ctx.question_analysis_source = "injected"`, and makes no LLM call. Every downstream stage reads `ctx.question_analysis` exactly as today.

- [ ] **Step 1: Write the failing test**

```python
def test_an_injected_analysis_skips_the_analyzer_call(monkeypatch):
    """The planner already decided. Re-deriving it costs a call per track."""
    from agent import planner as planner_module

    monkeypatch.setattr(
        planner_module,
        "analyze_question_active",
        lambda _ctx: pytest.fail("an injected analysis must not be re-derived"),
    )
    analysis = _question_analysis(query_type="data_explanation")

    ctx = process_query(
        "What was the May 2026 balancing price?",
        answer_mode="report",
        question_analysis=analysis,
    )

    assert ctx.question_analysis == analysis
    assert ctx.question_analysis_source == "injected"
```

Build `_question_analysis` with the existing helper in `tests/test_semantic_lock.py::_make_qa` — import it rather than duplicating the construction.

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_pipeline_analysis_injection.py -q`
Expected: FAIL with `TypeError: process_query() got an unexpected keyword argument 'question_analysis'`.

- [ ] **Step 3: Thread the parameter through**

Add `question_analysis: QuestionAnalysis | None = None` to `process_query` and `_process_query_impl`, and set it on the context before Stage 0.2 runs.

- [ ] **Step 4: Skip Stage 0.2 when the analysis is present**

At `agent/pipeline.py:3144`, guard the stage:

```python
    # Stage 0.2: structured question analysis
    if ctx.question_analysis is not None:
        ctx.question_analysis_source = "injected"
        _trace_stage(
            "stage_0_2_question_analyzer",
            time.time(),
            mode="injected",
            ok=True,
            error=False,
            query_type=ctx.question_analysis.classification.query_type.value,
            preferred_path=ctx.question_analysis.routing.preferred_path.value,
        )
    elif ENABLE_QUESTION_ANALYZER_SHADOW or ENABLE_QUESTION_ANALYZER_HINTS:
        ...  # existing body unchanged
```

Emitting the trace under `mode="injected"` keeps the stage visible in every trace; a silently absent stage reads as a pipeline that skipped analysis rather than one that was handed it.

- [ ] **Step 5: Run the new test and the suite**

Run: `python -m pytest tests/test_pipeline_analysis_injection.py -q`
Expected: PASS

Run: `python -m pytest tests/ -q`
Expected: PASS — the parameter defaults to `None`, so no existing caller changes.

- [ ] **Step 6: Commit**

```bash
git add agent/pipeline.py tests/test_pipeline_analysis_injection.py
git commit -m "Let a caller supply the question analysis instead of paying for it twice"
```

### Task 2.2: Map a research track to an analysis spec

> **RE-PLANNED 2026-08-07 after reading the contracts — DO NOT EXECUTE AS WRITTEN.**
>
> The task assumed `requested_metrics` translates into
> `analysis_requirements.derived_metrics`. It does not. The two are different
> vocabularies:
>
> | a track's `requested_metrics` | the analyzer's `DerivedMetricName` |
> |---|---|
> | `average_price`, `minimum_price`, `maximum_price`, `percent_change`, `import_dependency_ratio`, `generation_mix` | `mom_absolute_change`, `mom_percent_change`, `yoy_absolute_change`, `yoy_percent_change`, `share_delta_mom`, `correlation_to_target`, `trend_slope`, … |
>
> They barely overlap, and the gap is not cosmetic. `percent_change` cannot say
> whether it means month-on-month or year-on-year — **only the research
> question text carries that**, which is exactly what the per-track analyzer
> reads today ("how did it change from April 2026?" → `mom_*`). A mapping built
> from `requested_metrics` and `evidence_mode` provably cannot reproduce it.
>
> Getting it wrong is not a test failure, it is silent report degradation:
> emitting no derived metrics strips the MoM/YoY analysis, the why-context, and
> the derived charts; emitting the wrong ones re-triggers
> `missing_evidence_for_metrics`, the defect just fixed in Phase 0. And
> `query_type`, `preferred_path`, and `answer_kind` all steer routing, which the
> phased-audit rules say requires explicit disagreement review before cutover.
>
> **Corrected approach — contract first, then shadow:**
>
> - **Task 2.2a:** extend `ReportResearchTrack` with the analysis fields the
>   planner is already positioned to decide (`query_type`, `preferred_path`,
>   `answer_kind`, `derived_metrics`) and teach `llm_plan_report_research` to
>   emit them. An LLM still makes the semantic call — once, for the whole
>   report, in a call already being paid for — instead of four times from
>   re-parsed prose.
> - **Task 2.2b:** build `report_track_analysis_spec` over those fields, and run
>   it in **shadow**: the nested analyzer still decides, the spec is computed
>   alongside, and disagreements are logged
>   (`REPORT_TRACK_SPEC_DISAGREEMENT`, naming field, planner value, analyzer
>   value — enum values only, never query text).
> - **Task 2.2c:** review the disagreements on real reports. Cut over only when
>   they are understood, and keep the flag.
>
> This costs one extra deploy-and-observe cycle and removes the guesswork from
> the step that decides what every report analyses.

**Files:**
- Create: `agent/report_track_specs.py`
- Test: `tests/test_report_track_specs.py` (create)

**Interfaces:**
- Consumes: `ReportResearchTrack` from `contracts/report_research.py`.
- Produces: `report_track_analysis_spec(track: ReportResearchTrack, report_query: str) -> QuestionAnalysis` and `report_track_query(track: ReportResearchTrack) -> str` (the plain question text, used only for `ctx.query` display — no longer parsed by anything).

- [ ] **Step 1: Write the failing test**

```python
def test_a_track_spec_carries_the_planner_intent_without_re_asking():
    track = _plan().tracks[0]

    spec = report_track_analysis_spec(track, "May 2026 report")

    assert spec.classification.query_type.value in {
        "data_retrieval",
        "data_explanation",
        "comparison",
    }
    assert [
        metric.metric_name.value
        for metric in spec.analysis_requirements.derived_metrics
    ] == list(track.requested_metrics)
    assert spec.routing.preferred_path.value in {"tool", "sql", "knowledge"}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_report_track_specs.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the mapping**

Derive each field from the track rather than inventing it:
- `query_type` — from `track.evidence_mode`: `KNOWLEDGE` → `conceptual`, otherwise `data_explanation` when `track.requested_metrics` contains a change metric, else `data_retrieval`.
- `preferred_path` — `knowledge` for `ReportEvidenceMode.KNOWLEDGE`, else `tool`.
- `derived_metrics` — one `DerivedMetricRequest` per entry in `track.requested_metrics`.
- `confidence` — `1.0`. The planner is not guessing; this is its own output.

Read `contracts/question_analysis.py` for the exact required fields before writing — the model is strict (`extra="forbid"`) and every nested block must be populated.

- [ ] **Step 4: Run the test and the suite**

Run: `python -m pytest tests/test_report_track_specs.py tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/report_track_specs.py tests/test_report_track_specs.py
git commit -m "Map a research track to the analysis spec the planner already implied"
```

### Task 2.3: Use the spec and delete the prose round-trip

**Files:**
- Modify: `agent/report_research_execution.py:924` (`execute_report_track_analysis`), delete `build_report_track_analysis_query` at line 833
- Modify: `tests/test_report_research_execution.py`

**Interfaces:**
- Consumes: Task 2.1's `question_analysis=` parameter and Task 2.2's `report_track_analysis_spec`.
- Produces: `execute_report_track_analysis` unchanged in signature; internally passes the spec.

- [ ] **Step 1: Write the failing test**

```python
def test_track_analysis_passes_the_spec_and_never_re_asks(monkeypatch):
    captured = {}

    def query_pipeline(query, **kwargs):
        captured.update(kwargs)
        return _context_with_rows_and_a_missing_metric(query)

    execute_report_track_analysis(
        _QUERY,
        _plan().tracks[0],
        query_pipeline=query_pipeline,
    )

    assert captured["question_analysis"] is not None
    assert captured["answer_mode"] == "report"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/test_report_research_execution.py::test_track_analysis_passes_the_spec_and_never_re_asks -q`
Expected: FAIL — `question_analysis` is not among the kwargs.

- [ ] **Step 3: Pass the spec**

In `execute_report_track_analysis`, replace `build_report_track_analysis_query(report_query, track)` with `report_track_query(track)` for the display query and add `question_analysis=report_track_analysis_spec(track, report_query)` to the `query_pipeline(...)` call.

- [ ] **Step 4: Delete the prose builder**

Remove `build_report_track_analysis_query` and its tests (`test_track_analysis_query_uses_one_primary_question_and_bounded_coverage`, `test_track_analysis_query_reserves_room_for_report_context`). Those tests guard a 4,000-character prose budget that no longer exists.

- [ ] **Step 5: Full gate**

Run: `python -m pytest tests/ -q`
Run: `ruff check .`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Give each track the planner's intent instead of re-deriving it from prose"
```

- [ ] **Step 7: Measure against the baseline**

Deploy and re-run the May 2026 report. Compare against the Task 1.1 baseline: there must be **zero** `report_question_analyzer` stages in the log, and `REPORT_TRACK_ANALYSIS_ENABLED` must show the same `completed_count` and a comparable `analysis_item_count`. A drop in `analysis_item_count` means the mapping in Task 2.2 is losing intent — fix it before Phase 3.

---

## Phase 3 — Deduplicate track work

Three of four tracks called `get_prices` over the identical window on job 5e6b0cf3. Because `evidence_ref` is a content digest, those collapse to one manifest item assigned to three sections as *mandatory* citation — the structural cause of cross-section repetition.

### Task 3.1: Deduplicate by resolved tool and window

**Files:**
- Modify: `agent/report_track_specs.py` — add `report_track_work_key`
- Modify: `core/report_job_processor.py:697` (`_run_track_analysis`)
- Test: `tests/test_report_track_specs.py`, `tests/test_report_job_processor.py`

**Interfaces:**
- Produces: `report_track_work_key(track: ReportResearchTrack) -> tuple[str, ...]` — a hashable identity of `(sorted collector_ids, sorted requested_metrics, evidence_mode)`. Two tracks with equal keys run once and share the resulting packet.

- [ ] **Step 1: Write the failing test**

```python
def test_tracks_that_resolve_to_the_same_work_run_once(caplog):
    """Job 5e6b0cf3 fetched the same 61 rows of prices three times."""
    (
        research_plan,
        packets,
        manifest,
        decisions,
        gate,
        document_plan,
    ) = _document_components()
    runs = []

    def track_analyzer(_query, track, **_kwargs):
        runs.append(track.track_id)
        return {packet.track_id: packet for packet in packets}[track.track_id]

    processor = ReportJobProcessor(
        query_pipeline=lambda *_a, **_k: pytest.fail("no global enrichment"),
        track_analysis_mode="enabled",
        track_analyzer=track_analyzer,
        research_planner=lambda *_a, **_k: research_plan,
        research_executor=lambda *_a, **_k: packets,
        manifest_consolidator=lambda *_a, **_k: manifest,
        research_exhibit_builder=lambda *_a, **_k: decisions,
        evidence_gate_evaluator=lambda *_a, **_k: gate,
        document_planner=lambda *_a, **_k: document_plan,
        document_generator=lambda *_a, **_k: _valid_document_draft(
            document_plan, manifest
        ),
    )
    processor(_lease(query=_V2_QUERY), _Control())

    assert len(runs) == len({
        report_track_work_key(track) for track in research_plan.tracks
    })
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: FAIL — one run per track, not one per distinct key.

- [ ] **Step 3: Implement the key and the dedupe**

In `_run_track_analysis`, group `plan.tracks` by `report_track_work_key`, submit one future per group, and fan the resulting packet out to every track in the group via `merge_report_track_analysis_packet` (or `model_copy(update={"track_id": ...})` where no baseline exists — the packet's `track_id` must match the track it is stored under, or `ReportEvidencePacket` identity checks fail downstream).

- [ ] **Step 4: Add the telemetry**

Extend the `REPORT_TRACK_ANALYSIS_*` payload with `"deduplicated_track_count": len(plan.tracks) - len(groups)`. Without it, a dedupe that silently over-merges is invisible.

- [ ] **Step 5: Run the new tests and the suite**

Run: `python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Run one pipeline per distinct unit of track work, not per track"
```

---

## Phase 4 — One writer

### Task 4.1: Route every profile through the single document writer

**Files:**
- Modify: `agent/report_document_generation.py:784` (`generate_report_document`)
- Modify: `config.py` — add `ENABLE_REPORT_SINGLE_WRITER`, default `false`
- Modify: `config.py:371` — raise `REPORT_MAX_OUTPUT_TOKENS` default to `16384`
- Test: `tests/test_report_document_pipeline_v2.py`

**Interfaces:**
- Produces: `generate_report_document` with the flag on takes the `write_document` branch for every profile, not only `compact`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_full_profile_document_is_written_in_one_call(monkeypatch):
    from agent import report_document_generation as generation

    monkeypatch.setattr(
        generation, "ENABLE_REPORT_SINGLE_WRITER", True
    )
    (
        research_plan, packets, manifest, _, _, document_plan
    ) = _document_components()
    assert document_plan.profile.value != "compact"
    draft = _valid_document_draft(document_plan, manifest)
    calls = []

    generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_document=lambda *_args: (calls.append("write"), draft)[1],
        write_analysis_sections=lambda *_a, **_k: pytest.fail(
            "the batch writer must not run"
        ),
        repair_sections=lambda *_a, **_k: pytest.fail("no repair expected"),
    )

    assert calls == ["write"]
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: FAIL — the batch path runs for a full profile.

- [ ] **Step 3: Widen the branch condition**

```python
    if (
        write_document is not None
        or ENABLE_REPORT_SINGLE_WRITER
        or plan.profile.value == "compact"
    ):
```

- [ ] **Step 4: Raise the output ceiling**

The analysis batch alone emitted 4,810 completion tokens; analysis plus synthesis was 6,451 against an 8,192 cap, and a repair must re-emit the whole document. Set the `REPORT_MAX_OUTPUT_TOKENS` default to `16384`.

- [ ] **Step 5: Verify the whole-document path salvages and concedes**

The single-writer branch currently reaches the document repair loop and `_concede_length_shortfall`, but **not** `_sections_or_grounded_subset` — that is only wired into the batch path. Add the salvage call before the final raise in `generate_report_document`, or the grounded-subset protection shipped in `d794dd6` is lost the moment this flag goes on. Write a test for it:

```python
def test_the_single_writer_path_still_ships_the_grounded_subset(monkeypatch):
    from agent import report_document_generation as generation

    monkeypatch.setattr(generation, "ENABLE_REPORT_SINGLE_WRITER", True)
    ...  # mirror test_analysis_section_ships_grounded_subset_when_a_stray_number_survives
```

- [ ] **Step 6: Full gate, both flag positions**

Run: `python -m pytest tests/ -q`
Run: `$env:ENABLE_REPORT_SINGLE_WRITER='true'; python -m pytest tests/ -q; Remove-Item Env:\ENABLE_REPORT_SINGLE_WRITER`
Run: `ruff check .` and `python -m guardrails.redteam_gate`
Expected: all pass in both positions.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Write the whole report in one call behind ENABLE_REPORT_SINGLE_WRITER"
```

- [ ] **Step 8: Validate on a real report before deleting anything**

Turn the flag on in Railway and run the May 2026 report. Compare against baseline: total LLM calls should be 2–3, `finish_reason` must be `completed` (not `length` — if it is `length`, raise `REPORT_MAX_OUTPUT_TOKENS` further before continuing), and the document must still validate. **Do not start Task 4.2 until one real report has succeeded on this path.**

### Task 4.2: Delete the two-batch writer

**Files:**
- Modify: `agent/report_document_generation.py` — delete `_materialize_section_batch`, `_repair_section_batch`, `_repair_section_batch_until_valid`, `_without_free_unrendered_claims` call sites in the batch path, and the entire `else:` branch of `generate_report_document`
- Modify: `core/llm.py` — delete `_llm_write_report_section_batch`, `llm_write_report_analysis_sections`, `llm_write_report_synthesis_sections`
- Modify: `config.py` — delete `ENABLE_REPORT_SINGLE_WRITER` (the branch is now unconditional)

**Interfaces:**
- Consumes: a validated real report from Task 4.1 Step 8.
- Produces: `generate_report_document` with one writer branch.

- [ ] **Step 1: Preserve what the batch path owned**

`_without_free_unrendered_claims` (the unrendered-claim sweep from `d794dd6`) is called only from `_materialize_section_batch`. Before deleting that function, move the sweep into the single-writer path — call it per section after `ReportDocumentDraft.model_validate` and before `validate_report_document`. Its test `test_a_claim_the_prose_never_rendered_is_swept_not_repaired` must be rewritten against the new call site, **not** deleted.

- [ ] **Step 2: Run the suite to confirm the sweep survived the move**

Run: `python -m pytest tests/test_report_document_pipeline_v2.py -q`
Expected: PASS

- [ ] **Step 3: Commit the move separately**

```bash
git add -A
git commit -m "Move the unrendered-claim sweep onto the single-writer path"
```

- [ ] **Step 4: Delete the batch machinery**

Remove the functions listed above and the `else:` branch. Delete the tests that exercise batch orchestration specifically — `test_invalid_analysis_batch_stops_before_synthesis`, `test_batch_path_uses_two_budgeted_repairs_across_writer_batches`, `test_batch_repair_returning_the_wrong_section_set_is_rejected`, `test_schema_invalid_analysis_batch_repairs_from_the_raw_payload`, `test_exhausted_batch_repair_logs_the_result_it_could_not_fix`, `test_a_batch_that_reaches_salvage_unbudgeted_still_logs_its_state`. Each tests orchestration that no longer exists.

- [ ] **Step 5: Full gate**

Run: `python -m pytest tests/ -q`, `ruff check .`, `python -m guardrails.redteam_gate`
Expected: all pass.

- [ ] **Step 6: Commit and measure**

```bash
git add -A
git commit -m "Delete the two-batch report writer"
```

Record: `git diff --stat`. Expected: roughly −400 lines from `report_document_generation.py` and −200 from `core/llm.py`.

---

## Phase 5 — Separate "may cite" from "must cite"

`validate_report_section` uses one list as both whitelist and obligation, so shared narrative evidence broadcast to every analysis section forces every section to write about it.

### Task 5.1: Add an optional evidence set

**Files:**
- Modify: `contracts/report_document.py` (`ReportDocumentSectionSpec`)
- Modify: `agent/report_sections.py:189-219` (`validate_report_section`)
- Modify: `agent/report_document_planner.py:421` (evidence assignment)
- Test: `tests/test_report_sections.py`, `tests/test_report_document_pipeline_v2.py`

**Interfaces:**
- Produces: `ReportDocumentSectionSpec.optional_evidence_refs: List[EvidenceRef]`, defaulting to `[]`. `validate_report_section` allows `required ∪ optional` and requires only `required`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_report_document_pipeline_v2.py`, reusing its existing
`_document_components` and `_draft_section` helpers:

```python
def test_optional_evidence_may_be_cited_without_being_owed():
    """Shared context should be available to every section, mandatory in one."""
    (
        research_plan,
        _packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    spec = next(
        section
        for section in document_plan.sections
        if section.role is ReportDocumentSectionRole.ANALYSIS
    )
    # One shared ref moves from required to optional; the section cites only
    # what remains required.
    shared_ref = spec.required_evidence_refs[0]
    trimmed = spec.model_copy(
        update={
            "required_evidence_refs": spec.required_evidence_refs[1:],
            "optional_evidence_refs": [shared_ref],
        }
    )
    section = ReportSectionDraft.model_validate(
        _draft_section(trimmed, manifest)
    )

    validation = validate_report_section(section, trimmed, manifest)

    assert "REQUIRED_EVIDENCE_NOT_USED" not in validation.error_codes
    assert "EVIDENCE_REF_NOT_ALLOWED" not in validation.error_codes
```

Note that `_draft_section` builds `evidence_refs` from
`section.required_evidence_refs`, so the drafted section will not cite the
optional ref — which is exactly the case under test.

- [ ] **Step 2: Run it to confirm it fails**

Expected: FAIL — the field does not exist.

- [ ] **Step 3: Add the field and split the sets**

In `agent/report_sections.py`:

```python
    allowed_refs = set(section.required_evidence_refs) | set(
        section.optional_evidence_refs
    )
    ...
    if not set(section.required_evidence_refs).issubset(used_refs):
        errors.append("REQUIRED_EVIDENCE_NOT_USED")
```

- [ ] **Step 4: Move shared narrative refs to optional**

In `agent/report_document_planner.py`, build analysis sections with `required_evidence_refs=track_refs[:32]` and `optional_evidence_refs=shared_narrative_refs[:32]`. Keep the shared refs *required* on exactly one section — `limitations`, whose job is to describe collection boundaries — so the manifest item is still guaranteed at least one citation and cannot go unused.

- [ ] **Step 5: Full gate**

Run: `python -m pytest tests/ -q`, `ruff check .`, `python -m guardrails.redteam_gate`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Let a section cite shared context without being obliged to discuss it"
```

- [ ] **Step 7: Validate**

Run the May 2026 report. Compare section prose against the Task 1.1 baseline for the repeated framing you reported. If repetition persists with one writer *and* no shared obligation, the remaining cause is the word floor — revisit `report_section_word_floor_ratio`, not this contract.

---

## Out of scope, and why

**The grounding contract is unchanged by this plan.** Every phase here is orchestration. The writer still renders numbers into prose and emits matching coordinate metadata, verified afterwards — the coupling that has failed four distinct ways (`DERIVED_CLAIM_INVALID` → `UNGROUNDED_NUMERIC_CLAIM` → `DERIVED_CLAIM_NOT_USED` → `DIRECT_CLAIM_NOT_USED`). Replacing it with placeholder substitution is the separate structural fix already recorded in the backlog. **Sequence it after this plan** — Phase 4 roughly halves report output tokens, which is what makes placeholder substitution comfortable rather than tight.

**Chart and table selection are untouched.** The irrelevant chart and tariff table are defects in `agent/report_charts.py`, deterministic and independent of orchestration. They need their own plan.

**`REPORT_MAX_EXHIBITS = 4`** is hard-coded with no rationale and no env override, unlike every neighbouring limit. It did not bind on job 5e6b0cf3 — two of four charts were omitted for `INCOMPATIBLE_UNITS` and `INSUFFICIENT_CATEGORIES`, not truncated by the cap. Raising it changes nothing until chart selection is fixed. Leave it.

## Expected outcome

| | before | after |
|---|---|---|
| LLM calls per report | 8 | 2–3 |
| report code paths | 2 | 1 |
| lines in the report subsystem | ~9,989 | ~7,900 |
| duplicate tool fetches | 3 tracks × `get_prices` | 1 |
| sections obliged to cite shared context | all | 1 |
