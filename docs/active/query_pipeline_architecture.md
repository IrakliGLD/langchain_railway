# Query Pipeline Architecture

Current technical reference for the `langchain_railway` query pipeline, with the Ideal Decision Tree as a reference contract and an explicit list of remaining structural work.

**Last updated:** 2026-05-09
**Status:** Active. The runtime now closely matches the Ideal Decision Tree (§3.4): Stage 0.2 emits a full answer contract, evidence frames + generic renderer cover the four standard data shapes plus SCENARIO and FORECAST, evidence planner validates plans against `answer_kind`, vector retrieval is three-tier, and analyzer + summarizer prompts are question-type-aware with section-aware truncation. Remaining gap is Phase F (legacy stage consolidation).

---

## 1. Executive Summary

Stage 0.2 is one LLM call that emits the full answer contract. The runtime then executes that contract:

1. prepare context
2. structured question analyzer — emits `answer_kind`, `render_style`, `grouping`, `entity_scope`, `candidate_tools`, `evidence_roles`, `derived_metrics`, `visualization`
3. cross-check `answer_kind` against `query_type`-derived value, with a legal-list exception for high-confidence regulatory questions
4. conditionally retrieve vector knowledge — three tiers (FULL / LIGHT / SKIP) selected from `answer_kind` + `render_style`
5. derive response mode and resolution policy inline from the contract
6. evidence planner builds plan + validates it against `answer_kind` (LIST → entity step, COMPARISON → two-period evidence, etc.)
7. tool execution → canonical evidence frames (`ObservationFrame`, `EntitySetFrame`, `ComparisonFrame`) → inline validation
8. Stage 3 enrichment dispatches on `answer_kind` and emitted contract flags
9. generic renderer first (handles SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO / FORECAST from frames); falls back to LLM summarizer for narrative `render_style`
10. chart pipeline consumes the analyzer's `VisualizationInfo` (presentation, visual goal, measure transform, chart family, time grain, series split, sort/top-N) and renders multi-group plans

The remaining structural gap is Phase F: legacy parallel paths (Stage 0.7 keyword router fallback, Stages 1/2 SQL as separate stages, post-hoc provenance gate, the four-stage tool execution split) have not yet been collapsed into the eight-step ideal. They cause no observable bugs today but add maintenance surface and create occasional edge cases where legacy paths bypass frame construction.

---

## 2. What Changed Since 2026-04-12

- **Phases A through E of the prior phased plan are now landed.** Evidence planner reads `answer_kind` and validates plan shape; generic renderer covers SCENARIO and FORECAST with the five regex eligibility detectors removed; analyzer prompt is dynamically assembled with question-type-ordered blocks and section-aware truncation; vector retrieval has three tiers; fast/deep mode is configurable.
- **Visualization contract is fully implemented.** `VisualizationInfo` carries `primary_presentation`, `visual_goal`, `measure_transform`, `time_grain`, `series_split_mode`, `max_series`, `sort_rule`, `top_n`, `chart_intent`, `target_series`. Stage 5 consumes them via `agent/chart_pipeline.py` and `agent/chart_frame_builder.py`. Derived chart builders cover MoM/YoY, index growth, decomposition, forecast, and seasonal.
- **Cross-check policy refinement (2026-05-09).** Added an exception in `_cross_check_answer_kind`: when the LLM emits `answer_kind=LIST` with confidence ≥ 0.85 for a `regulatory_procedure` or `conceptual_definition` query, the LLM's LIST shape is trusted instead of being clobbered to KNOWLEDGE. Closes a quality regression where legal enumerations (eligible parties, requirements, documents) were paraphrased by narrative rendering.
- **Analyzer prompt vocabulary clarification (2026-05-09).** The `_ANALYZER_CORE_RULES` block explicitly lists the AnswerKind enum values, prohibits reusing `query_type` values for `answer_kind`, and adds a `query_type → answer_kind` default mapping. Eliminates a Pydantic validation crash where the LLM emitted `answer_kind=data_explanation` (a query_type value).
- **Answer-composer enumeration discipline (2026-05-09).** `Focus: Regulation` in the answer-composer skill gained an "Enumeration discipline" subsection with a worked example. The `conceptual_definition` and `regulatory_procedure` templates now require complete item-by-item rendering when the source enumerates items.
- **Router few-shot coverage (2026-05-09).** Added eligibility/participation example block (routes "who can …", "what documents …", "what conditions …" to `regulatory_procedure` + `answer_kind=list`) and supply-structure example block (routes "trend and structure of [supply | generation | mix]" to `data_retrieval` + `answer_kind=timeseries` + `preferred_path=tool`).
- **New developer-side skill** `skills/pipeline-failure-diagnostics` documents how to triage Q&A failures (latency log reading, failure taxonomy, fix layering principles). Not loaded into LLM prompts; it guides the developer when changes follow from a real failure.

---

## 3. Current Runtime Flow

### 3.1 High-Level Pipeline

```text
HTTP /ask
  -> Stage 0     prepare_context
  -> Stage 0.2   question analyzer (LLM) -> full contract
  -> answer_kind cross-check (with legal-list exception)
  -> Stage 0.3   vector knowledge retrieval (three-tier: FULL / LIGHT / SKIP)
  -> response_mode + resolution_policy derivation (inline from contract)
  -> Stage 0.4   evidence planner (validates plan against answer_kind)
  -> Stage 0.5/0.6 primary tool execution + canonical evidence frame construction + inline validation
  -> Stage 0.7   analyzer tool route fallback (legacy, pending removal in Phase F)
  -> Stage 0.8   evidence loop + evidence merge
  -> Stage 1/2   legacy planner + SQL fallback (pending fold-in in Phase F)
  -> Stage 3     analyzer enrichment (dispatches on answer_kind + contract flags)
  -> Stage 4     generic renderer (SCALAR/LIST/TIMESERIES/COMPARISON/SCENARIO/FORECAST) OR LLM structured summary
  -> Stage 5     chart pipeline (consumes VisualizationInfo, builds chart frames, multi-group)
```

### 3.2 Short-Circuit Paths

- `ResolutionPolicy.CLARIFY` → `summarizer.answer_clarify()`
- `ResponseMode.KNOWLEDGE_PRIMARY` → `summarizer.answer_conceptual()`
- Generic renderer succeeds (any of the six handled `answer_kind` shapes) → skip LLM summarizer
- Derived chart builder produces an answer-mode chart spec → preserved through Stage 5 instead of falling back to `ctx.df`

### 3.3 Current Decision Tree (Actual Runtime)

```text
HTTP /ask
│
├─ Stage 0: prepare_context
│   ├─ detect language, select light/analyst mode
│   ├─ heuristic conceptual classifier (fallback signal)
│   └─ initialize QueryContext
│
├─ Stage 0.2: question analyzer (LLM) — emits full contract
│   ├─ answer_kind, render_style, grouping, entity_scope
│   ├─ candidate_tools (ranked), candidate_topics, params_hint, filter
│   ├─ evidence_roles, needs_multi_tool, derived_metrics
│   ├─ visualization (primary_presentation, visual_goal, measure_transform,
│   │   time_grain, series_split_mode, max_series, sort_rule, top_n,
│   │   chart_intent, target_series)
│   └─ canonical_query_en, confidence
│
├─ Cross-check answer_kind (LLM-emitted vs query_type-derived)
│   ├─ legal-list exception: trust llm=LIST when query_type ∈
│   │   {regulatory_procedure, conceptual_definition} and confidence ≥ 0.85
│   ├─ otherwise prefer the safer of {TIMESERIES, EXPLANATION, KNOWLEDGE}
│   └─ log INFO when exception fires; WARNING on other disagreements
│
├─ Stage 0.3: vector knowledge retrieval (three-tier)
│   ├─ KNOWLEDGE / EXPLANATION → FULL (top-K=6, re-ranked)
│   ├─ data shape + NARRATIVE → LIGHT (top-K=2, no re-rank)
│   ├─ data shape + DETERMINISTIC → SKIP
│   └─ CLARIFY → SKIP
│
├─ Stage 0.4: evidence planner
│   ├─ reads answer_kind / render_style / candidate_tools
│   └─ validates plan: LIST → entity-enumeration, COMPARISON → two-period,
│       TIMESERIES → period range, SCENARIO → scenario params
│
├─ Tool execution + framing + inline validation
│   ├─ run tool; adapt result to ObservationFrame / EntitySetFrame / ComparisonFrame
│   ├─ bind provenance refs at construction time
│   └─ validate frame against answer_kind requirement
│
├─ Stage 3: analyzer enrichment
│   ├─ dispatches on contract flags (needs_correlation_context,
│   │   needs_driver_analysis, requested_derived_metrics)
│   └─ emits: MoM/YoY, correlation, scenario evidence, forecast trendline,
│       seasonal decomposition — all consumable by derived chart builders
│
├─ Stage 4: generic renderer first
│   ├─ SCALAR / LIST / TIMESERIES / COMPARISON → tabular
│   ├─ SCENARIO → payoff breakdown
│   ├─ FORECAST → trendline + R² caveat + seasonal
│   └─ on render_style=NARRATIVE → LLM structured summarizer
│
└─ Stage 5: chart pipeline
    ├─ consume VisualizationInfo
    ├─ choose chart frame source: derived chart builder OR canonical evidence frames
    ├─ apply measure_transform / time_grain
    └─ produce one chart per chart_group (multi-panel for split units)
```

### 3.4 Ideal Decision Tree

**Design principle: Stage 0.2 is the one LLM call. Make it emit the full contract. Everything after just executes — no re-interpretation.**

The current pipeline matches this tree closely. The main remaining mismatch is the persistence of legacy fallback stages (0.7, 1, 2) as parallel paths instead of being folded into the tool execution failure path.

```text
HTTP /ask
│
├─ Stage 0: prepare_context
│   → detect language, select light/analyst mode, init context
│   → cheap, no LLM
│
├─ Stage 0.2: question analyzer (THE LLM call — firm contract, no questioning after)
│   │
│   │ This is the single point where the question is interpreted.
│   │ Every field needed by downstream stages is emitted here.
│   │
│   │ Emitted fields:
│   │   query_type, preferred_path, candidate_tools, params_hint,
│   │   evidence_roles, derived_metrics, canonical_query_en,
│   │   answer_kind, render_style, grouping, entity_scope,
│   │   visualization (full)
│   │
│   │ NOTE: primary_tool is NOT emitted by the LLM. Tool selection
│   │ remains deterministic: candidate_tools[0] + evidence planner.
│   │
│   │ Active cross-check: LLM-emitted vs query_type-derived answer_kind.
│   │ Disagreement is logged; safer option preferred unless the
│   │ legal-list exception applies (LIST + regulatory/conceptual + high confidence).
│   │
│   │ Fallback when analyzer is disabled/fails:
│   │   → derive answer_kind from query_type mapping where unambiguous
│   │   → derive tool from keyword router (legacy)
│   │
│   → The analyzer's output is the contract. Downstream stages trust it.
│
├─ Stage 0.3: vector knowledge retrieval (three-tier)
│   ├─ KNOWLEDGE / EXPLANATION → FULL retrieval
│   ├─ data + NARRATIVE → LIGHT retrieval
│   └─ data + DETERMINISTIC, or CLARIFY → SKIP
│
├─ Stage 0.4: evidence planner
│   │ Reads the analyzer contract; validates plan against answer_kind.
│   │   answer_kind = LIST → planner ensures entity-enumeration step
│   │   answer_kind = COMPARISON → two periods or two entities
│   │   answer_kind = TIMESERIES → period range
│   │   answer_kind = SCENARIO → scenario params available
│   │ Tool selected from candidate_tools[0]; evidence roles known from 0.2.
│   → Ordered list of tool steps, validated against the answer contract
│
├─ Tool execution + evidence collection (single loop — TARGET)
│   │ Currently split across Stage 0.5 / 0.6 / 0.7 / 0.8; Phase F merges these.
│   │
│   │ for each evidence step:
│   │   1. execute tool (params from analyzer / planner)
│   │   2. validate relevance
│   │   3. normalize output into canonical evidence frame
│   │   4. validate frame against answer_kind requirements
│   │   5. store evidence by role, bind provenance refs at construction time
│   │
│   │ On tool failure:
│   │   ├─ [other plan steps remain] → continue, mark step failed
│   │   ├─ [no plan steps + SQL fallback possible] → try SQL escape hatch
│   │   └─ [nothing works] → downgrade to CLARIFY
│   │
│   │ Stage 0.7 (analyzer fallback) and Stages 1/2 (legacy SQL) are
│   │ pending fold-in via Phase F.
│   │
│   → Evidence frames collected, validated, provenance-bound
│
├─ Stage 3: analyzer enrichment — dispatches on answer_kind + contract flags
│   │ answer_kind = SCALAR/TIMESERIES + share signal → share enrichment
│   │ answer_kind = SCENARIO → scenario evidence dispatch
│   │ answer_kind = FORECAST → trendline pre-calculation
│   │ answer_kind = EXPLANATION → causal reasoning + correlation
│   │ answer_kind = COMPARISON → MoM/YoY derived metrics
│   │ Outputs written as canonical evidence frames consumable by
│   │ derived chart builders.
│
├─ Stage 4: answer rendering — switch on answer_kind
│   │
│   ├─ [render_style = DETERMINISTIC]
│   │   │ Generic renderer over evidence frames
│   │   ├─ SCALAR   → extract value + unit + period
│   │   ├─ LIST     → enumerate entities, group by reason
│   │   ├─ TIMESERIES → format period-indexed rows
│   │   ├─ COMPARISON → subject vs baseline + delta
│   │   ├─ SCENARIO → payoff breakdown (positive/negative, market vs combined)
│   │   └─ FORECAST → trendline + R² caveat + seasonal + assumptions
│   │   Provenance refs already bound from evidence collection.
│   │
│   └─ [render_style = NARRATIVE]
│       → LLM receives pre-structured evidence frames
│       → focused on explanation quality, not schema recovery
│       → provenance refs pre-bound
│
└─ Stage 5: chart pipeline — consumes VisualizationInfo, multi-group
    → Built from derived chart builder frames or canonical evidence frames
    → Not from raw ctx.df by default
```

---

## 4. Current Stage Semantics

### 4.1 Stage 0: Prepare Context

`agent/planner.py`. Detects language, picks light/analyst mode, runs heuristic conceptual classifier as a fallback signal, initialises `QueryContext`. Cheap, no LLM.

### 4.2 Stage 0.2: Structured Question Analyzer

`core/llm.py::llm_analyze_question()`. The semantic centre of the pipeline. Emits the full contract — `query_type`, `preferred_path`, `candidate_tools`, `params_hint`, `evidence_roles`, `derived_metrics`, `analysis_requirements`, `canonical_query_en`, `answer_kind`, `render_style`, `grouping`, `entity_scope`, `visualization`. Active cross-check against `query_type`-derived `answer_kind` runs every call (`agent/pipeline.py::_cross_check_answer_kind`); legal-list exception trusts high-confidence LIST for regulatory/conceptual queries.

Prompt assembly is dynamic: `_classify_analyzer_prompt_profile` + `_build_analyzer_prompt_blocks` choose ordered blocks per question family; section-aware truncation drops the least-relevant blocks first per `_ANALYZER_TRUNCATION_DATA` / `_ANALYZER_TRUNCATION_KNOWLEDGE`. Fast mode swaps the budget for `FAST_MODE_ANALYZER_BUDGET`.

### 4.3 Stage 0.3: Vector Knowledge Retrieval

Three-tier (`VectorRetrievalTier`): FULL for knowledge/explanation, LIGHT (top-K=2, no re-rank) for narrative data shapes, SKIP for deterministic data and CLARIFY. Selected in `agent/pipeline.py` from `answer_kind` + `render_style`.

### 4.4 Response Mode + Resolution Policy (Inline)

Derived inline from the contract — no separate stage. KNOWLEDGE / CLARIFY short-circuits; everything else continues to evidence planning.

### 4.5 Stage 0.4: Evidence Planner

`agent/evidence_planner.py`. Reads `answer_kind`, `render_style`, `candidate_tools`, `evidence_roles`. `_validate_plan_against_answer_kind` ensures planned steps will produce evidence matching the answer shape (LIST → entity-enumeration step; COMPARISON → two-period; TIMESERIES → period range; SCENARIO → scenario params). Mismatches are flagged at planning time, not after wasted tool calls.

### 4.6 Stage 0.5 / 0.6: Tool Execution + Evidence Frame Construction

Tool runs, output is adapted by `agent/frame_adapters.py` into one of `ObservationFrame`, `EntitySetFrame`, `ComparisonFrame`. Frames carry provenance refs bound at construction time. `agent/evidence_validator.py::validate_evidence` runs inline — correctable gaps are corrected; uncorrectable gaps surface to the planner.

### 4.7 Stage 0.7: Analyzer Tool Route Fallback (Legacy, Pending Removal)

`router.py::match_tool` runs as a final fallback when the evidence plan didn't yield a tool. `candidate_tools[0]` + evidence planner makes this redundant in principle. Pending Phase F removal (track hit rate first; remove if <5% of queries).

### 4.8 Stage 0.8: Evidence Loop And Merge

Iterates over remaining plan steps, framing and merging results. Pending fold-in into the single execution loop in Phase F.

### 4.9 Stage 1 / 2: Legacy Planner And SQL Fallback

Older parallel path that runs free-form SQL when no typed tool matches. Pending fold-in into the tool-execution failure path in Phase F.

### 4.10 Stage 3: Analyzer Enrichment

`agent/analyzer.py`. Dispatches on the analyzer-emitted contract flags (`needs_correlation_context`, `needs_driver_analysis`, requested `derived_metrics`, share-intent emitted by Stage 0.2). No regex eligibility detection. Outputs become `analysis_evidence` consumable by both the LLM summariser and the derived chart builders.

### 4.11 Stage 4: Generic Renderer + LLM Summariser

`agent/summarizer.py::summarize_data` first calls `_try_generic_renderer` (`agent/generic_renderer.py`). The renderer handles SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO / FORECAST from canonical evidence frames. `share_summary_override` is a deterministic specialized formatter for share-intent queries (alongside SCENARIO and FORECAST in the §3.4 Ideal Tree's "specialized formatters" category — see §6.3). When neither produces output and `render_style=NARRATIVE`, control passes to `llm_summarize_structured` with a focus-aware prompt assembled from `skills/answer-composer/`.

The five regex eligibility detectors that previously gated SCENARIO / FORECAST / TARIFF-LIST / RESIDUAL-WEIGHTED-PRICE / FORECAST-DIRECT have been removed. New question families that produce one of the six handled shapes need zero Stage 4 code.

### 4.12 Stage 5: Chart Pipeline

`agent/chart_pipeline.py` consumes `VisualizationInfo` directly. Chart-frame source is selected first: derived chart builders (`agent/derived_chart_builder.py`) produce specs for MoM/YoY, index growth, decomposition, forecast, and seasonal answers; otherwise canonical evidence frames feed `agent/chart_frame_builder.py`. Multi-group plans are preserved — `chart_groups` is iterated, with per-group `type`, `title`, `y_axis_label`, and `metrics` honoured.

---

## 5. Module Responsibilities

### `agent/pipeline.py`

Orchestrates the full pipeline. Owns: stage tracing, response-mode derivation, cross-check, evidence loop, error/fallback policy. Holds the legal-list cross-check exception.

### `agent/router.py`

Keyword + semantic tool router. Active only as a fallback when the evidence plan doesn't yield a tool (Stage 0.7) and as a tool-name resolver during recovery. Pending removal of the parallel Stage 0.7 invocation in Phase F.

### `agent/evidence_planner.py`

Builds and validates the evidence plan against the analyzer contract — including `answer_kind`-shape validation (`_validate_plan_against_answer_kind`).

### `agent/analyzer.py`

Stage 3 enrichment. Contract-driven dispatch on emitted flags and derived-metric requests. No regex eligibility detection.

### `agent/generic_renderer.py`

Stage 4 deterministic rendering for SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO / FORECAST.

### `agent/frame_adapters.py`

Per-tool adapters that normalise raw tool output into `ObservationFrame` / `EntitySetFrame` / `ComparisonFrame`.

### `agent/evidence_validator.py`

Inline validation of evidence frames against `answer_kind`.

### `agent/summarizer.py`

Stage 4 entry point. Tries the generic renderer first; otherwise calls `llm_summarize_structured` with focus-aware prompts. `share_summary_override` is a deliberate specialized formatter for share-intent queries (see §6.3). Conceptual answers go to `answer_conceptual`.

### `agent/chart_pipeline.py`, `agent/chart_frame_builder.py`, `agent/derived_chart_builder.py`

Stage 5. Visualization plan → chart frame → render. Derived chart builders cover MoM/YoY, index growth, decomposition, forecast, seasonal.

### `core/llm.py`

LLM call sites and prompt assembly. Owns: dynamic analyzer prompt blocks, `_classify_analyzer_prompt_profile`, `_build_analyzer_prompt_blocks`, `_ANALYZER_TRUNCATION_*` priorities, `_TRUNCATION_PRIORITY_*` summarizer profiles by `answer_kind`, fast-mode budget overrides, OpenAI fallback on Gemini failure.

### `contracts/question_analysis.py`, `question_analysis_catalogs.py`

The Stage 0.2 contract: `QuestionAnalysis` Pydantic model with `AnswerKind`, `RenderStyle`, `Grouping`, `VisualizationInfo`, `FilterCondition`, `MeasureTransform`, `VisualizationTimeGrain`, `SeriesSplitMode`, `SortRule`, `ChartFamily`, `VisualGoal`, `PresentationMode`, `SemanticRole`, `ChartIntent`. Catalogs supply the LLM-facing JSON that explains each enum.

### `contracts/evidence_frames.py`, `contracts/vector_knowledge.py`

`ObservationFrame`, `EntitySetFrame`, `ComparisonFrame`; `VectorRetrievalTier`.

### `knowledge/vector_retrieval.py`

Three-tier retrieval implementation.

### `skills/`

Runtime skills loaded into LLM prompts: `question-analyzer`, `sql-planner`, `answer-composer`, `energy-analyst`. Plus developer-only skills (NOT loaded into prompts): `developer-phased-audit`, `pipeline-failure-diagnostics`.

---

## 6. Remaining Architectural Gaps

Six structural gaps remain. None block correctness today; they are maintenance / clarity / latency improvements and an open quality area.

### 6.1 Pipeline Consolidation (Phase F)

The runtime still carries pre-contract structural artefacts:

- **Stage 0.7 — analyzer route fallback.** `router.match_tool()` runs after the evidence planner already selected a tool. When `candidate_tools[0]` is sound the call is a no-op; when the planner errored it can paper over the bug. Tracking its hit rate is the prerequisite for removal.
- **Stages 1 / 2 — legacy SQL.** Free-form SQL planning + execution as separate pipeline stages. Should be folded into the tool-execution failure path so SQL is attempted only when typed tools fail inside the execution loop.
- **Post-hoc provenance gate.** Provenance is now bound at frame construction. The post-hoc gate at the end of Stage 4 still runs — it is redundant for paths that go through canonical frames but still catches numeric hallucinations on legacy paths.
- **Four-stage tool execution split (0.5 / 0.6 / 0.7 / 0.8).** The Ideal Decision Tree calls for one execution loop. Today these are separate stages with their own tracing and fallback branches; merging them is the single largest simplification of `agent/pipeline.py`.

These items are low-correctness-risk because the contract-driven path produces the same outcome whether the legacy stages exist or not. They are high-clarity-impact because debugging today requires understanding four stages and their interactions instead of one loop.

### 6.2 Analyzer Misclassification — Quality, Not Architecture

The architecture trades regex brittleness for LLM non-determinism. The cross-check + evidence validator + legal-list exception are mitigations, not eliminations. The quality work happens in the analyzer prompt, the runtime skills, and the few-shot examples — not in additional pipeline stages.

The 2026-05-09 fix series is illustrative: a regulatory-eligibility question was mis-routed (`conceptual_definition` vs `regulatory_procedure`), the cross-check overrode the LLM's correct LIST shape, and the narrative template merged enumerated items. The fix was four small edits across the analyzer prompt, the cross-check policy, and the answer-composer skill — no new stage.

This area will keep producing per-question reports. The diagnostic playbook is in `skills/pipeline-failure-diagnostics/`; the fix layering principle is "prompt vs cross-check vs runtime skill — pick one layer based on where the contract was actually wrong, not where the symptom appeared."

### 6.3 `share_summary_override` Is A Specialized Formatter, Not Legacy Debt

Earlier versions of this document listed `share_summary_override` as legacy debt to be absorbed into the generic renderer. A 2026-05-10 audit (Phase F.1) determined this was a misclassification: `share_summary_override` is a deliberate **specialized formatter** for share-intent queries, in the same category as the SCENARIO and FORECAST specialized formatters that the §3.4 Ideal Decision Tree explicitly preserves alongside the generic renderer.

The artifact decomposes `share_all_ppa` into its renewable/thermal components and joins per-period prices — domain-specific knowledge that the generic renderer intentionally does not have. The decision to build it is gated on the structured analyzer signal `ctx.analyzer_indicates_share_intent`, not on regex. See the inline comment at [agent/analyzer.py:2020-2030](D:\Enaiapp\langchain_railway\agent\analyzer.py#L2020) for the design rationale.

No action required. This is not a Phase F item.

### 6.4 Filter Field — Implemented But Worth Auditing

`FilterCondition` (`metric / operator / value / unit`) exists on `ToolParamsHint` and is consumed by tool executors. Audit periodically that new threshold-style queries route through the structured filter rather than ad-hoc post-fetch filtering inside summarisers.

### 6.5 Cross-Tool Computation Patterns

The default for cross-tool computational questions ("weighted average price excluding regulated entities") is narrative rendering — let the LLM synthesise from pre-structured evidence. If a repeating pattern emerges, add a `derived_metric` type and let Stage 3 compute it from multi-tool evidence (same shape as MoM / YoY / correlation). Principle: narrative as default, derived metric only when the pattern repeats. No structural change required.

### 6.6 Chart-Building Polish

The visualisation contract is fully implemented and Stage 5 consumes it. Two remaining observations:

- **Decision rule for chart vs table on borderline `answer_kind=COMPARISON` questions.** Today `primary_presentation` is consulted, but when the analyzer leaves it null the code falls back to row-count heuristics. The right move when uncertain is `chart_plus_table` rather than chart-only.
- **Reference lines.** `include_reference_lines` was dropped because a bare bool can't carry which axis/value/label. If reference-line rendering becomes needed, introduce a concrete `ReferenceLineSpec` dataclass first.

---

## 7. Source Of Truth

Runtime behaviour described above is implemented in:

- `agent/pipeline.py`
- `agent/router.py`
- `agent/evidence_planner.py`
- `agent/evidence_validator.py`
- `agent/frame_adapters.py`
- `agent/analyzer.py`
- `agent/summarizer.py`
- `agent/generic_renderer.py`
- `agent/chart_pipeline.py`
- `agent/chart_frame_builder.py`
- `agent/derived_chart_builder.py`
- `core/llm.py`
- `knowledge/vector_retrieval.py`
- `contracts/question_analysis.py`
- `contracts/question_analysis_catalogs.py`
- `contracts/evidence_frames.py`
- `contracts/vector_knowledge.py`
- `schemas/question_analysis.schema.json`
- `skills/answer-composer/`, `skills/energy-analyst/`, `skills/question-analyzer/`, `skills/sql-planner/`
- `skills/developer-phased-audit/`, `skills/pipeline-failure-diagnostics/` (developer-only)

If this document and the code disagree, the code wins. Update this document.

---

## 8. Remaining Phased Plan

One phase, plus ongoing quality work. Phases A–E from the prior plan are complete and have been removed from this document.

### Phase F: Pipeline Consolidation — Remove Legacy Stages

**Problem being solved:** Stage 0.7, Stages 1/2, the post-hoc provenance gate, and the four-stage tool execution split are pre-contract artefacts that add maintenance surface and create occasional edge cases where legacy paths bypass frame construction. The Ideal Decision Tree calls for a single execution loop and folded-in SQL escape hatch.

**What to do:**

| # | Task | File |
|---|------|------|
| 1 | Track Stage 0.7 hit rate for two weeks. If <5% of queries and the evidence planner handles those cases, proceed with removal. If meaningful traffic, investigate why the evidence planner misses those cases first. **Done (Phase F.2, 2026-05-10)**: counters `stage_0_7_entered`, `stage_0_7_invocation_built`, `stage_0_7_used_result` exposed in `/metrics`. `used_result / requests` is the "paying its keep" rate; <5% is the cutoff for proceeding with task 2. | `agent/pipeline.py`, `utils/metrics.py` |
| 2 | Remove Stage 0.7: drop the `match_tool` fallback invocation, the recovery branch, and the associated stage tracing. | `agent/pipeline.py`, `agent/router.py` |
| 3 | Remove the post-hoc provenance gate. Provenance is already bound at frame construction. | `agent/pipeline.py`, `agent/summarizer.py` |
| 4 | Fold Stages 1 / 2 (legacy SQL) into the tool execution failure path: attempt SQL only when typed tools fail inside the execution loop. | `agent/pipeline.py` |
| 5 | Merge Stages 0.5 / 0.6 / 0.7 / 0.8 into a single execution loop iterating over evidence plan steps with inline frame construction and validation. | `agent/pipeline.py`, `agent/evidence_planner.py` |

**What to expect after:**

- Pipeline step count drops from ~10 to the eight-step ideal.
- No more edge cases where legacy paths bypass frame construction or evidence validation.
- `agent/pipeline.py` becomes significantly shorter and easier to reason about.
- Debugging is simpler: one execution loop instead of four stages with fallback branches.
- The Ideal Decision Tree (§3.4) becomes a literal description of the runtime, not an aspirational target.

**Validation:**

This phase is purely structural — behaviour should not change. Verify by running the full test suite and comparing outputs on a large query batch before and after. Any output difference indicates a case where the legacy path was producing different results than the new architecture, which should be investigated (it's likely a bug in one path or the other).

**Sequencing:**

The hit-rate tracking from item 1 can start in parallel with the analyzer-quality work in §6.2. Items 2 and 3 are independent. Items 4 and 5 are best done together because they affect the same call sites. Item 6 is independent and can land any time.

---

## 9. Standing Quality Workstreams

These are not phases — they are ongoing commitments triggered by failure reports.

### 9.1 Analyzer Routing Quality

Symptom class: a query is mis-routed at Stage 0.2 (wrong `query_type`, wrong `answer_kind`, low confidence on a question the human would answer easily).

Workflow: open `skills/pipeline-failure-diagnostics/references/log-reading.md`, classify the failure (Pattern C router misclassification, Pattern D heuristic disagreement), and propose a fix at the right layer per `fix-principles.md`. Few-shot example expansion in `core/llm.py::_ANALYZER_CORE_RULES` is usually correct; cross-check policy changes are rarer; runtime skill changes (answer-composer / energy-analyst) only when the analyzer was right but rendering lost information.

### 9.2 Summarizer Latency on Long Prompts

Symptom class: 504 retry on the summarizer call, total request time > 60s, prompt > 50 K chars.

Workflow: `pipeline-failure-diagnostics` Pattern E. Prefer reducing prompt size (`PROMPT_BUDGET_MAX_CHARS`, domain-knowledge cap, vector top-K) before changing model or retry count. Switching summarizer to `gemini-2.5-flash` is the safest single-knob latency win when quality measurements support it.

### 9.3 Grounding Failures

Symptom class: `provenance_gate gate_passed=false`, `summary_source=citation_gate_fallback`, the user gets a generic apology instead of the analytical answer.

Workflow: `pipeline-failure-diagnostics` Pattern F. Almost always one of: (1) `analyzer_available=false` from Pattern A, (2) data preview was truncated by the prompt budget and the LLM filled the gap with plausible-looking numbers, or (3) the LLM did arithmetic in its head instead of citing pre-computed `STATISTICS`. Fix at the right layer; do not lower the coverage threshold.

### 9.4 Visualization Plan Misalignment

Symptom class: chart contradicts the answer (chart shows raw rows when the answer discusses MoM change; chart present when the answer is a single number; multi-unit metrics squashed onto one axis).

Workflow: confirm the analyzer emitted `VisualizationInfo` correctly. If yes, the bug is in `agent/chart_pipeline.py` or a missing derived chart builder. If no, the bug is in the analyzer prompt's chart-policy hints.

---

## 10. Practical Conclusion

The pipeline is now contract-driven end to end. Stage 0.2 emits the answer contract, evidence frames + the generic renderer cover the six standard answer shapes, the evidence planner validates plan shape, vector retrieval is three-tier, and the visualisation plan flows through to Stage 5.

The remaining structural work is Phase F — collapsing legacy parallel stages into the eight-step ideal. None of it changes behaviour; it changes maintenance surface. The remaining quality work is in the analyzer prompt, runtime skills, and cross-check policy — guided by the `pipeline-failure-diagnostics` developer skill.

**The practical test:** when a new question family appears, does it require a Stage 4 patch?

- **For SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO / FORECAST:** no — the generic renderer handles it.
- **For EXPLANATION / KNOWLEDGE / CLARIFY:** no — narrative LLM summariser, guided by skills.
- **For the analyzer mis-routing the question entirely:** that is the live quality work — fixed by prompt few-shots and cross-check tuning, not new stages.
