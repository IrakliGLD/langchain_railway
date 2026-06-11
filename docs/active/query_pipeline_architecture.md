# Query Pipeline Architecture

Technical reference for the `langchain_railway` query pipeline. Describes the current runtime, the Ideal Decision Tree it targets, and the structural work that is still open.

**Last updated:** 2026-05-13
**Source of truth:** the code referenced inline below. When this document and the code disagree, the code wins — update this document.

---

## 1. Executive Summary

Stage 0.2 is the single **interpretation** point: the one LLM call that interprets the user question. It emits a full answer contract — `answer_kind`, `render_style`, `grouping`, `entity_scope`, `candidate_tools`, `params_hint`, `evidence_roles`, `derived_metrics`, `visualization` — and downstream stages execute it without re-interpretation. Other LLM calls do exist downstream (the narrative summarizer for EXPLANATION/KNOWLEDGE shapes, the FORECAST composer, the legacy SQL planner on the fallback path, and the OpenAI fallback on Gemini failure) — they *render or recover*, they do not re-interpret the question.

After the 2026-05-10 F.5 + §5.1 refactor series, the entire tool-execution surface is one function (`_execute_evidence_plan`) called once from `process_query`. It contains the four-strategy primary invocation picker (`_pick_primary_invocation`), the shared executor (`_execute_evidence_step`), the secondary evidence loop, and the post-loop driver-context enrichment. Evidence frames + the generic renderer cover six answer shapes (SCALAR, LIST, TIMESERIES, COMPARISON, SCENARIO, FORECAST) deterministically; narrative shapes (EXPLANATION, KNOWLEDGE, CLARIFY) go to a focus-aware LLM summarizer. The visualization plan flows through Stage 5 unchanged.

The remaining structural work is removing Stage 0.7 strategies from `_pick_primary_invocation` once two-week production hit-rate data confirms they can go (F.6). Quality work (analyzer routing accuracy, narrative grounding) is ongoing and uses the [`pipeline-failure-diagnostics`](../../skills/pipeline-failure-diagnostics/SKILL.md) developer skill as its playbook.

---

## 2. Current Runtime Flow

### 2.1 High-Level Pipeline

```text
HTTP /ask
  → Stage 0       prepare_context
  → Stage 0.2     question analyzer (LLM) → full answer contract
                  answer_kind cross-check (with legal-list exception)
  → Stage 0.3     vector knowledge retrieval (three-tier FULL/LIGHT/SKIP)
                  response_mode + resolution_policy derivation (inline)
  → Stage 0.4     evidence planner (validates plan against answer_kind)
  → Stage 0.5/0.7 primary tool execution
                  (four-strategy chain via _pick_primary_invocation;
                  the Stage 0.5 / Stage 0.7 split is a trace-shape concern
                  inside one orchestration block)
  → Stage 0.8     secondary evidence loop (additional plan steps)
                  → balancing driver-context enrichment (post-loop, conditional)
  → Stage agent   legacy agent loop (only when no authoritative analyzer)
  → Stages 1/2    legacy SQL fallback (only when typed tools didn't produce
                  primary data)
  → Stage 3       analyzer enrichment (dispatches on contract flags)
  → Stage 4       generic renderer OR specialized formatter OR LLM summarizer
  → Stage 5       chart pipeline (consumes VisualizationInfo, multi-group)
```

### 2.2 Short-Circuit Paths

- `ResolutionPolicy.CLARIFY` → `summarizer.answer_clarify()` and return.
- `ResponseMode.KNOWLEDGE_PRIMARY` → `summarizer.answer_conceptual()` and return.
- Generic renderer succeeds → skip LLM summarizer.
- `share_summary_override` (deterministic share-composition formatter) → skip LLM summarizer.
- Derived chart builder produces a chart spec → preserved through Stage 5 instead of falling back to `ctx.df`.
- Missing-derived-evidence check before Stage 4 → may CLARIFY if no usable analysis evidence exists.

### 2.3 Ideal Decision Tree

**Design principle: Stage 0.2 is the one *interpretation* call. Make it emit the full contract. Everything after executes or renders — no re-interpretation.** (Rendering may itself use an LLM — see §3.9 — but always against the Stage 0.2 contract, never against a fresh reading of the question.)

The current pipeline matches this tree closely. After §5.1 the entire tool-execution surface is one orchestration **function** (`_execute_evidence_plan`) called once from `process_query`; internally it still runs three passes (primary strategy chain, secondary loop, driver enrichment) rather than literally one for-loop over `evidence_plan`. Where this document says "single loop", read "single orchestration function".

```text
HTTP /ask
│
├─ Stage 0: prepare_context
│   → detect language, select light/analyst mode, init context
│   → cheap, no LLM
│
├─ Stage 0.2: question analyzer (THE LLM call — firm contract)
│   │ Single point where the question is interpreted. Every field
│   │ downstream stages need is emitted here.
│   │
│   │ Emitted fields:
│   │   query_type, preferred_path, candidate_tools, params_hint,
│   │   evidence_roles, derived_metrics, canonical_query_en,
│   │   answer_kind, render_style, grouping, entity_scope,
│   │   visualization (full).
│   │
│   │ NOTE: primary_tool is NOT emitted. candidate_tools[0] feeds the
│   │ evidence planner, but until F.6 lands the runtime picker is a
│   │ four-strategy priority cascade (§3.6: plan-driven → keyword router
│   │ → analyzer-built → router fallback) — not a single deterministic
│   │ source.
│   │
│   │ Active cross-check: LLM-emitted vs query_type-derived answer_kind.
│   │ Disagreement is logged; the safer option is preferred unless the
│   │ legal-list exception applies (LIST + regulatory/conceptual +
│   │ confidence ≥ 0.85).
│   │
│   │ Fallback when the analyzer is disabled/fails: derive answer_kind
│   │ from query_type mapping where unambiguous; derive tool from the
│   │ keyword router (legacy).
│   │
│   → The analyzer's output is the contract. Downstream stages trust it.
│
├─ Stage 0.3: vector knowledge retrieval (three-tier)
│   ├─ KNOWLEDGE / EXPLANATION → FULL (top-K=6, re-ranked)
│   ├─ data + NARRATIVE → LIGHT (top-K=2, no re-rank)
│   └─ data + DETERMINISTIC, or CLARIFY → SKIP
│
├─ Stage 0.4: evidence planner
│   │ Reads the analyzer contract; validates plan against answer_kind:
│   │   LIST → entity-enumeration step present
│   │   COMPARISON → two periods or two entities
│   │   TIMESERIES → period range
│   │   SCENARIO → scenario params available
│   → Ordered list of tool steps, validated against the answer contract.
│
├─ Tool execution + evidence collection (one orchestration function)
│   │ Implemented by _execute_evidence_plan (§3.6): primary execution
│   │ (strategy chain over the first plan step) + secondary loop (over
│   │ remaining steps) + driver enrichment, as three passes inside one
│   │ function. A literal single for-loop over evidence_plan remains a
│   │ nice-to-have, not a gap.
│   │
│   │ for each evidence step:
│   │   1. pick invocation (plan-driven; for the primary step also
│   │      try keyword router / analyzer-built / router fallback)
│   │   2. execute tool (caller passes its local execute_tool binding
│   │      so test patches still apply)
│   │   3. normalise output into canonical evidence frame
│   │   4. validate relevance (primary only today; secondary trusts
│   │      the planner's tool choice)
│   │   5. store evidence by role; bind provenance refs at construction
│   │
│   │ On primary failure:
│   │   ├─ analyzer-source → _attempt_analyzer_tool_recovery
│   │   ├─ otherwise → mark fallback reason; fall through to SQL
│   │   └─ no plan steps + SQL fallback possible → SQL escape hatch
│
├─ Stage 3: analyzer enrichment — dispatches on answer_kind + contract flags
│   │ answer_kind = SCALAR/TIMESERIES + share signal → share enrichment
│   │ answer_kind = SCENARIO → scenario evidence dispatch
│   │ answer_kind = FORECAST → trendline pre-calculation
│   │ answer_kind = EXPLANATION → causal reasoning + correlation
│   │ answer_kind = COMPARISON → MoM/YoY derived metrics
│   │ Outputs are canonical evidence frames consumable by derived
│   │ chart builders.
│
├─ Stage 4: answer rendering — switch on answer_kind
│   │
│   ├─ [render_style = DETERMINISTIC]
│   │   Generic renderer over evidence frames:
│   │   ├─ SCALAR     → value + unit + period
│   │   ├─ LIST       → enumerate entities, group by reason
│   │   ├─ TIMESERIES → format period-indexed rows
│   │   ├─ COMPARISON → subject vs baseline + delta
│   │   └─ SCENARIO   → payoff breakdown
│   │   Specialized formatter:
│   │   └─ share_summary_override (renewable/thermal PPA decomposition)
│   │
│   ├─ [answer_kind = FORECAST] — ALWAYS through LLM (since 2026-05-22).
│   │   Stage 3 enrichment pre-computes trendline values + R² and
│   │   emits the ``--- TRENDLINE FORECASTS ---`` block into
│   │   ``ctx.stats_hint``. The horizon is capped at history_years/2
│   │   by ``_cap_trendline_horizon_to_history_depth`` so the numbers
│   │   stay statistically defensible. The LLM consumes stats_hint
│   │   + forecast-caveats.md (loaded into skill guidance by
│   │   ``core/llm.py`` when forecast keywords appear) and produces
│   │   the narrative with R²-tiered reliability templates, the
│   │   long-horizon "focus on structural drivers" rule, and the
│   │   July 2027 target-market regime-break warning. The
│   │   deterministic ``generic_renderer._render_forecast`` was
│   │   orphan code from the production path and was deleted on
│   │   2026-06-10 (A2). Production trace c507e4d7
│   │   (2026-05-22) showed why this matters: the deterministic
│   │   renderer shipped "Winter (GEL): 12.49 GEL/MWh" — winter
│   │   forecast 10× lower than current — because it rubber-stamped
│   │   the one surviving trendline without judgment.
│   │
│   └─ [render_style = NARRATIVE]
│       → LLM summarizer receives pre-structured evidence frames
│       → focused on explanation quality; provenance refs pre-bound
│       → post-hoc provenance gate catches hallucinated numbers
│
└─ Stage 5: chart pipeline — consumes VisualizationInfo, multi-group
    → Built from derived chart builder specs (MoM/YoY, index growth,
      decomposition, forecast, seasonal) OR canonical evidence frames.
    → Not from raw ctx.df by default.
```

---

## 3. Current Stage Semantics

Per-stage description of what the runtime does today.

### 3.1 Stage 0 — Prepare Context

Detects language, picks light/analyst mode, runs the heuristic conceptual classifier as a fallback signal, initialises `QueryContext`. Cheap, no LLM. Also detects clarification-selection replies ("1", "option 2") and rewrites the query to the picked option before analysis.

### 3.2 Stage 0.2 — Structured Question Analyzer

The semantic centre of the pipeline. Emits the full contract listed in §2.3. Prompt assembly is dynamic: `_classify_analyzer_prompt_profile` + `_build_analyzer_prompt_blocks` choose ordered blocks per question family, and section-aware truncation drops the least-relevant blocks first per `_ANALYZER_TRUNCATION_DATA` / `_ANALYZER_TRUNCATION_KNOWLEDGE`. Fast mode swaps the budget for `FAST_MODE_ANALYZER_BUDGET`; deep mode reads `ANALYZER_PROMPT_BUDGET_MAX_CHARS` (separate from `SUMMARIZER_PROMPT_BUDGET_MAX_CHARS`; both default to the legacy `PROMPT_BUDGET_MAX_CHARS`).

Active cross-check (`_cross_check_answer_kind`) compares the LLM-emitted `answer_kind` against a `query_type`-derived value every call. On disagreement the safer option is preferred unless the legal-list exception fires (LIST + regulatory/conceptual + confidence ≥ 0.85). The scenario-metric override (EXPLANATION/SCALAR/TIMESERIES → SCENARIO when the analyzer emits a scenario-family derived metric) is additionally gated on a quantitative anchor in the user query — a digit, percent, or multiplicative/directional word — to prevent garbage renderer output when the analyzer hallucinates a `scenario_factor` from an anchorless question (2026-05-13 fix).

### 3.3 Stage 0.3 — Vector Knowledge Retrieval

Three-tier (`VectorRetrievalTier`): FULL for knowledge/explanation, LIGHT (top-K=2, no re-rank) for narrative data, SKIP for deterministic data and CLARIFY. The tier is computed by `_resolve_vector_retrieval_tier` from `answer_kind` + `render_style`.

**Known coupling risk:** the tier inherits Stage 0.2's classification. If the analyzer mislabels a question that genuinely needs regulatory/conceptual grounding as deterministic data, retrieval is SKIPped and the answer is silently ungrounded — the misclassification costs twice. Tiering is a deliberate cost optimization; treat retrieval-starved wrong answers as a §5.3 routing failure, not a retrieval bug.

**Cross-reference expansion (Phase A, env-gated).** Regulatory documents are riddled with cross-references like `მე-14 მუხლის მე-7 პუნქტი` ("paragraph 7 of article 14") that point to articles outside the top-K matched set. Phase A handles the cheapest slice — same-document adjacency — without any schema change. `resolve_adjacent_chunks` in [`knowledge/vector_retrieval.py`](../../knowledge/vector_retrieval.py) computes `(document_id, chunk_index ± 1)` for each top-K hit, deduplicates, excludes chunks already in the bundle, and fetches them via `KnowledgeVectorStore.fetch_chunks_by_index` using the existing `(document_id, chunk_index)` index. The result lives on `VectorKnowledgeBundle.adjacent_chunks`. Controlled by `VECTOR_ADJACENCY_MODE ∈ {off, shadow, on}` (default `off`): `shadow` fetches and emits a `stage_0_3_vector_knowledge_adjacency` trace event without changing pack output; `on` flips the pack function to append adjacency entries (tagged `| adjacent` in their header) after the primary chunks within the same `VECTOR_KNOWLEDGE_MAX_CHARS` budget. Primary chunks always win under budget pressure. See [`VECTOR_KNOWLEDGE_ROLLOUT.md`](VECTOR_KNOWLEDGE_ROLLOUT.md) "Adjacency expansion" for the rollout path.

**Reference expansion (Phase B, env-gated).** Phase B parses each chunk's text at ingest time to populate `outgoing_refs` (canonical `(kind, number, sub_kind, sub_number)` tuples covering Georgian suffix-/prefix-ordinal forms, decimal articles, ordinal-word paragraph qualifiers, Roman chapter refs, English/Russian variants — see `knowledge/vector_reference_parser.py`). At retrieval, `resolve_reference_chunks` follows each top-K chunk's article-kind refs and resolves them via the `(document_id, article_number)` partial index. Self-article anchors (`ამ მუხლის`) are tagged at parse time and skipped by the resolver. External-code refs (`კოდექსი`) are rejected at parse time to prevent false positives. Controlled by `VECTOR_REFERENCE_EXPANSION_MODE ∈ {off, shadow, on}` (default `off`); when `on`, resolved chunks pack with `| referenced` tag before any adjacency entries (references are higher signal than adjacency siblings). Per-chunk budget 3, total request budget 10 prevent expansion avalanches. See [`VECTOR_KNOWLEDGE_ROLLOUT.md`](VECTOR_KNOWLEDGE_ROLLOUT.md) "Reference expansion" for the schema migration prerequisite, re-ingestion requirement, rollout path, and known limitations (chapter resolution deferred; cross-document with quoted titles not addressed).

### 3.4 Response Mode + Resolution Policy (Inline)

Derived inline from the analyzer contract — no separate stage. `KNOWLEDGE_PRIMARY` and `CLARIFY` short-circuit to the relevant summarizer entry point. Everything else continues to evidence planning.

### 3.5 Stage 0.4 — Evidence Planner

Reads `answer_kind`, `render_style`, `candidate_tools`, `evidence_roles`. `_validate_plan_against_answer_kind` ensures planned steps will produce evidence matching the answer shape (LIST → entity-enumeration step; COMPARISON → two-period; TIMESERIES → period range; SCENARIO → scenario params). Mismatches are flagged at planning time, not after wasted tool calls.

### 3.6 Stages 0.5–0.8 — Evidence-Plan Execution

After §5.1 (commits `ea677dc`, `f55d18b`) the entire tool-execution surface is one function: `_execute_evidence_plan` in [`agent/pipeline.py`](../../agent/pipeline.py). `process_query` calls it once. The function runs three logical passes:

#### Pass 1 — Primary execution (strategy chain)

The strategy picker `_pick_primary_invocation` returns the first non-None invocation from a four-step chain:

1. **Plan-driven** — first unsatisfied step in `ctx.evidence_plan`.
2. **Keyword router** — `match_tool` on raw query (only when no authoritative analyzer).
3. **Analyzer-built** — `planner.build_tool_invocation_from_analysis` from the QuestionAnalysis.
4. **Authoritative router fallback** — `match_tool` reused, gated on `_should_attempt_authoritative_router_fallback`.

The historical split between "Stage 0.5" and "Stage 0.7" survives only as observability: strategies 1–2 fire under `stage_0_5_*` trace shapes, strategies 3–4 fire under `stage_0_7_*` shapes with `metrics.log_stage_0_7("entered" | "invocation_built" | "used_result")` counters. The execution body is the same `_execute_evidence_step` helper either way:

1. Execute tool (using a caller-supplied executor binding for testability).
2. Optionally emit `stage_0_6_tool_execute` trace + `metrics.log_tool_call`.
3. Normalise the DataFrame.
4. (Primary only) Stamp `ctx.df`, `ctx.cols`, `ctx.rows`, provenance refs, plan defaults.
5. Validate tool relevance (primary only).
6. On block: clear ctx and mark `tool_fallback_reason`; on pass: log "Typed/Analyzer tool relevance validated".
7. Store result in `ctx.evidence_collected[step.role]`; mark plan step satisfied.

Failure handling is source-aware: analyzer-source failures route to `_attempt_analyzer_tool_recovery`; plan/keyword-router failures mark the plan step and fall through to the legacy SQL escape hatch.

#### Pass 2 — Secondary evidence loop (was Stage 0.8)

When the evidence plan still has unsatisfied steps, the loop iterates over them with a per-loop budget (`EVIDENCE_LOOP_BUDGET_SECONDS`) and capped iterations (`_EVIDENCE_LOOP_MAX_STEPS=3`). Each step calls the same `_execute_evidence_step` helper with secondary semantics (`is_primary=False`, `validate_relevance=False`, no per-step trace or `log_tool_call`) — preserves the historical observability of one outer `stage_0_8_evidence_loop` trace rather than per-step.

The loop body lives in `_run_secondary_evidence_loop` in `pipeline.py` (moved out of `evidence_planner.py` in §5.1.a, commit `ea677dc`). `evidence_planner.execute_remaining_evidence` remains as a thin delegate so tests that monkey-patch `agent.evidence_planner.execute_tool` continue to intercept tool calls — the delegate passes its local `execute_tool` binding to the helper via the `executor=` parameter.

After the loop, `merge_evidence_into_context` joins secondary frames into `ctx.df` via date-aligned joins.

#### Pass 3 — Driver-context enrichment

When primary execution produced a usable result (`ctx.used_tool` and `ctx.tool_name` set), `_enrich_prices_with_balancing_driver_context` attaches source-price and contribution columns for balancing-price answers.

### 3.7 Stage agent_loop, Stages 1/2 — Legacy Fallbacks

The agent loop (`orchestrator.run_agent_loop`) is gated on `ENABLE_AGENT_LOOP and not ctx.has_authoritative_question_analysis and not analyzer_tool_failed` — it fires only when there is no firm Stage 0.2 contract. May set `ctx.used_tool=True` and exit early as `conceptual_exit` or `data_exit`.

Stages 1/2 (`planner.generate_plan` + `sql_executor.validate_and_execute`) fire when `not ctx.used_tool` after the above. The condition is the §2.3 ideal's "SQL escape hatch when no typed tool produced primary data". Conceptual or `skip_sql` paths route to `summarizer.answer_conceptual` and return.

### 3.8 Stage 3 — Analyzer Enrichment

`analyzer.enrich` dispatches on analyzer-emitted contract flags (`needs_correlation_context`, `needs_driver_analysis`, requested `derived_metrics`, `analyzer_indicates_share_intent`). Computes correlation, MoM/YoY, share decomposition, scenario evidence, forecast trendlines. No regex-based eligibility detection. Outputs become `ctx.analysis_evidence` and feed both the LLM summarizer and the derived chart builders.

A "missing requested evidence" check after Stage 3 can CLARIFY if the analyzer asked for derived metrics that Stage 3 couldn't produce, unless partial evidence is available.

### 3.9 Stage 4 — Generic Renderer + Specialized Formatter + LLM Summarizer

`summarizer.summarize_data` dispatch order:

1. **Generic renderer** (`generic_renderer.render`) — handles SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO from canonical evidence frames. FORECAST is excluded since 2026-05-22 (commit `ef048d6`); it routes through the LLM so `forecast-caveats.md` judgment can be applied to the deterministic trendline values pre-computed in Stage 3. Returns None when the shape isn't covered or evidence isn't suitable.
2. **share_summary_override** — deterministic specialized formatter for share-intent queries (renewable/thermal PPA decomposition + per-period price join). Sits alongside SCENARIO/FORECAST as a permanent specialized formatter per §2.3 (see §5.3 for why it's not legacy debt).
3. **Other specialized direct answers** — regulated-tariff-list, residual-weighted-price.
4. **LLM `llm_summarize_structured`** — for `render_style=NARRATIVE` shapes. Receives focus-aware prompts assembled from [`skills/answer-composer/`](../../skills/answer-composer/SKILL.md). Prompt budget reads `SUMMARIZER_PROMPT_BUDGET_MAX_CHARS` (independent of the analyzer budget). Truncation profile selected by `answer_kind`: `_TRUNCATION_PRIORITY_DATA` / `_KNOWLEDGE` / `_EXPLANATION` / `_FORECAST_SCENARIO`; the profile ordering follows three invariants pinned by tests — `UNTRUSTED_CONVERSATION_HISTORY` dropped first in every profile, `UNTRUSTED_EXTERNAL_SOURCE_PASSAGES` preserved last for KNOWLEDGE, `UNTRUSTED_STATISTICS` preserved last for data-grounded profiles (see [`tests/test_vector_retrieval_tier.py`](../../tests/test_vector_retrieval_tier.py)).

A post-hoc provenance gate runs after the summary. It is a no-op for canonical-frame paths (claims list is empty → `gate_passed=True, reason="no_claims"`); for narrative LLM summaries it catches numeric hallucinations and replaces the answer with a `citation_gate_fallback` on coverage failure.

**Coverage boundary (accepted gap, not an oversight):** the gate therefore protects only narrative LLM answers. Deterministic renders (generic renderer + specialized formatters) ship with **no post-hoc numeric check** — their correctness rests entirely on evidence-frame construction, which is validated at attach time (`evidence_validator`) for *shape*, not values. If a frame is built from the wrong rows, the renderer prints the wrong number confidently. Mitigation lives upstream (planner validation, tool relevance checks), not in Stage 4.

### 3.10 Stage 5 — Chart Pipeline

`chart_pipeline.build_chart` consumes the analyzer's `VisualizationInfo` directly. The chart frame source is selected first: derived chart builders produce specs for MoM/YoY, index growth, decomposition, forecast, and seasonal answers; otherwise canonical evidence frames feed `chart_frame_builder`. Multi-group plans are iterated — `chart_groups` produces one chart per group, with per-group `type`, `title`, `y_axis_label`, and `metrics` honoured.

---

## 4. Module Responsibilities & Source of Truth

Runtime modules and the files they live in. If this list disagrees with the codebase, the codebase wins.

### Orchestration

- **[`agent/pipeline.py`](../../agent/pipeline.py)** — `process_query` orchestrates the full pipeline. Stage tracing via `_emit_trace_stage` + local `_trace_stage` closure. Holds: `_cross_check_answer_kind` (with the legal-list exception), `_pick_primary_invocation` (the four-strategy chain), `_execute_evidence_step` (the shared tool executor), `_run_secondary_evidence_loop` (the Stage 0.8 body, moved from `evidence_planner.py` in §5.1.a), `_execute_evidence_plan` (the §5.1.b consolidated function combining primary execution + secondary loop + driver enrichment), `_attempt_analyzer_tool_recovery` (analyzer-source recovery candidate). After §5.1 the function `process_query` calls one line for the entire tool-execution surface: `ctx = _execute_evidence_plan(ctx)`.

### Analyzer + Routing

- **[`core/llm.py`](../../core/llm.py)** — All LLM call sites and prompt assembly. Owns `_classify_analyzer_prompt_profile`, `_build_analyzer_prompt_blocks`, the catalogue JSON cached at module load, `_ANALYZER_TRUNCATION_*` priorities, `_TRUNCATION_PRIORITY_*` summarizer profiles by `answer_kind`, fast-mode budget overrides, OpenAI fallback on Gemini failure.
- **[`agent/planner.py`](../../agent/planner.py)** — `prepare_context`, language detection, mode selection, heuristic conceptual classifier, `build_tool_invocation_from_analysis` (strategy 3 in the primary picker), legacy `generate_plan` for SQL fallback.
- **[`agent/router.py`](../../agent/router.py)** — `match_tool` keyword+semantic tool router (strategies 2 and 4 in the primary picker).
- **[`contracts/question_analysis.py`](../../contracts/question_analysis.py)**, **[`question_analysis_catalogs.py`](../../contracts/question_analysis_catalogs.py)** — the Stage 0.2 contract: `QuestionAnalysis` Pydantic model with `AnswerKind`, `RenderStyle`, `Grouping`, `VisualizationInfo`, `FilterCondition`, `MeasureTransform`, `VisualizationTimeGrain`, `SeriesSplitMode`, `SortRule`, `ChartFamily`, `VisualGoal`, `PresentationMode`, `SemanticRole`, `ChartIntent`. Catalogs supply the LLM-facing JSON that explains each enum.
- **[`schemas/question_analysis.schema.json`](../../schemas/question_analysis.schema.json)** — JSON-schema snapshot of the Pydantic model, asserted by `test_question_analysis_contract.py::test_schema_snapshot_matches_runtime_model`.

### Evidence Collection

- **[`agent/evidence_planner.py`](../../agent/evidence_planner.py)** — builds and validates the evidence plan (`_validate_plan_against_answer_kind`); holds `merge_evidence_into_context` (date-aligned joins of secondary frames into `ctx.df`). `execute_remaining_evidence` is a thin delegate after §5.1.a — the secondary loop body itself lives in `pipeline.py` as `_run_secondary_evidence_loop`; the delegate stays so tests that monkey-patch `agent.evidence_planner.execute_tool` keep working.
- **[`agent/evidence_validator.py`](../../agent/evidence_validator.py)** — `validate_evidence` runs inline during frame attachment; checks frames against `answer_kind` shape requirements (shared with the planner via [`agent/shape_requirements.py`](../../agent/shape_requirements.py)).
- **[`agent/frame_adapters.py`](../../agent/frame_adapters.py)** — per-tool adapters that normalise raw tool output into `ObservationFrame` / `EntitySetFrame` / `ComparisonFrame`.
- **[`agent/tools/`](../../agent/tools/)** — typed tool implementations.
- **[`agent/tool_adapter.py`](../../agent/tool_adapter.py)** — runs typed tools with timeouts; produces compact preview text for LLM contexts.
- **[`contracts/evidence_frames.py`](../../contracts/evidence_frames.py)** — `ObservationFrame`, `EntitySetFrame`, `ComparisonFrame` definitions.
- **[`contracts/vector_knowledge.py`](../../contracts/vector_knowledge.py)** — `VectorRetrievalTier` enum + vector-knowledge bundle types.
- **[`knowledge/vector_retrieval.py`](../../knowledge/vector_retrieval.py)** — three-tier retrieval implementation.
- **[`agent/provenance.py`](../../agent/provenance.py)** — `stamp_provenance`, `clear_provenance`, `tool_invocation_hash`, `sql_query_hash`.

### Analysis & Enrichment

- **[`agent/analyzer.py`](../../agent/analyzer.py)** — `enrich` runs Stage 3. Contract-driven dispatch on emitted flags and derived-metric requests.
- **[`agent/metric_registry.py`](../../agent/metric_registry.py)** — per-metric computation registry (MoM, YoY, CAGR, share-decomposition, correlation). `analyzer.py` dispatches to these instead of a monolithic if/elif chain.
- **[`agent/aggregation.py`](../../agent/aggregation.py)** — aggregation-intent detection + SQL guard helpers.

### Rendering

- **[`agent/summarizer.py`](../../agent/summarizer.py)** — Stage 4 entry. Tries the generic renderer first; otherwise calls `llm_summarize_structured` with focus-aware prompts. `share_summary_override` is a deliberate specialized formatter (see §5.3). Conceptual answers go to `answer_conceptual`. Owns `_enforce_provenance_gate` (the post-hoc grounding check).
- **[`agent/generic_renderer.py`](../../agent/generic_renderer.py)** — deterministic rendering for SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO from evidence frames. FORECAST excluded since 2026-05-22 — routes through LLM (see §2.3 ideal-tree).
- **[`agent/chart_pipeline.py`](../../agent/chart_pipeline.py)** — Stage 5 entry. Selects chart frame source, applies `measure_transform` / `time_grain`, iterates `chart_groups`.
- **[`agent/chart_frame_builder.py`](../../agent/chart_frame_builder.py)** — builds chart-shaped frames from canonical evidence frames.
- **[`agent/derived_chart_builder.py`](../../agent/derived_chart_builder.py)** — specs for MoM/YoY, index growth, decomposition, forecast, seasonal charts.
- **[`visualization/chart_selector.py`](../../visualization/chart_selector.py)** — `should_generate_chart` gate (consults `VisualizationInfo` + row count + answer_kind).

### Skills

Loaded into LLM prompts at request time:

- **[`skills/question-analyzer/`](../../skills/question-analyzer/)** — JSON-contract guidance for the Stage 0.2 prompt.
- **[`skills/sql-planner/`](../../skills/sql-planner/)** — chart-strategy rules, focus guidance, SQL patterns for the legacy planner.
- **[`skills/answer-composer/`](../../skills/answer-composer/)** — Stage 4 narrative templates and focus catalogs (Focus: Regulation, Focus: Balancing, etc.).
- **[`skills/energy-analyst/`](../../skills/energy-analyst/)** — domain reasoning references (entity taxonomy, driver framework, seasonal rules) for energy-domain summaries.

Developer-only (NOT loaded into prompts):

- **[`skills/developer-phased-audit/`](../../skills/developer-phased-audit/)** — workflow for phased implementation.
- **[`skills/pipeline-failure-diagnostics/`](../../skills/pipeline-failure-diagnostics/)** — log-reading playbook and failure taxonomy for triaging Q&A failures.

### Legacy / Fallback

- **[`agent/orchestrator.py`](../../agent/orchestrator.py)** — legacy agent loop. Fires only when there is no authoritative analyzer.
- **[`agent/sql_executor.py`](../../agent/sql_executor.py)** — Stage 2 SQL execution + safety checks for the legacy SQL escape hatch.

---

## 4.1 Deployment Constraint: Single Replica

The service holds request-scoped *and* cross-request state **in process memory**: rate-limit
buckets (`main.py`), session-bound conversation history (`utils/session_memory.py`), the LLM
response cache (`core/llm.py`), and circuit-breaker state (`utils/resilience.py`).

**The deployment assumption is exactly one worker process / one replica.** With N replicas:
rate limits multiply by N, sessions issued on one replica are unknown to the others, and
cache/breaker state diverges. A shared-store migration (Redis) was evaluated and **declined**
on 2026-06-10 (owner decision: no new runtime infrastructure — see
`medium_fix_plan_2026-06-10.md` P5). If horizontal scaling ever becomes necessary, that
decision is the one to revisit *first*; do not scale out without it.

---

## 5. What Still Needs to Be Fixed

Priority-sorted. Each item is paired with the verification needed before removal/refactor.

### 5.1 Pipeline Consolidation — Single Execution Loop (DONE)

**What was done (2026-05-10):** Merged primary execution + secondary evidence loop + driver-context enrichment into one `_execute_evidence_plan` function in `pipeline.py`. `process_query` calls it once.

| Sub-phase | Scope | Commit |
|---|---|---|
| §5.1.a | Move `execute_remaining_evidence` body from `evidence_planner.py` into `_run_secondary_evidence_loop` in `pipeline.py`; remove the F.5c circular-import workaround. `evidence_planner.execute_remaining_evidence` becomes a thin delegate so tests that monkey-patch `agent.evidence_planner.execute_tool` keep working via the helper's `executor=` parameter. | `ea677dc` |
| §5.1.b | Introduce `_execute_evidence_plan(ctx)` containing the three passes (primary strategy chain + secondary loop + driver enrichment) + circuit-breaker preflight. Replace the corresponding ~245-line block in `process_query` with one call. | `f55d18b` |
| §5.1.c | Doc update describing the unified function; §3.6 absorbs the former §3.7 since Stages 0.5–0.8 are now one function. | this commit |

**Verification:** the [targeted test suite](../../skills/developer-phased-audit/references/targeted-suite.md) passes at every sub-phase. Diff symmetry inspection confirms every `metrics.log_*` line and every `_trace_stage`/`_emit_trace_stage` call from the old block has a matching counterpart in the new function — pure structural move with no logic change. Trace shapes (`stage_0_5_plan_driven`, `stage_0_5_router_match`, `stage_0_6_tool_execute`, `stage_0_7_analyzer_route`, `stage_0_7_analyzer_tool_execute`, `stage_0_8_evidence_loop`) and metric counters fire under identical conditions to the pre-§5.1 form.

After §5.1, the §2.3 Ideal Decision Tree's "Tool execution + evidence collection" node is implemented by `_execute_evidence_plan` — one orchestration function with three internal passes (see the §2.3 note on terminology).

### 5.2 Stage 0.7 Removal (F.6, gated)

**What:** Remove the analyzer-built (strategy 3) and authoritative router-fallback (strategy 4) branches from `_pick_primary_invocation`. With them gone, Stage 0.7 disappears entirely; the strategy chain becomes plan-driven + keyword-router (the latter only when no authoritative analyzer).

**Gate:** Two-week production data from the F.2 hit-rate counters (`stage_0_7_entered`, `stage_0_7_invocation_built`, `stage_0_7_used_result`) showing `used_result / requests < 5%`. Until that is observed, removal is premature.

**Risk:** Small once the data clears. The change is deletion + a few related metric/trace removals.

**Files:** [`agent/pipeline.py`](../../agent/pipeline.py), [`agent/router.py`](../../agent/router.py).

### 5.3 Analyzer Misclassification — Standing Quality Area

The architecture trades regex brittleness for LLM non-determinism. Cross-check, evidence validator, and the legal-list exception are mitigations, not eliminations. Mis-routing surfaces as wrong `query_type`, wrong `answer_kind`, or a confidence flag mismatch, and the fix layer depends on which contract was actually wrong.

**Where the work lives:** analyzer prompt (few-shot examples, vocabulary), `_cross_check_answer_kind` policy, runtime skills (`answer-composer`, `energy-analyst`).

**Playbook:** [`skills/pipeline-failure-diagnostics/references/failure-taxonomy.md`](../../skills/pipeline-failure-diagnostics/references/failure-taxonomy.md) Pattern A (schema crash → cascade), Pattern B (cross-check override), Pattern C (router misclassification), Pattern D (heuristic disagreement). The 2026-05-09 fix series ("who can trade…", "why did price change…", "trend and structure of supply") is the canonical worked example: four edits across prompt + cross-check + answer-composer skill, no new stage. The 2026-05-13 log-driven series is the second worked example: five edits across post-LLM narrative scrubbing, the SCENARIO override gate, clarify-selection detection, per-stage prompt budgets, and truncation-priority invariants — again no new stage.

This is ongoing — every new failure report potentially adds a few-shot example, a cross-check refinement, or an answer-composer rule.

### 5.4 `share_summary_override` Is A Specialized Formatter — Not Removable

Earlier versions of this document listed `share_summary_override` as legacy debt to be absorbed into the generic renderer. A 2026-05-10 audit determined this was a misclassification: it is a deliberate **specialized formatter** for share-intent queries, in the same category as SCENARIO and FORECAST per §2.3.

The artifact decomposes `share_all_ppa` into its renewable/thermal components and joins per-period prices — domain-specific knowledge that the generic renderer intentionally does not have. The decision to build it is gated on the structured analyzer signal `ctx.analyzer_indicates_share_intent`, not on regex. See the inline comment at [`agent/analyzer.py`](../../agent/analyzer.py) §"Share summary override" for the design rationale.

**No action required.**

### 5.5 Post-Hoc Provenance Gate Is The Narrative Safety Net — Not Removable

Earlier versions of this document proposed removing the post-hoc provenance gate as redundant after frame-construction-time provenance binding. The 2026-05-10 audit determined this is wrong:

- For **canonical-frame paths** (generic renderer output) the gate is already a no-op — `summary_claims=[]` makes it return `gate_passed=True, reason="no_claims"` immediately.
- For **narrative LLM summaries** the gate is the safety net against numeric hallucination — it fired on the 2026-05-09 verification log when an ungrounded narrative produced `summary_source=structured_summary_grounding_fallback`. Frame-construction provenance binding cannot replace it because the narrative content is LLM-generated, not extracted.

**No action required.**

### 5.6 Small Open Items

- **`FilterCondition` audit.** The contract exists on `ToolParamsHint` and is consumed by tool executors. Audit periodically that new threshold-style queries route through the structured filter rather than ad-hoc post-fetch filtering inside summarizers. ([`contracts/question_analysis.py`](../../contracts/question_analysis.py))
- **Cross-tool computational queries.** Default is narrative rendering — let the LLM synthesise from pre-structured evidence. If a repeating pattern emerges, add a `derived_metric` type and let Stage 3 compute it (same shape as MoM/YoY/correlation). Principle: narrative as default, derived metric only when the pattern repeats. No structural change required.
- **Borderline `answer_kind=COMPARISON` chart vs table.** Today `primary_presentation` is consulted, but when the analyzer leaves it null the code falls back to row-count heuristics. The right move when uncertain is `chart_plus_table` rather than chart-only. ([`agent/chart_pipeline.py`](../../agent/chart_pipeline.py))
- **Reference lines.** `include_reference_lines` was dropped because a bare bool can't carry which axis/value/label. If reference-line rendering becomes needed, introduce a concrete `ReferenceLineSpec` dataclass first.

### 5.7 Triage Playbook (operational, not architectural)

For triaging Q&A failures — latency spikes, grounding failures, schema validation crashes, routing misclassification — consult [`skills/pipeline-failure-diagnostics/`](../../skills/pipeline-failure-diagnostics/). The skill is the source of truth for failure patterns and fix-layer selection; this architecture document deliberately does not duplicate that content.

---

## 6. Practical Conclusion

The pipeline is contract-driven end to end. Stage 0.2 emits the answer contract; evidence frames + the generic renderer cover five standard answer shapes deterministically (SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO); FORECAST plus the narrative shapes (EXPLANATION / KNOWLEDGE) go to a focus-aware LLM summarizer with pre-computed evidence in `stats_hint`; the visualization plan flows to Stage 5 without re-interpretation.

After the 2026-05-10 F.5 + §5.1 refactor series, the entire tool-execution surface is one function `_execute_evidence_plan` called once from `process_query` — one orchestration function with three internal passes (see §2.3 note). The remaining structural work is removing Stage 0.7 strategies once F.2 hit-rate data clears.

**The practical test:** when a new question family appears, does it require a Stage 4 patch?

- **SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO** → no, the generic renderer handles it.
- **FORECAST** → no, LLM summarizer with pre-computed trendline evidence in `stats_hint` and `forecast-caveats.md` in skill guidance.
- **EXPLANATION / KNOWLEDGE / CLARIFY** → no, narrative LLM summarizer guided by skills.
- **Share-composition queries** → no, `share_summary_override` (specialized formatter) handles it.
- **Analyzer mis-routing the question entirely** → that's the live quality work in §5.3 — fixed by prompt few-shots, cross-check tuning, or runtime-skill edits, not new pipeline stages.
