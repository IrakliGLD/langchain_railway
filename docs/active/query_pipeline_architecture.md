# Query Pipeline Architecture

Technical reference for the `langchain_railway` query pipeline. Describes the current runtime, the Ideal Decision Tree it targets, and the structural work that is still open.

**Last updated:** 2026-07-16 (F3 request-scoped deterministic canaries for P4 behavior activation and P4.B distinct terminal-outcome rendering. Earlier updates retain the analyzer/evidence architecture history below.)
**Source of truth:** the code referenced inline below. When this document and the code disagree, the code wins — update this document.

---

## 1. Executive Summary

Stage 0.2 is the single **interpretation** point: the one LLM call that interprets the user question. It emits a full answer contract — `answer_kind`, `render_style`, `grouping`, `entity_scope`, `candidate_tools`, `params_hint`, `evidence_roles`, `derived_metrics`, `visualization` — and downstream stages execute it without re-interpretation. Other LLM calls do exist downstream (the narrative summarizer for EXPLANATION/KNOWLEDGE shapes, the FORECAST composer, the legacy SQL planner on the fallback path, and the OpenAI fallback on Gemini failure) — they *render or recover*, they do not re-interpret the question.

After the 2026-05-10 F.5 + §5.1 refactor series, the entire tool-execution surface is one function (`_execute_evidence_plan`) called once from `process_query`. It contains the four-strategy primary invocation picker (`_pick_primary_invocation`), the shared executor (`_execute_evidence_step`), the secondary evidence loop, and the post-loop driver-context enrichment. Evidence frames + the generic renderer cover five answer shapes (SCALAR, LIST, TIMESERIES, COMPARISON, SCENARIO) deterministically; FORECAST routes through the LLM over pre-computed trendline evidence (§3.9); narrative shapes (EXPLANATION, KNOWLEDGE, CLARIFY) go to a focus-aware LLM summarizer. The visualization plan flows through Stage 5 unchanged.

The legacy agent loop is **deleted** (architecture-audit P2, 2026-07): with an authoritative Stage 0.2 contract it never fired, and analyzer-failure / no-tool cases fall through to the SQL fallback (§3.7). After the P0-4 decomposition, `process_query` itself is a thin stage driver built from `StageResult` stage functions (§4 Orchestration). The remaining structural work is removing Stage 0.7 strategies from `_pick_primary_invocation` once two-week production hit-rate data confirms they can go (F.6). Quality work (analyzer routing accuracy, narrative grounding) is ongoing and uses the [`pipeline-failure-diagnostics`](../../skills/pipeline-failure-diagnostics/SKILL.md) developer skill as its playbook.

The 2026-07-08 design-gap series closed the review's "the pipeline is a monologue" findings: every response carries an **answer-provenance block** (§3.9); deterministic renders get a **shadow fitness check** (§3.9); routing failures feed a **self-growing golden set** (§5.3); and two flag-gated-OFF capabilities await their §5 criteria — **contract continuity** (follow-ups as deltas over the previous turn's contract, §3.2/§5.7) and **evidence-triggered re-analysis** (one same-interpreter retry on surprising evidence, §3.6/§5.8). The ontology-agreement shadow (§5.9) measures when the first planner rule can migrate into the contract.

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
  → Stage 0.9     surprising-evidence detection (always; counters/trace)
                  → flag-gated ONE re-analysis (ENABLE_EVIDENCE_REANALYSIS
                    plus ENAI_EVIDENCE_REANALYSIS_PERCENT)
  → Stages 1/2    legacy SQL fallback (only when typed tools didn't produce
                  primary data; the agent loop that used to sit before this
                  was deleted — §3.7)
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
│   ├─ KNOWLEDGE / EXPLANATION → FULL (top-K=6, over-fetched candidate pool)
│   ├─ data + NARRATIVE → LIGHT (top-K=2, tighter candidate pool)
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
│   │
│   │ After execution: surprising-evidence detection (always; counters) →
│   │ flag-gated ONE re-analysis with the anomaly as trusted context —
│   │ the same Stage 0.2 interpreter, never a loop (§3.6).
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

**Contract continuity (2026-07-07, injection flag-gated OFF: `ENABLE_CONTRACT_CONTINUITY`).** After every successful request, a compact routed-contract snapshot ([`agent/contract_continuity.py`](../../agent/contract_continuity.py)) is stored on the session (`utils/session_memory.set_last_contract`, same TTL as history; storage is always on). With the flag on, the next turn's analyzer prompt receives it as a `TRUSTED_PREVIOUS_CONTRACT` block pinned right after the user question, instructing the LLM to interpret follow-ups ("and for 2023", "same in USD") as deltas over the previous contract instead of re-deriving intent from history text — which the truncation policy drops first. Truncation placement: drops second (right after history) in both analyzer profiles; for the clarify profile it moves to the preserved end alongside history. The snapshot participates in the analyzer cache key, so follow-ups can never hit a stale cached interpretation. Cutover criteria: §5.7.

### 3.3 Stage 0.3 — Vector Knowledge Retrieval

Three-tier (`VectorRetrievalTier`): FULL for knowledge/explanation, LIGHT (top-K=2, tighter candidate pool, no boost-term expansion) for narrative data, SKIP for deterministic data and CLARIFY. The tier is computed by `_resolve_vector_retrieval_tier` from `answer_kind` + `render_style`. There is no separate re-rank pass — candidates are over-fetched and ordered by similarity.

**Embedding caching (2026-07-07).** The embedding provider is a per-config singleton (`knowledge/vector_embeddings.get_embedding_provider`) so retrieval reuses one SDK client instead of constructing one per request, and query embeddings are memoized at the retrieval call site (`VECTOR_QUERY_EMBEDDING_CACHE_SIZE`, default 256, ≤0 disables) so repeated questions skip the embedding API round trip. Explicitly injected providers bypass the memo.

**Known coupling risk:** the tier inherits Stage 0.2's classification. If the analyzer mislabels a question that genuinely needs regulatory/conceptual grounding as deterministic data, retrieval is SKIPped and the answer is silently ungrounded — the misclassification costs twice. Tiering is a deliberate cost optimization; treat retrieval-starved wrong answers as a §5.3 routing failure, not a retrieval bug.

**Cross-reference expansion (Phase A, env-gated).** Regulatory documents are riddled with cross-references like `მე-14 მუხლის მე-7 პუნქტი` ("paragraph 7 of article 14") that point to articles outside the top-K matched set. Phase A handles the cheapest slice — same-document adjacency — without any schema change. `resolve_adjacent_chunks` in [`knowledge/vector_retrieval.py`](../../knowledge/vector_retrieval.py) computes `(document_id, chunk_index ± 1)` for each top-K hit, deduplicates, excludes chunks already in the bundle, and fetches them via `KnowledgeVectorStore.fetch_chunks_by_index` using the existing `(document_id, chunk_index)` index. The result lives on `VectorKnowledgeBundle.adjacent_chunks`. Controlled by `VECTOR_ADJACENCY_MODE ∈ {off, shadow, on}` (default `off`): `shadow` fetches and emits a `stage_0_3_vector_knowledge_adjacency` trace event without changing pack output; `on` flips the pack function to append adjacency entries (tagged `| adjacent` in their header) after the primary chunks within the same `VECTOR_KNOWLEDGE_MAX_CHARS` budget. Primary chunks always win under budget pressure. See [`VECTOR_KNOWLEDGE_ROLLOUT.md`](VECTOR_KNOWLEDGE_ROLLOUT.md) "Adjacency expansion" for the rollout path.

**Reference expansion (Phase B, env-gated).** Phase B parses each chunk's text at ingest time to populate `outgoing_refs` (canonical `(kind, number, sub_kind, sub_number)` tuples covering Georgian suffix-/prefix-ordinal forms, decimal articles, ordinal-word paragraph qualifiers, Roman chapter refs, English/Russian variants — see `knowledge/vector_reference_parser.py`). At retrieval, `resolve_reference_chunks` follows each top-K chunk's article-kind refs and resolves them via the `(document_id, article_number)` partial index. Self-article anchors (`ამ მუხლის`) are tagged at parse time and skipped by the resolver. External-code refs (`კოდექსი`) are rejected at parse time to prevent false positives. Controlled by `VECTOR_REFERENCE_EXPANSION_MODE ∈ {off, shadow, on}` (default `off`); when `on`, resolved chunks pack with `| referenced` tag before any adjacency entries (references are higher signal than adjacency siblings). Per-chunk budget 3, total request budget 10 prevent expansion avalanches. See [`VECTOR_KNOWLEDGE_ROLLOUT.md`](VECTOR_KNOWLEDGE_ROLLOUT.md) "Reference expansion" for the schema migration prerequisite, re-ingestion requirement, rollout path, and known limitations (chapter resolution deferred; cross-document with quoted titles not addressed).

### 3.4 Response Mode + Resolution Policy (Inline)

Derived inline from the analyzer contract — no separate stage (`_apply_response_mode` in `pipeline.py`). `KNOWLEDGE_PRIMARY` and `CLARIFY` short-circuit to the relevant summarizer entry point via the `_early_answer_conceptual` / `_early_answer_clarify` stage functions. Everything else continues to evidence planning.

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

**Concurrent prefetch (2026-07-07).** When `EVIDENCE_PARALLEL_SECONDARY` is on (default) and ≥2 steps remain, the pure-I/O tool calls are prefetched on a 2-worker pool (sized to the `core/db.py` budget of 5 connections), but every ctx mutation is applied serially in plan order by handing each prefetched result to `_execute_evidence_step` through its `executor=` parameter — storage order, log order, trace shapes, and metric counters are identical to the serial path. One semantic shift: the loop budget bounds *waiting* rather than *starting* (a step still running at the deadline is marked `evidence_loop_budget_exceeded` instead of blocking the loop unboundedly; its DB work stays bounded by the statement timeout). The env flag is a kill switch.

The loop body lives in `_run_secondary_evidence_loop` in `pipeline.py` (moved out of `evidence_planner.py` in §5.1.a, commit `ea677dc`). `evidence_planner.execute_remaining_evidence` remains as a thin delegate so tests that monkey-patch `agent.evidence_planner.execute_tool` continue to intercept tool calls — the delegate passes its local `execute_tool` binding to the helper via the `executor=` parameter.

After the loop, `merge_evidence_into_context` joins secondary frames into `ctx.df` via date-aligned joins.

#### Pass 3 — Driver-context enrichment

When primary execution produced a usable result (`ctx.used_tool` and `ctx.tool_name` set), `_enrich_prices_with_balancing_driver_context` attaches source-price and contribution columns for balancing-price answers.

#### Stage 0.9 — Evidence-anomaly detection + gated re-analysis (2026-07-07)

After `_execute_evidence_plan` returns, `_detect_evidence_anomaly` checks data-shaped, authoritative contracts for surprising primary evidence: `primary_empty` (tool succeeded, zero rows) and `period_gap` (rendered rows entirely outside the requested period, reusing `agent/render_fitness.py`). Detection is always on — `evidence_anomaly_events` counters plus a `stage_0_9_evidence_anomaly` trace size the blast radius. Behind `ENABLE_EVIDENCE_REANALYSIS` (default OFF), `_attempt_evidence_reanalysis` runs **one** retry: the SAME Stage 0.2 interpreter re-runs with the anomaly attached as a `TRUSTED_EVIDENCE_ANOMALY` block (cache-key participating), the finalize cross-check re-applies, per-attempt evidence state resets to defaults, and the plan rebuilds and re-executes. `reanalysis_attempted` guards against loops; the vector bundle from the first pass is kept (tier changes out of scope). This closes the open-loop gap without reintroducing an agent loop — one interpretation, of more information. Enablement criteria: §5.8. F3 adds a deterministic request cohort, so enabling the master flag with `ENAI_EVIDENCE_REANALYSIS_PERCENT=5` activates only the selected actors while holdbacks keep detection-only behavior.

### 3.7 Stages 1/2 — Legacy SQL Fallback

The legacy agent loop (`orchestrator.run_agent_loop`) was **deleted** in the architecture-audit P2 cleanup (2026-07). With Stage 0.2 authoritative it never fired in production, and analyzer-failure / no-tool cases fall through to the generate-plan/SQL fallback below — a more capable path than the retired keyword-driven loop. `agent/orchestrator.py` no longer exists. The `ENABLE_AGENT_LOOP` config flag and its `agent_loop_blocked_by_policy` trace field were **removed entirely** in F10 B5.A (2026-07-19) — they were inert residue after the loop deletion; no flag, branch, or trace field remains.

Stages 1/2 (`planner.generate_plan` + `sql_executor.validate_and_execute`) fire when `not ctx.used_tool` after evidence-plan execution. The condition is the §2.3 ideal's "SQL escape hatch when no typed tool produced primary data". Conceptual or `skip_sql` paths route to `summarizer.answer_conceptual` and return (terminal). The stage body lives in `_run_generate_sql_stage`, one of the `StageResult` stage functions extracted from `process_query` (audit P0-4d).

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

**Answer-provenance surface (2026-07-07).** Every `/ask` response carries `chart_metadata.answer_provenance` — a read-only projection of pipeline state built by [`agent/answer_provenance.py`](../../agent/answer_provenance.py): `answer_path` (deterministic_render / specialized_formatter / narrative_llm / conceptual / clarify / guardrail_fallback), `summary_source`, `data_source`, `tool_name`, `used_sql`, `retrieval_tier`, the analyzer's routed `query_type`/`answer_kind`/`confidence`, and the grounding-gate outcome. `guardrail_fallback` explicitly identifies answers replaced by grounding or citation safety gates; these responses carry `grounding_gate.passed=false`. Additive and backward compatible; lets the client express calibrated trust instead of presenting every answer identically.

**Coverage boundary (accepted gap, narrowed 2026-07-07):** the gate therefore protects only narrative LLM answers. Deterministic renders (generic renderer + specialized formatters) ship with **no post-hoc numeric check** — their correctness rests entirely on evidence-frame construction, which is validated at attach time (`evidence_validator`) for *shape*, not values. If a frame is built from the wrong rows, the renderer prints the wrong number confidently. Mitigation lives upstream (planner validation, tool relevance checks), not in Stage 4. Since 2026-07-07 a **shadow fitness check** ([`agent/render_fitness.py`](../../agent/render_fitness.py)) makes the failure class visible without changing behavior: violations (`empty_result_rendered`, `period_coverage_gap`, `requested_entities_missing`) emit a `stage_4_render_fitness` trace event and `render_fitness_events` counters. CLARIFY-on-violation cutover is a future, separately gated decision.

### 3.10 Stage 5 — Chart Pipeline

`chart_pipeline.build_chart` consumes the analyzer's `VisualizationInfo` directly. The chart frame source is selected first: derived chart builders produce specs for MoM/YoY, index growth, decomposition, forecast, and seasonal answers; otherwise canonical evidence frames feed `chart_frame_builder`. Multi-group plans are iterated — `chart_groups` produces one chart per group, with per-group `type`, `title`, `y_axis_label`, and `metrics` honoured.

**Design principle — intent vs realization:** Stage 0.2 emits visualization *intent* (it cannot know row counts or data shape); Stage 5 owns *realization* (the gates, the uncertainty default below, row-count checks). Do not "fix" Stage-5 gates by pushing data-dependent visualization decisions into the analyzer — it cannot make them well.

**Uncertainty default (2026-07-07).** When the analyzer leaves `primary_presentation` null on a COMPARISON answer, Stage 5 applies `chart_plus_table` via `effective_primary_presentation` ([`visualization/chart_selector.py`](../../visualization/chart_selector.py)) — the reader gets the chart *and* the exact values instead of a row-count-heuristic chart-only guess (closed the former §5.6 item). Explicit presentations, including `text`/`table`, always win; analyzer-emission observability (planning cross-check, trace serialization) still reads the raw contract.

---

## 4. Module Responsibilities & Source of Truth

Runtime modules and the files they live in. If this list disagrees with the codebase, the codebase wins.

### Orchestration

- **[`main.py`](../../main.py)** — HTTP composition root. It owns public routes and middleware and delegates startup/readiness and actor-bound conversation state through the stable runtime interfaces below.
- **[`core/application_runtime.py`](../../core/application_runtime.py)** — application lifecycle, schema reflection/warm-up, and typed readiness snapshots. It does not own request handlers or public response schemas.
- **[`core/session_runtime.py`](../../core/session_runtime.py)** — actor/authentication-mode-bound turn orchestration over the session repository, including hostile-history normalization and continuity updates.
- **[`agent/pipeline.py`](../../agent/pipeline.py)** — compatibility-preserving pipeline composition module and `process_query` entry point. It still contains the existing stage implementations and tool-execution helpers, while deadline-aware stage sequencing is delegated to `PipelineStageOrchestrator`. It must not be described as a ~230-line thin driver: the incremental F9 extraction intentionally preserved module-level patch points and public behavior rather than performing a big-bang rewrite.
- **[`agent/stage_orchestrator.py`](../../agent/stage_orchestrator.py)** — stable deadline-first context-transition interface. It checks remaining budget before each stage/effect/terminal transition and adopts the context returned by nested sub-pipelines.

### Analyzer + Routing

- **[`core/llm.py`](../../core/llm.py)** — compatibility-preserving LLM stage-entry and prompt-composition surface. Provider selection and fallback policy remain visible here, but each bounded native provider delivery attempt is delegated to `ProviderInvocationRuntime`; module-level symbols retained for tests and extension points are compatibility exports, not separate provider implementations.
- **[`core/llm_runtime.py`](../../core/llm_runtime.py)** — provider client factories, `LLMResponseCache`, token/cost accounting (Q1 extraction).
- **[`core/provider_invocation.py`](../../core/provider_invocation.py)** — one bounded, delivery-aware native provider attempt, including per-provider timeout arguments, breaker checks, attempt claiming/finalization, and ambiguous-delivery classification. It does not own provider selection or fallback ordering.
- **[`core/llm_payloads.py`](../../core/llm_payloads.py)** — JSON extraction from model output, relative-date coercion, schema-aware null-list coercion, `QuestionAnalysis` payload sanitizers (Q2 extraction; pure functions).
- **[`core/prompt_budget.py`](../../core/prompt_budget.py)** — prompt-budget enforcement + section-aware truncation engine; owns the `_TRUNCATION_PRIORITY_*` summarizer profiles selected per `answer_kind` (Q3a extraction; the `_enforce_prompt_budget` entry point remains in `core.llm` for the patch surface).
- **[`core/query_classifier.py`](../../core/query_classifier.py)** — dependency-free heuristic query-classification helpers (P0-1 extraction); `core.llm` re-exports them for backward compatibility.
- **[`agent/planner.py`](../../agent/planner.py)** — context preparation, mode selection, heuristic conceptual classification, `build_tool_invocation_from_analysis` (strategy 3 in the primary picker), and legacy `generate_plan` for SQL fallback. Structured question-contract finalization is delegated to `question_interpretation.py`.
- **[`agent/question_interpretation.py`](../../agent/question_interpretation.py)** — finalizes analyzer output and deterministic query facts into the authoritative `QuestionAnalysis` answer contract. `planner.py` retains compatibility-facing entry points where existing callers require them.
- **[`agent/router.py`](../../agent/router.py)** — `match_tool` keyword+semantic tool router (strategies 2 and 4 in the primary picker). Also `extract_granularity`, the single authority for yearly/monthly granularity detection shared by the deterministic router and its semantic fallback.
- **[`contracts/intent_lexicon.py`](../../contracts/intent_lexicon.py)** — multilingual (EN/KA/RU) intent keyword lexicon, one named constant per concept (A3). Consumers import the sets; the matching logic stays at the call sites. Migrated so far: `sql_executor`, `planner`, `models`; still to migrate (A3.d): `utils/query_validation.py`, `agent/router.py`.
- **[`agent/contract_continuity.py`](../../agent/contract_continuity.py)** — compact routed-contract snapshot for the next turn's `TRUSTED_PREVIOUS_CONTRACT` block (§3.2; stored via `utils/session_memory.set_last_contract`).
- **[`agent/fixture_candidates.py`](../../agent/fixture_candidates.py)** — `routed_fields_snapshot` (single authority, reused by the golden-set runner) + `ROUTING_FIXTURE_CANDIDATE` emission at the cross-check-disagreement and provenance-gate-failure sites (§5.3).
- **[`contracts/question_analysis.py`](../../contracts/question_analysis.py)**, **[`question_analysis_catalogs.py`](../../contracts/question_analysis_catalogs.py)** — the Stage 0.2 contract: `QuestionAnalysis` Pydantic model with `AnswerKind`, `RenderStyle`, `Grouping`, `VisualizationInfo`, `FilterCondition`, `MeasureTransform`, `VisualizationTimeGrain`, `SeriesSplitMode`, `SortRule`, `ChartFamily`, `VisualGoal`, `PresentationMode`, `SemanticRole`, `ChartIntent`. Catalogs supply the LLM-facing JSON that explains each enum.
- **[`schemas/question_analysis.schema.json`](../../schemas/question_analysis.schema.json)** — JSON-schema snapshot of the Pydantic model, asserted by `test_question_analysis_contract.py::test_schema_snapshot_matches_runtime_model`.

### Evidence Collection

- **[`agent/evidence_planner.py`](../../agent/evidence_planner.py)** — builds evidence plans and holds `merge_evidence_into_context` (date-aligned joins of secondary frames into `ctx.df`). `execute_remaining_evidence` remains a compatibility delegate so established patch points continue to work.
- **[`agent/plan_validation.py`](../../agent/plan_validation.py)** — typed validation of a proposed evidence plan against the finalized answer contract. The pipeline enforces or observes the typed result according to the configured rollout mode before any tool/DB call.
- **[`agent/evidence_validator.py`](../../agent/evidence_validator.py)** — `validate_evidence` runs inline during frame attachment; checks frames against `answer_kind` shape requirements (shared with the planner via [`agent/shape_requirements.py`](../../agent/shape_requirements.py)).
- **[`agent/evidence_finalizer.py`](../../agent/evidence_finalizer.py)** — the one frame-construction, validation, provenance-binding, and stale-frame-invalidation routine for every path that creates or mutates tabular evidence; rollout remains off/shadow/enforce as documented in §4.2.
- **[`agent/frame_adapters.py`](../../agent/frame_adapters.py)** — per-tool adapters that normalise raw tool output into `ObservationFrame` / `EntitySetFrame` / `ComparisonFrame`.
- **[`agent/tools/`](../../agent/tools/)** — typed tool implementations. (`agent/tool_adapter.py` — the agent loop's timeout/preview wrapper — was deleted with the loop; the pipeline binds `agent.tools.execute_tool` directly, bounded by the DB statement timeout.)
- **[`contracts/evidence_frames.py`](../../contracts/evidence_frames.py)** — `ObservationFrame`, `EntitySetFrame`, `ComparisonFrame` definitions.
- **[`contracts/vector_knowledge.py`](../../contracts/vector_knowledge.py)** — `VectorRetrievalTier` enum + vector-knowledge bundle types.
- **[`knowledge/vector_retrieval.py`](../../knowledge/vector_retrieval.py)** — three-tier retrieval implementation.
- **[`agent/provenance.py`](../../agent/provenance.py)** — `stamp_provenance`, `clear_provenance`, `tool_invocation_hash`, `sql_query_hash`.

### Analysis & Enrichment

- **[`agent/analyzer.py`](../../agent/analyzer.py)** — computation implementation used by Stage 3 for contract-driven enrichment.
- **[`agent/evidence_derivation.py`](../../agent/evidence_derivation.py)** — Stage 3 boundary that identifies requested derived metrics, invokes analyzer enrichment, adopts the returned context, and preserves/rebinds authoritative provenance.
- **[`agent/metric_registry.py`](../../agent/metric_registry.py)** — per-metric computation registry (MoM, YoY, CAGR, share-decomposition, correlation). `analyzer.py` dispatches to these instead of a monolithic if/elif chain.
- **[`agent/aggregation.py`](../../agent/aggregation.py)** — aggregation-intent detection + SQL guard helpers.

### Rendering

- **[`agent/summarizer.py`](../../agent/summarizer.py)** — Stage 4 dispatch and compatibility surface. It tries the generic renderer first; otherwise calls `llm_summarize_structured` with focus-aware prompts. `share_summary_override` remains a deliberate specialized formatter and conceptual answers go to `answer_conceptual`.
- **[`agent/summary_grounding.py`](../../agent/summary_grounding.py)** — numeric grounding, claim provenance, deterministic fallback policy, and the post-hoc provenance gate. `summarizer.py` imports these functions to preserve its established public/patch surface.
- **[`agent/generic_renderer.py`](../../agent/generic_renderer.py)** — deterministic rendering for SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO from evidence frames. FORECAST excluded since 2026-05-22 — routes through LLM (see §2.3 ideal-tree).
- **[`agent/chart_pipeline.py`](../../agent/chart_pipeline.py)** — Stage 5 entry. Selects chart frame source, applies `measure_transform` / `time_grain`, iterates `chart_groups`.
- **[`agent/chart_frame_builder.py`](../../agent/chart_frame_builder.py)** — builds chart-shaped frames from canonical evidence frames.
- **[`agent/derived_chart_builder.py`](../../agent/derived_chart_builder.py)** — specs for MoM/YoY, index growth, decomposition, forecast, seasonal charts.
- **[`visualization/chart_selector.py`](../../visualization/chart_selector.py)** — `should_generate_chart` gate (consults `VisualizationInfo` + row count + answer_kind).
- **[`contracts/summary.py`](../../contracts/summary.py)** — the summary-generation result contract shared by the summarizer, guardrails, and tests (P0-1 extraction from `core/llm.py`).
- **[`agent/render_fitness.py`](../../agent/render_fitness.py)** — shadow fitness checks on deterministic renders (§3.9); `period_bounds_from_hint`/`df_date_span` are also reused by Stage 0.9 anomaly detection.
- **[`agent/answer_provenance.py`](../../agent/answer_provenance.py)** — the response-facing `answer_provenance` block (§3.9); read-only, contractually un-crashable projection of `QueryContext`.

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

- **[`agent/sql_executor.py`](../../agent/sql_executor.py)** — Stage 2 SQL execution + safety checks for the legacy SQL escape hatch.
- **[`core/sql_generator.py`](../../core/sql_generator.py)** — SQL sanitization, AST-based table-whitelist validation, synonym auto-correction for the fallback SQL.
- **[`core/query_executor.py`](../../core/query_executor.py)** — pooled, read-only, timeout-guarded query execution.
- **[`core/db.py`](../../core/db.py)** — the SQLAlchemy engine, single source of truth (audit P1); imported downward by both `core.query_executor` and `knowledge.vector_store`.
- `agent/orchestrator.py` (the legacy agent loop) is **deleted** — see §3.7.

---

## 4.1 Deployment Constraint: Single Replica

The service holds request-scoped *and* cross-request state **in process memory**: rate-limit
buckets (`main.py`), session-bound conversation history and last-contract snapshots
(`utils/session_memory.py`), the LLM response cache (`core/llm_runtime.py`), and
circuit-breaker state (`utils/resilience.py`).

**The deployment assumption is exactly one worker process / one replica.** With N replicas:
rate limits multiply by N, sessions issued on one replica are unknown to the others, and
cache/breaker state diverges. A shared-store migration (Redis) was evaluated and **declined**
on 2026-06-10 (owner decision: no new runtime infrastructure — see
`medium_fix_plan_2026-06-10.md` P5). If horizontal scaling ever becomes necessary, that
decision is the one to revisit *first*; do not scale out without it.

## 4.2 Request-scoped P4 rollout

Behavior-changing P4 features are assigned once when `QueryContext` is created by
[`agent/p4_rollout.py`](../../agent/p4_rollout.py). Assignment is deterministic
and isolated per gate. The key precedence is gateway-verified actor ID, signed
server session ID, then request ID. Raw identifiers and hashes never enter
metrics. A partial rollout without a stable identifier is ineligible and fails
closed into the gate's holdback behavior.

The master controls remain authoritative and keep their safe defaults:
`ENAI_EVIDENCE_FINALIZATION_MODE=shadow`,
`ENAI_PLAN_VALIDATION_MODE=warn`,
`ENAI_ENABLE_HONEST_TERMINAL_OUTCOMES=false`, and
`ENABLE_EVIDENCE_REANALYSIS=false`. When a master control requests active
behavior, its corresponding integer percentage
(`ENAI_EVIDENCE_FINALIZATION_ENFORCE_PERCENT`,
`ENAI_PLAN_VALIDATION_ENFORCE_PERCENT`,
`ENAI_HONEST_TERMINAL_OUTCOMES_PERCENT`, or
`ENAI_EVIDENCE_REANALYSIS_PERCENT`) selects 0–100 percent of stable traffic.
Percentages default to 100 so explicitly enabling an existing master control
retains its historical all-traffic meaning. Holdbacks resolve respectively to
shadow, warn, legacy conceptual routing, and anomaly detection without retry.

`p4_rollout_events` exposes only `<gate>:<active|holdback|disabled|ineligible>`
counts. Production order and rollback evidence are defined in
[`p4_f3_canonical_pipeline_activation_runbook_2026-07-16.md`](p4_f3_canonical_pipeline_activation_runbook_2026-07-16.md).

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

**Status (2026-07):** the sibling half of audit item A4 — legacy agent-loop removal — is done (§3.7). What remains gated on production data is only this strategy 3–4 deletion.

**Gate:** Two-week production data from the F.2 hit-rate counters (`stage_0_7_entered`, `stage_0_7_invocation_built`, `stage_0_7_used_result`) showing `used_result / requests < 5%`. Until that is observed, removal is premature.

**Risk:** Small once the data clears. The change is deletion + a few related metric/trace removals.

**Files:** [`agent/pipeline.py`](../../agent/pipeline.py), [`agent/router.py`](../../agent/router.py).

### 5.3 Analyzer Misclassification — Standing Quality Area

The architecture trades regex brittleness for LLM non-determinism. Cross-check, evidence validator, and the legal-list exception are mitigations, not eliminations. Mis-routing surfaces as wrong `query_type`, wrong `answer_kind`, or a confidence flag mismatch, and the fix layer depends on which contract was actually wrong.

**Where the work lives:** analyzer prompt (few-shot examples, vocabulary), `_cross_check_answer_kind` policy, runtime skills (`answer-composer`, `energy-analyst`).

**Playbook:** [`skills/pipeline-failure-diagnostics/references/failure-taxonomy.md`](../../skills/pipeline-failure-diagnostics/references/failure-taxonomy.md) Pattern A (schema crash → cascade), Pattern B (cross-check override), Pattern C (router misclassification), Pattern D (heuristic disagreement). The 2026-05-09 fix series ("who can trade…", "why did price change…", "trend and structure of supply") is the canonical worked example: four edits across prompt + cross-check + answer-composer skill, no new stage. The 2026-05-13 log-driven series is the second worked example: five edits across post-LLM narrative scrubbing, the SCENARIO override gate, clarify-selection detection, per-stage prompt budgets, and truncation-priority invariants — again no new stage.

This is ongoing — every new failure report potentially adds a few-shot example, a cross-check refinement, or an answer-composer rule.

**Regression harness (2026-07-07):** [`evaluation/routing_golden_set.py`](../../evaluation/routing_golden_set.py) runs the live analyzer plus the finalize cross-check over [`evaluation/routing_golden_set.json`](../../evaluation/routing_golden_set.json) and scores the routed contract fields (`query_type`, `answer_kind`, `render_style`, `preferred_path`, `top_tool`). Run it after every analyzer-prompt, cross-check, or runtime-skill edit — it is the pre-deploy complement to the A5 production disagreement counters. Extend the fixture set with each fixed failure so the fix cannot silently regress. Fixture structure and enum validity are pinned by `tests/test_routing_golden_set.py`; the live run needs real LLM keys and is operator-run, not CI.

**Self-growing loop (2026-07-07):** production cross-check disagreements and provenance-gate failures emit single-line `ROUTING_FIXTURE_CANDIDATE {json}` log events ([`agent/fixture_candidates.py`](../../agent/fixture_candidates.py)); [`evaluation/harvest_fixture_candidates.py`](../../evaluation/harvest_fixture_candidates.py) converts a pasted log file into candidate fixtures with `expected` intentionally blank — a human verifies and labels before adopting a candidate into the golden set. This closes the loop: production failure → candidate fixture → labeled regression case.

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

- **Cross-tool computational queries.** Default is narrative rendering — let the LLM synthesise from pre-structured evidence. If a repeating pattern emerges, add a `derived_metric` type and let Stage 3 compute it (same shape as MoM/YoY/correlation). Principle: narrative as default, derived metric only when the pattern repeats. No structural change required.
- **Reference lines.** `include_reference_lines` was dropped because a bare bool can't carry which axis/value/label. If reference-line rendering becomes needed, introduce a concrete `ReferenceLineSpec` dataclass first.

Resolved 2026-07-07 and removed from this list: the borderline-COMPARISON item (null `primary_presentation` now defaults to `chart_plus_table` — see §3.10) and the `FilterCondition` audit (performed; clean — the contract flows `params_hint.filter` → `adapt_tool_result` → `_apply_filter` in every frame adapter, and no ad-hoc post-fetch threshold filtering exists in the summarizer/renderer layer).

### 5.7 Contract Continuity (slice 1 live, injection gated)

Storage of the previous turn's routed-contract snapshot is live (§3.2); prompt injection sits behind `ENABLE_CONTRACT_CONTINUITY` (default off). **Cutover:** (1) teach the golden-set runner to chain turns so the `c00x` continuity fixtures can execute, and get them green; (2) enable the flag in production and watch `analyzer_cross_check_events` stay at-or-below baseline for two weeks. Rollback is an env flip. The eventual slice 2 — the analyzer emitting a contract *delta* instead of a full re-interpretation — is deliberately deferred until slice 1 proves the trusted-block mechanism in production.

### 5.8 Evidence-Triggered Re-Analysis (detection live, retry gated)

Anomaly detection and counters are live (§3.6 Stage 0.9); the single retry sits behind `ENABLE_EVIDENCE_REANALYSIS` (default off) and the deterministic `ENAI_EVIDENCE_REANALYSIS_PERCENT` cohort. **Enablement:** two weeks of `evidence_anomaly_events` from production to size the blast radius (expected rare — these are the residue of routing errors), the routing golden set green with the flag on, then 0 → 5 → 25 → 100 percent in separate observation windows. Holdbacks continue detection without retry. Rollback is setting the master flag false. If anomaly rates are high, that is §5.3 analyzer-quality work first — the retry is a safety net, not a crutch.

### 5.9 Evidence-Ontology Consolidation (migration, shadow since 2026-07-07)

**Principle:** the contract should own the evidence ontology; the planner becomes purely mechanical (validate + order, never invent). Today the planner still *adds* steps from its own rule tables — domain knowledge living outside the contract, and the reason tool-routing unification is hard.

**Migration pattern (one rule per slice):** teach the analyzer to emit the evidence intent (skill guidance), keep the planner rule authoritative, and measure agreement via `evidence_rule_agreement_events` (`<rule>:agree` / `<rule>:disagree` in `/metrics`). **Cutover per rule:** ≥95% agreement over 14 days → delete the planner rule in a dedicated session, gated by the routing golden set plus disagreement review.

**Slice 1 (live):** `threshold_share_price_context` — threshold-share queries asking for price context. The analyzer prompt (question-analyzer skill, Example 2d) now teaches emitting `get_prices` as a secondary candidate; the planner rule at `_add_steps_from_rules` still adds the step and logs agreement. The `%`-only detection gap was fixed 2026-07-08 (`_SHARE_THRESHOLD_PATTERNS` now also match spelled-out "percent"/"pct"; golden-set case r014). KA/RU threshold phrasings remain fixture-driven — harvest them from production, don't guess.

### 5.10 Triage Playbook (operational, not architectural)

For triaging Q&A failures — latency spikes, grounding failures, schema validation crashes, routing misclassification — consult [`skills/pipeline-failure-diagnostics/`](../../skills/pipeline-failure-diagnostics/). The skill is the source of truth for failure patterns and fix-layer selection; this architecture document deliberately does not duplicate that content.

---

## 6. Practical Conclusion

The pipeline is contract-driven end to end. Stage 0.2 emits the answer contract; evidence frames + the generic renderer cover five standard answer shapes deterministically (SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO); FORECAST plus the narrative shapes (EXPLANATION / KNOWLEDGE) go to a focus-aware LLM summarizer with pre-computed evidence in `stats_hint`; the visualization plan flows to Stage 5 without re-interpretation.

After the 2026-05-10 F.5 + §5.1 refactor series, the entire tool-execution surface is one function `_execute_evidence_plan` called once from `process_query` — one orchestration function with three internal passes (see §2.3 note); after the P0-4 decomposition, `process_query` itself is a thin ~230-line driver over `StageResult` stage functions. The legacy agent loop is deleted (§3.7). The remaining structural work is removing Stage 0.7 strategies once F.2 hit-rate data clears.

**The practical test:** when a new question family appears, does it require a Stage 4 patch?

- **SCALAR / LIST / TIMESERIES / COMPARISON / SCENARIO** → no, the generic renderer handles it.
- **FORECAST** → no, LLM summarizer with pre-computed trendline evidence in `stats_hint` and `forecast-caveats.md` in skill guidance.
- **EXPLANATION / KNOWLEDGE / CLARIFY** → no, narrative LLM summarizer guided by skills.
- **Share-composition queries** → no, `share_summary_override` (specialized formatter) handles it.
- **Analyzer mis-routing the question entirely** → that's the live quality work in §5.3 — fixed by prompt few-shots, cross-check tuning, or runtime-skill edits, not new pipeline stages.
