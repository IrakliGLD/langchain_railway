# Query Pipeline Architecture

Current technical reference for the `langchain_railway` query pipeline, with an explicit architectural assessment and recommended redesign.

**Last updated:** 2026-04-12  
**Status:** Active — updated to reflect the full analyzer contract, canonical evidence frames, generic renderer, and evidence validation

---

## 1. Executive Summary

The pipeline is driven by a single LLM call at Stage 0.2 that emits a full answer contract. The current runtime flow is:

1. prepare context
2. run the structured LLM question analyzer — emits `answer_kind`, `render_style`, `grouping`, `entity_scope`, `candidate_tools`, `evidence_roles`, `derived_metrics`
3. cross-check `answer_kind` against `query_type`-derived value; prefer safer option on disagreement
4. conditionally retrieve vector knowledge (skip for deterministic data paths)
5. derive response mode and resolution policy inline from the contract
6. build a deterministic evidence plan from analyzer output
7. execute tool steps, normalize results into canonical evidence frames (`ObservationFrame`, `EntitySetFrame`, `ComparisonFrame`), validate evidence inline
8. run deterministic analysis enrichment (Stage 3)
9. try the generic tabular renderer first (handles SCALAR/LIST/TIMESERIES/COMPARISON from evidence frames); fall back to specialized formatters or LLM summarizer
10. chart generation

The remaining structural gap is that some downstream stages still use parallel legacy paths alongside the `answer_kind` contract: Stage 3 enrichment retains regex-based signal detection for some paths, Stage 4 has five regex-based fallback detectors behind the generic renderer, and the evidence planner does not yet read `answer_kind` to validate that evidence steps match the answer shape. See section 12 for the full gap analysis.

---

## 2. What Changed Since The Previous Architecture Doc

Changes since the previous version of this document (2026-04-05):

- Stage 0.2 now emits a full answer contract: `answer_kind`, `render_style`, `grouping`, `entity_scope` alongside existing `candidate_tools`, `params_hint`, `evidence_roles`, `derived_metrics`.
- An active cross-check compares LLM-emitted `answer_kind` against a `query_type`-derived value and prefers the safer option on disagreement.
- Vector knowledge retrieval is now conditional: skipped entirely for deterministic data paths (`answer_kind` not in {KNOWLEDGE, EXPLANATION, CLARIFY} + `render_style` = DETERMINISTIC).
- Canonical evidence frames (`ObservationFrame`, `EntitySetFrame`, `ComparisonFrame`) are constructed inline during tool execution, with provenance bound at construction time.
- A generic tabular renderer handles SCALAR, LIST, TIMESERIES, and COMPARISON from evidence frames — these answer shapes require zero per-question code in Stage 4.
- Evidence validation runs inline during frame construction, detecting correctable and uncorrectable gaps.
- Section-aware prompt truncation with data/knowledge priority orderings manages prompt budget.
- Conditional prompt blocks in the summarizer include guidance based on query focus and content signals.

Earlier changes (still accurate):

- The evidence planner is a first-class stage and is the main mechanism for multi-dataset questions.
- Stage 0.5 is often plan-driven, not just keyword-driven.
- The agent loop is a constrained fallback and only runs when there is no authoritative analyzer route.
- Stage 4 contains deterministic direct-answer branches that bypass the LLM.

---

## 3. Current Runtime Flow

### 3.1 High-Level Pipeline

```text
HTTP /ask
  -> Stage 0     prepare_context
  -> Stage 0.2   question analyzer (LLM) -> full contract: answer_kind, render_style, grouping, entity_scope
  -> answer_kind  cross-check (LLM-emitted vs query_type-derived)
  -> Stage 0.3   vector knowledge retrieval (conditional: skip for deterministic data)
  -> response_mode + resolution_policy derivation (inline from contract)
  -> Stage 0.4   evidence planner
  -> Stage 0.5/0.6 primary tool execution + evidence frame construction + evidence validation
  -> Stage 0.7   analyzer tool route fallback (legacy, pending removal)
  -> Stage 0.8   evidence loop + evidence merge
  -> Stage 1/2   legacy planner + SQL fallback (only if no usable tool path)
  -> Stage 3     analyzer enrichment
  -> Stage 4     generic renderer (first) OR specialized formatters OR LLM structured summary
  -> Stage 5     chart builder
```

### 3.2 Short-Circuit Paths

There are several early exits:

- `ResolutionPolicy.CLARIFY` -> `summarizer.answer_clarify()`
- `ResponseMode.KNOWLEDGE_PRIMARY` -> `summarizer.answer_conceptual()`
- Generic renderer succeeds (SCALAR/LIST/TIMESERIES/COMPARISON from evidence frames) -> skip LLM summarizer
- Specialized deterministic formatters (scenario, forecast) -> skip LLM summarizer

This means the pipeline is not a simple linear chain. It is a policy-driven decision graph with a deterministic-first answer strategy.

### 3.3 Current Decision Tree

This is the actual branching logic as implemented in `pipeline.py::process_query()`.

```text
HTTP /ask
│
├─ Stage 0: prepare_context
│   → detect language, select light/analyst mode, heuristic conceptual detection
│
├─ Stage 0.2: question analyzer (LLM)
│   ├─ [analyzer enabled in ACTIVE mode + succeeds]
│   │   → ctx.has_authoritative_question_analysis = True
│   │   → ctx.semantic_locked = True
│   │   → produces QuestionAnalysis:
│   │     query_type, preferred_path, candidate_tools,
│   │     evidence_roles, derived_metrics, canonical_query_en,
│   │     answer_kind, render_style, grouping, entity_scope
│   │   → fallback: if answer_kind not emitted, derive from query_type mapping
│   │   → fallback: if render_style not emitted, default to NARRATIVE
│   │
│   ├─ [analyzer enabled in SHADOW mode + succeeds]
│   │   → analysis stored for observability only
│   │   → NOT authoritative — all downstream decisions use heuristics
│   │
│   └─ [analyzer disabled or fails]
│       → no analysis; heuristic fallback for everything downstream
│
├─ answer_kind cross-check (always, even when analyzer succeeds)
│   → derive answer_kind from query_type mapping
│   → if LLM-emitted and derived disagree, prefer the safer option
│     (TIMESERIES, EXPLANATION, KNOWLEDGE are considered "safe")
│
├─ Resolve downstream query
│   ├─ [authoritative analysis] → use canonical_query_en
│   └─ [no analysis]           → use raw query
│
├─ Stage 0.3: vector knowledge retrieval (CONDITIONAL)
│   ├─ [answer_kind in {KNOWLEDGE, EXPLANATION, CLARIFY} or render_style != DETERMINISTIC]
│   │   → retrieve domain/policy passages for the resolved query
│   │   → pack for later Stage 4 prompts
│   └─ [answer_kind is data + render_style = DETERMINISTIC]
│       → SKIP retrieval entirely (deterministic data path never uses vector knowledge)
│
├─ Response Mode Derivation (inline, single source of truth, set once)
│   ├─ [has authoritative analysis]
│   │   ├─ query_type in {conceptual_definition, regulatory_procedure}
│   │   │   → KNOWLEDGE_PRIMARY
│   │   ├─ query_type in {data_retrieval, data_explanation, factual_lookup}
│   │   │   → DATA_PRIMARY
│   │   └─ query_type in {comparison, forecast, ambiguous, unsupported}
│   │       ├─ preferred_path = "knowledge" → KNOWLEDGE_PRIMARY
│   │       └─ preferred_path != "knowledge" → DATA_PRIMARY
│   │
│   └─ [no authoritative analysis]
│       ├─ heuristic is_conceptual = true  → KNOWLEDGE_PRIMARY
│       └─ heuristic is_conceptual = false → DATA_PRIMARY
│
├─ Resolution Policy Derivation
│   ├─ [has authoritative analysis + preferred_path in {CLARIFY, REJECT}]
│   │   → CLARIFY
│   └─ [otherwise]
│       → ANSWER
│
├─ Policy Short-Circuits *** EXIT POINTS ***
│   ├─ [CLARIFY]
│   │   → summarizer.answer_clarify() → RETURN
│   │
│   └─ [KNOWLEDGE_PRIMARY]
│       → summarizer.answer_conceptual() → RETURN
│
│   *** Only DATA_PRIMARY + ANSWER continues past this point ***
│
├─ Stage 0.4: evidence planner
│   ├─ [enabled + has authoritative analysis]
│   │   → expand analysis into ordered evidence steps
│   │   → each step: {tool_name, params, role, satisfied: false}
│   │   → roles: PRIMARY_DATA, COMPOSITION_CONTEXT, TARIFF_CONTEXT, CORRELATION_DRIVER
│   │   → NOTE: does not yet read answer_kind to validate steps match answer shape
│   │
│   └─ [disabled or no analysis]
│       → no plan; rely on keyword/analyzer routing below
│
├─ Stage 0.5: primary tool routing *** THREE-WAY BRANCH ***
│   │
│   ├─ [evidence plan exists + has unsatisfied step]
│   │   → plan-driven: use first unsatisfied step as ToolInvocation
│   │   → confidence = 0.85, reason = "evidence_plan:{role}"
│   │
│   ├─ [no plan + has authoritative analysis]
│   │   → skip keyword router entirely (invocation = None)
│   │   → fall through to Stage 0.7
│   │
│   └─ [no plan + no authoritative analysis]
│       → keyword router match_tool():
│         4-tier ladder: tariffs(0.92) → composition(0.94) → generation(0.85) → prices(0.82)
│         optional semantic fallback (token similarity scoring)
│
├─ Stage 0.6: tool execution + evidence frame construction (if Stage 0.5 produced an invocation)
│   ├─ [execute + relevance validated]
│   │   → normalize tool result into canonical evidence frame:
│   │     ├─ ObservationFrame  {period, entity_id, entity_label, metric, value, unit}
│   │     ├─ EntitySetFrame    {entity_id, entity_label, membership_reason}
│   │     └─ ComparisonFrame   {subject, baseline, value, delta}
│   │   → validate evidence frame against answer_kind requirements
│   │   → bind provenance refs at construction time
│   │   → store result on ctx, stamp provenance
│   │   → if evidence plan: mark matching step satisfied
│   │
│   ├─ [execute + relevance BLOCKED]
│   │   → clear result, ctx.used_tool = false
│   │   → fall through to Stage 0.7
│   │
│   └─ [execution ERROR]
│       → mark plan step failed, ctx.used_tool = false
│       → fall through to Stage 0.7
│
├─ Stage 0.7: analyzer tool route fallback (only if Stage 0.5/0.6 produced no result)
│   │ NOTE: pending removal — candidate_tools[0] + evidence planner should make this unnecessary
│   │
│   ├─ [has authoritative analysis + hints enabled]
│   │   → build ToolInvocation from analyzer candidates
│   │   ├─ [built + executed + relevant] → store result + build evidence frame
│   │   ├─ [built + executed + failed + plan has unsatisfied steps]
│   │   │   → mark step failed, let Stage 0.8 handle
│   │   ├─ [built + executed + failed + no plan steps]
│   │   │   → attempt limited recovery (composition→prices swap, resolved-query re-route)
│   │   └─ [build failed]
│   │       → attempt recovery or log miss
│   │
│   └─ [no authoritative analysis or hints disabled]
│       → log router miss
│
├─ Stage 0.8: evidence loop (if plan has unsatisfied steps)
│   → execute remaining plan steps
│   → normalize each result into canonical evidence frames
│   → store evidence by role in ctx.evidence_collected
│   → merge secondary datasets into primary frame by date
│   → record join provenance
│
├─ Stage 1/2: legacy fallback (only if NO tool succeeded AND no satisfied plan)
│   │ Additional gate: does NOT run if authoritative analysis is active
│   │
│   ├─ [agent loop enabled + no authoritative analysis]
│   │   → orchestrator.run_agent_loop()
│   │   ├─ conceptual_exit → RETURN
│   │   ├─ data_exit + relevant → continue
│   │   └─ fallback_exit → fall through to SQL
│   │
│   └─ [generate plan + SQL]
│       ├─ plan says conceptual or skip_sql → answer_conceptual() → RETURN
│       ├─ SQL execute blocked → answer_conceptual() → RETURN
│       └─ SQL execute succeeds → continue
│
├─ Stage 3: analyzer enrichment
│   → share resolution + summary construction + grounding hints
│   → scenario evidence dispatch (partially uses answer_kind, partially uses regex)
│   → forecast/CAGR, correlation, "why" causal reasoning
│   → trendline pre-calculation, MoM/YoY, seasonal signals
│   → build structured analysis_evidence records
│   → outputs written as canonical evidence frames
│
├─ Post-Stage 3: evidence readiness check
│   ├─ [missing requested evidence + no evidence at all]
│   │   → CLARIFY → RETURN
│   ├─ [missing requested evidence + partial evidence]
│   │   → continue with warning
│   └─ [all evidence present]
│       → continue
│
├─ Stage 4: answer dispatch *** GENERIC RENDERER FIRST, then fallback ladder ***
│   │
│   ├─ [generic renderer succeeds] *** NEW: first-choice path ***
│   │   → switch(answer_kind) over canonical evidence frames:
│   │     SCALAR → extract value + unit + period from ObservationFrame
│   │     LIST → enumerate entities from EntitySetFrame
│   │     TIMESERIES → format period-indexed rows from ObservationFrame
│   │     COMPARISON → format subject vs baseline + delta from ComparisonFrame
│   │   → provenance refs already bound from evidence collection
│   │   → SKIP LLM summarizer entirely (deterministic, confidence 1.0)
│   │
│   ├─ [generic renderer returns None → fallback to legacy regex ladder]
│   │
│   ├─ [ctx.share_summary_override populated]
│   │   → pass-through Stage 3's deterministic share answer (confidence 1.0)
│   │
│   ├─ [scenario eligible: answer_kind=SCENARIO or regex-based detection]
│   │   → deterministic scenario formatter (confidence 0.95)
│   │
│   ├─ [regulated tariff list signal: regex-based detection]
│   │   → deterministic tariff list formatter (confidence 0.98)
│   │
│   ├─ [residual weighted-price signal: regex-based detection]
│   │   → deterministic residual calculation formatter (confidence 0.95)
│   │
│   ├─ [trendline forecast eligible: answer_kind=FORECAST or regex-based detection]
│   │   → deterministic forecast formatter (confidence 0.95)
│   │
│   └─ [none of the above]
│       → LLM llm_summarize_structured()
│       ├─ [grounding passes] → final answer
│       └─ [grounding fails]
│           ├─ scenario fallback available → deterministic scenario answer
│           └─ no fallback → generic grounding-failure message
│
└─ Stage 5: chart builder
    → infer chart type, limit series, construct payload
    → RETURN
```

### 3.4 Ideal Decision Tree

**Design principle: Stage 0.2 is the one LLM call. Make it emit the full contract. Everything after just executes — no re-interpretation.**

The current pipeline has 13 steps because downstream stages constantly re-interpret what the analyzer already understood. The ideal pipeline is shorter: 0.2 produces a firm plan, and every subsequent stage is pure execution.

**Current → Ideal step reduction:**

```text
Current (13 steps):
  0 → 0.2 → 0.3 → mode deriv → 0.4 → 0.5 → 0.6 → 0.7 → 0.8 → 1/2 → 3 → 4 → prov → 5

Ideal (8 steps):
  0 → 0.2 → 0.3 → 0.4 → tool exec → evidence collect → 3 → 4 → 5
```

**What was eliminated or merged:**

| Removed step | Why it existed | How it's absorbed |
|---|---|---|
| Mode derivation (separate step) | Translates analyzer output into response_mode | 0.2 emits `answer_kind` + `render_style` directly; mode is trivially derived inline |
| Stage 0.2b AnswerSpec construction | Builds typed contract from analyzer | Unnecessary — 0.2 emits the contract itself |
| Stage 0.7 analyzer fallback | 0.5 skips router when analyzer active but doesn't use analyzer to route | `candidate_tools[0]` + evidence planner selects tool deterministically; no fallback needed |
| Stage 0.8b evidence validation | Validates evidence against answer contract | Merged into evidence collection — validate as you collect, not as a separate pass |
| Stage 0.8c canonical framing | Normalizes DataFrames into evidence frames | Merged into evidence collection — frame as you collect |
| Stage 1/2 legacy SQL | Fallback when no tool matches | Rare escape hatch inside tool execution failure path, not a separate pipeline stage |
| Provenance gate (separate step) | Post-hoc numeric grounding check | Provenance bound at construction time in Stage 4; separate gate is redundant |

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
│   │ Nothing downstream re-parses the query text.
│   │
│   │ Current QuestionAnalysis fields (kept):
│   │   query_type, preferred_path, candidate_tools, params_hint,
│   │   evidence_roles, derived_metrics, canonical_query_en, etc.
│   │
│   │ New fields added to QuestionAnalysis:
│   │
│   │   answer_kind: scalar | list | timeseries | comparison |
│   │                explanation | forecast | scenario | knowledge | clarify
│   │
│   │   render_style: deterministic | narrative
│   │     (LLM decides: "is this a data lookup or does the user want explanation?")
│   │
│   │   grouping: none | by_entity | by_period | by_metric
│   │
│   │   entity_scope: e.g., "regulated_plants", "all", specific entity names
│   │
│   │   NOTE: primary_tool is NOT emitted by the LLM. Tool selection remains
│   │   deterministic: candidate_tools[0] + evidence planner. Only 4 tools exist;
│   │   the mapping is trivial and should not be an LLM decision.
│   │
│   │ Why the LLM should emit answer shape, not downstream code:
│   │   - "which plants are regulated?" → LLM says LIST + entity_scope=regulated
│   │     (today: query_type=data_retrieval, Stage 4 regex detects "which" + "regulated")
│   │   - "how did price change Jan vs Feb?" → LLM says COMPARISON
│   │     (today: query_type=data_retrieval, regex detects "vs" in 3 places)
│   │   - "show monthly prices for 2025" → LLM says TIMESERIES + grouping=by_period
│   │     (today: query_type=data_retrieval, no way to distinguish from SCALAR until Stage 4)
│   │   - "explain why prices rose" → LLM says render_style=narrative + answer_kind=explanation
│   │     (today: _EXPLANATION_ROUTING_SIGNALS regex checked in 3 different places)
│   │
│   │ Fallback when analyzer is disabled/fails:
│   │   → derive answer_kind from query_type mapping + keyword heuristics
│   │   → derive tool from keyword router (existing match_tool logic)
│   │   → render_style defaults to narrative (safer)
│   │
│   │ Active cross-check (always, even when analyzer succeeds):
│   │   → also derive answer_kind from query_type mapping
│   │   → if LLM-emitted and derived answer_kind disagree, log warning
│   │   → prefer the safer option (see Section 8.7)
│   │
│   │ After this point, response_mode and resolution_policy are derived
│   │ trivially inline (no separate stage):
│   │   answer_kind in {knowledge, clarify} → short-circuit RETURN
│   │   render_style = narrative + answer_kind = explanation → KNOWLEDGE check
│   │   everything else → continue to tool execution
│   │
│   → The analyzer's output is the contract. Downstream stages trust it.
│
├─ Stage 0.3: vector knowledge retrieval (conditional)
│   ├─ [answer_kind in {knowledge, explanation}] → full retrieval
│   ├─ [answer_kind is data + render_style = narrative] → light retrieval
│   └─ [answer_kind is data + render_style = deterministic] → skip
│
├─ Stage 0.4: evidence planner
│   │ Reads the analyzer contract to build evidence steps.
│   │ The contract tells the planner what it needs to satisfy:
│   │
│   │   answer_kind = LIST → planner ensures entity-enumeration step
│   │   answer_kind = COMPARISON → planner ensures two periods or two entities
│   │   answer_kind = TIMESERIES → planner ensures period range
│   │   answer_kind = SCENARIO → planner ensures scenario params available
│   │
│   │ Tool is selected deterministically from candidate_tools[0] — no keyword routing needed.
│   │ Evidence roles are known from 0.2 — secondary datasets planned directly.
│   │
│   → Ordered list of tool steps, validated against the answer contract
│
├─ Tool execution + evidence collection (merges current 0.5/0.6/0.7/0.8)
│   │
│   │ Tool selection is deterministic:
│   │   candidate_tools[0] from analyzer + evidence planner expansion.
│   │   No LLM decision needed — only 4 tools exist.
│   │
│   │ Simple execution loop — no routing decisions, just run the plan:
│   │
│   │ for each evidence step:
│   │   1. execute tool (params from analyzer / planner)
│   │   2. validate relevance
│   │   3. normalize output into canonical evidence frame:
│   │      ├─ ObservationFrame {period, entity_id, entity_label, metric, value, unit}
│   │      ├─ EntitySetFrame   {entity_id, entity_label, membership_reason}
│   │      └─ ComparisonFrame  {subject, baseline, value, delta}
│   │      (adapters contain domain-specific column mapping — see note in 8.3)
│   │   4. validate frame against answer_kind requirements
│   │   5. store evidence by role, bind provenance refs
│   │
│   │ On tool failure:
│   │   ├─ [other plan steps remain] → continue, mark step failed
│   │   ├─ [no plan steps + SQL fallback possible] → try SQL escape hatch
│   │   └─ [nothing works] → downgrade to CLARIFY
│   │
│   │ Stage 0.7 (analyzer fallback) is eliminated:
│   │   candidate_tools[0] + evidence planner already selects the tool.
│   │
│   │ Legacy SQL (Stage 1/2) is folded in as a failure-path escape hatch,
│   │   not a separate pipeline stage.
│   │
│   → Evidence frames collected, validated, provenance-bound
│
├─ [CHANGED] Stage 3: analyzer enrichment — switches on answer_kind
│   │
│   │ Same computation as today, but dispatch uses answer_kind instead of
│   │ its own signal detection (share-intent regex, scenario-eligibility
│   │ checks, forecast-mode keywords, why-mode detection):
│   │
│   │   answer_kind = SCALAR/TIMESERIES + subject_domain = shares → share enrichment
│   │   answer_kind = SCENARIO → scenario evidence dispatch
│   │   answer_kind = FORECAST → trendline pre-calculation
│   │   answer_kind = EXPLANATION → "why" causal reasoning + correlation
│   │   answer_kind = COMPARISON → MoM/YoY derived metrics
│   │
│   │ Outputs written as canonical evidence frames.
│   │ Share-summary grounding pattern generalized to all paths.
│
├─ Stage 4: answer rendering *** SINGLE SWITCH on answer_kind ***
│   │
│   │ No regex. No query-signal detection. No re-interpretation.
│   │ Just: switch(answer_kind) with evidence frames as input.
│   │
│   ├─ [render_style = DETERMINISTIC]
│   │   │
│   │   │ Generic tabular renderer (one function, handles most cases):
│   │   │
│   │   ├─ SCALAR   → extract value + unit + period from ObservationFrame
│   │   ├─ LIST     → enumerate entities from EntitySetFrame, group by reason
│   │   ├─ TIMESERIES → format period-indexed rows from ObservationFrame
│   │   ├─ COMPARISON → format subject vs baseline + delta from ComparisonFrame
│   │   │
│   │   │ Specialized formatters (only for domain-specific decomposition):
│   │   ├─ SCENARIO → payoff breakdown (positive/negative, market vs combined)
│   │   ├─ FORECAST → trendline + R² caveat + seasonal + assumptions
│   │   │
│   │   │ Provenance refs already bound from evidence collection.
│   │   │ No separate provenance gate needed.
│   │   │
│   │   → New question family that is SCALAR/LIST/TIMESERIES/COMPARISON:
│   │     works automatically. Zero Stage 4 code. Zero regex.
│   │
│   └─ [render_style = NARRATIVE]
│       → LLM receives pre-structured evidence frames
│       → focused on explanation quality, not schema recovery
│       → provenance refs pre-bound
│
└─ Stage 5: chart builder (unchanged)
```

**Step count comparison:**

| | Before (doc v1) | Current | Ideal (target) |
|---|---|---|---|
| Total steps | 13 | ~10 (evidence frames + generic renderer inline, but legacy stages not yet removed) | 8 |
| Steps that interpret query semantics | 5+ (analyzer, mode deriv, router, evidence planner, Stage 4 regex) | 2 (analyzer + legacy regex fallbacks in Stage 3/4) | 1 (analyzer only) |
| Separate fallback/recovery stages | 3 (Stage 0.7, Stage 1/2, provenance gate) | 3 (still present, pending removal) | 0 (folded into execution loop) |
| LLM calls for data questions | 1 (analyzer) + usually 1 (Stage 4 summarizer) | 1 (analyzer) + 0 when generic renderer succeeds | 1 (analyzer) + only for narrative render_style |

**Key differences summary — current vs ideal:**

| Decision Point | Current (implemented) | Ideal (remaining gap) |
|---|---|---|
| `answer_kind` source | LLM emits in QuestionAnalysis; cross-checked against `query_type` derivation | Done ✓ |
| Tool selection | `candidate_tools[0]` + evidence planner; keyword router as fallback | Remove keyword router fallback (Stage 0.7) |
| Tool routing mechanism | 3-way branch still exists (plan / analyzer-skip / keyword router) | Collapse to direct execution from evidence plan |
| Evidence framing | Canonical frames built during collection via `frame_adapters.py` | Done ✓ |
| Evidence validation | Validated during collection against `answer_kind` via `evidence_validator.py` | Done ✓ — but evidence planner does not yet pre-validate steps against `answer_kind` |
| Stage 4 dispatch | Generic renderer first (answer_kind switch); 5 regex detectors as fallback | Remove regex fallback detectors; extend generic renderer to SCENARIO/FORECAST |
| Stage 4 rendering | Generic renderer handles SCALAR/LIST/TIMESERIES/COMPARISON | Extend to SCENARIO/FORECAST |
| Grounding | Provenance bound at construction time; post-hoc gate still runs on legacy paths | Remove redundant post-hoc gate |
| Vector knowledge | Conditional: skip for deterministic data paths | Add light retrieval tier for narrative data |
| New question family requires | Usually nothing for SCALAR/LIST/TIMESERIES/COMPARISON | Extend to SCENARIO/FORECAST so all standard shapes need nothing |

---

## 4. Current Stage Semantics

### 4.1 Stage 0: Prepare Context

Owned by `agent/planner.py`.

Responsibilities:

- detect language
- select `light` vs `analyst` mode
- run heuristic conceptual detection for fallback compatibility
- initialize query context

This stage is intentionally cheap and does not decide the final answer path by itself.

### 4.2 Stage 0.2: Structured Question Analyzer

Owned by `core/llm.py::llm_analyze_question()` and consumed in `agent/pipeline.py`.

Responsibilities:

- normalize the question into a strict `QuestionAnalysis` contract
- choose `query_type` and `preferred_path`
- emit the full answer contract: `answer_kind`, `render_style`, `grouping`, `entity_scope`
- propose `candidate_tools` with scores and `params_hint`
- emit `analysis_requirements` including `derived_metrics`
- mark `needs_multi_tool` and `evidence_roles` for multi-dataset questions
- provide `canonical_query_en`

This is the semantic center of the pipeline. When active, it is the single authoritative source for:

- answer shape (`answer_kind` + `render_style` + `grouping`)
- response mode and resolution policy (derived inline from the contract)
- tool selection (`candidate_tools[0]` + evidence planner)
- requested derived metrics and evidence roles

An active cross-check (`_cross_check_answer_kind` in `agent/pipeline.py:235`) compares the LLM-emitted `answer_kind` against a deterministic `query_type`-derived value and prefers the safer option on disagreement. If the LLM does not emit `answer_kind`, it is derived from the `query_type` mapping. If `render_style` is missing, it defaults to NARRATIVE (safer).

### 4.3 Stage 0.3: Vector Knowledge Retrieval (Conditional)

Owned by `knowledge/vector_retrieval.py`. Gated by `agent/pipeline.py:1083-1098`.

Responsibilities:

- retrieve policy/regulation/domain passages for the resolved query
- pack passages for Stage 4 prompts
- run in active or shadow mode

Conditional execution based on the analyzer contract:

- `answer_kind` in {KNOWLEDGE, EXPLANATION, CLARIFY} or `render_style` != DETERMINISTIC → **run retrieval**
- `answer_kind` is data + `render_style` = DETERMINISTIC → **skip entirely**

This avoids wasting latency and prompt budget on deterministic data paths that never use vector knowledge. A light retrieval tier (top-K=2, no re-rank) for narrative data questions is planned but not yet implemented (see section 12.1 Gap 4).

### 4.4 Response Mode Derivation (Inline)

Owned by `agent/pipeline.py` (`_derive_response_mode` at line 147, `_derive_resolution_policy` at line 188).

This is not a separate pipeline stage — it is an inline derivation from the analyzer contract:

- `conceptual_definition` and `regulatory_procedure` → `KNOWLEDGE_PRIMARY`
- `data_retrieval`, `data_explanation`, `factual_lookup` → `DATA_PRIMARY`
- `comparison`, `forecast`, `ambiguous`, `unsupported` → use `preferred_path` as tie-breaker
- `preferred_path` in {CLARIFY, REJECT} → `ResolutionPolicy.CLARIFY`

Legacy flag `ctx.is_conceptual` is kept in sync for backward compatibility but no stage should re-derive it independently.

### 4.5 Stage 0.4: Evidence Planner

Owned by `agent/evidence_planner.py`.

Responsibilities:

- expand `QuestionAnalysis` into ordered evidence steps
- choose the primary dataset from `candidate_tools`
- add secondary evidence roles: `composition_context`, `tariff_context`, `correlation_driver`
- keep time windows aligned across evidence sources

The planner reads `candidate_tools` and `evidence_roles` from the analyzer contract. It does NOT yet read `answer_kind` or `render_style` to validate that evidence steps satisfy the answer shape — this is a known gap (see section 12.1 Gap 3).

### 4.6 Stage 0.5 / 0.6: Primary Tool Execution + Evidence Frame Construction

Owned by `agent/pipeline.py`, `agent/router.py`, `agent/frame_adapters.py`, and typed tools in `agent/tools/`.

Current behavior:

- when an evidence plan exists, Stage 0.5 prefers the first unsatisfied plan step
- raw-query keyword routing is fallback behavior when no authoritative analyzer is available
- Stage 0.6 executes the tool, validates relevance, and stamps provenance
- on successful execution, `_build_and_attach_evidence_frame()` (`pipeline.py:827`) normalizes the tool result into a canonical evidence frame (`ObservationFrame`, `EntitySetFrame`, or `ComparisonFrame`) using per-tool adapters in `agent/frame_adapters.py`
- `validate_evidence()` (`agent/evidence_validator.py`, called at `pipeline.py:869`) checks the frame against `answer_kind` requirements inline
- provenance refs are bound at frame construction time

The keyword router is no longer the main semantic entry point in analyzer-enabled mode.

### 4.7 Stage 0.7: Analyzer Tool Route Fallback (Legacy, Pending Removal)

Responsibilities:

- if Stage 0.5 does not yield an invocation, convert analyzer candidates into a concrete tool call
- execute that tool, build evidence frame, validate relevance
- optionally fall through to limited recovery logic

This is a constrained second-chance routing step. With `candidate_tools[0]` + evidence planner handling tool selection, Stage 0.7 should be unnecessary. It is pending removal once its hit rate is confirmed to be negligible (see section 12.2).

### 4.8 Stage 0.8: Evidence Loop And Merge

Owned by `agent/evidence_planner.py`.

Responsibilities:

- execute remaining unsatisfied evidence-plan steps
- normalize each result into canonical evidence frames
- store evidence by role in `ctx.evidence_collected`
- merge secondary datasets into the primary frame by date
- record join provenance
- optionally restore the primary dataset from collected evidence

This is the stage that makes multi-tool reasoning concrete.

### 4.9 Stage 1 / 2: Legacy Planner And SQL Fallback

Responsibilities:

- generate plan + SQL only when deterministic tool paths are exhausted
- validate SQL
- execute SQL under safety constraints

This path still matters, but it is now clearly a fallback path, not the dominant architecture.

### 4.10 Stage 3: Analyzer Enrichment

Owned by `agent/analyzer.py`.

Responsibilities:

- compute deterministic derived metrics
- compute comparisons, trends, correlations, seasonal signals
- build analysis evidence (outputs written as canonical evidence frames)
- emit contradiction guards and overrides
- dispatch partially uses `answer_kind` (scenario at `summarizer.py:1177`, forecast at `summarizer.py:1536`) but retains parallel legacy regex paths for share intent, correlation context, and forecast mode

Remaining gap: not yet refactored to a single `switch(answer_kind)` dispatch (see section 12.1 Gap 1).

### 4.11 Stage 4: Generic Renderer Plus Specialized Formatters Plus LLM Summarization

Owned by `agent/summarizer.py` and `agent/generic_renderer.py`.

Current behavior (first match wins):

1. **Generic tabular renderer** (`_try_generic_renderer` at `summarizer.py:1878`) — tries first. Switches on `answer_kind` over canonical evidence frames: SCALAR, LIST, TIMESERIES, COMPARISON. If it succeeds, the LLM summarizer is skipped entirely.
2. **Legacy regex-based fallback ladder** (if generic renderer returns None):
   - share summary pass-through
   - scenario formatter (partially uses `answer_kind`, partially regex)
   - regulated tariff list formatter (regex)
   - residual weighted-price formatter (regex)
   - trendline forecast formatter (partially uses `answer_kind`, partially regex)
3. **LLM `llm_summarize_structured()`** — receives conditional prompt blocks based on query focus and content signals
4. **Grounding check** on LLM output

The generic renderer is the primary deterministic path and handles most standard answer shapes. The five regex-based detectors are legacy fallbacks pending absorption into the generic renderer (see section 12.1 Gap 2).

### 4.12 Stage 5: Chart Pipeline

Responsibilities:

- infer chart type
- limit series
- decide skip conditions
- construct chart payload

This stage remains mostly deterministic and isolated from the main architecture issues.

---

## 5. Current Module Responsibilities

### `agent/pipeline.py`

Main orchestrator. Owns stage ordering, policy decisions, fallback order, answer_kind cross-check, evidence frame construction (`_build_and_attach_evidence_frame`), conditional vector retrieval gating, and integration logic between planner, tools, analyzer, summarizer, and charting.

### `agent/router.py`

Deterministic extraction and fallback routing:

- query keyword heuristics
- semantic fallback scoring
- date, currency, metric, and entity extraction

Acts partly as parameter extractor and partly as fallback route selector (Stage 0.7). Route selection role is diminishing as `candidate_tools[0]` + evidence planner handles most cases.

### `agent/evidence_planner.py`

Architectural backbone for multi-evidence questions:

- expands analyzer `candidate_tools` and `evidence_roles` into tool steps
- resolves aligned parameters
- executes remaining evidence steps
- merges evidence into the main frame

Does not yet read `answer_kind` to validate steps match the answer shape.

### `agent/analyzer.py`

Deterministic enrichment layer (Stage 3). Produces derived metrics and contextual evidence. Dispatch partially uses `answer_kind`, partially uses legacy regex detection.

### `agent/generic_renderer.py`

Generic tabular answer renderer. Switches on `answer_kind` over canonical evidence frames:

- SCALAR → single value + unit + period from `ObservationFrame`
- LIST → entity enumeration from `EntitySetFrame`
- TIMESERIES → period-indexed table from `ObservationFrame`
- COMPARISON → subject vs baseline + delta from `ComparisonFrame`

Called first in Stage 4. When it succeeds, the LLM summarizer is skipped.

### `agent/frame_adapters.py`

Per-tool evidence frame adapters. Contains `adapt_tool_result()` which normalizes raw tool DataFrames into canonical evidence frames with domain-specific column mapping. Written once per tool, stable.

### `agent/evidence_validator.py`

Evidence validation. Contains `validate_evidence()` which checks whether evidence frames satisfy `answer_kind` requirements. Called inline during frame construction at `pipeline.py:869`.

### `agent/summarizer.py`

Mixed responsibility module:

- conceptual answer generation
- clarify answer generation
- legacy regex-based deterministic direct answers (scenario, tariff list, residual weighted-price, forecast)
- LLM structured summarization with conditional prompt blocks
- grounding checks

The generic renderer has absorbed most standard deterministic answers. The remaining regex-based formatters are pending migration.

### `core/llm.py`

LLM access layer:

- question analyzer prompt assembly and budget enforcement
- structured summarizer prompt with conditional skill-based guidance blocks
- section-aware prompt truncation with data/knowledge priority orderings
- domain knowledge retrieval and prompt selection
- model selection, resilience, caching, timeout retry with reduced budget

### `contracts/question_analysis.py`

Defines the `QuestionAnalysis` pydantic model including `AnswerKind`, `RenderStyle`, `Grouping` enums, `ToolCandidate`, `ToolingInfo`, `RoutingInfo`, and the full analyzer contract schema.

### `contracts/question_analysis_catalogs.py`

Static catalogs injected into the analyzer prompt: query type guide, answer kind guide, filter guide, topic catalog, tool catalog (with `combined_with` rules), derived metric catalog, chart policy.

### `contracts/evidence_frames.py`

Defines `ObservationFrame`, `EntitySetFrame`, `ComparisonFrame` dataclasses used by frame adapters and the generic renderer.

---

## 6. Architectural Assessment

The system has made substantial progress toward a contract-driven architecture. The analyzer emits a full answer contract, canonical evidence frames normalize tool output, and the generic renderer handles most standard shapes. The remaining issues are concentrated in legacy parallel paths that have not yet been removed.

### 6.1 Query Semantics Are Still Re-Implemented In Some Places — PARTIALLY RESOLVED

The analyzer contract (`answer_kind`, `render_style`, `grouping`, `entity_scope`) is now the authoritative semantic source. Response mode and resolution policy derive from it inline.

Remaining duplication:

- Stage 3 enrichment still uses parallel regex signal detection alongside `answer_kind` for some paths
- Stage 4 has five regex-based fallback detectors behind the generic renderer
- Explanation routing (`pipeline.py:711`) checks `answer_kind` first but falls through to keyword detection
- Evidence planner reads `candidate_tools` and `evidence_roles` but does not read `answer_kind`

Impact: new question families that fit standard shapes (SCALAR/LIST/TIMESERIES/COMPARISON) need zero per-question code. But SCENARIO and FORECAST still require regex-based detection, and Stage 3 enrichment for correlation and share intent still uses legacy paths.

### 6.2 Tool Outputs Are Now Answer-Shaped Via Evidence Frames — RESOLVED

Canonical evidence frames (`ObservationFrame`, `EntitySetFrame`, `ComparisonFrame`) normalize raw tool output during collection. Per-tool frame adapters in `agent/frame_adapters.py` handle domain-specific column mapping. The generic renderer operates on these stable frame types, not on raw tool-shaped DataFrames.

Remaining gap: not all tool execution paths go through frame construction yet (some legacy paths in Stage 0.7 and Stage 0.8 may bypass framing).

### 6.3 Stage 4 Business Logic Is Reducing — PARTIALLY RESOLVED

The generic renderer has absorbed SCALAR, LIST, TIMESERIES, and COMPARISON deterministic answers. When it succeeds, Stage 4 is a clean `answer_kind` switch with zero regex.

Remaining: five regex-based fallback detectors (scenario, tariff list, residual weighted-price, forecast, share summary) still live in `agent/summarizer.py`. These will be absorbed when the generic renderer is extended to handle SCENARIO and FORECAST.

### 6.4 Grounding Is Moving Toward Construction-Time — PARTIALLY RESOLVED

Provenance refs are now bound at evidence frame construction time (`pipeline.py:848-851`). Evidence validation runs inline during collection (`evidence_validator.py`).

Remaining: the separate post-hoc provenance gate still runs after Stage 4. It is redundant for paths that go through canonical frames but still needed for legacy paths.

### 6.5 Evidence Planning Reads The Contract But Not The Answer Shape — PARTIALLY RESOLVED

The evidence planner reads `candidate_tools` and `evidence_roles` from the analyzer contract. Evidence validation checks frames against `answer_kind` inline.

Remaining gap: the planner itself does not read `answer_kind` to validate that planned steps will produce evidence matching the answer shape. For example, it does not ensure LIST questions get entity-enumeration steps or COMPARISON questions get two-period evidence. This means some evidence gaps are caught late by the validator rather than prevented by the planner.

### 6.6 LLM Efficiency Has Improved But Prompt Budget Needs Work — PARTIALLY RESOLVED

Improvements:

- The analyzer emits the full answer contract, eliminating downstream re-interpretation for standard shapes
- Vector knowledge skips for deterministic data paths (no wasted retrieval or prompt space)
- The generic renderer skips the LLM summarizer entirely for SCALAR/LIST/TIMESERIES/COMPARISON
- Conditional prompt blocks in the summarizer include guidance based on query focus

Remaining inefficiencies (see section 13 for detailed analysis):

- The analyzer prompt includes all catalog blocks unconditionally (~3,000-5,000 chars even when irrelevant)
- Prompt truncation is response_mode-aware but not question-type-aware
- Domain knowledge loaded unconditionally for energy-domain focuses even on deterministic data paths
- No "fast" vs "deep" mode for users who want quick answers vs thorough analysis

---

## 7. Why Individual Question Fixes Keep Happening

The frequency of per-question fixes has decreased significantly since the analyzer contract, evidence frames, and generic renderer were implemented. SCALAR/LIST/TIMESERIES/COMPARISON questions that fit canonical evidence frames now work with zero per-question code.

However, fixes still occur in two areas:

1. **SCENARIO and FORECAST questions** — the generic renderer does not handle these shapes yet, so they still go through regex-based eligibility detectors and specialized formatters. New phrasings that don't match the regex require patches.
2. **Stage 3 enrichment routing** — correlation, share intent, and forecast enrichment still use parallel legacy regex detection alongside `answer_kind`, meaning new analytical patterns may need regex additions.

### 7.1 Per-Question Fix Taxonomy — Updated Status

| Fix category | Example | Root cause | Status |
|---|---|---|---|
| **Stage 4 regex miss** | New phrasing for "list regulated plants" doesn't match regex | Answer shape detected by keyword instead of contract | **RESOLVED for SCALAR/LIST/TIMESERIES/COMPARISON** — generic renderer uses `answer_kind`. Still affects SCENARIO/FORECAST. |
| **Missing deterministic formatter** | New answer shape needs a bespoke builder | No generic renderer for the answer_kind | **RESOLVED for standard shapes** — generic renderer handles them. SCENARIO/FORECAST still need specialized formatters. |
| **Tool returns wrong shape** | Tool returns tariff-group evidence, not entity list | Tool output is tool-shaped | **RESOLVED** — canonical evidence frames + per-tool adapters normalize output. |
| **Evidence planner miss** | Comparison needs two datasets but planner fetches one | Evidence expansion rules don't cover this | **OPEN** — planner does not read `answer_kind` to validate steps match shape. |
| **Analyzer misclassification** | Question classified with wrong `answer_kind` | LLM misclassification | **MITIGATED** — active cross-check prefers safer option. Residual risk remains. |
| **Grounding failure on valid data** | Correct answer fails post-hoc token check | Provenance binding too late | **MOSTLY RESOLVED** — construction-time binding in evidence frames. Post-hoc gate still runs on legacy paths. |

### 7.2 The Remaining Generalization Gap

The two structural generalizations previously missing — `answer_kind` in the analyzer contract and a generic tabular renderer — are now implemented. The remaining gap is:

1. **Extend the generic renderer to SCENARIO and FORECAST** — absorbing the five regex-based fallback detectors in `summarizer.py`. This would eliminate the last per-question fix surface in Stage 4.
2. **Evidence planner validation against `answer_kind`** — ensuring evidence steps match the answer shape before collection, rather than catching mismatches after the fact in the evidence validator.
3. **Stage 3 unified `answer_kind` dispatch** — removing legacy regex signal detection so enrichment routing is a clean switch on `answer_kind`.

---

## 8. Recommended Target Architecture

**Design principle: make Stage 0.2 emit the full contract. Everything after just executes.**

### 8.1 Strengthen The Analyzer Contract (Stage 0.2) — DONE

Implemented. `QuestionAnalysis` now emits `answer_kind`, `render_style`, `grouping`, `entity_scope` alongside existing `candidate_tools`, `params_hint`, `evidence_roles`, `derived_metrics`. Active cross-check runs on every call (`agent/pipeline.py:235-274`). Fallback derivation from `query_type` + keyword heuristics active (`agent/pipeline.py:1070`). Tool selection remains deterministic via `candidate_tools[0]` + evidence planner — `primary_tool` is NOT emitted.

### 8.2 Normalize Tool Outputs Into Canonical Evidence Frames — DONE

Implemented. `ObservationFrame`, `EntitySetFrame`, `ComparisonFrame` defined in `contracts/evidence_frames.py`. Per-tool frame adapters in `agent/frame_adapters.py` with `adapt_tool_result()`. Frames constructed inline during evidence collection at `agent/pipeline.py:827-866`. Provenance refs bound at construction time.

### 8.3 Build A Generic Tabular Answer Renderer — DONE

Implemented. `agent/generic_renderer.py` handles SCALAR, LIST, TIMESERIES, COMPARISON from canonical evidence frames. Called first in Stage 4 via `_try_generic_renderer()` at `agent/summarizer.py:1878`. New question families that produce these shapes work with zero Stage 4 code.

Remaining: the generic renderer does not yet handle SCENARIO or FORECAST — those still use specialized formatters with regex-based eligibility detection. See section 12.1 Gap 2 for the recommendation to extend the renderer.

### 8.4 Validate Evidence During Collection — DONE

Implemented. `agent/evidence_validator.py` with `validate_evidence()` called at `agent/pipeline.py:869` during frame attachment. Detects correctable and uncorrectable gaps.

### 8.5 Stage 3 Enrichment Dispatch On `answer_kind` — PARTIAL

Stage 3 uses `answer_kind` for some dispatch paths (scenario at `summarizer.py:1177`, forecast at `summarizer.py:1536`) but retains parallel legacy regex detection for share intent, correlation context, and forecast mode. See section 12.1 Gap 1.

### 8.6 Make Vector Knowledge Conditional — PARTIAL

Skip for deterministic data is implemented (`agent/pipeline.py:1083-1098`). Light retrieval tier for narrative data is NOT yet implemented — currently binary skip/full. See section 12.1 Gap 4.

### 8.7 Keep SQL As The Escape Hatch

Folded into the tool execution failure path, not a separate pipeline stage. Fires only when:

- typed tools fail or produce no result
- unusual aggregations not covered by typed tools
- narrow expert-mode fallback

### 8.8 Risk: Trading Regex Brittleness For LLM Non-Determinism

The target architecture trades one failure mode for another:

- **Current failure mode (regex brittleness):** loud. Regex miss → falls to LLM summarizer → grounding may catch the mismatch → user sees a fallback or generic answer. The failure is visible.
- **New failure mode (LLM non-determinism):** silent. LLM emits wrong `answer_kind` → generic renderer confidently formats the wrong answer shape → no grounding failure because the evidence frame is technically valid. The failure is invisible — the user gets a well-formatted wrong answer.

Additionally, different phrasings of the same question may produce different `answer_kind` values. "List prices for each month" could be LIST or TIMESERIES depending on phrasing. The current regex system is brittle but consistent — same keywords always produce the same path. The LLM is more flexible but less predictable.

**Mitigations:**

1. **Active cross-check (not just fallback):** The `query_type` → `answer_kind` derivation should always run, even when the analyzer succeeds. If LLM-emitted and derived `answer_kind` disagree, log a warning and prefer the safer option (e.g., prefer NARRATIVE over DETERMINISTIC when in doubt, since the LLM summarizer can handle ambiguity).

2. **Evidence validation catches structural mismatches:** Section 8.4's validation catches cases like COMPARISON with only one period — the frame doesn't have the required structure, so the system re-plans or degrades. This is a safety net for misclassification.

3. **Phase 1 shadow mode must measure consistency:** Before switching behavior, shadow mode must validate not just accuracy ("did the LLM pick the right answer_kind?") but consistency ("do semantically equivalent paraphrases get the same answer_kind?"). If "list regulated plants" → LIST but "which plants are regulated" → SCALAR, the contract needs tightening (constrained examples in the prompt, or fewer `answer_kind` values with `grouping` carrying the variance).

### 8.9 Handling Edge Cases: Filters And Cross-Tool Computations

The architecture in 8.1–8.8 covers ~80% of question diversity automatically. Two remaining gaps (~5% of questions) need small extensions, not new architecture.

**Gap 1 — Filtered/conditional questions** (e.g., "show months where price exceeded 15 tetri", "list entities where tariff increased"):

The analyzer already emits `sql_hints` with `aggregation`, `dimensions`, `entities`. The missing piece is a value-based filter condition in the contract. Solution: a `filter` field on `ToolParamsHint` with `{metric, operator (gt/lt/eq/gte/lte), value, unit}`. The LLM at 0.2 emits the filter (it already understands threshold conditions). Evidence collection applies the filter after fetching, before framing. `answer_kind` stays LIST or TIMESERIES — no new rendering logic needed. ~20 lines of code in the evidence pipeline, not a new stage or concept.

**Gap 2 — Cross-tool computational questions** (e.g., "weighted average price excluding regulated entities"):

Currently handled by `_build_residual_weighted_price_direct_answer` bespoke formatter. Default solution: `render_style = narrative` + `answer_kind = EXPLANATION` — let the LLM synthesize the cross-tool computation from pre-structured evidence. This already works for "why did price change" questions. If a repeating pattern appears: add a new `derived_metric` type to the catalog (e.g., `weighted_residual`). Stage 3 computes it from multi-tool evidence, same as existing MoM/YoY/correlation patterns. Principle: narrative rendering as default, derived_metric only when a pattern repeats.

| Gap | Solution | Where it lives | New code |
|---|---|---|---|
| Value filtering | `filter` field in ToolParamsHint | contracts + evidence collection | ~20 lines |
| Cross-tool computation | narrative rendering (default) OR new derived_metric (if pattern repeats) | existing LLM path OR catalogs + Stage 3 | 0–50 lines per pattern |

---

## 9. Recommended Refactor Plan — Remaining Work

Phases 1 and 2 are complete. Phase 3 is partially complete. Phase 4 is the remaining cleanup.

### Phase 1: Strengthen The Analyzer Contract — DONE

Completed. `QuestionAnalysis` emits `answer_kind`, `render_style`, `grouping`, `entity_scope`. Active cross-check runs on every call. Fallback derivation active. Mode derivation is inline.

### Phase 2: Canonical Evidence Frames + Generic Renderer — DONE

Completed. `ObservationFrame`, `EntitySetFrame`, `ComparisonFrame` defined. Per-tool frame adapters built. Generic renderer handles SCALAR/LIST/TIMESERIES/COMPARISON. Evidence frames constructed inline during collection. Provenance bound at construction.

### Phase 3: Evidence Validation + Conditional Knowledge — PARTIAL

**Done:**
- Evidence validation during collection (`agent/evidence_validator.py`, called at `pipeline.py:869`)
- Vector knowledge conditional skip for deterministic data paths (`pipeline.py:1083-1098`)

**Remaining:**
- Evidence planner should read `answer_kind` to validate evidence steps match the answer shape (zero references to `answer_kind` in `agent/evidence_planner.py`)
- Light vector retrieval tier (top-K=2, no re-rank) for narrative data questions
- `filter` field in `ToolParamsHint` for value-based filtering (Section 8.9)

Files: `agent/evidence_planner.py`, `knowledge/vector_retrieval.py`, `agent/pipeline.py`, `contracts/question_analysis.py`

### Phase 4: Consolidate Pipeline Steps — REMAINING

Scope (all items still pending):

- Extend generic renderer to handle SCENARIO and FORECAST, absorbing the five regex-based detectors in `agent/summarizer.py` (`_is_deterministic_scenario_eligible`, `_has_regulated_tariff_list_query_signal`, `_has_residual_weighted_price_query_signal`, `_is_forecast_direct_answer_eligible`, share_summary_override)
- Refactor Stage 3 enrichment dispatch to a single `switch(answer_kind)` — remove parallel legacy regex paths in `agent/analyzer.py`
- Remove explanation routing keyword fallback in `agent/pipeline.py:711-749` (gate behind `not ctx.has_authoritative_question_analysis`)
- Remove Stage 0.7 analyzer fallback (`agent/pipeline.py:1370-1504`) — `candidate_tools[0]` + evidence planner makes it unnecessary
- Fold legacy SQL (Stages 1/2) into tool execution failure path
- Remove separate provenance gate (provenance already bound at construction)
- Merge tool execution stages (0.5/0.6/0.7/0.8) into a single execution loop

Expected payoff:

- pipeline goes from current state to the 8-step ideal
- per-question fixes drop to near-zero for all standard answer shapes
- remaining fix surface is only: analyzer prompt tuning or specialized formatter additions (rare)

Files: `agent/summarizer.py`, `agent/generic_renderer.py`, `agent/analyzer.py`, `agent/pipeline.py`, `agent/router.py`

---

## 10. Practical Conclusion

The pipeline has made significant progress. Stage 0.2 now emits the full answer contract. Canonical evidence frames and the generic tabular renderer handle SCALAR/LIST/TIMESERIES/COMPARISON with zero per-question code. Evidence validation runs inline. Vector knowledge skips for deterministic data.

**The practical test:** when a new question family appears, does it require a Stage 4 patch?

- **For SCALAR/LIST/TIMESERIES/COMPARISON:** no — the generic renderer handles it automatically
- **For SCENARIO/FORECAST:** yes — these still use specialized formatters with regex-based eligibility detection
- **For EXPLANATION/KNOWLEDGE:** no — these route to the LLM summarizer via `render_style=narrative`

**Remaining structural work:** extend the generic renderer to absorb SCENARIO and FORECAST, refactor Stage 3 to a pure `answer_kind` switch, remove the five legacy regex detectors, and consolidate the execution stages. The prompt budget system also needs question-type-aware ordering and truncation (see sections 13-14).

---

## 11. Source Of Truth

This document reflects the runtime behavior currently implemented in:

- `agent/pipeline.py`
- `agent/router.py`
- `agent/evidence_planner.py`
- `agent/analyzer.py`
- `agent/summarizer.py`
- `agent/generic_renderer.py`
- `agent/frame_adapters.py`
- `agent/evidence_validator.py`
- `core/llm.py`
- `knowledge/vector_retrieval.py`
- `contracts/question_analysis.py`
- `contracts/question_analysis_catalogs.py`
- `contracts/evidence_frames.py`

If code and documentation diverge, update this file together with the relevant pipeline change.

---

## 12. Code Audit — Remaining Gaps Against Ideal Decision Tree (Section 3.4)

*Audit date: 2026-04-12. The core of the ideal decision tree is now implemented: Stage 0.2 emits the full contract (`answer_kind`, `render_style`, `grouping`, `entity_scope`), answer_kind cross-check runs on every call, vector knowledge skips for deterministic data, response mode derives inline, canonical evidence frames and the generic tabular renderer handle SCALAR/LIST/TIMESERIES/COMPARISON, and evidence validation runs inline during frame construction. This section documents only the remaining gaps and forward-looking recommendations.*

### 12.1 Partial Implementations — Gaps Remain

**Gap 1: Stage 3 dispatch not yet a single `switch(answer_kind)`**

Doc reference: lines 405-418. Stage 3 should dispatch purely on `answer_kind`.

Current state: The analyzer enrichment uses `answer_kind` for some paths — scenario eligibility checks `answer_kind == SCENARIO` (`agent/summarizer.py:1177-1185`), forecast checks `answer_kind == FORECAST` (`agent/summarizer.py:1536-1537`). But correlation and forecast enrichment in `agent/analyzer.py` still read `needs_driver_analysis`, `needs_correlation_context` flags alongside regex-based signal detection (share-intent regex, why-mode keywords) rather than switching purely on `answer_kind`.

What to change: Refactor `analyzer.py` enrichment to dispatch on `answer_kind` alone: EXPLANATION → correlation + causal reasoning, COMPARISON → MoM/YoY derived metrics, SCENARIO → scenario evidence, FORECAST → trendline pre-calculation.

Why: Eliminates duplicate signal detection. New `answer_kind` values route correctly without adding new regex patterns.

**Gap 2: Stage 4 still has regex-based fallback detectors**

Doc reference: lines 420-443. Stage 4 should be a single `switch(answer_kind)` with no regex.

Current state: The generic renderer is tried first (`agent/summarizer.py:1878`), which is a clean answer_kind switch. But when it returns None, Stage 4 falls through to five regex-based detectors: `_is_deterministic_scenario_eligible` (line 1174), `_has_regulated_tariff_list_query_signal` (line 1212), `_has_residual_weighted_price_query_signal`, `_is_forecast_direct_answer_eligible` (line 1531), and the share_summary_override check. These use keyword/regex matching.

What to change: Extend the generic renderer to handle scenario and forecast deterministic paths using `answer_kind`, then remove the regex detectors.

Why: The ideal says "No regex. No query-signal detection." Removing these detectors means new question families within existing answer_kind shapes need zero Stage 4 code.

What should be improved: The generic renderer currently returns None for scenario and forecast — it only handles SCALAR, LIST, TIMESERIES, COMPARISON. Adding SCENARIO and FORECAST to the renderer would absorb all five regex fallback paths.

**Gap 3: Evidence planner does not read `answer_kind` or `render_style`**

Doc reference: lines 359-371. The planner should use the contract to ensure evidence steps satisfy the answer shape.

Current state: `agent/evidence_planner.py` reads `candidate_tools` and `evidence_roles` from the analyzer contract but has zero references to `answer_kind` or `render_style` (confirmed by search).

What to change: Add answer_kind-aware evidence step validation:
- `answer_kind = LIST` → planner ensures entity-enumeration step
- `answer_kind = COMPARISON` → planner ensures two-period or two-entity evidence
- `answer_kind = TIMESERIES` → planner ensures period range coverage
- `answer_kind = SCENARIO` → planner ensures scenario params available

Why: Without this, the planner can build evidence steps that don't satisfy the answer shape, causing late-discovery gaps in the evidence validator.

**Gap 4: Light vector retrieval tier missing**

Doc reference: line 356. Three tiers: full retrieval for knowledge/explanation, light retrieval for narrative data, skip for deterministic data.

Current state: Binary skip/full only (`agent/pipeline.py:1083-1098`). When `answer_kind` is data + `render_style` is NARRATIVE, full retrieval runs. This over-retrieves for questions like "show monthly prices with context" where one relevant passage would suffice.

What to change: Add a light retrieval mode (top-K=2, no re-ranking) to `knowledge/vector_retrieval.py` and gate it in `agent/pipeline.py`:
- knowledge/explanation → full (top-K=6, re-ranked)
- narrative data → light (top-K=2, no re-rank)
- deterministic data → skip

Why: Reduces ~200-400ms latency and ~3,000 chars of prompt for narrative data questions that only need minimal context.

**Gap 5: Explanation routing still has keyword fallback**

Current state: `_should_route_tool_as_explanation()` at `agent/pipeline.py:711-749` checks `answer_kind == EXPLANATION` first (line 718) then falls through to keyword-based `_EXPLANATION_ROUTING_SIGNALS` when `answer_kind` is None.

What to change: Once `answer_kind` is reliably emitted (it is for all active analyzer paths), remove the keyword fallback or gate it behind `not ctx.has_authoritative_question_analysis`.

Why: Dual paths create risk of disagreement between answer_kind and keyword detection.

### 12.2 Not Yet Implemented

**1. Merged tool execution loop (doc lines 373-403)**

Code still has separate Stage 0.5 (plan-driven/keyword routing), Stage 0.6 (execution), Stage 0.7 (analyzer fallback at `agent/pipeline.py:1370-1504`), and Stage 0.8 (evidence loop). These have not been merged into a single execution loop as the ideal tree specifies.

Recommendation: Medium-term. Merge into a single loop that iterates over evidence plan steps with inline validation and framing. This is an architectural simplification that reduces stage count but does not change behavior.

**2. Stage 0.7 still active (doc lines 397-398)**

`match_tool()` from `agent/router.py` is still called as a fallback at `pipeline.py:1396`. The ideal says `candidate_tools[0]` + evidence planner makes Stage 0.7 unnecessary.

Recommendation: Track Stage 0.7 hit rate. If it triggers on <5% of queries and the evidence planner handles those cases, remove it. If it still catches meaningful traffic, investigate why the evidence planner misses those cases.

**3. SQL as failure-path escape hatch only (doc lines 394-401)**

Stages 1/2 remain as separate pipeline stages. The ideal says SQL should only be attempted inside the tool execution failure path, not as a separate parallel stage.

Recommendation: Low priority. Stages 1/2 are already rarely hit when the analyzer is active. Folding them in is architectural cleanliness, not a behavioral change.

### 12.3 Prioritized Recommendation Summary

| Priority | Item | Impact | Effort | Primary File |
|----------|------|--------|--------|-------------|
| 1 | Evidence planner reads answer_kind | Ensures evidence matches answer shape; eliminates late-discovery gaps | Medium | `agent/evidence_planner.py` |
| 2 | Extend generic renderer to SCENARIO + FORECAST; remove regex detectors from Stage 4 | New question families need zero Stage 4 code | Medium | `agent/summarizer.py`, `agent/generic_renderer.py` |
| 3 | Stage 3 unified answer_kind dispatch | Removes duplicate signal detection in analyzer enrichment | Medium | `agent/analyzer.py` |
| 4 | Light vector retrieval tier | ~200-400ms latency reduction for narrative data | Small | `knowledge/vector_retrieval.py`, `agent/pipeline.py` |
| 5 | Remove explanation routing keyword fallback | Eliminates dual-path disagreement risk | Small | `agent/pipeline.py` |
| 6 | Merge tool execution stages (0.5-0.8 → single loop) | Architectural simplification | Large | `agent/pipeline.py`, `agent/evidence_planner.py` |
| 7 | Remove separate provenance gate | Redundant since frame construction binds provenance | Small | `agent/pipeline.py`, `agent/summarizer.py` |
| 8 | Fold SQL into failure path | Architectural cleanliness | Medium | `agent/pipeline.py` |
| 9 | Remove Stage 0.7 | Dead code once planner is reliable | Small | `agent/pipeline.py` |

---

## 13. Prompt Budget, Latency, And Fast/Deep Mode Recommendations

### 13.1 Current Prompt Budget Analysis

The pipeline manages prompt size through several mechanisms:

- **Global budget**: `PROMPT_BUDGET_MAX_CHARS = 45000` (`config.py:100`) with 10% headroom → effective ceiling of ~40,500 chars.
- **Two truncation orderings**: `_TRUNCATION_PRIORITY_DATA` (sacrifice knowledge before data) and `_TRUNCATION_PRIORITY_KNOWLEDGE` (sacrifice data before knowledge), defined at `core/llm.py:2554-2577`. Selected by `response_mode`: knowledge_primary gets `_TRUNCATION_PRIORITY_KNOWLEDGE`, everything else gets `_TRUNCATION_PRIORITY_DATA`.
- **Analyzer prompt**: Calls `_enforce_prompt_budget(prompt, label="question_analysis")` at `core/llm.py:2165` with **no** `truncation_priority` kwarg, falling back to the default `_TRUNCATION_PRIORITY` ordering. The analyzer prompt has no question-type awareness in its truncation.
- **Summarizer prompt**: History capped at `SESSION_HISTORY_MAX_TURNS=3` turns (`config.py:106`), answers truncated to 500 chars per turn.
- **Vector knowledge**: Capped at `VECTOR_KNOWLEDGE_MAX_CHARS=9000` (`config.py`), top-K=6 chunks.
- **Domain knowledge in planner**: Limited to `max(1200, PROMPT_BUDGET_MAX_CHARS // 3)` chars (`core/llm.py:995-1006`).
- **Timeout retry**: On Gemini timeout, retries with 75% budget (`core/llm.py:2485-2520`).

### 13.2 Identified Inefficiencies

**1. Analyzer prompt includes all catalog blocks unconditionally (~3,000-5,000 chars)**

The seven catalog/guide blocks at `core/llm.py:2076-2095` (QUERY_TYPE_GUIDE, ANSWER_KIND_GUIDE, FILTER_GUIDE, TOPIC_CATALOG, TOOL_CATALOG, CHART_POLICY_HINTS, DERIVED_METRIC_CATALOG) are always present regardless of question type. For a knowledge question like "What is balancing price?", the TOOL_CATALOG (~2,000 chars with combined_with rules and metric hints), DERIVED_METRIC_CATALOG (~1,500 chars with scenario instructions), and CHART_POLICY_HINTS (~400 chars) are irrelevant.

**2. Analyzer catalog blocks are not individually truncatable**

The section-aware truncation at `core/llm.py:2631-2684` relies on `UNTRUSTED_*:\n<<<...>>>` tags to identify sections. In the analyzer prompt, only `UNTRUSTED_USER_QUESTION` and `UNTRUSTED_CONVERSATION_HISTORY` use this format. The catalog blocks (QUERY_TYPE_GUIDE, TOOL_CATALOG, etc.) use a different tag format and cannot be individually truncated by the budget system. When the analyzer prompt exceeds budget, the system can only trim history and then falls through to the head+tail fallback.

**3. Conversation history always included in analyzer prompt**

`core/llm.py:2073` always includes `history_str` in the analyzer prompt. For first-turn questions this is a zero-cost no-op, but for follow-up questions history adds 1,500-3,000 chars. Many follow-ups are self-contained ("What was the balancing price in March 2025?" after a January question) and don't need history for correct classification.

**4. Scenario/forecast rules always present in the 65-line rule block**

Lines `core/llm.py:2142-2163` contain detailed scenario extraction rules (scenario_scale, scenario_offset, scenario_payoff, chart_intent, target_series) totaling ~1,500 chars. These are relevant for ~10% of queries but always occupy prompt budget.

**5. Domain knowledge loaded unconditionally for energy focuses**

`core/llm.py:2389-2396` loads `seasonal-rules.md` and `entity-taxonomy.md` for any energy-domain focus (balancing, generation, trade, energy_security). For deterministic data lookups, the generic renderer bypasses the LLM summarizer entirely, making this loaded knowledge unused. Overhead: 2,000-4,000 chars.

**6. Truncation priority is response_mode-aware but not question-type-aware**

The two truncation orderings distinguish data vs knowledge response modes. But within data questions, a forecast and a simple scalar lookup get identical truncation priority. For a forecast, the DERIVED_METRIC_CATALOG and trendline rules are critical; for a scalar lookup they are dispensable. The truncation system cannot make this distinction.

### 13.3 Question-Type-Aware Truncation Design

Extend the two existing truncation profiles to four, selected by `answer_kind` instead of `response_mode`:

```
_TRUNCATION_PRIORITY_DATA_HEAVY (deterministic SCALAR / LIST / TIMESERIES / COMPARISON):
  1. UNTRUSTED_CONVERSATION_HISTORY
  2. UNTRUSTED_DOMAIN_KNOWLEDGE
  3. UNTRUSTED_EXTERNAL_SOURCE_PASSAGES
  4. UNTRUSTED_DATA_PREVIEW
  5. UNTRUSTED_STATISTICS
  Rationale: deterministic paths never use knowledge; data and stats are critical.

_TRUNCATION_PRIORITY_KNOWLEDGE_HEAVY (KNOWLEDGE / CLARIFY):
  1. UNTRUSTED_CONVERSATION_HISTORY
  2. UNTRUSTED_DATA_PREVIEW
  3. UNTRUSTED_STATISTICS
  4. UNTRUSTED_DOMAIN_KNOWLEDGE
  5. UNTRUSTED_EXTERNAL_SOURCE_PASSAGES
  Rationale: knowledge questions need passages and domain knowledge; data is irrelevant.

_TRUNCATION_PRIORITY_EXPLANATION (EXPLANATION + NARRATIVE render_style):
  1. UNTRUSTED_CONVERSATION_HISTORY
  2. UNTRUSTED_STATISTICS
  3. UNTRUSTED_DATA_PREVIEW
  4. UNTRUSTED_DOMAIN_KNOWLEDGE
  5. UNTRUSTED_EXTERNAL_SOURCE_PASSAGES
  Rationale: explanations need both data context and knowledge; stats are derivable from data.

_TRUNCATION_PRIORITY_FORECAST_SCENARIO (FORECAST / SCENARIO):
  1. UNTRUSTED_CONVERSATION_HISTORY
  2. UNTRUSTED_EXTERNAL_SOURCE_PASSAGES
  3. UNTRUSTED_DOMAIN_KNOWLEDGE
  4. UNTRUSTED_DATA_PREVIEW
  5. UNTRUSTED_STATISTICS
  Rationale: forecasts/scenarios need stats (trendline data, scenario results) most; knowledge is secondary.
```

Selection logic (replaces the current `response_mode`-based selection in `core/llm.py`):

```python
if qa and qa.answer_kind in {AnswerKind.KNOWLEDGE, AnswerKind.CLARIFY}:
    priority = _TRUNCATION_PRIORITY_KNOWLEDGE_HEAVY
elif qa and qa.answer_kind == AnswerKind.EXPLANATION:
    priority = _TRUNCATION_PRIORITY_EXPLANATION
elif qa and qa.answer_kind in {AnswerKind.FORECAST, AnswerKind.SCENARIO}:
    priority = _TRUNCATION_PRIORITY_FORECAST_SCENARIO
else:
    priority = _TRUNCATION_PRIORITY_DATA_HEAVY
```

### 13.4 Fast Vs Deep Mode

Expose two modes to the user — "fast" for quick data lookups and "deep" for thorough analytical answers. The user signals which mode they want via an API parameter, a UI toggle, or a keyword prefix in the question (e.g., "quick: what is the balancing price?").

**Fast mode (latency target: <2s total pipeline):**

| Component | Fast Behavior |
|-----------|--------------|
| Analyzer prompt budget | 20,000 chars (vs 45,000) |
| Analyzer catalogs | Strip CHART_POLICY, DERIVED_METRIC_CATALOG for non-analytical questions; strip FILTER_GUIDE for non-threshold questions |
| Vector retrieval | Skip entirely regardless of answer_kind |
| Domain knowledge | Skip in summarizer |
| Conversation history | Include only if question is <20 chars or has anaphoric references |
| Summarizer for deterministic render_style | Skip LLM entirely — use generic renderer only |
| Summarizer for narrative render_style | Use reduced prompt (no domain knowledge, no vector knowledge) |
| Thinking budget | `ROUTER_THINKING_BUDGET = 512` (vs 2048) |

**Deep mode (current behavior, the default):**

All existing behavior unchanged. Full 45,000 char budget, full vector retrieval, all catalog blocks, all domain knowledge, full thinking budget.

**Configuration:**

- New env var: `PIPELINE_MODE = os.getenv("PIPELINE_MODE", "deep")`
- Per-request override via API parameter (allows the frontend to send a fast/deep signal)
- Budget selection: `_enforce_prompt_budget` already accepts `budget_override`; pass `FAST_MODE_ANALYZER_BUDGET = 20000` when in fast mode.

**How the user signals fast/deep:**

Option A — explicit API parameter: the request payload includes `"mode": "fast"` or `"mode": "deep"`. The frontend exposes this as a toggle or button.

Option B — keyword prefix: the user writes "quick:" or "fast:" at the start of their question. Stage 0 detects this prefix, strips it from the query, and sets `ctx.pipeline_mode = "fast"`. Transparent to the rest of the pipeline.

Option C — auto-detect: classify_query_type pre-classifies the question. Simple factual_lookup and data_retrieval with no analytical signals default to fast; explanations, forecasts, scenarios default to deep. The user can override.

Recommended: Start with Option A (explicit API parameter) because it gives the user control without auto-detection risk. Add Option B as a convenience shortcut. Reserve Option C for a future iteration after measuring fast-mode accuracy.

### 13.5 Conditional Block Inclusion Strategy For The Analyzer Prompt

The analyzer prompt at `core/llm.py:2069-2163` includes all blocks statically. Making blocks conditional based on a lightweight pre-classification (`classify_query_type` at `core/llm.py:403`, which is cheap — no LLM call) reduces prompt size for question types that don't need every block:

| Block | Always? | Include When | Omit When | Savings |
|-------|---------|-------------|-----------|---------|
| UNTRUSTED_USER_QUESTION | Yes | Always (protected) | Never | — |
| QUERY_TYPE_GUIDE | Yes | Always (needed for classification) | Never | — |
| ANSWER_KIND_GUIDE | Yes | Always (needed for answer_kind) | Never | — |
| TOPIC_CATALOG | Yes | Always (needed for candidate_topics) | Never | — |
| Schema + core rules | Yes | Always (contract definition) | Never | — |
| UNTRUSTED_CONVERSATION_HISTORY | Conditional | History exists AND (question <20 chars OR contains anaphoric tokens: "it", "the same", "those", "this", "them", "and also", "what about") | No history, or self-contained long question | ~1,500-3,000 chars |
| TOOL_CATALOG | Conditional | Pre-classified as data-path question | Pre-classified as `conceptual_definition` or `regulatory_procedure` | ~2,000 chars |
| FILTER_GUIDE | Conditional | Question contains numeric token + threshold verb ("above", "exceed", "more than", "%") | No threshold language | ~600 chars |
| CHART_POLICY_HINTS | Conditional | Question contains chart/visual keywords ("chart", "graph", "plot", "show", "visualize") or pre-classified as timeseries/comparison | Knowledge, scalar, clarify with no chart language | ~400 chars |
| DERIVED_METRIC_CATALOG | Conditional | Pre-classified as `data_explanation`, `comparison`, `forecast`, or question has analytical signals ("why", "change", "trend", "compare", "scenario", "what if") | Knowledge, simple data_retrieval, factual_lookup with no analytical signals | ~1,500 chars |
| Scenario extraction rules | Conditional | Question contains hypothetical language ("what if", "CfD", "PPA", "payoff", "strike price", "hypothetical") | No hypothetical language | ~800 chars |
| Chart-intent rules | Conditional | Same gate as CHART_POLICY_HINTS | Same | ~400 chars |

Total potential savings for a simple knowledge question: ~5,700 chars (~14% of budget).
Total potential savings for a simple scalar lookup: ~3,600 chars (~9% of budget).

### 13.6 Latency Reduction Recommendations

| # | Recommendation | Estimated Saving | Effort | Location |
|---|---------------|-----------------|--------|----------|
| 1 | Fast mode: skip LLM summarizer for deterministic render_style | 800-1,500ms | Medium | `agent/summarizer.py` — already partially done via generic renderer; extend to scenario/forecast |
| 2 | Fast mode: reduce analyzer thinking budget to 512 tokens | 200-500ms | Trivial | `config.py` — `ROUTER_THINKING_BUDGET` |
| 3 | Light vector retrieval (top-K=2, no re-rank) for narrative data | 200-400ms | Medium | `knowledge/vector_retrieval.py` |
| 4 | Conditional catalog blocks in analyzer prompt | 100-200ms (smaller prompt → faster LLM processing) | Medium | `core/llm.py:2069-2095` |
| 5 | Conditional history in analyzer prompt | 50-100ms | Small | `core/llm.py:2045,2073` |
| 6 | Skip domain knowledge for deterministic data paths in summarizer | 50-100ms (file I/O + prompt size) | Small | `core/llm.py:2389-2396` |
| 7 | Cache compact-JSON catalog strings at module load time (avoid reserializing on every call) | 10-20ms | Trivial | `core/llm.py:2047-2053` — `_compact_json()` is called per-request on static data |

Combined fast-mode savings for a simple deterministic lookup: estimated 1,200-2,500ms (dominated by skipping the summarizer LLM call).

---

## 14. Stage 0.2 Contract-First, Question-Type-Aware Prompt Design

### 14.1 Design Principle

**Contract first, then question-type-specific ordering, then question-type-specific truncation.**

The analyzer prompt should be structured as three layers:

1. **Non-negotiable core** — always present, never truncated. Contains the minimum contract the ideal decision tree depends on: user question, output schema, and the rules that govern contract fields.
2. **Question-type-ordered blocks** — the most important reference material for the current question family appears first (closest to the core), where the LLM pays the most attention. Less relevant blocks appear later, in truncation-friendly positions.
3. **Optional conditional blocks** — only included when lightweight pre-signals (keyword detection via `classify_query_type`) indicate they are needed. Omitted otherwise.

The ordering matters because: (a) LLMs attend more strongly to content near the instruction and near the end; placing the most relevant context immediately after the core improves classification accuracy, and (b) when truncation occurs, it removes from the bottom of the priority list, so less-relevant blocks are dropped first.

### 14.2 Protected Core Specification

These blocks are ALWAYS present and NEVER truncated:

```
1. UNTRUSTED_USER_QUESTION          — the raw user question; anchor for all classification
2. QUERY_TYPE_GUIDE                 — needed for query_type emission (~800 chars)
3. ANSWER_KIND_GUIDE                — needed for answer_kind emission (~1,200 chars)
4. Output schema                    — the JSON schema hint for QuestionAnalysis
5. Core rules (~30 lines)           — rules for answer_kind, render_style, grouping,
                                      entity_scope, preferred_path, candidate_tools,
                                      params_hint, canonical_query_en, confidence,
                                      preferred_path routing, and metric vocabulary
```

The core rules are the subset of the current 65-line rule block (`core/llm.py:2100-2163`) that define contract semantics. Rules that are specific to rare question types (scenario extraction, chart intent, season comparison) move to conditional add-on blocks.

Estimated protected core size: ~4,000-5,000 chars. This leaves 35,000-36,000 chars of effective budget for question-type-specific content.

### 14.3 Question-Type-Based Block Ordering

After the protected core, blocks are ordered by relevance to the question type. The pre-classification uses the keyword-based `classify_query_type()` (`core/llm.py:403`), which runs before the analyzer call and costs zero LLM tokens.

**For factual_lookup, data_retrieval, comparison questions:**

```
Protected core
→ TOOL_CATALOG           (most relevant: which tool, what params, combined_with rules)
→ FILTER_GUIDE           (if threshold language detected)
→ DERIVED_METRIC_CATALOG (if analytical signals present)
→ CHART_POLICY_HINTS     (if chart language or timeseries)
→ TOPIC_CATALOG           (least relevant for data — the LLM mainly needs tool mapping)
→ CONVERSATION_HISTORY   (if anaphoric references detected)
```

Rationale: data questions mainly need correct tool planning and parameter hints. The tool catalog, date/filter rules, and derived-metric rules matter more than topic catalog or narrative knowledge.

**For data_explanation questions:**

```
Protected core
→ TOOL_CATALOG           (needed for multi-tool evidence planning)
→ DERIVED_METRIC_CATALOG (evidence-role guidance, derived metrics for explanation)
→ TOPIC_CATALOG           (may need knowledge routing for causal reasoning)
→ FILTER_GUIDE           (if threshold language)
→ CHART_POLICY_HINTS     (if chart language)
→ CONVERSATION_HISTORY   (if anaphoric)
```

Rationale: explanation quality depends on planning the right supporting evidence (correct tool combination, derived metrics). The tool catalog and combined_with rules come first because missing them causes incorrect evidence plans.

**For conceptual_definition, regulatory_procedure questions:**

```
Protected core
→ TOPIC_CATALOG           (most relevant: which knowledge domain to route to)
→ CONVERSATION_HISTORY   (if anaphoric — context matters for knowledge follow-ups)
→ [TOOL_CATALOG omitted]
→ [DERIVED_METRIC_CATALOG omitted]
→ [CHART_POLICY_HINTS omitted]
→ [FILTER_GUIDE omitted]
```

Rationale: the main decision is whether to route to knowledge and which topic domain. Tool details, metric catalogs, and chart policies add noise. Omitting them saves ~4,500 chars.

**For forecast, scenario questions:**

```
Protected core
→ DERIVED_METRIC_CATALOG (most relevant: scenario/forecast metric types and extraction rules)
→ TOOL_CATALOG           (needed for tool params)
→ Scenario extraction rules (conditional add-on: scenario_scale, scenario_offset, scenario_payoff)
→ CHART_POLICY_HINTS     (forecasts often produce charts)
→ TOPIC_CATALOG
→ FILTER_GUIDE           (rarely needed)
→ CONVERSATION_HISTORY   (if anaphoric)
```

Rationale: missing scenario/forecast rules causes the highest semantic loss — incorrect `scenario_factor`, wrong `scenario_volume`, or missing seasonal decomposition.

**For clarify-turn follow-up replies (free-text answers to a prior clarification request):**

```
Protected core
→ CONVERSATION_HISTORY   (promoted to the front of the middle order)
→ remaining middle blocks keep the base order for the underlying question family
  (data-shaped replies still keep TOOL_CATALOG ahead of TOPIC_CATALOG;
   knowledge-shaped replies still keep TOPIC_CATALOG ahead of TOOL_CATALOG)
```

Rationale: the strongest pre-analyzer clarify signal is not a short query or a bare question mark; it is a prior assistant clarification turn. When the current user turn is answering that prompt, conversation history becomes load-bearing for disambiguation, but the non-history ordering should still follow the underlying data vs. knowledge family.

### 14.4 Conditional Block Inclusion Rules

| Block | Include When | Detection Method |
|-------|-------------|-----------------|
| CONVERSATION_HISTORY | History exists AND (question contains anaphoric signals such as "it", "the same", "those", "that one", "this", "them", "and also", "what about", Georgian/Russian equivalents OR the latest assistant answer is a clarification prompt such as "Please choose one of these directions", "Reply with the option number", or "Which ... did you mean?") | Regex on question text + marker scan over the last assistant answer from raw `conversation_history` |
| TOOL_CATALOG | Pre-classified as data-path (not `conceptual_definition`, not `regulatory_procedure`) | `classify_query_type()` result |
| FILTER_GUIDE | Question contains numeric token adjacent to comparison word, or "%", or threshold verbs ("above", "below", "exceed", "more than", "less than") | Regex: `\d+.*?(above\|below\|exceed\|more than\|less than)` or reverse |
| CHART_POLICY_HINTS | Question contains chart/visual keywords ("chart", "graph", "plot", "show", "visualize", "diagram") OR pre-classified as timeseries/comparison | Keyword check |
| DERIVED_METRIC_CATALOG | Pre-classified as data_explanation, comparison, forecast, or question contains analytical signals ("why", "change", "trend", "compare", "forecast", "scenario", "what if") | Keyword check |
| Scenario extraction rules | Question contains: "what if", "hypothetical", "CfD", "PPA", "payoff", "strike price", "financial compensation", "what would" | Keyword check |
| Chart-intent rules | Same gate as CHART_POLICY_HINTS | Same |
| Season comparison guidance | Question contains: "summer vs winter", "seasonal", "season", "heating season", "cooling" | Keyword check |

### 14.5 Analyzer-Specific Truncation Priorities

**Implemented design**: The analyzer prompt now wraps catalog blocks in truncation-compatible section tags, and section-aware truncation can drop them individually. The protected core remains pinned via `CONTRACT_*` blocks and is never truncated.

The current runtime uses two **base** truncation priorities plus one **clarify overlay**:

```
_ANALYZER_TRUNCATION_DATA (data-path questions):
  1. UNTRUSTED_CONVERSATION_HISTORY
  2. UNTRUSTED_CHART_POLICY_HINTS
  3. UNTRUSTED_TOPIC_CATALOG
  4. UNTRUSTED_FILTER_GUIDE
  5. UNTRUSTED_DERIVED_METRIC_CATALOG
  6. UNTRUSTED_TOOL_CATALOG              (last resort — tool info is critical for data)

_ANALYZER_TRUNCATION_KNOWLEDGE (knowledge-path questions):
  1. UNTRUSTED_CONVERSATION_HISTORY
  2. UNTRUSTED_TOOL_CATALOG              (first non-history block — not needed for knowledge)
  3. UNTRUSTED_DERIVED_METRIC_CATALOG
  4. UNTRUSTED_CHART_POLICY_HINTS
  5. UNTRUSTED_FILTER_GUIDE
  6. UNTRUSTED_TOPIC_CATALOG              (last resort — topic info is critical)

CLARIFY overlay:
  - Start from the base DATA or KNOWLEDGE priority chosen by the underlying
    question family.
  - Move UNTRUSTED_CONVERSATION_HISTORY to the LAST truncation slot.
  - Preserve the relative order of all non-history blocks.
```

Selection at Stage 0.2:
```python
_pre_type = classify_query_type(user_query)
_prompt_profile = _classify_analyzer_prompt_profile(conversation_history, _pre_type)

if _pre_type == "regulatory_procedure":
    _ana_priority = _ANALYZER_TRUNCATION_KNOWLEDGE
else:
    _ana_priority = _ANALYZER_TRUNCATION_DATA

if _prompt_profile == "clarify":
    _ana_priority = move_history_to_end(_ana_priority)

prompt = _enforce_prompt_budget(prompt, label="question_analysis", truncation_priority=_ana_priority)
```

This keeps the spec intent ("history is critical for disambiguation") while avoiding a single global CLARIFY list that would accidentally make data clarifications topic-heavy.

### 14.6 Deterministic Choices Stay Out Of The Prompt

Already implemented. The prompt states "candidate_tools and candidate_topics are ranked candidates, not final decisions" (`core/llm.py:2115`). `primary_tool` is not in the schema or the rules. Tool selection remains `candidate_tools[0]` + evidence planner. This is correct and should not change.

### 14.7 Vector And Domain Knowledge Conditional Loading

Three-tier strategy to replace the current binary skip/full:

| answer_kind | render_style | Vector Retrieval | Domain Knowledge |
|-------------|-------------|-----------------|------------------|
| KNOWLEDGE | any | Full (top-K=6, re-ranked) | Full |
| EXPLANATION | any | Full (top-K=6, re-ranked) | Full |
| SCALAR, LIST, TIMESERIES, COMPARISON | NARRATIVE | Light (top-K=2, no re-rank) | Conditional (only if energy focus) |
| SCALAR, LIST, TIMESERIES, COMPARISON | DETERMINISTIC | Skip | Skip |
| FORECAST, SCENARIO | NARRATIVE | Light (top-K=2, no re-rank) | Seasonal rules only |
| FORECAST, SCENARIO | DETERMINISTIC | Skip | Skip |
| CLARIFY | any | Skip | Skip |

Implementation:
- Add `VectorRetrievalTier` enum (`FULL`, `LIGHT`, `SKIP`) to `contracts/` or `knowledge/vector_retrieval.py`
- `agent/pipeline.py:1083`: compute tier from `answer_kind` + `render_style`, pass to `retrieve_vector_knowledge()`
- `knowledge/vector_retrieval.py`: when `tier == LIGHT`, use `top_k=2` and skip re-ranking

### 14.8 Rare Rule Extraction

Move these from the always-present rule block to conditional add-on blocks:

| Rule Lines (core/llm.py) | Content | Include When |
|--------------------------|---------|-------------|
| 2142-2152 | Scenario extraction rules (scenario_scale, scenario_offset, scenario_payoff, scenario_aggregation, scenario_volume) | Hypothetical language detected |
| 2155-2163 | Chart intent and target_series rules (valid chart_intent values, valid target_series roles) | Chart language detected |
| 2138-2140 | Season comparison guidance (derived_metrics[].season field) | Seasonal comparison language detected |

This reduces the always-present rule block from ~65 lines to ~45 lines, saving ~1,000 chars for the majority of queries that are not scenarios, forecasts, or chart requests.

### 14.9 Implementation Steps

| Step | Files | Description |
|------|-------|-------------|
| 1 | `core/llm.py` | Create `_build_analyzer_prompt_blocks(user_query, history_str, pre_type)` that returns ordered blocks based on pre-classified question type |
| 2 | `core/llm.py` | Wrap catalog blocks in `UNTRUSTED_*:\n<<<...>>>` tags for truncation compatibility |
| 3 | `core/llm.py:~2551` | Define `_ANALYZER_TRUNCATION_DATA`, `_ANALYZER_TRUNCATION_KNOWLEDGE`, `_ANALYZER_TRUNCATION_CLARIFY` |
| 4 | `core/llm.py:2069-2163` | Replace static prompt concatenation in `llm_analyze_question()` with dynamic block assembly from step 1 |
| 5 | `core/llm.py:2165` | Pass question-type-specific truncation priority to `_enforce_prompt_budget()` |
| 6 | `core/llm.py:2100-2163` | Extract scenario, chart-intent, and season rules into conditional add-on blocks |
| 7 | `agent/pipeline.py:1083` | Implement three-tier vector retrieval selection |
| 8 | `knowledge/vector_retrieval.py` | Add `tier` parameter that adjusts `top_k` and `search_multiplier` |
| 9 | `config.py` | Add `PIPELINE_MODE`, `FAST_MODE_ANALYZER_BUDGET`, `FAST_MODE_SUMMARIZER_BUDGET` |
| 10 | `core/llm.py`, `agent/summarizer.py` | Fast mode: reduced budgets, skip domain knowledge, bypass summarizer LLM for deterministic paths |

### 14.10 Backward Compatibility And Validation

- **Block ordering changes** are internal to prompt assembly; the QuestionAnalysis output schema is unchanged. Downstream stages see the same contract.
- **UNTRUSTED_* wrapping** of catalog blocks adds section markers but does not change content.
- **Truncation priority changes** use the existing `truncation_priority` parameter of `_enforce_prompt_budget()`; no API change.
- **Fast/deep mode** is opt-in via config or API parameter; default is `deep` (current behavior).
- **Conditional block inclusion** may change analyzer behavior because different prompts produce different LLM outputs. Validation approach: run in shadow mode first — build the new prompt alongside the old one, send only the old one to the LLM, but log what the new prompt would have been. Compare prompt sizes and (in a later phase) compare classification agreement between old and new prompts on a test set.
- **Three-tier vector retrieval** affects only the `light` tier (new behavior); `full` and `skip` are unchanged. Validate by comparing answer quality on narrative data questions with full vs light retrieval.

### 14.11 Summary

The contract-first, question-type-aware design follows this principle:

> Protected core → question-type-ordered blocks → question-type-specific truncation.

For the core: user question, output schema, and the rules for `answer_kind`, `render_style`, `grouping`, `entity_scope`, `preferred_path`, `candidate_tools`, and `params_hint`. This is the minimum the ideal decision tree depends on.

For ordering: the most important reference material for the current question family appears first. Data questions front-load the tool catalog; knowledge questions front-load the topic catalog; forecast/scenario questions front-load the derived metric catalog; and free-text replies to prior clarification turns front-load conversation history via a Stage 0.2 prompt-profile overlay.

For truncation: when budget pressure forces cuts, the system drops the least relevant blocks for the current question type — and clarify-turn replies additionally keep conversation history the longest without discarding the underlying data-vs-knowledge priority.

For conditional inclusion: blocks that are irrelevant to the current question type are omitted entirely, not just pushed to the back. This is cleaner than truncation because it saves the budget without any content loss.

The result: the same budget delivers more relevant context per question, and truncation preserves the content that matters most for correct classification.

---

## 15. Phased Implementation Plan

Six phases, each independently deployable. Ordered so that each phase builds on the last but delivers value on its own. Every phase can be validated in isolation before moving to the next.

---

### Phase A: Evidence Planner Reads `answer_kind` + Remove Dual-Path Routing

**Problem being solved:** The evidence planner builds evidence steps without knowing what answer shape they need to satisfy. A COMPARISON question might get single-period evidence. A LIST question might get timeseries evidence. These mismatches are only caught late by the evidence validator, after wasted tool calls and latency. Separately, the explanation routing function has two paths (answer_kind check + keyword fallback) that can disagree.

**What to do:**

| # | Task | File | Reference |
|---|------|------|-----------|
| 1 | Add `answer_kind` and `render_style` reading to evidence planner: validate that planned steps can produce the expected answer shape | `agent/evidence_planner.py` | Section 12.1 Gap 3 |
| 2 | Add shape-specific plan validation rules: LIST → entity-enumeration step present; COMPARISON → two-period or two-entity evidence; TIMESERIES → period range; SCENARIO → scenario params | `agent/evidence_planner.py` | Section 12.1 Gap 3 |
| 3 | Gate explanation routing keyword fallback behind `not ctx.has_authoritative_question_analysis` so it only fires when the analyzer is absent | `agent/pipeline.py:711-749` | Section 12.1 Gap 5 |

**What to expect after:**
- Evidence gaps for shape mismatches are prevented at planning time, not discovered post-collection
- Fewer wasted tool calls for questions where the planner would have fetched the wrong evidence shape
- No risk of `answer_kind` and keyword-based explanation detection disagreeing when the analyzer is active
- Zero schema or API changes — purely internal planner logic

**Validation:** Run the existing test suite. Shadow-log planner decisions: for each query, log whether the new validation would have changed the plan vs what the old planner produced. Check that no valid plans are rejected.

---

### Phase B: Extend Generic Renderer + Remove Legacy Regex Detectors

**Problem being solved:** The generic renderer handles SCALAR, LIST, TIMESERIES, COMPARISON — but SCENARIO and FORECAST still use five regex-based eligibility detectors and specialized formatters in `agent/summarizer.py`. This means: (a) new phrasings for scenario/forecast questions that don't match the regex need per-question patches, (b) Stage 3 enrichment uses its own parallel signal detection alongside `answer_kind`, creating a second place where dispatch logic must be maintained, and (c) Stage 4 has two parallel dispatch mechanisms (generic renderer + regex ladder).

**What to do:**

| # | Task | File | Reference |
|---|------|------|-----------|
| 1 | Extend `agent/generic_renderer.py` to handle `answer_kind = SCENARIO`: format payoff breakdown (positive/negative sums, market vs combined) from scenario evidence frames | `agent/generic_renderer.py` | Section 12.1 Gap 2 |
| 2 | Extend `agent/generic_renderer.py` to handle `answer_kind = FORECAST`: format trendline + R² caveat + seasonal breakdown from forecast evidence frames | `agent/generic_renderer.py` | Section 12.1 Gap 2 |
| 3 | Remove `_is_deterministic_scenario_eligible` regex detector | `agent/summarizer.py:1174` | Section 12.1 Gap 2 |
| 4 | Remove `_has_regulated_tariff_list_query_signal` regex detector | `agent/summarizer.py:1212` | Section 12.1 Gap 2 |
| 5 | Remove `_has_residual_weighted_price_query_signal` regex detector | `agent/summarizer.py` | Section 12.1 Gap 2 |
| 6 | Remove `_is_forecast_direct_answer_eligible` regex detector | `agent/summarizer.py:1531` | Section 12.1 Gap 2 |
| 7 | Remove `share_summary_override` pass-through check from Stage 4 dispatch (absorb into generic renderer LIST/SCALAR path) | `agent/summarizer.py` | Section 12.1 Gap 2 |
| 8 | Refactor Stage 3 enrichment dispatch in `agent/analyzer.py` to a single `switch(answer_kind)`: EXPLANATION → correlation + causal reasoning, COMPARISON → MoM/YoY, SCENARIO → scenario evidence, FORECAST → trendline pre-calculation | `agent/analyzer.py` | Section 12.1 Gap 1 |
| 9 | Remove legacy regex signal detection from Stage 3: share-intent regex, `needs_driver_analysis` flag check, why-mode keywords, scenario-eligibility regex, forecast-mode keywords | `agent/analyzer.py`, `agent/summarizer.py` | Section 12.1 Gap 1 |

**What to expect after:**
- **Stage 4 becomes a single `switch(answer_kind)` with no regex** — the core promise of the ideal decision tree (section 3.4 lines 420-443)
- New question families for ANY standard answer shape (including SCENARIO and FORECAST) need zero Stage 4 code and zero regex additions
- Stage 3 enrichment dispatch is a clean switch — no parallel signal detection, no regex maintenance
- The per-question fix surface in `summarizer.py` drops to near-zero
- The five removed functions (~400 lines of regex logic) are replaced by ~100 lines of generic renderer extensions

**Validation:** For each removed regex detector, ensure the generic renderer produces identical or better output on the existing test queries. Run scenario and forecast test cases end-to-end. Shadow-compare old vs new output for a test batch.

**Depends on:** Phase A (evidence planner should validate shape before this phase removes the fallback detectors that sometimes caught shape mismatches late).

---

### Phase C: Analyzer Prompt Optimization — Contract-First, Question-Type-Aware

**Problem being solved:** The analyzer prompt at `core/llm.py:2069-2163` includes all seven catalog blocks unconditionally (~3,000-5,000 chars of catalogs always present). For a knowledge question, the TOOL_CATALOG, DERIVED_METRIC_CATALOG, and CHART_POLICY_HINTS are irrelevant noise. For a simple scalar lookup, scenario extraction rules (~1,500 chars) are wasted. The catalog blocks are not wrapped in `UNTRUSTED_*` tags, so the section-aware truncation system cannot truncate them individually — it can only trim history, then falls through to a destructive head+tail split. The prompt ordering is static: the most relevant block for a given question type may appear last, where LLM attention is weakest.

**What to do:**

| # | Task | File | Reference |
|---|------|------|-----------|
| 1 | Wrap catalog blocks (TOOL_CATALOG, DERIVED_METRIC_CATALOG, TOPIC_CATALOG, FILTER_GUIDE, CHART_POLICY_HINTS) in `UNTRUSTED_*:\n<<<...>>>` tags for truncation compatibility | `core/llm.py:2076-2095` | Section 14.5 |
| 2 | Extract scenario extraction rules (lines 2142-2152), chart-intent rules (2155-2163), and season comparison guidance (2138-2140) from the always-present rule block into conditional add-on blocks | `core/llm.py:2100-2163` | Section 14.8 |
| 3 | Create `_classify_analyzer_prompt_profile(conversation_history, pre_type)` and `_build_analyzer_prompt_blocks(user_query, history_str, pre_type, prompt_profile)` helpers so Stage 0.2 can separately model query shape vs. clarify-turn context | `core/llm.py` | Section 14.3, 14.9 step 1 |
| 4 | Replace static prompt concatenation in `llm_analyze_question()` with dynamic block assembly from the helper | `core/llm.py:2069-2163` | Section 14.9 step 4 |
| 5 | Implement conditional block inclusion: omit TOOL_CATALOG for knowledge questions, omit DERIVED_METRIC_CATALOG for simple lookups, omit CHART_POLICY for non-chart questions, and include/promote CONVERSATION_HISTORY when anaphoric signals or prior clarify-turn markers are present | `core/llm.py` | Section 14.4, 13.5 |
| 6 | Define base analyzer-specific truncation priority lists (`_ANALYZER_TRUNCATION_DATA`, `_KNOWLEDGE`) and implement CLARIFY as a history-priority overlay instead of a separate global list | `core/llm.py:~2551` | Section 14.5 |
| 7 | Pass prompt-profile-aware truncation priority to `_enforce_prompt_budget()` at the analyzer call site | `core/llm.py:2165` | Section 14.5, 14.9 step 5 |
| 8 | Cache `_compact_json()` results for static catalogs at module load time instead of reserializing per request | `core/llm.py:2047-2053` | Section 13.6 item 7 |

**What to expect after:**
- Knowledge questions save ~4,500 chars of irrelevant tool/metric/chart catalogs
- Simple scalar lookups save ~3,600 chars of unnecessary scenario rules and derived metric catalogs
- When truncation is forced (long history or very large catalogs), the system drops the least relevant blocks for the current question type — not a global ordering that is wrong for half the question families
- Analyzer prompt processing is ~100-200ms faster for questions with fewer blocks
- The always-present rule block shrinks from ~65 to ~45 lines
- Catalog serialization overhead eliminated (~10-20ms per call)
- No behavioral change for the default case — the same blocks are present, just ordered differently and sometimes omitted. Schema and output contract unchanged.

**Validation:** Shadow mode: build both old and new prompts, send only the old one, log both. Compare: (a) prompt sizes, (b) which blocks were omitted, (c) after enabling new prompts in a later step, compare classification agreement on a test batch. Watch for regressions where an omitted block was actually needed (e.g., a question classified as "knowledge" by the keyword heuristic but actually needing tool info).

**Independent of:** Phases A and B. Can be done in parallel if desired.

---

### Phase D: Three-Tier Vector Retrieval + Summarizer Prompt Optimization

**Problem being solved:** Vector retrieval is currently binary: full (top-K=6, re-ranked) or skip. Narrative data questions ("show monthly prices with some context") get full retrieval even though they only need one or two passages. This wastes ~200-400ms on retrieval and adds ~3,000-6,000 chars of vector knowledge to the summarizer prompt. Separately, the summarizer loads domain knowledge (seasonal-rules.md, entity-taxonomy.md) unconditionally for energy-domain focuses, even when the generic renderer will bypass the LLM summarizer entirely for deterministic paths. The summarizer truncation is response_mode-aware (data vs knowledge) but not answer_kind-aware (forecast vs scalar vs explanation).

**What to do:**

| # | Task | File | Reference |
|---|------|------|-----------|
| 1 | Add `VectorRetrievalTier` enum (`FULL`, `LIGHT`, `SKIP`) | `knowledge/vector_retrieval.py` or `contracts/` | Section 14.7 |
| 2 | Compute tier from `answer_kind` + `render_style` in pipeline (knowledge/explanation → FULL, narrative data → LIGHT, deterministic data → SKIP, clarify → SKIP) | `agent/pipeline.py:1083-1098` | Section 14.7 |
| 3 | Implement LIGHT tier in `retrieve_vector_knowledge()`: `top_k=2`, skip re-ranking | `knowledge/vector_retrieval.py` | Section 12.1 Gap 4 |
| 4 | Skip domain knowledge loading in summarizer when `render_style = DETERMINISTIC` (generic renderer will handle it without the LLM) | `core/llm.py:2389-2396` | Section 13.2 item 5 |
| 5 | Extend summarizer truncation from 2 profiles to 4: add `_TRUNCATION_PRIORITY_EXPLANATION` and `_TRUNCATION_PRIORITY_FORECAST_SCENARIO`; select by `answer_kind` instead of `response_mode` | `core/llm.py:2554-2577` | Section 13.3 |

**What to expect after:**
- Narrative data questions save ~200-400ms retrieval latency and ~3,000 chars of vector knowledge prompt
- Deterministic data paths skip ~2,000-4,000 chars of domain knowledge that was loaded but never used
- Forecast/scenario questions retain stats and data preview during truncation (critical for correct answers) while shedding knowledge passages first
- Explanation questions retain both data and knowledge context while shedding stats first (derivable from data)
- Overall: the pipeline spends prompt budget on what the specific question type actually needs

**Validation:** Compare answer quality on narrative data questions with full vs light retrieval. Verify that deterministic data paths still produce correct answers without domain knowledge. Test truncation behavior on queries that exceed the budget — confirm that forecast questions keep stats and knowledge questions keep passages.

**Depends on:** Phase A (evidence planner should know about answer_kind before retrieval tiers change what evidence is available). Independent of Phases B and C.

---

### Phase E: Fast / Deep Mode

**Problem being solved:** All queries currently get the same full-budget treatment regardless of complexity. A simple "What is the balancing price in March 2025?" goes through the same 45,000-char analyzer budget, full thinking budget, full summarizer prompt, and (if narrative) full vector retrieval as a complex "Explain why balancing prices changed and correlate with generation mix composition shifts." The user has no way to signal "I just need a quick number" vs "give me a thorough analysis." This creates unnecessary latency for simple lookups.

**What to do:**

| # | Task | File | Reference |
|---|------|------|-----------|
| 1 | Add `PIPELINE_MODE` config (`deep` default) and per-request `mode` API parameter | `config.py`, API handler | Section 13.4 |
| 2 | Add `FAST_MODE_ANALYZER_BUDGET = 20000` and `FAST_MODE_SUMMARIZER_BUDGET = 15000` config vars | `config.py` | Section 13.4 |
| 3 | Pass `budget_override` to `_enforce_prompt_budget()` when in fast mode | `core/llm.py` | Section 13.4 |
| 4 | Reduce `ROUTER_THINKING_BUDGET` to 512 tokens in fast mode | `core/llm.py` | Section 13.6 item 2 |
| 5 | In fast mode: skip vector retrieval entirely regardless of answer_kind | `agent/pipeline.py` | Section 13.4 |
| 6 | In fast mode: skip domain knowledge in summarizer | `core/llm.py:2389-2396` | Section 13.4 |
| 7 | In fast mode + deterministic render_style: skip LLM summarizer entirely (generic renderer only) | `agent/summarizer.py` | Section 13.4, 13.6 item 1 |
| 8 | In fast mode + narrative render_style: use reduced summarizer prompt (no domain knowledge, no vector knowledge) | `core/llm.py` | Section 13.4 |
| 9 | Implement keyword prefix detection ("quick:", "fast:") as convenience shortcut — strip prefix, set `ctx.pipeline_mode = "fast"` | `agent/pipeline.py` Stage 0 | Section 13.4 Option B |

**What to expect after:**
- Simple deterministic lookups complete in <2s (down from ~3-5s), because they skip: full-budget analyzer prompt processing, vector retrieval, domain knowledge loading, and the LLM summarizer call
- Users can choose their latency/thoroughness trade-off per question
- The API parameter lets the frontend expose a toggle ("Quick answer" / "Detailed analysis")
- Default behavior (`deep`) is completely unchanged — zero regression risk for existing users
- Combined fast-mode savings for a simple deterministic lookup: ~1,200-2,500ms (dominated by skipping the summarizer LLM call)

**Validation:** Benchmark fast mode latency on a set of simple factual/data queries. Compare answer quality: fast-mode answers for simple queries should be identical (generic renderer produces the same output). For narrative fast-mode, compare against deep-mode to quantify quality trade-off.

**Depends on:** Phase B (generic renderer must handle SCENARIO/FORECAST so fast-mode deterministic bypass covers all standard shapes). Phase C (conditional block inclusion) and Phase D (three-tier retrieval) are nice-to-have but not strictly required — fast mode can use cruder gates (skip everything) initially.

---

### Phase F: Pipeline Consolidation — Remove Legacy Stages

**Problem being solved:** The pipeline still carries structural artifacts from before the analyzer contract, evidence frames, and generic renderer: Stage 0.7 (analyzer fallback routing that `candidate_tools[0]` + evidence planner has made redundant), separate Stages 1/2 (legacy SQL as a parallel path instead of a failure-path escape hatch), a separate provenance gate (redundant since provenance is bound at frame construction), and the four-stage tool execution structure (0.5/0.6/0.7/0.8) that the ideal tree collapses into one loop. These add maintenance surface, make the pipeline harder to reason about, and create edge cases where legacy paths bypass the new architecture (e.g., Stage 0.7 results may not go through frame construction).

**What to do:**

| # | Task | File | Reference |
|---|------|------|-----------|
| 1 | Track Stage 0.7 hit rate for 2 weeks. If <5% of queries and the evidence planner handles those cases, proceed to remove | `agent/pipeline.py` | Section 12.2 item 2 |
| 2 | Remove Stage 0.7 (`pipeline.py:1370-1504`): `match_tool()` fallback, recovery logic, and the associated stage tracing | `agent/pipeline.py` | Section 12.2 item 2 |
| 3 | Remove the separate post-hoc provenance gate (provenance already bound at frame construction time) | `agent/pipeline.py`, `agent/summarizer.py` | Section 12.3 item 7 |
| 4 | Fold Stages 1/2 (legacy SQL) into the tool execution failure path: attempt SQL only when typed tools fail inside the execution loop, not as a separate parallel stage | `agent/pipeline.py` | Section 12.2 item 3 |
| 5 | Merge Stages 0.5/0.6/0.7/0.8 into a single execution loop that iterates over evidence plan steps with inline frame construction and validation | `agent/pipeline.py`, `agent/evidence_planner.py` | Section 12.2 item 1 |

**What to expect after:**
- Pipeline step count drops from ~10 to the target 8 (matching section 3.4's ideal tree)
- No more edge cases where legacy paths bypass frame construction or evidence validation
- `agent/pipeline.py` becomes significantly shorter and easier to reason about
- Debugging is simpler: one execution loop instead of four stages with fallback branches
- The ideal decision tree in section 3.4 becomes a literal description of the runtime, not an aspirational target

**Validation:** This phase is purely structural — behavior should not change. Verify by running the full test suite and comparing outputs on a large query batch before and after. Any output difference indicates a case where the legacy path was producing different results than the new architecture, which should be investigated (it's likely a bug in one path or the other).

**Depends on:** All previous phases. Phase B (generic renderer covers all shapes, so there's no need for regex fallback paths that Stage 0.7 might have fed). Phase A (evidence planner validates shape, so there's no need for Stage 0.7 to catch planning misses). Stage 0.7 hit rate data from step 1 should be collected starting now, in parallel with earlier phases.

---

### Phase Dependency Summary

```text
Phase A: Evidence planner + answer_kind ─────────────────────┐
  │                                                          │
  ▼                                                          │
Phase B: Generic renderer SCENARIO/FORECAST + remove regex   │
  │                                                          │
  ▼                                                          │
Phase E: Fast / Deep mode                                    │
                                                             │
Phase C: Analyzer prompt optimization ───── (independent) ───┤
                                                             │
Phase D: Three-tier vector + summarizer truncation ──────────┤
                                                             │
                                                             ▼
                                              Phase F: Pipeline consolidation
```

Phases A → B → E are sequential (each depends on the previous). Phases C and D are independent and can run in parallel with B or E. Phase F depends on all others.

### Estimated Impact Summary

| Phase | Per-question fix reduction | Latency improvement | Prompt budget improvement | Effort |
|-------|--------------------------|--------------------|--------------------------|---------| 
| A | Prevents evidence shape mismatches at planning time | — | — | Small |
| B | Eliminates all regex-based Stage 4 dispatch; Stage 3 becomes clean switch | — | — | Medium |
| C | — | ~100-200ms (smaller analyzer prompt) | ~3,600-5,700 chars saved per query | Medium |
| D | — | ~200-400ms (light retrieval) | ~2,000-6,000 chars saved per query | Medium |
| E | — | ~1,200-2,500ms for simple queries in fast mode | 50% budget reduction in fast mode | Medium |
| F | Eliminates edge cases from legacy paths | Marginal (fewer stages) | — | Large |
