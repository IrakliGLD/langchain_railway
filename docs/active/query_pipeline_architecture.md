# Query Pipeline Architecture

Current technical reference for the `langchain_railway` query pipeline, with an explicit architectural assessment and recommended redesign.

**Last updated:** 2026-04-05  
**Status:** Active and corrected for the current evidence-planner pipeline

---

## 1. Executive Summary

The pipeline is no longer best described as "keyword router first, then analyzer fallback." The current runtime flow is:

1. prepare context
2. run the structured LLM question analyzer
3. retrieve vector knowledge
4. derive response mode and resolution policy
5. build a deterministic evidence plan
6. execute the primary tool step
7. execute remaining evidence-plan steps and merge them
8. run deterministic analysis enrichment
9. choose a deterministic direct answer when possible, otherwise call the summarizer LLM
10. run provenance gating and chart generation

The system works, but it has a meaningful architectural weakness: semantics are spread across too many layers. Query meaning is interpreted in the analyzer, again in router heuristics, again in evidence-planner rules, and again in Stage 4 summarizer special cases. That is the main reason individual user questions can require targeted fixes.

The problem is not that the codebase is generally "bad." The main problem is missing middle-layer structure:

- tools return tool-shaped data, not answer-shaped evidence
- the analyzer produces routing hints, not a strong answer contract
- the summarizer has accumulated business logic that should live earlier in the pipeline

This also means the LLM is not used as efficiently as it could be. It is strong at normalization, ambiguity handling, and narrative explanation, but today it is also being used to bridge gaps between raw tool output and final answer shape.

---

## 2. What Changed Since The Previous Architecture Doc

The previous version of this document is outdated in several important ways:

- The authoritative LLM question analyzer is now central to routing whenever hints mode is enabled.
- Vector knowledge retrieval now runs before response-mode derivation and can influence Stage 4 context.
- The evidence planner is a first-class stage and is the main mechanism for multi-dataset questions.
- Stage 0.5 is often plan-driven, not just keyword-driven.
- The agent loop is no longer the general next step after a router miss. It is now a constrained fallback and only runs when there is no authoritative analyzer route.
- Stage 4 is not just "LLM summarization + provenance gate." It now contains multiple deterministic direct-answer branches that bypass the LLM entirely.

---

## 3. Current Runtime Flow

### 3.1 High-Level Pipeline

```text
HTTP /ask
  -> Stage 0   prepare_context
  -> Stage 0.2 question analyzer (LLM, structured JSON)
  -> Stage 0.3 vector knowledge retrieval
  -> response_mode + resolution_policy derivation
  -> Stage 0.4 evidence planner
  -> Stage 0.5 / 0.6 primary tool execution
  -> Stage 0.7 analyzer tool route fallback
  -> Stage 0.8 evidence loop + evidence merge
  -> Stage 1/2 legacy planner + SQL fallback (only if no usable tool path)
  -> Stage 3 analyzer enrichment
  -> Stage 4 deterministic direct answer OR LLM structured summary
  -> provenance gate
  -> Stage 5 chart builder
```

### 3.2 Short-Circuit Paths

There are several early exits:

- `ResolutionPolicy.CLARIFY` -> `summarizer.answer_clarify()`
- `ResponseMode.KNOWLEDGE_PRIMARY` -> `summarizer.answer_conceptual()`
- deterministic Stage 4 direct-answer branches -> skip Stage 4 LLM

This means the pipeline is not a simple linear chain anymore. It is a policy-driven decision graph.

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
│   │   → produces QuestionAnalysis: query_type, preferred_path, candidate_tools,
│   │     evidence_roles, derived_metrics, canonical_query_en
│   │
│   ├─ [analyzer enabled in SHADOW mode + succeeds]
│   │   → analysis stored for observability only
│   │   → NOT authoritative — all downstream decisions use heuristics
│   │
│   └─ [analyzer disabled or fails]
│       → no analysis; heuristic fallback for everything downstream
│
├─ Resolve downstream query
│   ├─ [authoritative analysis] → use canonical_query_en
│   └─ [no analysis]           → use raw query
│
├─ Stage 0.3: vector knowledge retrieval
│   → retrieve domain/policy passages for the resolved query
│   → pack for later Stage 4 prompts (active or shadow)
│
├─ Response Mode Derivation (single source of truth, set once)
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
├─ Stage 0.6: tool execution (if Stage 0.5 produced an invocation)
│   ├─ [execute + relevance validated]
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
│   ├─ [has authoritative analysis + hints enabled]
│   │   → build ToolInvocation from analyzer candidates
│   │   ├─ [built + executed + relevant] → store result
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
│   → scenario evidence dispatch
│   → forecast/CAGR, correlation, "why" causal reasoning
│   → trendline pre-calculation, MoM/YoY, seasonal signals
│   → build structured analysis_evidence records
│
├─ Post-Stage 3: evidence readiness check
│   ├─ [missing requested evidence + no evidence at all]
│   │   → CLARIFY → RETURN
│   ├─ [missing requested evidence + partial evidence]
│   │   → continue with warning
│   └─ [all evidence present]
│       → continue
│
├─ Stage 4: answer dispatch *** DECISION LADDER (first match wins) ***
│   │
│   ├─ [ctx.share_summary_override populated]
│   │   → pass-through Stage 3's deterministic share answer (confidence 1.0)
│   │
│   ├─ [scenario eligible: scenario records in analysis_evidence
│   │    + query_type in {data_retrieval, data_explanation}
│   │    + no explanation signals]
│   │   → deterministic scenario formatter (confidence 0.95)
│   │
│   ├─ [regulated tariff list signal: get_tariffs tool
│   │    + regulation keywords + list keywords + no explanation signals]
│   │   → deterministic tariff list formatter (confidence 0.98)
│   │
│   ├─ [residual weighted-price signal: weighted-average keywords
│   │    + balancing context + residual/remaining scope]
│   │   → deterministic residual calculation formatter (confidence 0.95)
│   │
│   ├─ [trendline forecast eligible: trendline data in stats_hint
│   │    + forecast query_type or forecast keywords]
│   │   → deterministic forecast formatter (confidence 0.95)
│   │
│   └─ [none of the above]
│       → LLM llm_summarize_structured()
│       ├─ [grounding passes] → final answer
│       └─ [grounding fails]
│           ├─ scenario fallback available → deterministic scenario answer
│           └─ no fallback → generic grounding-failure message
│
├─ Provenance gate
│   → validate summary claims against source data
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

| | Current | Ideal |
|---|---|---|
| Total steps | 13 | 8 |
| Steps that interpret query semantics | 5+ (analyzer, mode deriv, router, evidence planner rules, Stage 4 regex) | 1 (analyzer) |
| Separate fallback/recovery stages | 3 (Stage 0.7, Stage 1/2, provenance gate) | 0 (folded into execution loop) |
| LLM calls for data questions | 1 (analyzer) + sometimes 1 (Stage 4 summarizer) | 1 (analyzer) + only for narrative render_style |

**Key differences summary:**

| Decision Point | Current | Ideal |
|---|---|---|
| `answer_kind` source | Not emitted; inferred from `query_type` in Stage 4 regex | LLM emits directly in QuestionAnalysis; cross-checked against `query_type` derivation |
| Tool selection | Keyword router or analyzer-candidate fallback | Deterministic: `candidate_tools[0]` + evidence planner (not LLM-emitted) |
| Tool routing mechanism | 3-way branch (plan / analyzer-skip / keyword router) + fallback stage | Direct execution from evidence plan; no routing stages |
| Evidence framing | Raw DataFrames passed to Stage 4 | Canonical frames built during collection |
| Evidence validation | None — Stage 4 discovers gaps late | Validated during collection against answer_kind |
| Stage 4 dispatch | 5 query-signal regex detectors | Single switch on `answer_kind` |
| Stage 4 rendering | One bespoke formatter per answer family | Generic renderer for SCALAR/LIST/TIMESERIES/COMPARISON |
| Grounding | Post-hoc token overlap | Provenance bound at construction time; no separate gate |
| Vector knowledge | Always collected | Conditional on answer_kind |
| New question family requires | Regex detector + bespoke formatter + Stage 4 patch | Usually nothing — generic renderer handles it |

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

- normalize the question into a strict `QuestionAnalysis`
- choose `query_type`
- choose `preferred_path`
- propose candidate tools
- emit `analysis_requirements`
- optionally mark `needs_multi_tool` and `evidence_roles`
- provide `canonical_query_en`

This is the semantic center of the current system. When active, it is the authoritative source for:

- response mode
- clarify vs answer
- requested derived metrics
- evidence-planning inputs

### 4.3 Stage 0.3: Vector Knowledge Retrieval

Owned by `knowledge/vector_retrieval.py`.

Responsibilities:

- retrieve policy/regulation/domain passages for the resolved query
- pack passages for Stage 4 prompts
- run in active or shadow mode

This stage is useful for policy, procedure, and explanation support, but it increases prompt volume and can become expensive if overused for simple fact/list answers.

### 4.4 Response Mode Derivation

Owned by `agent/pipeline.py`.

Responsibilities:

- convert question-analysis output into one authoritative runtime mode
- derive `ResponseMode`
- derive `ResolutionPolicy`
- synchronize legacy flags such as `ctx.is_conceptual`

Current behavior is intentionally simple, but also coarse:

- `conceptual_definition` and `regulatory_procedure` -> `KNOWLEDGE_PRIMARY`
- `data_retrieval`, `data_explanation`, `factual_lookup` -> `DATA_PRIMARY`
- otherwise use `preferred_path` as the tie-breaker

This simplicity is operationally helpful, but it is also one reason some semantically different questions collapse into the same downstream handling.

### 4.5 Stage 0.4: Evidence Planner

Owned by `agent/evidence_planner.py`.

Responsibilities:

- expand `QuestionAnalysis` into ordered evidence steps
- choose the primary dataset
- add secondary evidence roles such as:
  - `composition_context`
  - `tariff_context`
  - `correlation_driver`
- keep time windows aligned across evidence sources

This is the most important recent architectural addition. It makes the pipeline much more explicit about multi-dataset questions than the older "tool plus ad hoc enrichment" model.

### 4.6 Stage 0.5 / 0.6: Primary Tool Execution

Owned by `agent/pipeline.py`, `agent/router.py`, and typed tools in `agent/tools/`.

Important current behavior:

- when an evidence plan exists, Stage 0.5 prefers the first unsatisfied plan step
- raw-query keyword routing is mostly fallback behavior when no authoritative analyzer is available
- Stage 0.6 executes the tool and stamps provenance
- tool relevance is validated after execution

This is a major change from the old documentation. The keyword router is no longer the main semantic entry point in analyzer-enabled mode.

### 4.7 Stage 0.7: Analyzer Tool Route Fallback

Responsibilities:

- if Stage 0.5 does not yield an invocation, convert analyzer candidates into a concrete tool call
- execute that tool
- validate relevance
- optionally fall through to limited recovery logic

This is now a constrained second-chance routing step, not the main route-selection mechanism.

### 4.8 Stage 0.8: Evidence Loop And Merge

Owned by `agent/evidence_planner.py`.

Responsibilities:

- execute remaining unsatisfied evidence-plan steps
- store evidence by role
- merge secondary datasets into the primary frame
- record join provenance
- optionally restore the primary dataset from collected evidence

This is the stage that makes multi-tool reasoning concrete. It is one of the strongest parts of the current architecture.

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
- build analysis evidence
- emit contradiction guards and overrides

This stage is where numeric interpretation becomes more semantic, but it still operates on raw tool-shaped frames and merged columns.

### 4.11 Stage 4: Deterministic Direct Answers Plus LLM Summarization

Owned by `agent/summarizer.py`.

Current behavior:

- first try deterministic direct answers
- otherwise call `llm_summarize_structured()`
- then run summary grounding and provenance gating

The important update here is that Stage 4 now contains several deterministic answer builders, including direct handling for:

- share summaries
- scenario queries
- regulated tariff list queries
- residual weighted-price queries
- trendline forecast answers

This is a sign of progress, but also a sign that Stage 4 has become a catch-all for missing architecture in earlier stages.

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

Main orchestrator. Owns stage ordering, policy decisions, fallback order, and integration logic between planner, tools, analyzer, summarizer, and charting.

### `agent/router.py`

Deterministic extraction and fallback routing:

- query keyword heuristics
- semantic fallback scoring
- date, currency, metric, and entity extraction

Today it acts partly as parameter extractor and partly as route selector.

### `agent/evidence_planner.py`

The current architectural backbone for multi-evidence questions:

- expands analyzer output into tool steps
- resolves aligned parameters
- executes remaining evidence
- merges evidence into the main frame

### `agent/analyzer.py`

Deterministic enrichment layer. Produces derived metrics and contextual evidence that Stage 4 can use.

### `agent/summarizer.py`

Mixed responsibility module:

- conceptual answer generation
- clarify answer generation
- deterministic direct answers
- LLM structured summarization
- grounding checks
- provenance preparation

This module currently contains more business logic than ideal.

### `core/llm.py`

LLM access layer:

- question analyzer prompt
- structured summarizer prompt
- domain knowledge retrieval prompt selection
- model selection, resilience, caching, prompt budgeting

---

## 6. Architectural Assessment

The system has strong deterministic assets, but there is a real architecture issue. The main issue is not "too much LLM" or "too little LLM" in isolation. The main issue is that the LLM and deterministic layers do not meet through a strong enough contract.

### 6.1 Query Semantics Are Re-Implemented In Multiple Places

The same intent is interpreted in several layers:

- question analyzer prompt and schema
- response-mode derivation
- router keyword/entity extraction
- evidence-planner expansion rules
- summarizer query-signal detection

Impact:

- fixes often land as local patches rather than reusable architecture
- new query families tend to require touching multiple modules
- debugging becomes difficult because there is no single semantic source of truth after Stage 0.2

### 6.2 Tool Outputs Are Tool-Shaped, Not Answer-Shaped

Typed tools return pivots, alias groups, and dataset-specific columns. That is useful for computation, but not always ideal for final answers.

Symptom:

- a user asks for a list of regulated plants
- the tool returns tariff-group evidence rather than a canonical "regulated entities" list
- the summarizer then needs special logic to convert raw columns into the expected answer shape

Impact:

- Stage 4 accumulates ad hoc answer builders
- broad natural-language questions fail unless special handling is added

### 6.3 Stage 4 Owns Too Much Business Logic

The summarizer now contains:

- prompt construction
- grounding policy selection
- deterministic list answers
- deterministic scenario answers
- deterministic forecast answers
- provenance token matching

This is too much responsibility for one stage.

Impact:

- answer correctness depends on Stage 4 knowing too much about tool schemas
- direct-answer logic grows one case at a time
- the LLM is sometimes skipped for the right reason, but without a general pattern

### 6.4 Grounding Is Mostly Post-Hoc

Today the system often generates or assembles the answer first and then checks whether the numbers can be grounded.

Impact:

- failures are discovered late
- user-visible fallbacks can happen even when the underlying evidence was sufficient
- the answer builder is not explicitly tied to evidence structure

### 6.5 Evidence Planning Is Better Than The Old Model, But It Stops Too Early

The evidence planner is good at selecting datasets, but it does not yet produce a canonical answer contract.

It decides:

- what evidence to fetch
- how to align periods
- which secondary sources to add

It does not decide strongly enough:

- what answer kind is expected
- what final entity set or aggregation should be rendered
- whether the output should be deterministic list/scalar/comparison/explanation

Impact:

- Stage 4 must still infer answer shape from query text and raw columns

### 6.6 The LLM Is Not Used Efficiently

The LLM is valuable for:

- resolving ambiguous follow-ups
- normalizing user intent
- identifying evidence needs
- producing concise explanation text when data is already structured
- answering knowledge/procedure questions

It is less efficient when used to recover answer structure that deterministic code could have defined earlier.

Current inefficiencies:

- analyzer output does not fully specify answer shape, so downstream code re-interprets the question
- Stage 4 often receives very large prompts containing preview, statistics, history, domain knowledge, and vector knowledge even when the answer is fundamentally a deterministic list or scalar
- special-case direct answers exist because the generic LLM summary path is not reliable enough for those shapes
- vector/domain knowledge can be attached late even when the user asked for a simple data retrieval

---

## 7. Why Individual Question Fixes Keep Happening

Individual fixes are not always a sign of poor engineering. Some are normal in language systems.

However, repeated fixes in the same area are a signal of missing generalization. In this codebase, repeated fixes are most likely when all three conditions are true:

1. the analyzer correctly identifies the broad path
2. the tool retrieves technically relevant evidence
3. the final answer shape is not represented explicitly anywhere before Stage 4

When that happens, the system tends to need one more of the following:

- router lexical expansion
- evidence-planner expansion rule
- summarizer direct-answer branch
- grounding exception or enrichment logic

That pattern indicates a structural gap, not just random edge cases.

### 7.1 Per-Question Fix Taxonomy

Not all per-question fixes are the same. Breaking them into categories reveals which ones the recommended architecture would eliminate and which require additional changes.

| Fix category | Example | Root cause | Would AnswerSpec alone fix it? |
|---|---|---|---|
| **Stage 4 regex miss** | New phrasing for "list regulated plants" doesn't match `_has_regulated_tariff_list_query_signal` | Answer shape detected by keyword matching instead of typed contract | **Yes** — `answer_kind = LIST` replaces regex |
| **Missing deterministic formatter** | New answer shape (e.g., residual weighted-price) needs a bespoke builder in summarizer.py | No generic renderer for the answer_kind; each shape is hand-coded | **Partially** — dispatch is cleaner, but still need to write a formatter |
| **Tool returns wrong shape** | User asks for regulated plants, tool returns tariff-group evidence, not an entity list | Tool output is tool-shaped, not answer-shaped | **Partially** — canonical frames help, but adapters need per-tool logic |
| **Evidence planner miss** | Comparison question needs two datasets but planner only fetches one | Evidence expansion rules don't cover this combination | **No** — AnswerSpec doesn't change evidence planning |
| **Analyzer misclassification** | "Which plants are regulated?" classified as `data_retrieval` instead of `factual_lookup` with list intent | LLM prompt doesn't emit answer_kind; post-hoc derivation can't recover | **No** — AnswerSpec derives from analyzer output, so bad classification propagates |
| **Grounding failure on valid data** | Answer is correct but post-hoc token overlap check fails | Provenance binding is too late | **Yes** — construction-time binding eliminates most of these |

### 7.2 The Generalization Gap

The AnswerSpec proposal in Section 8.1 eliminates the top row (regex dispatch) and bottom row (grounding). But the middle rows — missing formatters, wrong tool shape, evidence planning misses, analyzer misclassification — still produce per-question fixes.

The key missing generalization is: **there is no generic answer renderer that can handle an arbitrary SCALAR/LIST/TIMESERIES/COMPARISON question from canonical evidence frames without a bespoke formatter.** Today each answer shape (share summary, scenario, tariff list, residual weighted-price, trendline forecast) has its own hand-coded builder. When a new question family appears, a new builder must be written.

Three additional changes are needed to close this gap:

1. **Move `answer_kind` into the analyzer LLM prompt** — let the LLM emit answer_kind directly as part of `QuestionAnalysis`, rather than deriving it post-hoc from `query_type`. The LLM is better at understanding "this is a list question" or "this is a comparison" than a mapping table.

2. **Build a generic tabular answer renderer** — a single renderer that can format any `ObservationFrame`, `EntitySetFrame`, or `ComparisonFrame` into a user-facing answer based on `answer_kind` + `grouping` + `entity_scope`, without needing a bespoke function per question family. Specialized formatters (like the scenario payoff breakdown) remain for cases where the generic renderer is insufficient.

3. **Add evidence planner validation against AnswerSpec** — after collecting evidence, check whether the evidence satisfies the AnswerSpec's requirements (correct entity scope, expected metric present, enough periods for a comparison). If not, re-plan or clarify before reaching Stage 4.

---

## 8. Recommended Target Architecture

**Design principle: make Stage 0.2 emit the full contract. Everything after just executes.**

The LLM at Stage 0.2 is the smartest part of the pipeline. It already understands whether the user wants a list, a comparison, a timeseries, or an explanation. But today it only says `query_type = data_retrieval`, and then 5+ downstream stages try to reconstruct what the LLM already knew. That reconstruction is where per-question fixes land.

The fix is not more LLM calls. It is: strengthen the one LLM call so that nothing downstream needs to re-interpret the question.

### 8.1 Strengthen The Analyzer Contract (Stage 0.2)

This is the highest-impact change. Extend `QuestionAnalysis` so the LLM emits everything downstream stages need:

**New fields:**

| Field | Values | What it replaces |
|---|---|---|
| `answer_kind` | `scalar`, `list`, `timeseries`, `comparison`, `explanation`, `forecast`, `scenario`, `knowledge`, `clarify` | Stage 4 regex detectors + `_EXPLANATION_ROUTING_SIGNALS` checks + mode derivation logic |
| `render_style` | `deterministic`, `narrative` | `_should_route_tool_as_explanation()` + per-branch explanation-signal checks |
| `grouping` | `none`, `by_entity`, `by_period`, `by_metric` | Implicit; today inferred ad-hoc in Stage 4 formatters |
| `entity_scope` | e.g., `regulated_plants`, `all`, specific names | Implicit; today inferred from tool params or query text regex |

**Tool selection stays deterministic, not LLM-emitted.** There are only 4 tools. The mapping from domain to tool is trivial: prices → `get_prices`, tariffs → `get_tariffs`, generation → `get_generation_mix`, shares → `get_balancing_composition`. The LLM already emits `candidate_tools` with scores. Asking it to also emit `primary_tool` would add a redundant field that can conflict with `candidate_tools[0]`, requiring arbitration logic worse than the current router. Tool selection is a deterministic function of `candidate_tools[0]` + evidence planner — 3 lines of code, not an LLM decision.

**Why the LLM should decide answer shape, not downstream code:**

| User question | LLM understands | Today's pipeline infers |
|---|---|---|
| "which plants are regulated?" | LIST + entity_scope=regulated | `query_type=data_retrieval` → Stage 4 regex detects "which" + "regulated" |
| "how did price change Jan vs Feb?" | COMPARISON | `query_type=data_retrieval` → regex detects "vs" in 3 places |
| "show monthly prices for 2025" | TIMESERIES + grouping=by_period | `query_type=data_retrieval` → no way to distinguish from SCALAR until Stage 4 |
| "explain why prices rose" | render_style=narrative + EXPLANATION | `_EXPLANATION_ROUTING_SIGNALS` checked in pipeline.py, router.py, summarizer.py |
| "what if gas price increases 20%?" | SCENARIO + render_style=deterministic | `derived_metrics` contains scenario_payoff → Stage 4 `_is_deterministic_scenario_eligible()` regex |

**Fallback when analyzer is disabled or fails:**

- derive `answer_kind` from `query_type` mapping + keyword heuristics (existing logic, kept as active cross-check — see Section 8.7)
- derive tool selection from keyword router (existing `match_tool()`)
- default `render_style` to `narrative` (safer)

**What this eliminates:**

- response-mode derivation as a separate step (trivially inline: `answer_kind in {knowledge, clarify}` → short-circuit)
- Stage 0.7 analyzer fallback (evidence planner + `candidate_tools[0]` selects tool deterministically)
- all 5 query-signal regex detectors in Stage 4
- `_EXPLANATION_ROUTING_SIGNALS` checks scattered across 3 files
- Stage 3's own signal detection (share-intent regex, scenario-eligibility, forecast-mode keywords) — replaced by `answer_kind` switch

### 8.2 Normalize Tool Outputs Into Canonical Evidence Frames

During evidence collection (not as a separate stage), normalize tool DataFrames into typed frames:

- `ObservationFrame` — `{period, entity_id, entity_label, metric, value, unit, provenance_refs}`
- `EntitySetFrame` — `{entity_id, entity_label, membership_reason, provenance_refs}`
- `ComparisonFrame` — `{subject, baseline, comparison_value, delta, provenance_refs}`

Normalization happens inline as each tool result is collected. Not a separate pipeline stage.

Why this helps:

- answer builders operate on stable field names, not raw column names like `p_bal_gel` or `share_import`
- the generic renderer (8.3) becomes possible
- provenance refs are bound at collection time, eliminating the separate provenance gate

### 8.3 Build A Generic Tabular Answer Renderer

The change that most directly stops per-question fixes.

Currently each deterministic answer type has a bespoke builder. When a new question family appears, a new builder must be written. The generic renderer replaces most of them with one function:

| `answer_kind` | Input frame | Output |
|---|---|---|
| `SCALAR` | `ObservationFrame` | Single value + unit + period |
| `LIST` | `EntitySetFrame` | Bulleted entity list grouped by reason |
| `TIMESERIES` | `ObservationFrame` | Period-indexed table applying `grouping` |
| `COMPARISON` | `ComparisonFrame` | Subject vs baseline + delta + percent |

Specialized formatters remain only for answer kinds with domain-specific decomposition:

- `SCENARIO` — payoff breakdown (positive/negative sums, market vs combined)
- `FORECAST` — trendline + R² caveat + seasonal breakdown + regulatory assumptions

**Where the domain logic goes:** The generic renderer does not eliminate domain-specific formatting — it moves it to the evidence frame adapters. Each tool's adapter still needs domain-aware column mapping:

- `get_tariffs` adapter maps `regulated_hpp_tariff_` column prefixes → display names ("Enguri HPP") in `EntitySetFrame.entity_label`
- `get_prices` adapter maps `p_bal_gel`/`p_bal_usd` → dual-currency `ObservationFrame` entries with correct units
- `get_balancing_composition` adapter maps `share_import`, `share_thermal_ppa` → labeled entity shares

This adapter logic is written once per tool and is stable — it does not grow per question family. The net win is: domain logic at the adapter boundary is write-once, vs. domain logic in Stage 4 regex that grows every time a new question pattern appears.

**The practical test:** when a new question family appears that produces SCALAR, LIST, TIMESERIES, or COMPARISON, does it require new code? With the generic renderer: no, if the canonical evidence frame adapter is correct. Zero Stage 4 patches. Possibly a new adapter if a new tool is added (rare — 4 tools have been stable).

### 8.4 Validate Evidence During Collection

As each tool result is collected and framed, validate it against `answer_kind`:

- `LIST` → frame must contain entity identifiers
- `COMPARISON` → evidence must have two periods or two entities
- `TIMESERIES` → evidence must have >= 2 period observations
- `SCALAR` → evidence must contain the requested metric
- `FORECAST` → evidence must have trend data or enough history

On failure:

1. **Re-plannable** (missing secondary dataset) → planner adds corrective step, re-executes
2. **Not re-plannable** (fundamental mismatch) → downgrade `render_style` to `NARRATIVE` or `CLARIFY`

This is part of the evidence collection loop, not a separate pipeline stage.

Why this helps:

- gaps caught during collection, not discovered as Stage 4 failures
- evidence planner becomes self-correcting
- Stage 4 never receives evidence that cannot produce the expected answer

### 8.5 Stage 3 Enrichment Dispatch On `answer_kind`

Stage 3 (`analyzer.py`) contains ~90% of the pipeline's business logic but currently uses its own signal detection to decide what enrichment to run:

| Enrichment | Current signal detection | With `answer_kind` |
|---|---|---|
| Share resolution | `analyzer_indicates_share_intent` regex OR "share" keyword in query | `answer_kind` in {SCALAR, TIMESERIES} + `subject_domain` = shares |
| Scenario evidence | `_is_deterministic_scenario_eligible()` checks | `answer_kind` = SCENARIO |
| Forecast/CAGR | `query_type == "forecast"` OR forecast keywords | `answer_kind` = FORECAST |
| "Why" causal reasoning | `needs_driver_analysis` OR "why"/"explain" keywords | `answer_kind` = EXPLANATION |
| Standalone MoM/YoY | analyst-mode flag + derived_metrics presence | `answer_kind` = COMPARISON (or any data kind with derived_metrics) |

This eliminates a significant per-question fix surface: when a new analytical pattern is added to Stage 3, it switches on `answer_kind` instead of adding a new signal detector to `analyzer.py`. The enrichment dispatch becomes a simple switch like Stage 4's answer dispatch.

### 8.6 Make Vector Knowledge Conditional

- `answer_kind` in {`knowledge`, `explanation`} → full vector retrieval
- `render_style = narrative` for data questions → light retrieval
- `render_style = deterministic` → skip vector retrieval entirely

Why this helps: lower prompt size, lower latency, less irrelevant context in data answers.

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

## 9. Recommended Refactor Plan

Ordered by impact on per-question fix rate. Each phase is independently deployable. Four phases, not seven — the previous plan had too many separate steps for what is fundamentally one architectural change: make 0.2 firm, then execute.

### Phase 1: Strengthen The Analyzer Contract

**The single highest-impact change.** Everything else depends on this.

Scope:

- extend `QuestionAnalysis` schema in `contracts/question_analysis.py` with `answer_kind`, `render_style`, `grouping`, `entity_scope` (NOT `primary_tool` — tool selection stays deterministic via `candidate_tools[0]`)
- update the analyzer LLM prompt in `core/llm.py` to emit these fields
- add fallback derivation from `query_type` + keyword heuristics for when analyzer is disabled/fails
- implement active cross-check: always derive `answer_kind` from `query_type` mapping alongside the LLM-emitted value; log disagreements; prefer safer option on mismatch (see Section 8.8)
- run in shadow mode first: log analyzer-emitted `answer_kind` vs current Stage 4 dispatch decision to validate **both accuracy and consistency**
- consistency validation: test that semantically equivalent paraphrases produce the same `answer_kind` (e.g., "list regulated plants" vs "which plants are regulated" vs "show me the regulated power plants" should all produce LIST). If consistency is low, tighten via constrained examples in the prompt or reduce `answer_kind` cardinality with `grouping` carrying the variance

Expected payoff:

- validates that the LLM can reliably and consistently classify answer shape
- zero behavioral change initially (shadow mode)
- once validated: eliminates all 5 query-signal regex detectors in `summarizer.py`, eliminates `_EXPLANATION_ROUTING_SIGNALS` checks in 3 files, eliminates Stage 0.7 fallback routing, collapses mode derivation into inline code

Files: `contracts/question_analysis.py`, `core/llm.py`, `agent/pipeline.py`

### Phase 2: Canonical Evidence Frames + Generic Renderer

These two changes are coupled — the renderer needs the frames, and the frames are pointless without the renderer.

Scope:

- define `ObservationFrame`, `EntitySetFrame`, `ComparisonFrame` dataclasses
- build per-tool frame adapters with domain-specific column mapping (e.g., `regulated_hpp_tariff_` → "Enguri HPP" in `entity_label`; `p_bal_gel`/`p_bal_usd` → dual-currency ObservationFrame entries)
- normalize tool outputs into canonical frames inline during evidence collection (not a separate stage)
- build the generic tabular renderer: one function that formats any frame based on `answer_kind` + `grouping`
- migrate existing deterministic answer branches one at a time:
  1. regulated tariff list → generic LIST renderer over EntitySetFrame
  2. residual weighted-price → generic SCALAR renderer over ObservationFrame
  3. share summary → keep as-is (already works as Stage 3 pass-through)
- keep scenario and forecast as specialized formatters
- switch Stage 4 dispatch from regex to `answer_kind`
- switch Stage 3 enrichment dispatch from signal detection to `answer_kind` (see Section 8.5)
- bind provenance refs during frame construction (eliminates separate provenance gate)

Expected payoff:

- most new question families work with zero Stage 4 code
- Stage 4 shrinks from mixed-responsibility module to simple switch + generic renderer
- Stage 3 enrichment dispatch simplifies (no more share-intent regex, scenario-eligibility checks, forecast-mode keywords)
- provenance is structural, not post-hoc token matching
- domain logic lives in write-once tool adapters, not in growing Stage 4 regex

Files: new `contracts/evidence_frames.py`, new `agent/frame_adapters.py`, new `agent/generic_renderer.py`, `agent/pipeline.py`, `agent/summarizer.py`, `agent/analyzer.py`

### Phase 3: Evidence Validation + Conditional Knowledge

Scope:

- add `filter` field to `ToolParamsHint` in `contracts/question_analysis.py` for value-based filtering (see Section 8.9)
- apply filter during evidence collection (after fetch, before frame construction)
- validate evidence against `answer_kind` during collection (LIST needs entities, COMPARISON needs two periods, etc.)
- re-plan on correctable gaps, degrade gracefully on uncorrectable ones
- make vector knowledge retrieval conditional on `answer_kind` (skip for deterministic data answers)

Expected payoff:

- evidence gaps caught during collection, not discovered as Stage 4 failures
- lower latency and cost for simple data questions
- pipeline becomes self-correcting for common evidence gaps

Files: `agent/evidence_planner.py`, `agent/pipeline.py`, `knowledge/vector_retrieval.py`

### Phase 4: Consolidate Pipeline Steps

Once Phases 1-3 are in place, clean up the pipeline:

Scope:

- remove Stage 0.7 (analyzer fallback) — `candidate_tools[0]` + evidence planner makes it unnecessary
- fold mode derivation into 0.2 output processing (inline, not separate step)
- fold legacy SQL into tool execution failure path (not separate stage)
- remove separate provenance gate (provenance bound at construction)
- remove `_EXPLANATION_ROUTING_SIGNALS`, `_has_regulated_tariff_list_query_signal`, `_is_deterministic_scenario_eligible`, `_has_residual_weighted_price_query_signal`, `_is_forecast_direct_answer_eligible` — all replaced by `answer_kind` switch

Expected payoff:

- pipeline goes from 13 steps to 8
- per-question fixes drop to near-zero for SCALAR/LIST/TIMESERIES/COMPARISON
- remaining fix surface is only: analyzer prompt tuning (improve LLM classification) or specialized formatter additions (rare)

Files: `agent/pipeline.py`, `agent/summarizer.py`, `agent/router.py`

---

## 10. Practical Conclusion

The pipeline works. The evidence planner, structured analyzer, and deterministic direct answers are real progress. But the per-question fix pattern is a clear signal that something structural is missing.

**The root cause is simple:** the LLM at Stage 0.2 already understands what the user wants — list, comparison, timeseries, scalar — but it only says `query_type = data_retrieval`. Then 5+ downstream stages try to reconstruct what the LLM already knew, using regex and keyword matching. Each new question family that doesn't match the existing regex needs a patch.

**The fix is also simple:** make Stage 0.2 emit the full answer contract (`answer_kind`, `render_style`, `grouping`, `entity_scope`), then build one generic renderer that can format any SCALAR/LIST/TIMESERIES/COMPARISON from canonical evidence frames. No regex. No per-question formatters. No re-interpretation.

The result is a shorter pipeline (13 steps → 8) that does more with less code, because the LLM's understanding is trusted and executed, not discarded and reconstructed.

**The practical test:** when a new question family appears, does it require a Stage 4 patch?

- **Today:** almost always yes (new regex detector + new bespoke formatter)
- **After Phase 1 (analyzer contract):** only if Stage 4 dispatch is wrong (regex replaced by `answer_kind` switch)
- **After Phase 2 (generic renderer):** only if the answer has domain-specific decomposition the generic renderer can't express (scenario payoff breakdowns, forecast reliability caveats)
- **For most data questions:** the answer just works

---

## 11. Source Of Truth

This document reflects the runtime behavior currently implemented in:

- `agent/pipeline.py`
- `agent/router.py`
- `agent/evidence_planner.py`
- `agent/analyzer.py`
- `agent/summarizer.py`
- `core/llm.py`
- `knowledge/vector_retrieval.py`

If code and documentation diverge, update this file together with the relevant pipeline change.
