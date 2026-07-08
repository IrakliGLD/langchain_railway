# Design-Gap Closures Implementation Plan (six items from the 2026-07-07 design review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six design gaps identified in the query-pipeline design review — answer provenance surface, render fitness checks, self-growing golden set, conversational contract continuity, evidence-ontology agreement shadow, and evidence-triggered re-analysis — each as an independently shippable slice.

**Architecture:** Every behavior-affecting slice follows the repo's shadow-first rollout pattern (observe → counter → env-gated cutover); pure-additive surfaces default on. All slices preserve the single-interpretation principle: Stage 0.2 stays the only place the question is interpreted (re-analysis re-runs that same stage once with more information, never a new interpreter).

**Tech Stack:** Python 3.11+ (CI) / 3.14 (local), FastAPI, Pydantic v2 contracts, pytest (no real LLM/DB in `tests/`).

## Global Constraints

- Working directory for ALL commands: `D:\Enaiapp\langchain_railway` (the git repo root).
- No new runtime infrastructure or services (owner decision 2026-06-10).
- `tests/` never calls a real LLM or DB; mock at the module-path the code under test resolves.
- Monkeypatch surface rule: before moving/renaming any symbol, `grep tests/ -rn "<qualified.name>"`; if patched, keep a re-export and late binding.
- Prompt-truncation invariants pinned by `tests/test_vector_retrieval_tier.py` MUST keep holding: `UNTRUSTED_CONVERSATION_HISTORY` drops first in every profile; `UNTRUSTED_STATISTICS` preserved last for data profiles; `UNTRUSTED_EXTERNAL_SOURCE_PASSAGES` preserved last for KNOWLEDGE.
- Gate per task: focused test file(s) green, then ONE full targeted pass `python -m pytest tests/ --ignore=tests/security -q` (~3–4 min). Never two full passes at once.
- Update `docs/active/query_pipeline_architecture.md` in the same task that changes the behavior it documents.
- Commit per task, conventional style, ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Push only after the final task.
- Env flags: behavior-affecting → default OFF (`ENABLE_CONTRACT_CONTINUITY`, `ENABLE_EVIDENCE_REANALYSIS`); shadow/observability-only and pure-additive → default ON.

---

### Task 1: Answer-provenance block in the /ask response (design item 5)

**Files:**
- Create: `agent/answer_provenance.py`
- Create: `tests/test_answer_provenance.py`
- Modify: `models.py` (add `vector_retrieval_tier: str = ""` field to `QueryContext`, in the Stage 0.3 outputs region)
- Modify: `agent/pipeline.py` (one line in `process_query` right after `_retrieval_tier = _resolve_vector_tier(ctx)`: `ctx.vector_retrieval_tier = _retrieval_tier.value`)
- Modify: `main.py` (`response_chart_meta.update({...})` block, ~line 1068: add `"answer_provenance": build_answer_provenance(ctx),` and import)
- Modify: `docs/active/query_pipeline_architecture.md` (§3.9 tail: one paragraph "Answer provenance surface (2026-07-07)")

**Interfaces:**
- Produces: `agent.answer_provenance.build_answer_provenance(ctx) -> dict` — keys: `answer_path` (str: `"deterministic_render" | "specialized_formatter" | "narrative_llm" | "conceptual" | "clarify" | "unknown"`), `summary_source` (str), `data_source` (str: ctx.provenance_source), `tool_name` (str), `used_sql` (bool), `retrieval_tier` (str), `analyzer` (dict: `authoritative` bool, `query_type` str, `answer_kind` str, `confidence` float), `grounding_gate` (dict: `passed` bool, `reason` str, `coverage` float).
- Consumes: existing `QueryContext` fields only (`summary_source`, `provenance_source`, `tool_name`, `used_tool`, `safe_sql`, `summary_provenance_gate_*`, `question_analysis`, `has_authoritative_question_analysis`, new `vector_retrieval_tier`).

`answer_path` derivation (single source of truth, keep in the module):
```python
_PATH_BY_SUMMARY_SOURCE = {
    "generic_renderer": "deterministic_render",
    "deterministic_share_summary": "specialized_formatter",
    "deterministic_regulated_tariff_list_direct": "specialized_formatter",
    "deterministic_residual_weighted_price_direct": "specialized_formatter",
    "structured_conceptual_summary": "conceptual",
    "legacy_conceptual_text_fallback": "conceptual",
    "clarification_request": "clarify",
}
```
Anything else non-empty (`structured_summary`, `structured_summary_grounding_fallback`, `citation_gate_fallback`, `absence_claim_guardrail`) → `"narrative_llm"`; empty → `"unknown"`.

- [ ] **Step 1: Write the failing tests** — `tests/test_answer_provenance.py` with the standard env-setdefault header (copy the 6 `os.environ.setdefault` lines from `tests/test_evidence_planner.py`), then:

```python
from agent.answer_provenance import build_answer_provenance
from models import QueryContext


def _ctx(**overrides):
    ctx = QueryContext(query="test")
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def test_deterministic_render_path():
    ctx = _ctx(summary_source="generic_renderer", used_tool=True,
               tool_name="get_prices", provenance_source="tool",
               vector_retrieval_tier="skip")
    block = build_answer_provenance(ctx)
    assert block["answer_path"] == "deterministic_render"
    assert block["tool_name"] == "get_prices"
    assert block["used_sql"] is False
    assert block["retrieval_tier"] == "skip"


def test_narrative_and_gate_fields():
    ctx = _ctx(summary_source="structured_summary_grounding_fallback",
               summary_provenance_gate_passed=False,
               summary_provenance_gate_reason="coverage_below_threshold",
               summary_provenance_coverage=0.42)
    block = build_answer_provenance(ctx)
    assert block["answer_path"] == "narrative_llm"
    assert block["grounding_gate"] == {
        "passed": False, "reason": "coverage_below_threshold", "coverage": 0.42,
    }


def test_clarify_and_unknown_paths():
    assert build_answer_provenance(_ctx(summary_source="clarification_request"))["answer_path"] == "clarify"
    assert build_answer_provenance(_ctx())["answer_path"] == "unknown"


def test_analyzer_subblock_without_analysis():
    block = build_answer_provenance(_ctx())
    assert block["analyzer"] == {
        "authoritative": False, "query_type": "", "answer_kind": "", "confidence": 0.0,
    }
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_answer_provenance.py -q` → FAIL (`ModuleNotFoundError: agent.answer_provenance`).

- [ ] **Step 3: Implement `agent/answer_provenance.py`**

```python
"""Answer-provenance surface: how this answer was produced, for the client.

Pure read-only projection of QueryContext — no pipeline behavior. Rides in
chart_metadata.answer_provenance so UIs can express calibrated trust
(architecture §3.9). Additive and backward compatible.
"""
from __future__ import annotations

from models import QueryContext

_PATH_BY_SUMMARY_SOURCE = {
    "generic_renderer": "deterministic_render",
    "deterministic_share_summary": "specialized_formatter",
    "deterministic_regulated_tariff_list_direct": "specialized_formatter",
    "deterministic_residual_weighted_price_direct": "specialized_formatter",
    "structured_conceptual_summary": "conceptual",
    "legacy_conceptual_text_fallback": "conceptual",
    "clarification_request": "clarify",
}


def _answer_path(summary_source: str) -> str:
    if not summary_source:
        return "unknown"
    return _PATH_BY_SUMMARY_SOURCE.get(summary_source, "narrative_llm")


def build_answer_provenance(ctx: QueryContext) -> dict:
    qa = ctx.question_analysis
    authoritative = bool(ctx.has_authoritative_question_analysis)
    return {
        "answer_path": _answer_path(str(ctx.summary_source or "")),
        "summary_source": str(ctx.summary_source or ""),
        "data_source": str(ctx.provenance_source or ""),
        "tool_name": str(ctx.tool_name or "") if ctx.used_tool else "",
        "used_sql": bool(ctx.safe_sql) and not bool(ctx.used_tool),
        "retrieval_tier": str(getattr(ctx, "vector_retrieval_tier", "") or ""),
        "analyzer": {
            "authoritative": authoritative,
            "query_type": qa.classification.query_type.value if (qa and authoritative) else "",
            "answer_kind": (qa.answer_kind.value if (qa and authoritative and qa.answer_kind) else ""),
            "confidence": float(qa.classification.confidence) if (qa and authoritative) else 0.0,
        },
        "grounding_gate": {
            "passed": bool(ctx.summary_provenance_gate_passed),
            "reason": str(ctx.summary_provenance_gate_reason or ""),
            "coverage": float(ctx.summary_provenance_coverage),
        },
    }
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_answer_provenance.py -q` → PASS.
- [ ] **Step 5: Wire in** — add `vector_retrieval_tier: str = ""` to `QueryContext` in `models.py`; in `agent/pipeline.py process_query` set `ctx.vector_retrieval_tier = _retrieval_tier.value` right after `_retrieval_tier = _resolve_vector_tier(ctx)`; in `main.py` add `from agent.answer_provenance import build_answer_provenance` (imports region) and `"answer_provenance": build_answer_provenance(ctx),` inside the `response_chart_meta.update({...})` dict.
- [ ] **Step 6: Doc** — §3.9 tail paragraph: the response's `chart_metadata.answer_provenance` block (fields list, additive, derived read-only from ctx).
- [ ] **Step 7: Gate** — focused file + full targeted pass green.
- [ ] **Step 8: Commit** — `feat(api): answer-provenance block in chart_metadata`.

---

### Task 2: Render-fitness shadow checks (design item 3)

**Files:**
- Create: `agent/render_fitness.py`
- Create: `tests/test_render_fitness.py`
- Modify: `utils/metrics.py` (add counter dict + `log_render_fitness(tag: str)` following the `log_analyzer_cross_check` pattern; expose in the metrics snapshot the same way neighboring counters are)
- Modify: `agent/summarizer.py` in `summarize_data`, immediately after each deterministic `ctx.summary_source = ...` assignment block completes (single hook point just before returning the deterministic result — locate the shared return after lines ~2492–2525): call the check
- Modify: `docs/active/query_pipeline_architecture.md` (§3.9 coverage-boundary paragraph: add the shadow-fitness sentence)

**Interfaces:**
- Produces: `agent.render_fitness.evaluate_render_fitness(ctx) -> list[str]` returning violation tags from: `"empty_result_rendered"`, `"period_coverage_gap"`, `"requested_entities_missing"`. Also `period_bounds_from_hint(qa) -> tuple[str, str] | None` (reused by Task 6).
- Consumes: `visualization.chart_selector.detect_column_types` for time-column detection; `qa.tooling.candidate_tools[0].params_hint` for requested period/entities.

Shadow only: violations emit `trace_detail(log, ctx, "stage_4_render_fitness", "violations", tags=...)` + `metrics.log_render_fitness(tag)` per tag. No behavior change, no flag (observability is unconditional, like other trace_detail sites).

- [ ] **Step 1: Failing tests** — `tests/test_render_fitness.py` (env header as Task 1):

```python
import pandas as pd

from agent.render_fitness import evaluate_render_fitness, period_bounds_from_hint
from models import QueryContext
from contracts.question_analysis import QuestionAnalysis

# Reuse the analyzer-payload helper style from tests/test_evidence_planner.py:
# build a minimal QuestionAnalysis payload with tooling.candidate_tools[0].params_hint
# carrying start_date/end_date/entities. Copy _make_qa_payload from that file and
# extend params_hint (do not import across test files).


def _ctx_with(df: pd.DataFrame, qa: QuestionAnalysis | None) -> QueryContext:
    ctx = QueryContext(query="test")
    ctx.df = df
    ctx.rows = [tuple(r) for r in df.itertuples(index=False)]
    ctx.cols = list(df.columns)
    ctx.question_analysis = qa
    ctx.question_analysis_source = "llm_active"
    ctx.summary_source = "generic_renderer"
    return ctx


def test_empty_result_rendered_flagged():
    ctx = _ctx_with(pd.DataFrame({"date": [], "p_bal_gel": []}), qa=None)
    assert "empty_result_rendered" in evaluate_render_fitness(ctx)


def test_period_coverage_gap_flagged_when_disjoint():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-06-01"], "p_bal_gel": [1.0, 2.0]})
    qa = _qa_with_hint(start_date="2024-01-01", end_date="2024-12-31")
    assert "period_coverage_gap" in evaluate_render_fitness(_ctx_with(df, qa))


def test_overlapping_period_not_flagged():
    df = pd.DataFrame({"date": ["2024-03-01", "2024-06-01"], "p_bal_gel": [1.0, 2.0]})
    qa = _qa_with_hint(start_date="2024-01-01", end_date="2024-12-31")
    assert "period_coverage_gap" not in evaluate_render_fitness(_ctx_with(df, qa))


def test_no_hint_no_period_check():
    df = pd.DataFrame({"date": ["2020-01-01"], "p_bal_gel": [1.0]})
    assert evaluate_render_fitness(_ctx_with(df, None)) == []
```

- [ ] **Step 2: Verify failure**, **Step 3: implement**:

```python
"""Shadow fitness checks for deterministic renders (architecture §3.9).

Deterministic renders ship with no post-hoc numeric gate; these checks make
the "right shape, wrong rows" class VISIBLE (trace + counter) without
changing behavior. Cutover to CLARIFY-on-violation is a future, separately
gated decision.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from visualization.chart_selector import detect_column_types


def period_bounds_from_hint(question_analysis) -> tuple[str, str] | None:
    tooling = getattr(question_analysis, "tooling", None)
    tools = getattr(tooling, "candidate_tools", None) or []
    hint = tools[0].params_hint if tools else None
    if hint is None or not (hint.start_date and hint.end_date):
        return None
    return str(hint.start_date), str(hint.end_date)


def _df_date_span(df: pd.DataFrame) -> tuple[date, date] | None:
    if df is None or df.empty:
        return None
    time_cols, _, _ = detect_column_types(list(df.columns))
    if not time_cols:
        return None
    series = pd.to_datetime(df[time_cols[0]], errors="coerce").dropna()
    if series.empty:
        return None
    return series.min().date(), series.max().date()


def evaluate_render_fitness(ctx) -> list[str]:
    tags: list[str] = []
    df = ctx.df
    if df is None or df.empty or not ctx.rows:
        tags.append("empty_result_rendered")
        return tags

    qa = ctx.question_analysis if ctx.has_authoritative_question_analysis else None
    bounds = period_bounds_from_hint(qa) if qa is not None else None
    if bounds:
        span = _df_date_span(df)
        if span:
            req_start, req_end = date.fromisoformat(bounds[0]), date.fromisoformat(bounds[1])
            if span[1] < req_start or span[0] > req_end:
                tags.append("period_coverage_gap")

    if qa is not None:
        tools = qa.tooling.candidate_tools or []
        hint = tools[0].params_hint if tools else None
        entities = list(getattr(hint, "entities", []) or [])
        if entities:
            haystack = " ".join(str(c).lower() for c in df.columns)
            haystack += " " + " ".join(str(v).lower() for v in df.iloc[:, :].astype(str).values.flatten()[:500])
            if not any(e.lower() in haystack for e in entities):
                tags.append("requested_entities_missing")
    return tags
```

- [ ] **Step 4: Verify pass.**
- [ ] **Step 5: Wire** — `utils/metrics.py`: `self.render_fitness_events: dict[str, int] = {}` in `__init__` + `def log_render_fitness(self, tag: str)` incrementing under the existing lock pattern; snapshot exposure alongside `analyzer_cross_check_events`. `agent/summarizer.py`: at the single point where a deterministic `summary_source` has just been set and the function is about to return the deterministic result, add:

```python
from agent.render_fitness import evaluate_render_fitness  # module top

_fitness_tags = evaluate_render_fitness(ctx)
if _fitness_tags:
    for _tag in _fitness_tags:
        metrics.log_render_fitness(_tag)
    trace_detail(log, ctx, "stage_4_render_fitness", "violations", tags=_fitness_tags,
                 summary_source=ctx.summary_source)
```

Add a wiring test in `tests/test_render_fitness.py`: monkeypatch `agent.summarizer.evaluate_render_fitness` to return `["empty_result_rendered"]`, run a deterministic-summary path fixture from `tests/test_guardrails.py` style, assert `metrics.render_fitness_events` incremented.
- [ ] **Step 6: Doc** — §3.9: extend the coverage-boundary paragraph: "Since 2026-07-07 a shadow fitness check (`agent/render_fitness.py`) makes violations visible (`stage_4_render_fitness` trace + `render_fitness_events` counters) without changing behavior; CLARIFY-on-violation cutover is a future gated decision."
- [ ] **Step 7: Gate**, **Step 8: Commit** — `feat(render): shadow fitness checks for deterministic renders`.

---

### Task 3: Self-growing golden set — fixture-candidate emission + harvester (design item 6)

**Files:**
- Create: `agent/fixture_candidates.py`
- Create: `evaluation/harvest_fixture_candidates.py`
- Create: `tests/test_fixture_candidates.py`
- Modify: `agent/pipeline.py` — in `_cross_check_answer_kind`, immediately after `metrics.log_analyzer_cross_check("disagreement")`: `log_fixture_candidate("cross_check_disagreement", ctx)`
- Modify: `agent/summarizer.py` — where `ctx.summary_source = "citation_gate_fallback"` is set (~line 1074): `log_fixture_candidate("provenance_gate_failure", ctx)`
- Modify: `evaluation/routing_golden_set.py` — replace its private `_routed_fields` with an import: `from agent.fixture_candidates import routed_fields_snapshot` (keep the runner's ctx-level wrapper calling it with `ctx.question_analysis`)
- Modify: `docs/active/query_pipeline_architecture.md` (§5.3 regression-harness paragraph: add the harvest loop sentence)

**Interfaces:**
- Produces: `agent.fixture_candidates.routed_fields_snapshot(qa) -> dict` (keys: query_type, answer_kind, render_style, preferred_path, top_tool; tolerant of None qa → {}); `log_fixture_candidates.log_fixture_candidate(trigger: str, ctx) -> None` emitting exactly one log line: `ROUTING_FIXTURE_CANDIDATE {compact-json}` with keys `trigger, query, routed, trace_id` (json.dumps with `ensure_ascii=False, separators=(",", ":")`; strip newlines from query).
- Produces: `evaluation/harvest_fixture_candidates.py` CLI: `python evaluation/harvest_fixture_candidates.py <logfile> [--out cands.json]` → parses marker lines, dedupes by query (first occurrence wins), emits golden-set-shaped cases `{"id": "cand_001", "query": ..., "note": "trigger=...; observed=<routed json>; VERIFY expected before adopting", "expected": {}}`. Pure stdlib; no config import (must run keyless).

- [ ] **Step 1: Failing tests** — round-trip: `log_fixture_candidate` output line (capture via `caplog` on logger `"Enai"`) is parseable by the harvester's `parse_candidates(lines)` function; dedupe works; KA/RU query survives (`ensure_ascii=False`); `routed_fields_snapshot(None) == {}`.
- [ ] **Step 2: Verify failure**, **Step 3: implement** both modules (harvester exposes `parse_candidates(lines: list[str]) -> list[dict]` + `main(argv)`; marker constant shared by importing `MARKER` from `agent.fixture_candidates` — evaluation may depend on agent, never the reverse).
- [ ] **Step 4: Verify pass**, **Step 5: wire the two emit sites + runner reuse** (re-run `tests/test_routing_golden_set.py` to prove the runner refactor holds).
- [ ] **Step 6: Doc** — §5.3: "Production disagreements and gate failures emit `ROUTING_FIXTURE_CANDIDATE` lines; `evaluation/harvest_fixture_candidates.py` converts a pasted log into candidate fixtures (expected fields intentionally blank — a human labels before adoption)."
- [ ] **Step 7: Gate**, **Step 8: Commit** — `feat(eval): self-growing golden set via fixture-candidate emission`.

---

### Task 4: Conversational contract continuity, slice 1 (design item 2) — flag-gated OFF

**Files:**
- Create: `agent/contract_continuity.py`
- Create: `tests/test_contract_continuity.py`
- Modify: `utils/session_memory.py` (store/fetch last-contract snapshot on the session record, same TTL/cleanup as history)
- Modify: `config.py` (`ENABLE_CONTRACT_CONTINUITY`, default `"false"`, next to the other `ENABLE_*` flags)
- Modify: `models.py` (`previous_contract_snapshot: str = ""` on `QueryContext`, inputs region)
- Modify: `main.py` (fetch snapshot before `process_query` and pass via the new `QueryContext` field — `process_query` takes `query/conversation_history/trace_id/session_id`; add keyword `previous_contract_snapshot: str = ""` to `process_query` and set it on ctx at construction; store snapshot after success: `set_last_contract(session_id, continuity_snapshot_json(ctx))`)
- Modify: `agent/planner.py` `analyze_question` (~line 1850): pass `previous_contract=ctx.previous_contract_snapshot if ENABLE_CONTRACT_CONTINUITY else ""` into `llm_analyze_question`
- Modify: `core/llm.py` `llm_analyze_question` (line 2651): new kwarg `previous_contract: str = ""`; **include it in `cache_input`** (append `f"|prev={previous_contract}"`); thread into `_build_analyzer_prompt_blocks` (new parameter) which, when non-empty, inserts block `("TRUSTED_PREVIOUS_CONTRACT", guidance_text + snapshot)` immediately after `("UNTRUSTED_USER_QUESTION", ...)`
- Modify: `core/llm.py` `_ANALYZER_TRUNCATION_DATA` / `_ANALYZER_TRUNCATION_KNOWLEDGE` (~line 2596): insert `"TRUSTED_PREVIOUS_CONTRACT"` as the SECOND drop (right after `UNTRUSTED_CONVERSATION_HISTORY`) in both profiles — preserves every pinned invariant
- Modify: `docs/active/query_pipeline_architecture.md` (§3.2 new paragraph; new §5 entry with cutover criterion)

**Interfaces:**
- Produces: `agent.contract_continuity.continuity_snapshot_json(ctx) -> str` — compact JSON (`""` when no authoritative analysis) with keys: `query_type, answer_kind, render_style, preferred_path, top_tool, params_hint` (metric/currency/granularity/start_date/end_date/entities of candidate_tools[0], nulls dropped), `entity_scope`, `canonical_query_en`. Hard cap 2000 chars (truncate → `""` rather than emit partial JSON).
- Produces: `utils.session_memory.set_last_contract(session_id: str, snapshot_json: str) -> None`, `get_last_contract(session_id: str) -> str` (empty string when absent/expired).
- Block guidance text (verbatim, part of the block body):
  `"Previous turn's routed contract (trusted context from this session, not user input). If the new question is a follow-up or delta (e.g. 'and for 2023', 'same in USD'), interpret it relative to this contract. If the new question is self-contained, ignore this block."`

- [ ] **Step 1: Failing tests** — snapshot builder (authoritative ctx → expected keys; non-authoritative → `""`; >2000 chars → `""`); session-memory round trip + expiry (mirror existing session tests' monkeypatch of time); block insertion (`_build_analyzer_prompt_blocks(..., previous_contract="{}")` contains the tag right after `UNTRUSTED_USER_QUESTION`; absent when `""`); truncation-order test asserting the new section drops after history but before everything else in BOTH analyzer profiles; cache-key test (two `llm_analyze_question` calls differing only in `previous_contract` must produce different `cache_input` — assert via monkeypatched `llm_cache.get` capturing keys, mocked `_invoke_with_resilience`).
- [ ] **Step 2: Verify failure**, **Step 3–4: implement + pass** (respect the patched-symbol rule: `llm_analyze_question` internals resolve `_invoke_with_resilience` etc. through `core.llm` module globals — do not move anything).
- [ ] **Step 5: Wire main.py + planner + config flag** (flag OFF: injection suppressed; storage always on — harmless, enables instant flag flip).
- [ ] **Step 6: Doc** — §3.2: "Contract continuity (2026-07-07, flag-gated OFF)…" describing the trusted block, storage, and that history remains the untrusted fallback; new §5 entry: cutover = golden-set green including new follow-up fixtures + two weeks of `analyzer_cross_check_events` at-or-below baseline with the flag on in production.
- [ ] **Step 7: Add 3 follow-up fixtures** to `evaluation/routing_golden_set.json` (e.g. "and for 2023?" following r002's series question — note field documents the required previous contract; the runner does not yet chain turns, so mark these `"skip_unless_continuity": true` and teach `load_fixtures` to skip them unless `--include-continuity` is passed — small runner addition with its own fixture-contract test).
- [ ] **Step 8: Gate**, **Step 9: Commit** — `feat(analyzer): contract-continuity slice 1 (trusted previous-contract block, flag-gated off)`.

---

### Task 5: Evidence-ontology agreement shadow, slice 1 (design item 4)

**Files:**
- Modify: `agent/evidence_planner.py` — at the point where `threshold_share_with_prices` causes the planner RULE to add the prices step (the `_share_query_requests_price_context` consumers, ~lines 212/336): after the rule fires, compute analyzer agreement and log
- Modify: `utils/metrics.py` — `evidence_rule_agreement_events: dict[str, int]` + `log_evidence_rule_agreement(rule: str, agree: bool)` (increments `f"{rule}:{'agree' if agree else 'disagree'}"`)
- Modify: `skills/question-analyzer/` guidance (the tool/evidence-roles section): add one rule — threshold-share questions with price context should emit `get_prices` among `candidate_tools` (additive analyzer behavior; planner rule stays authoritative)
- Create: `tests/test_evidence_rule_agreement.py`
- Modify: `docs/active/query_pipeline_architecture.md` (new §5 entry: migration policy + cutover criterion)

**Interfaces:**
- Agreement predicate (implement inline where the rule fires):
```python
def _analyzer_emitted_price_context(qa) -> bool:
    if qa is None:
        return False
    tools = qa.tooling.candidate_tools or []
    return any(t.name == "get_prices" for t in tools[1:])
```
- Consumes: existing `threshold_share_with_prices` boolean and `ctx.question_analysis`.

- [ ] **Step 1: Failing tests** — build plans via `build_evidence_plan` with `_make_qa_payload`-style fixtures (threshold-share query + candidate_tools with/without a secondary `get_prices`); assert `metrics.evidence_rule_agreement_events` gets `"threshold_share_price_context:agree"` / `":disagree"`; assert plan content unchanged versus a pre-change golden copy of the same fixture (no behavior change).
- [ ] **Step 2–4: fail → implement → pass.**
- [ ] **Step 5: Skill-guidance edit** (one rule line in the question-analyzer skill file that lists tool-emission guidance — locate via `grep -rn "candidate_tools" skills/question-analyzer/`).
- [ ] **Step 6: Doc** — new §5 entry "Evidence-ontology consolidation (migration, shadow)": principle (planner becomes purely mechanical; contract owns roles), slice-1 rule, cutover = ≥95% agreement over 14 days of `evidence_rule_agreement_events`, then delete the planner rule in a dedicated session (golden set + disagreement review gate that change).
- [ ] **Step 7: Gate**, **Step 8: Commit** — `feat(planner): evidence-ontology agreement shadow (threshold-share slice)`.

---

### Task 6: Evidence-anomaly detection + flag-gated single re-analysis (design item 1) — flag OFF

Depends on: Task 2 (`period_bounds_from_hint`), Task 4 (trusted-block mechanism in `_build_analyzer_prompt_blocks` — reuse with tag `TRUSTED_EVIDENCE_ANOMALY`).

**Files:**
- Modify: `agent/pipeline.py` — new `_detect_evidence_anomaly(ctx) -> str | None` + hook in `process_query` immediately after `ctx = _execute_evidence_plan(ctx)`; new `_attempt_evidence_reanalysis(ctx, anomaly: str) -> QueryContext` (flag-gated)
- Modify: `config.py` — `ENABLE_EVIDENCE_REANALYSIS`, default `"false"`
- Modify: `models.py` — `evidence_anomaly: str = ""`, `reanalysis_attempted: bool = False` on `QueryContext`
- Modify: `core/llm.py` — `llm_analyze_question` gains `evidence_anomaly_note: str = ""` (same threading pattern as Task 4's `previous_contract`, same cache-key inclusion, block inserted after `TRUSTED_PREVIOUS_CONTRACT`; truncation lists updated the same way)
- Create: `tests/test_evidence_reanalysis.py`
- Modify: `docs/active/query_pipeline_architecture.md` (§2.3 tree note + §3.6 tail + new §5 entry)

**Interfaces:**
- `_detect_evidence_anomaly(ctx)` returns first matching tag or None:
  - `"primary_empty"`: `ctx.used_tool and not ctx.rows` and finalized `answer_kind` in `{scalar, list, timeseries, comparison, scenario, forecast}`
  - `"period_gap"`: reuse `period_bounds_from_hint` + `_df_date_span` from `agent/render_fitness.py` (import; same disjoint predicate as Task 2)
- Shadow always: `metrics.log_evidence_anomaly(tag)` (new counter dict, same pattern as Task 5) + `_emit_trace_stage(ctx, "stage_0_9_evidence_anomaly", t0, tag=tag)`.
- Gated retry (flag ON only, once per request): set `ctx.evidence_anomaly = tag`, `ctx.reanalysis_attempted = True`; re-run `planner.analyze_question_active(ctx)` with the anomaly note (`f"Prior execution anomaly: {tag}. The first interpretation produced no usable evidence for the interpreted period/tool; reconsider the interpretation."` threaded via the new kwarg), then `_finalize_answer_kind(ctx)`, rebuild the plan (`evidence_planner.build_evidence_plan`), clear per-attempt fields (`used_tool/tool_name/df/rows/cols/evidence_collected/evidence_plan` reset exactly as a fresh request would have them — enumerate each field in the implementation, resetting via the same defaults `QueryContext` declares), and call `_execute_evidence_plan(ctx)` once more. Emit `_emit_trace_stage(ctx, "stage_0_9_reanalysis", t0, anomaly=tag, tool_changed=..., kind_changed=...)`.

- [ ] **Step 1: Failing tests** (all with monkeypatched `planner.analyze_question_active`, `evidence_planner.build_evidence_plan`, and a fake `_execute_evidence_plan` counting invocations):
  - flag OFF + empty primary → anomaly counter incremented, `_execute_evidence_plan` called exactly once, ctx untouched;
  - flag ON + empty primary → analyzer re-run exactly once, executor called exactly twice, `reanalysis_attempted is True`;
  - flag ON + second anomaly after retry → NO third attempt;
  - no anomaly → no counter, no retry.
- [ ] **Step 2–4: fail → implement → pass.**
- [ ] **Step 5: Doc** — §2.3 tree gains one line under tool execution ("on evidence anomaly (flag-gated): one re-analysis with the anomaly as trusted context — still the same Stage 0.2 interpreter, §3.6"); §3.6 tail paragraph; §5 entry with enablement criterion (two weeks of `evidence_anomaly` counters to size the blast radius, then enable in production; golden set must stay green with flag on).
- [ ] **Step 6: Gate**, **Step 7: Commit** — `feat(pipeline): evidence-anomaly shadow + flag-gated single re-analysis`.

---

### Task 7: Architecture-doc coherence pass, memory, push

**Files:**
- Modify: `docs/active/query_pipeline_architecture.md`
- Modify: `C:\Users\Administrator\.claude\projects\D--Enaiapp\memory\project_architecture_audit_backlog.md`

- [ ] **Step 1:** Header "Last updated" → date + one-line scope; §1 executive summary sentence for the new surfaces (provenance block, shadow checks, flag-gated continuity/re-analysis); verify §2.3 tree matches the six additions; add the named design principle to §3.10: "Stage 0.2 emits visualization *intent*; Stage 5 owns *realization*."
- [ ] **Step 2:** Confirm every §5 entry added by Tasks 4–6 states its cutover criterion and owner action.
- [ ] **Step 3:** Full targeted suite one final time → expected: all green.
- [ ] **Step 4:** Update memory (slices landed, flags default-off inventory, cutover criteria).
- [ ] **Step 5:** Commit doc pass + `git push`.

---

## Self-Review (performed at write time)

- **Coverage:** all six review items map to Tasks 1–6; the seventh review note (intent-vs-realization principle) lands in Task 7. ✓
- **Placeholders:** Task 2 tests reference `_qa_with_hint` — defined by instruction to copy/extend `_make_qa_payload` within the new test file (explicit, not "similar to"); Task 5 skill-file edit includes its locating grep. Remaining code blocks are complete. ✓
- **Type consistency:** `routed_fields_snapshot(qa)` (Task 3) matches the runner refactor; `period_bounds_from_hint` (Task 2) matches Task 6's import; `previous_contract`/`evidence_anomaly_note` kwargs both follow the same threading + cache-key pattern. ✓
- **Sequencing:** 1→2→3 independent; 4 before 6 (block mechanism); 5 independent. Suite gate after every task. ✓
