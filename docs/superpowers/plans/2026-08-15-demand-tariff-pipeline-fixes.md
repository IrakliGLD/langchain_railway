# Demand-Tariff Pipeline Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make retail end-user tariff questions return correct, fully-shipped answers from `demand_tariff_mv` instead of wrong-view data or grounding-gutted stubs.

**Architecture:** Four defects surfaced in the 2026-08-15 production trace, plus four domain rules the model needs in order to answer retail-tariff questions correctly at all. Two defects (empty `stats_hint`, failed grounding) share one root cause that already has a tested fix in this repo which was never applied to the SQL path — so the fix is to call the existing helper at the frame boundary, not to patch consumers. The third is a routing defect requiring a deterministic guard. The fourth is demoted to measurement because the evidence does not yet establish it is a defect. The domain rules — company identities, category discipline, VAT basis, and the wholesale comparison basis — are encoded as content with tests that assert the load-bearing facts.

**Tech Stack:** Python 3.11 (container) / 3.14 (local), pandas, pytest, SQLAlchemy, FastAPI.

## Global Constraints

- Targeted suite must be green before each phase is considered done: `python -m pytest tests/ --ignore=tests/security -q` from repo root.
- `ruff check .` must pass.
- Baseline at plan authoring: 2,625 passed, 0 failures.
- TDD is mandatory for Phases 1 and 2 (production code): red, verify red, minimal green, verify green. Phases 3-5 change prompt/knowledge content; their tests assert observable consequences (grounding-corpus contents, example SQL shape, specific load-bearing facts), never the presence of a paragraph.
- Do not change `strict_numeric` grounding thresholds. The gate is behaving correctly; the corpus it reads is the defect.
- Local imports for `analysis.system_quantities` inside functions, matching the existing precedent at `agent/pipeline.py:1145` (circular-import caution).
- `D:\export_enai\_repo_sync` is **read-only reference**. Never edit it.
- `DB_SCHEMA_DOC` has a 9,000-char tripwire and sits at 7,583. Content goes in `knowledge/network_supply_tariffs.md` (selectively loaded) unless it is a one-line operational rule. Do not raise the cap to make room.

---

## Evidence Base

From the 2026-08-15 container trace, four requests:

| Req | Question | Path | Outcome |
| --- | --- | --- | --- |
| 2 | household consumer tariff dynamics | sql → `demand_tariff_mv` | 528 rows; grounding 5/15; shipped 273 of 1,587 chars |
| 3 | **distribution** tariff dynamics | tool → `get_tariffs` | answered with Enguri/Vardnili/Dzevrula **HPP** tariffs; grounding passed; shipped in full |
| 4 | Telmico tariff dynamics | sql → `demand_tariff_mv` | 528 rows; grounding 3/9; shipped 278 of 1,363 chars |

Confirmed mechanically (not inferred):

- `pd.DataFrame({'value':[Decimal('0.15289')]}).select_dtypes(include='number')` → `[]`. PostgreSQL `numeric` arrives as `Decimal`/object dtype.
- Four consumers rely on `select_dtypes(include="number")`: `agent/analyzer.py:3105`, `agent/analyzer.py:3433`, `agent/summary_grounding.py:196`, `analysis/seasonal_stats.py:56`.
- `analysis/system_quantities.coerce_decimal_columns_to_float` exists and its docstring names this exact failure chain, including *"stats_hint is just 'Rows: N'"*. `stats_hint_len=9` == `len("Rows: 528")`.
- `agent/pipeline.py:1141-1146` already calls that helper for driver-enrichment frames. The main SQL path does not.

---

## File Structure

| File | Responsibility | Phase |
| --- | --- | --- |
| `core/query_executor.py` | Builds the SQL result DataFrame. Single boundary where Decimal coercion belongs. | 1 |
| `tests/test_sql_decimal_coercion.py` (new) | Owns the reproduction + regression tests for numeric dtype on the SQL path. | 0, 1 |
| `agent/evidence_planner.py` | Deterministic evidence planning. Gains the retail-vs-generation guard. | 2 |
| `tests/test_evidence_planner.py` | Existing home for planner routing tests. | 2 |
| `contracts/question_analysis_catalogs.py` | Tool catalog the analyzer reads. `get_tariffs.avoid_for` gains retail. | 2 |
| `context.py` | `DB_SCHEMA_DOC` — terse operational rules only (unit, VAT basis, comparison formula). Budget-capped at 9,000 chars, currently 7,583. | 3, 5 |
| `knowledge/sql_example_selector.py` | `END_USER_PRICE_EXAMPLES` — published totals, category discipline, wholesale comparison. | 3, 4, 5 |
| `knowledge/network_supply_tariffs.md` | The rich domain content: company identities, service territories, VAT, comparison basis. Selectively loaded, so it carries the detail that will not fit in `DB_SCHEMA_DOC`. | 3, 4, 5 |
| `tests/test_sql_example_selector.py` | Existing home for example-content assertions. | 3, 4, 5 |
| `tests/test_context.py` | Existing home for `DB_SCHEMA_DOC` assertions and its size tripwire. | 3, 5 |
| `agent/tools/end_user_price_tools.py` (new) | The `get_end_user_prices` tool and the canonical category matrix. Single source of truth for the 8 categories. | 7 |
| `tests/test_end_user_price_tool.py` (new) | Tool contract: categories, VAT, benchmark, rejection of bad input. | 7 |
| `agent/tools/registry.py` | `TOOL_REGISTRY` gains the tool. | 7 |
| `agent/planner.py` | `resolve_tool_params` gains the `entity_scope` → `(supplier, category)` mapping. | 7 |
| `tests/test_network_supply_knowledge.py` (new) | Asserts the load-bearing domain facts, and that the knowledge file has not drifted from the tool's matrix. | 4, 5, 7 |

## Domain Rules (source of truth for Phases 3-5)

Verified read-only against `D:\export_enai\_repo_sync` on 2026-08-15. **That folder must not be edited.**

**Companies and roles.** Six actors, three activities. Never substitute one for another's component — they are separate legal entities.

| Code | Full name | Role | Territory |
| --- | --- | --- | --- |
| `gse` | Georgian State Electrosystem | TSO / transmission | national |
| `telasi` | Telasi | distribution | Tbilisi |
| `telmico` | Tbilisi Electricity Supply Company | supply | Tbilisi |
| `epg` | Energo-Pro Georgia | distribution | outside Tbilisi, plus some Tbilisi suburbs |
| `eps` | EP Georgia Supply | supply | outside Tbilisi, plus some Tbilisi suburbs |

Supplier→distributor pairing is fixed: `telmico`→`telasi`, `eps`→`epg`.

**VAT.** `demand_tariff_mv.value` and its `final_price` rows are **net of VAT**
(`networkSupplyChart.js:153` — "`net` is what the view publishes as final_price; VAT is
levied on top", `VAT_RATE = 0.18`). Decision: report net by default and say so; add the 18%
and give the gross total only when the question asks what a consumer actually pays.

**Units.** Tariffs are GEL/kWh; wholesale prices are GEL/MWh. The dashboard normalises
**downward** — every price is divided by `KWH_PER_MWH = 1000` so the comparison happens in
GEL/kWh (`networkSupplyWholesaleChart.js:16-20`). Never convert tariffs up to GEL/MWh.

**Wholesale comparison basis.** The published supply tariff already bundles the guaranteed
capacity charge, so the two sides only line up once it is accounted for. The dashboard adds it
to the **wholesale** side, not off the tariffs, "so the regulated lines show what is actually
charged" (`networkSupplyWholesaleChart.js:85-95`). Benchmark, per month:

```
(p_bal_gel + p_gcap_gel) / 1000   -- GEL/kWh, from public.price_with_usd
```

The dashboard also draws a lower band edge at `p_bal_gel × 0.95 + p_gcap_gel`. **Decision: do
not reproduce the band.** It is a chart-reading aid; prose uses the single line above.

**Category discipline.** Eight final-price categories, each existing for both suppliers (16
published prices per month). A component must be looked up with that category's own keys, and
rows 6 and 8 carry a real irregularity: the supply component is filed under
`level_2_cat = 'other'` while the matching distribution component has a **blank** one. Matching
`level_2_cat` uniformly across all three components silently drops the distribution row.

| # | volate | l1 | l2 | supply activity | distribution l2 |
| - | --- | --- | --- | --- | --- |
| 1 | `220/380` | `com` | `other` | `public` | `other` |
| 2 | `220/380` | `com` | `small` | `universal` | `small` |
| 3 | `220/380` | `hh` | `cat1` | `universal` | `cat1` |
| 4 | `220/380` | `hh` | `cat2` | `universal` | `cat2` |
| 5 | `220/380` | `hh` | `cat3` | `universal` | `cat3` |
| 6 | `3.3-6-10` | `com` | `other` | `public` | **`''`** |
| 7 | `3.3-6-10` | `hh` | `''` | `universal` | `''` |
| 8 | `35-110` | `com` | `other` | `public` | **`''`** |

---

## Phase 0: Reproduce the Production Symptom

**Goal:** One test that fails for the same reason production fails. Without this, later phases prove only that mechanisms changed, not that the symptom is gone.

### Task 0.1: Failing reproduction test

**Files:**
- Create: `tests/test_sql_decimal_coercion.py`

**Interfaces:**
- Consumes: `analysis.stats.quick_stats`, `agent.analyzer._append_column_aggregates`, `models.QueryContext`
- Produces: `make_demand_tariff_frame(rows=…)` helper reused by Tasks 1.2 and 1.3.

- [ ] **Step 1: Write the failing test**

```python
"""Regression tests for PostgreSQL numeric -> Decimal dtype on the SQL path.

PostgreSQL ``numeric`` columns arrive in pandas as ``Decimal`` objects with
``object`` dtype.  Every downstream consumer that uses
``select_dtypes(include="number")`` then silently skips them, leaving
``stats_hint`` at just "Rows: N" and the grounding corpus without aggregates.
Production trace 2026-08-15 (demand_tariff_mv, stats_hint_len=9).
"""
from decimal import Decimal

import pandas as pd


def make_demand_tariff_frame(rows: int = 6) -> pd.DataFrame:
    """A demand_tariff_mv-shaped frame with Decimal values, as psycopg returns."""
    return pd.DataFrame(
        {
            "date": [f"2026-0{(i % 6) + 1}-01" for i in range(rows)],
            "company": ["telmico"] * rows,
            "activity": ["universal"] * rows,
            "level_1_cat": ["hh"] * rows,
            "level_2_cat": ["cat2"] * rows,
            "value": [Decimal("0.15289") + Decimal(i) / 1000 for i in range(rows)],
        }
    )


def test_decimal_value_column_is_numeric_after_sql_execution():
    """The SQL path must hand downstream consumers a float column, not object.

    Fails before the fix: ``value`` is object dtype, so select_dtypes finds
    nothing and every aggregate consumer silently no-ops.
    """
    from core.query_executor import coerce_result_frame

    df = coerce_result_frame(make_demand_tariff_frame())

    assert "value" in df.select_dtypes(include="number").columns, (
        "Decimal column still invisible to select_dtypes(include='number')"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sql_decimal_coercion.py::test_decimal_value_column_is_numeric_after_sql_execution -v`
Expected: FAIL with `ImportError: cannot import name 'coerce_result_frame' from 'core.query_executor'`

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_sql_decimal_coercion.py
git commit -m "test: reproduce Decimal dtype loss on the SQL result path"
```

---

## Phase 1: Coerce Decimal at the SQL Frame Boundary

**Goal:** Fix Issues 2 and 3a at the root. One coercion at the boundary heals all four `select_dtypes` consumers.

**Why here and not at the consumers:** `agent/pipeline.py:1141` already establishes the boundary-coercion pattern with a comment naming the same trap. Patching four consumers would leave the fifth to be written later.

### Task 1.1: Add the boundary coercion

**Files:**
- Modify: `core/query_executor.py:192`
- Test: `tests/test_sql_decimal_coercion.py`

**Interfaces:**
- Consumes: `analysis.system_quantities.coerce_decimal_columns_to_float(df) -> tuple[pd.DataFrame, list[str]]`
- Produces: `core.query_executor.coerce_result_frame(df: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write minimal implementation**

```python
def coerce_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Make PostgreSQL ``numeric`` columns visible to numeric consumers.

    psycopg returns ``numeric`` as ``Decimal``, which pandas stores as object
    dtype.  ``select_dtypes(include="number")`` then skips those columns, so
    per-column aggregates, grounding tokens, seasonal stats and chart builders
    all silently see zero numeric columns.  Coerce once here, at the single
    boundary where SQL rows become a frame, rather than at each consumer.

    Mirrors the existing treatment of driver-enrichment frames in
    ``agent/pipeline.py``.
    """
    if df is None or df.empty:
        return df
    # Local import: analysis.* is imported lazily elsewhere to avoid cycles.
    from analysis.system_quantities import coerce_decimal_columns_to_float

    coerced, changed_columns = coerce_decimal_columns_to_float(df)
    if changed_columns:
        log.info(
            "Coerced %d Decimal column(s) to float on the SQL result frame",
            len(changed_columns),
        )
    return coerced
```

Then at line 192, replace `df = pd.DataFrame(rows, columns=cols)` with:

```python
        df = coerce_result_frame(pd.DataFrame(rows, columns=cols))
```

- [ ] **Step 2: Run the Phase 0 test to verify it passes**

Run: `python -m pytest tests/test_sql_decimal_coercion.py -v`
Expected: PASS

- [ ] **Step 3: Run the full targeted suite**

Run: `python -m pytest tests/ --ignore=tests/security -q`
Expected: 2,625 passed + 1 new = 2,626, 0 failures. **Any failure here is a real signal** — the SQL path now yields float where tests may have asserted Decimal/object. Investigate, do not paper over.

- [ ] **Step 4: Commit**

```bash
git add core/query_executor.py tests/test_sql_decimal_coercion.py
git commit -m "fix: coerce PostgreSQL Decimal columns to float on the SQL result frame"
```

### Task 1.2: Prove stats_hint recovers

**Files:**
- Modify: `tests/test_sql_decimal_coercion.py`

- [ ] **Step 1: Write the failing test**

```python
def test_column_aggregates_reach_stats_hint_for_a_decimal_frame():
    """stats_hint must carry aggregates, not just the row count.

    Production symptom: stats_hint_len=9, which is exactly len("Rows: 528").
    With no statistics the model computes its own, and strict_numeric
    grounding then rejects every computed figure.
    """
    from agent.analyzer import _append_column_aggregates
    from core.query_executor import coerce_result_frame
    from models import QueryContext

    ctx = QueryContext(query="household tariff dynamics")
    ctx.df = coerce_result_frame(make_demand_tariff_frame(rows=12))
    ctx.stats_hint = "Rows: 12"

    _append_column_aggregates(ctx)

    assert "Column Aggregates" in ctx.stats_hint
    assert len(ctx.stats_hint) > len("Rows: 12")
```

- [ ] **Step 2: Run to verify it passes** (Task 1.1 already made this pass)

Run: `python -m pytest tests/test_sql_decimal_coercion.py -v`
Expected: PASS. **If it fails, `QueryContext` construction differs — read `models.py` and fix the fixture, not the assertion.**

- [ ] **Step 3: Commit**

```bash
git add tests/test_sql_decimal_coercion.py
git commit -m "test: pin stats_hint aggregates for Decimal-valued SQL frames"
```

### Task 1.3: Prove the grounding corpus recovers

**Files:**
- Modify: `tests/test_sql_decimal_coercion.py`

- [ ] **Step 1: Write the test**

```python
def test_aggregate_tokens_reach_the_grounding_corpus():
    """agent/summary_grounding.py:196 has the same select_dtypes dependency.

    Without numeric dtypes it contributes no aggregate tokens, so an answer
    citing a mean or max cannot match and the gate strips it.
    """
    from agent.summary_grounding import _add_aggregate_tokens
    from core.query_executor import coerce_result_frame
    from models import QueryContext

    ctx = QueryContext(query="household tariff dynamics")
    ctx.df = coerce_result_frame(make_demand_tariff_frame(rows=12))

    tokens: set[str] = set()
    _add_aggregate_tokens(tokens, ctx)

    assert tokens, "no aggregate tokens produced for a Decimal-valued frame"
```

- [ ] **Step 2: Run to verify it passes**

Run: `python -m pytest tests/test_sql_decimal_coercion.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_sql_decimal_coercion.py
git commit -m "test: pin grounding aggregate tokens for Decimal-valued SQL frames"
```

---

## Phase 2: Stop `get_tariffs` Answering Retail Questions

**Goal:** Fix Issue 1 — the silent-wrong-answer defect. A distribution-tariff question must not be served generation-side plant tariffs.

**Mechanism:** The analyzer emitted `candidate_tools: ["get_tariffs"]` and `evidence_planner` honoured it, bypassing Stage 2. When the planner produces no steps, the pipeline falls through to SQL generation (observed in requests 2 and 4). So suppressing the step is sufficient — no new tool is required.

**Deliberately not in scope:** creating a `get_end_user_tariffs` tool. That is a larger change; the guard makes the wrong answer impossible first.

### Task 2.1: Guard against retail questions taking the generation tool

**Files:**
- Modify: `agent/evidence_planner.py` — inside `_expand_evidence_steps`, immediately before the primary step is constructed (~line 246-263)
- Test: `tests/test_evidence_planner.py`

**Interfaces (verified against the code, do not improvise):**
- Public entry is `build_evidence_plan(ctx: QueryContext) -> QueryContext`. It reads `ctx.question_analysis` and writes `ctx.evidence_plan` (a list of step dicts) and `ctx.evidence_plan_source`. It does **not** take `(qa, query)` and does **not** return a list.
- It delegates to `_expand_evidence_steps(qa, ctx.query) -> list[dict]`. The guard belongs there.
- Topics live at `qa.knowledge.candidate_topics`, a list of `TopicCandidate`; each item's name is `item.name`, a `KnowledgeTopicName` enum, so the string is `item.name.value`. There is **no** `qa.candidate_topics`.
- `qa.entity_scope` is a top-level `Optional[str]`.
- Existing fixtures: `_make_qa_payload(query_type=…, preferred_path=…, tools=…) -> dict` and `_ctx_with_qa(payload) -> QueryContext`. There is no `_make_qa`. `_make_qa_payload` does not expose topics or entity scope, so set them on the returned payload directly.
- Produces: `agent.evidence_planner._is_retail_tariff_question(qa) -> bool`

- [ ] **Step 1: Write the failing test**

```python
def test_retail_tariff_question_does_not_plan_the_generation_tariff_tool():
    """Production trace 2026-08-15 request 3.

    "How are distribution tariffs trending?" planned get_tariffs, which reads
    tariff_with_usd, and answered with Enguri/Vardnili/Dzevrula HPP tariffs at
    confidence 0.98 with the grounding gate satisfied -- a fluent, fully
    shipped, entirely wrong answer.
    """
    payload = _make_qa_payload(
        query_type="data_retrieval",
        preferred_path="tool",
        tools=[{"name": "get_tariffs", "score": 0.85, "reason": "tariff data"}],
    )
    payload["knowledge"] = {
        "candidate_topics": [{"name": "network_supply_tariffs", "score": 0.9}]
    }
    payload["entity_scope"] = "distribution_network_tariffs"

    ctx = build_evidence_plan(_ctx_with_qa(payload))

    assert [step["tool_name"] for step in ctx.evidence_plan] == [], (
        "retail tariff question still routed to the generation-side tool"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evidence_planner.py::test_retail_tariff_question_does_not_plan_the_generation_tariff_tool -v`
Expected: FAIL — `assert ['get_tariffs'] == []`

If it instead errors on payload validation, read `contracts/question_analysis.py` for the required shape of `knowledge.candidate_topics` and fix the fixture — never relax the contract to make a test pass.

- [ ] **Step 3: Write minimal implementation**

```python
def _is_retail_tariff_question(qa) -> bool:
    """True when the question is about end-user/network tariffs, not plant tariffs.

    ``get_tariffs`` reads ``tariff_with_usd`` -- what a regulated PLANT is paid.
    Retail questions need ``demand_tariff_mv`` via the SQL path.  Requiring the
    retail topic AND the absence of the generation topic keeps a genuine
    generation question (which may mention both) on the tool.
    """
    topics = {
        candidate.name.value
        for candidate in (qa.knowledge.candidate_topics or [])
    }
    return (
        KnowledgeTopicName.NETWORK_SUPPLY_TARIFFS.value in topics
        and KnowledgeTopicName.TARIFFS.value not in topics
    )
```

`KnowledgeTopicName` is already imported in this module; confirm before adding an import.

Then guard the primary step inside `_expand_evidence_steps`, immediately before `primary_step` is constructed:

```python
    if top_name == ToolName.GET_TARIFFS.value and _is_retail_tariff_question(qa):
        log.info(
            "Evidence plan: suppressing get_tariffs for a retail-tariff question; "
            "falling through to SQL against demand_tariff_mv"
        )
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evidence_planner.py::test_retail_tariff_question_does_not_plan_the_generation_tariff_tool -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/evidence_planner.py tests/test_evidence_planner.py
git commit -m "fix: keep retail tariff questions off the generation-side tariff tool"
```

### Task 2.2: Prove the guard does not over-block

**Files:**
- Modify: `tests/test_evidence_planner.py`

- [ ] **Step 1: Write both negative tests**

```python
def _tariff_payload(topics: list[str]) -> dict:
    payload = _make_qa_payload(
        query_type="data_retrieval",
        preferred_path="tool",
        tools=[{"name": "get_tariffs", "score": 0.85, "reason": "tariff data"}],
    )
    payload["knowledge"] = {
        "candidate_topics": [{"name": name, "score": 0.9} for name in topics]
    }
    payload["entity_scope"] = "regulated_plants"
    return payload


def test_generation_tariff_question_still_plans_the_tariff_tool():
    """Regression guard: the fix must not starve plant-tariff questions."""
    ctx = build_evidence_plan(_ctx_with_qa(_tariff_payload(["tariffs"])))

    assert "get_tariffs" in [step["tool_name"] for step in ctx.evidence_plan]


def test_question_naming_both_tariff_topics_keeps_the_generation_tool():
    """Ambiguous questions keep current behaviour rather than losing evidence.

    Suppression requires the retail topic AND the absence of the generation
    topic, so a question the analyzer tagged with both is left alone.
    """
    ctx = build_evidence_plan(
        _ctx_with_qa(_tariff_payload(["tariffs", "network_supply_tariffs"]))
    )

    assert "get_tariffs" in [step["tool_name"] for step in ctx.evidence_plan]


def test_retail_guard_leaves_non_tariff_tools_alone():
    """The guard keys on the tool, not just the topic."""
    payload = _make_qa_payload(
        query_type="data_retrieval",
        preferred_path="tool",
        tools=[{"name": "get_prices", "score": 0.9, "reason": "price data"}],
    )
    payload["knowledge"] = {
        "candidate_topics": [{"name": "network_supply_tariffs", "score": 0.9}]
    }

    ctx = build_evidence_plan(_ctx_with_qa(payload))

    assert "get_prices" in [step["tool_name"] for step in ctx.evidence_plan]
```

- [ ] **Step 2: Run to verify both pass**

Run: `python -m pytest tests/test_evidence_planner.py -k tariff -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_evidence_planner.py
git commit -m "test: pin that the retail guard leaves generation tariff routing intact"
```

### Task 2.3: Tell the analyzer, so the guard is a backstop not the only defence

**Files:**
- Modify: `contracts/question_analysis_catalogs.py` (`get_tariffs` entry, ~line 236-240)
- Test: `tests/test_question_analysis_catalogs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_generation_tariff_tool_warns_against_retail_questions():
    from contracts.question_analysis_catalogs import QUESTION_ANALYSIS_TOOL_CATALOG

    entry = _entry(QUESTION_ANALYSIS_TOOL_CATALOG, "get_tariffs")

    assert "end-user" in entry["avoid_for"].lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_question_analysis_catalogs.py::test_generation_tariff_tool_warns_against_retail_questions -v`
Expected: FAIL — `avoid_for` currently reads "Balancing price questions, conceptual definitions, generation mix questions, and policy/status questions about liberalization or deregulation."

The constant is `QUESTION_ANALYSIS_TOOL_CATALOG` (`contracts/question_analysis_catalogs.py:208`) and the `get_tariffs` entry begins at line 236. `_entry` already exists in this test module.

- [ ] **Step 3: Write minimal implementation**

Append to `get_tariffs.avoid_for`:

```
 Also avoid for end-user, retail, distribution, transmission or supply tariffs -- those are network_supply_tariffs and are served from demand_tariff_mv, not this tool.
```

- [ ] **Step 4: Run to verify it passes, then check the prompt budget**

Run: `python -m pytest tests/test_question_analysis_catalogs.py -v`
Expected: PASS

Then, because the analyzer prompt is already over budget in 36/40 variants:

```bash
python -c "import tests.test_analyzer_prompt_order as t; print(sum(1 for v in t._budgeted_prompts().values() if '[truncated]' in v))"
```
Expected: still `36`. If it rises, shorten the wording rather than accepting the regression.

- [ ] **Step 5: Regenerate the golden fixture and commit**

```bash
python -c "import tests.test_analyzer_prompt_order as t; t.regenerate()"
python -m pytest tests/ --ignore=tests/security -q
git add contracts/question_analysis_catalogs.py tests/test_question_analysis_catalogs.py tests/fixtures/analyzer_prompt_legacy_hashes.json
git commit -m "fix: steer the analyzer away from the generation tariff tool for retail questions"
```

---

## Phase 3: Report End-User Totals from Published Rows, in Stored Units

**Goal:** Fix Issue 3b. Two ungroundable habits: converting GEL/kWh to GEL/MWh (×1000 — `0.15289` reported as `152.89`), and summing three components into a total that appears in no row.

**Anchor:** `demand_tariff_mv` already contains `activity = 'final_price'` rows — the regulator's own published total. A total quoted from that row is grounded by construction. This replaces prose-only guidance with a testable consequence.

### Task 3.1: Make the SQL examples prefer the published total

**Files:**
- Modify: `knowledge/sql_example_selector.py` (`END_USER_PRICE_EXAMPLES`)
- Test: `tests/test_sql_example_selector.py`

- [ ] **Step 1: Write the failing test**

```python
def test_end_user_examples_prefer_the_published_total_over_a_computed_sum():
    """A summed total exists in no row, so strict_numeric grounding strips it.

    The final_price row is the regulator's own total and IS in the frame, so
    quoting it is grounded by construction.
    """
    sql = _sql_blocks(END_USER_PRICE_EXAMPLES, keep_comments=True)

    assert "final_price" in sql
    assert "do not sum" in sql.lower() or "rather than re-summing" in sql.lower()


def test_end_user_examples_do_not_convert_to_per_mwh():
    """demand_tariff_mv is GEL/kWh. Converting produces numbers absent from
    the frame -- production trace showed 152.89/186.89/224.89, i.e. the
    GEL/kWh values times 1000."""
    sql = _sql_blocks(END_USER_PRICE_EXAMPLES)

    assert "* 1000" not in sql and "*1000" not in sql
    assert "gel_mwh" not in sql.lower()
```

- [ ] **Step 2: Run to verify the first fails**

Run: `python -m pytest tests/test_sql_example_selector.py -k end_user -v`
Expected: `test_end_user_examples_prefer_the_published_total_over_a_computed_sum` FAILS (no such comment yet); the second passes already.

- [ ] **Step 3: Write minimal implementation**

In `EXAMPLE 11.1`, change the reconciliation comment to state the preference explicitly, and reorder so the published total is the reported figure:

```sql
-- Report the published final_price row as the total. Do not sum the three
-- components for the headline number: a computed sum appears in no row of
-- the frame, so the grounding gate strips it from the answer. The component
-- SUMs below are for the breakdown and as a cross-check only.
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sql_example_selector.py -k end_user -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add knowledge/sql_example_selector.py tests/test_sql_example_selector.py
git commit -m "fix: report end-user totals from the published final_price row"
```

### Task 3.2: State the reporting unit in the schema doc

**Files:**
- Modify: `context.py` (`DB_SCHEMA_DOC`, demand_tariff_mv block)
- Test: `tests/test_context.py`

- [ ] **Step 1: Write the failing test**

```python
    def test_schema_doc_forbids_converting_end_user_tariffs_to_per_mwh(self):
        """Production trace: 0.15289 GEL/kWh was reported as 152.89, i.e.
        converted to GEL/MWh to match the convention used by every other
        view. The converted value is in no row, so grounding strips it."""
        from context import DB_SCHEMA_DOC

        assert "Report demand_tariff_mv values in GEL/kWh as stored" in DB_SCHEMA_DOC
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_context.py -k per_mwh -v`
Expected: FAIL — string absent

- [ ] **Step 3: Write minimal implementation**

Add to the `demand_tariff_mv` block of `DB_SCHEMA_DOC`:

```
- Report demand_tariff_mv values in GEL/kWh as stored. Do NOT convert to GEL/MWh: the
  converted number exists in no row and will be stripped by the grounding gate.
```

- [ ] **Step 4: Run to verify it passes, and that the size tripwire holds**

Run: `python -m pytest tests/test_context.py -v`
Expected: PASS, including `test_schema_doc_stays_within_a_sane_prompt_budget` (cap 9,000 chars).

- [ ] **Step 5: Commit**

```bash
git add context.py tests/test_context.py
git commit -m "fix: pin GEL/kWh as the reporting unit for end-user tariffs"
```

### Task 3.3: Correct the VAT claim and state the reporting rules in the knowledge file

**Files:**
- Modify: `knowledge/network_supply_tariffs.md` (Data Mapping → Critical usage notes, and Verification status)

**Why this task exists:** the committed knowledge file currently says the VAT treatment
"is not determinable from the data alone" and lists it as open item 1. That is now settled and
**wrong as written** — leaving it would have the model hedge on a question it can answer.

- [ ] **Step 1: Add the reporting rules**

```markdown
- **Report values in GEL/kWh, as stored.** Do not convert tariffs up to GEL/MWh. The
  converted figure appears in no row of the view, so the grounding gate strips it and the
  answer ships truncated. When a comparison to wholesale prices is required, convert the
  *price* down instead (see "Comparing to the wholesale price").
- **Quote `final_price` for the total.** A summed three-component total exists in no row.
  Use the components for the breakdown and `final_price` for the headline number.
- **`value` and `final_price` are NET of VAT.** VAT is 18% and is levied on top. Report the
  net figure by default and say it is net of VAT; give `net × 1.18` only when the question
  asks what a consumer actually pays.
```

- [ ] **Step 2: Replace open item 1 in the Verification status section**

Delete the "VAT treatment" open item and replace it with a settled statement:

```markdown
1. ~~VAT treatment~~ **Settled 2026-08-15.** The view stores tariffs **net of VAT**; the
   dashboard levies 18% on top of the published `final_price`. Earlier drafts of this file
   recorded this as undeterminable — that was wrong.
```

Renumber the remaining open items.

- [ ] **Step 3: Run the full targeted suite**

Run: `python -m pytest tests/ --ignore=tests/security -q`
Expected: 0 failures

- [ ] **Step 4: Commit**

```bash
git add knowledge/network_supply_tariffs.md
git commit -m "docs: settle the VAT basis and state end-user reporting rules"
```

---

## Phase 4: Company Identities and Category Discipline

**Goal:** Address notes 2 and 3. The model must name the right company, not claim the wrong
territory, and never build a final price by mixing categories or substituting one company's
component for another's.

**Testing note:** this phase changes retrieved content, not control flow. Tests assert that the
specific load-bearing facts are present and that the category table is internally consistent —
never that a paragraph exists.

### Task 4.1: Encode company identities and territories

**Files:**
- Modify: `knowledge/network_supply_tariffs.md` (§ Structure of the end-user price)
- Test: `tests/test_network_supply_knowledge.py` (new)

- [ ] **Step 1: Write the failing test**

```python
"""Content contract for the retail-tariff knowledge topic.

These assert load-bearing FACTS, not prose. Each one maps to a way an answer
would be wrong: naming the wrong company, claiming the wrong territory, or
mixing two categories when assembling a final price.
"""
import pathlib

KNOWLEDGE = (
    pathlib.Path(__file__).resolve().parents[1] / "knowledge" / "network_supply_tariffs.md"
).read_text(encoding="utf-8")


def test_every_company_code_has_its_full_legal_name():
    for code, name in [
        ("gse", "Georgian State Electrosystem"),
        ("telmico", "Tbilisi Electricity Supply Company"),
        ("eps", "EP Georgia Supply"),
        ("epg", "Energo-Pro Georgia"),
    ]:
        assert name in KNOWLEDGE, f"{code} is missing its full name ({name})"


def test_service_territories_are_stated():
    assert "Tbilisi" in KNOWLEDGE
    assert "suburb" in KNOWLEDGE.lower(), (
        "EPG/EPS also serve some Tbilisi suburbs; without this the model will "
        "claim Telasi/Telmico serve Tbilisi exclusively"
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_network_supply_knowledge.py -v`
Expected: FAIL — the full legal names are not in the file.

- [ ] **Step 3: Write minimal implementation**

Replace the company table in the knowledge file with:

```markdown
| Component | Code | Full name | Role | Territory |
| ------------- | --------- | --------- | ---- | --------- |
| Transmission | `gse` | Georgian State Electrosystem | TSO, transmission network | national |
| Distribution | `telasi` | Telasi | distribution network | Tbilisi |
| Supply | `telmico` | Tbilisi Electricity Supply Company | supply service | Tbilisi |
| Distribution | `epg` | Energo-Pro Georgia | distribution network | outside Tbilisi, plus some Tbilisi suburbs |
| Supply | `eps` | EP Georgia Supply | supply service | outside Tbilisi, plus some Tbilisi suburbs |

Telasi and Telmico operate in Tbilisi, the capital. Energo-Pro Georgia and EP Georgia Supply
operate across the rest of the country **and also cover some suburbs of Tbilisi** — so
"Tbilisi" alone does not determine the supplier.

The distributor and the supplier on the same network are **different legal entities**. Never
use one in place of the other when assembling a price.
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_network_supply_knowledge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add knowledge/network_supply_tariffs.md tests/test_network_supply_knowledge.py
git commit -m "docs: name the network and supply companies and their territories"
```

### Task 4.2: Pin the eight categories against mixing

**Files:**
- Modify: `knowledge/network_supply_tariffs.md` (§ Consumer categories)
- Test: `tests/test_network_supply_knowledge.py`

- [ ] **Step 1: Write the failing test**

```python
def test_category_table_flags_the_level_2_irregularity():
    """Rows 6 and 8 file the supply component under 'other' while the matching
    distribution component has a blank level_2_cat. Matching level_2_cat
    uniformly across all three components silently drops the distribution row
    and produces an incomplete price."""
    assert "level_2_cat" in KNOWLEDGE
    lowered = KNOWLEDGE.lower()
    assert "irregular" in lowered or "mismatch" in lowered


def test_knowledge_forbids_mixing_categories():
    lowered = KNOWLEDGE.lower()
    assert "never mix" in lowered or "do not mix" in lowered, (
        "the file must state that components from different categories cannot "
        "be combined into one final price"
    )
```

- [ ] **Step 2: Run to verify `test_knowledge_forbids_mixing_categories` fails**

Run: `python -m pytest tests/test_network_supply_knowledge.py -v`
Expected: the mixing test FAILS; the irregularity test already passes.

- [ ] **Step 3: Write minimal implementation**

Add beneath the category table:

```markdown
**Never mix categories.** A final end-user price is assembled from three components that all
belong to the *same* `(supplier, volate, level_1_cat, level_2_cat)` category, plus the single
national transmission row. Taking the distribution component from one category and the supply
component from another produces a number that corresponds to no real tariff. There are 8
categories × 2 suppliers = 16 published prices per month; each is self-contained.
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_network_supply_knowledge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add knowledge/network_supply_tariffs.md tests/test_network_supply_knowledge.py
git commit -m "docs: forbid mixing categories when assembling an end-user price"
```

---

## Phase 5: Compare End-User Price to the Wholesale Price

**Goal:** Address note 4. The comparison is only valid once the guaranteed capacity charge is
accounted for, and both sides must be in GEL/kWh.

**Grounding constraint:** the benchmark `(p_bal_gel + p_gcap_gel) / 1000` is a **derived**
value present in no row of either view. Unless it is serialized into `stats_hint`, the
strict_numeric gate strips it — the exact failure Phase 1 fixes for column aggregates. Task 5.2
is therefore not optional garnish; without it the comparison answer ships truncated.

### Task 5.1: Add the comparison SQL example

**Files:**
- Modify: `knowledge/sql_example_selector.py` (`END_USER_PRICE_EXAMPLES`)
- Test: `tests/test_sql_example_selector.py`

- [ ] **Step 1: Write the failing test**

```python
def test_end_user_examples_cover_the_wholesale_comparison():
    """The supply tariff bundles the guaranteed capacity charge, so a bare
    balancing price is not comparable to an end-user price."""
    sql = _sql_blocks(END_USER_PRICE_EXAMPLES, keep_comments=True)

    assert "p_gcap_gel" in sql, "comparison must add the guaranteed capacity charge"
    assert "p_bal_gel" in sql
    assert "1000" in sql, "wholesale prices must be converted down to GEL/kWh"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_sql_example_selector.py -k wholesale -v`
Expected: FAIL — no example references `p_gcap_gel`.

- [ ] **Step 3: Write minimal implementation**

Append to `END_USER_PRICE_EXAMPLES`:

```python
EXAMPLE 11.4 - End-User Price vs Wholesale:
Query: "How does the household end-user price compare with the wholesale price?"
Plan:
{
  "intent": "comparison",
  "target": "end_user_vs_wholesale",
  "period": "recent"
}
---SQL---
-- The published supply tariff already bundles the guaranteed capacity charge, so
-- the two sides only line up once it is added to the WHOLESALE side. Adding it there
-- (rather than subtracting it from the tariff) keeps the regulated figure equal to
-- what is actually charged.
-- Prices are GEL/MWh and tariffs are GEL/kWh: divide the price by 1000, never
-- multiply the tariff by 1000.
SELECT
    d.date,
    d.value                                          AS end_user_gel_kwh,
    (p.p_bal_gel + p.p_gcap_gel) / 1000.0            AS wholesale_benchmark_gel_kwh,
    d.value - (p.p_bal_gel + p.p_gcap_gel) / 1000.0  AS spread_gel_kwh
FROM demand_tariff_mv d
JOIN price_with_usd p ON p.date = d.date
WHERE d.activity = 'final_price'
  AND d.company = 'telmico'
  AND d.volate = '220/380'
  AND d.level_1_cat = 'hh'
  AND d.level_2_cat = 'cat2'
  AND d.date <= (SELECT MAX(date) FROM demand_tariff_mv WHERE activity = 'final_price')
ORDER BY d.date;
"""
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sql_example_selector.py -k "wholesale or end_user" -v`
Expected: PASS. The existing `test_end_user_examples_do_not_convert_to_per_mwh` must still
pass — the example divides the price, it never multiplies the tariff.

- [ ] **Step 5: Commit**

```bash
git add knowledge/sql_example_selector.py tests/test_sql_example_selector.py
git commit -m "feat: add the end-user vs wholesale comparison SQL example"
```

### Task 5.2: Make derived comparison values groundable

**Files:**
- Modify: `knowledge/network_supply_tariffs.md` (new § Comparing to the wholesale price)
- Test: `tests/test_network_supply_knowledge.py`

- [ ] **Step 1: Write the failing test**

```python
def test_knowledge_states_the_wholesale_comparison_basis():
    assert "p_gcap_gel" in KNOWLEDGE
    assert "guaranteed capacity" in KNOWLEDGE.lower()
    lowered = KNOWLEDGE.lower()
    assert "added to the wholesale" in lowered or "add it to the balancing" in lowered
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_network_supply_knowledge.py -k wholesale -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```markdown
## Comparing to the wholesale price

The regulated supply tariff already bundles the guaranteed capacity fee, so a bare balancing
price is **not** comparable to an end-user price. Add the capacity charge to the wholesale
side rather than subtracting it from the tariff — that way the regulated figure stays equal to
what is actually charged and the adjustment sits on one series.

Benchmark, per month, from `public.price_with_usd`:

    (p_bal_gel + p_gcap_gel) / 1000   -- GEL/kWh

Both prices are GEL/MWh, so divide by 1000 to reach the tariff's unit. Never multiply the
tariff by 1000 instead: the resulting figure appears in no row and the grounding gate will
strip it from the answer.

Compare against the **net** `final_price` (not the VAT-inclusive figure) — the wholesale price
is itself net of VAT, so comparing gross to net overstates the spread by 18%.
```

- [ ] **Step 4: Run to verify it passes, and check the schema-doc budget**

Run: `python -m pytest tests/test_network_supply_knowledge.py tests/test_context.py -v`
Expected: PASS, including the 9,000-char `DB_SCHEMA_DOC` tripwire. **If `DB_SCHEMA_DOC` is
near the cap, keep this content in the knowledge file and add only the one-line formula to the
schema doc — do not raise the cap to make room.**

- [ ] **Step 5: Commit**

```bash
git add knowledge/network_supply_tariffs.md tests/test_network_supply_knowledge.py
git commit -m "docs: state the wholesale comparison basis for end-user prices"
```

### Task 5.3: Add the one-line comparison rule to the schema doc

**Files:**
- Modify: `context.py` (`DB_SCHEMA_DOC`, demand_tariff_mv block)
- Test: `tests/test_context.py`

- [ ] **Step 1: Write the failing test**

```python
    def test_schema_doc_states_the_wholesale_comparison_basis(self):
        """Without this the model compares a bare balancing price to a tariff
        that already bundles the guaranteed capacity charge."""
        from context import DB_SCHEMA_DOC

        assert "(p_bal_gel + p_gcap_gel) / 1000" in DB_SCHEMA_DOC
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_context.py -k wholesale_comparison -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Add exactly two lines to the `demand_tariff_mv` block:

```
- value/final_price are NET of VAT (18% is levied on top). Report net by default.
- To compare with wholesale: benchmark = (p_bal_gel + p_gcap_gel) / 1000 GEL/kWh, joining
  price_with_usd on date. The supply tariff already bundles the capacity fee.
```

- [ ] **Step 4: Run to verify it passes and the tripwire holds**

Run: `python -m pytest tests/test_context.py -v`
Expected: PASS, including `test_schema_doc_stays_within_a_sane_prompt_budget`.

- [ ] **Step 5: Commit**

```bash
git add context.py tests/test_context.py
git commit -m "feat: state the VAT basis and wholesale comparison formula in the schema doc"
```

---

## Phase 6: Measure Before Changing SQL Generation

**Goal:** Establish whether Issue 4 is a defect at all. Requests 2 and 4 each returned exactly 528 rows (66 months × 8 series) with `has_period=False`. That may be a *complete* category set, which is legitimate for a "how are tariffs trending" question — or it may be the model failing to narrow.

**Deliberately not doing:** adding category or period filters to SQL generation. Changing generation on a hunch risks narrowing legitimately broad questions. Measure first.

### Task 6.1: Log the shape of demand_tariff_mv result frames

**Files:**
- Modify: `agent/sql_executor.py` — call the new helper where `ctx.df = df` is set on the SQL path (line 437)
- Test: `tests/test_sql_decimal_coercion.py`

**Interfaces (verified):**
- `agent/sql_executor.py` already has a module logger: `log = logging.getLogger("Enai")` at line 25. Do not create another.
- Produces: `agent.sql_executor.log_result_frame_shape(df: pd.DataFrame, table: str = "") -> None`

- [ ] **Step 1: Write the failing test**

```python
def test_wide_result_frames_log_their_dimension_cardinality(caplog):
    """Diagnostic: distinguish 'complete category set' from 'failed to narrow'.

    Both are 528 rows; only the per-dimension cardinality tells them apart.
    """
    import logging

    from agent.sql_executor import log_result_frame_shape

    df = make_demand_tariff_frame(rows=12)
    with caplog.at_level(logging.INFO, logger="Enai"):
        log_result_frame_shape(df, table="demand_tariff_mv")

    assert "distinct_company=1" in caplog.text
    assert "distinct_level_2_cat=1" in caplog.text
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_sql_decimal_coercion.py -k dimension_cardinality -v`
Expected: FAIL with `ImportError: cannot import name 'log_result_frame_shape'`

- [ ] **Step 3: Write minimal implementation**

```python
_SHAPE_DIMENSIONS = ("company", "activity", "volate", "level_1_cat", "level_2_cat", "entity", "technology")


def log_result_frame_shape(df, table: str = "") -> None:
    """Log per-dimension cardinality of a result frame.

    A 528-row frame is either a complete category set or a query that failed
    to narrow.  Row count alone cannot distinguish them; distinct counts per
    dimension can.
    """
    if df is None or df.empty:
        return
    parts = [
        f"distinct_{col}={df[col].nunique()}"
        for col in _SHAPE_DIMENSIONS
        if col in df.columns
    ]
    if parts:
        log.info("result_frame_shape table=%s rows=%d %s", table, len(df), " ".join(parts))
```

Call it where `ctx.df = df` is set on the SQL path.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sql_decimal_coercion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/sql_executor.py tests/test_sql_decimal_coercion.py
git commit -m "feat: log per-dimension cardinality of SQL result frames"
```

### Task 6.2: Decide with data

- [ ] **Step 1:** Deploy Phases 1-4 and re-run the four trace queries.
- [ ] **Step 2:** Read `result_frame_shape` for the retail queries. If `distinct_level_2_cat` equals the full vocabulary for a question naming one band, the model is failing to narrow — open a follow-up for SQL-generation guidance. If it matches the question's scope, close Issue 4 as not-a-defect.
- [ ] **Step 3:** Record the outcome in this plan file and commit.

---

## Phase 7: The `get_end_user_prices` Typed Tool

**Goal:** Replace Phase 2's *suppression* with a *destination*. A typed tool encodes the
category matrix, supplier pairing, VAT and the wholesale benchmark in code rather than in
prompt text, and returns derived values as real columns — so they are grounded by construction
instead of relying on the model to compute them and the gate to tolerate it.

**What this supersedes.** Once this phase lands:
- Phase 2's guard stops being "return `[]`" and becomes "route to `get_end_user_prices`". Its
  three tests carry over unchanged — they assert `get_tariffs` is not chosen, which stays true.
- Phase 5's grounding contingency is unnecessary: the benchmark is a returned column.
- Phase 6's narrowing question is moot: the tool takes explicit params.

**Naming.** `get_end_user_prices`, chosen deliberately over `get_demand_tariffs` /
`get_demand_prices`. Two tools one token apart is what produced Issue 1 in the first place;
this name is distinct from both `get_tariffs` (generation) and `get_prices` (wholesale).

**Verified contracts (do not improvise):**
- `ToolResult = Tuple[pd.DataFrame, List[str], List[tuple]]` (`agent/tools/types.py`).
- Tools are plain keyword-argument functions; see `get_tariffs` at `agent/tools/tariff_tools.py:170`.
- Helpers available from `agent/tools/common.py`: `normalize_date`, `normalize_limit`,
  `get_sort_direction`, `run_text_query`, `run_statement`, `last_day_of_month`.
- `TOOL_REGISTRY` is a flat `dict[str, Callable]` in `agent/tools/registry.py`.
- `resolve_tool_params(qa, tool_name, raw_query, *, hint=None) -> Optional[dict]` lives at
  `agent/planner.py:982` and is shared by the router and the evidence planner. Returning
  `None` means "unknown tool"; raise `ValueError` for an unresolvable reference.

### Task 7.1: The category matrix as one source of truth

**Files:**
- Create: `agent/tools/end_user_price_tools.py`
- Test: `tests/test_end_user_price_tool.py` (new)

**Interfaces:**
- Produces: `END_USER_CATEGORIES: tuple[EndUserCategory, ...]`, `SUPPLIER_TO_DISTRIBUTOR: dict[str, str]`, `TRANSMISSION_ROW: dict[str, str]`, `category_id(volate, level_1_cat, level_2_cat) -> str`

- [ ] **Step 1: Write the failing test**

```python
"""Contract for the end-user price tool.

The eight categories are ported from the company_mapping/category_mapping CTEs
of public.demand_tariff_mv. Rows 6 and 8 carry a real irregularity: the supply
component is filed under level_2_cat 'other' while the matching distribution
component has a blank one. Matching level_2_cat uniformly across all three
components silently drops the distribution row.
"""


def test_eight_categories_exist_for_both_suppliers():
    from agent.tools.end_user_price_tools import END_USER_CATEGORIES, SUPPLIER_TO_DISTRIBUTOR

    assert len(END_USER_CATEGORIES) == 8
    assert SUPPLIER_TO_DISTRIBUTOR == {"telmico": "telasi", "eps": "epg"}


def test_commercial_medium_and_high_voltage_use_a_blank_distribution_subclass():
    """Categories 6 and 8. Getting this wrong yields an incomplete price."""
    from agent.tools.end_user_price_tools import END_USER_CATEGORIES

    by_id = {category.id: category for category in END_USER_CATEGORIES}

    for category_id in ("3.3-6-10|com|other", "35-110|com|other"):
        category = by_id[category_id]
        assert category.supply_level_2 == "other"
        assert category.distribution_level_2 == ""


def test_household_categories_match_on_all_three_components():
    from agent.tools.end_user_price_tools import END_USER_CATEGORIES

    by_id = {category.id: category for category in END_USER_CATEGORIES}
    category = by_id["220/380|hh|cat2"]

    assert category.supply_activity == "universal"
    assert category.supply_level_2 == "cat2"
    assert category.distribution_level_2 == "cat2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_end_user_price_tool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.tools.end_user_price_tools'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Typed retrieval tool for regulated end-user (retail) electricity prices.

Reads ``public.demand_tariff_mv``.  The category matrix below is ported from
that view's own ``company_mapping`` / ``category_mapping`` CTEs -- if the view
is redefined these must move with it.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class EndUserCategory:
    """One final-price category and where its two variable components live."""

    id: str
    label: str
    volate: str
    level_1_cat: str
    level_2_cat: str
    supply_activity: str
    supply_level_2: str
    distribution_level_2: str


def category_id(volate: str, level_1_cat: str, level_2_cat: str) -> str:
    return f"{volate}|{level_1_cat}|{level_2_cat}"


SUPPLIER_TO_DISTRIBUTOR: Dict[str, str] = {"telmico": "telasi", "eps": "epg"}

TRANSMISSION_ROW = {
    "company": "gse",
    "activity": "transmission",
    "volate": "",
    "level_1_cat": "",
    "level_2_cat": "",
}

VAT_RATE = 0.18

END_USER_CATEGORIES: Tuple[EndUserCategory, ...] = (
    EndUserCategory("220/380|com|other", "Commercial - other (220/380)", "220/380", "com", "other", "public", "other", "other"),
    EndUserCategory("220/380|com|small", "Commercial - small (220/380)", "220/380", "com", "small", "universal", "small", "small"),
    EndUserCategory("220/380|hh|cat1", "Household cat 1, <=101 kWh (220/380)", "220/380", "hh", "cat1", "universal", "cat1", "cat1"),
    EndUserCategory("220/380|hh|cat2", "Household cat 2, 101-301 kWh (220/380)", "220/380", "hh", "cat2", "universal", "cat2", "cat2"),
    EndUserCategory("220/380|hh|cat3", "Household cat 3, >301 kWh (220/380)", "220/380", "hh", "cat3", "universal", "cat3", "cat3"),
    # Rows 6 and 8: supply is filed under 'other', distribution under a blank
    # sub-class.  This asymmetry is real -- do not "normalise" it away.
    EndUserCategory("3.3-6-10|com|other", "Commercial - other (3.3-6-10)", "3.3-6-10", "com", "other", "public", "other", ""),
    EndUserCategory("3.3-6-10|hh|", "Household (3.3-6-10)", "3.3-6-10", "hh", "", "universal", "", ""),
    EndUserCategory("35-110|com|other", "Commercial - other (35-110)", "35-110", "com", "other", "public", "other", ""),
)

CATEGORY_BY_ID: Dict[str, EndUserCategory] = {c.id: c for c in END_USER_CATEGORIES}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_end_user_price_tool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/tools/end_user_price_tools.py tests/test_end_user_price_tool.py
git commit -m "feat: add the end-user price category matrix"
```

### Task 7.2: The tool itself

**Files:**
- Modify: `agent/tools/end_user_price_tools.py`
- Test: `tests/test_end_user_price_tool.py`

**Interfaces:**
- Produces:

```python
def get_end_user_prices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    supplier: Optional[str] = None,          # 'telmico' | 'eps' | None = both
    category: Optional[str] = None,          # a category id | None = all eight
    include_vat: bool = False,
    include_wholesale_benchmark: bool = False,
    currency: str = "gel",
    limit: int = MAX_ROWS,
) -> ToolResult
```

Returned columns: `date, supplier, category, category_label, distribution, supply,
transmission, final_price_net` plus `vat, total_gross` when `include_vat`, plus
`wholesale_benchmark, spread` when `include_wholesale_benchmark`.

- [ ] **Step 1: Write the failing test**

```python
def test_tool_returns_components_and_the_published_net_total(monkeypatch):
    """The tool returns the breakdown AND the published final_price, so the
    answer never has to compute a total that exists in no row."""
    import agent.tools.end_user_price_tools as tool_module

    captured = {}

    def fake_run_text_query(sql, params=None):
        captured["sql"] = sql
        captured["params"] = params
        import pandas as pd

        df = pd.DataFrame(
            [{
                "date": "2026-06-01", "supplier": "telmico",
                "category": "220/380|hh|cat2", "category_label": "Household cat 2",
                "distribution": 0.0812, "supply": 0.1104,
                "transmission": 0.0067, "final_price_net": 0.1983,
            }]
        )
        return df, list(df.columns), [tuple(r) for r in df.itertuples(index=False, name=None)]

    monkeypatch.setattr(tool_module, "run_text_query", fake_run_text_query)

    df, cols, rows = tool_module.get_end_user_prices(
        supplier="telmico", category="220/380|hh|cat2"
    )

    assert "final_price_net" in cols
    assert {"distribution", "supply", "transmission"} <= set(cols)
    assert "final_price" in captured["sql"]


def test_vat_is_added_only_when_requested(monkeypatch):
    """value/final_price are NET of VAT; 18% is levied on top."""
    import agent.tools.end_user_price_tools as tool_module

    monkeypatch.setattr(tool_module, "run_text_query", _stub_single_row())

    _, cols_net, _ = tool_module.get_end_user_prices(include_vat=False)
    assert "total_gross" not in cols_net

    df, cols_gross, _ = tool_module.get_end_user_prices(include_vat=True)
    assert {"vat", "total_gross"} <= set(cols_gross)
    assert round(float(df["total_gross"].iloc[0]), 4) == round(0.1983 * 1.18, 4)


def test_wholesale_benchmark_adds_the_capacity_charge_and_converts_down(monkeypatch):
    """Benchmark = (p_bal_gel + p_gcap_gel) / 1000, in GEL/kWh.

    The supply tariff already bundles the guaranteed capacity fee, so the
    charge goes on the WHOLESALE side; and prices are GEL/MWh, so they are
    divided by 1000 rather than the tariff being multiplied by it.
    """
    import agent.tools.end_user_price_tools as tool_module

    captured = {}

    def fake_run_text_query(sql, params=None):
        captured["sql"] = sql
        return _stub_single_row()(sql, params)

    monkeypatch.setattr(tool_module, "run_text_query", fake_run_text_query)

    tool_module.get_end_user_prices(include_wholesale_benchmark=True)

    assert "p_gcap_gel" in captured["sql"]
    assert "p_bal_gel" in captured["sql"]
    assert "1000" in captured["sql"]
    assert "* 1000" not in captured["sql"], "tariffs must never be scaled up"


def test_unknown_category_is_rejected():
    from agent.tools.end_user_price_tools import get_end_user_prices

    with pytest.raises(ValueError, match="Unknown end-user category"):
        get_end_user_prices(category="not-a-category")


def test_unknown_supplier_is_rejected():
    from agent.tools.end_user_price_tools import get_end_user_prices

    with pytest.raises(ValueError, match="Unknown supplier"):
        get_end_user_prices(supplier="telasi")  # a distributor, not a supplier
```

Add the shared stub helper at the top of the test module:

```python
def _stub_single_row():
    import pandas as pd

    def _run(sql, params=None):
        df = pd.DataFrame(
            [{
                "date": "2026-06-01", "supplier": "telmico",
                "category": "220/380|hh|cat2", "category_label": "Household cat 2",
                "distribution": 0.0812, "supply": 0.1104,
                "transmission": 0.0067, "final_price_net": 0.1983,
            }]
        )
        return df, list(df.columns), [tuple(r) for r in df.itertuples(index=False, name=None)]

    return _run
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_end_user_price_tool.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'get_end_user_prices'`

- [ ] **Step 3: Write minimal implementation**

Build one SQL statement that pivots the three activities per `(date, supplier, category)`,
joins the published `final_price` row, bounds the window at the last `final_price` month, and
optionally joins `price_with_usd` for the benchmark. Compute `vat`/`total_gross` in pandas
after the query so the arithmetic is visible and testable. Raise `ValueError` for an unknown
category or supplier. Reject any `supplier` not in `SUPPLIER_TO_DISTRIBUTOR` — passing a
distributor code is the most likely caller mistake.

Drop any `(category, date)` where a component is missing rather than emitting a partial
stack — this mirrors the view's own INNER JOIN, and a partial total is worse than an absent
one.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_end_user_price_tool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/tools/end_user_price_tools.py tests/test_end_user_price_tool.py
git commit -m "feat: add the get_end_user_prices typed tool"
```

### Task 7.3: Register the tool

**Files:**
- Modify: `agent/tools/registry.py`, `contracts/question_analysis.py`, `agent/tools/capabilities.py`
- Test: `tests/test_end_user_price_tool.py`, `tests/test_plan_validation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_tool_is_registered_and_executable():
    from agent.tools.registry import TOOL_REGISTRY, list_tools

    assert "get_end_user_prices" in TOOL_REGISTRY
    assert "get_end_user_prices" in list_tools()


def test_tool_name_enum_carries_the_tool():
    from contracts.question_analysis import ToolName

    assert ToolName.GET_END_USER_PRICES.value == "get_end_user_prices"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_end_user_price_tool.py -k register -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

- `ToolName.GET_END_USER_PRICES = "get_end_user_prices"` in `contracts/question_analysis.py`
- `"get_end_user_prices": get_end_user_prices` in `TOOL_REGISTRY`
- Add `ToolName.GET_END_USER_PRICES.value` to `_DEFAULT_MULTI_PERIOD_TOOLS` in
  `agent/tools/capabilities.py` — an unbounded call returns a bounded history, same as its peers.

**The drift guard added earlier will fail until you classify the tool.** That is the guard
working: `test_every_tool_is_explicitly_classified` fails for any `ToolName` member that is in
neither the multi-period set nor `KNOWN_SINGLE_PERIOD_TOOLS`.

- [ ] **Step 4: Regenerate the schema snapshot and run the suite**

```bash
python -c "import json,os,sys; os.environ.setdefault('SUPABASE_DB_URL','postgresql://x/y'); sys.path.insert(0,'.'); from contracts.question_analysis import QuestionAnalysis; from pathlib import Path; Path('schemas/question_analysis.schema.json').write_text(json.dumps(QuestionAnalysis.model_json_schema(), indent=2, ensure_ascii=False)+chr(10), encoding='utf-8')"
python -m pytest tests/ --ignore=tests/security -q
```
Expected: 0 failures

- [ ] **Step 5: Commit**

```bash
git add agent/tools/registry.py contracts/question_analysis.py agent/tools/capabilities.py schemas/question_analysis.schema.json tests/
git commit -m "feat: register get_end_user_prices as a typed tool"
```

### Task 7.4: Resolve params from analyzer output

**Goal:** Turn the analyzer's freeform `entity_scope` (e.g. `household_consumers`,
`distribution_network_tariffs`, `Telmico`) into a strict `(supplier, category)` pair. **This is
the hard part of the phase and the likeliest source of bugs** — an unmatched scope must degrade
to "all suppliers, all categories", never to a wrong category.

**Files:**
- Modify: `agent/planner.py` (`resolve_tool_params`, line 982)
- Test: `tests/test_planner_tool_params.py` or the existing planner test module

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize(
    "entity_scope,expected_supplier,expected_category",
    [
        ("Telmico", "telmico", None),
        ("household_consumers", None, None),
        ("distribution_network_tariffs", None, None),
        ("", None, None),
        ("something_unrecognised", None, None),
    ],
)
def test_end_user_scope_never_resolves_to_a_wrong_category(
    entity_scope, expected_supplier, expected_category
):
    """An unrecognised scope must widen, never guess.

    Returning a specific category for a scope we did not understand produces a
    confidently wrong answer -- the exact failure mode this whole plan exists
    to remove.
    """
    qa = _qa_with_scope(entity_scope)

    params = resolve_tool_params(qa, "get_end_user_prices", "test query")

    assert params is not None
    assert params.get("supplier") == expected_supplier
    assert params.get("category") == expected_category
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_planner_tool_params.py -k end_user_scope -v`
Expected: FAIL — `resolve_tool_params` returns `None` for an unknown tool.

- [ ] **Step 3: Write minimal implementation**

Add a branch for `get_end_user_prices` that resolves dates exactly as the other tools do, maps a
recognised supplier name to its code, maps a recognised category phrase to a category id, and
leaves both as `None` otherwise. Do not invent a default category.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_planner_tool_params.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/planner.py tests/
git commit -m "feat: resolve end-user price tool params from analyzer output"
```

### Task 7.5: Route to the tool instead of suppressing

**Files:**
- Modify: `agent/evidence_planner.py` (the guard from Task 2.1)
- Test: `tests/test_evidence_planner.py`

- [ ] **Step 1: Write the failing test**

```python
def test_retail_tariff_question_routes_to_the_end_user_price_tool():
    """Phase 2 suppressed the wrong tool; Phase 7 supplies the right one."""
    payload = _make_qa_payload(
        query_type="data_retrieval",
        preferred_path="tool",
        tools=[{"name": "get_tariffs", "score": 0.85, "reason": "tariff data"}],
    )
    payload["knowledge"] = {
        "candidate_topics": [{"name": "network_supply_tariffs", "score": 0.9}]
    }
    payload["entity_scope"] = "distribution_network_tariffs"

    ctx = build_evidence_plan(_ctx_with_qa(payload))

    assert [step["tool_name"] for step in ctx.evidence_plan] == ["get_end_user_prices"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_evidence_planner.py -k routes_to_the_end_user -v`
Expected: FAIL — `assert [] == ['get_end_user_prices']` (the Task 2.1 guard returns an empty plan)

- [ ] **Step 3: Write minimal implementation**

Change the Task 2.1 guard from `return []` to substituting the tool:

```python
    if top_name == ToolName.GET_TARIFFS.value and _is_retail_tariff_question(qa):
        log.info(
            "Evidence plan: retail-tariff question -- substituting %s for get_tariffs",
            ToolName.GET_END_USER_PRICES.value,
        )
        top_name = ToolName.GET_END_USER_PRICES.value
```

The Task 2.2 tests still pass: they assert `get_tariffs` is *not* selected for retail and *is*
selected for generation, both of which remain true.

- [ ] **Step 4: Run the whole planner suite**

Run: `python -m pytest tests/test_evidence_planner.py -v`
Expected: PASS, including the three Task 2.2 guards.

- [ ] **Step 5: Commit**

```bash
git add agent/evidence_planner.py tests/test_evidence_planner.py
git commit -m "feat: route retail tariff questions to get_end_user_prices"
```

### Task 7.6: Add the tool to the analyzer catalog

**Files:**
- Modify: `contracts/question_analysis_catalogs.py` (`QUESTION_ANALYSIS_TOOL_CATALOG`)
- Test: `tests/test_question_analysis_catalogs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_end_user_price_tool_is_catalogued_and_distinct_from_the_others():
    from contracts.question_analysis_catalogs import QUESTION_ANALYSIS_TOOL_CATALOG

    entry = _entry(QUESTION_ANALYSIS_TOOL_CATALOG, "get_end_user_prices")

    assert "end-user" in entry["use_for"].lower()
    assert "get_tariffs" in entry["avoid_for"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_question_analysis_catalogs.py -k end_user_price_tool -v`
Expected: FAIL — no such entry

- [ ] **Step 3: Write minimal implementation**

```python
    {
        "name": "get_end_user_prices",
        "concepts": ["end-user price", "retail tariff", "household tariff",
                      "distribution tariff", "supply tariff", "telasi", "telmico"],
        "use_for": "Regulated end-user (retail) electricity prices in GEL/kWh and their distribution/supply/transmission components, by supplier and consumer category.",
        "avoid_for": "What a generating plant is paid -- use get_tariffs. Wholesale/balancing prices -- use get_prices.",
        "main_params": ["supplier", "category", "start_date", "end_date", "include_vat"],
    },
```

- [ ] **Step 4: Verify the prompt budget did not regress, then regenerate the fixture**

```bash
python -c "import tests.test_analyzer_prompt_order as t; print(sum(1 for v in t._budgeted_prompts().values() if '[truncated]' in v))"
```
Expected: still `36`. If it rises, shorten `use_for` — do not accept the regression.

```bash
python -c "import tests.test_analyzer_prompt_order as t; t.regenerate()"
python -m pytest tests/ --ignore=tests/security -q
```

- [ ] **Step 5: Commit**

```bash
git add contracts/question_analysis_catalogs.py tests/ tests/fixtures/analyzer_prompt_legacy_hashes.json
git commit -m "feat: catalogue get_end_user_prices for the analyzer"
```

### Task 7.7: Keep the category matrix and the knowledge file in step

**Files:**
- Test: `tests/test_network_supply_knowledge.py`

- [ ] **Step 1: Write the test**

```python
def test_knowledge_file_lists_every_tool_category():
    """The tool's matrix is the source of truth. If the knowledge file drifts
    from it, the model reads one set of categories and the tool serves another.
    """
    from agent.tools.end_user_price_tools import END_USER_CATEGORIES

    for category in END_USER_CATEGORIES:
        assert category.volate in KNOWLEDGE, f"{category.volate} missing from the topic file"
```

- [ ] **Step 2: Run to verify it passes**

Run: `python -m pytest tests/test_network_supply_knowledge.py -v`
Expected: PASS (Phase 4 already wrote the voltages in)

- [ ] **Step 3: Commit**

```bash
git add tests/test_network_supply_knowledge.py
git commit -m "test: pin the tool category matrix against the knowledge topic"
```

---

## Verification

After all phases:

- [ ] `python -m pytest tests/ --ignore=tests/security -q` — 0 failures
- [ ] `ruff check .` — clean
- [ ] `python scripts/generate_requirements_lock.py --check` — exit 0
- [ ] Re-run the four production queries and confirm:
  - the distribution-tariff question reads `demand_tariff_mv`, not `tariff_with_usd`
  - `stats_hint_len` is well above 9 on the SQL path
  - `Grounding fail` no longer appears for retail queries
  - `shipped_answer_chars` is close to `model_answer_chars`
- [ ] Ask four domain questions and check the answers by hand:
  - *"Who distributes electricity in Tbilisi?"* → Telasi (distribution) and Telmico as supplier; must not claim Energo-Pro Georgia is absent from Tbilisi suburbs
  - *"What does a household in Tbilisi pay per kWh?"* → a net GEL/kWh figure matching a real `final_price` row, labelled net of VAT
  - *"…including VAT?"* → that figure × 1.18
  - *"How does it compare with the wholesale price?"* → benchmark `(p_bal_gel + p_gcap_gel) / 1000`, in GEL/kWh, comparing against the **net** price

## Risks

| Risk | Handling |
| --- | --- |
| Phase 1 changes dtypes for **every** SQL answer, not just the new views | This is the intended blast radius — the bug always applied. The full suite is the gate; investigate any failure rather than suppressing it. |
| Richer grounding corpus could make the gate too permissive | Aggregates are computed from the real frame, so a fabricated number still cannot match unless it is a genuine aggregate. Rationale already argued in `_add_aggregate_tokens`. Do not widen further. |
| Phase 2 guard over-blocks | Two negative tests (2.2) pin generation-side routing. Suppression requires the retail topic **and** absence of the generation topic. |
| Phase 2 leaves no retail tool until Phase 7 | Intentional sequencing. The ~20-line guard stops wrong answers on day one; the tool is the structural fix. Task 7.5 converts the guard from suppression into routing, and the Phase 2 tests carry over unchanged. |
| Task 7.4 (`entity_scope` → category) is the phase's real risk | An unrecognised scope must **widen** to all suppliers/categories, never guess a specific one. Guessing reintroduces exactly the confidently-wrong failure this plan exists to remove. The parametrised test pins that behaviour for five scope shapes including the empty and unrecognised cases. |
| A fifth tool adds to the analyzer's tool catalog | Task 7.6 checks the truncation count stays at 36 before accepting the change. |
| Analyzer prompt budget | Task 2.3 checks the truncation count stays at 36 and regenerates the golden fixture. |
| Phases 4-5 add knowledge content, which competes for the summarizer's budget | `UNTRUSTED_DOMAIN_KNOWLEDGE` is shed first under pressure. Keep additions tight and prefer the knowledge file (selectively loaded per topic) over `DB_SCHEMA_DOC` (sent on every planning call). |
| Gross (×1.18) and the wholesale benchmark are derived values | Both are absent from every row, so `strict_numeric` will strip them. Task 5.2 states the rule; if verification still shows them stripped, serialize them into `stats_hint` rather than weakening the gate. |
| The 0.95 lower band edge is not reproduced | Deliberate — its rationale is not documented in the dashboard and encoding it blind would invent precision. Prose uses the single benchmark line. |

## Out of Scope

- The `time_month` phantom column in the few-shot examples (separate task).
- The analyzer prompt being over budget in 36/40 variants (separate task).
- `dates_mv` / `monthly_cpi_mv` missing from `DB_SCHEMA_DOC` (separate task).
