# Report Charting Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the report chart layer producing charts that mix incompatible
quantities, and close the composition exhibit that has been omitted four runs
running.

**Architecture:** Every defect here is the report path re-deciding something
Standard's chart layer already decides correctly. `visualization/chart_selector`
owns dimension inference and chart-type selection; `agent/derived_chart_builder`
owns which columns a composition may plot. The report builder consults the first
for *type* but never for *which slices*, and keeps its own answer for the
second. Each task moves one decision to the module that already owns it.

**Tech Stack:** Python 3.11 (container) / 3.14 (local), pydantic v2, pytest.

## Global Constraints

- Gate for every task: `python -m pytest tests/ -q`, `ruff check .`, and
  `python -m guardrails.redteam_gate` (score must stay ≥ 0.92).
- Run from `D:\Enaiapp\langchain_railway`, not from `D:\Enaiapp`.
- Pre-set env before pytest, using exactly these values — `tests/test_main.py`
  asserts against them and shorter secrets fail 16 auth tests for the wrong
  reason: `SUPABASE_DB_URL=postgresql://user:pass@localhost/db`,
  `ENAI_GATEWAY_SECRET=test-gateway-key`,
  `ENAI_SESSION_SIGNING_SECRET=test-session-key`,
  `ENAI_EVALUATE_SECRET=test-evaluate-key`, `MODEL_TYPE=openai`,
  `OPENAI_API_KEY=test-openai-key`, `NVIDIA_API_KEY=test-nvidia-key`.
- Every test must fail before the fix and pass after. A test that passes both
  ways is a plan failure. Verify by reverting the single changed line.
- Branch: `agent/track-scoped-report-analysis` (HEAD `2d9995d`).
- `infer_dimension(column)` (`visualization/chart_selector.py:172`) returns
  exactly one of: `"xrate"`, `"share"`, `"index"`, `"energy_qty"`,
  `"price_tariff"`, `"other"`. This vocabulary is the authority; do not add a
  second one.

---

## Direction: protecting Standard while fixing the report (2026-08-09, after job fbc46aa4)

Standard's chart output is better than the report's, and the reason is now
specific rather than a general impression. **The report leans on the one rule
in the system that Standard treats as a last resort.**

### The seam

| Module | Used by | Rule for touching it |
|---|---|---|
| `agent/chart_pipeline.py` | Standard only (Stage 5, via `agent/pipeline.py`) | Refactor only. Never change its decisions. |
| `agent/report_charts.py` | Report only | Free to change. Prefer every fix here. |
| `visualization/chart_selector.py` | **Both** | Change only behind a characterization suite. |
| `agent/derived_chart_builder.py` | **Both**, via `analyzer.enrich` in Stage 3 | Highest risk in the codebase. A boolean-dtype bug here once killed derived charts in *both* modes. |

### Why Standard is better: it has three protections the report has none of

Standard reaches the shared `select_chart_type` **last**
(`chart_pipeline.py:1528`), after `preferred_chart_family` and
`_chart_type_for_visual_goal`, and then applies a corrective pass. The report
calls `select_chart_type` as its **only** authority
(`report_charts.py:297`).

The rules differ in exactly the way that produces the reported defect:

| Where | Composition test | Effect |
|---|---|---|
| `chart_pipeline.py:1563` (Standard) | `dimensions == {"share"}` | Exact — a mixed set never pies |
| `chart_pipeline.py:1536` (Standard) | `dimensions == {"share"}` | Exact |
| `chart_pipeline.py:1542` (Standard) | price/xrate present and share absent → force `line` | A corrective the report never runs |
| `chart_selector.py:242` (**shared**) | `"share" in dimensions` | **Membership** — a mixed set pies |

So a report composition whose columns infer to `{"share", "energy_qty"}` takes
the membership branch and gets a pie with shares and thousand MWh in the same
whole. Standard's own adjacent code shows it never intended membership; it just
never had to fix the shared function because it rarely reaches it.

### The rule I will follow

**Do not tighten the shared rule to fix the report. Give the report the rules
Standard already has.** Tightening `select_chart_type` would be a one-line fix
that silently re-decides every Standard chart that falls through to it, and
"rarely reaches it" is not "never reaches it".

Order of work, each step gated on the one before:

1. **Characterize first.** Pin Standard's current chart-type decision across
   the `(visual_goal, has_time, has_categories, dimensions, category_count)`
   matrix, including the corrective pass. These tests must pass unchanged
   before *and* after every later step. This is the regression net, and it is
   written before anything moves.
2. **Extract, do not edit.** Lift Standard's corrective rules into a shared,
   named function that `chart_pipeline` then calls in place of its inline copy.
   Pure refactor: Standard's output must be byte-identical, proven by step 1.
3. **Then let the report call it.** The report gains the exactness test and the
   price/xrate corrective. Standard's behaviour never changed — only its code
   location did.
4. **Report-only rules stay report-only.** `REPORT_CHART_INCOMPATIBLE_UNITS`
   and the omission machinery have no Standard counterpart and belong in
   `agent/report_charts.py`.

If a step cannot keep Standard byte-identical, stop and re-plan rather than
accept the drift.

### Tasks 1–3 need re-validating first

Job `fbc46aa4` moved the ground under this plan, which was written from four
earlier runs. It built **three of four** charts:

- `generation_and_cross_border_flows_composition` — pie, 8 categories,
  `series_count: 1`. The mixed-unit pie did **not** reproduce.
- `regulated_tariffs_table` — built (it was omitted as
  `REPORT_CHART_INCOMPATIBLE_EVIDENCE` the run before).
- `prices_and_balancing_composition` — omitted `REPORT_CHART_INCOMPATIBLE_UNITS`,
  which is the units guard working, not failing.
- The composition omission Task 3 investigates now reports a *different* reason
  code than the `REPORT_CHART_INSUFFICIENT_CATEGORIES` it was written against.

So step 0 is to re-run the plan's premises against current behaviour and delete
whatever the intervening fixes already closed. The membership-vs-equality gap
above is real and provable at rest; the rest of the plan is not yet re-confirmed.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `agent/report_charts.py` | Turns chart requests + manifest into artefacts | Modify — slice selection and unit assignment |
| `tests/test_report_charts.py` | Chart builder behaviour | Modify — new cases |
| `agent/report_research_execution.py` | Builds exhibit candidates | Modify only in Task 3, and only if Task 3's diagnosis says so |

---

## Background: what the last four runs show

Job `26f3bbf6` built `supply_mix_and_cross_border_composition` as a pie with
**8 categories**. The same track logged:

```
_limit_derived_num_cols capped 10 → 8 series:
['share_hydro','share_thermal','share_wind','share_solar',
 'quantity_hydro','quantity_solar','quantity_thermal','quantity_wind']
```

Four ratios and four thousand-MWh quantities as slices of one whole.

`_composition_snapshot_type` (`agent/report_charts.py:284`) already delegates
the *type* question to Standard, and its docstring records that job `83010f04`
"pied a GEL price, an FX rate and two quantities as slices of one whole and
stamped the first unit it found onto all of them". That fix was half done:
Standard answers "pie" because `"share" in dimensions`, but nothing removes the
non-share columns, and `report_charts.py:800-805` still does

```python
units={"value": next((units[c] for c in pivot_columns if c in units), "")}
```

— the same first-unit stamp the docstring says was fixed. Standard's own
composition builder does the missing half
(`agent/derived_chart_builder.py:749-752`):

```python
share_cols = [
    c for c in dim_map
    if dim_map[c] == "share" and not c.lower().endswith("_total")
]
```

Separately, `balancing_market_composition_composition` has been omitted with
`REPORT_CHART_INSUFFICIENT_CATEGORIES` and "composition snapshot with 1
categories" in jobs `5cb4d210`, `106b043c`, `70692961` and `26f3bbf6`. Task 3
diagnoses that rather than guessing; four runs of speculation have not settled
which table reaches the builder with one category.

---

## Task 1: A composition plots one kind of quantity

**Files:**
- Modify: `agent/report_charts.py:753-808` (the wide-frame composition branch)
- Test: `tests/test_report_charts.py`

**Interfaces:**
- Consumes: `infer_dimension(column) -> str` from
  `visualization/chart_selector` (already imported at
  `agent/report_charts.py:22`).
- Produces: `_composition_slice_columns(columns: list[str], units: dict[str, str]) -> list[str]`
  in `agent/report_charts.py`, returning the subset of `columns` that share one
  dimension — preferring `"share"`, else the largest group, ties broken by
  first appearance.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report_charts.py`:

```python
def test_a_composition_pie_plots_one_kind_of_quantity():
    """Slices of one whole must be the same kind of number.

    Job 26f3bbf6 pied four shares and four thousand-MWh quantities together
    and labelled every slice with whichever unit came first — the same defect
    job 83010f04 recorded, whose fix only covered the chart-type half.
    """

    from agent.report_charts import _composition_slice_columns

    columns = [
        "share_hydro",
        "quantity_hydro",
        "share_thermal",
        "quantity_thermal",
    ]
    units = {
        "share_hydro": "ratio",
        "quantity_hydro": "thousand MWh",
        "share_thermal": "ratio",
        "quantity_thermal": "thousand MWh",
    }

    assert _composition_slice_columns(columns, units) == [
        "share_hydro",
        "share_thermal",
    ]


def test_a_composition_without_shares_still_plots_one_dimension():
    """No share columns: take the largest single-dimension group, not all."""

    from agent.report_charts import _composition_slice_columns

    columns = ["quantity_hydro", "p_bal_gel", "quantity_thermal"]
    units = {
        "quantity_hydro": "thousand MWh",
        "p_bal_gel": "GEL/MWh",
        "quantity_thermal": "thousand MWh",
    }

    assert _composition_slice_columns(columns, units) == [
        "quantity_hydro",
        "quantity_thermal",
    ]
```

- [ ] **Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_report_charts.py -q -k "one_kind_of_quantity or one_dimension"
```

Expected: FAIL — `ImportError: cannot import name '_composition_slice_columns'`.

- [ ] **Step 3: Add the selector**

In `agent/report_charts.py`, directly above `_composition_snapshot_type`:

```python
def _composition_slice_columns(
    columns: list[str],
    units: dict[str, str],
) -> list[str]:
    """Return the columns that can be slices of one whole.

    A pie asserts that its slices sum to something. Four ratios and four
    thousand-MWh quantities do not, and job 26f3bbf6 charted exactly that.
    ``_composition_snapshot_type`` already asks Standard whether a pie is
    right, and Standard says yes as soon as one share column is present — it
    answers the type question, not which columns belong. Standard's own
    composition builder answers that second question by keeping only the
    ``share`` dimension, so this keeps the same rule and falls back to the
    largest single dimension when no share column exists.

    Grouped by ``infer_dimension`` and the declared unit together: two share
    columns on different scales are still not one whole.
    """

    grouped: dict[tuple[str, str], list[str]] = {}
    for column in columns:
        key = (infer_dimension(column), str(units.get(column, "")).strip())
        grouped.setdefault(key, []).append(column)
    if not grouped:
        return []
    preferred = [
        members for (dimension, _unit), members in grouped.items()
        if dimension == "share"
    ]
    if preferred:
        return max(preferred, key=len)
    return max(grouped.values(), key=len)
```

- [ ] **Step 4: Run the test and watch it pass**

```bash
python -m pytest tests/test_report_charts.py -q -k "one_kind_of_quantity or one_dimension"
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Write the failing test for the builder**

Append to `tests/test_report_charts.py`:

```python
def test_a_wide_composition_chart_drops_the_other_dimension():
    """The builder must plot only the slices, and label them one unit."""

    from agent.report_charts import build_report_chart_requests
    from contracts.report import ReportChartPurpose, ReportChartRequest

    table, manifest = _wide_composition_manifest()
    decision = build_report_chart_requests(
        [
            ReportChartRequest(
                chart_id="mix_composition",
                section_id="mix",
                purpose=ReportChartPurpose.COMPOSITION,
                title="May 2026 supply mix",
                evidence_refs=[table["evidence_ref"]],
                x_field="date",
                series_fields=[
                    "share_hydro",
                    "quantity_hydro",
                    "share_thermal",
                    "quantity_thermal",
                ],
                required=False,
            )
        ],
        manifest,
    )[0]

    assert decision.status == "built"
    categories = {row["category"] for row in decision.artifact.data}
    assert categories == {"share_hydro", "share_thermal"}
    assert set(decision.artifact.metadata.unit_by_series.values()) == {"ratio"}
```

Add this helper directly above that test, mirroring the fixtures already in the
module:

```python
def _wide_composition_manifest():
    """One period, four numeric columns, two dimensions."""

    from contracts.report_evidence import ReportEvidenceManifest

    table = _table_item()
    table["title"] = "Observed generation and supply mix"
    table["columns"] = [
        "date",
        "share_hydro",
        "quantity_hydro",
        "share_thermal",
        "quantity_thermal",
    ]
    table["rows"] = [
        {
            "date": "2026-05",
            "share_hydro": 0.99,
            "quantity_hydro": 1180.0,
            "share_thermal": 0.01,
            "quantity_thermal": 12.0,
        }
    ]
    table["unit_by_column"] = {
        "share_hydro": "ratio",
        "quantity_hydro": "thousand MWh",
        "share_thermal": "ratio",
        "quantity_thermal": "thousand MWh",
    }
    table["total_row_count"] = 1
    manifest = ReportEvidenceManifest.model_validate(
        {**_manifest(), "items": [table]}
    )
    return table, manifest
```

If `_table_item` and `_manifest` are not already imported in
`tests/test_report_charts.py`, add
`from tests.fixtures_report_manifest import _manifest, _table_item` — that hub
module is where the other chart tests get their fixtures. Confirm the exact
names with `grep -n "_table_item\|_manifest" tests/test_report_charts.py | head`
before writing, and use whatever that file already imports rather than adding a
second fixture source.

- [ ] **Step 6: Run it and watch it fail**

```bash
python -m pytest tests/test_report_charts.py -q -k "drops_the_other_dimension"
```

Expected: FAIL — `categories` contains all four columns.

- [ ] **Step 7: Use the selector in the wide-frame branch**

In `agent/report_charts.py`, in the `if temporal and len(numeric) >= 2:`
composition branch, replace

```python
                pivot_columns = [
                    column
                    for column in numeric[:_MAXIMUM_CHART_SERIES]
                    if _is_numeric(latest.get(column))
                ]
```

with

```python
                pivot_columns = _composition_slice_columns(
                    [
                        column
                        for column in numeric[:_MAXIMUM_CHART_SERIES]
                        if _is_numeric(latest.get(column))
                    ],
                    units,
                )
```

and replace the `units=` argument of the PIE `_built(...)` call with

```python
                        units={
                            "value": next(
                                (
                                    units[column]
                                    for column in pivot_columns
                                    if column in units
                                ),
                                "",
                            )
                        },
```

left exactly as it is — it is now correct, because every remaining column
shares one unit. Add a comment above it saying so:

```python
                        # Safe only because _composition_slice_columns left one
                        # unit standing; this line stamped a mixed pie before.
```

- [ ] **Step 8: Run the test and watch it pass**

```bash
python -m pytest tests/test_report_charts.py -q
```

Expected: PASS. If a previously-passing chart test now fails, read whether the
chart it expected mixed dimensions; change its expectation only when that is
demonstrably true, never to make it green.

- [ ] **Step 9: Full gate and commit**

```bash
python -m pytest tests/ -q
```

```bash
git add -A && git commit -m "fix(charts): a composition plots one kind of quantity"
```

---

## Task 2: The category-axis branch honours the same rule

`report_charts.py:702-752` handles the long-form composition — a table with a
category column — and picks `snapshot_series = (chart.series_fields or [numeric[0]])[:_MAXIMUM_CHART_SERIES]`.
Nothing there restricts the measure to one dimension either. A melted frame
carrying both a share and a quantity per category hits the same defect by a
different route, and this branch is the one the anchored-explanation frames
reach.

**Files:**
- Modify: `agent/report_charts.py:702-752`
- Test: `tests/test_report_charts.py`

**Interfaces:**
- Consumes: `_composition_slice_columns` from Task 1.

- [ ] **Step 1: Write the failing test**

```python
def test_a_categorical_composition_chart_measures_one_dimension():
    """The long-form branch needs the same rule as the wide one.

    A melted frame carrying a share and a quantity for each category would
    otherwise plot both as slices, the way the wide branch did on job 26f3bbf6.
    """

    from agent.report_charts import build_report_chart_requests
    from contracts.report import ReportChartPurpose, ReportChartRequest
    from contracts.report_evidence import ReportEvidenceManifest

    table = _table_item()
    table["title"] = "Composition: focus periods"
    table["columns"] = ["category", "share_value", "quantity_value"]
    table["rows"] = [
        {"category": "Hydro", "share_value": 0.99, "quantity_value": 1180.0},
        {"category": "Thermal", "share_value": 0.01, "quantity_value": 12.0},
    ]
    table["unit_by_column"] = {
        "share_value": "ratio",
        "quantity_value": "thousand MWh",
    }
    table["total_row_count"] = 2
    manifest = ReportEvidenceManifest.model_validate(
        {**_manifest(), "items": [table]}
    )

    decision = build_report_chart_requests(
        [
            ReportChartRequest(
                chart_id="mix_composition",
                section_id="mix",
                purpose=ReportChartPurpose.COMPOSITION,
                title="May 2026 supply mix",
                evidence_refs=[table["evidence_ref"]],
                x_field="category",
                series_fields=["share_value", "quantity_value"],
                required=False,
            )
        ],
        manifest,
    )[0]

    assert decision.status == "built"
    assert decision.artifact.metadata.series == ["share_value"]
```

- [ ] **Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_report_charts.py -q -k "measures_one_dimension"
```

Expected: FAIL — `series` is `['share_value', 'quantity_value']`.

- [ ] **Step 3: Apply the rule**

In the `if categorical:` composition branch, replace

```python
                snapshot_series = (chart.series_fields or [numeric[0]])[:_MAXIMUM_CHART_SERIES]
```

with

```python
                # Same rule as the wide branch: slices of one whole are one
                # kind of number, whichever shape the table arrives in.
                snapshot_series = _composition_slice_columns(
                    [
                        column
                        for column in (chart.series_fields or [numeric[0]])
                        if column in numeric
                    ],
                    units,
                )[:_MAXIMUM_CHART_SERIES]
                if not snapshot_series:
                    decisions.append(
                        _omitted(chart, "REPORT_CHART_NO_NUMERIC_EVIDENCE")
                    )
                    continue
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_report_charts.py -q
```

Expected: PASS.

- [ ] **Step 5: Full gate and commit**

```bash
python -m pytest tests/ -q
```

```bash
git add -A && git commit -m "fix(charts): the categorical composition branch measures one dimension"
```

---

## Task 3: Find out why the balancing composition has one category

Omitted with `REPORT_CHART_INSUFFICIENT_CATEGORIES` and "composition snapshot
with 1 categories" in four consecutive jobs. I have speculated about the cause
three times and been wrong each time, so this task measures instead.

The two branches that can log one category are
`report_charts.py:723` (`len(composition_rows)`, the categorical branch after
filtering to the latest period) and `report_charts.py:765`
(`len(pivot_columns)`, the wide branch). The log line alone does not say which.

**Files:**
- Modify: `agent/report_charts.py` (the two `REPORT_CHART_INSUFFICIENT_CATEGORIES` sites)
- Test: `tests/test_report_charts.py`

**Interfaces:**
- Consumes: `_omitted(chart, reason_code)` as it exists today.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test**

```python
def test_an_omitted_composition_says_which_branch_refused_it(caplog):
    """Four jobs logged "1 categories" without saying which rule fired.

    The categorical branch counts rows after filtering to the latest period;
    the wide branch counts pivot columns. Both emit the same code.
    """

    import json
    import logging

    from agent.report_charts import build_report_chart_requests
    from contracts.report import ReportChartPurpose, ReportChartRequest
    from contracts.report_evidence import ReportEvidenceManifest

    table = _table_item()
    table["columns"] = ["date", "category", "share_value"]
    table["rows"] = [{"date": "2026-05", "category": "Hydro", "share_value": 1.0}]
    table["unit_by_column"] = {"share_value": "ratio"}
    table["total_row_count"] = 1
    manifest = ReportEvidenceManifest.model_validate(
        {**_manifest(), "items": [table]}
    )

    with caplog.at_level(logging.INFO, logger="Enai.ReportCharts"):
        decision = build_report_chart_requests(
            [
                ReportChartRequest(
                    chart_id="mix_composition",
                    section_id="mix",
                    purpose=ReportChartPurpose.COMPOSITION,
                    title="May 2026 supply mix",
                    evidence_refs=[table["evidence_ref"]],
                    x_field="category",
                    series_fields=["share_value"],
                    required=False,
                )
            ],
            manifest,
        )[0]

    assert decision.status == "omitted"
    assert decision.reason_code == "REPORT_CHART_INSUFFICIENT_CATEGORIES"
    logged = [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("REPORT_CHART_COMPOSITION_REFUSED ")
    ]
    assert logged, "the refusal did not say which branch fired"
    assert logged[0]["branch"] == "categorical"
    assert logged[0]["category_count"] == 1
```

- [ ] **Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_report_charts.py -q -k "which_branch_refused"
```

Expected: FAIL — `the refusal did not say which branch fired`.

- [ ] **Step 3: Name the branch**

Add above `build_report_chart_requests` in `agent/report_charts.py`:

```python
def _log_composition_refusal(
    chart,
    *,
    branch: str,
    category_count: int,
) -> None:
    """Say which composition rule refused, and on what count.

    Both branches emit REPORT_CHART_INSUFFICIENT_CATEGORIES, so four
    consecutive jobs logged "1 categories" without saying whether that was one
    row after the latest-period filter or one plottable column.
    """

    _LOGGER.info(
        "REPORT_CHART_COMPOSITION_REFUSED %s",
        json.dumps(
            {
                "branch": branch,
                "category_count": category_count,
                "chart_id": chart.chart_id,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
```

At the categorical site (`if snapshot_type == "pie" and len(composition_rows) < 2:`)
insert before `decisions.append(...)`:

```python
                    _log_composition_refusal(
                        chart,
                        branch="categorical",
                        category_count=len(composition_rows),
                    )
```

At the wide site (`if len(pivot_columns) < 2:`) insert before
`decisions.append(...)`:

```python
                    _log_composition_refusal(
                        chart,
                        branch="wide_pivot",
                        category_count=len(pivot_columns),
                    )
```

Confirm `json` and `_LOGGER` are already imported in the module — both are used
by `_chart_decision_log`; if not, add `import json` to the stdlib block.

- [ ] **Step 4: Run the test and watch it pass**

```bash
python -m pytest tests/test_report_charts.py -q
```

Expected: PASS.

- [ ] **Step 5: Full gate and commit**

```bash
python -m pytest tests/ -q
```

```bash
git add -A && git commit -m "fix(charts): say which composition rule refused an exhibit"
```

- [ ] **Step 6: Deploy and capture one run**

Ask for the `REPORT_CHART_COMPOSITION_REFUSED` line from one May 2026 report.
**Do not start Task 3b until that line exists.**

---

## Task 3b: Fix what Task 3 names

Blocked on the production line from Task 3 Step 6. Written so the sequence is
explicit, not so it can be started early.

| `branch` | Meaning | Fix |
|---|---|---|
| `categorical` | The latest period has one category row. The table is long-form but the period filter left a single slice. | Either the frame carries one category (evidence problem — the candidate should not have offered it, extend Task 3 of the previous plan) or the latest-period filter is matching on a formatted string that only one row shares. Check `str(row.get(time_column)) == latest_period` against the actual period spellings before changing anything. |
| `wide_pivot` | Only one numeric column survived `_is_numeric(latest.get(column))`. | The latest row has nulls in the other columns. Prefer the latest row that has at least two non-null slice columns rather than `rows[-1]` unconditionally. |

---

## Deferred, with rationale

**`price_performance_trend` built with one series.** Job `26f3bbf6` logged
`_limit_derived_num_cols capped 2 → 1 series (max_series=1, query_keywords=True)`
and the exhibit rendered a single line. That cap is Standard's, it is
query-driven, and a one-series price trend is a legitimate chart — so this is
recorded, not fixed. Revisit only if a run shows a trend that clearly needed its
second series.

**Sourcing report units from `ObservationFrame`.** Still the general fix behind
this whole class of defect: `contracts/evidence_frames.py::ObservationFrame`
carries `unit` per row and is built on every track in shadow mode, while
`agent/report_evidence.py` infers units from column names. Every unit-related
patch in the previous plan, and Task 1's grouping key here, are fallback-path
work that a frame-sourced manifest would delete. Separate plan; the frame is
still shadow-gated behind `ENAI_EVIDENCE_FINALIZATION_MODE`.

---

## Self-Review

**Spec coverage.** The reported defect (pie mixing shares and thousand MWh) →
Tasks 1 and 2, covering both branches that can build a composition. The
four-run omission → Task 3 (+3b), diagnostic-first because three prior guesses
were wrong. "Look at how Standard treats charts" → Task 1 adopts
`derived_chart_builder`'s share-filter rule and `chart_selector`'s dimension
vocabulary rather than adding a third; the Background section records that
`_composition_snapshot_type` already delegates the type half.

**Placeholders.** Task 3b is the only task without code and is explicitly gated
on a production line that does not exist yet; both its branches name the exact
next check. Task 1 Step 5 asks the implementer to confirm fixture import names
with a `grep` before writing — a verification step, not a placeholder, and
necessary because `tests/fixtures_report_manifest.py` is a shared hub whose
exports nine modules depend on.

**Type consistency.** `_composition_slice_columns(columns: list[str], units: dict[str, str]) -> list[str]`
is defined in Task 1 and consumed unchanged in Task 2.
`_log_composition_refusal(chart, *, branch: str, category_count: int) -> None`
is defined and used only in Task 3. `infer_dimension` returns the six-value
vocabulary quoted in Global Constraints, which is what Task 1's grouping key
relies on.

**Ordering.** Task 2 depends on Task 1's helper. Task 3 is independent of both
and should ship in the same deploy so its line arrives sooner.
