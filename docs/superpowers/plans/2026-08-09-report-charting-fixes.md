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

### Correcting this direction (self-review)

The first version of this section proposed extracting Standard's correctives
into a shared function that both callers use. **That was wrong on the terms
set for this work**, in four ways worth recording so the reasoning is not
repeated.

1. **"Extract into shared" edits Standard's call path.** The mandate is *no
   impact on Standard*. A refactor proven byte-identical over a test suite is
   evidence, not proof — it is only as good as the domain tested. Not touching
   the file is strictly stronger than testing that touching it was harmless.
2. **It converts a one-time risk into permanent coupling.** Once both callers
   share a mutable rule, the next report-side tweak moves Standard silently.
   That is the same "two masters, one authority" shape recorded in
   [[project_report_repair_needs_named_offenders]], just inverted.
3. **A hand-picked characterization matrix is theatre.** Choosing the cases
   means characterizing what I already thought of, which is never the
   combination that breaks. The input domain here is *finite* — six dimension
   values, a small goal enum, two booleans, a handful of count boundaries — so
   it can be enumerated exhaustively instead of sampled.
4. **"Standard rarely reaches the shared selector" was a guess.** It is now
   measured, and the precise statement is much more useful:
   `_chart_type_for_visual_goal` returns non-`None` for *every* recognised
   `visual_goal`, so **Standard reaches `select_chart_type` only when the
   analyzer emitted no `visual_goal` at all.** For a composition goal Standard
   returns `bar` when `dimensions != {"share"}` (`chart_pipeline.py:1565`) and
   never pies a mixed set.

The first version also skipped shadow mode, which the phased-audit workflow
requires before any behaviour cutover.

### The general defect, stated without reference to pies

> **The report re-decides a question Standard already answers, using a subset
> of Standard's inputs and none of Standard's correctives.**

Charting is one instance. The guardrails reading context instead of the
question, the document plan re-deciding manifest membership, and the gate and
assembler judging length independently were the same shape. A fix that only
corrects the pie branch buys nothing against the next instance, so the
deliverable is a **mechanism that makes report-vs-Standard divergence
mechanically visible**, with the pie as its first application.

### The rule

**Where the report needs a decision Standard already makes, the report gets
its own copy plus a machine-checked equivalence to Standard — never a shared
mutable rule, and never an edit to Standard's path.**

Duplication is normally a smell, and the audit checklist asks about it
directly. It is the right trade here because the duplicate is *guarded by
construction*: the equivalence test fails the moment the two diverge in either
direction, which converts silent drift into a forced decision. It also leaves
room for the report to diverge where it legitimately must — it has omission
semantics (`REPORT_CHART_INCOMPATIBLE_UNITS`) that Standard has no counterpart
for.

Layering: **agree with Standard on what the shape wants, then apply the
report's own admission gate on top.** Only the first half is equivalence-tested.

---

## Phased plan

Hard constraint, checked at every phase: `git diff --stat` for
`agent/chart_pipeline.py` and `visualization/chart_selector.py` must be
**empty**. If a phase cannot hold that, stop and re-plan.

### Phase 0 — Measure, assume nothing — **DONE**

No production code changed. Answers, from the code:

**Q1. Which inputs reach `select_chart_type` from Standard?**
Exactly one combination: **`preferred_chart_family is None` and
`visual_goal is None`.** Proven exhaustively rather than sampled —
`_chart_type_for_visual_goal` returns non-`None` for every one of the seven
`VisualGoal` members (`trend`/`relationship`→line, `composition`/
`decomposition`→stackedbar|pie|bar, `ranking`→bar, `compare`→line|bar,
`threshold_scan`→line|bar), so its `return None` is reachable only when the
field is absent, and `visual_goal: Optional[VisualGoal] = None`
(`contracts/question_analysis.py:475`) makes absence a real state.
`preferred_chart_family` short-circuits earlier still (`chart_pipeline.py:1516`).

**Q2. Callers of `select_chart_type`.** Exactly two:
`chart_pipeline.py:1528` (Standard) and `report_charts.py:297` (report, via
`_composition_snapshot_type`). `main.py:189` imports it but never calls it —
ruff's `F401` is globally disabled for intentional re-exports, and nothing
imports these four names back out of `main`. Not a caller.

**Q3. How often does `preferred_chart_family` fire?** Not answerable from the
code — it is analyzer output. Unresolved, and it does not block: when it is
set, neither the goal rule nor the shared selector runs, so it can only
*reduce* the population this work touches.

**Q4. Does the report have a `visual_goal` equivalent?**
**Yes, and this reframes the defect.** `ReportChartPurpose` (`contracts/report.py:338`)
carries `TREND / COMPARISON / COMPOSITION / RELATIONSHIP / FORECAST / TABLE`,
overlapping `VisualGoal` on four members. The report already branches on it —
`if chart.purpose is ReportChartPurpose.COMPOSITION:` (`report_charts.py:701`).

So the report is **not** missing the goal. It has it, routes on it, and then
**discards it at the final step**: inside the composition branch it calls
`_composition_snapshot_type`, which asks the goal-*less* `select_chart_type`
the very question Standard answers with its goal-*aware* rule. The two differ
only in exactness, which is precisely the reported defect:

| For `has_time=False, has_categories=True` | Pure share set | Mixed set e.g. `{share, energy_qty}` |
|---|---|---|
| Standard goal rule (`chart_pipeline.py:1563`) | `pie` | **`bar`** |
| Shared fallback (`chart_selector.py:242`) | `pie` | **`pie`** ← the mixed-unit pie |

**Consequences for the plan.** The fix is smaller and safer than assumed: the
report does not need Standard's whole chain, only the composition rule its own
`purpose` already entitles it to. Both `_composition_snapshot_type` call sites
benefit — the second (`report_charts.py:765`) already treats a non-`pie` answer
as "chart the series over time as a line, which is what Standard renders for
this shape", so a corrected rule makes that comment true instead of aspirational.

**Risk note.** Phase 1's golden must still cover Standard's *whole* decision
surface, not just the composition branch, because Phase 2's equivalence test is
only as trustworthy as the golden behind it.

### Phase 1 — Freeze Standard mechanically — **DONE**

`tests/test_chart_type_decision_golden.py` + `tests/fixtures/chart_type_decision_golden.json`.
**6,400 entries, zero production changes.** Two suites: a semantic one over the
full powerset of the six `infer_dimension` values × eight goals (seven plus
`None`) × `has_time` × `has_categories` × three category-count boundaries, and
an override one holding the semantic core small so the short-circuit matrix
(explicit group type, explicit user type, preferred family) is itself
exhaustive. `_choose_chart_type` is the whole decision surface, so that is what
is pinned.

**Proven to discriminate, not assumed to.** Mutating the shared selector's
`"share" in dimensions` to `dimensions == {"share"}`:

- on the pie branch → 
  `cats=1|dims=eiopsx|goal=-|n=8|time=0: pie → bar`, caught;
- on the *stacked-bar* branch, which was mutated by accident first →
  `cats=1|dims=eiopsx|goal=-|n=1|time=1: stackedbar → line`, also caught.

The second is the argument for enumeration over curation: nobody would have
hand-written a case for a six-dimension set with time and categories, and the
net caught it anyway. Standard restored to an empty diff after both.

**Audit finding, recorded and deliberately not acted on.** The golden documents
that *Standard itself* answers `pie` for a fully mixed dimension set when
`visual_goal` is absent (`cats=1|dims=eiopsx|goal=-|n=8|time=0 → pie`). The
weak membership rule is therefore not purely a report problem — Standard is
exposed to it too, just only on the no-goal fall-through Phase 0 identified.
Fixing that is out of scope: the mandate is no impact on Standard, and this
golden is exactly what would have to change to do it. Raise it as its own
decision later, with its own evidence.

### Phase 2 — Give the report its own decision module — **DONE**

`agent/report_chart_rules.py` (report-only) + `tests/test_report_chart_rules.py`.
Nothing imports the module yet — verified by grep, not by intent.
`chart_pipeline.py` and `chart_selector.py`: **zero diff**.

**Phase 1's open risk resolved first.** The golden pins `_choose_chart_type`
only, so Phase 2 could not rely on it until the type decision was shown to be
the actual lever. Confirmed at rest against real column names:

| Columns | Inferred | Report answers today |
|---|---|---|
| `share_hydro, share_thermal, share_wind` | `{share}` | `pie` ✓ |
| `share_hydro, quantity_hydro` | `{energy_qty, share}` | **`pie`** ← the defect |
| `quantity_hydro, quantity_thermal` | `{energy_qty}` | `bar` ✓ |

The mixed-unit pie reproduces without a production run. The category-axis
branch is not at risk — `_chart_candidates` already narrows it to one series —
so the exposure is the temporal-pivot branch (`report_charts.py:753`), where
several numeric columns become slices of one whole and only the type decision
stands between mixed units and a pie.

**The rule, three lines, in Standard's order:** exact `{"share"}` under the
category ceiling → `pie`; a continuous measure (`price_tariff`/`xrate`) with no
share to anchor it → `line`, which is Standard's corrective pass and something
the report never ran; everything else → `bar`.

**Equivalence, and proof it discriminates.** The suite replays all 192 golden
points that ask the report's question (64 dimension sets × 3 counts at
`goal=composition, time=0, cats=1`) and asserts the copy matches Standard
exactly. Because an equivalence test that cannot fail is worthless, all three
rules were mutated and each was caught:

| Mutation | Failures |
|---|---|
| `== {"share"}` → `"share" in` | 5 |
| pie ceiling 8 → 9 | 6 |
| drop the `line` corrective | 9 |

A guard test also asserts the compared slice is neither empty nor trivial —
Standard must answer all three of `pie`/`bar`/`line` across it, or agreeing
with it would prove nothing.

**Process note.** The mutation loop reverted with `git checkout`, which silently
does nothing to an untracked file, so three mutations accumulated in the new
module before the state was noticed and restored. Commit a new file before
mutation-testing it.

### Phase 3 — Shadow — **SHIPPED, awaiting a run**

`_composition_snapshot_type` now computes both answers, logs
`REPORT_CHART_TYPE_DISAGREEMENT` (applied, shadow, dimensions, columns,
category_count) when they differ, and **returns the old answer unchanged**.
Both call sites go through this one function, so a single seam covers the
category-axis branch and the temporal-pivot branch. `chart_pipeline.py` and
`chart_selector.py`: **zero diff**.

**Phase 2's open risk resolved first.** The rule assumes the period has
collapsed, so both call sites had to be confirmed as snapshot questions:
`report_charts.py:709-721` filters to `latest_period` before asking, and
`report_charts.py:754` pivots `rows[-1]`. Both single-period. `has_time=False`
is correct.

Two tests pin the shadow guarantee: a mixed set still returns `pie` while
logging `shadow: "bar"`, and an agreeing set logs nothing — a line on every
chart would drown the signal.

**Predictions, recorded before the run so the review can falsify them.**
From job `fbc46aa4`:

- `generation_and_cross_border_flows_composition` — built `pie`, `series_count: 1`,
  so its dimension set is almost certainly `{share}` alone. **Expect no
  disagreement.**
- `prices_and_balancing_composition` — logged `Chart type: bar (categorical
  comparison, no time)`, so the type decision ran and found no `share`. If its
  columns carry `price_tariff`, the corrective makes the shadow `line`.
  **Expect one disagreement, `bar` → `line`.**

If the run shows a disagreement neither prediction covers, that is the
interesting case and it gets read before anything is cut over.

### Phase 3 result — the shadow falsified the rule. **Cutover blocked.**

Job `e4049b2d` produced the first disagreement, and it says the rule is right
but the *input* is corrupt:

```
REPORT_CHART_TYPE_DISAGREEMENT {"applied":"bar","shadow":"line","category_count":1,
 "columns":["Balancing electricity price (GEL/MWh)","Balancing electricity price (USD/MWh)",
            "Price Deregulated Ren Gel","Price Deregulated Ren Usd",
            "Share Deregulated Ren","Share Import","Share Regulated Hpp"],
 "dimensions":["energy_qty","other","price_tariff"]}
```

Seven columns, three of them shares, and `share` is **not in the inferred set**.
The report is passing `infer_dimension` **display labels** instead of column
identifiers, and every single one is misclassified:

| Passed | Inferred | Identifier | Correct |
|---|---|---|---|
| `Balancing electricity price (GEL/MWh)` | `energy_qty` | `p_bal_gel` | `price_tariff` |
| `Share Deregulated Ren` | `other` | `share_deregulated_ren` | `share` |
| `Share Import` | `other` | `share_import` | `share` |
| `Price Deregulated Ren Gel` | `price_tariff` | `price_deregulated_ren_gel` | `price_tariff` |

A balancing **price** is read as an energy quantity because its label ends in
"MWh". Shares are read as "other" because the label is title-cased prose.

**This is the actual root cause of the charting complaints**, and it subsumes
the ones this plan was written against:

- the mixed-unit pie — the `{"share"}` exactness test can never pass on a
  dimension set that never contains `share`, and the membership test fires on
  noise instead;
- `REPORT_CHART_INCOMPATIBLE_UNITS` — `_axis_metadata` groups by
  `(dimension, unit)`, so garbage dimensions manufacture spurious axis groups
  and omit a chart that was fine;
- **my own Phase 2 rule would have made this case worse.** It answered `line`
  because it saw `price_tariff` and no `share` — a line chart of a
  one-category snapshot. The shadow is the only reason that did not ship.

**Where the labels come from.** `agent/report_research_execution._derived_chart_evidence_items`
builds manifest tables from `chart_override_specs`, whose row keys are the
chart's display labels. The code says so itself at line 957: *"its columns
still read 'Balancing electricity price (GEL/MWh)'"* — and works around it
**for units only**, via `declared_units`. That workaround was added earlier in
this session; it patched the unit half of this exact problem and left the
dimension half, which is precisely the partial fix that was supposed to be
avoided. The spec keeps no record of the source identifiers
(`metadata` holds only `title`, `xAxisTitle`, `yAxisTitle`, `axisMode`,
`role`, `type`, `labels`, `data`), so they are gone by the time the report
sees them.

**Two candidate fixes, and the choice matters for the mandate:**

- **A — preserve the mapping at the source.** `dispatch_derived_chart` emits a
  label→identifier map in `metadata`. Correct and complete, but
  `agent/derived_chart_builder.py` is **shared** with Standard. Purely
  additive (Standard reads none of the new key), yet it breaks the zero-diff
  rule this work has held for four phases.
- **C — invert the labelling in the report.** `_field_label` is a pure
  function over two finite dictionaries plus a title-case fallback, so it can
  be inverted report-side and the inverse can be exhaustively tested against
  both dictionaries. Report-only, zero diff to Standard, fragile only where a
  label is ambiguous — which a test over the full dictionary can prove or
  disprove up front.

Recommendation: **C**, on the mandate. Establish the inverse and prove it total
over `COLUMN_LABELS`/`DERIVED_LABELS` before using it; fall back to the label
unchanged where inversion is not provably unique, so the worst case is today's
behaviour.

### Phase 3b + 4 — Label inverse and cutover, shipped together — **DONE**

Option **C**. `agent/report_chart_rules` now owns the label map *and* its
inverse, so the two cannot drift, and `report_charts` imports rather than
keeping a second copy.

**The inverse was proven before it was used**, as its own gated step: 96 known
labels, **zero collisions**, and a test asserts
`evidence_column_identifier(field_label(x)) == x` for every one. Unknown text
falls back to its snake_case form and then to itself, so anything the inverse
cannot place behaves exactly as it does today.

All three report dimension-inference sites now read the identifier:
`_axis_metadata`, `_composition_snapshot_type`, and `_plottable_series`.

**Why the cutover could not wait for its own commit.** Recovering the
identifiers makes the dimension set truthful — and the *old* membership rule
answers `pie` for `{price_tariff, share}` the moment it can see the share it
was previously blind to:

| | dimensions seen | old rule | new rule |
|---|---|---|---|
| before the inverse | `{energy_qty, other, price_tariff}` | `bar` (by accident) | `line` (wrong) |
| after the inverse | `{price_tariff, share}` | **`pie`** (the defect) | `bar` (correct) |

Shipping the label fix alone would have *introduced* the mixed-unit pie it
exists to prevent, because the garbage dimensions had been suppressing it by
accident. The two land together or not at all. A test pins exactly this, and a
second pins that a pure-share composition is still a pie — the cutover must not
cost the compositions that already worked.

`REPORT_CHART_TYPE_DISAGREEMENT` is kept, with `applied`/`previous` swapped, so
the effect of the change stays visible in production rather than going dark at
cutover.

### Phase 4 — Cutover (superseded, folded into 3b above)

Switch the report to the new module once the disagreements are understood and
each is an improvement. Phase 1 golden unchanged, Standard files still zero
diff.

### Phase 5 — Re-validate Tasks 1–3

Re-run the premises below against post-cutover behaviour and delete whatever
the intervening fixes already closed. Do not execute a task whose premise no
longer reproduces.

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

## Tasks 1-3 — re-validated and closed, not executed (Phase 5)

Every premise was re-tested at rest against post-cutover behaviour. None
reproduces. Executing them would have added machinery for defects that no
longer exist.

| Task | Premise | Re-validation | Verdict |
|---|---|---|---|
| **1** — a wide composition must plot one kind of quantity, via a new `_composition_slice_columns` | shares and quantities pied as slices of one whole | wide frame with `share_hydro` + `quantity_hydro` now builds **line, dual axis** | **Closed.** A pie only fires when *every* column is a share, so slices are homogeneous by construction. The filter is unnecessary. |
| **2** — the category-axis branch needs the same rule | a melted frame with a share and a quantity per category hits it by another route | categorical frame with `share_tech` + `quantity_tech` now builds **bar, dual axis** | **Closed.** Same reason; two units on two axes is a legitimate rendering, not the defect. |
| **3** — find out which branch logs "1 categories" | the code was identical for two branches | shipped in `6c7e723` as the `detail` object naming `branch`, row count, and columns | **Closed.** This *was* the deliverable. |

**Both user-facing symptoms verified closed at rest:**

- *"the pie chart combined shares, thousand MWh dimensions"* — a pie now
  requires `dimensions == {"share"}`; the control case (pure shares by
  category) still pies, so the good shape was not lost.
- *"one is not rendered"* — job `e4049b2d` omitted
  `prices_balancing_analysis_composition` as `REPORT_CHART_INCOMPATIBLE_UNITS`.
  The evidence was fine; the dimensions were not. Label-reading produced three
  spurious groups and `_axis_metadata` refuses more than two. With identifiers
  it is two groups and a dual axis, and the chart builds.

Both are pinned by regression tests rather than left to the next run.

**What Phase 5 did not close.** The 8-series charts are untouched — series
*count* is a different question from series *compatibility*, and no evidence
yet says eight is wrong. The granularity fallback seen on job `4ea18b2b`
(`unsupported_tool_granularity_hint:day` sending a prices track to composition
data) is a routing fault upstream of charting and belongs to its own
investigation.

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
