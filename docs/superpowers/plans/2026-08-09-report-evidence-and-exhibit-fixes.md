# Report Evidence and Exhibit Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the report pipeline discarding its first writer call, make every
evidence table citable, and make exhibit candidates obey the rules the chart
builder actually enforces.

**Architecture:** Every defect below is the same shape — a component knows
something (a unit, a transform, an axis rule, a validation location) and drops
it at a module boundary, leaving the next component to guess from a string or
fail without saying why. Each task moves the knowledge to where the decision is
made, or names it in telemetry when it cannot yet be moved. No task adds a new
inference rule.

**Tech Stack:** Python 3.11 (container) / 3.14 (local), pydantic v2 strict
contracts, pytest.

## Global Constraints

- Gate for every task: `python -m pytest tests/ -q`, `ruff check .`, and
  `python -m guardrails.redteam_gate` (score must stay ≥ 0.92).
- Run everything from `D:\Enaiapp\langchain_railway`, not from `D:\Enaiapp`.
- Pre-set env before pytest so `config` imports:
  `SUPABASE_DB_URL=postgresql://user:pass@localhost/db`,
  `ENAI_GATEWAY_SECRET=test-gateway-key`,
  `ENAI_SESSION_SIGNING_SECRET=test-session-key`,
  `ENAI_EVALUATE_SECRET=test-evaluate-key`, `MODEL_TYPE=openai`,
  `OPENAI_API_KEY=test-openai-key`, `NVIDIA_API_KEY=test-nvidia-key`.
  Use these exact values — `tests/test_main.py` asserts against them, and
  shorter secrets fail 16 auth tests for the wrong reason.
- Telemetry may carry field names, enum values and identifiers from our own
  contracts. It must never carry query text, evidence values, or pydantic
  error messages/inputs.
- Every test must be shown to fail before the fix and pass after. A test that
  passes both ways is a plan failure, not a completed task.
- Branch: `agent/track-scoped-report-analysis` (current HEAD `4522996`).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `utils/validation_diagnostics.py` | Single authority for turning a `ValidationError` into safe field locations | **Create** |
| `core/report_job_processor.py` | Job lifecycle; already has a private copy of the above | Modify — delegate |
| `agent/report_document_generation.py` | Document gate; swallows two `ValidationError`s | Modify — name the offenders |
| `agent/report_research_execution.py` | Builds evidence items and exhibit candidates from a track's pipeline context | Modify — long-form units, categorical axis, unresolvable metrics |
| `agent/evidence_derivation.py` | Decides which requested metrics went unmet | Modify — separate unresolvable from unmet |
| `agent/report_research_planner.py` | Chooses the planner's topic knowledge | Modify — drop the chat fallback |
| `core/llm.py` | Planner prompt assembly | Modify — add the topic catalog |

---

## Task 1: Name the fields that invalidate a section batch

The first generative call of every report is thrown away. On jobs `40e55527`
and `5cb4d210` the `report_analysis_writer` returned, and one millisecond later
the gate logged `DOCUMENT_SCHEMA_INVALID` with `word_count: 0` and
`section_word_counts: {}` — the signature of the bare `except ValidationError`
in `_validate_section_batch`. Two repair calls then followed, so three of the
five budgeted calls went to producing sections the first call was asked for.

The cause is not yet knowable: `ReportDocumentRepair.model_validate` can reject
for a missing per-section `contract_version`, a duplicate paragraph
(`ReportSectionDraft._validate_paragraph_uniqueness`), or a duplicate section id
— and the code discards `exc.errors()` without reading it. This task does not
guess. It makes the next production run say which.

**Files:**
- Create: `utils/validation_diagnostics.py`
- Modify: `core/report_job_processor.py:193-216`
- Modify: `agent/report_document_generation.py:459-465`, `:1116-1124`
- Test: `tests/test_validation_diagnostics.py` (create),
  `tests/test_report_document_pipeline_v2.py`

**Interfaces:**
- Produces: `validation_error_locations(exc: Exception) -> list[str]` in
  `utils/validation_diagnostics.py`, returning up to 8 dotted `loc` paths
  (`"sections.0.paragraphs.2.text"`), de-duplicated, in first-seen order.
  Returns `[]` for any exception without a usable `errors()`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test for the shared helper**

Create `tests/test_validation_diagnostics.py`:

```python
"""The one place a ValidationError becomes safe telemetry."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError


class _Inner(BaseModel):
    text: str


class _Outer(BaseModel):
    items: list[_Inner]


def test_locations_name_the_rejected_fields_and_nothing_else():
    from utils.validation_diagnostics import validation_error_locations

    with pytest.raises(ValidationError) as caught:
        _Outer.model_validate({"items": [{"text": 1}, {}]})

    locations = validation_error_locations(caught.value)

    assert locations == ["items.0.text", "items.1.text"]


def test_a_non_pydantic_exception_yields_no_locations():
    """Callers log this on any failure path; it must never raise itself."""

    from utils.validation_diagnostics import validation_error_locations

    assert validation_error_locations(RuntimeError("boom")) == []
```

- [ ] **Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_validation_diagnostics.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'utils.validation_diagnostics'`.

- [ ] **Step 3: Create the helper**

Create `utils/validation_diagnostics.py`:

```python
"""Turn a pydantic ValidationError into telemetry that carries no data.

Error locations are field names and indices from our own contracts. The
messages and inputs beside them are the rejected values themselves, which is
why only ``loc`` is ever read.
"""

from __future__ import annotations

_MAXIMUM_REPORTED_LOCATIONS = 8


def validation_error_locations(exc: Exception) -> list[str]:
    """Return the dotted field paths a ValidationError rejected."""

    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return []
    try:
        raw = errors()
    except Exception:  # pragma: no cover - defensive
        return []
    located: list[str] = []
    for entry in list(raw)[:_MAXIMUM_REPORTED_LOCATIONS]:
        location = ".".join(
            str(part) for part in (entry or {}).get("loc", ()) if part != ""
        )
        if location and location not in located:
            located.append(location)
    return located
```

- [ ] **Step 4: Run it and watch it pass**

```bash
python -m pytest tests/test_validation_diagnostics.py -q
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Delegate the job processor's private copy**

In `core/report_job_processor.py`, replace the body of
`_diagnostic_error_locations` (lines 193-216) with a delegation, keeping its
`_diagnostic_identifier` sanitisation so that call site's behaviour is
unchanged:

```python
def _diagnostic_error_locations(exc: Exception) -> list[str]:
    """Name the schema fields a ValidationError rejected, never their values.

    Sanitised through ``_diagnostic_identifier`` because this line is emitted
    on the job's own failure telemetry, where the identifier vocabulary is
    fixed.
    """

    located: list[str] = []
    for location in validation_error_locations(exc):
        candidate = _diagnostic_identifier(location)
        if candidate not in located:
            located.append(candidate)
    return located
```

Add to the import block:

```python
from utils.validation_diagnostics import validation_error_locations
```

- [ ] **Step 6: Write the failing test for the document gate**

Append to `tests/test_report_document_pipeline_v2.py`:

```python
def test_an_invalid_section_batch_names_the_fields_that_rejected_it(caplog):
    """DOCUMENT_SCHEMA_INVALID with no locations is unfixable from a log.

    Jobs 40e55527 and 5cb4d210 both discarded the analysis writer's whole
    output one millisecond after it returned, then spent two repair calls
    rebuilding it. The code caught ValidationError and dropped exc.errors(),
    so three runs could not say whether the cause was a missing
    contract_version, a duplicate paragraph, or a duplicate section id.
    """

    import json
    import logging

    from agent.report_document_generation import _validate_section_batch

    (
        research_plan,
        _packets,
        manifest,
        _a,
        _b,
        document_plan,
    ) = _document_components()

    with caplog.at_level(logging.WARNING, logger="Enai.ReportDocument"):
        sections, validation = _validate_section_batch(
            {"contract_version": "report-document-repair-v1", "sections": [{}]},
            document_plan,
            manifest,
            section_ids=[document_plan.sections[0].section_id],
            research_plan=research_plan,
        )

    assert sections is None
    assert validation.document_errors == ["DOCUMENT_SCHEMA_INVALID"]
    logged = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("REPORT_DOCUMENT_SCHEMA_INVALID ")
    ]
    assert logged, "the rejection was not reported"
    assert logged[0]["invalid_fields"], "no field was named"
    assert logged[0]["stage"] == "section_batch"
```

- [ ] **Step 7: Run it and watch it fail**

```bash
python -m pytest tests/test_report_document_pipeline_v2.py -q -k "names_the_fields"
```

Expected: FAIL — `AssertionError: the rejection was not reported`.

- [ ] **Step 8: Name the offenders at both sites**

In `agent/report_document_generation.py`, add the import:

```python
from utils.validation_diagnostics import validation_error_locations
```

Add this helper directly above `_validate_section_batch`:

```python
def _log_schema_rejection(
    exc: Exception,
    *,
    stage: str,
    section_ids: Sequence[str],
) -> None:
    """Record which fields invalidated a model payload.

    ``DOCUMENT_SCHEMA_INVALID`` is the only blocking code with no named
    offender, which is why three production runs could not say what was wrong
    with the writer's output.
    """

    _LOGGER.warning(
        "REPORT_DOCUMENT_SCHEMA_INVALID %s",
        json.dumps(
            {
                "expected_section_ids": list(section_ids),
                "invalid_fields": validation_error_locations(exc),
                "stage": stage,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
```

Change the `except` at line 459 from `except ValidationError:` to:

```python
    except ValidationError as exc:
        _log_schema_rejection(exc, stage="section_batch", section_ids=expected_ids)
```

Change the `except` at line 1116 from `except ValidationError:` to:

```python
    except ValidationError as exc:
        _log_schema_rejection(
            exc,
            stage="whole_document",
            section_ids=[section.section_id for section in plan.sections],
        )
```

If `json` is not already imported in this module, add `import json` to the
stdlib block.

- [ ] **Step 9: Run the test and the module's suite**

```bash
python -m pytest tests/test_report_document_pipeline_v2.py tests/test_validation_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 10: Full gate and commit**

```bash
python -m pytest tests/ -q
```

```bash
git add -A && git commit -m "fix(report): name the fields that invalidate a section batch"
```

- [ ] **Step 11: Deploy and capture one run**

This task's deliverable is the next production log line. Ask for
`REPORT_DOCUMENT_SCHEMA_INVALID` from one May 2026 report. **Do not start
Task 1b until that line exists** — the fix depends entirely on which fields it
names.

---

## Task 1b: Fix what Task 1 names

Blocked on the production line from Task 1 Step 11. Written here so the
sequence is explicit, not so it can be started early.

The three candidate causes and the fix each implies:

| `invalid_fields` shows | Cause | Fix |
|---|---|---|
| `sections.N.contract_version` | `payload_bindings` binds `contract_version` at the top level only; `ReportSectionDraft` requires its own | Bind `contract_version` into each section in `_invoke_report_document_contract`, the same way the top level is bound |
| `sections.N.paragraphs` | Two paragraphs in one section share text; `_validate_paragraph_uniqueness` rejects the batch | The writer prompt must forbid repeating a paragraph, and the gate should reject the section rather than the whole batch |
| `sections` (length or ids) | The writer returned a different section set than `section_ids` | Already covered by `SECTION_SET_MISMATCH`; investigate why it reached the schema path instead |

---

## Task 2: A long-form derived frame declares its measure unit

Four evidence tables per report are still unciteable, all with
`columns=value`. They come from the anchored-explanation builder
(`agent/derived_chart_builder.py:771-799`), which melts share columns into
`{date, category, value}` rows. The column is called `value`, so name-based
inference has nothing to work with — but the builder already knows what it
holds and writes it into `metadata.yAxisTitle` as `"Share"`.

This extends the override added in commit `2e52cec` for percent transforms
rather than adding a second mechanism, and reuses the `_LABEL_DECLARED_UNITS`
vocabulary already introduced there, so there is one table of unit spellings.

**Files:**
- Modify: `agent/report_research_execution.py` (`_derived_chart_evidence_items`)
- Modify: `agent/report_evidence.py` (export `_LABEL_DECLARED_UNITS` lookup)
- Test: `tests/test_report_research_execution.py`

**Interfaces:**
- Consumes: `make_report_table_evidence_item(..., unit_by_column=...)` from
  commit `2e52cec`, and `_LABEL_DECLARED_UNITS` from `agent/report_evidence.py`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report_research_execution.py`:

```python
def test_a_long_form_frame_takes_its_unit_from_the_axis_it_declares():
    """A melted composition frame calls its measure column "value".

    Name inference has nothing to read, so on jobs 40e55527 and 5cb4d210 four
    evidence tables per report declared no unit and nothing in them could be
    cited. The builder already knows: it writes "Share" into yAxisTitle.
    """

    def query_pipeline(query, **_kwargs):
        return QueryContext(
            query=query,
            cols=["date", "share_hydro"],
            rows=[["2026-04", 0.61], ["2026-05", 0.72]],
            provenance_cols=["date", "share_hydro"],
            provenance_rows=[["2026-04", 0.61], ["2026-05", 0.72]],
            provenance_refs=["query:track:composition"],
            provenance_source="pipeline",
            stats_hint="Hydro share rose over the month.",
            chart_override_specs=[
                {
                    "type": "stackedbar",
                    "data": [
                        {"date": "2026-05", "category": "Share Hydro", "value": 0.72},
                        {"date": "2026-05", "category": "Share Thermal", "value": 0.28},
                    ],
                    "metadata": {
                        "title": "Composition: focus periods",
                        "yAxisTitle": "Share",
                        "role": "component_primary",
                    },
                }
            ],
            answer_mode="report",
        )

    packet = execute_report_track_analysis(
        _QUERY,
        _plan().tracks[0],
        query_pipeline=query_pipeline,
    )

    melted = next(
        item
        for item in packet.items
        if item.kind is ReportEvidenceKind.TABLE
        and "value" in item.columns
    )
    assert melted.unit_by_column.get("value") == "ratio"
    assert melted.citable_numeric_columns() == ["value"]
```

- [ ] **Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_report_research_execution.py -q -k "long_form_frame"
```

Expected: FAIL — `assert None == 'ratio'`.

- [ ] **Step 3: Publish the unit vocabulary**

In `agent/report_evidence.py`, add below `_LABEL_DECLARED_UNITS`:

```python
def declared_unit_spelling(label: str) -> str:
    """Return the manifest's spelling of a unit named in prose, or "".

    The axis title a chart builder writes is the same vocabulary a column
    label carries in parentheses, so both resolve through one table.
    """

    return _LABEL_DECLARED_UNITS.get(" ".join(str(label or "").lower().split()), "")
```

- [ ] **Step 4: Use it for the measure column**

In `agent/report_research_execution.py`, inside `_derived_chart_evidence_items`,
replace the `percent_units` block with:

```python
        transform = str(metadata.get("measureTransform") or "").lower()
        numeric_columns = [
            column
            for column in columns
            if any(
                isinstance(row.get(column), Real)
                and not isinstance(row.get(column), bool)
                for row in rows
            )
        ]
        if any(token in transform for token in ("pct", "percent")):
            # A percent-change panel keeps the labels of the levels it was
            # computed from, so inference would declare those levels' units.
            declared_units = {column: "%" for column in numeric_columns}
        else:
            # A melted frame names its measure column "value" and puts what it
            # measures on the axis instead.
            axis_unit = declared_unit_spelling(metadata.get("yAxisTitle", ""))
            declared_units = (
                {column: axis_unit for column in numeric_columns}
                if axis_unit and len(numeric_columns) == 1
                else {}
            )
```

and pass `unit_by_column=declared_units` to `make_report_table_evidence_item`.

Add `declared_unit_spelling` to the existing
`from agent.report_evidence import (...)` block.

- [ ] **Step 5: Run the test and watch it pass**

```bash
python -m pytest tests/test_report_research_execution.py -q -k "long_form_frame or change_panel"
```

Expected: PASS, 2 tests. The `change_panel` test guards that the percent case
still wins over the axis case.

- [ ] **Step 6: Full gate and commit**

```bash
python -m pytest tests/ -q
```

```bash
git add -A && git commit -m "fix(report): a melted frame takes its unit from the axis it declares"
```

---

## Task 3: A composition exhibit picks an axis the builder accepts

Job `5cb4d210` omitted `balancing_price_and_composition_composition` with
`REPORT_CHART_CATEGORY_REQUIRED`. `_chart_candidates` derives its own
`dimension_fields` (everything non-numeric) and falls back to
`dimension_fields[0]` when no preferred token matches — which can be the date
column. `build_report_chart_requests` then classifies columns separately into
`numeric` / `temporal` / `categorical` and, for a composition chart with a
categorical column present, requires `x_field` to be one
(`agent/report_charts.py:704`). Two authorities, one disagreement, one dropped
exhibit.

`chart_column_roles` already exists for exactly this — "Expose the builder's own
axis typing so planning can respect it" — and `core/llm.py:3896` already uses
it. `_chart_candidates` does not.

**Files:**
- Modify: `agent/report_research_execution.py` (`_chart_candidates`)
- Test: `tests/test_report_research_execution.py`

**Interfaces:**
- Consumes: `chart_column_roles(item) -> dict[str, list[str]]` from
  `agent/report_charts.py:355`, returning keys `numeric`, `temporal`,
  `categorical`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report_research_execution.py`:

```python
def test_a_composition_exhibit_picks_an_axis_the_builder_accepts():
    """A melted frame has both a period and a category column.

    The candidate builder took the first dimension it found, which was the
    date, and the chart builder then refused the exhibit because a composition
    chart needs a categorical axis. Job 5cb4d210 lost its balancing
    composition chart to that disagreement.
    """

    from agent.report_charts import chart_column_roles

    payload = _plan().tracks[1].model_dump(mode="json")
    payload["expected_exhibits"] = ["composition"]
    track = ReportResearchTrack.model_validate(payload)

    def query_pipeline(query, **_kwargs):
        return QueryContext(
            query=query,
            cols=["date", "share_hydro"],
            rows=[["2026-05", 0.72]],
            provenance_cols=["date", "share_hydro"],
            provenance_rows=[["2026-05", 0.72]],
            provenance_refs=["query:track:composition"],
            provenance_source="pipeline",
            stats_hint="Hydro dominated the mix.",
            chart_override_specs=[
                {
                    "type": "stackedbar",
                    "data": [
                        {"date": "2026-05", "category": "Share Hydro", "value": 0.72},
                        {"date": "2026-05", "category": "Share Thermal", "value": 0.28},
                    ],
                    "metadata": {
                        "title": "Composition: focus periods",
                        "yAxisTitle": "Share",
                    },
                }
            ],
            answer_mode="report",
        )

    packet = execute_report_track_analysis(
        _QUERY,
        track,
        query_pipeline=query_pipeline,
    )

    candidate = next(
        entry
        for entry in packet.chart_candidates
        if entry.purpose.value == "composition"
    )
    item = {
        entry.evidence_ref: entry for entry in packet.items
    }[candidate.evidence_refs[0]]
    # The rule the builder applies, asserted against what the candidate chose.
    assert candidate.x_field in chart_column_roles(item)["categorical"]
```

- [ ] **Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_report_research_execution.py -q -k "axis_the_builder_accepts"
```

Expected: FAIL — `assert 'date' in ['category']`.

- [ ] **Step 3: Take the axis typing from the builder**

In `agent/report_research_execution.py`, add to the imports:

```python
from agent.report_charts import chart_column_roles
```

Inside `_chart_candidates`, replace the `numeric_fields` / `dimension_fields`
derivation in the ranked-tables loop (the second occurrence, inside
`for _table_index, item in ranked_tables:`) with:

```python
            roles = chart_column_roles(item)
            numeric_fields = roles["numeric"]
            # A composition chart slices one whole by category, so a period
            # column cannot be its axis; every other purpose plots against
            # time when it can.
            dimension_fields = (
                roles["categorical"]
                if purpose is ReportChartPurpose.COMPOSITION
                else [*roles["temporal"], *roles["categorical"]]
            )
            if not numeric_fields or not dimension_fields:
                continue
```

and delete the now-unreachable `if not numeric_fields: continue` below it.

- [ ] **Step 4: Run the test and watch it pass**

```bash
python -m pytest tests/test_report_research_execution.py tests/test_report_charts.py -q
```

Expected: PASS. If a previously-passing exhibit test now fails, read whether
the exhibit it expected was one the builder would have refused; fix the test's
expectation only when that is demonstrably true, never to make it green.

- [ ] **Step 5: Full gate and commit**

```bash
python -m pytest tests/ -q
```

```bash
git add -A && git commit -m "fix(report): a composition exhibit picks an axis the builder accepts"
```

---

## Task 4: A metric that names no column is not a coverage gap

Job `5cb4d210` marked `generation_mix_and_cross_border_flows` PARTIAL with
`MISSING_DERIVED_METRIC_MOM_ABSOLUTE_CHANGE` and
`..._MOM_PERCENT_CHANGE`. The analyzer requested those against a metric column
the fetched frame does not contain, so `dispatch_metric` returned `None` and
`missing_requested_evidence` reported them as unmet.

A gap means "evidence the report should have had and did not". A metric naming
a column that was never in the frame is a different fact — the analyzer asked
for something uncomputable — and conflating them degrades a track that
collected everything it was asked to collect. This task separates them: the
unresolvable request is logged by name, and only genuinely unmet metrics
produce a gap.

**Files:**
- Modify: `agent/evidence_derivation.py:30-41`
- Test: `tests/test_evidence_derivation.py`

**Interfaces:**
- Produces: `unresolvable_requested_metrics(ctx) -> list[str]` in
  `agent/evidence_derivation.py`, naming requested derived metrics whose
  `metric` column is absent from `ctx.df`.
- `missing_requested_evidence(ctx) -> list[str]` keeps its signature and stops
  returning unresolvable names.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evidence_derivation.py`:

```python
def test_a_metric_naming_an_absent_column_is_not_a_coverage_gap():
    """A gap means evidence we should have had, not a request we could not read.

    Job 5cb4d210 marked a track PARTIAL for two MoM metrics the analyzer
    requested against a column the fetched frame never contained. The track had
    collected everything it was asked to collect.
    """

    import pandas as pd

    from agent import evidence_derivation
    from models import QueryContext

    ctx = QueryContext(
        query="What were the generation mix, imports, and exports in May 2026?",
        df=pd.DataFrame(
            {
                "date": ["2026-04", "2026-05"],
                "quantity_hydro": [1.0, 2.0],
            }
        ),
        requested_derived_metrics=["mom_absolute_change"],
        analysis_evidence=[],
    )
    ctx.analysis_requirement_metrics = {"mom_absolute_change": "generation"}

    assert evidence_derivation.unresolvable_requested_metrics(ctx) == [
        "mom_absolute_change"
    ]
    assert evidence_derivation.missing_requested_evidence(ctx) == []
```

- [ ] **Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_evidence_derivation.py -q -k "absent_column"
```

Expected: FAIL — `AttributeError: module 'agent.evidence_derivation' has no attribute 'unresolvable_requested_metrics'`.

- [ ] **Step 3: Read how the request's column is reachable from ctx**

```bash
grep -n "requested_derived_metric_names" -A 25 agent/evidence_derivation.py
```

`requested_derived_metric_names` reads `metric_name` off each
`DerivedMetricRequest`; the column it applies to is the sibling `metric` field.
Confirm the accessor before writing Step 4, and use the same traversal — do not
add a second way to read the analyzer's requests.

- [ ] **Step 4: Separate the two facts**

In `agent/evidence_derivation.py`:

```python
def _requested_metric_columns(ctx: QueryContext) -> dict[str, str]:
    """Map each requested derived metric to the column it applies to."""

    analysis = getattr(ctx, "question_analysis", None)
    requirements = getattr(analysis, "analysis_requirements", None)
    columns: dict[str, str] = {}
    for request in (getattr(requirements, "derived_metrics", None) or []):
        name = str(getattr(request.metric_name, "value", request.metric_name) or "")
        column = str(getattr(request, "metric", "") or "")
        if name and column and name not in columns:
            columns[name] = column
    return columns


def unresolvable_requested_metrics(ctx: QueryContext) -> list[str]:
    """Return requested metrics whose column is absent from the frame.

    Distinct from a coverage gap: nothing was lost, the request could not be
    read against the evidence that was fetched.
    """

    frame = getattr(ctx, "df", None)
    if frame is None or not len(getattr(frame, "columns", [])):
        return []
    available = set(frame.columns)
    return [
        name
        for name, column in _requested_metric_columns(ctx).items()
        if column not in available
    ]
```

and in `missing_requested_evidence`, after `evidence_names` is built:

```python
    unresolvable = set(unresolvable_requested_metrics(ctx))
    return [
        name
        for name in requested
        if name not in evidence_names and name not in unresolvable
    ]
```

- [ ] **Step 5: Report the unresolvable request where the gap used to appear**

In `agent/pipeline.py`, beside the existing
`ctx.missing_evidence_for_metrics = _missing_requested_evidence(ctx)`, add the
new list to the `evidence_readiness` trace so the fact is still visible:

```python
        unresolvable_requested_metrics=_unresolvable_requested_metrics(ctx),
```

importing it alongside `missing_requested_evidence` in the existing
`from agent.evidence_derivation import (...)` block as
`unresolvable_requested_metrics as _unresolvable_requested_metrics`.

- [ ] **Step 6: Run the tests**

```bash
python -m pytest tests/test_evidence_derivation.py tests/test_guardrails.py -q
```

Expected: PASS.

- [ ] **Step 7: Full gate and commit**

```bash
python -m pytest tests/ -q
```

```bash
git add -A && git commit -m "fix(analyzer): a metric naming an absent column is not a coverage gap"
```

---

## Task 5: The planner sees what the system knows

`_planning_topic_knowledge` resolves topics by substring-matching TOPIC_MAP
keywords against the request. "create a report about may 2026 - what happenned
in this month, what were changes in main areas" contains no market keyword and
does not match the conceptual patterns (`what is`, `what are`, `define`,
`explain`), so it falls through to the chat default
`{"balancing_price", "sql_examples"}`. The planner decomposing a whole-market
monthly report is handed balancing-price notes and SQL query examples, and
nothing on tariffs, generation mix, cross-border trade, or market structure —
the topics all four of its tracks then covered.

It also cannot see that those files exist. `QUESTION_ANALYSIS_TOPIC_CATALOG`
goes to the analyzer as `UNTRUSTED_TOPIC_CATALOG`; the planner prompt carries
`COLLECTOR_CATALOG`, `OUTPUT_JSON_SCHEMA`, `MAX_RESEARCH_TRACKS`,
`MAX_TOTAL_EXHIBITS`, `REQUIRED_EXHIBITS`, `TOPIC_KNOWLEDGE`,
`USER_REPORT_REQUEST` — no topic catalog.

The catalog is byte-identical on every request, so it belongs in the constants
block where the provider's prefix cache absorbs it. It is also what a planner
actually needs: it allocates tracks, it does not answer questions, and per-track
vector retrieval still fetches the content afterwards.

**Files:**
- Modify: `core/llm.py:3806-3821` (planner prompt)
- Modify: `agent/report_research_planner.py:195-223`
- Test: `tests/test_report_research_planner_v2.py`

**Interfaces:**
- Consumes: `_TOPIC_CATALOG_JSON` (`core/llm.py:2421`), already built from
  `QUESTION_ANALYSIS_TOPIC_CATALOG`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_report_research_planner_v2.py`:

```python
def test_report_planning_does_not_fall_back_to_the_chat_default():
    """A broad report request matches no topic keyword.

    infer_topic_matches then returns the chat default
    {"balancing_price", "sql_examples"}, so the planner for a whole-market
    monthly report received balancing-price notes and SQL query syntax, and
    nothing about the tariffs, generation mix, or market structure its own
    tracks went on to cover.
    """

    from agent import report_research_planner

    knowledge = report_research_planner._planning_topic_knowledge(
        "create a report about may 2026 - what happenned in this month, "
        "what were changes in main areas"
    )

    assert "SELECT" not in knowledge.upper(), "SQL examples reached the planner"


def test_a_matched_topic_still_reaches_the_planner():
    """The fallback is the problem, not the lookup."""

    from agent import report_research_planner

    knowledge = report_research_planner._planning_topic_knowledge(
        "report on regulated tariff levels"
    )

    assert knowledge.strip(), "a keyword-matched request lost its knowledge"


def test_the_planner_prompt_lists_the_knowledge_topics(monkeypatch):
    """The planner cannot allocate a knowledge track for a topic it cannot see."""

    from core import llm

    captured: dict[str, str] = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after prompt assembly")

    monkeypatch.setattr(llm, "_invoke_report_document_contract", _capture)
    try:
        llm.llm_plan_report_research(
            "create a report about may 2026",
            language_code="en",
            max_tracks=4,
        )
    except Exception:
        pass

    assert "KNOWLEDGE_TOPIC_CATALOG:" in captured.get("prompt", "")
```

- [ ] **Step 2: Run them and watch the first and third fail**

```bash
python -m pytest tests/test_report_research_planner_v2.py -q -k "chat_default or lists_the_knowledge_topics or matched_topic"
```

Expected: `chat_default` FAILS (SQL reaches the planner),
`lists_the_knowledge_topics` FAILS (no catalog in the prompt),
`matched_topic` PASSES (it already works, and pins that Step 3 does not break it).

The third test asserts against the planner's own call path; if
`llm_plan_report_research` does not route through
`_invoke_report_document_contract`, capture the `prompt` local by patching
whatever it does call, and keep the assertion identical.

- [ ] **Step 3: Use report-shaped topic selection**

In `agent/report_research_planner.py`, replace the body of
`_planning_topic_knowledge` up to the budget clip with:

```python
    try:
        topics = _direct_topic_matches(query)
        if not topics:
            # infer_topic_matches falls back to a chat default
            # ({"balancing_price", "sql_examples"}) tuned for one question. A
            # broad report matches no keyword, and SQL syntax is noise to a
            # planner. Send nothing and let KNOWLEDGE_TOPIC_CATALOG carry it.
            return ""
        knowledge = get_knowledge_for_topics(sorted(topics), fallback_query="")
    except Exception:  # pragma: no cover - defensive
        return ""
```

and change the import of `infer_topic_matches` to also bring
`_direct_topic_matches` from `knowledge`.

- [ ] **Step 4: Put the catalog in the cacheable prefix**

In `core/llm.py`, in the `prompt` assembly at line 3806, insert after
`COLLECTOR_CATALOG`:

```python
        "KNOWLEDGE_TOPIC_CATALOG:\n"
        f"{_TOPIC_CATALOG_JSON}\n\n"
```

and extend the planner system message, after the sentence ending
"...retrieves only approved knowledge Markdown passages.":

```
KNOWLEDGE_TOPIC_CATALOG lists every topic the approved knowledge covers. Use
it to decide which knowledge tracks are worth planning; a topic in the catalog
can be retrieved even when TOPIC_KNOWLEDGE below is empty.
```

- [ ] **Step 5: Run the tests and watch them pass**

```bash
python -m pytest tests/test_report_research_planner_v2.py -q
```

Expected: PASS.

- [ ] **Step 6: Full gate and commit**

```bash
python -m pytest tests/ -q
```

```bash
git add -A && git commit -m "feat(report): plan against the knowledge topic catalog, not a chat fallback"
```

---

## Task 6: The comparison vocabulary comes from the catalog

Commit `2e52cec` added `delta`, `growth`, `index`, `mom`, `pct`, `yoy` to a
hand-maintained blacklist so a period-over-period frame stops outranking the
levels it was computed from. A blacklist is a symptom fix: a new
`DerivedMetricName` breaks exhibit selection silently. `DerivedMetricName` is a
closed catalog, so the comparison vocabulary can be derived from it.

**Files:**
- Modify: `agent/report_research_execution.py` (`_chart_candidates`)
- Test: `tests/test_report_research_execution.py`

**Interfaces:**
- Consumes: `DerivedMetricName` from `contracts/question_analysis.py`.
- Produces: `_COMPARISON_TOKENS: frozenset[str]` in
  `agent/report_research_execution.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report_research_execution.py`:

```python
def test_the_comparison_vocabulary_covers_every_catalogued_metric():
    """A hand-listed vocabulary rots the moment a metric name is added."""

    from contracts.question_analysis import DerivedMetricName
    from agent.report_research_execution import _COMPARISON_TOKENS

    for metric in DerivedMetricName:
        subject_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", metric.value.casefold())
            if token not in _COMPARISON_TOKENS
        }
        assert subject_tokens != set(), (
            f"{metric.value} has no subject left after removing comparisons"
        )
        assert "mom" not in subject_tokens
        assert "yoy" not in subject_tokens
```

Add `import re` at the top of the test module if it is not already imported.

- [ ] **Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_report_research_execution.py -q -k "comparison_vocabulary"
```

Expected: FAIL — `ImportError: cannot import name '_COMPARISON_TOKENS'`.

- [ ] **Step 3: Derive the vocabulary**

In `agent/report_research_execution.py`, replace the inline set inside
`_chart_candidates` with a module constant above it:

```python
# A requested metric names a subject and a comparison — "share_delta_mom" is
# the share, compared month over month. Only the subject should steer which
# table an exhibit is drawn from: on job 40e55527 the comparison words scored
# the month-on-month change panel above the levels it was computed from, and
# the balancing composition exhibit was built from one row of deltas, then
# omitted for having a single category. Derived from the catalog rather than
# listed by hand, so a new DerivedMetricName cannot quietly break selection.
_COMPARISON_TOKENS = frozenset(
    {
        "absolute",
        "average",
        "change",
        "delta",
        "growth",
        "index",
        "maximum",
        "minimum",
        "mom",
        "pct",
        "percent",
        "ratio",
        "slope",
        "trend",
        "yoy",
    }
)
```

and use it in the comprehension:

```python
    requested_tokens = {
        token
        for metric in requested_metrics
        for token in re.findall(r"[a-z0-9]+", metric.casefold())
        if token not in _COMPARISON_TOKENS
    }
```

Run the Step 1 test; if it reports a catalogued metric left with no subject
tokens, remove that token from the set rather than weakening the assertion —
the test exists to keep the vocabulary honest against the catalog.

- [ ] **Step 4: Run the tests and watch them pass**

```bash
python -m pytest tests/test_report_research_execution.py -q
```

Expected: PASS, including
`test_a_composition_exhibit_is_drawn_from_levels_not_from_changes`.

- [ ] **Step 5: Full gate and commit**

```bash
python -m pytest tests/ -q
```

```bash
git add -A && git commit -m "refactor(report): derive the comparison vocabulary from the metric catalog"
```

---

## Deferred, with rationale

**Report evidence sourced from the canonical frame.**
`contracts/evidence_frames.py::ObservationFrame` already carries
`period, entity_id, entity_label, metric, value, unit` per row, and the logs
show it built on every track (`evidence_finalization ... "mode": "shadow"`).
`agent/report_evidence.py` contains no reference to it: the manifest is built
from raw `ctx.cols`/`ctx.rows` and infers units from column names. Tasks 2 and
the commits before it are all fallback-path patches for that one omission.
Sourcing units from the frame would delete most of them — but it touches the
grounding contract, the frame is still in shadow, and there is an activation
runbook (`ENAI_EVIDENCE_FINALIZATION_MODE=enforce`) to respect. Separate plan,
after this one lands.

**The knowledge-routing slice of Task 2.2c.** Shadow data across jobs
`40e55527` and `5cb4d210` shows the planner right on `preferred_path` and
`query_type` for knowledge tracks (`market_design_context`, twice) and the
analyzer right on `derived_metrics` and `answer_kind`. That argues for adopting
the planner's routing only where it says `knowledge`. It is a routing change
and needs its own disagreement review, so it does not belong in a plan of
evidence fixes.

**Prefix caching of the analyzer fan-out.** The four
`report_question_analyzer` calls each carry ~11 K prompt tokens within ten
seconds and all report `cached_prompt_tokens=0`, because they run in parallel
and none has completed to populate an entry. Serialising the first would light
up the other three at a latency cost; the knowledge-routing slice above removes
those calls on knowledge tracks outright, which is strictly better. Revisit
only if that slice is rejected.

---

## Follow-on work (2026-08-09, after job `fe331b3c`)

**Item 2 — the knowledge-routing slice — is done** (`e248b8d`), and the run
confirmed it: `planner_knowledge_track_ids=["documented_market_context"]`, three
analyzer calls instead of four, no guardrail line, no 61-row price fetch, tokens
89,379 → 74,287. It is no longer deferred; the paragraph above is kept for the
reasoning that led to it.

**Item 3 — the two repair calls — is open, and starts diagnostically**
(`71f89c2`). `REQUIRED_EVIDENCE_NOT_USED` and `UNGROUNDED_NUMERIC_CLAIM` each
named a section and nothing else, so five runs could not say which ref went
uncited or which number went unbacked. `REPORT_DOCUMENT_DIAGNOSTIC` now carries
`uncited_required_refs` and `ungrounded_value_hints`, and both repair prompts
name the skipped assignments. **No repair-loop behaviour was changed** — the
next deployed run names the offenders, and the fix follows from what it names.
Every cause guessed at before it was named this session turned out wrong.

**The window-expansion guardrail is fixed** (`d63d61a`). The `Applied quantity
trend guardrail` line on `generation_and_cross_border_supply` was not a trend
question: three guardrails matched their *positive* conditions against the whole
track query, so the report context's "evolution" decided the route and the
month was replaced by the full history. Positive conditions now read
`_asked_question_surface`; negative brakes still read everything. Documented in
`docs/active/query_pipeline_architecture.md` §3.2.

**Still open:** the composition chart omitted for a fifth consecutive run —
covered by `2026-08-09-report-charting-fixes.md`, deferred by the user.

### The evidence caps do not fit each other (allocation fixed; caps unchanged)

`ReportEvidencePacket.items` holds up to **12**, `research_max_tracks` is
**4**, and `ReportEvidenceManifest.items` holds **32** with one slot always
spent on the limitation note — so consolidation keeps **31**. Three
well-populated tracks (36) already overflow; four (48) overflow badly. Job
`e3f43e84` discarded 17 items.

Consolidation fills the manifest in packet order, so the overflow falls
entirely on the **last** tracks: an early track keeps all twelve items while a
late one can reach the writer with none. That is the root cause of the
`document_plan_ready` crash — the crash itself is fixed (the plan no longer
references dropped evidence, `d68f030`) and the loss is now named by
`REPORT_MANIFEST_TRUNCATED`, but the allocation is still first-come.

**Round-robin shipped.** Consolidation now takes one item from every packet
before any packet takes seconds. Each packet already orders its own evidence
most-important-first, so a round takes each track's next-best item and the
shortfall is shared. On the three-track fixture, packet order kept
`{prices: 12, security: 12, market_model: 8}`; rounds keep an even split, and
a track with fewer items simply stops contributing and leaves its rounds to
the others. The whole suite passed unchanged, so nothing depended on the old
ordering — `evidence_ref` is a content hash, and section evidence is ordered
by track, not by manifest position.

The two alternatives were rejected: a **per-track quota of `31 // tracks`**
wastes slots whenever a track has fewer items, and **raising the manifest cap**
moves the problem downstream into the writer's prompt budget, which is where
it was tuned away from.

**The caps themselves are unchanged and still do not fit.** Round-robin shares
the loss fairly; it does not stop 48 items being cut to 31. Whether the packet
cap, the track cap, or the manifest cap is the wrong number is a sizing
question that wants real distribution data — `REPORT_MANIFEST_TRUNCATED` now
reports it every run.

**A second, independent way to lose evidence** lives in
`build_report_manifest_from_items`: a 768 KiB byte budget, also spent
first-come, so one wide table can starve everything after it and the item
count never predicts it. Now reported as `REPORT_MANIFEST_ITEM_OVERSIZED`.
Left first-come for now — unlike the item cap it rarely binds, and the log
will say if that is wrong.

---

## Self-Review

**Spec coverage.** The four items raised: wasted first writer call → Task 1
(+1b); `value` column → Task 2; unresolvable MoM → Task 4; composition
`CATEGORY_REQUIRED` → Task 3. The two items from the preceding discussion:
planner knowledge → Task 5; blacklist → Task 6. No item is unassigned.

**Placeholders.** Task 1b is the one task without code, and it is explicitly
gated on a production log line that does not exist yet; its three branches each
name the exact fix. Task 4 Step 3 is a verification step, not a placeholder —
it names the command and what to confirm.

**Type consistency.** `validation_error_locations(exc) -> list[str]` is used
identically in Tasks 1's three call sites. `declared_unit_spelling(label) -> str`
(Task 2) and `chart_column_roles(item) -> dict[str, list[str]]` (Task 3, existing)
match their consumers. `unresolvable_requested_metrics(ctx) -> list[str]`
(Task 4) matches its import alias in `agent/pipeline.py`.

**Ordering.** Tasks 2, 3, 4, 5, 6 are independent and may be done in any order
or in parallel. Task 1 should go first only because its deliverable is a
deployed log line, and the sooner it ships the sooner 1b is unblocked.
