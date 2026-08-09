"""The report's own copy of Standard's chart-type rules.

Standard answers "what should this shape render as" through a chain: the
analyzer's explicit family, then a rule keyed on ``visual_goal``, then a
goal-less fallback selector, then two corrective passes
(``agent/chart_pipeline._choose_chart_type``). The report has the equivalent of
a goal — ``ReportChartPurpose`` — and already routes on it, but at the final
step it asked the goal-*less* fallback the question Standard answers with its
goal-*aware* rule. The two differ in exactness, and that is the whole defect: a
composition whose columns infer to ``{"share", "energy_qty"}`` takes the
fallback's ``"share" in dimensions`` branch and pies shares and thousand MWh as
slices of one whole, where Standard's ``dimensions == {"share"}`` returns a bar.

**Why a copy and not a shared function.** Standard's chart output is the better
tested of the two and must not move. Extracting the rule into something both
call would edit Standard's call path, and would leave a mutable surface where a
later report-side tweak moves Standard silently. A copy paired with
``tests/test_report_chart_rules.py``'s equivalence check against the frozen
golden turns drift into a failing test instead: the duplicate is guarded by
construction, in both directions.

This module answers only the shared question — what the shape wants. The
report's own admission rules (``REPORT_CHART_INCOMPATIBLE_UNITS`` and the rest
of the omission machinery) have no Standard counterpart and stay in
``agent/report_charts.py``, layered on top of this answer.
"""

from __future__ import annotations

from collections.abc import Iterable

from context import COLUMN_LABELS, DERIVED_LABELS
from visualization.chart_selector import infer_dimension

_SUMMARY_FIELD_LABELS = {
    "first": "First value",
    "first_period": "First period",
    "largest_decrease": "Largest decrease",
    "largest_decrease_period": "Largest decrease period",
    "largest_increase": "Largest increase",
    "largest_increase_period": "Largest increase period",
    "last": "Last value",
    "last_period": "Last period",
    "mean": "Mean",
    "metric": "Metric",
    "maximum": "Maximum",
    "maximum_period": "Maximum period",
    "minimum": "Minimum",
    "minimum_period": "Minimum period",
    "observations": "Observations",
    "segment": "Segment",
    "std_dev": "Standard deviation",
}
KNOWN_FIELD_LABELS = {
    **COLUMN_LABELS,
    **DERIVED_LABELS,
    **_SUMMARY_FIELD_LABELS,
}


def field_label(field: str) -> str:
    """Render a column identifier for a human reader."""

    return KNOWN_FIELD_LABELS.get(
        field,
        field.replace("_", " ").strip().title(),
    )


# The inverse lives beside the map it inverts so the two cannot drift, and is
# proven total over every known label by
# tests/test_report_chart_rules.py. Built once: 96 labels, no collisions, so
# the mapping is unambiguous in both directions.
_IDENTIFIER_BY_LABEL = {
    label: identifier for identifier, label in KNOWN_FIELD_LABELS.items()
}


def evidence_column_identifier(column: str) -> str:
    """Recover the identifier a manifest column's display label came from.

    A derived-chart exhibit reaches the manifest with the chart's *labels* as
    its column names — ``_derived_chart_evidence_items`` says so itself, and
    already works around it for units. Everything that reads meaning out of a
    column name is wrong on those tables: on job e4049b2d a balancing price
    inferred ``energy_qty`` because its label ends in "MWh", and every share
    inferred ``other`` because the label is title-cased prose.

    Falls back to the label's own snake_case form, then to the label
    unchanged, so a name this cannot place behaves exactly as it does today.
    """

    text = str(column or "")
    mapped = _IDENTIFIER_BY_LABEL.get(text)
    if mapped is not None:
        return mapped
    return text.strip().lower().replace(" ", "_") or text


def evidence_dimension(column: str) -> str:
    """Infer a column's dimension from its identifier, not its label."""

    return infer_dimension(evidence_column_identifier(column))

# Mirrors agent/chart_pipeline._chart_type_for_visual_goal for
# visual_goal="composition", and the corrective pass that follows it. Kept as
# named constants so the equivalence test failure points at a rule rather than
# a magic number.
MAXIMUM_PIE_CATEGORIES = 8
_MAXIMUM_PIE_CATEGORIES = MAXIMUM_PIE_CATEGORIES
_CONTINUOUS_DIMENSIONS = frozenset({"price_tariff", "xrate"})


def composition_chart_type(
    dimensions: Iterable[str],
    category_count: int,
) -> str:
    """Return what Standard renders a composition snapshot as.

    The snapshot has already collapsed to a single period, so the question is
    asked as categories-without-time regardless of any date column in the
    source table — which is what makes this comparable to Standard's
    ``has_time=False, has_categories=True`` answer.

    Three rules, in Standard's order:

    1. Parts of one whole, few enough to read as slices, and measured in the
       *same* thing — a pie. Exactness is the point: a set carrying anything
       besides ``share`` is not one whole.
    2. A continuous measure with no share to anchor it is never a slice or a
       bar of a composition; Standard forces it to a line.
    3. Everything else is a bar.
    """

    dimension_set = set(dimensions)
    if (
        category_count <= _MAXIMUM_PIE_CATEGORIES
        and dimension_set == {"share"}
    ):
        return "pie"
    if (
        dimension_set & _CONTINUOUS_DIMENSIONS
        and "share" not in dimension_set
    ):
        return "line"
    return "bar"


def composition_chart_type_for_columns(
    columns: Iterable[str],
    category_count: int,
) -> str:
    """Answer for a set of evidence columns rather than inferred dimensions."""

    return composition_chart_type(
        {infer_dimension(column) for column in columns},
        category_count,
    )
