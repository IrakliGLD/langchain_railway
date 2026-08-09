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

from visualization.chart_selector import infer_dimension

# Mirrors agent/chart_pipeline._chart_type_for_visual_goal for
# visual_goal="composition", and the corrective pass that follows it. Kept as
# named constants so the equivalence test failure points at a rule rather than
# a magic number.
_MAXIMUM_PIE_CATEGORIES = 8
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
