"""Deterministic report chart materialization from manifest table evidence."""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from agent.report_chart_rules import (
    KNOWN_FIELD_LABELS,
    MAXIMUM_PIE_CATEGORIES,
    composition_chart_type,
    evidence_dimension,
    field_label,
)
from config import SUMMER_MONTHS, WINTER_MONTHS
from context import COLUMN_LABELS, DERIVED_LABELS
from contracts.report import ReportChartPurpose, ReportChartRequest, ReportPlan
from contracts.report_charts import (
    ReportChartArtifact,
    ReportChartBuildDecision,
    ReportChartMetadata,
    ReportChartType,
)
from contracts.report_evidence import ReportEvidenceKind, ReportEvidenceManifest
from contracts.report_research import ReportEvidencePacket
from visualization.chart_selector import infer_dimension, select_chart_type

_TIME_COLUMN_NAMES = {
    "date",
    "datetime",
    "month",
    "period",
    "quarter",
    "timestamp",
    "year",
}
_TIME_VALUE_PATTERN = re.compile(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?(?:Q[1-4])?$")
# Below this a table is already readable, and summarizing it would destroy
# information rather than condense it.
_TABLE_SUMMARY_ROW_THRESHOLD = 6
# Mirrors ReportChartArtifact.metadata.series (max_length=8): a builder that
# emitted more would fail contract validation rather than render.
_MAXIMUM_CHART_SERIES = 8
_LOGGER = logging.getLogger("Enai.ReportCharts")
# The label map and its inverse live together in report_chart_rules so the
# two cannot drift; a stale inverse would mislabel exactly the columns it
# exists to recover.
_field_label = field_label


def _chart_decision_log(
    decision: ReportChartBuildDecision,
    *,
    chart_type: ReportChartType | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    artifact = decision.artifact
    _LOGGER.info(
        "REPORT_CHART_DECISION %s",
        json.dumps(
            {
                "axis_mode": (
                    artifact.metadata.axis_mode
                    if artifact is not None
                    else ""
                ),
                # A reason code names the chart, not the cause. Two branches
                # raise REPORT_CHART_INSUFFICIENT_CATEGORIES and the line read
                # identically for both, so job 4ea18b2b could not say which one
                # dropped its composition.
                "detail": detail or {},
                "chart_id": decision.chart_id,
                "chart_type": (
                    chart_type.value if chart_type is not None else ""
                ),
                "reason_code": decision.reason_code,
                "series_count": (
                    len(artifact.metadata.series)
                    if artifact is not None
                    else 0
                ),
                "status": decision.status,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _axis_metadata(
    chart_type: ReportChartType,
    series: list[str],
    units: dict[str, str],
) -> tuple[str, dict[str, str], dict[str, str]] | None:
    dimensions = {name: evidence_dimension(name) for name in series}
    if chart_type not in {ReportChartType.LINE, ReportChartType.BAR}:
        return "single", {}, dimensions
    groups: list[tuple[str, str]] = []
    group_by_series: dict[str, tuple[str, str]] = {}
    for name in series:
        group = (dimensions[name], units.get(name, ""))
        if group not in groups:
            groups.append(group)
        group_by_series[name] = group
    if len(groups) <= 1:
        return "single", {}, dimensions
    if len(groups) > 2:
        return None
    axis_by_series = {
        name: ("left" if groups.index(group_by_series[name]) == 0 else "right")
        for name in series
    }
    return "dual", axis_by_series, dimensions


def _period_month(value: Any) -> int | None:
    """Extract the calendar month from a period literal like ``2026-04``."""

    match = re.match(r"^(\d{4})-(\d{2})", str(value or ""))
    if match is None:
        return None
    month = int(match.group(2))
    return month if 1 <= month <= 12 else None


def _summary_statistics_rows(
    rows: list[dict[str, Any]],
    numeric_columns: list[str],
    time_column: str | None,
) -> list[dict[str, Any]]:
    """Summarize a long series instead of printing every observation.

    A report needs analytics: job 83010f04 shipped all 138 monthly prices as an
    exhibit. Seasons split on the shared SUMMER_MONTHS authority rather than a
    local month list, so a report and an answer never disagree about which
    months are summer.
    """

    segments: list[tuple[str, list[dict[str, Any]]]] = [("total", rows)]
    if time_column is not None:
        summer = [
            row
            for row in rows
            if (_period_month(row.get(time_column)) or 0) in SUMMER_MONTHS
        ]
        winter = [
            row
            for row in rows
            if (_period_month(row.get(time_column)) or 0) in WINTER_MONTHS
        ]
        if summer and winter:
            segments = [("summer", summer), ("winter", winter), ("total", rows)]

    summary: list[dict[str, Any]] = []
    for column in numeric_columns:
        for segment, segment_rows in segments:
            observations = [
                (row.get(time_column) if time_column else None, float(row[column]))
                for row in segment_rows
                if _is_numeric(row.get(column))
            ]
            values = [value for _period, value in observations]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                if len(values) > 1
                else 0.0
            )
            summary_row = {
                    "segment": segment,
                    "metric": column,
                    "mean": round(mean, 4),
                    "std_dev": round(math.sqrt(variance), 4),
                    "minimum": round(min(values), 4),
                    "maximum": round(max(values), 4),
                    "observations": len(values),
                }
            if time_column is not None and segment == "total":
                ordered = sorted(
                    observations,
                    key=lambda item: str(item[0]),
                )
                minimum = min(ordered, key=lambda item: item[1])
                maximum = max(ordered, key=lambda item: item[1])
                summary_row.update(
                    {
                        "first": round(ordered[0][1], 4),
                        "first_period": ordered[0][0],
                        "last": round(ordered[-1][1], 4),
                        "last_period": ordered[-1][0],
                        "minimum_period": minimum[0],
                        "maximum_period": maximum[0],
                    }
                )
                changes = [
                    (
                        current[0],
                        current[1] - previous[1],
                    )
                    for previous, current in zip(
                        ordered,
                        ordered[1:],
                        strict=False,
                    )
                ]
                increases = [change for change in changes if change[1] > 0]
                decreases = [change for change in changes if change[1] < 0]
                if increases:
                    period, change = max(increases, key=lambda item: item[1])
                    summary_row.update(
                        {
                            "largest_increase": round(change, 4),
                            "largest_increase_period": period,
                        }
                    )
                if decreases:
                    period, change = min(decreases, key=lambda item: item[1])
                    summary_row.update(
                        {
                            "largest_decrease": round(change, 4),
                            "largest_decrease_period": period,
                        }
                    )
            summary.append(summary_row)
    return summary


_ISO_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"(?:[T ](?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?)?$"
)


def _normalized_axis_values(values: list[Any]) -> list[Any] | None:
    """Render a temporal axis at the granularity the series actually carries.

    ``date`` is temporal by column name, so its ISO-timestamp values never met
    _TIME_VALUE_PATTERN and reached the axis whole: job 4bd4d24f labelled a
    monthly series ``2026-06-01T00:00:00``.

    Only components that are constant *and* at their default across every row
    are dropped, so this never merges distinct observations — a daily series
    keeps its day, and a non-midnight time is real data rather than padding.
    Returns None when the column is not a uniform timestamp, leaving it as-is.
    """

    matches = [_ISO_TIMESTAMP_PATTERN.match(str(value)) for value in values]
    if not matches or any(match is None for match in matches):
        return None
    if any(
        part not in (None, "00")
        for match in matches
        for part in (match["hour"], match["minute"], match["second"])
    ):
        return None
    if all(match["day"] == "01" for match in matches):
        if all(match["month"] == "01" for match in matches):
            return [match["year"] for match in matches]
        return [f"{match['year']}-{match['month']}" for match in matches]
    return [
        f"{match['year']}-{match['month']}-{match['day']}" for match in matches
    ]


def _composition_snapshot_type(columns: list[str], category_count: int) -> str:
    """Ask Standard's selector what a composition snapshot should render as.

    Reports used to reach PIE for any composition request, so job 83010f04
    pied a GEL price, an FX rate and two quantities as slices of one whole and
    stamped the first unit it found onto all of them. Standard already answers
    this correctly — a pie needs a ``share`` dimension — so it owns the rule
    and this calls it rather than keeping a second copy that can drift.

    The snapshot has already collapsed to one period, so it is asked as
    categories-without-time regardless of the source table's date column.

    ``select_chart_type`` is Standard's *goal-less* fallback, reached only when
    the analyzer emits no ``visual_goal`` at all, and its pie branch tests
    ``"share" in dimensions``. The report has a goal —
    ``ReportChartPurpose.COMPOSITION`` — and is entitled to the rule Standard
    applies when it has one, which tests ``dimensions == {"share"}``. That
    exactness is what keeps prices out of a pie of shares. The previous answer
    is still computed, and reported whenever it differs, so the effect of the
    change stays visible in production.
    """

    dimensions = {evidence_dimension(column) for column in columns}
    previous = select_chart_type(
        has_time=False,
        has_categories=True,
        dimensions=dimensions,
        category_count=category_count,
    )
    applied = composition_chart_type(dimensions, category_count)
    if previous != applied:
        _LOGGER.info(
            "REPORT_CHART_TYPE_DISAGREEMENT %s",
            json.dumps(
                {
                    "applied": applied,
                    "category_count": category_count,
                    "columns": sorted(columns)[:8],
                    "dimensions": sorted(dimensions),
                    "previous": previous,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    return applied


def _latest_period_rows(
    rows: list[dict[str, Any]],
    temporal: list[str],
) -> list[dict[str, Any]]:
    """Collapse a frame to the most recent period it carries."""

    if not temporal:
        return rows
    time_column = temporal[0]
    periods = [
        str(row.get(time_column))
        for row in rows
        if row.get(time_column) is not None
    ]
    if not periods:
        return rows
    latest = max(periods)
    return [row for row in rows if str(row.get(time_column)) == latest]


def _ranked_by_contribution(
    columns: list[str],
    latest: dict[str, Any],
) -> list[str]:
    """Order components by their magnitude in the period being shown.

    Table order is declaration accident — a dominant component could be lost
    because of where it sat in the SELECT. This is the contribution ordering
    Standard's own composition budget uses.
    """

    return sorted(
        columns,
        key=lambda column: (
            -abs(float(latest.get(column) or 0.0)),
            columns.index(column),
        ),
    )


def _largest_contributors(
    chart,
    columns: list[str],
    latest: dict[str, Any],
) -> list[str]:
    """Trim a series list to the display budget, largest contributors first.

    ``metadata.series`` holds at most eight, so something has to go once a
    frame carries more. A line chart claims no total, so dropping the smallest
    is honest — but say what went, or a thin chart is a mystery.
    """

    if len(columns) <= _MAXIMUM_CHART_SERIES:
        return columns
    # Rank *within* a dimension and then take from each in turn. Contribution
    # is only meaningful between comparable measures: ranking a quantity in
    # thousands against a share in 0..1 makes every quantity outrank every
    # share, and on job b3153071 that dropped five shares and kept every
    # quantity — one of which survived only because it happened to be zero.
    by_dimension: dict[str, list[str]] = {}
    for column in columns:
        by_dimension.setdefault(evidence_dimension(column), []).append(column)
    ranked = {
        dimension: _ranked_by_contribution(members, latest)
        for dimension, members in by_dimension.items()
    }
    kept: set[str] = set()
    for depth in range(max(len(members) for members in ranked.values())):
        for members in ranked.values():
            if len(kept) >= _MAXIMUM_CHART_SERIES:
                break
            if depth < len(members):
                kept.add(members[depth])
        if len(kept) >= _MAXIMUM_CHART_SERIES:
            break
    dropped = [column for column in columns if column not in kept]
    _LOGGER.info(
        "REPORT_CHART_SERIES_DROPPED %s",
        json.dumps(
            {
                "chart_id": chart.chart_id,
                "dropped": dropped[:12],
                "dropped_count": len(dropped),
                "kept_count": len(kept),
                "reason": "series_budget",
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    # Keep the frame's own column order among the survivors: the ranking picks
    # who stays, not how the chart reads.
    return [column for column in columns if column in kept]


def _composition_slices(
    chart,
    columns: list[str],
    latest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return slices that always account for the whole composition.

    Too many components to read is a display problem, not a reason to answer a
    different question: a composition was asked for, so a composition is what
    should come back. The largest stay as their own slices and the tail is
    summed into ``Other``, which is what keeps the pie honest — eleven equal
    components used to render as eight slices totalling 0.727.

    An ``Other`` slice grounds nothing, and needs to ground nothing: chart data
    never reaches the writer's prompt and nothing downstream reads it, so the
    prose still cites manifest cells exclusively.
    """

    slices = [
        {"category": column, "value": latest.get(column)}
        for column in columns
    ]
    if len(slices) <= MAXIMUM_PIE_CATEGORIES:
        return slices
    kept = set(
        _ranked_by_contribution(columns, latest)[: MAXIMUM_PIE_CATEGORIES - 1]
    )
    rolled_up = [column for column in columns if column not in kept]
    _LOGGER.info(
        "REPORT_CHART_SLICES_ROLLED_UP %s",
        json.dumps(
            {
                "chart_id": chart.chart_id,
                "rolled_up": rolled_up[:12],
                "rolled_up_count": len(rolled_up),
                "slice_count": MAXIMUM_PIE_CATEGORIES,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return [
        *(row for row in slices if row["category"] in kept),
        {
            "category": "Other",
            "value": sum(
                float(latest.get(column) or 0.0) for column in rolled_up
            ),
        },
    ]


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _table_rows_for_chart(chart, manifest: ReportEvidenceManifest):
    item_by_ref = manifest.item_by_ref()
    if any(ref not in item_by_ref for ref in chart.evidence_refs):
        return None
    items = [item_by_ref[ref] for ref in chart.evidence_refs]
    if not items or any(item.kind is not ReportEvidenceKind.TABLE for item in items):
        return None
    columns = items[0].columns
    if any(item.columns != columns for item in items[1:]):
        return None
    rows = [row for item in items for row in item.rows]
    units = {
        column: unit
        for item in items
        for column, unit in item.unit_by_column.items()
    }
    return columns, rows[:200], units


def _infer_columns(columns: list[str], rows: list[dict[str, Any]]):
    temporal = [
        column
        for column in columns
        if (
            column.lower() in _TIME_COLUMN_NAMES
            or any(
                isinstance(row.get(column), str)
                and bool(_TIME_VALUE_PATTERN.match(row[column]))
                for row in rows
            )
        )
    ]
    numeric = [
        column
        for column in columns
        if column not in temporal
        and any(_is_numeric(row.get(column)) for row in rows)
    ]
    categorical = [
        column
        for column in columns
        if column not in numeric and column not in temporal
    ]
    return numeric, temporal, categorical


def chart_column_roles(item) -> dict[str, list[str]]:
    """Expose the builder's own axis typing so planning can respect it."""

    numeric, temporal, categorical = _infer_columns(
        list(item.columns),
        list(item.rows),
    )
    return {
        "numeric": numeric,
        "temporal": temporal,
        "categorical": categorical,
    }


def demote_unbuildable_required_charts(
    plan: ReportPlan,
    chart_decisions: list[ReportChartBuildDecision],
) -> tuple[ReportPlan, list[ReportChartBuildDecision]]:
    """Keep an unbuildable chart request visible without failing the report.

    A chart marked required may be exactly what the request asked for, so it is
    neither pruned nor allowed to kill the job: the request stays in the plan,
    the omission stays in the decisions, and assembly discloses it. `required`
    never affects buildability, so this needs no second build pass.
    """

    unbuildable = {
        decision.chart_id
        for decision in chart_decisions
        if decision.required and decision.status != "built"
    }
    if not unbuildable:
        return plan, chart_decisions

    payload = plan.model_dump(mode="json")
    for chart in payload["charts"]:
        if chart["chart_id"] in unbuildable:
            chart["required"] = False
    demoted_decisions = [
        decision.model_copy(update={"required": False})
        if decision.chart_id in unbuildable
        else decision
        for decision in chart_decisions
    ]
    return ReportPlan.model_validate(payload), demoted_decisions


def _omitted(
    chart,
    code: str,
    detail: dict[str, Any] | None = None,
) -> ReportChartBuildDecision:
    decision = ReportChartBuildDecision(
        chart_id=chart.chart_id,
        required=chart.required,
        status="omitted",
        reason_code=code,
        artifact=None,
    )
    _chart_decision_log(decision, detail=detail)
    return decision


def _axis_is_unique(rows: list[dict[str, Any]], column: str) -> bool:
    """Return whether one value of ``column`` labels exactly one row."""

    labels = [str(row.get(column)) for row in rows]
    return bool(labels) and len(labels) == len(set(labels))


def _comparison_projection(
    rows: list[dict[str, Any]],
    *,
    x_axis: str,
    series: list[str],
    categorical: list[str],
    temporal: list[str],
    units: dict[str, str],
) -> tuple[list[dict[str, Any]] | None, list[str], dict[str, str]]:
    """Return ``(rows, [x_axis, *series], units)`` a bar chart can label.

    Passes a already-unique axis through untouched. A repeated axis means the
    frame is keyed by more than one dimension, so it is pivoted to one row per
    category with a series per period. Returns ``None`` rows when neither holds
    — an unlabelable chart is worse than a declared missing one.
    """

    if _axis_is_unique(rows, x_axis):
        return rows, [x_axis, *series], units
    category = next(
        (column for column in categorical if column != x_axis),
        x_axis if x_axis in categorical else None,
    )
    period = next((column for column in temporal if column != category), None)
    if category is None or period is None or len(series) != 1:
        return None, [], {}
    measure = series[0]
    periods = list(
        dict.fromkeys(
            str(row.get(period))
            for row in rows
            if row.get(period) is not None
        )
    )
    if not 2 <= len(periods) <= _MAXIMUM_CHART_SERIES:
        return None, [], {}
    pivoted: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get(category))
        value = row.get(measure)
        if not _is_numeric(value):
            continue
        pivoted.setdefault(label, {category: label})[
            str(row.get(period))
        ] = value
    if len(pivoted) < 2:
        return None, [], {}
    measure_unit = units.get(measure, "")
    return (
        list(pivoted.values()),
        [category, *periods],
        {name: measure_unit for name in periods},
    )


def _built(
    chart,
    *,
    chart_type: ReportChartType,
    data: list[dict[str, Any]],
    x_axis: str,
    series: list[str],
    units: dict[str, str],
    context_columns: tuple[str, ...] = (),
) -> ReportChartBuildDecision:
    axis_metadata = _axis_metadata(chart_type, series, units)
    if axis_metadata is None:
        # More than two (dimension, unit) groups. Naming them separates a
        # genuine three-unit frame from a spurious third group produced by a
        # mis-inferred dimension, which read identically before.
        return _omitted(
            chart,
            "REPORT_CHART_INCOMPATIBLE_UNITS",
            {
                "axis_groups": sorted(
                    {
                        (evidence_dimension(name), units.get(name, ""))
                        for name in series
                    },
                    key=lambda group: [
                        evidence_dimension(name) for name in series
                    ].index(group[0]),
                ),
                "series": series[:8],
            },
        )
    axis_mode, axis_by_series, dimension_by_series = axis_metadata
    # Project rows onto the declared axis and series. metadata.series caps the
    # legend, but passing rows through verbatim let the payload carry every
    # column of the source table — 31 of them on the enriched balancing frame —
    # so a renderer keying off the row dicts drew all of them while metadata
    # truthfully claimed eight. Projecting here makes the wider chart
    # unrepresentable rather than merely undeclared, and shrinks the payload.
    # ``context_columns`` keeps a non-series column a reader still needs — the
    # period a composition snapshot was filtered to, which is otherwise
    # unrecoverable from the rows. One temporal column cannot recreate the
    # defect above, which was thirty numeric ones.
    projected_columns = [x_axis, *series, *context_columns]
    projected_data = [
        {
            column: row[column]
            for column in projected_columns
            if column in row
        }
        for row in data
    ]
    if projected_data and all(x_axis in row for row in projected_data):
        normalized = _normalized_axis_values(
            [row[x_axis] for row in projected_data]
        )
        if normalized is not None:
            projected_data = [
                {**row, x_axis: value}
                for row, value in zip(projected_data, normalized, strict=True)
            ]
    label_by_field = {
        field: _field_label(field)
        for field in dict.fromkeys(projected_columns)
    }
    label_by_value = {
        value: _field_label(value)
        for row in projected_data
        for value in row.values()
        if isinstance(value, str) and value in KNOWN_FIELD_LABELS
    }
    artifact = ReportChartArtifact(
        chart_id=chart.chart_id,
        section_id=chart.section_id,
        type=chart_type,
        data=projected_data,
        metadata=ReportChartMetadata(
            title=chart.title,
            deterministic=True,
            evidence_refs=chart.evidence_refs,
            x_axis=x_axis,
            series=series,
            unit_by_series={
                series_name: units[series_name]
                for series_name in series
                if series_name in units
            },
            label_by_field=label_by_field,
            label_by_value=label_by_value,
            dimension_by_series=dimension_by_series,
            axis_mode=axis_mode,
            axis_by_series=axis_by_series,
        ),
    )
    decision = ReportChartBuildDecision(
        chart_id=chart.chart_id,
        required=chart.required,
        status="built",
        reason_code="",
        artifact=artifact,
    )
    # A built chart used to log only a count. "series_count": 8 cannot say
    # whether the eight are prices, shares, or a mix that should never have
    # shared an axis, so no production log could tell what was depicted.
    _chart_decision_log(
        decision,
        chart_type=chart_type,
        detail={
            "dimensions": dimension_by_series,
            "row_count": len(projected_data),
            "series": series[:8],
            "units": artifact.metadata.unit_by_series,
            "x_axis": x_axis,
        },
    )
    return decision


def build_report_chart_requests(
    chart_requests: list[ReportChartRequest],
    manifest: ReportEvidenceManifest,
) -> list[ReportChartBuildDecision]:
    decisions: list[ReportChartBuildDecision] = []
    for chart in chart_requests:
        table = _table_rows_for_chart(chart, manifest)
        if table is None:
            decisions.append(_omitted(chart, "REPORT_CHART_INCOMPATIBLE_EVIDENCE"))
            continue
        columns, rows, units = table
        requested_fields = set(chart.series_fields)
        if chart.x_field is not None:
            requested_fields.add(chart.x_field)
        if not requested_fields.issubset(set(columns)):
            decisions.append(_omitted(chart, "REPORT_CHART_UNKNOWN_FIELD"))
            continue
        numeric, temporal, categorical = _infer_columns(columns, rows)
        if not rows:
            decisions.append(_omitted(chart, "REPORT_CHART_NO_EVIDENCE_ROWS"))
            continue

        if chart.purpose is ReportChartPurpose.TABLE:
            x_axis = chart.x_field or (temporal or categorical or columns)[0]
            # The cap has to wrap the whole choice. Written as
            # ``series_fields or [...][:_MAXIMUM_CHART_SERIES]`` it binds to the fallback alone, so
            # a supplied list skipped the cap entirely. Unreachable today only
            # because the contract caps series_fields at 8 as well; relying on
            # that leaves the guard here silently doing nothing.
            series = (
                chart.series_fields
                or [column for column in columns if column != x_axis]
            )[:_MAXIMUM_CHART_SERIES]
            if not series:
                decisions.append(_omitted(chart, "REPORT_CHART_TABLE_FIELDS_REQUIRED"))
                continue
            summary = (
                _summary_statistics_rows(
                    rows,
                    [column for column in series if column in numeric],
                    temporal[0] if temporal else None,
                )
                if len(rows) > _TABLE_SUMMARY_ROW_THRESHOLD
                else []
            )
            if summary:
                decisions.append(
                    _built(
                        chart,
                        chart_type=ReportChartType.TABLE,
                        data=summary,
                        x_axis="segment",
                        series=[
                            "metric",
                            "mean",
                            "std_dev",
                            "minimum",
                            "maximum",
                            "observations",
                        ],
                        units=units,
                        context_columns=(
                            "first",
                            "first_period",
                            "last",
                            "last_period",
                            "minimum_period",
                            "maximum_period",
                            "largest_increase",
                            "largest_increase_period",
                            "largest_decrease",
                            "largest_decrease_period",
                        ),
                    )
                )
                continue
            decisions.append(
                _built(
                    chart,
                    chart_type=ReportChartType.TABLE,
                    data=rows,
                    x_axis=x_axis,
                    series=series,
                    units=units,
                )
            )
            continue

        if not numeric:
            decisions.append(_omitted(chart, "REPORT_CHART_NO_NUMERIC_EVIDENCE"))
            continue

        requested_series = chart.series_fields or numeric
        if any(series not in numeric for series in requested_series):
            decisions.append(_omitted(chart, "REPORT_CHART_NUMERIC_SERIES_REQUIRED"))
            continue

        if chart.purpose in {
            ReportChartPurpose.TREND,
            ReportChartPurpose.FORECAST,
        }:
            x_axis = chart.x_field or (temporal[0] if temporal else None)
            if x_axis is None or x_axis not in temporal:
                decisions.append(_omitted(chart, "REPORT_CHART_TIME_AXIS_REQUIRED"))
                continue
            decisions.append(
                _built(
                    chart,
                    chart_type=ReportChartType.LINE,
                    data=rows,
                    x_axis=x_axis,
                    series=requested_series[:_MAXIMUM_CHART_SERIES],
                    units=units,
                )
            )
            continue

        if chart.purpose is ReportChartPurpose.RELATIONSHIP:
            if (
                chart.x_field is None
                or not chart.series_fields
                or chart.x_field not in numeric
            ):
                decisions.append(
                    _omitted(chart, "REPORT_CHART_EXPLICIT_AXES_REQUIRED")
                )
                continue
            decisions.append(
                _built(
                    chart,
                    chart_type=ReportChartType.SCATTER,
                    data=rows,
                    x_axis=chart.x_field,
                    series=chart.series_fields,
                    units=units,
                )
            )
            continue

        if chart.purpose is ReportChartPurpose.COMPOSITION:
            composition_rows = _latest_period_rows(rows, temporal)
            # A single category is not a composition — but the frame may still
            # hold one. get_balancing_composition returns eight share *columns*
            # beside a `segment` column carrying the same value on every row,
            # so the category axis collapses to one slice while the components
            # sit unused in ``numeric``. That omitted the balancing composition
            # in every run from 26f3bbf6 onward. Where the components can be
            # pivoted, prefer them over nothing; the category axis still wins
            # whenever it actually has categories.
            pivot_available = bool(temporal) and len(numeric) >= 2
            use_category_axis = bool(categorical) and not (
                pivot_available and len(composition_rows) < 2
            )
            if use_category_axis:
                x_axis = chart.x_field or categorical[0]
                if x_axis not in categorical:
                    decisions.append(
                        _omitted(chart, "REPORT_CHART_CATEGORY_REQUIRED")
                    )
                    continue
                snapshot_series = (chart.series_fields or [numeric[0]])[:_MAXIMUM_CHART_SERIES]
                snapshot_type = _composition_snapshot_type(
                    snapshot_series,
                    len(composition_rows),
                )
                if snapshot_type == "pie" and len(composition_rows) < 2:
                    decisions.append(
                        _omitted(
                            chart,
                            "REPORT_CHART_INSUFFICIENT_CATEGORIES",
                            {
                                "branch": "category_axis",
                                "categorical_columns": categorical[:8],
                                "category_count": len(composition_rows),
                                "numeric_columns": numeric[:8],
                                "row_count": len(rows),
                                "series": snapshot_series,
                                "temporal_columns": temporal[:4],
                                "x_axis": x_axis,
                            },
                        )
                    )
                    continue
                decisions.append(
                    _built(
                        chart,
                        chart_type=(
                            ReportChartType.PIE
                            if snapshot_type == "pie"
                            else ReportChartType.BAR
                        ),
                        data=composition_rows,
                        x_axis=x_axis,
                        series=snapshot_series,
                        units=units,
                        context_columns=(
                            (temporal[0],) if temporal else ()
                        ),
                    )
                )
                continue
            if temporal and len(numeric) >= 2:
                latest = rows[-1]
                # Every component, not the first eight. The slice budget used
                # to be applied *before* the type was chosen, so
                # ``len(pivot_columns)`` could never exceed the pie ceiling and
                # the "few enough to read as slices" gate could never fail: an
                # eleven-part composition rendered as eight slices summing to
                # 0.727 and presented as the whole.
                pivot_columns = [
                    column
                    for column in numeric
                    if _is_numeric(latest.get(column))
                ]
                if not pivot_columns:
                    decisions.append(
                        _omitted(chart, "REPORT_CHART_NO_NUMERIC_EVIDENCE")
                    )
                    continue
                snapshot_type = _composition_snapshot_type(
                    pivot_columns,
                    len(pivot_columns),
                )
                if snapshot_type != "pie" and _composition_snapshot_type(
                    pivot_columns,
                    MAXIMUM_PIE_CATEGORIES,
                ) == "pie":
                    # Parts of one whole, only too many of them to read. That
                    # is a display problem, and the tail rolls into "Other"
                    # rather than the reader getting a trend they did not ask
                    # for. Asking the same rule with a count that fits is what
                    # separates "too many" from "not a composition at all".
                    snapshot_type = "pie"
                if snapshot_type != "pie":
                    # Not parts of one whole. Chart the series over time
                    # instead, which is what Standard renders for this shape.
                    # A line is a readability budget rather than a claim about
                    # totals, so trimming is honest here — but it must drop the
                    # least of the composition, not whatever the table happened
                    # to declare last.
                    decisions.append(
                        _built(
                            chart,
                            chart_type=ReportChartType.LINE,
                            data=rows,
                            x_axis=temporal[0],
                            series=_largest_contributors(
                                chart,
                                pivot_columns,
                                latest,
                            ),
                            units=units,
                        )
                    )
                    continue
                if len(pivot_columns) < 2:
                    decisions.append(
                        _omitted(
                            chart,
                            "REPORT_CHART_INSUFFICIENT_CATEGORIES",
                            {
                                "branch": "temporal_pivot",
                                "categorical_columns": categorical[:8],
                                "category_count": len(pivot_columns),
                                "numeric_columns": numeric[:8],
                                "pivot_columns": pivot_columns,
                                "row_count": len(rows),
                                "temporal_columns": temporal[:4],
                            },
                        )
                    )
                    continue
                decisions.append(
                    _built(
                        chart,
                        chart_type=ReportChartType.PIE,
                        data=_composition_slices(
                            chart,
                            pivot_columns,
                            latest,
                        ),
                        x_axis="category",
                        series=["value"],
                        units={
                            "value": next(
                                (units[column] for column in pivot_columns if column in units),
                                "",
                            )
                        },
                    )
                )
                continue
            decisions.append(_omitted(chart, "REPORT_CHART_CATEGORY_REQUIRED"))
            continue

        x_axis = chart.x_field or ((categorical or temporal)[0] if (categorical or temporal) else None)
        if x_axis:
            # A bar chart says "this value belongs to this label". When the
            # axis column repeats, several bars carry the same label and the
            # column that actually separates them appears nowhere: job
            # 5e6b0cf3 drew eight bars under four repeated month labels for a
            # frame keyed by (period, technology). Pivot that shape into one
            # row per category with a series per period — the comparison the
            # reader asked for, expressible in the existing contract.
            plot_rows, plot_series, plot_units = _comparison_projection(
                rows,
                x_axis=x_axis,
                series=requested_series[:_MAXIMUM_CHART_SERIES],
                categorical=categorical,
                temporal=temporal,
                units=units,
            )
            if plot_rows is None:
                decisions.append(
                    _omitted(chart, "REPORT_CHART_AMBIGUOUS_CATEGORY_AXIS")
                )
                continue
            decisions.append(
                _built(
                    chart,
                    chart_type=ReportChartType.BAR,
                    data=plot_rows,
                    x_axis=plot_series[0],
                    series=plot_series[1:],
                    units=plot_units,
                )
            )
            continue
        if len(rows) == 1 and len(numeric) >= 2:
            comparison_rows = [
                {"category": column, "value": rows[0].get(column)}
                for column in numeric[:_MAXIMUM_CHART_SERIES]
            ]
            decisions.append(
                _built(
                    chart,
                    chart_type=ReportChartType.BAR,
                    data=comparison_rows,
                    x_axis="category",
                    series=["value"],
                    units={},
                )
            )
            continue
        decisions.append(_omitted(chart, "REPORT_CHART_CATEGORY_REQUIRED"))
    return decisions


def build_report_charts(
    plan: ReportPlan,
    manifest: ReportEvidenceManifest,
) -> list[ReportChartBuildDecision]:
    return build_report_chart_requests(list(plan.charts), manifest)


def build_report_research_exhibits(
    packets: list[ReportEvidencePacket],
    manifest: ReportEvidenceManifest,
) -> list[ReportChartBuildDecision]:
    """Materialize packet candidates directly from their verified tables."""

    requests: list[ReportChartRequest] = []
    seen_chart_ids: set[str] = set()
    for packet in packets:
        for candidate in packet.chart_candidates:
            if candidate.chart_id in seen_chart_ids:
                raise ValueError(
                    "Research exhibit chart IDs must be globally unique."
                )
            seen_chart_ids.add(candidate.chart_id)
            requests.append(
                ReportChartRequest(
                    chart_id=candidate.chart_id,
                    section_id=packet.track_id,
                    purpose=candidate.purpose,
                    title=candidate.title,
                    evidence_refs=candidate.evidence_refs,
                    x_field=candidate.x_field,
                    series_fields=candidate.series_fields,
                    required=candidate.required,
                )
            )
    return build_report_chart_requests(requests, manifest)
