"""Deterministic report chart materialization from manifest table evidence."""

from __future__ import annotations

import math
import re
from typing import Any

from config import SUMMER_MONTHS, WINTER_MONTHS
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
            values = [
                float(row[column])
                for row in segment_rows
                if _is_numeric(row.get(column))
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                if len(values) > 1
                else 0.0
            )
            summary.append(
                {
                    "segment": segment,
                    "metric": column,
                    "mean": round(mean, 4),
                    "std_dev": round(math.sqrt(variance), 4),
                    "minimum": round(min(values), 4),
                    "maximum": round(max(values), 4),
                    "observations": len(values),
                }
            )
    return summary


def _composition_snapshot_type(columns: list[str], category_count: int) -> str:
    """Ask Standard's selector what a composition snapshot should render as.

    Reports used to reach PIE for any composition request, so job 83010f04
    pied a GEL price, an FX rate and two quantities as slices of one whole and
    stamped the first unit it found onto all of them. Standard already answers
    this correctly — a pie needs a ``share`` dimension — so it owns the rule
    and this calls it rather than keeping a second copy that can drift.

    The snapshot has already collapsed to one period, so it is asked as
    categories-without-time regardless of the source table's date column.
    """

    return select_chart_type(
        has_time=False,
        has_categories=True,
        dimensions={infer_dimension(column) for column in columns},
        category_count=category_count,
    )


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


def _omitted(chart, code: str) -> ReportChartBuildDecision:
    return ReportChartBuildDecision(
        chart_id=chart.chart_id,
        required=chart.required,
        status="omitted",
        reason_code=code,
        artifact=None,
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
        ),
    )
    return ReportChartBuildDecision(
        chart_id=chart.chart_id,
        required=chart.required,
        status="built",
        reason_code="",
        artifact=artifact,
    )


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
            if categorical:
                x_axis = chart.x_field or categorical[0]
                if x_axis not in categorical:
                    decisions.append(
                        _omitted(chart, "REPORT_CHART_CATEGORY_REQUIRED")
                    )
                    continue
                composition_rows = rows
                if temporal:
                    time_column = temporal[0]
                    latest_period = max(
                        str(row.get(time_column))
                        for row in rows
                        if row.get(time_column) is not None
                    )
                    composition_rows = [
                        row
                        for row in rows
                        if str(row.get(time_column)) == latest_period
                    ]
                snapshot_series = (chart.series_fields or [numeric[0]])[:_MAXIMUM_CHART_SERIES]
                decisions.append(
                    _built(
                        chart,
                        chart_type=(
                            ReportChartType.PIE
                            if _composition_snapshot_type(
                                snapshot_series,
                                len(composition_rows),
                            ) == "pie"
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
                pivot_columns = [
                    column
                    for column in numeric[:_MAXIMUM_CHART_SERIES]
                    if _is_numeric(latest.get(column))
                ]
                if not pivot_columns:
                    decisions.append(
                        _omitted(chart, "REPORT_CHART_NO_NUMERIC_EVIDENCE")
                    )
                    continue
                if _composition_snapshot_type(
                    pivot_columns,
                    len(pivot_columns),
                ) != "pie":
                    # Not parts of one whole. Chart the series over time
                    # instead, which is what Standard renders for this shape.
                    decisions.append(
                        _built(
                            chart,
                            chart_type=ReportChartType.LINE,
                            data=rows,
                            x_axis=temporal[0],
                            series=pivot_columns,
                            units=units,
                        )
                    )
                    continue
                decisions.append(
                    _built(
                        chart,
                        chart_type=ReportChartType.PIE,
                        data=[
                            {"category": column, "value": latest.get(column)}
                            for column in pivot_columns
                        ],
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
            decisions.append(
                _built(
                    chart,
                    chart_type=ReportChartType.BAR,
                    data=rows,
                    x_axis=x_axis,
                    series=requested_series[:_MAXIMUM_CHART_SERIES],
                    units=units,
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
