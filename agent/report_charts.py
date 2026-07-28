"""Deterministic report chart materialization from manifest table evidence."""

from __future__ import annotations

import re
from typing import Any

from contracts.report import ReportChartPurpose, ReportChartRequest, ReportPlan
from contracts.report_charts import (
    ReportChartArtifact,
    ReportChartBuildDecision,
    ReportChartMetadata,
    ReportChartType,
)
from contracts.report_evidence import ReportEvidenceKind, ReportEvidenceManifest
from contracts.report_research import ReportEvidencePacket

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
) -> ReportChartBuildDecision:
    artifact = ReportChartArtifact(
        chart_id=chart.chart_id,
        section_id=chart.section_id,
        type=chart_type,
        data=data,
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
            series = chart.series_fields or [
                column for column in columns if column != x_axis
            ][:8]
            if not series:
                decisions.append(_omitted(chart, "REPORT_CHART_TABLE_FIELDS_REQUIRED"))
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
                    series=requested_series[:8],
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
                decisions.append(
                    _built(
                        chart,
                        chart_type=ReportChartType.PIE,
                        data=composition_rows,
                        x_axis=x_axis,
                        series=(chart.series_fields or [numeric[0]])[:8],
                        units=units,
                    )
                )
                continue
            if temporal and len(numeric) >= 2:
                latest = rows[-1]
                composition_rows = [
                    {"category": column, "value": latest.get(column)}
                    for column in numeric[:8]
                    if _is_numeric(latest.get(column))
                ]
                if not composition_rows:
                    decisions.append(
                        _omitted(chart, "REPORT_CHART_NO_NUMERIC_EVIDENCE")
                    )
                    continue
                decisions.append(
                    _built(
                        chart,
                        chart_type=ReportChartType.PIE,
                        data=composition_rows,
                        x_axis="category",
                        series=["value"],
                        units={
                            "value": next(
                                (units[column] for column in numeric if column in units),
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
                    series=requested_series[:8],
                    units=units,
                )
            )
            continue
        if len(rows) == 1 and len(numeric) >= 2:
            comparison_rows = [
                {"category": column, "value": rows[0].get(column)}
                for column in numeric[:8]
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
