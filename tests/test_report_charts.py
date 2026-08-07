"""Deterministic report chart and shadow-evaluation tests."""

from __future__ import annotations

import logging
import os
from copy import deepcopy

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from agent.report_charts import build_report_charts
from agent.report_evaluation import evaluate_report_plan
from agent.report_planner import ReportPlanEvidenceError, validate_report_plan_evidence
from contracts.report import ReportPlan
from contracts.report_evidence import ReportEvidenceManifest
from tests.test_report_planner import _manifest, _plan_payload


def test_trend_chart_is_built_deterministically_from_manifest_rows():
    plan = ReportPlan.model_validate(_plan_payload())

    first = build_report_charts(plan, _manifest())
    second = build_report_charts(plan, _manifest())

    assert first == second
    assert len(first) == 1
    decision = first[0]
    assert decision.status == "built"
    assert decision.artifact is not None
    assert decision.artifact.type == "line"
    assert decision.artifact.data == [
        {"period": "2026-01", "price": 120.0},
        {"period": "2026-02", "price": 130.0},
    ]
    assert decision.artifact.metadata.x_axis == "period"
    assert decision.artifact.metadata.series == ["price"]
    assert decision.artifact.metadata.deterministic is True


def test_unrenderable_required_chart_is_omitted_with_a_typed_reason():
    payload = _plan_payload()
    payload["charts"][0]["purpose"] = "relationship"
    plan = ReportPlan.model_validate(payload)

    decisions = build_report_charts(plan, _manifest())

    assert decisions[0].status == "omitted"
    assert decisions[0].reason_code == "REPORT_CHART_EXPLICIT_AXES_REQUIRED"
    assert decisions[0].artifact is None


def test_relationship_chart_requires_verified_explicit_numeric_axes():
    manifest_payload = _manifest().model_dump(mode="json")
    table = manifest_payload["items"][0]
    table["columns"].append("hydro_generation")
    table["rows"][0]["hydro_generation"] = 410.0
    table["rows"][1]["hydro_generation"] = 390.0
    table["unit_by_column"]["hydro_generation"] = "GWh"
    manifest = ReportEvidenceManifest.model_validate(manifest_payload)

    payload = _plan_payload()
    payload["charts"][0].update(
        {
            "purpose": "relationship",
            "x_field": "price",
            "series_fields": ["hydro_generation"],
        }
    )
    plan = ReportPlan.model_validate(payload)
    validate_report_plan_evidence(plan, manifest)

    decision = build_report_charts(plan, manifest)[0]

    assert decision.status == "built"
    assert decision.artifact is not None
    assert decision.artifact.type == "scatter"
    assert decision.artifact.metadata.x_axis == "price"
    assert decision.artifact.metadata.series == ["hydro_generation"]

    invalid_payload = deepcopy(payload)
    invalid_payload["charts"][0]["series_fields"] = ["invented_metric"]
    with pytest.raises(ReportPlanEvidenceError, match="unknown chart fields"):
        validate_report_plan_evidence(
            ReportPlan.model_validate(invalid_payload),
            manifest,
        )


def test_shadow_evaluation_is_content_free_and_blocks_required_chart_omission():
    valid = evaluate_report_plan(
        ReportPlan.model_validate(_plan_payload()),
        _manifest(),
    )

    assert valid.contract_version == "report-evaluation-v1"
    assert valid.ready_for_generation is True
    assert valid.evidence_reference_coverage == 1.0
    assert valid.required_chart_build_rate == 1.0
    assert valid.findings == []
    assert "120.0" not in valid.model_dump_json()

    payload = deepcopy(_plan_payload())
    payload["charts"][0]["purpose"] = "relationship"
    invalid = evaluate_report_plan(ReportPlan.model_validate(payload), _manifest())

    assert invalid.ready_for_generation is False
    assert invalid.required_chart_build_rate == 0.0
    assert "REQUIRED_CHART_OMITTED" in invalid.findings


def test_shadow_evaluation_fails_closed_without_throwing_on_unknown_chart_evidence():
    payload = deepcopy(_plan_payload())
    payload["charts"][0]["evidence_refs"] = [
        "evidence:table:" + "9" * 16
    ]
    invalid = evaluate_report_plan(ReportPlan.model_validate(payload), _manifest())

    assert invalid.ready_for_generation is False
    assert invalid.evidence_reference_coverage < 1.0
    assert "PLAN_EVIDENCE_INVALID" in invalid.findings
    assert "REQUIRED_CHART_OMITTED" in invalid.findings


def test_chart_column_roles_expose_the_axis_types_the_builder_uses():
    from agent.report_charts import chart_column_roles

    table = _manifest().items[0]

    roles = chart_column_roles(table)

    assert roles["temporal"] == ["period"]
    assert roles["numeric"] == ["price"]
    assert roles["categorical"] == []


def test_demotion_clears_required_on_a_chart_that_cannot_build():
    from agent.report_charts import demote_unbuildable_required_charts

    payload = _plan_payload()
    payload["charts"][0]["purpose"] = "relationship"
    plan = ReportPlan.model_validate(payload)
    decisions = build_report_charts(plan, _manifest())
    assert [decision.status for decision in decisions] == ["omitted"]
    assert [decision.required for decision in decisions] == [True]

    demoted_plan, demoted_decisions = demote_unbuildable_required_charts(
        plan,
        decisions,
    )

    assert [chart.required for chart in demoted_plan.charts] == [False]
    assert [decision.required for decision in demoted_decisions] == [False]
    assert [decision.reason_code for decision in demoted_decisions] == [
        "REPORT_CHART_EXPLICIT_AXES_REQUIRED"
    ]


def test_demotion_leaves_a_buildable_required_chart_untouched():
    from agent.report_charts import demote_unbuildable_required_charts

    plan = ReportPlan.model_validate(_plan_payload())
    decisions = build_report_charts(plan, _manifest())
    assert [decision.status for decision in decisions] == ["built"]

    demoted_plan, demoted_decisions = demote_unbuildable_required_charts(
        plan,
        decisions,
    )

    assert demoted_plan is plan
    assert demoted_decisions is decisions


def _mixed_unit_manifest() -> ReportEvidenceManifest:
    """A table whose numeric columns share no unit — a price, an FX rate, a quantity."""
    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["columns"] = ["period", "p_bal_gel", "xrate", "quantity_hydro"]
    table["rows"] = [
        {
            "period": "2026-01",
            "p_bal_gel": 137.86,
            "xrate": 2.6693,
            "quantity_hydro": 812.4,
        },
        {
            "period": "2026-02",
            "p_bal_gel": 140.99,
            "xrate": 2.6453,
            "quantity_hydro": 903.1,
        },
    ]
    table["unit_by_column"] = {
        "p_bal_gel": "GEL/MWh",
        "xrate": "GEL/USD",
        "quantity_hydro": "MWh",
    }
    return ReportEvidenceManifest.model_validate(payload)


def _composition_plan() -> ReportPlan:
    payload = _plan_payload()
    payload["charts"][0]["purpose"] = "composition"
    payload["charts"][0].pop("series_fields", None)
    payload["charts"][0].pop("x_field", None)
    return ReportPlan.model_validate(payload)


def test_composition_never_pies_columns_that_share_no_unit():
    """A price, an exchange rate and a quantity are not slices of one whole.

    Production job 83010f04 rendered p_bal_gel, p_bal_usd, xrate,
    quantity_hydro and quantity_thermal as one pie. The builder pivoted every
    numeric column of the last row and stamped the first unit it found onto
    all of them.
    """

    decisions = build_report_charts(_composition_plan(), _mixed_unit_manifest())

    assert decisions
    artifact = decisions[0].artifact
    if artifact is not None:
        assert artifact.type != "pie"


def test_time_series_with_three_incompatible_dimensions_is_omitted():
    """Two axes cannot represent price, exchange rate, and quantity."""

    decisions = build_report_charts(_composition_plan(), _mixed_unit_manifest())

    assert decisions[0].artifact is None
    assert decisions[0].reason_code == "REPORT_CHART_INCOMPATIBLE_UNITS"


def test_mixed_dimension_line_declares_human_labels_and_dual_axes():
    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["columns"] = ["period", "share_tech", "quantity_tech"]
    table["rows"] = [
        {"period": "2026-01", "share_tech": 0.4, "quantity_tech": 800.0},
        {"period": "2026-02", "share_tech": 0.6, "quantity_tech": 1200.0},
    ]
    table["unit_by_column"] = {
        "share_tech": "ratio",
        "quantity_tech": "thousand MWh",
    }
    manifest = ReportEvidenceManifest.model_validate(payload)
    plan_payload = _plan_payload()
    plan_payload["charts"][0].update(
        {
            "purpose": "trend",
            "x_field": "period",
            "series_fields": ["share_tech", "quantity_tech"],
        }
    )

    decision = build_report_charts(
        ReportPlan.model_validate(plan_payload),
        manifest,
    )[0]

    artifact = decision.artifact
    assert artifact is not None
    assert artifact.metadata.axis_mode == "dual"
    assert artifact.metadata.axis_by_series == {
        "share_tech": "left",
        "quantity_tech": "right",
    }
    assert artifact.metadata.dimension_by_series == {
        "share_tech": "share",
        "quantity_tech": "energy_qty",
    }
    assert artifact.metadata.label_by_field == {
        "period": "Period",
        "share_tech": "Share Tech",
        "quantity_tech": "Quantity (thousand MWh)",
    }


def test_line_with_more_than_two_incompatible_axis_groups_is_omitted():
    plan_payload = _plan_payload()
    plan_payload["charts"][0].update(
        {
            "purpose": "trend",
            "x_field": "period",
            "series_fields": ["p_bal_gel", "xrate", "quantity_hydro"],
        }
    )

    decision = build_report_charts(
        ReportPlan.model_validate(plan_payload),
        _mixed_unit_manifest(),
    )[0]

    assert decision.status == "omitted"
    assert decision.reason_code == "REPORT_CHART_INCOMPATIBLE_UNITS"


def test_one_category_composition_is_omitted_as_noninformative():
    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["columns"] = ["type_tech", "share_tech"]
    table["rows"] = [{"type_tech": "hydro", "share_tech": 1.0}]
    table["unit_by_column"] = {"share_tech": "ratio"}
    table["total_row_count"] = 1
    manifest = ReportEvidenceManifest.model_validate(payload)
    plan_payload = _plan_payload()
    plan_payload["charts"][0].update(
        {
            "purpose": "composition",
            "x_field": "type_tech",
            "series_fields": ["share_tech"],
        }
    )

    decision = build_report_charts(
        ReportPlan.model_validate(plan_payload),
        manifest,
    )[0]

    assert decision.status == "omitted"
    assert decision.reason_code == "REPORT_CHART_INSUFFICIENT_CATEGORIES"


def test_chart_build_decision_is_logged_without_evidence_values(caplog):
    with caplog.at_level(logging.INFO, logger="Enai.ReportCharts"):
        build_report_charts(
            ReportPlan.model_validate(_plan_payload()),
            _manifest(),
        )

    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_CHART_DECISION ")
    )
    assert '"status":"built"' in record
    assert '"chart_id":"price_trend"' in record
    assert "120.0" not in record


def test_composition_still_pies_genuine_shares():
    """The share case is what a pie is for, and must keep working."""
    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["columns"] = ["period", "share_hydro", "share_thermal"]
    table["rows"] = [
        {"period": "2026-01", "share_hydro": 0.8, "share_thermal": 0.2},
        {"period": "2026-02", "share_hydro": 0.7, "share_thermal": 0.3},
    ]
    table["unit_by_column"] = {
        "share_hydro": "share (0-1)",
        "share_thermal": "share (0-1)",
    }
    manifest = ReportEvidenceManifest.model_validate(payload)

    decisions = build_report_charts(_composition_plan(), manifest)

    artifact = decisions[0].artifact
    assert artifact is not None
    assert artifact.type == "pie"


def _monthly_price_manifest(months: int = 14) -> ReportEvidenceManifest:
    """A year-plus of monthly prices — the raw list a report should not print."""
    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["columns"] = ["period", "p_bal_gel"]
    table["rows"] = [
        {
            "period": f"2025-{month:02d}" if month <= 12 else f"2026-{month - 12:02d}",
            "p_bal_gel": 100.0 + month,
        }
        for month in range(1, months + 1)
    ]
    table["unit_by_column"] = {"p_bal_gel": "GEL/MWh"}
    table["total_row_count"] = months
    return ReportEvidenceManifest.model_validate(payload)


def _table_plan() -> ReportPlan:
    payload = _plan_payload()
    payload["charts"][0]["purpose"] = "table"
    payload["charts"][0].pop("series_fields", None)
    payload["charts"][0].pop("x_field", None)
    return ReportPlan.model_validate(payload)


def test_report_table_summarizes_instead_of_listing_every_row():
    """A report needs analytics, not the dataset.

    Job 83010f04 printed all 138 monthly prices as an exhibit.
    """
    decisions = build_report_charts(_table_plan(), _monthly_price_manifest())

    artifact = decisions[0].artifact
    assert artifact is not None
    assert artifact.type == "table"
    # One row per segment, not one per observation.
    assert len(artifact.data) < 14
    segments = {str(row.get("segment")) for row in artifact.data}
    assert segments == {"summer", "winter", "total"}


def test_report_table_reports_mean_stdev_min_and_max():
    decisions = build_report_charts(_table_plan(), _monthly_price_manifest())

    artifact = decisions[0].artifact
    assert artifact is not None
    total = next(
        row for row in artifact.data if row.get("segment") == "total"
    )
    assert set(total) >= {
        "segment",
        "metric",
        "mean",
        "std_dev",
        "minimum",
        "maximum",
        "observations",
    }
    # 14 months of 101..114 GEL/MWh.
    assert total["minimum"] == 101.0
    assert total["maximum"] == 114.0
    assert total["observations"] == 14
    assert total["metric"] == "p_bal_gel"
    assert total["first_period"] == "2025-01"
    assert total["last_period"] == "2026-02"
    assert total["minimum_period"] == "2025-01"
    assert total["maximum_period"] == "2026-02"
    assert total["largest_increase"] == 1.0
    assert total["largest_increase_period"] == "2025-02"
    assert artifact.metadata.label_by_field["std_dev"] == "Standard deviation"
    assert artifact.metadata.label_by_value["p_bal_gel"] == (
        "Balancing electricity price (GEL/MWh)"
    )


def test_report_table_splits_summer_from_winter():
    """SUMMER_MONTHS is April-July; the split must use that one authority."""
    decisions = build_report_charts(_table_plan(), _monthly_price_manifest())

    rows = {
        str(row["segment"]): row
        for row in decisions[0].artifact.data
        if row.get("metric") == "p_bal_gel"
    }
    # 2025-04..07 -> 104..107, plus 2026-04..05 is absent at 14 months.
    assert rows["summer"]["observations"] == 4
    assert rows["summer"]["minimum"] == 104.0
    assert rows["summer"]["maximum"] == 107.0
    assert rows["winter"]["observations"] == 10


def test_report_table_keeps_a_short_table_verbatim():
    """Summarizing three rows would destroy information, not condense it."""
    decisions = build_report_charts(_table_plan(), _monthly_price_manifest(months=3))

    artifact = decisions[0].artifact
    assert artifact is not None
    assert len(artifact.data) == 3
    assert "segment" not in artifact.data[0]


def _wide_manifest(numeric_columns: int = 20) -> ReportEvidenceManifest:
    """The enriched frame shape: one period column plus many driver columns."""
    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    names = [f"driver_{index:02d}_gel" for index in range(numeric_columns)]
    table["columns"] = ["period", *names]
    table["rows"] = [
        {"period": "2026-01", **{name: 1.0 + index for index, name in enumerate(names)}},
        {"period": "2026-02", **{name: 2.0 + index for index, name in enumerate(names)}},
    ]
    table["unit_by_column"] = {name: "GEL/MWh" for name in names}
    return ReportEvidenceManifest.model_validate(payload)


def _wide_plan(purpose: str, *, explicit_series: bool) -> ReportPlan:
    payload = _plan_payload()
    chart = payload["charts"][0]
    chart["purpose"] = purpose
    chart.pop("x_field", None)
    if explicit_series:
        # The contract caps series_fields at 8, so this is the widest a planner
        # can legally request.
        chart["series_fields"] = [f"driver_{index:02d}_gel" for index in range(8)]
    else:
        chart.pop("series_fields", None)
    return ReportPlan.model_validate(payload)


@pytest.mark.parametrize("purpose", ["trend", "table"])
@pytest.mark.parametrize("explicit_series", [False, True])
def test_report_charts_never_render_an_unreadable_number_of_series(
    purpose,
    explicit_series,
):
    """A legend of twenty series is not a chart, it is a wall.

    The enriched balancing frame carries 31 driver columns, so any exhibit
    built over it can request far more series than a reader can follow.
    """
    decisions = build_report_charts(
        _wide_plan(purpose, explicit_series=explicit_series),
        _wide_manifest(),
    )

    artifact = decisions[0].artifact
    assert artifact is not None, decisions[0].omitted_reason
    assert len(artifact.metadata.series) <= 8


def test_chart_data_carries_only_the_declared_axis_and_series():
    """metadata.series caps the legend; data must not smuggle the rest.

    The enriched balancing frame has 31 driver columns. _built passed rows
    through verbatim, so a renderer keying off the row dicts drew every column
    while metadata truthfully claimed eight — the report chart that came back
    as an unreadable wall of lines.
    """
    decisions = build_report_charts(
        _wide_plan("trend", explicit_series=False),
        _wide_manifest(numeric_columns=20),
    )

    artifact = decisions[0].artifact
    assert artifact is not None
    declared = {artifact.metadata.x_axis, *artifact.metadata.series}
    for row in artifact.data:
        assert set(row) <= declared, (
            f"row carries undeclared columns: {sorted(set(row) - declared)}"
        )


def test_chart_data_projection_preserves_every_declared_value():
    """Projection must drop columns, never rows or declared values."""
    manifest = _wide_manifest(numeric_columns=20)
    decisions = build_report_charts(
        _wide_plan("trend", explicit_series=False),
        manifest,
    )

    artifact = decisions[0].artifact
    source_rows = manifest.items[0].rows
    assert len(artifact.data) == len(source_rows)
    for projected, source in zip(artifact.data, source_rows, strict=True):
        for column in projected:
            assert projected[column] == source[column]


def _temporal_manifest(period_values: list[str]) -> ReportEvidenceManifest:
    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["columns"] = ["date", "p_bal_gel"]
    table["rows"] = [
        {"date": value, "p_bal_gel": 100.0 + index}
        for index, value in enumerate(period_values)
    ]
    table["unit_by_column"] = {"p_bal_gel": "GEL/MWh"}
    table["total_row_count"] = len(period_values)
    return ReportEvidenceManifest.model_validate(payload)


def _temporal_plan() -> ReportPlan:
    payload = _plan_payload()
    chart = payload["charts"][0]
    chart["purpose"] = "trend"
    chart["x_field"] = "date"
    chart["series_fields"] = ["p_bal_gel"]
    return ReportPlan.model_validate(payload)


def _axis_values(period_values: list[str]) -> list[str]:
    decisions = build_report_charts(
        _temporal_plan(),
        _temporal_manifest(period_values),
    )
    artifact = decisions[0].artifact
    assert artifact is not None, decisions[0].reason_code
    return [row["date"] for row in artifact.data]


def test_monthly_series_renders_month_labels_not_midnight_timestamps():
    """Job 4bd4d24f axis read 2026-06-01T00:00:00 for a monthly series.

    ``date`` is detected as temporal by column name, so its ISO-timestamp
    values never met _TIME_VALUE_PATTERN and were never normalized.
    """
    assert _axis_values(
        ["2026-04-01T00:00:00", "2026-05-01T00:00:00", "2026-06-01T00:00:00"]
    ) == ["2026-04", "2026-05", "2026-06"]


def test_daily_series_keeps_its_day_component():
    """Collapsing to month would merge distinct observations."""
    assert _axis_values(
        ["2026-06-01T00:00:00", "2026-06-02T00:00:00", "2026-06-03T00:00:00"]
    ) == ["2026-06-01", "2026-06-02", "2026-06-03"]


def test_annual_series_renders_years():
    assert _axis_values(
        ["2024-01-01T00:00:00", "2025-01-01T00:00:00", "2026-01-01T00:00:00"]
    ) == ["2024", "2025", "2026"]


def test_already_normalized_periods_are_left_alone():
    assert _axis_values(["2026-04", "2026-05"]) == ["2026-04", "2026-05"]


def test_intraday_timestamps_are_not_truncated():
    """A non-midnight component is real data, not padding."""
    values = ["2026-06-01T09:30:00", "2026-06-01T10:30:00"]
    assert _axis_values(values) == values
