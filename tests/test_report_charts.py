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
from contracts.report import ReportPlan
from contracts.report_charts import ReportChartType
from contracts.report_evidence import ReportEvidenceManifest
from tests.fixtures_report_manifest import _manifest, _plan_payload


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

    decision = build_report_charts(plan, manifest)[0]

    assert decision.status == "built"
    assert decision.artifact is not None
    assert decision.artifact.type == "scatter"
    assert decision.artifact.metadata.x_axis == "price"
    assert decision.artifact.metadata.series == ["hydro_generation"]


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


def _long_frame_manifest() -> ReportEvidenceManifest:
    """A frame keyed by two dimensions: one row per (period, technology)."""
    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["columns"] = ["period", "type_tech", "share_tech"]
    table["rows"] = [
        {"period": period, "type_tech": technology, "share_tech": share}
        for period, technology, share in (
            ("2026-04", "hydro", 0.8143),
            ("2026-04", "thermal", 0.1719),
            ("2026-05", "hydro", 0.9898),
            ("2026-05", "thermal", 0.0),
        )
    ]
    table["unit_by_column"] = {"share_tech": "ratio"}
    table["total_row_count"] = 4
    return ReportEvidenceManifest.model_validate(payload)


def _comparison_plan() -> ReportPlan:
    payload = _plan_payload()
    payload["charts"][0]["purpose"] = "comparison"
    payload["charts"][0]["x_field"] = "period"
    payload["charts"][0]["series_fields"] = ["share_tech"]
    return ReportPlan.model_validate(payload)


def test_a_two_dimensional_frame_is_pivoted_not_drawn_under_repeated_labels():
    """Job 5e6b0cf3 drew eight bars under four repeated month labels.

    The frame was keyed by (period, technology) and the axis took period, so
    several bars shared a label and the technology separating them appeared
    nowhere on the chart. The comparison the reader asked for is one bar group
    per technology with a series per month.
    """

    decisions = build_report_charts(_comparison_plan(), _long_frame_manifest())

    artifact = decisions[0].artifact
    assert artifact is not None, decisions[0].reason_code
    labels = [row["type_tech"] for row in artifact.data]
    assert labels == sorted(set(labels)), labels
    assert {"2026-04", "2026-05"}.issubset(set(artifact.metadata.series))
    hydro = next(row for row in artifact.data if row["type_tech"] == "hydro")
    assert hydro["2026-04"] == 0.8143
    assert hydro["2026-05"] == 0.9898


def test_a_unique_axis_is_still_charted_verbatim():
    """The pivot must not disturb the one-row-per-label case."""

    payload = _plan_payload()
    payload["charts"][0]["purpose"] = "comparison"
    payload["charts"][0]["x_field"] = "period"
    payload["charts"][0]["series_fields"] = ["p_bal_gel"]
    plan = ReportPlan.model_validate(payload)

    decisions = build_report_charts(plan, _mixed_unit_manifest())

    artifact = decisions[0].artifact
    assert artifact is not None, decisions[0].reason_code
    assert [row["period"] for row in artifact.data] == ["2026-01", "2026-02"]


def test_an_unlabelable_comparison_is_omitted_rather_than_drawn():
    """No category to pivot onto means no honest bar chart."""

    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["columns"] = ["period", "p_bal_gel"]
    table["rows"] = [
        {"period": "2026-04", "p_bal_gel": 155.6},
        {"period": "2026-04", "p_bal_gel": 137.8},
    ]
    table["unit_by_column"] = {"p_bal_gel": "GEL/MWh"}
    table["total_row_count"] = 2
    manifest = ReportEvidenceManifest.model_validate(payload)
    payload_plan = _plan_payload()
    payload_plan["charts"][0]["purpose"] = "comparison"
    payload_plan["charts"][0]["x_field"] = "period"
    payload_plan["charts"][0]["series_fields"] = ["p_bal_gel"]

    decisions = build_report_charts(
        ReportPlan.model_validate(payload_plan),
        manifest,
    )

    assert decisions[0].artifact is None
    assert decisions[0].reason_code == "REPORT_CHART_AMBIGUOUS_CATEGORY_AXIS"


def test_a_composition_of_shares_and_prices_is_not_a_pie(caplog):
    """Cutover, and why it could not wait for its own commit.

    Recovering the identifiers made the dimension set truthful, and the *old*
    membership rule answers pie for {price_tariff, share} the moment it can
    see the share it was previously blind to. Shipping the label fix alone
    would therefore have introduced the mixed-unit pie it exists to prevent:
    before, the garbage dimensions produced a bar by accident. The rule and
    the input have to land together.
    """

    import json

    from agent.report_charts import _composition_snapshot_type

    mixed = [
        "Balancing electricity price (GEL/MWh)",
        "Share Import",
        "Share Regulated Hpp",
    ]

    with caplog.at_level(logging.INFO, logger="Enai.ReportCharts"):
        answer = _composition_snapshot_type(mixed, 3)

    assert answer == "bar"
    logged = [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("REPORT_CHART_TYPE_DISAGREEMENT ")
    ]
    assert logged, "the change from the previous rule was not reported"
    assert logged[0]["applied"] == "bar"
    assert logged[0]["previous"] == "pie"
    assert logged[0]["dimensions"] == ["price_tariff", "share"]


def test_a_composition_of_pure_shares_is_still_a_pie(caplog):
    """The cutover must not cost the compositions that already worked."""

    from agent.report_charts import _composition_snapshot_type

    pure = ["Share Import", "Share Regulated Hpp", "Share Deregulated Ren"]

    with caplog.at_level(logging.INFO, logger="Enai.ReportCharts"):
        answer = _composition_snapshot_type(pure, 3)

    assert answer == "pie"
    assert not [
        record
        for record in caplog.records
        if record.getMessage().startswith("REPORT_CHART_TYPE_DISAGREEMENT ")
    ]


def test_a_labelled_price_and_share_chart_is_no_longer_omitted():
    """The chart that did not render, pinned end to end.

    Job e4049b2d omitted prices_balancing_analysis_composition as
    REPORT_CHART_INCOMPATIBLE_UNITS. The evidence was fine: the dimensions
    were not. Reading them off display labels produced three spurious groups
    -- energy_qty for a price whose label ends in "MWh", "other" for each
    share -- and _axis_metadata refuses more than two. With identifiers it is
    two groups and a dual axis.
    """

    from agent.report_chart_rules import evidence_dimension
    from agent.report_charts import _axis_metadata
    from contracts.report_charts import ReportChartType

    series = [
        "Balancing electricity price (GEL/MWh)",
        "Share Import",
        "Share Regulated Hpp",
    ]
    units = {
        "Balancing electricity price (GEL/MWh)": "GEL/MWh",
        "Share Import": "%",
        "Share Regulated Hpp": "%",
    }

    assert {evidence_dimension(name) for name in series} == {
        "price_tariff",
        "share",
    }
    resolved = _axis_metadata(ReportChartType.BAR, series, units)
    assert resolved is not None, "the chart is still being omitted"
    assert resolved[0] == "dual"


def test_a_wide_mixed_composition_charts_over_time_instead_of_pieing():
    """Task 1's premise, re-validated: the exactness rule already covers it.

    A wide frame holding both shares and quantities used to reach the pie
    branch. It now takes the not-parts-of-one-whole path and charts the series
    over time, which is what Standard renders for this shape -- so the
    per-slice dimension filter Task 1 proposed is unnecessary.
    """

    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["columns"] = ["date", "share_hydro", "quantity_hydro"]
    table["rows"] = [
        {"date": "2026-04", "share_hydro": 0.6, "quantity_hydro": 100.0},
        {"date": "2026-05", "share_hydro": 0.8, "quantity_hydro": 120.0},
    ]
    table["unit_by_column"] = {
        "share_hydro": "ratio",
        "quantity_hydro": "thousand MWh",
    }
    table["total_row_count"] = 2
    plan_payload = _plan_payload()
    chart = plan_payload["charts"][0]
    chart.update({"purpose": "composition", "series_fields": []})
    chart.pop("x_field", None)

    decision = build_report_charts(
        ReportPlan.model_validate(plan_payload),
        ReportEvidenceManifest.model_validate(payload),
    )[0]

    assert decision.status == "built"
    assert decision.artifact.type.value != "pie"


def _composition_manifest(values_by_component, *, periods=2):
    """A wide, category-less composition frame -- the temporal-pivot shape."""

    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    components = list(values_by_component)
    table["columns"] = ["date", *components]
    table["rows"] = [
        {
            "date": f"2026-{month:02d}",
            **{name: value for name, value in values_by_component.items()},
        }
        for month in range(4, 4 + periods)
    ]
    table["unit_by_column"] = {name: "ratio" for name in components}
    table["total_row_count"] = periods
    plan_payload = _plan_payload()
    chart = plan_payload["charts"][0]
    chart.update({"purpose": "composition", "series_fields": []})
    chart.pop("x_field", None)
    return (
        ReportPlan.model_validate(plan_payload),
        ReportEvidenceManifest.model_validate(payload),
    )


def _pie_slice_total(artifact):
    return sum(
        row["value"]
        for row in artifact.data
        if isinstance(row.get("value"), (int, float))
    )


def test_a_pie_never_shows_part_of_a_whole_as_the_whole():
    """Truncate-then-classify let any composition through the category gate.

    pivot_columns was cut to eight before the type was chosen, so
    len(pivot_columns) could never exceed eight and the "few enough to read as
    slices" test could never fail. Eleven components rendered as a pie of eight
    summing to 0.727, presented as 100%.
    """

    plan, manifest = _composition_manifest(
        {f"share_c{index}": 1 / 11 for index in range(11)}
    )

    decision = build_report_charts(plan, manifest)[0]

    assert decision.status == "built"
    assert decision.artifact.type is not ReportChartType.PIE, (
        "eleven components are too many to read as slices"
    )


def test_a_pie_within_the_slice_budget_is_still_complete():
    """The gate must keep letting through the compositions that fit."""

    plan, manifest = _composition_manifest(
        {f"share_c{index}": 1 / 8 for index in range(8)}
    )

    decision = build_report_charts(plan, manifest)[0]

    assert decision.artifact.type is ReportChartType.PIE
    assert _pie_slice_total(decision.artifact) == pytest.approx(1.0)


def test_an_overflowing_composition_keeps_its_largest_components(caplog):
    """`numeric[:8]` dropped by table order, not by importance.

    Whichever components happened to be declared last were the ones lost, so a
    dominant component could vanish because of column ordering. Rank by
    contribution, and say what went.
    """

    import json

    # Declared smallest-first, so table order and importance disagree.
    values = {f"share_c{index}": (index + 1) / 55 for index in range(10)}
    plan, manifest = _composition_manifest(values)

    with caplog.at_level(logging.INFO, logger="Enai.ReportCharts"):
        decision = build_report_charts(plan, manifest)[0]

    series = list(decision.artifact.metadata.series)
    assert len(series) == 8
    # The two smallest go, not the two declared last.
    assert "share_c0" not in series and "share_c1" not in series
    assert "share_c9" in series and "share_c8" in series

    dropped = [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("REPORT_CHART_SERIES_DROPPED ")
    ]
    assert dropped, "the dropped components were not reported"
    assert dropped[0]["dropped"] == ["share_c0", "share_c1"]
    assert dropped[0]["kept_count"] == 8
