"""Tests for agent.metric_registry — dispatch registry and individual compute functions."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import numpy as np
import pandas as pd
import pytest

from agent.metric_registry import (
    MetricContext,
    METRIC_REGISTRY,
    compute_mom,
    compute_yoy,
    compute_share_delta_mom,
    compute_correlation,
    compute_trend_slope,
    compute_scenario,
    dispatch_metric,
    row_value,
    share_value,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(months=3):
    """Build a simple price+share DataFrame with sequential months."""
    dates = pd.date_range("2024-01-01", periods=months, freq="MS")
    return pd.DataFrame({
        "date": dates,
        "p_bal_gel": [50.0 + i * 5 for i in range(months)],
        "p_bal_usd": [18.0 + i * 2 for i in range(months)],
        "xrate": [2.7 + i * 0.1 for i in range(months)],
        "share_import": [0.30 - i * 0.02 for i in range(months)],
        "share_thermal_ppa": [0.40 + i * 0.01 for i in range(months)],
    })


def _make_context(df=None, months=3):
    """Build a MetricContext suitable for most tests."""
    if df is None:
        df = _make_df(months)
    current_row = df.tail(1)
    previous_row = df.iloc[[-2]] if len(df) >= 2 else pd.DataFrame()
    current_ts = pd.Timestamp(current_row["date"].iloc[0])
    previous_ts = pd.Timestamp(previous_row["date"].iloc[0]) if not previous_row.empty else None

    cur_shares = {
        "share_import": float(current_row["share_import"].iloc[0]),
        "share_thermal_ppa": float(current_row["share_thermal_ppa"].iloc[0]),
    }
    prev_shares = {
        "share_import": float(previous_row["share_import"].iloc[0]) if not previous_row.empty else 0.0,
        "share_thermal_ppa": float(previous_row["share_thermal_ppa"].iloc[0]) if not previous_row.empty else 0.0,
    }

    # YoY row — same month last year (won't exist in 3-month fixture)
    yoy_row = pd.DataFrame()
    yoy_ts = None
    yoy_shares: dict[str, float] = {}

    return MetricContext(
        df=df,
        time_col="date",
        current_ts=current_ts,
        current_row=current_row,
        previous_ts=previous_ts,
        previous_row=previous_row,
        cur_shares=cur_shares,
        prev_shares=prev_shares,
        yoy_row=yoy_row,
        yoy_ts=yoy_ts,
        yoy_shares=yoy_shares,
        correlation_results={},
    )


def _base_record(metric_name, metric, current_ts=None):
    return {
        "record_type": "derived",
        "derived_metric_name": metric_name,
        "metric": metric,
        "target_metric": None,
        "period": str(current_ts) if current_ts else None,
        "comparison_period": None,
        "current_value": None,
        "previous_value": None,
        "absolute_change": None,
        "percent_change": None,
        "correlation_value": None,
        "trend_slope": None,
        "formula": "",
    }


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_row_value_finds_alias(self):
        df = pd.DataFrame({"p_bal_gel": [42.0]})
        assert row_value(df, "p_bal_gel") == 42.0

    def test_row_value_returns_none_for_missing(self):
        df = pd.DataFrame({"other_col": [1.0]})
        assert row_value(df, "p_bal_gel") is None

    def test_row_value_empty_frame(self):
        assert row_value(pd.DataFrame(), "p_bal_gel") is None

    def test_share_value_existing(self):
        assert share_value({"share_import": 0.3}, "share_import") == 0.3

    def test_share_value_missing(self):
        assert share_value({}, "share_import") is None


# ---------------------------------------------------------------------------
# MoM tests
# ---------------------------------------------------------------------------

class TestComputeMom:
    def test_mom_absolute_change(self):
        mctx = _make_context()
        record = _base_record("mom_absolute_change", "p_bal_gel", mctx.current_ts)
        result = compute_mom({}, record, mctx)

        assert result is not None
        assert result["absolute_change"] == pytest.approx(5.0, abs=1e-4)
        assert result["current_value"] == pytest.approx(60.0, abs=1e-4)
        assert result["previous_value"] == pytest.approx(55.0, abs=1e-4)
        assert result["formula"] == "current_value - previous_value"
        assert len(result["source_cells"]) == 2

    def test_mom_percent_change(self):
        mctx = _make_context()
        record = _base_record("mom_percent_change", "p_bal_gel", mctx.current_ts)
        result = compute_mom({}, record, mctx)

        assert result is not None
        expected_pct = (5.0 / 55.0) * 100.0
        assert result["percent_change"] == pytest.approx(expected_pct, abs=0.01)

    def test_mom_share_metric(self):
        mctx = _make_context()
        record = _base_record("mom_absolute_change", "share_import", mctx.current_ts)
        result = compute_mom({}, record, mctx)

        assert result is not None
        assert result["absolute_change"] == pytest.approx(-0.02, abs=1e-4)

    def test_mom_missing_data_returns_none(self):
        mctx = _make_context()
        record = _base_record("mom_absolute_change", "nonexistent_metric", mctx.current_ts)
        result = compute_mom({}, record, mctx)
        assert result is None


# ---------------------------------------------------------------------------
# YoY tests
# ---------------------------------------------------------------------------

class TestComputeYoy:
    def test_yoy_with_data(self):
        # Build a 13-month dataset so YoY has a match
        df = _make_df(months=13)
        mctx = _make_context(df, months=13)
        # Manually set yoy row
        mctx.yoy_row = df.iloc[[0]]
        mctx.yoy_ts = pd.Timestamp(df["date"].iloc[0])

        record = _base_record("yoy_absolute_change", "p_bal_gel", mctx.current_ts)
        result = compute_yoy({}, record, mctx)

        assert result is not None
        assert result["absolute_change"] == pytest.approx(60.0, abs=1e-4)  # 110 - 50
        assert result["formula"] == "current_value - previous_value_same_period_last_year"
        assert len(result["source_cells"]) == 2
        assert result["source_cells"][1]["role"] == "yoy_previous"

    def test_yoy_no_data_returns_none(self):
        mctx = _make_context()  # 3 months, no YoY match
        record = _base_record("yoy_absolute_change", "p_bal_gel", mctx.current_ts)
        result = compute_yoy({}, record, mctx)
        assert result is None


# ---------------------------------------------------------------------------
# Share delta tests
# ---------------------------------------------------------------------------

class TestComputeShareDelta:
    def test_share_delta_mom(self):
        mctx = _make_context()
        record = _base_record("share_delta_mom", "share_import", mctx.current_ts)
        result = compute_share_delta_mom({}, record, mctx)

        assert result is not None
        assert result["absolute_change"] == pytest.approx(-0.02, abs=1e-4)
        assert result["formula"] == "current_share - previous_share"

    def test_share_delta_missing_share(self):
        mctx = _make_context()
        record = _base_record("share_delta_mom", "share_nonexistent", mctx.current_ts)
        result = compute_share_delta_mom({}, record, mctx)
        assert result is None


# ---------------------------------------------------------------------------
# Correlation tests
# ---------------------------------------------------------------------------

class TestComputeCorrelation:
    def test_correlation_found(self):
        mctx = _make_context()
        mctx.correlation_results = {"p_bal_gel": {"share_import": 0.85}}
        record = _base_record("correlation_to_target", "share_import", mctx.current_ts)
        result = compute_correlation({}, record, mctx)

        assert result is not None
        assert result["correlation_value"] == pytest.approx(0.85, abs=1e-4)
        assert "corr(share_import, p_bal_gel)" in result["formula"]
        assert "source_cells" in result
        assert len(result["source_cells"]) == 2
        roles = {c["role"] for c in result["source_cells"]}
        assert "source_series" in roles
        assert "target_series" in roles

    def test_correlation_not_found(self):
        mctx = _make_context()
        mctx.correlation_results = {}
        record = _base_record("correlation_to_target", "share_import", mctx.current_ts)
        result = compute_correlation({}, record, mctx)
        assert result is None


# ---------------------------------------------------------------------------
# Trend slope tests
# ---------------------------------------------------------------------------

class TestComputeTrendSlope:
    def test_trend_slope_positive(self):
        mctx = _make_context(months=5)
        record = _base_record("trend_slope", "p_bal_gel", mctx.current_ts)
        result = compute_trend_slope({}, record, mctx)

        assert result is not None
        assert result["trend_slope"] == pytest.approx(5.0, abs=0.1)
        assert result["source_row_count"] == 5
        assert "source_cells" in result
        assert result["source_cells"][0]["role"] == "trend_series"
        assert "min_value" in result["source_cells"][0]
        assert "max_value" in result["source_cells"][0]

    def test_trend_slope_too_few_rows(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")],
            "p_bal_gel": [50.0],
            "share_import": [0.3],
            "share_thermal_ppa": [0.4],
        })
        mctx = _make_context(df, months=1)
        record = _base_record("trend_slope", "p_bal_gel", mctx.current_ts)
        result = compute_trend_slope({}, record, mctx)
        assert result is None

    def test_trend_slope_missing_column(self):
        mctx = _make_context()
        record = _base_record("trend_slope", "nonexistent", mctx.current_ts)
        result = compute_trend_slope({}, record, mctx)
        assert result is None


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------

class TestComputeScenario:
    def test_scenario_scale(self):
        mctx = _make_context()
        record = _base_record("scenario_scale", "p_bal_gel", mctx.current_ts)
        request = {"scenario_factor": 1.2, "scenario_volume": 1, "scenario_aggregation": "sum"}
        result = compute_scenario(request, record, mctx)

        assert result is not None
        assert result["record_type"] == "scenario"
        assert result["scenario_factor"] == 1.2
        # baseline = sum(50, 55, 60) = 165; scaled = 165 * 1.2 = 198
        assert result["aggregate_result"] == pytest.approx(198.0, abs=0.1)
        assert result["baseline_aggregate"] == pytest.approx(165.0, abs=0.1)
        assert result["delta_aggregate"] == pytest.approx(33.0, abs=0.1)
        assert "source_cells" in result
        assert result["source_cells"][0]["role"] == "scenario_series"
        assert "min_value" in result["source_cells"][0]

    def test_scenario_offset(self):
        mctx = _make_context()
        record = _base_record("scenario_offset", "p_bal_gel", mctx.current_ts)
        request = {"scenario_factor": 10, "scenario_volume": 1, "scenario_aggregation": "sum"}
        result = compute_scenario(request, record, mctx)

        assert result is not None
        # baseline = 165; offset = sum(60, 65, 70) = 195
        assert result["aggregate_result"] == pytest.approx(195.0, abs=0.1)
        assert result["delta_aggregate"] == pytest.approx(30.0, abs=0.1)

    def test_scenario_payoff(self):
        mctx = _make_context()
        record = _base_record("scenario_payoff", "p_bal_gel", mctx.current_ts)
        request = {"scenario_factor": 55.0, "scenario_volume": 100, "scenario_aggregation": "sum"}
        result = compute_scenario(request, record, mctx)

        assert result is not None
        assert result["positive_sum"] is not None or result["negative_sum"] is not None
        assert result["market_component_aggregate"] is not None
        assert result["combined_total_aggregate"] is not None
        assert result["scenario_volume"] == 100

    def test_scenario_identity_scale_skipped(self):
        mctx = _make_context()
        record = _base_record("scenario_scale", "p_bal_gel", mctx.current_ts)
        request = {"scenario_factor": 1.0, "scenario_volume": 1, "scenario_aggregation": "sum"}
        result = compute_scenario(request, record, mctx)
        assert result is None

    def test_scenario_zero_offset_skipped(self):
        mctx = _make_context()
        record = _base_record("scenario_offset", "p_bal_gel", mctx.current_ts)
        request = {"scenario_factor": 0, "scenario_volume": 1, "scenario_aggregation": "sum"}
        result = compute_scenario(request, record, mctx)
        assert result is None

    def test_scenario_missing_column(self):
        mctx = _make_context()
        record = _base_record("scenario_scale", "nonexistent", mctx.current_ts)
        request = {"scenario_factor": 1.5}
        result = compute_scenario(request, record, mctx)
        assert result is None


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_known_metric_dispatches(self):
        mctx = _make_context()
        record = _base_record("mom_absolute_change", "p_bal_gel", mctx.current_ts)
        result = dispatch_metric({}, record, mctx)
        assert result is not None
        assert result["absolute_change"] is not None

    def test_unknown_metric_returns_none(self):
        mctx = _make_context()
        record = _base_record("unknown_metric_type", "p_bal_gel", mctx.current_ts)
        result = dispatch_metric({}, record, mctx)
        assert result is None

    def test_all_registered_names(self):
        expected = {
            "mom_absolute_change", "mom_percent_change",
            "yoy_absolute_change", "yoy_percent_change",
            "share_delta_mom",
            "correlation_to_target",
            "trend_slope",
            "scenario_scale", "scenario_offset", "scenario_payoff",
        }
        assert set(METRIC_REGISTRY.keys()) == expected

    def test_cfd_scheme_in_balancing_share_metadata(self):
        """CfD_scheme must be tracked as a separate share — USD-linked, moderate cost."""
        from config_metrics.metric_config import BALANCING_SHARE_METADATA
        assert "share_cfd_scheme" in BALANCING_SHARE_METADATA
        meta = BALANCING_SHARE_METADATA["share_cfd_scheme"]
        assert meta["usd_linked"] is True
        assert meta["cost"] in ("moderate", "expensive")  # aligned with support-scheme treatment
        # share_all_ppa must NOT include CfD (literal PPA only)
        assert "share_cfd_scheme" not in BALANCING_SHARE_METADATA.get("share_all_ppa", {}).get("components", [])


# ---------------------------------------------------------------------------
# Integration: end-to-end via _build_requested_analysis_evidence
# ---------------------------------------------------------------------------

class TestBuildRequestedAnalysisEvidence:
    """Regression tests ensuring the refactored dispatch produces the same
    evidence records as the prior if/elif chain."""

    def test_mom_evidence_via_analyzer(self):
        from agent.analyzer import _build_requested_analysis_evidence
        from unittest.mock import MagicMock

        df = _make_df(months=3)
        current_row = df.tail(1)
        previous_row = df.iloc[[-2]]
        current_ts = pd.Timestamp(current_row["date"].iloc[0])
        previous_ts = pd.Timestamp(previous_row["date"].iloc[0])

        ctx = MagicMock()
        ctx.question_analysis = None
        ctx.question_analysis_source = ""
        ctx.query = "test"
        ctx.correlation_results = {}

        # Monkey-patch _active_analysis_requests to return a known request
        import agent.analyzer as az
        original = az._active_analysis_requests
        az._active_analysis_requests = lambda _ctx: [
            {"metric_name": "mom_absolute_change", "metric": "p_bal_gel"},
        ]
        try:
            result = _build_requested_analysis_evidence(
                ctx, df, "date",
                current_ts, current_row,
                previous_ts, previous_row,
                {"share_import": 0.26},
                {"share_import": 0.28},
            )
        finally:
            az._active_analysis_requests = original

        assert not result.empty
        row = result.iloc[0]
        assert row["derived_metric_name"] == "mom_absolute_change"
        assert row["absolute_change"] == pytest.approx(5.0, abs=1e-4)
        assert row["source_cells"] is not None
        assert len(row["source_cells"]) == 2

    def test_scenario_evidence_via_analyzer(self):
        from agent.analyzer import _build_requested_analysis_evidence
        from unittest.mock import MagicMock

        df = _make_df(months=5)
        current_row = df.tail(1)
        previous_row = df.iloc[[-2]]
        current_ts = pd.Timestamp(current_row["date"].iloc[0])
        previous_ts = pd.Timestamp(previous_row["date"].iloc[0])

        ctx = MagicMock()
        ctx.question_analysis = None
        ctx.question_analysis_source = ""
        ctx.query = "test"
        ctx.correlation_results = {}

        import agent.analyzer as az
        original = az._active_analysis_requests
        az._active_analysis_requests = lambda _ctx: [
            {
                "metric_name": "scenario_scale",
                "metric": "p_bal_gel",
                "scenario_factor": 1.5,
                "scenario_volume": 1,
                "scenario_aggregation": "sum",
            },
        ]
        try:
            result = _build_requested_analysis_evidence(
                ctx, df, "date",
                current_ts, current_row,
                previous_ts, previous_row,
                {}, {},
            )
        finally:
            az._active_analysis_requests = original

        assert not result.empty
        row = result.iloc[0]
        assert row["record_type"] == "scenario"
        assert row["scenario_factor"] == 1.5
        assert row["baseline_aggregate"] is not None
        assert row["delta_aggregate"] is not None
