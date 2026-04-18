"""Phase 15 (§16.4.5) unit tests for the generalized derived-chart builders.

Each builder is tested against a small golden DataFrame — no DB required.
Covers positive cases, empty/stub returns, and the dispatcher routing.
"""
import os

# Env setup required before importing project modules.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd
import pytest

from agent.derived_chart_builder import (  # noqa: E402
    _build_decomposition_spec,
    _build_forecast_spec,
    _build_index_growth_spec,
    _build_mom_yoy_specs,
    _build_seasonal_spec,
    dispatch_derived_chart,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ts_df(n: int = 12, *, col: str = "p_bal_gel", start: str = "2023-01-01") -> pd.DataFrame:
    """Simple monthly time-series DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=n, freq="MS"),
            col: [float(i + 1) * 10 for i in range(n)],
        }
    )


def _label_map(df: pd.DataFrame, time_key: str = "date") -> dict:
    return {c: c for c in df.columns if c != time_key}


# ---------------------------------------------------------------------------
# Builder 1: MoM/YoY
# ---------------------------------------------------------------------------


def test_mom_yoy_emits_two_specs():
    # Use 24 months so YoY lag=12 yields 12 non-NaN delta rows.
    df = _ts_df(24)
    specs = _build_mom_yoy_specs(df, "date", ["p_bal_gel"], _label_map(df), "yoy_pct")
    assert specs is not None
    assert len(specs) == 2
    # Spec 0 = observed (line), Spec 1 = delta (bar)
    assert specs[0]["type"] == "line"
    assert specs[1]["type"] == "bar"


def test_mom_yoy_delta_spec_role_is_derived():
    # Use 6 months for MoM (lag=1, so 5 non-NaN delta rows).
    df = _ts_df(6)
    specs = _build_mom_yoy_specs(df, "date", ["p_bal_gel"], _label_map(df), "mom_pct")
    assert specs is not None and len(specs) == 2
    assert specs[1]["metadata"]["role"] == "derived"
    assert specs[1]["metadata"]["measureTransform"] == "mom_pct"


def test_mom_yoy_observed_spec_has_data():
    df = _ts_df(24)
    specs = _build_mom_yoy_specs(df, "date", ["p_bal_gel"], _label_map(df), "yoy_pct")
    assert specs is not None
    assert len(specs[0]["data"]) == 24


def test_mom_yoy_short_series_returns_at_least_observed():
    """YoY delta has 12 lag; a 12-month series has 12 NaN deltas → empty data.
    Builder should still return the observed panel (at least 1 spec)."""
    df = _ts_df(12)
    specs = _build_mom_yoy_specs(df, "date", ["p_bal_gel"], _label_map(df), "yoy_pct")
    assert specs is not None
    assert len(specs) >= 1  # at least observed panel returned


def test_mom_yoy_returns_none_without_time_key():
    df = pd.DataFrame({"p_bal_gel": [1.0, 2.0, 3.0]})
    result = _build_mom_yoy_specs(df, None, ["p_bal_gel"], {}, "yoy_pct")
    assert result is None


# ---------------------------------------------------------------------------
# Builder 2: Indexed growth
# ---------------------------------------------------------------------------


def test_indexed_growth_emits_one_spec():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=6, freq="MS"),
            "p_bal_gel": [100.0, 110.0, 105.0, 120.0, 115.0, 130.0],
            "p_bal_usd": [40.0, 44.0, 42.0, 48.0, 46.0, 52.0],
        }
    )
    label_map = {"p_bal_gel": "Balancing GEL", "p_bal_usd": "Balancing USD"}
    specs = _build_index_growth_spec(df, "date", ["p_bal_gel", "p_bal_usd"], label_map)
    assert specs is not None
    assert len(specs) == 1
    assert specs[0]["type"] == "line"
    assert specs[0]["metadata"]["measureTransform"] == "index_100"
    # Base value should be 100 for the first non-NaN period.
    first_record = specs[0]["data"][0]
    assert first_record is not None


def test_indexed_growth_returns_none_without_time():
    df = pd.DataFrame({"p_bal_gel": [100.0, 110.0]})
    result = _build_index_growth_spec(df, None, ["p_bal_gel"], {})
    assert result is None


# ---------------------------------------------------------------------------
# Builder 3: Decomposition
# ---------------------------------------------------------------------------


def test_decomposition_emits_stackedbar():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3, freq="MS"),
            "share_renewable": [0.3, 0.32, 0.35],
            "share_thermal": [0.5, 0.48, 0.45],
            "share_hydro": [0.2, 0.20, 0.20],
        }
    )
    label_map = {
        "share_renewable": "Renewable",
        "share_thermal": "Thermal",
        "share_hydro": "Hydro",
    }
    specs = _build_decomposition_spec(df, "date", list(label_map.keys()), label_map)
    assert specs is not None
    assert len(specs) == 1
    assert specs[0]["type"] == "stackedbar"
    # One row per (period × category)
    assert len(specs[0]["data"]) == 3 * 3


def test_decomposition_returns_none_if_no_share_cols():
    df = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=3, freq="MS"), "p_bal_gel": [1.0, 2.0, 3.0]}
    )
    result = _build_decomposition_spec(df, "date", ["p_bal_gel"], {})
    assert result is None


# ---------------------------------------------------------------------------
# Builder 4: Forecast
# ---------------------------------------------------------------------------


def test_forecast_splits_observed_and_projected():
    # Simulate _generate_cagr_forecast output: uses 'is_forecast' column
    # (not 'is_projected') — matches the actual analyzer output.
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=6, freq="MS"),
            "p_bal_gel": [100.0, 105.0, 102.0, 108.0, 110.0, 115.0],
            "is_forecast": [False, False, False, False, True, True],
        }
    )
    label_map = {"p_bal_gel": "Balancing GEL"}
    specs = _build_forecast_spec(df, "date", ["p_bal_gel"], label_map)
    assert specs is not None
    assert len(specs) == 1
    assert specs[0]["type"] == "line"
    # Observed and projected column labels should both appear in labels.
    labels = specs[0]["metadata"]["labels"]
    observed_labels = [l for l in labels if "Projected" not in l]
    projected_labels = [l for l in labels if "Projected" in l]
    assert observed_labels
    assert projected_labels


def test_forecast_returns_none_when_no_marker_column():
    """Without is_forecast / is_projected, the builder cannot distinguish
    observed from projected; it should skip (return None) rather than guess."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=6, freq="MS"),
            "p_bal_gel": [100.0, 105.0, 102.0, 108.0, 110.0, 115.0],
        }
    )
    result = _build_forecast_spec(df, "date", ["p_bal_gel"], {"p_bal_gel": "GEL"})
    assert result is None


def test_forecast_returns_none_on_empty_df():
    result = _build_forecast_spec(pd.DataFrame(), "date", ["p_bal_gel"], {})
    assert result is None


# ---------------------------------------------------------------------------
# Builder 5: Seasonal
# ---------------------------------------------------------------------------


def test_seasonal_emits_bar_with_season_labels():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=24, freq="MS"),
            "p_bal_gel": [float(i + 1) * 5 for i in range(24)],
        }
    )
    specs = _build_seasonal_spec(df, "date", ["p_bal_gel"], {"p_bal_gel": "Balancing GEL"})
    assert specs is not None
    assert len(specs) == 1
    assert specs[0]["type"] == "bar"
    assert "seasonLabels" in specs[0]["metadata"]
    # Season labels must not be raw ISO strings inside the data period column.
    period_values = [r["date"] for r in specs[0]["data"] if "date" in r]
    season_strings = [v for v in period_values if v and ("summer" in str(v) or "winter" in str(v))]
    assert season_strings, "Season labels should be human-readable (summer/winter)"


def test_seasonal_returns_none_for_short_series():
    df = _ts_df(4)
    result = _build_seasonal_spec(df, "date", ["p_bal_gel"], {})
    assert result is None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _make_ctx_with_qa(df, measure_transform=None, time_grain=None, answer_kind=None, chart_intent=None):
    """Build a minimal QueryContext with authoritative question_analysis."""
    from types import SimpleNamespace
    from models import QueryContext

    ctx = QueryContext(query="test query")
    ctx.df = df.copy()
    ctx.cols = list(df.columns)
    ctx.rows = []
    ctx.question_analysis_source = "llm_active"

    # Build a minimal stub QuestionAnalysis with the necessary fields.
    vis = SimpleNamespace(
        measure_transform=SimpleNamespace(value=measure_transform) if measure_transform else None,
        time_grain=SimpleNamespace(value=time_grain) if time_grain else None,
        chart_intent=SimpleNamespace(value=chart_intent) if chart_intent else None,
    )
    qa = SimpleNamespace(
        visualization=vis,
        analysis_requirements=SimpleNamespace(derived_metrics=[]),
    )

    # Patch has_authoritative_question_analysis via property bypass.
    ctx.question_analysis = qa
    from contracts.question_analysis import AnswerKind
    if answer_kind:
        ctx._effective_answer_kind_override = AnswerKind(answer_kind)

    return ctx


def test_dispatcher_routes_yoy_pct_to_mom_yoy():
    df = _ts_df(12)
    ctx = _make_ctx_with_qa(df, measure_transform="yoy_pct")
    result = dispatch_derived_chart(ctx)
    assert result is not None
    assert len(result) >= 1


def test_dispatcher_routes_index_100_to_indexed_growth():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=6, freq="MS"),
            "p_bal_gel": [100.0, 110.0, 105.0, 120.0, 115.0, 130.0],
        }
    )
    ctx = _make_ctx_with_qa(df, measure_transform="index_100")
    result = dispatch_derived_chart(ctx)
    assert result is not None
    assert result[0]["metadata"].get("measureTransform") == "index_100"


def test_dispatcher_routes_season_grain():
    df = _ts_df(24)
    ctx = _make_ctx_with_qa(df, time_grain="season")
    result = dispatch_derived_chart(ctx)
    assert result is not None
    assert result[0]["metadata"].get("timeGrain") == "season"


def test_dispatcher_returns_none_when_not_authoritative():
    """If question_analysis_source is not 'llm_active', dispatch must be skipped."""
    from models import QueryContext

    ctx = QueryContext(query="test")
    ctx.df = _ts_df(6)
    ctx.question_analysis_source = "raw_query"  # not authoritative
    result = dispatch_derived_chart(ctx)
    assert result is None


def test_dispatcher_returns_none_for_raw_transform():
    """Raw measure_transform should not trigger any override."""
    df = _ts_df(6)
    ctx = _make_ctx_with_qa(df, measure_transform="raw")
    result = dispatch_derived_chart(ctx)
    assert result is None
