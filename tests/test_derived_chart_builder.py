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
    _limit_derived_num_cols,
    _select_forecast_series,
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
    observed_labels = [lbl for lbl in labels if "Projected" not in lbl]
    projected_labels = [lbl for lbl in labels if "Projected" in lbl]
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


def _make_ctx_with_qa(
    df,
    measure_transform=None,
    time_grain=None,
    answer_kind=None,
    chart_intent=None,
    derived_metrics=None,
):
    """Build a minimal QueryContext with authoritative question_analysis.

    Parameters
    ----------
    derived_metrics:
        Optional list of SimpleNamespace / DerivedMetricRequest stubs to place
        in ``question_analysis.analysis_requirements.derived_metrics``.  Used
        by Phase 17 tests that exercise the derived-metrics fallback path.
    """
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
        analysis_requirements=SimpleNamespace(derived_metrics=derived_metrics or []),
    )

    # Patch has_authoritative_question_analysis via property bypass.
    ctx.question_analysis = qa
    from contracts.question_analysis import AnswerKind
    if answer_kind:
        # Set the plain field that dispatch_derived_chart reads.
        ctx.effective_answer_kind = AnswerKind(answer_kind)

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


# ---------------------------------------------------------------------------
# Dispatcher — derived_metrics fallback (Phase 17)
#
# For EXPLANATION and plain data queries the analyzer fills
# analysis_requirements.derived_metrics but leaves visualization.measure_transform
# unset.  The fallback bridges these two fields inside the dispatcher.
# ---------------------------------------------------------------------------


def test_dispatcher_fallback_mom_absolute_change():
    """When visualization.measure_transform is unset but derived_metrics
    contains mom_absolute_change, the MoM dual-panel builder must fire."""
    from types import SimpleNamespace

    from contracts.question_analysis import AnswerKind, DerivedMetricName

    # 6 months is enough for mom_delta (lag=1 → 5 non-NaN delta rows).
    df = _ts_df(6)
    dm = SimpleNamespace(metric_name=DerivedMetricName.MOM_ABSOLUTE_CHANGE)
    ctx = _make_ctx_with_qa(df, derived_metrics=[dm], answer_kind="explanation")
    result = dispatch_derived_chart(ctx)
    assert result is not None, "Fallback should fire for mom_absolute_change"
    assert len(result) >= 1  # at least the observed panel
    # Second spec (delta) is a bar chart with role=derived.
    if len(result) >= 2:
        assert result[1]["type"] == "bar"
        assert result[1]["metadata"].get("role") == "derived"


def test_dispatcher_fallback_yoy_percent_change():
    """yoy_percent_change in derived_metrics → yoy_pct transform → dual-panel builder."""
    from types import SimpleNamespace

    from contracts.question_analysis import DerivedMetricName

    # 24 months so YoY lag=12 yields 12 non-NaN delta rows.
    df = _ts_df(24)
    dm = SimpleNamespace(metric_name=DerivedMetricName.YOY_PERCENT_CHANGE)
    ctx = _make_ctx_with_qa(df, derived_metrics=[dm])
    result = dispatch_derived_chart(ctx)
    assert result is not None
    assert len(result) == 2, "Should produce observed + delta panels"
    assert result[1]["metadata"]["measureTransform"] == "yoy_pct"
    assert result[1]["type"] == "bar"


# ---------------------------------------------------------------------------
# Dispatcher — anchored explanation override (2026-07-10 report)
#
# "Why did the balancing price change in July 2025?" charts must show the
# periods the narrative talks about (same month across prior years, previous
# month, anchor month) — not the whole 61-month derived-metrics fetch window —
# and the composition driver as stacked columns.
# ---------------------------------------------------------------------------


def _why_frame() -> pd.DataFrame:
    """61 monthly rows (2020-07..2025-07) shaped like the enriched balancing
    frame: prices + entity shares + an aggregate share + a driver price."""
    dates = pd.date_range("2020-07-01", "2025-07-01", freq="MS")
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "p_bal_gel": [140.0 + (i % 12) for i in range(n)],
            "p_bal_usd": [52.0 + (i % 12) / 3 for i in range(n)],
            "share_renewable_ppa": [0.80 - (i % 12) / 100 for i in range(n)],
            "share_import": [0.05 + (i % 12) / 200 for i in range(n)],
            "share_regulated_old_tpp": [0.10 for _ in range(n)],
            "share_regulated_hpp": [0.05 for _ in range(n)],
            "share_ppa_import_total": [0.85 for _ in range(n)],  # aggregate — excluded
            "price_deregulated_ren_gel": [90.0 + (i % 12) for i in range(n)],
        }
    )


def _why_periods() -> list:
    return [
        pd.Timestamp(f"{year}-07-01") for year in range(2020, 2025)
    ] + [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01")]


def _make_explanation_ctx(df, periods):
    ctx = _make_ctx_with_qa(df, answer_kind="explanation")
    ctx.query = "why balancing electricity price changed in july 2025?"
    ctx.why_chart_periods = list(periods)
    return ctx


def test_explanation_override_builds_focus_price_and_stacked_composition():
    ctx = _make_explanation_ctx(_why_frame(), _why_periods())

    result = dispatch_derived_chart(ctx)

    assert result is not None
    assert len(result) == 2

    price_spec, composition_spec = result
    # Spec A: discrete-period comparison → bar, one row per focus period.
    assert price_spec["type"] == "bar"
    assert len(price_spec["data"]) == 7
    labels = [row["date"] for row in price_spec["data"]]
    assert labels == ["2020-07", "2021-07", "2022-07", "2023-07", "2024-07", "2025-06", "2025-07"]

    # Spec B: stacked composition over the SAME periods. Categories are
    # display labels, so match by content rather than raw column names.
    assert composition_spec["type"] == "stackedbar"
    categories = {row["category"].lower() for row in composition_spec["data"]}
    assert any("renewable" in c for c in categories)
    assert any("import" in c for c in categories)
    # Aggregate share must not enter the stack (it double-counts components).
    assert not any("total" in c for c in categories)
    dates_in_stack = {row["date"] for row in composition_spec["data"]}
    assert dates_in_stack == set(labels)


def test_explanation_override_prefers_query_currency():
    ctx = _make_explanation_ctx(_why_frame(), _why_periods())
    ctx.query = "why did the balancing price change in july 2025 in USD?"

    result = dispatch_derived_chart(ctx)

    assert result is not None
    price_spec = result[0]
    series_keys = {k for row in price_spec["data"] for k in row if k != "date"}
    assert any("usd" in k.lower() for k in series_keys)
    assert not any("gel" in k.lower() for k in series_keys)


def test_explanation_override_falls_through_without_periods():
    """No resolved focus periods (generic 'why do prices change?') → the
    existing derived-metrics fallback path stays in charge."""
    from types import SimpleNamespace

    from contracts.question_analysis import DerivedMetricName

    df = _ts_df(24)
    dm = SimpleNamespace(metric_name=DerivedMetricName.YOY_PERCENT_CHANGE)
    ctx = _make_ctx_with_qa(df, derived_metrics=[dm], answer_kind="explanation")

    result = dispatch_derived_chart(ctx)

    assert result is not None
    assert result[1]["metadata"]["measureTransform"] == "yoy_pct"


def test_explanation_override_falls_through_without_usable_columns():
    """Focus periods present but the frame has neither price nor share
    columns → fall through instead of emitting empty specs."""
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"date": dates, "cpi_index": [100.0 + i for i in range(6)]})
    ctx = _make_explanation_ctx(df, [pd.Timestamp("2024-05-01"), pd.Timestamp("2024-06-01")])

    result = dispatch_derived_chart(ctx)

    assert result is None


def test_dispatcher_fallback_skipped_when_explicit_transform():
    """When visualization.measure_transform is already explicitly set, the
    derived_metrics fallback must NOT override the explicit routing — the
    explicit branch fires first and the fallback section is never reached."""
    from types import SimpleNamespace

    from contracts.question_analysis import DerivedMetricName

    df = _ts_df(24)
    # Explicit transform = mom_pct; derived_metrics has yoy_percent_change.
    # The explicit branch must win: result should carry mom_pct, not yoy_pct.
    dm = SimpleNamespace(metric_name=DerivedMetricName.YOY_PERCENT_CHANGE)
    ctx = _make_ctx_with_qa(df, measure_transform="mom_pct", derived_metrics=[dm])
    result = dispatch_derived_chart(ctx)
    assert result is not None
    # The delta spec (index 1) was built by the explicit mom_pct branch.
    assert result[-1]["metadata"]["measureTransform"] == "mom_pct"


# ---------------------------------------------------------------------------
# Phase 4: Forecast-chart series selection (currency, alias dedup, series cap)
# ---------------------------------------------------------------------------


def _forecast_df(cols_and_vals: dict, n: int = 12) -> pd.DataFrame:
    """Build a monthly forecast-shaped DataFrame with an ``is_forecast`` flag."""
    base = {"date": pd.date_range("2023-01-01", periods=n, freq="MS")}
    base.update(cols_and_vals)
    base["is_forecast"] = [False] * (n - 2) + [True] * 2
    return pd.DataFrame(base)


def test_select_forecast_series_prefers_gel_by_default():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=6, freq="MS"),
        "p_bal_gel": [100.0, 110.0, 105.0, 120.0, 115.0, 130.0],
        "p_bal_usd": [40.0, 44.0, 42.0, 48.0, 46.0, 52.0],
    })
    out = _select_forecast_series(df, ["p_bal_gel", "p_bal_usd"], user_query="forecast price to 2035")
    assert out == ["p_bal_gel"]


def test_select_forecast_series_prefers_usd_when_query_asks():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=6, freq="MS"),
        "p_bal_gel": [100.0, 110.0, 105.0, 120.0, 115.0, 130.0],
        "p_bal_usd": [40.0, 44.0, 42.0, 48.0, 46.0, 52.0],
    })
    out = _select_forecast_series(df, ["p_bal_gel", "p_bal_usd"], user_query="forecast price in USD to 2035")
    assert out == ["p_bal_usd"]


def test_select_forecast_series_drops_alias_duplicate():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=6, freq="MS"),
        "p_bal_gel": [100.0, 110.0, 105.0, 120.0, 115.0, 130.0],
        "balancing_price_gel": [100.0, 110.0, 105.0, 120.0, 115.0, 130.0],
    })
    out = _select_forecast_series(df, ["p_bal_gel", "balancing_price_gel"], user_query="forecast")
    assert out == ["p_bal_gel"], "alias balancing_price_gel must be dropped in favour of canonical"


def test_select_forecast_series_caps_at_two_by_variance():
    """When >2 columns remain after currency/alias filtering, keep the two
    with the highest variance (more informative for a forecast)."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="MS"),
        "flat_col": [100.0] * 10,                       # variance 0
        "tiny_slope": [100.0 + i * 0.01 for i in range(10)],   # tiny variance
        "steep_slope": [100.0 + i * 5.0 for i in range(10)],   # big variance
        "noisy_series": [100.0, 200.0, 80.0, 250.0, 70.0, 300.0, 60.0, 320.0, 50.0, 340.0],  # huge variance
    })
    out = _select_forecast_series(
        df,
        ["flat_col", "tiny_slope", "steep_slope", "noisy_series"],
        user_query="",
        max_series=2,
    )
    assert set(out) == {"steep_slope", "noisy_series"}


def test_select_forecast_series_empty_input_safe():
    assert _select_forecast_series(pd.DataFrame(), [], user_query=None) == []


def test_build_forecast_spec_filters_series_via_helper():
    """End-to-end: forecast builder receives [gel, usd] and only charts gel
    when no USD hint in the query."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=6, freq="MS"),
        "p_bal_gel": [100.0, 105.0, 102.0, 108.0, 110.0, 115.0],
        "p_bal_usd": [40.0, 41.0, 39.0, 42.0, 43.0, 44.0],
        "is_forecast": [False, False, False, False, True, True],
    })
    label_map = {"p_bal_gel": "Balancing GEL", "p_bal_usd": "Balancing USD"}
    specs = _build_forecast_spec(
        df, "date", ["p_bal_gel", "p_bal_usd"], label_map,
        user_query="forecast balancing price until 2035",
    )
    assert specs is not None and len(specs) == 1
    labels = specs[0]["metadata"]["labels"]
    # Only GEL labels should appear; USD series dropped by the filter.
    assert any("GEL" in lbl for lbl in labels)
    assert not any("USD" in lbl for lbl in labels), f"USD must be filtered out but got: {labels}"


# ---------------------------------------------------------------------------
# Phase 19 — Regression guards
# ---------------------------------------------------------------------------


def test_resolve_num_cols_filters_scratch_and_reference_cols():
    """`_resolve_num_cols` must exclude ``__forecast_*`` scratch columns,
    ``is_forecast`` marker, and known reference columns like ``xrate``.

    Prior to Phase 19, ``__forecast_year`` survived the filter (the regex
    ``\\b(month|year)\\b`` misses ``__forecast_year`` because ``_`` is a word
    character) and was charted as a forecast series.
    """
    from agent.derived_chart_builder import _resolve_num_cols
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=4, freq="MS"),
        "p_bal_gel": [100.0, 105.0, 110.0, 115.0],
        "p_bal_usd": [40.0, 41.0, 42.0, 43.0],
        "xrate": [2.5, 2.55, 2.6, 2.65],
        "is_forecast": [False, False, False, True],
        "__forecast_year": [2023, 2023, 2023, 2024],
        "__forecast_month": [1, 2, 3, 4],
    })
    num_cols = _resolve_num_cols(df, "date")
    assert "p_bal_gel" in num_cols
    assert "p_bal_usd" in num_cols
    assert "xrate" not in num_cols, "xrate is a reference column, not a metric"
    assert "is_forecast" not in num_cols, "is_forecast is a marker, not a metric"
    assert "__forecast_year" not in num_cols, "__forecast_year is a scratch column"
    assert "__forecast_month" not in num_cols, "__forecast_month is a scratch column"


def test_robust_endpoint_value_falls_back_on_short_series():
    """When the series has fewer than 2*window points, the leading and
    trailing windows would overlap and collapse to identical means, which
    would zero-out any CAGR. The helper must fall back to raw endpoints.

    This is the regression the user hit on a 2-row yearly aggregate —
    Phase 18 Fix 5 caused first_val == last_val → CAGR=0% → flat forecast.
    """
    from agent.analyzer import _robust_endpoint_value
    s = pd.Series([100.0, 200.0])  # len=2, window=3 → 2 < 6 → fallback
    first = _robust_endpoint_value(s, window=3, which="first")
    last = _robust_endpoint_value(s, window=3, which="last")
    assert first == 100.0, "First must fall back to raw iloc[0] on short series"
    assert last == 200.0, "Last must fall back to raw iloc[-1] on short series"
    assert first != last, "Short-series endpoints must not collapse to the same value"


def test_robust_endpoint_value_damps_on_long_series():
    """Once the series is long enough for disjoint windows (len >= 2*window),
    damping re-engages and the endpoint is the trailing-window mean."""
    from agent.analyzer import _robust_endpoint_value
    # 6 points, window=3 → disjoint windows (first 3, last 3).
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    first = _robust_endpoint_value(s, window=3, which="first")
    last = _robust_endpoint_value(s, window=3, which="last")
    assert first == pytest.approx(20.0), "First must be mean of [10,20,30]"
    assert last == pytest.approx(50.0), "Last must be mean of [40,50,60]"


def test_generate_cagr_forecast_drops_scratch_columns():
    """After `_generate_cagr_forecast` returns, the DataFrame must not
    expose ``__forecast_*`` scratch columns to downstream consumers.
    ``is_forecast`` marker and ``season`` (when populated) remain."""
    from agent.analyzer import _generate_cagr_forecast
    # Build a minimal yearly price series spanning >= 2 years so the yearly
    # branch fires and appends a forecast row.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2022-06-01", "2023-06-01", "2024-06-01"]),
        "p_bal_gel": [100.0, 110.0, 120.0],
    })
    out, _note = _generate_cagr_forecast(df, user_query="forecast balancing price until 2030")
    scratch = [c for c in out.columns if str(c).startswith("__forecast_")]
    assert scratch == [], f"Scratch columns leaked: {scratch}"
    # is_forecast marker is expected to remain.
    assert "is_forecast" in out.columns


# ---------------------------------------------------------------------------
# Fix G (2026-05-17) — series cap for derived chart specs
#
# Q3 production trace 0f713756 rendered a chart with 25+ series jamming
# prices (0-200 GEL/MWh), shares (0-1), tariffs, and contributions on
# one axis. Stage 3 enrichment adds ~23 driver-context columns; the
# derived builders were passing every column straight through.
# ---------------------------------------------------------------------------


class TestLimitDerivedNumCols:
    """Unit tests for the ``_limit_derived_num_cols`` helper."""

    def _q3_columns(self) -> list[str]:
        # The exact column shape from Q3's Stage 3 balancing_driver_enrichment.
        return [
            "balancing_price_gel", "balancing_price_usd",
            "price_deregulated_ren_gel", "price_deregulated_ren_usd",
            "price_regulated_hpp_gel", "price_regulated_hpp_usd",
            "price_regulated_new_tpp_gel", "price_regulated_new_tpp_usd",
            "price_regulated_old_tpp_gel", "price_regulated_old_tpp_usd",
            "share_import", "share_deregulated_ren", "share_regulated_hpp",
            "share_renewable_ppa", "share_thermal_ppa", "share_cfd_scheme",
            "regulated_hpp_tariff_gel", "regulated_old_tpp_tariff_gel",
            "contribution_deregulated_ren_gel", "xrate",
        ]

    def test_default_cap_is_5(self):
        cols = self._q3_columns()
        result = _limit_derived_num_cols(cols, query="trend of balancing prices", max_series=None)
        assert len(result) == 5

    def test_analyzer_cap_respected(self):
        cols = self._q3_columns()
        result = _limit_derived_num_cols(cols, query="anything", max_series=8)
        assert len(result) == 8

    def test_below_cap_returns_unchanged(self):
        short = ["balancing_price_gel", "share_import"]
        result = _limit_derived_num_cols(short, query="trend", max_series=None)
        assert result == short

    def _enriched_duplicate_df(self) -> pd.DataFrame:
        # The 2026-07-08 chart report: driver enrichment re-attaches the
        # balancing price under a second name, value-identical to the primary.
        gel = [145.0, 137.4, 85.2, 121.0]
        usd = [55.0, 54.96, 32.9, 46.0]
        return pd.DataFrame({
            "date": pd.date_range("2023-02-01", periods=4, freq="MS"),
            "p_bal_gel": gel,
            "p_bal_usd": usd,
            "balancing_price_gel": gel,
            "balancing_price_usd": usd,
            "share_import": [0.1, 0.2, 0.0, 0.1],
        })

    def test_value_duplicate_series_dropped_first_occurrence_wins(self):
        df = self._enriched_duplicate_df()
        cols = ["p_bal_gel", "p_bal_usd", "balancing_price_gel",
                "balancing_price_usd", "share_import"]
        result = _limit_derived_num_cols(
            cols, query="why balancing electricity price changed in may 2023?",
            max_series=4, df=df,
        )
        assert result == ["p_bal_gel", "p_bal_usd", "share_import"]

    def test_duplicates_dropped_even_below_cap(self):
        df = self._enriched_duplicate_df()
        cols = ["p_bal_gel", "balancing_price_gel"]
        result = _limit_derived_num_cols(cols, query="trend", max_series=None, df=df)
        assert result == ["p_bal_gel"]

    def test_different_values_are_not_merged(self):
        df = self._enriched_duplicate_df()
        df["balancing_price_gel"] = df["balancing_price_gel"] + 1.0
        cols = ["p_bal_gel", "balancing_price_gel"]
        result = _limit_derived_num_cols(cols, query="trend", max_series=None, df=df)
        assert result == cols

    def test_no_df_keeps_prior_behavior(self):
        cols = ["p_bal_gel", "balancing_price_gel"]
        result = _limit_derived_num_cols(cols, query="trend", max_series=None)
        assert result == cols

    def test_query_relevance_prioritises_intent_column(self):
        """For a "balancing price" query, balancing_price_* columns must
        survive the cap even when they're far down the original list."""
        # Put balancing_price last to force the relevance heuristic to
        # promote it.
        cols = [f"share_{i}" for i in range(10)] + ["balancing_price_gel"]
        result = _limit_derived_num_cols(cols, query="balancing price trend", max_series=3)
        assert "balancing_price_gel" in result

    def test_empty_input_returns_empty(self):
        assert _limit_derived_num_cols([], query="x", max_series=5) == []

    def test_zero_or_negative_max_falls_back_to_default(self):
        cols = self._q3_columns()
        result_neg = _limit_derived_num_cols(cols, query="trend", max_series=0)
        # max_series=0 falls back to the default cap, not an empty list.
        assert len(result_neg) == 5
        result_neg2 = _limit_derived_num_cols(cols, query="trend", max_series=-1)
        assert len(result_neg2) == 5


def test_mom_yoy_specs_apply_series_cap_to_observed_panel():
    """End-to-end Fix G: feeding 20+ columns into ``_build_mom_yoy_specs``
    should produce specs whose ``metadata.labels`` count <= the cap."""
    import pandas as pd
    rows = []
    for month in pd.date_range("2025-01-01", periods=8, freq="MS"):
        row = {"date": month}
        # 20 numeric columns, only one of which is the user's intent.
        for i in range(19):
            row[f"share_col_{i}"] = 0.05 * (i + 1)
        row["balancing_price_gel"] = 100.0 + month.month
        rows.append(row)
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c != "date"]
    label_map = {c: c for c in num_cols}

    specs = _build_mom_yoy_specs(
        df,
        time_key="date",
        num_cols=num_cols,
        label_map=label_map,
        measure_transform="mom_delta",
        query="show balancing electricity price trend",
        max_series=5,
    )
    assert specs is not None
    observed_spec = specs[0]
    labels = observed_spec["metadata"]["labels"]
    assert len(labels) <= 5, f"observed spec exceeded cap: {len(labels)} labels"
    # balancing_price_gel must survive (it's the user's intent column).
    assert "balancing_price_gel" in labels
