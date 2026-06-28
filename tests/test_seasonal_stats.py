"""Focused tests for analysis.seasonal_stats."""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from analysis.seasonal_stats import (
    calculate_seasonal_stats,
    detect_monthly_timeseries,
    format_seasonal_stats,
)
from analysis.stats import is_intensive_metric


def _monthly(value_col: str, base: float, per_year: float) -> "pd.DataFrame":
    rows = []
    for year in range(2015, 2026):
        for month in range(1, 13):
            rows.append({"date": f"{year}-{month:02d}", value_col: base + (year - 2015) * per_year})
    return pd.DataFrame(rows)


def test_is_intensive_metric_classification():
    assert is_intensive_metric("p_bal_gel") is True
    assert is_intensive_metric("tariff_usd") is True
    assert is_intensive_metric("xrate") is True
    assert is_intensive_metric("quantity_tech") is False
    assert is_intensive_metric("generation") is False


def test_price_series_reports_average_levels_not_summed():
    """A price (~123 → ~158/MWh) must report yearly AVERAGE levels, never the
    12x annual sum (the 1,482-GEL/MWh bug, prod trace d04b96ae)."""
    df = _monthly("p_bal_gel", base=123.0, per_year=3.5)  # 2015≈123, 2025≈158

    stats = calculate_seasonal_stats(df, "date", "p_bal_gel")

    assert stats["aggregate_kind"] == "average"
    # Levels must be in the hundreds, NOT ~12x (which would be >1,400).
    assert 100 < stats["first_year_total"] < 200
    assert 100 < stats["last_year_total"] < 200
    # The growth % is unchanged by mean-vs-sum (ratio-invariant).
    assert stats["overall_growth_pct"] > 0
    assert "avg" in format_seasonal_stats(stats)


def test_quantity_series_still_reports_annual_totals():
    """Extensive quantities keep summing to a meaningful annual total."""
    df = _monthly("generation", base=100.0, per_year=10.0)

    stats = calculate_seasonal_stats(df, "date", "generation")

    assert stats["aggregate_kind"] == "total"
    # 12 months × ~100 ≈ 1,200 annual total.
    assert stats["first_year_total"] > 1000


def test_detect_monthly_timeseries_returns_none_for_empty_df():
    df = pd.DataFrame(columns=["date", "p_bal_gel"])

    assert detect_monthly_timeseries(df) is None


def test_detect_monthly_timeseries_returns_none_for_all_null_time_values():
    df = pd.DataFrame(
        {
            "date": [None, None],
            "p_bal_gel": [50.0, 55.0],
        },
    )

    assert detect_monthly_timeseries(df) is None
