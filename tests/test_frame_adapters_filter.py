"""Tests for the price threshold filter vocabulary handling in frame_adapters.

The analyzer emits filter.metric in tool-params vocabulary (``balancing``,
``deregulated``, ``guaranteed_capacity``, ``exchange_rate``) but frame rows
carry display-oriented metric names (``balancing_price`` etc).  These tests
lock in the mapping so filter thresholds actually filter instead of silently
passing every row through.
"""
import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import frame_adapters  # noqa: E402
from contracts.question_analysis import FilterCondition, FilterOperator  # noqa: E402


def _price_df():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "p_bal_gel": [10.0, 17.5, 9.0, 22.0],
        "p_dereg_gel": [8.0, 12.0, 14.0, 19.0],
    })


def test_balancing_threshold_filter_applies_to_frame_rows():
    df = _price_df()
    filt = FilterCondition(
        metric="balancing",
        operator=FilterOperator.GT,
        value=15.0,
        unit="tetri",
    )
    frame = frame_adapters.adapt_prices(df, filter_cond=filt)

    balancing_rows = [r for r in frame.rows if r["metric"] == "balancing_price"]
    dereg_rows = [r for r in frame.rows if r["metric"] == "deregulated_price"]

    # Only balancing rows above threshold survive; deregulated rows untouched.
    assert [r["value"] for r in balancing_rows] == [17.5, 22.0]
    assert [r["value"] for r in dereg_rows] == [8.0, 12.0, 14.0, 19.0]


def test_deregulated_threshold_filter_applies_to_frame_rows():
    df = _price_df()
    filt = FilterCondition(
        metric="deregulated",
        operator=FilterOperator.GTE,
        value=14.0,
    )
    frame = frame_adapters.adapt_prices(df, filter_cond=filt)

    bal_rows = [r for r in frame.rows if r["metric"] == "balancing_price"]
    dereg_rows = [r for r in frame.rows if r["metric"] == "deregulated_price"]

    assert [r["value"] for r in dereg_rows] == [14.0, 19.0]
    assert len(bal_rows) == 4  # unfiltered


def test_metric_matches_recognizes_contract_vocabulary():
    assert frame_adapters._metric_matches("balancing_price", "balancing")
    assert frame_adapters._metric_matches("deregulated_price", "deregulated")
    assert frame_adapters._metric_matches("guaranteed_capacity_price", "guaranteed_capacity")
    assert frame_adapters._metric_matches("exchange_rate", "exchange_rate")
    # Exact equality still works for display-vocabulary callers.
    assert frame_adapters._metric_matches("balancing_price", "balancing_price")
    # Empty contract metric matches everything.
    assert frame_adapters._metric_matches("balancing_price", "")
    # Unknown contract metric does not match unrelated rows.
    assert not frame_adapters._metric_matches("balancing_price", "deregulated")
