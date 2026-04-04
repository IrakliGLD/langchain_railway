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

from analysis.seasonal_stats import detect_monthly_timeseries


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
