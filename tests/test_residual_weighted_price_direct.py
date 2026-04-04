"""Focused tests for deterministic residual weighted-price answers."""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import summarizer
from models import QueryContext


def test_explicit_residual_component_query_filters_to_threshold_months():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01", "2021-07-01", "2021-08-01"]),
            "share_ppa_import_total": [0.999, 0.994, 0.880],
            "residual_contribution_ppa_import_gel": [72.7272, 66.6362, 44.0],
            "residual_contribution_ppa_import_usd": [25.225, 21.7686, 15.0],
        }
    )
    ctx = QueryContext(
        query=(
            "Calculate the weighted average balancing price for electricity from Renewable PPA, Import, "
            "Thermal Generation PPA, and CfD Scheme for June 2020, July 2021, and August 2021, only if "
            "these entities collectively contribute 99% or more to the total balancing composition in those months."
        ),
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
    )

    out = summarizer.summarize_data(ctx)

    assert out.summary_source == "deterministic_residual_weighted_price_direct"
    assert "Renewable PPA + Import + Thermal Generation PPA + CfD Scheme" in out.summary
    assert "June 2020" in out.summary
    assert "July 2021" in out.summary
    assert "August 2021" not in out.summary
    assert "99.0%" in out.summary

