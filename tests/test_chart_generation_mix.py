"""2026-07-09 chart fixes: 'period' time-key recognition (Bug A) and
generation-mix composition restricted to generation techs (Bug B)."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd

from agent.chart_pipeline import _prepare_chart_source
from agent.derived_chart_builder import _build_decomposition_spec, _resolve_time_key
from models import QueryContext

# ---------------------------------------------------------------------------
# Bug A — "period" recognised as the time axis (not clobbered to nanoseconds)
# ---------------------------------------------------------------------------

def test_period_recognised_as_time_key_and_not_numeric():
    df = pd.DataFrame({
        "period": pd.to_datetime(["2015-01-01", "2015-02-01", "2015-03-01"]),
        "quantity_hydro": [800.0, 810.0, 790.0],
        "quantity_thermal": [200.0, 210.0, 190.0],
    })
    ctx = QueryContext(query="generation mix")
    ctx.df = df
    ctx.cols = list(df.columns)
    prepared, time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert time_key == "period"
    assert "period" not in num_cols  # never plotted as a data series
    # datetime preserved — NOT coerced to int64 nanosecond epoch.
    assert pd.api.types.is_datetime64_any_dtype(prepared["period"])


def test_resolve_time_key_finds_period():
    df = pd.DataFrame({"period": [1], "quantity_hydro": [1.0]})
    assert _resolve_time_key(df) == "period"


# ---------------------------------------------------------------------------
# Bug B — generation-mix decomposition keeps only generation techs
# ---------------------------------------------------------------------------

def _gen_mix_frame():
    return pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "share_hydro": [0.35, 0.40],
        "share_thermal": [0.10, 0.12],
        "share_wind": [0.02, 0.03],
        "share_export": [0.05, 0.06],   # demand-side — must be excluded
        "share_losses": [0.08, 0.07],   # demand-side — must be excluded
    })


def test_decomposition_excludes_demand_shares():
    df = _gen_mix_frame()
    num_cols = [c for c in df.columns if c != "period"]
    spec = _build_decomposition_spec(df, "period", num_cols, {c: c for c in df.columns})
    categories = {r["category"] for r in spec[0]["data"]}
    assert categories == {"share_hydro", "share_thermal", "share_wind"}
    # dates are formatted YYYY-MM, not nanoseconds
    assert spec[0]["data"][0]["date"] == "2024-01"


def test_non_generation_decomposition_is_untouched():
    # Balancing composition has no demand/transit share columns → unchanged.
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01"]),
        "share_import": [0.3],
        "share_regulated_hpp": [0.7],
    })
    spec = _build_decomposition_spec(
        df, "period", ["share_import", "share_regulated_hpp"],
        {c: c for c in df.columns},
    )
    categories = {r["category"] for r in spec[0]["data"]}
    assert categories == {"share_import", "share_regulated_hpp"}


def test_generation_aggregate_shares_are_kept():
    """Regression: the exclusion approach must NOT drop generation aggregates
    such as share_renewable — only demand/transit shares are removed."""
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "share_renewable": [0.30, 0.32],
        "share_thermal": [0.50, 0.48],
        "share_hydro": [0.20, 0.20],
        "share_export": [0.05, 0.06],   # demand — excluded
    })
    num_cols = [c for c in df.columns if c != "period"]
    spec = _build_decomposition_spec(df, "period", num_cols, {c: c for c in df.columns})
    categories = {r["category"] for r in spec[0]["data"]}
    assert categories == {"share_renewable", "share_thermal", "share_hydro"}
