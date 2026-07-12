import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.analyzer import (
    _build_correlation_matrix_from_frame,
    _build_correlation_metadata,
    _scope_correlation_frame,
)
from models import QueryContext


def test_correlation_fallback_is_limited_to_requested_period():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=60, freq="MS"),
        "p_bal_gel": range(60),
        "xrate": range(60),
    })
    ctx = QueryContext(
        query="correlation in 2023",
        tool_params={"start_date": "2023-01-01", "end_date": "2023-12-31"},
    )
    scoped = _scope_correlation_frame(ctx, df)
    assert len(scoped) == 12
    assert scoped["date"].min() == pd.Timestamp("2023-01-01")
    assert scoped["date"].max() == pd.Timestamp("2023-12-01")


def test_correlation_metadata_exposes_pairwise_n_period_and_uncertainty():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=12, freq="MS"),
        "p_bal_gel": range(12),
        "xrate": [None, None] + list(range(10)),
    })
    results = _build_correlation_matrix_from_frame(df)
    metadata = _build_correlation_metadata(df, results)
    entry = metadata["p_bal_gel"]["xrate"]
    assert entry["sample_size"] == 10
    assert entry["period_start"] == "2023-03-01"
    assert entry["period_end"] == "2023-12-01"
    assert entry["uncertainty"] == "high"
    assert entry["confidence_interval_95"] is None  # perfect correlation boundary
