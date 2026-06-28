"""Regression tests for PostgreSQL Decimal→float64 coercion in tool dataframes.

Root-cause regression (2026-05-17, Q2 trace 5b117786): PostgreSQL ``numeric``
columns arrive in pandas as Python ``Decimal`` objects with ``object`` dtype.
``pd.api.types.is_numeric_dtype`` returns False for object dtype, so
downstream code that depends on ``select_dtypes(include="number")``
silently skips every column:

  - ``agent/analyzer.py::_append_column_aggregates`` returns immediately
    when ``numeric_cols == []``, so no per-column sum/mean/min/max enters
    ``stats_hint``.
  - ``analysis/stats.py::quick_stats`` short-circuits at line 108
    (``if not numeric_cols: return "\\n".join(out)``), returning only
    "Rows: N" — exactly 8 chars for a 12-row month-window.

Consequence: the LLM receives no pre-computed statistics for an analyst-
mode "compare averages" query and must compute aggregates from 12 raw
rows. It hallucinates plausible-but-wrong numbers; the grounding gate
correctly rejects them; the conservative fallback replaces the answer.

These tests anchor the fix at the source of the problem: every tool
result must come out of ``normalize_tool_dataframe`` with proper float64
numeric dtypes.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


def test_coerce_decimal_columns_to_float_promotes_object_columns():
    """Decimal-typed object columns must be promoted to float64."""
    from analysis.system_quantities import coerce_decimal_columns_to_float

    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "price_gel": [Decimal("100.5"), Decimal("105.3"), Decimal("110.1")],
        "share_pct": [Decimal("0.42"), Decimal("0.44"), Decimal("0.46")],
        "label": ["a", "b", "c"],  # non-Decimal object column, must stay object
    })
    assert df["price_gel"].dtype == object
    assert df["share_pct"].dtype == object

    coerced, converted = coerce_decimal_columns_to_float(df)

    # Numeric columns now float64 and visible to select_dtypes
    assert coerced["price_gel"].dtype == "float64"
    assert coerced["share_pct"].dtype == "float64"
    assert "price_gel" in converted
    assert "share_pct" in converted
    # Non-Decimal object column untouched
    assert coerced["label"].dtype == object
    assert "label" not in converted
    # Date column untouched
    assert pd.api.types.is_datetime64_any_dtype(coerced["date"])

    # The fix is observable via select_dtypes — the critical signal the
    # downstream pipeline relies on.
    numeric_cols = coerced.select_dtypes(include="number").columns.tolist()
    assert "price_gel" in numeric_cols
    assert "share_pct" in numeric_cols


def test_coerce_decimal_handles_empty_and_none_safely():
    """Empty df and None must short-circuit cleanly."""
    from analysis.system_quantities import coerce_decimal_columns_to_float

    out_empty, converted_empty = coerce_decimal_columns_to_float(pd.DataFrame())
    assert out_empty.empty
    assert converted_empty == []

    out_none, converted_none = coerce_decimal_columns_to_float(None)
    assert out_none is None
    assert converted_none == []


def test_coerce_decimal_preserves_mixed_object_columns():
    """A column with mixed Decimal/string content (rare but possible) must
    not be coerced — leaving it as object is safer than producing NaN."""
    from analysis.system_quantities import coerce_decimal_columns_to_float

    df = pd.DataFrame({
        "mixed_field": ["category_label", Decimal("100.5"), "another_label"],
    })
    # First non-null is a string, not Decimal → column is NOT coerced
    coerced, converted = coerce_decimal_columns_to_float(df)
    assert coerced["mixed_field"].dtype == object
    assert "mixed_field" not in converted


def test_normalize_tool_dataframe_coerces_decimal_columns():
    """End-to-end: normalize_tool_dataframe must coerce Decimal columns for
    every tool, not just the ones with explicit normalization branches.
    """
    from analysis.system_quantities import normalize_tool_dataframe

    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-{:02d}-01".format(m) for m in range(1, 4)]),
        "p_bal_gel": [Decimal("95"), Decimal("100"), Decimal("110")],
        "p_bal_usd": [Decimal("35"), Decimal("37"), Decimal("41")],
        "xrate": [Decimal("2.71")] * 3,
    })
    # Before: all numeric columns are object dtype
    assert df.select_dtypes(include="number").columns.tolist() == []

    normalized = normalize_tool_dataframe("get_prices", df)

    # After: all numeric columns visible to select_dtypes
    numeric = normalized.select_dtypes(include="number").columns.tolist()
    assert "p_bal_gel" in numeric
    assert "p_bal_usd" in numeric
    assert "xrate" in numeric


def test_quick_stats_produces_full_summary_after_decimal_coercion():
    """Integration: with the fix in place, ``quick_stats`` on a 12-row
    Decimal-typed monthly frame must produce a full numeric summary
    (not just "Rows: 12"). Anchors the production regression directly.
    """
    from analysis.stats import quick_stats
    from analysis.system_quantities import normalize_tool_dataframe

    # Mirror the failing trace's data shape: 12 monthly rows, Decimal cols.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-{:02d}-01".format(m) for m in range(1, 13)]),
        "p_bal_gel": [Decimal(str(95 + m)) for m in range(12)],
        "price_regulated_hpp_gel": [Decimal(str(15.0 + m * 0.1)) for m in range(12)],
        "price_regulated_new_tpp_gel": [Decimal(str(180 + m)) for m in range(12)],
        "price_regulated_old_tpp_gel": [Decimal(str(200 + m)) for m in range(12)],
    })

    # Before coercion: stats are 8 chars ("Rows: 12") — the production bug.
    rows_before = [tuple(r) for r in df.itertuples(index=False, name=None)]
    cols = list(df.columns)
    stats_before = quick_stats(rows_before, cols)
    assert len(stats_before) == 8
    assert stats_before == "Rows: 12"

    # After coercion: real numeric summary (hundreds of chars).
    df_after = normalize_tool_dataframe("get_prices", df)
    rows_after = [tuple(r) for r in df_after.itertuples(index=False, name=None)]
    stats_after = quick_stats(rows_after, cols)
    assert len(stats_after) > 100, (
        f"Expected full numeric summary; got {len(stats_after)} chars: {stats_after!r}"
    )
    # The numeric summary section must be present.
    assert "Numeric summary" in stats_after
    # Per-column means must appear (mean of [180..191] = 185.5).
    assert "185.5" in stats_after  # mean of price_regulated_new_tpp_gel


def test_append_column_aggregates_produces_output_after_decimal_coercion():
    """Integration: ``_append_column_aggregates`` must populate stats_hint
    with per-column sum/mean/min/max once Decimal columns are coerced.
    """
    from agent.analyzer import _append_column_aggregates
    from analysis.system_quantities import normalize_tool_dataframe
    from models import QueryContext

    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-{:02d}-01".format(m) for m in range(1, 13)]),
        "p_bal_gel": [Decimal(str(95 + m)) for m in range(12)],
    })

    # Without coercion: select_dtypes(include="number") is empty,
    # _append_column_aggregates returns without appending.
    ctx_uncoerced = QueryContext(query="test", df=df, cols=list(df.columns))
    _append_column_aggregates(ctx_uncoerced)
    assert ctx_uncoerced.stats_hint == "", (
        "Pre-fix: object-dtype Decimals were invisible to select_dtypes "
        "so aggregates silently produced nothing."
    )

    # With coercion (normalize_tool_dataframe is the entry point):
    df_coerced = normalize_tool_dataframe("get_prices", df)
    ctx_coerced = QueryContext(query="test", df=df_coerced, cols=list(df_coerced.columns))
    _append_column_aggregates(ctx_coerced)
    # ``_append_column_aggregates`` applies COLUMN_LABELS for human-readable
    # labels; "p_bal_gel" becomes "Balancing electricity price (GEL/MWh)".
    assert "Balancing electricity price" in ctx_coerced.stats_hint
    # p_bal_gel is an intensive (per-MWh) price: aggregates expose mean/min/max
    # but NOT sum — a summed price is meaningless (see is_intensive_metric).
    assert "sum=" not in ctx_coerced.stats_hint
    assert "mean=100.5000" in ctx_coerced.stats_hint
    assert "min=95.0000" in ctx_coerced.stats_hint
    assert "max=106.0000" in ctx_coerced.stats_hint
