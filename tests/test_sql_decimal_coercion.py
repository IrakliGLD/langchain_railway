"""Regression tests for PostgreSQL numeric -> Decimal dtype on the SQL path.

PostgreSQL ``numeric`` columns arrive in pandas as ``Decimal`` objects with
``object`` dtype.  Every downstream consumer that uses
``select_dtypes(include="number")`` then silently skips them, leaving
``stats_hint`` at just "Rows: N" and the grounding corpus without aggregates.

Production trace 2026-08-15 (``demand_tariff_mv``): ``stats_hint_len=9``, which
is exactly ``len("Rows: 528")``.  With no statistics the model computed its own
figures, and the strict_numeric grounding gate correctly rejected them --
shipping 273 of 1,587 answer characters.

``analysis/system_quantities.coerce_decimal_columns_to_float`` already exists
and documents this whole chain; it was applied to driver-enrichment frames in
``agent/pipeline.py`` and never to the main SQL result path.
"""
import os

# config.py validates its settings at import time, and core.query_executor
# imports it. Match the preamble used by tests/test_plan_validation.py so this
# module is import-order independent.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from decimal import Decimal  # noqa: E402

import pandas as pd  # noqa: E402


def make_demand_tariff_frame(rows: int = 6) -> pd.DataFrame:
    """A demand_tariff_mv-shaped frame with Decimal values, as psycopg returns."""
    return pd.DataFrame(
        {
            "date": [f"2026-0{(i % 6) + 1}-01" for i in range(rows)],
            "company": ["telmico"] * rows,
            "activity": ["universal"] * rows,
            "level_1_cat": ["hh"] * rows,
            "level_2_cat": ["cat2"] * rows,
            "value": [Decimal("0.15289") + Decimal(i) / 1000 for i in range(rows)],
        }
    )


def test_decimal_value_column_is_numeric_after_sql_execution():
    """The SQL path must hand downstream consumers a float column, not object.

    Fails before the fix: ``value`` is object dtype, so select_dtypes finds
    nothing and every aggregate consumer silently no-ops.
    """
    from core.query_executor import coerce_result_frame

    df = coerce_result_frame(make_demand_tariff_frame())

    assert "value" in df.select_dtypes(include="number").columns, (
        "Decimal column still invisible to select_dtypes(include='number')"
    )


def test_raw_decimal_frame_is_invisible_to_numeric_selection():
    """Pins the defect itself, so the tests above cannot silently stop testing it.

    If pandas ever starts treating Decimal as numeric, this fails and the
    coercion can be reconsidered -- rather than the regression tests quietly
    passing for a different reason.
    """
    raw = make_demand_tariff_frame()

    assert raw["value"].dtype == object
    assert raw.select_dtypes(include="number").columns.tolist() == []


def test_column_aggregates_reach_stats_hint_only_after_coercion():
    """stats_hint must carry aggregates, not just the row count.

    Production symptom: stats_hint_len=9, which is exactly len("Rows: 528").
    With no statistics the model computes its own figures and the
    strict_numeric grounding gate then rejects every one of them.
    """
    from agent.analyzer import _append_column_aggregates
    from core.query_executor import coerce_result_frame
    from models import QueryContext

    raw_ctx = QueryContext(query="household tariff dynamics")
    raw_ctx.df = make_demand_tariff_frame(rows=12)
    raw_ctx.stats_hint = "Rows: 12"
    _append_column_aggregates(raw_ctx)
    assert raw_ctx.stats_hint == "Rows: 12", (
        "expected the raw Decimal frame to contribute no aggregates"
    )

    coerced_ctx = QueryContext(query="household tariff dynamics")
    coerced_ctx.df = coerce_result_frame(make_demand_tariff_frame(rows=12))
    coerced_ctx.stats_hint = "Rows: 12"
    _append_column_aggregates(coerced_ctx)

    assert "Column Aggregates" in coerced_ctx.stats_hint
    assert len(coerced_ctx.stats_hint) > len("Rows: 12")


def test_grounding_aggregate_tokens_appear_only_after_coercion():
    """agent/summary_grounding.py has the same select_dtypes dependency.

    Without numeric dtypes it contributes no aggregate tokens, so an answer
    citing a mean or a max cannot match the corpus and the gate strips it.
    """
    from agent.summary_grounding import _add_aggregate_tokens
    from core.query_executor import coerce_result_frame
    from models import QueryContext

    raw_ctx = QueryContext(query="household tariff dynamics")
    raw_ctx.df = make_demand_tariff_frame(rows=12)
    raw_tokens: set[str] = set()
    _add_aggregate_tokens(raw_tokens, raw_ctx)
    assert raw_tokens == set(), "expected no aggregate tokens from a raw Decimal frame"

    coerced_ctx = QueryContext(query="household tariff dynamics")
    coerced_ctx.df = coerce_result_frame(make_demand_tariff_frame(rows=12))
    coerced_tokens: set[str] = set()
    _add_aggregate_tokens(coerced_tokens, coerced_ctx)

    assert coerced_tokens, "no aggregate tokens produced for a coerced frame"
