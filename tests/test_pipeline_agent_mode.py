"""
Integration-style tests for the pipeline's generate-plan/SQL fallback path and stage timings.
"""
import os
from typing import Any

import pandas as pd
import sqlalchemy

# Ensure config validation passes and avoid real DB engine creation at import time.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


class _DummyResult:
    def fetchall(self):
        return []

    def keys(self):
        return []


class _DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args: Any, **kwargs: Any):
        return _DummyResult()


class _DummyEngine:
    def connect(self):
        return _DummyConnection()


sqlalchemy.create_engine = lambda *args, **kwargs: _DummyEngine()  # type: ignore[assignment]

from agent import pipeline  # noqa: E402


def _minimal_df():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")],
        "p_bal_usd": [10.0],
    })
    return df, list(df.columns), [tuple(r) for r in df.itertuples(index=False, name=None)]


def test_pipeline_records_stage_timings(monkeypatch):
    monkeypatch.setattr(pipeline, "ENABLE_TYPED_TOOLS", False)
    monkeypatch.setattr(pipeline.planner, "prepare_context", lambda ctx: ctx)

    def _planner(ctx):
        ctx.is_conceptual = False
        ctx.skip_sql = False
        return ctx

    def _sql(ctx):
        df, cols, rows = _minimal_df()
        ctx.df = df
        ctx.cols = cols
        ctx.rows = rows
        return ctx

    monkeypatch.setattr(pipeline.planner, "generate_plan", _planner)
    monkeypatch.setattr(pipeline.sql_executor, "validate_and_execute", _sql)
    monkeypatch.setattr(pipeline.analyzer, "enrich", lambda ctx: ctx)
    monkeypatch.setattr(pipeline.summarizer, "summarize_data", lambda ctx: setattr(ctx, "summary", "ok") or ctx)
    monkeypatch.setattr(pipeline.chart_pipeline, "build_chart", lambda ctx: setattr(ctx, "chart_type", "line") or ctx)

    out = pipeline.process_query("Show balancing price trend", trace_id="trace-x", session_id="session-y")
    assert out.stage_timings_ms["stage_0_prepare_context"] >= 0.0
    assert out.stage_timings_ms["stage_1_generate_plan"] >= 0.0
    assert out.stage_timings_ms["stage_2_sql_execute"] >= 0.0
    assert out.stage_timings_ms["stage_4_summarize_data"] >= 0.0
    assert out.stage_timings_ms["stage_5_chart_build"] >= 0.0
