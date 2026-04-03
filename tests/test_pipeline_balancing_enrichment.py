"""Focused tests for balancing driver enrichment in agent.pipeline."""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import pipeline
from agent.tools.types import ToolInvocation
from models import QueryContext


class _FakeConn:
    def execute(self, *_args, **_kwargs):
        return None


class _FakeEngine:
    class _Ctx:
        def __init__(self, conn):
            self._conn = conn

        def __enter__(self):
            return self._conn

        def __exit__(self, exc_type, exc, tb):
            return False

    def __init__(self):
        self.conn = _FakeConn()

    def connect(self):
        return self._Ctx(self.conn)


def _seed_ctx(query: str) -> QueryContext:
    ctx = QueryContext(query=query)
    ctx.df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01"],
            "p_bal_gel": [150.0, 180.0],
        },
    )
    ctx.cols = list(ctx.df.columns)
    ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False, name=None)]
    return ctx


def test_balancing_driver_enrichment_adds_source_price_columns(monkeypatch):
    monkeypatch.setattr(pipeline, "ENGINE", _FakeEngine())
    monkeypatch.setattr(
        pipeline,
        "compute_entity_price_contributions",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-02-01"],
                "price_deregulated_hydro_gel": [45.0, 48.0],
                "price_regulated_hpp_gel": [32.0, 33.0],
                "contribution_regulated_hpp_gel": [6.4, 5.9],
                "residual_contribution_ppa_import_gel": [101.0, 126.0],
            },
        ),
    )

    ctx = _seed_ctx("Compare balancing electricity prices in January and February 2024")
    invocation = ToolInvocation(
        name="get_prices",
        params={
            "metric": "balancing",
            "currency": "gel",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
        },
    )

    out = pipeline._enrich_prices_with_balancing_driver_context(
        ctx,
        invocation,
        is_explanation=False,
    )

    assert "price_deregulated_hydro_gel" in out.cols
    assert "price_regulated_hpp_gel" in out.cols
    assert "contribution_regulated_hpp_gel" in out.cols
    assert "residual_contribution_ppa_import_gel" in out.cols
    assert "balancing_driver_context" in out.evidence_collected
    assert out.evidence_collected["balancing_driver_context"]["params"]["currency"] == "gel"
    assert len(out.join_provenance) == 1
    assert out.join_provenance[0]["secondary_tool"] == "compute_entity_price_contributions"
    assert out.join_provenance[0]["role"] == "balancing_driver_context"
    assert "price_deregulated_hydro_gel" in out.join_provenance[0]["columns_added"]


def test_balancing_driver_enrichment_falls_back_to_composition_for_comparison_queries(monkeypatch):
    monkeypatch.setattr(pipeline, "ENGINE", _FakeEngine())
    monkeypatch.setattr(
        pipeline,
        "compute_entity_price_contributions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("db failure")),
    )

    def _fake_execute_tool(invocation: ToolInvocation):
        assert invocation.name == "get_balancing_composition"
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-02-01"],
                "share_import": [0.20, 0.25],
                "share_regulated_hpp": [0.30, 0.22],
            },
        )
        cols = list(df.columns)
        rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
        return df, cols, rows

    monkeypatch.setattr(pipeline, "execute_tool", _fake_execute_tool)

    ctx = _seed_ctx("Compare balancing electricity prices in January and February 2024")
    invocation = ToolInvocation(
        name="get_prices",
        params={
            "metric": "balancing",
            "currency": "gel",
        },
    )

    out = pipeline._enrich_prices_with_balancing_driver_context(
        ctx,
        invocation,
        is_explanation=False,
    )

    assert "share_import" in out.cols
    assert "share_regulated_hpp" in out.cols
