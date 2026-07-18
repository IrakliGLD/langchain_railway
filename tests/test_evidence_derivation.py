"""Characterization tests for Stage 3 evidence derivation boundaries."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd

from agent import evidence_derivation
from agent.provenance import sql_query_hash, tool_invocation_hash
from models import QueryContext


def test_derive_evidence_stamps_new_analyzer_columns_with_source_provenance(monkeypatch):
    ctx = QueryContext(query="derive")
    ctx.df = pd.DataFrame({"date": ["2024-01-01"], "value": [10.0]})
    ctx.cols = ["date", "value"]
    ctx.rows = [("2024-01-01", 10.0)]
    ctx.used_tool = True
    ctx.tool_name = "get_prices"
    ctx.tool_params = {"currency": "gel"}

    def enrich(current):
        current.df["derived"] = [12.0]
        current.cols = list(current.df.columns)
        current.rows = [tuple(row) for row in current.df.itertuples(index=False, name=None)]
        return current

    monkeypatch.setattr(evidence_derivation.analyzer, "enrich", enrich)

    result = evidence_derivation.derive_evidence(ctx)

    expected_hash = tool_invocation_hash(ctx.tool_name, ctx.tool_params)
    assert result is ctx
    assert ctx.provenance_source == "tool"
    assert ctx.provenance_query_hash == expected_hash
    assert ctx.provenance_cols == ["date", "value", "derived"]
    assert ctx.provenance_rows == [("2024-01-01", 10.0, 12.0)]


def test_requested_and_missing_metric_names_are_stable_and_deduplicated():
    ctx = SimpleNamespace(
        has_authoritative_question_analysis=True,
        question_analysis=SimpleNamespace(
            analysis_requirements=SimpleNamespace(
                derived_metrics=[
                    SimpleNamespace(metric_name=SimpleNamespace(value="mom_percent_change")),
                    SimpleNamespace(metric_name=SimpleNamespace(value="mom_percent_change")),
                    SimpleNamespace(metric_name=SimpleNamespace(value="yoy_percent_change")),
                ]
            )
        ),
        requested_derived_metrics=["mom_percent_change", "yoy_percent_change"],
        analysis_evidence=[
            {"derived_metric_name": "mom_percent_change"},
            {"derived_metric_name": ""},
        ],
    )

    assert evidence_derivation.requested_derived_metric_names(ctx) == [
        "mom_percent_change",
        "yoy_percent_change",
    ]
    assert evidence_derivation.missing_requested_evidence(ctx) == [
        "yoy_percent_change"
    ]


def test_metric_helpers_return_empty_without_authoritative_or_requested_metrics():
    ctx = SimpleNamespace(
        has_authoritative_question_analysis=False,
        requested_derived_metrics=[],
    )

    assert evidence_derivation.requested_derived_metric_names(ctx) == []
    assert evidence_derivation.missing_requested_evidence(ctx) == []


def test_derive_evidence_does_not_restamp_when_analyzer_adds_no_columns(monkeypatch):
    ctx = QueryContext(query="unchanged")
    ctx.cols = ["value"]
    ctx.rows = [(10.0,)]
    ctx.provenance_cols = ["value"]
    ctx.provenance_rows = [(10.0,)]
    monkeypatch.setattr(evidence_derivation.analyzer, "enrich", lambda current: current)

    result = evidence_derivation.derive_evidence(ctx)

    assert result is ctx
    assert ctx.provenance_cols == ["value"]
    assert ctx.provenance_rows == [(10.0,)]


def test_derive_evidence_uses_sql_identity_for_sql_fallback(monkeypatch):
    ctx = QueryContext(query="derive from SQL")
    ctx.cols = ["date", "derived"]
    ctx.rows = [("2024-01-01", 12.0)]
    ctx.safe_sql = "SELECT date, derived FROM safe_view"
    monkeypatch.setattr(evidence_derivation.analyzer, "enrich", lambda current: current)

    evidence_derivation.derive_evidence(ctx)

    assert ctx.provenance_source == "sql"
    assert ctx.provenance_query_hash == sql_query_hash(ctx.safe_sql)
