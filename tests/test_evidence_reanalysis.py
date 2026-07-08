"""Tests for evidence-anomaly detection + flag-gated re-analysis (design item 1)."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd

from agent import pipeline
from contracts.question_analysis import QuestionAnalysis
from models import QueryContext
from utils.metrics import metrics


def _qa_payload(answer_kind: str = "timeseries", start_date=None, end_date=None) -> dict:
    params_hint = {}
    if start_date:
        params_hint["start_date"] = start_date
    if end_date:
        params_hint["end_date"] = end_date
    tool = {"name": "get_prices", "score": 0.9, "reason": "price data"}
    if params_hint:
        tool["params_hint"] = params_hint
    return {
        "version": "question_analysis_v1",
        "raw_query": "monthly balancing prices for 2024",
        "canonical_query_en": "monthly balancing prices for 2024",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "data_retrieval",
            "analysis_mode": "light",
            "intent": "price series",
            "needs_clarification": False,
            "confidence": 0.9,
        },
        "routing": {
            "preferred_path": "tool",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": True,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {},
        "tooling": {"candidate_tools": [tool]},
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
        "answer_kind": answer_kind,
    }


def _ctx(answer_kind: str = "timeseries", *, used_tool: bool, rows: list,
         df: pd.DataFrame | None = None, start_date=None, end_date=None) -> QueryContext:
    ctx = QueryContext(query="monthly balancing prices for 2024")
    ctx.question_analysis = QuestionAnalysis(
        **_qa_payload(answer_kind, start_date=start_date, end_date=end_date)
    )
    ctx.question_analysis_source = "llm_active"
    ctx.used_tool = used_tool
    ctx.tool_name = "get_prices" if used_tool else None
    ctx.rows = rows
    ctx.df = df if df is not None else pd.DataFrame()
    return ctx


# --- detector ----------------------------------------------------------------

def test_primary_empty_detected():
    ctx = _ctx(used_tool=True, rows=[])
    assert pipeline._detect_evidence_anomaly(ctx) == "primary_empty"


def test_period_gap_detected_when_disjoint():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-06-01"], "p_bal_gel": [1.0, 2.0]})
    ctx = _ctx(used_tool=True, rows=[(1,), (2,)], df=df,
               start_date="2024-01-01", end_date="2024-12-31")
    assert pipeline._detect_evidence_anomaly(ctx) == "period_gap"


def test_no_anomaly_for_overlapping_period():
    df = pd.DataFrame({"date": ["2024-03-01"], "p_bal_gel": [1.0]})
    ctx = _ctx(used_tool=True, rows=[(1,)], df=df,
               start_date="2024-01-01", end_date="2024-12-31")
    assert pipeline._detect_evidence_anomaly(ctx) is None


def test_no_anomaly_without_tool_or_for_knowledge_shapes():
    assert pipeline._detect_evidence_anomaly(_ctx(used_tool=False, rows=[])) is None
    assert pipeline._detect_evidence_anomaly(
        _ctx("knowledge", used_tool=True, rows=[])
    ) is None
    assert pipeline._detect_evidence_anomaly(QueryContext(query="q")) is None


# --- gated re-analysis ---------------------------------------------------------

def _wire_reanalysis_mocks(monkeypatch, calls: dict):
    def _fake_analyze(ctx):
        calls["analyze"] = calls.get("analyze", 0) + 1
        calls["anomaly_note"] = ctx.evidence_anomaly
        return ctx

    def _fake_build_plan(ctx):
        calls["plan"] = calls.get("plan", 0) + 1
        return ctx

    def _fake_execute(ctx):
        calls["execute"] = calls.get("execute", 0) + 1
        return ctx

    monkeypatch.setattr(pipeline.planner, "analyze_question_active", _fake_analyze)
    monkeypatch.setattr(pipeline.evidence_planner, "build_evidence_plan", _fake_build_plan)
    monkeypatch.setattr(pipeline, "_execute_evidence_plan", _fake_execute)
    monkeypatch.setattr(pipeline, "_finalize_answer_kind", lambda ctx: None)


def test_reanalysis_resets_state_and_runs_once(monkeypatch):
    calls: dict = {}
    _wire_reanalysis_mocks(monkeypatch, calls)

    ctx = _ctx(used_tool=True, rows=[])
    ctx.evidence_collected = {"primary_data": {"tool": "get_prices"}}
    ctx.evidence_plan = [{"role": "primary_data", "tool_name": "get_prices",
                          "params": {}, "satisfied": True}]

    out = pipeline._attempt_evidence_reanalysis(ctx, "primary_empty")

    assert out.reanalysis_attempted is True
    assert "primary_empty" in calls["anomaly_note"]
    assert calls == {"analyze": 1, "plan": 1, "execute": 1,
                     "anomaly_note": calls["anomaly_note"]}
    assert out.used_tool is False
    assert out.evidence_collected == {}
    assert out.evidence_plan == []
    # Trace emitted with the change flags.
    assert "stage_0_9_reanalysis" in out.stage_timings_ms


def test_hook_counts_but_never_retries_when_flag_off(monkeypatch):
    metrics.evidence_anomaly_events.clear()
    monkeypatch.setattr(pipeline, "ENABLE_EVIDENCE_REANALYSIS", False)
    retried = {"n": 0}
    monkeypatch.setattr(
        pipeline, "_attempt_evidence_reanalysis",
        lambda ctx, anomaly: retried.__setitem__("n", retried["n"] + 1) or ctx,
    )

    ctx = _ctx(used_tool=True, rows=[])
    anomaly = pipeline._detect_evidence_anomaly(ctx)
    assert anomaly == "primary_empty"
    # Mirror the hook body:
    metrics.log_evidence_anomaly(anomaly)
    if pipeline.ENABLE_EVIDENCE_REANALYSIS and not ctx.reanalysis_attempted:
        pipeline._attempt_evidence_reanalysis(ctx, anomaly)

    assert metrics.evidence_anomaly_events.get("primary_empty") == 1
    assert retried["n"] == 0


def test_no_second_attempt_after_retry(monkeypatch):
    calls: dict = {}
    _wire_reanalysis_mocks(monkeypatch, calls)
    monkeypatch.setattr(pipeline, "ENABLE_EVIDENCE_REANALYSIS", True)

    ctx = _ctx(used_tool=True, rows=[])
    ctx = pipeline._attempt_evidence_reanalysis(ctx, "primary_empty")
    assert ctx.reanalysis_attempted is True

    # The hook's guard must prevent a second attempt even if an anomaly persists.
    if pipeline.ENABLE_EVIDENCE_REANALYSIS and not ctx.reanalysis_attempted:
        pipeline._attempt_evidence_reanalysis(ctx, "primary_empty")
    assert calls["analyze"] == 1
