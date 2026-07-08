"""Tests for agent.render_fitness (design item 3 — shadow fitness checks)."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd

from agent.render_fitness import evaluate_render_fitness, period_bounds_from_hint
from contracts.question_analysis import QuestionAnalysis
from models import QueryContext


def _qa_with_hint(start_date=None, end_date=None, entities=None) -> QuestionAnalysis:
    """Minimal valid QuestionAnalysis whose top tool carries a params hint.

    Payload shape mirrors tests/test_evidence_planner.py::_make_qa_payload
    (kept local per house test style — no cross-test-file imports).
    """
    params_hint = {}
    if start_date:
        params_hint["start_date"] = start_date
    if end_date:
        params_hint["end_date"] = end_date
    if entities:
        params_hint["entities"] = entities
    tool = {"name": "get_prices", "score": 0.9, "reason": "price data"}
    if params_hint:
        tool["params_hint"] = params_hint
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "test query",
        "canonical_query_en": "test query in English",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "data_retrieval",
            "analysis_mode": "light",
            "intent": "test intent",
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
    }
    return QuestionAnalysis(**payload)


def _ctx_with(df: pd.DataFrame, qa: QuestionAnalysis | None) -> QueryContext:
    ctx = QueryContext(query="test")
    ctx.df = df
    ctx.rows = [tuple(r) for r in df.itertuples(index=False)]
    ctx.cols = list(df.columns)
    ctx.question_analysis = qa
    if qa is not None:
        ctx.question_analysis_source = "llm_active"
    ctx.summary_source = "generic_renderer"
    return ctx


def test_empty_result_rendered_flagged():
    ctx = _ctx_with(pd.DataFrame({"date": [], "p_bal_gel": []}), qa=None)
    assert evaluate_render_fitness(ctx) == ["empty_result_rendered"]


def test_period_coverage_gap_flagged_when_disjoint():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-06-01"], "p_bal_gel": [1.0, 2.0]})
    qa = _qa_with_hint(start_date="2024-01-01", end_date="2024-12-31")
    assert "period_coverage_gap" in evaluate_render_fitness(_ctx_with(df, qa))


def test_overlapping_period_not_flagged():
    df = pd.DataFrame({"date": ["2024-03-01", "2024-06-01"], "p_bal_gel": [1.0, 2.0]})
    qa = _qa_with_hint(start_date="2024-01-01", end_date="2024-12-31")
    assert "period_coverage_gap" not in evaluate_render_fitness(_ctx_with(df, qa))


def test_requested_entities_missing_flagged():
    df = pd.DataFrame({"date": ["2024-03-01"], "p_bal_gel": [1.0]})
    qa = _qa_with_hint(start_date="2024-01-01", end_date="2024-12-31",
                       entities=["enguri"])
    assert "requested_entities_missing" in evaluate_render_fitness(_ctx_with(df, qa))


def test_present_entity_not_flagged():
    df = pd.DataFrame({"date": ["2024-03-01"], "entity": ["Enguri"], "p_bal_gel": [1.0]})
    qa = _qa_with_hint(start_date="2024-01-01", end_date="2024-12-31",
                       entities=["enguri"])
    assert "requested_entities_missing" not in evaluate_render_fitness(_ctx_with(df, qa))


def test_no_hint_no_period_check():
    df = pd.DataFrame({"date": ["2020-01-01"], "p_bal_gel": [1.0]})
    assert evaluate_render_fitness(_ctx_with(df, None)) == []


def test_period_bounds_from_hint_none_paths():
    assert period_bounds_from_hint(None) is None
    assert period_bounds_from_hint(_qa_with_hint()) is None
    assert period_bounds_from_hint(
        _qa_with_hint(start_date="2024-01-01", end_date="2024-06-30")
    ) == ("2024-01-01", "2024-06-30")


def test_tolerates_partial_question_analysis():
    """Fitness is shadow observability — duck-typed fakes must not crash it."""
    from types import SimpleNamespace

    df = pd.DataFrame({"date": ["2024-03-01"], "p_bal_gel": [1.0]})
    ctx = QueryContext(query="test")
    ctx.df = df
    ctx.rows = [tuple(r) for r in df.itertuples(index=False)]
    ctx.cols = list(df.columns)
    ctx.question_analysis = SimpleNamespace()  # no .tooling at all
    ctx.question_analysis_source = "llm_active"
    assert evaluate_render_fitness(ctx) == []


def test_summarizer_wiring_logs_violations(monkeypatch):
    """summarize_data must emit counters for deterministic-render violations."""
    from agent import summarizer
    from utils.metrics import metrics

    monkeypatch.setattr(summarizer, "_try_generic_renderer", lambda ctx: "answer text")
    monkeypatch.setattr(
        summarizer, "evaluate_render_fitness", lambda ctx: ["empty_result_rendered"]
    )
    metrics.render_fitness_events.clear()

    ctx = QueryContext(query="test")
    ctx = summarizer.summarize_data(ctx)

    assert ctx.summary_source == "generic_renderer"
    assert metrics.render_fitness_events.get("empty_result_rendered") == 1
