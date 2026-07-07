"""Tests for the ontology-migration agreement shadow (design item 4, slice 1)."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.evidence_planner import build_evidence_plan
from contracts.question_analysis import QuestionAnalysis
from models import QueryContext
from utils.metrics import metrics

# NOTE: the planner's _SHARE_THRESHOLD_PATTERNS require a literal "%" —
# "10 percent" spelled out does not trigger the rule (pre-existing gap,
# candidate for the routing golden set).
_THRESHOLD_QUERY = (
    "which sources have a share above 10% in balancing electricity "
    "and at what price?"
)


def _qa_payload(tools: list[dict]) -> dict:
    return {
        "version": "question_analysis_v1",
        "raw_query": _THRESHOLD_QUERY,
        "canonical_query_en": _THRESHOLD_QUERY,
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "data_retrieval",
            "analysis_mode": "light",
            "intent": "threshold share with price context",
            "needs_clarification": False,
            "confidence": 0.9,
        },
        "routing": {
            "preferred_path": "tool",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": True,
            "needs_multi_tool": True,
            "evidence_roles": [],
        },
        "knowledge": {},
        "tooling": {"candidate_tools": tools},
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


def _ctx(tools: list[dict]) -> QueryContext:
    ctx = QueryContext(query=_THRESHOLD_QUERY)
    ctx.question_analysis = QuestionAnalysis(**_qa_payload(tools))
    ctx.question_analysis_source = "llm_active"
    return ctx


def _run_plan(tools: list[dict]):
    metrics.evidence_rule_agreement_events.clear()
    ctx = _ctx(tools)
    ctx = build_evidence_plan(ctx)
    return ctx


def test_disagree_counted_when_analyzer_omits_prices():
    ctx = _run_plan([
        {"name": "get_balancing_composition", "score": 0.9, "reason": "shares"},
    ])
    assert metrics.evidence_rule_agreement_events.get(
        "threshold_share_price_context:disagree") == 1
    # Behavior unchanged: the planner rule still adds the prices step.
    assert any(s["tool_name"] == "get_prices" for s in ctx.evidence_plan)


def test_agree_counted_when_analyzer_emits_prices_secondary():
    ctx = _run_plan([
        {"name": "get_balancing_composition", "score": 0.9, "reason": "shares"},
        {"name": "get_prices", "score": 0.7, "reason": "price context"},
    ])
    assert metrics.evidence_rule_agreement_events.get(
        "threshold_share_price_context:agree") == 1
    assert any(s["tool_name"] == "get_prices" for s in ctx.evidence_plan)


def test_no_counter_for_non_threshold_query():
    metrics.evidence_rule_agreement_events.clear()
    ctx = QueryContext(query="show monthly balancing prices for 2023")
    payload = _qa_payload([
        {"name": "get_prices", "score": 0.9, "reason": "price data"},
    ])
    payload["raw_query"] = ctx.query
    payload["canonical_query_en"] = ctx.query
    payload["routing"]["needs_multi_tool"] = False
    ctx.question_analysis = QuestionAnalysis(**payload)
    ctx.question_analysis_source = "llm_active"
    build_evidence_plan(ctx)
    assert metrics.evidence_rule_agreement_events == {}
