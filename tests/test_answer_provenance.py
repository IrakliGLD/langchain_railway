"""Tests for agent.answer_provenance (design item 5 — answer-provenance surface)."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.answer_provenance import build_answer_provenance
from models import QueryContext


def _ctx(**overrides):
    ctx = QueryContext(query="test")
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def test_deterministic_render_path():
    ctx = _ctx(summary_source="generic_renderer", used_tool=True,
               tool_name="get_prices", provenance_source="tool",
               vector_retrieval_tier="skip")
    block = build_answer_provenance(ctx)
    assert block["answer_path"] == "deterministic_render"
    assert block["tool_name"] == "get_prices"
    assert block["used_sql"] is False
    assert block["retrieval_tier"] == "skip"


def test_specialized_formatter_path():
    ctx = _ctx(summary_source="deterministic_share_summary")
    assert build_answer_provenance(ctx)["answer_path"] == "specialized_formatter"


def test_narrative_and_gate_fields():
    ctx = _ctx(summary_source="structured_summary_grounding_fallback",
               summary_provenance_gate_passed=False,
               summary_provenance_gate_reason="coverage_below_threshold",
               summary_provenance_coverage=0.42)
    block = build_answer_provenance(ctx)
    assert block["answer_path"] == "narrative_llm"
    assert block["grounding_gate"] == {
        "passed": False, "reason": "coverage_below_threshold", "coverage": 0.42,
    }


def test_sql_fallback_flagged():
    ctx = _ctx(summary_source="structured_summary", used_tool=False,
               safe_sql="SELECT 1", tool_name="")
    block = build_answer_provenance(ctx)
    assert block["used_sql"] is True
    assert block["tool_name"] == ""


def test_clarify_and_unknown_paths():
    assert build_answer_provenance(_ctx(summary_source="clarification_request"))["answer_path"] == "clarify"
    assert build_answer_provenance(_ctx())["answer_path"] == "unknown"


def test_analyzer_subblock_without_analysis():
    block = build_answer_provenance(_ctx())
    assert block["analyzer"] == {
        "authoritative": False, "query_type": "", "answer_kind": "", "confidence": 0.0,
    }


def test_never_crashes_on_partial_context():
    """The block must never be able to fail a response — /ask tests use
    duck-typed SimpleNamespace stand-ins for QueryContext."""
    from types import SimpleNamespace

    block = build_answer_provenance(SimpleNamespace())
    assert block["answer_path"] == "unknown"
    assert block["analyzer"]["authoritative"] is False
    assert block["grounding_gate"]["passed"] is False
