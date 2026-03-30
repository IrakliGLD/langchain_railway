"""
Tests for resolved-query propagation, entity normalization, and
context-preserving fallback (Phases 1–4 of the analyzer-coercion fix).
"""
import os
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import sqlalchemy
from langchain_core.messages import AIMessage

# Ensure config validation passes before importing project modules.
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

from models import QueryContext  # noqa: E402
from agent.planner import normalize_balancing_entities  # noqa: E402
from agent.tools.composition_tools import ALLOWED_BALANCING_ENTITIES  # noqa: E402
from agent import orchestrator  # noqa: E402
from agent.tool_adapter import ToolExecutionResult  # noqa: E402


# ---------------------------------------------------------------------------
# Phase 2: normalize_balancing_entities
# ---------------------------------------------------------------------------


def test_normalize_passthrough_valid_entity():
    """Valid entities pass through unchanged."""
    assert normalize_balancing_entities(["import"]) == ["import"]


def test_normalize_underscore_form():
    """Space-separated form is normalized to underscore."""
    result = normalize_balancing_entities(["deregulated hydro"])
    assert "deregulated_hydro" in result


def test_normalize_cheap_energy_expands_to_cheap_tier():
    """'cheap energy' expands to all cheap-tier entities."""
    result = normalize_balancing_entities(["cheap energy"])
    assert "regulated_hpp" in result
    assert "deregulated_hydro" in result
    # Should not include expensive entities
    assert "thermal_ppa" not in result
    assert "import" not in result


def test_normalize_expensive_expands_to_expensive_tier():
    """'expensive sources' expands to expensive-tier entities."""
    result = normalize_balancing_entities(["expensive sources"])
    assert "thermal_ppa" in result
    assert "import" in result
    assert "regulated_new_tpp" in result
    # Should not include cheap entities
    assert "regulated_hpp" not in result


def test_normalize_mixed_valid_and_semantic():
    """Mix of valid entity and semantic concept resolves correctly."""
    result = normalize_balancing_entities(["import", "cheap energy"])
    assert "import" in result
    assert "regulated_hpp" in result
    assert "deregulated_hydro" in result


def test_normalize_mixed_valid_and_unresolved_returns_none():
    """Any unresolved entity should fail closed instead of partially executing."""
    result = normalize_balancing_entities(["import", "completely_invalid_xyz"])
    assert result is None


def test_normalize_label_substring_match():
    """Label-based resolution works for partial matches."""
    result = normalize_balancing_entities(["hydro"])
    assert len(result) > 0
    # Should match at least deregulated_hydro via label "deregulated hydro"
    assert "deregulated_hydro" in result


def test_normalize_nonsense_returns_none():
    """Unresolvable entities signal unresolved_concept via None."""
    result = normalize_balancing_entities(["completely_invalid_xyz"])
    assert result is None


def test_normalize_empty_input():
    """Empty input returns empty list (tool default: all entities)."""
    assert normalize_balancing_entities([]) == []


def test_normalize_preserves_allowed_order():
    """Output order matches ALLOWED_BALANCING_ENTITIES order."""
    result = normalize_balancing_entities(["thermal_ppa", "import"])
    allowed_list = [e for e in ALLOWED_BALANCING_ENTITIES if e.lower() in {r.lower() for r in result}]
    assert result == allowed_list


# ---------------------------------------------------------------------------
# Phase 4: agent loop context enrichment
# ---------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    def invoke(self, messages):
        self._last_messages = messages
        if self._idx >= len(self._responses):
            return AIMessage(content="", tool_calls=[])
        res = self._responses[self._idx]
        self._idx += 1
        return res


def test_agent_loop_includes_resolved_query(monkeypatch):
    """When resolved_query differs from query, the agent sees the resolved form."""
    llm = FakeLLM([AIMessage(content="Answer based on context.", tool_calls=[])])

    ctx = QueryContext(query="why did it increase?")
    ctx.resolved_query = "Why did the balancing electricity price increase in Georgia?"

    out = orchestrator.run_agent_loop(ctx, llm=llm)

    user_msg = llm._last_messages[1].content
    assert "Resolved query:" in user_msg
    assert "balancing electricity price" in user_msg
    assert "why did it increase?" in user_msg


def test_agent_loop_includes_conversation_history_for_followup(monkeypatch):
    """When query is a follow-up (resolved_query differs), history is injected."""
    llm = FakeLLM([AIMessage(content="Follow-up answer.", tool_calls=[])])

    ctx = QueryContext(query="what about cheap energy?")
    ctx.resolved_query = "What is the share of cheap energy sources in balancing composition?"
    ctx.conversation_history = [
        {"question": "Show balancing price trend", "answer": "Prices increased 15% in 2024."},
    ]

    out = orchestrator.run_agent_loop(ctx, llm=llm)

    user_msg = llm._last_messages[1].content
    assert "Previous conversation:" in user_msg
    assert "balancing price trend" in user_msg
    assert "15%" in user_msg


def test_agent_loop_skips_history_for_standalone_query(monkeypatch):
    """When query is standalone (no resolved_query diff), history is NOT injected."""
    llm = FakeLLM([AIMessage(content="Direct answer.", tool_calls=[])])

    ctx = QueryContext(query="Show balancing price in 2024")
    ctx.conversation_history = [
        {"question": "Unrelated prior question", "answer": "Unrelated prior answer."},
    ]

    out = orchestrator.run_agent_loop(ctx, llm=llm)

    user_msg = llm._last_messages[1].content
    assert "Previous conversation:" not in user_msg
    assert "Unrelated prior" not in user_msg


def test_agent_loop_includes_tool_fallback_reason(monkeypatch):
    """When a tool failed before reaching agent loop, the reason is included."""
    llm = FakeLLM([AIMessage(content="Let me help.", tool_calls=[])])

    ctx = QueryContext(query="cheap energy composition")
    ctx.tool_fallback_reason = "analyzer_tool_execution_error:Unsupported balancing entity: cheap energy"

    out = orchestrator.run_agent_loop(ctx, llm=llm)

    user_msg = llm._last_messages[1].content
    assert "previous tool attempt failed" in user_msg.lower()
    assert "cheap energy" in user_msg


def test_agent_loop_plain_query_when_no_extra_context(monkeypatch):
    """First-turn query with no history or resolved_query stays unchanged."""
    llm = FakeLLM([AIMessage(content="Simple answer.", tool_calls=[])])

    ctx = QueryContext(query="Show balancing price in 2024")

    out = orchestrator.run_agent_loop(ctx, llm=llm)

    user_msg = llm._last_messages[1].content
    assert user_msg == "Show balancing price in 2024"


# ---------------------------------------------------------------------------
# Phase 1: resolved_query is set on ctx
# ---------------------------------------------------------------------------


def test_resolved_query_field_defaults_empty():
    """resolved_query defaults to empty string on fresh QueryContext."""
    ctx = QueryContext(query="test")
    assert ctx.resolved_query == ""
