"""
Tests for Phase 4 bounded agent loop orchestration.
"""
import os

import pandas as pd
from langchain_core.messages import AIMessage

# Ensure config validation passes before importing project modules.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("APP_SECRET_KEY", "test-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from models import QueryContext  # noqa: E402
from agent import orchestrator  # noqa: E402
from agent.provenance import tool_invocation_hash  # noqa: E402
from agent.tool_adapter import ToolExecutionResult  # noqa: E402
from utils.metrics import Metrics  # noqa: E402
from core import llm as llm_module  # noqa: E402


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    def invoke(self, _messages):
        if self._idx >= len(self._responses):
            return AIMessage(content="", tool_calls=[])
        res = self._responses[self._idx]
        self._idx += 1
        return res


def _tool_result(name: str) -> ToolExecutionResult:
    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
        "value": [10.0, 11.0],
    })
    rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    return ToolExecutionResult(
        name=name,
        params={},
        success=True,
        df=df,
        cols=list(df.columns),
        rows=rows,
        row_count=len(rows),
        preview="preview",
    )


def test_agent_data_exit_selects_primary_dataset(monkeypatch):
    responses = [
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "get_prices", "args": {"currency": "usd"}}]),
        AIMessage(content="Done.", tool_calls=[]),
    ]
    llm = FakeLLM(responses)
    monkeypatch.setattr(orchestrator, "execute_tool_for_agent", lambda *args, **kwargs: _tool_result("get_prices"))

    ctx = QueryContext(query="Show balancing price trend from 2024 to 2025")
    out = orchestrator.run_agent_loop(ctx, llm=llm)

    assert out.agent_outcome == "data_exit"
    assert out.used_tool is True
    assert out.tool_name == "get_prices"
    assert len(out.rows) == 2
    assert out.is_conceptual is False
    assert out.provenance_source == "tool"
    assert out.provenance_query_hash == tool_invocation_hash("get_prices", {})
    assert out.provenance_cols == ["date", "value"]
    assert len(out.provenance_rows) == 2


def test_agent_conceptual_exit_with_summary(monkeypatch):
    responses = [
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "get_prices", "args": {"currency": "usd"}}]),
        AIMessage(content="Yes, prices increased by about 5 USD/MWh.", tool_calls=[]),
    ]
    llm = FakeLLM(responses)
    monkeypatch.setattr(orchestrator, "execute_tool_for_agent", lambda *args, **kwargs: _tool_result("get_prices"))

    ctx = QueryContext(query="Did prices increase recently?")
    out = orchestrator.run_agent_loop(ctx, llm=llm)

    assert out.agent_outcome == "conceptual_exit"
    assert out.is_conceptual is True
    assert "increased" in out.summary.lower()


def test_agent_fallback_on_max_rounds_with_ambiguous_datasets(monkeypatch):
    responses = [
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "get_prices", "args": {}}]),
        AIMessage(content="", tool_calls=[{"id": "call_2", "name": "get_tariffs", "args": {}}]),
    ]
    llm = FakeLLM(responses)

    def _fake_tool(tool_name, *_args, **_kwargs):
        return _tool_result(tool_name)

    monkeypatch.setattr(orchestrator, "AGENT_MAX_ROUNDS", 2)
    monkeypatch.setattr(orchestrator, "execute_tool_for_agent", _fake_tool)

    ctx = QueryContext(query="energy analysis")
    out = orchestrator.run_agent_loop(ctx, llm=llm)

    assert out.agent_outcome == "fallback_exit"
    assert out.used_tool is False
    assert "max_rounds" in out.agent_fallback_reason


def test_agent_toolset_does_not_expose_sql_generation():
    tools = orchestrator.available_agent_tools()
    assert "generate_sql_query" not in tools
    assert "get_prices" in tools


def test_agent_loop_logs_request_token_usage(monkeypatch):
    m = Metrics()
    monkeypatch.setattr(orchestrator, "metrics", m)
    monkeypatch.setattr(llm_module, "metrics", m)
    m.start_request_telemetry("trace-agent")

    llm = FakeLLM(
        [
            AIMessage(
                content="Conceptual answer",
                tool_calls=[],
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )
        ]
    )

    ctx = QueryContext(query="Explain balancing price")
    out = orchestrator.run_agent_loop(ctx, llm=llm)
    snapshot = m.finalize_request_telemetry()

    assert out.agent_outcome == "conceptual_exit"
    assert snapshot["trace_id"] == "trace-agent"
    assert snapshot["llm_calls"] == 1
    assert snapshot["prompt_tokens"] == 12
    assert snapshot["completion_tokens"] == 8
    assert snapshot["total_tokens"] == 20
