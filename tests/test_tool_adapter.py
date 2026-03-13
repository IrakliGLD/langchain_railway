"""
Tests for Phase 4 agent tool adapter.
"""
import os

import pandas as pd

# Ensure config validation passes before importing project modules.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import tool_adapter  # noqa: E402


def test_execute_tool_for_agent_success_preview_and_contract(monkeypatch):
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "p_bal_usd": [float(i) for i in range(12)],
    })
    rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    cols = list(df.columns)

    def _fake_execute_tool(_invocation):
        return df, cols, rows

    monkeypatch.setattr(tool_adapter, "execute_tool", _fake_execute_tool)

    result = tool_adapter.execute_tool_for_agent(
        tool_name="get_prices",
        params={"currency": "usd"},
        preview_rows=3,
        preview_max_chars=2000,
    )

    assert result.success is True
    assert result.row_count == 12
    assert result.cols == cols
    assert result.rows == rows
    assert all(isinstance(r, tuple) for r in result.rows)
    assert "... and 9 more rows." in result.preview


def test_execute_tool_for_agent_failure(monkeypatch):
    def _fake_execute_tool(_invocation):
        raise ValueError("boom")

    monkeypatch.setattr(tool_adapter, "execute_tool", _fake_execute_tool)

    result = tool_adapter.execute_tool_for_agent(
        tool_name="get_prices",
        params={},
    )

    assert result.success is False
    assert "boom" in result.error
    assert result.row_count == 0


def test_format_tool_preview_message_error_path():
    result = tool_adapter.ToolExecutionResult(
        name="get_prices",
        params={},
        success=False,
        error="unsupported params",
        cols=[],
        rows=[],
    )
    rendered = tool_adapter.format_tool_preview_message("ds_1", result)
    assert "status: error" in rendered
    assert "unsupported params" in rendered
