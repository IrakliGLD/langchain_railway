"""
Agent tool adapter for Phase 4.

Executes typed tools and returns:
- full dataframe payload (kept out-of-band in orchestrator memory)
- compact preview text for LLM context
"""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from config import (
    AGENT_TOOL_PREVIEW_MAX_CHARS,
    AGENT_TOOL_PREVIEW_ROWS,
    AGENT_TOOL_TIMEOUT_SECONDS,
    AGENT_TOOL_RETRY_ATTEMPTS,
)
from agent.tools import execute_tool
from agent.tools.types import ToolInvocation

log = logging.getLogger("Enai")


# Typed exceptions keep retries reserved for transient execution failures.
class ToolTimeoutError(TimeoutError):
    """Raised when typed tool execution exceeds the configured timeout."""


class ToolNonRetriableError(RuntimeError):
    """Raised when validation/input errors should not be retried."""


@dataclass
class ToolExecutionResult:
    """Normalized tool execution payload for orchestrator use."""

    name: str
    params: Dict[str, Any]
    success: bool
    error: str = ""
    df: Optional[pd.DataFrame] = None
    cols: Optional[List[str]] = None
    rows: Optional[List[tuple]] = None
    row_count: int = 0
    preview: str = ""


# Preview helpers keep LLM-visible tool output compact and deterministic.
def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 32)
    return text[:keep] + "\n... [truncated]"


def _frame_preview(df: pd.DataFrame, max_rows: int) -> str:
    if df is None or df.empty:
        return "(no rows)"

    preview_df = df.head(max_rows).copy()
    preview = preview_df.to_string(index=False)
    if len(df) > max_rows:
        preview += f"\n... and {len(df) - max_rows} more rows."
    return preview


# Run tools in a worker thread so the agent loop can enforce hard timeouts.
def _run_tool_once(invocation: ToolInvocation, timeout_seconds: int):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(execute_tool, invocation)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise ToolTimeoutError(
            f"Tool '{invocation.name}' timed out after {timeout_seconds}s"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


@retry(
    stop=stop_after_attempt(AGENT_TOOL_RETRY_ATTEMPTS),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_not_exception_type(ToolNonRetriableError),
    reraise=True,
)
def _execute_tool_with_retry(invocation: ToolInvocation, timeout_seconds: int):
    try:
        return _run_tool_once(invocation, timeout_seconds=timeout_seconds)
    except (ValueError, TypeError) as exc:
        # Validation/schema errors are deterministic; retrying won't help.
        raise ToolNonRetriableError(str(exc)) from exc


def format_tool_preview_message(dataset_id: str, result: ToolExecutionResult) -> str:
    """Render a compact LLM-facing tool result message."""
    if not result.success:
        return (
            f"dataset_id: {dataset_id}\n"
            f"tool: {result.name}\n"
            f"status: error\n"
            f"error: {result.error}"
        )

    columns = ", ".join(result.cols or [])
    return (
        f"dataset_id: {dataset_id}\n"
        f"tool: {result.name}\n"
        f"status: ok\n"
        f"rows: {result.row_count}\n"
        f"columns: {columns}\n"
        f"preview:\n{result.preview}"
    )


# Main adapter entrypoint: execute the tool, then package a safe preview and raw data.
def execute_tool_for_agent(
    tool_name: str,
    params: Optional[Dict[str, Any]],
    preview_rows: int = AGENT_TOOL_PREVIEW_ROWS,
    preview_max_chars: int = AGENT_TOOL_PREVIEW_MAX_CHARS,
    timeout_seconds: int = AGENT_TOOL_TIMEOUT_SECONDS,
) -> ToolExecutionResult:
    """Execute a typed tool and prepare an LLM-safe preview."""
    safe_params = dict(params or {})
    invocation = ToolInvocation(name=tool_name, params=safe_params)

    try:
        df, cols, rows = _execute_tool_with_retry(
            invocation=invocation,
            timeout_seconds=max(1, int(timeout_seconds)),
        )
        out_cols = list(cols)
        out_rows = [tuple(r) for r in rows]
        preview = _frame_preview(df, max_rows=max(1, int(preview_rows)))
        preview = _truncate_text(preview, max_chars=max(200, int(preview_max_chars)))
        return ToolExecutionResult(
            name=tool_name,
            params=safe_params,
            success=True,
            df=df,
            cols=out_cols,
            rows=out_rows,
            row_count=len(out_rows),
            preview=preview,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        log.warning("Agent tool execution failed: tool=%s err=%s", tool_name, exc)
        return ToolExecutionResult(
            name=tool_name,
            params=safe_params,
            success=False,
            error=str(exc),
            cols=[],
            rows=[],
            row_count=0,
            preview="",
        )
