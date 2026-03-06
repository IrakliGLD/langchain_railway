"""
Shared helpers for typed retrieval tools.
"""
from datetime import date
import datetime as dt
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import text

from config import MAX_ROWS
from core.query_executor import ENGINE, check_dataframe_memory
from .types import ToolResult


def normalize_limit(limit: Optional[int]) -> int:
    """Clamp tool row limits to a safe bounded value."""
    if limit is None:
        return MAX_ROWS
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        return MAX_ROWS
    return max(1, min(parsed, MAX_ROWS))


def normalize_date(value: Optional[str]) -> Optional[str]:
    """Normalize user/router date into YYYY-MM-DD string for bind params."""
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    if len(raw) == 4 and raw.isdigit():
        return f"{raw}-01-01"
    if len(raw) == 7 and raw[4] == "-":
        return f"{raw}-01"

    try:
        parsed = dt.date.fromisoformat(raw)
        return parsed.isoformat()
    except ValueError:
        return None


def last_day_of_month(year: int, month: int) -> str:
    """Return YYYY-MM-DD for the last day of the given month."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return (next_month - dt.timedelta(days=1)).isoformat()


def run_text_query(sql: str, params: Optional[Dict[str, Any]] = None) -> ToolResult:
    """Execute read-only SQL text and return (df, cols, rows)."""
    params = params or {}
    with ENGINE.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        result = conn.execute(text(sql), params)
        rows = [tuple(r) for r in result.fetchall()]
        cols = list(result.keys())
        df = pd.DataFrame(rows, columns=cols)

    check_dataframe_memory(df)
    return df, cols, rows


def run_statement(statement: Any, params: Optional[Dict[str, Any]] = None) -> ToolResult:
    """Execute a SQLAlchemy statement object and return (df, cols, rows)."""
    params = params or {}
    with ENGINE.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        result = conn.execute(statement, params)
        rows = [tuple(r) for r in result.fetchall()]
        cols = list(result.keys())
        df = pd.DataFrame(rows, columns=cols)

    check_dataframe_memory(df)
    return df, cols, rows
