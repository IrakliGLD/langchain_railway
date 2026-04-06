"""
Shared helpers for tabular provenance stamping.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

from models import QueryContext


# Hash helpers provide stable identifiers for SQL and tool-derived evidence.
def tool_invocation_hash(tool_name: str, tool_params: dict) -> str:
    """Build a stable short hash for a typed-tool invocation."""
    payload = {
        "tool": str(tool_name or ""),
        "params": dict(tool_params or {}),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def sql_query_hash(sql_text: str) -> str:
    """Build a stable short hash for SQL provenance tracking."""
    return hashlib.sha256(str(sql_text or "").encode("utf-8")).hexdigest()[:16]


# Context mutators copy the exact rows/columns used for downstream claim tracing.
def stamp_provenance(
    ctx: QueryContext,
    cols: Sequence[Any],
    rows: Iterable[Iterable[Any]],
    *,
    source: str,
    query_hash: str,
) -> QueryContext:
    """Attach exact source rows and identity used for downstream summarization."""
    ctx.provenance_cols = list(cols or [])
    ctx.provenance_rows = [tuple(r) for r in (rows or [])]
    ctx.provenance_source = str(source or "")
    ctx.provenance_query_hash = str(query_hash or "")
    return ctx


def clear_provenance(ctx: QueryContext) -> QueryContext:
    """Reset provenance metadata when execution is blocked or fails."""
    ctx.provenance_cols = []
    ctx.provenance_rows = []
    ctx.provenance_source = ""
    ctx.provenance_query_hash = ""
    return ctx
