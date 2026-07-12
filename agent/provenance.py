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


def source_rows_hash(cols: Sequence[Any], rows: Iterable[Iterable[Any]]) -> str:
    """Hash the exact tabular source shape and values used downstream."""
    payload = {
        "cols": [str(col) for col in (cols or [])],
        "rows": [list(row) for row in (rows or [])],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def build_provenance_refs(
    cols: Sequence[Any],
    rows: Iterable[Iterable[Any]],
    *,
    source: str,
    query_hash: str,
    parent_refs: Iterable[str] = (),
) -> list[str]:
    """Build stable query/source references and retain every parent reference."""
    materialized_rows = [tuple(row) for row in (rows or [])]
    refs = [str(ref) for ref in parent_refs if str(ref)]
    if query_hash:
        refs.append(f"query:{source or 'unknown'}:{query_hash}")
    refs.append(f"source:rows:{source_rows_hash(cols, materialized_rows)}")
    return list(dict.fromkeys(refs))


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
    materialized_rows = [tuple(r) for r in (rows or [])]
    ctx.provenance_cols = list(cols or [])
    ctx.provenance_rows = materialized_rows
    ctx.provenance_source = str(source or "")
    ctx.provenance_query_hash = str(query_hash or "")
    ctx.provenance_refs = build_provenance_refs(
        ctx.provenance_cols,
        materialized_rows,
        source=ctx.provenance_source,
        query_hash=ctx.provenance_query_hash,
        parent_refs=ctx.provenance_refs,
    )
    return ctx


def clear_provenance(ctx: QueryContext) -> QueryContext:
    """Reset provenance metadata when execution is blocked or fails."""
    ctx.provenance_cols = []
    ctx.provenance_rows = []
    ctx.provenance_source = ""
    ctx.provenance_query_hash = ""
    ctx.provenance_refs = []
    return ctx
