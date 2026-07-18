"""Stage 3 analytical derivation with provenance-safe context adoption."""

from __future__ import annotations

from agent import analyzer
from agent.provenance import (
    sql_query_hash,
    stamp_provenance,
    tool_invocation_hash,
)
from models import QueryContext


def requested_derived_metric_names(ctx: QueryContext) -> list[str]:
    """Return active analyzer-requested derived metrics in stable order."""
    if not ctx.has_authoritative_question_analysis:
        return []

    names: list[str] = []
    seen: set[str] = set()
    for metric in ctx.question_analysis.analysis_requirements.derived_metrics or []:
        name = getattr(metric.metric_name, "value", str(metric.metric_name or "")).strip()
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def missing_requested_evidence(ctx: QueryContext) -> list[str]:
    """Return requested derived metrics that Stage 3 did not materialize."""
    requested = list(ctx.requested_derived_metrics or [])
    if not requested:
        return []

    evidence_names = {
        str(record.get("derived_metric_name") or "").strip()
        for record in (ctx.analysis_evidence or [])
        if str(record.get("derived_metric_name") or "").strip()
    }
    return [name for name in requested if name not in evidence_names]


def derive_evidence(ctx: QueryContext) -> QueryContext:
    """Run Stage 3 once and provenance-stamp its finalized tabular result."""
    ctx = analyzer.enrich(ctx)
    if not (
        ctx.rows
        and ctx.cols
        and set(ctx.cols) - set(ctx.provenance_cols or [])
    ):
        return ctx

    inferred_source = str(ctx.provenance_source or ("tool" if ctx.used_tool else "sql"))
    inferred_hash = str(ctx.provenance_query_hash or "")
    if not inferred_hash:
        if ctx.used_tool and ctx.tool_name:
            inferred_hash = tool_invocation_hash(ctx.tool_name, ctx.tool_params)
        elif ctx.safe_sql:
            inferred_hash = sql_query_hash(ctx.safe_sql)
    stamp_provenance(
        ctx,
        ctx.cols,
        ctx.rows,
        source=inferred_source or "tool",
        query_hash=inferred_hash or sql_query_hash(f"{ctx.query}|stage3_enriched"),
    )
    return ctx
