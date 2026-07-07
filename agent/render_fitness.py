"""Shadow fitness checks for deterministic renders (architecture §3.9).

Deterministic renders ship with no post-hoc numeric gate; these checks make
the "right shape, wrong rows" failure class VISIBLE (trace + counter) without
changing behavior. Cutover to CLARIFY-on-violation is a future, separately
gated decision.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from visualization.chart_selector import detect_column_types


def period_bounds_from_hint(question_analysis) -> tuple[str, str] | None:
    """Requested (start_date, end_date) from the top candidate tool's params hint.

    Tolerant of partial/fake analysis objects — fitness checking is shadow
    observability and must never be able to raise into the pipeline.
    """
    tooling = getattr(question_analysis, "tooling", None)
    tools = getattr(tooling, "candidate_tools", None) or []
    hint = getattr(tools[0], "params_hint", None) if tools else None
    start = getattr(hint, "start_date", None)
    end = getattr(hint, "end_date", None)
    if not (start and end):
        return None
    return str(start), str(end)


def _df_date_span(df: pd.DataFrame) -> tuple[date, date] | None:
    if df is None or df.empty:
        return None
    time_cols, _, _ = detect_column_types(list(df.columns))
    if not time_cols:
        return None
    series = pd.to_datetime(df[time_cols[0]], errors="coerce").dropna()
    if series.empty:
        return None
    return series.min().date(), series.max().date()


def evaluate_render_fitness(ctx) -> list[str]:
    """Return violation tags for the deterministic render about to ship.

    Tags: ``empty_result_rendered`` (no rows behind the render),
    ``period_coverage_gap`` (rendered rows entirely outside the requested
    period), ``requested_entities_missing`` (none of the requested entities
    appear in the rendered frame). Shadow-only — the caller logs, never blocks.
    """
    tags: list[str] = []
    df = ctx.df
    if df is None or df.empty or not ctx.rows:
        tags.append("empty_result_rendered")
        return tags

    qa = ctx.question_analysis if ctx.has_authoritative_question_analysis else None

    bounds = period_bounds_from_hint(qa) if qa is not None else None
    if bounds:
        span = _df_date_span(df)
        if span:
            req_start = date.fromisoformat(bounds[0])
            req_end = date.fromisoformat(bounds[1])
            if span[1] < req_start or span[0] > req_end:
                tags.append("period_coverage_gap")

    if qa is not None:
        tools = getattr(getattr(qa, "tooling", None), "candidate_tools", None) or []
        hint = getattr(tools[0], "params_hint", None) if tools else None
        entities = list(getattr(hint, "entities", []) or [])
        if entities:
            haystack = " ".join(str(c).lower() for c in df.columns)
            haystack += " " + " ".join(
                str(v).lower() for v in df.astype(str).values.flatten()[:500]
            )
            if not any(e.lower() in haystack for e in entities):
                tags.append("requested_entities_missing")
    return tags
