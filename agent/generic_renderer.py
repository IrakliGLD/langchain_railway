"""Generic tabular renderer — one function that formats any canonical frame.

Replaces per-question bespoke formatters in summarizer.py for SCALAR, LIST,
TIMESERIES, and COMPARISON answer kinds.  Scenario and forecast remain
specialized.
"""

from __future__ import annotations

import logging
from typing import Optional

from contracts.evidence_frames import (
    CanonicalFrame,
    ComparisonFrame,
    EntitySetFrame,
    ObservationFrame,
)
from contracts.question_analysis import AnswerKind, Grouping

log = logging.getLogger("Enai")


def render(
    frame: CanonicalFrame,
    answer_kind: AnswerKind,
    grouping: Optional[Grouping] = None,
    entity_scope: Optional[str] = None,
    language_code: str = "en",
) -> str | None:
    """Format a canonical frame into a user-facing text answer.

    Returns None if the frame/answer_kind combination is not handled by the
    generic renderer (caller should fall back to LLM narrative or specialized
    formatter).
    """
    if answer_kind == AnswerKind.SCALAR:
        return _render_scalar(frame)
    if answer_kind == AnswerKind.LIST:
        return _render_list(frame, entity_scope)
    if answer_kind == AnswerKind.TIMESERIES:
        return _render_timeseries(frame, grouping)
    if answer_kind == AnswerKind.COMPARISON:
        return _render_comparison(frame)
    # SCENARIO, FORECAST, EXPLANATION, KNOWLEDGE, CLARIFY — not handled here.
    return None


# ---------------------------------------------------------------------------
# SCALAR — extract single value + unit + period
# ---------------------------------------------------------------------------

def _render_scalar(frame: CanonicalFrame) -> str | None:
    if not isinstance(frame, ObservationFrame) or frame.is_empty():
        return None

    # Take the last row (most recent observation)
    row = frame.rows[-1]
    val = row.get("value")
    unit = row.get("unit", "")
    period = row.get("period", "")
    metric_label = row.get("entity_label") or row.get("metric", "")

    if val is None:
        return None

    formatted_val = _fmt_number(val)
    parts = [metric_label, ":", formatted_val, unit]
    if period:
        parts.append(f"({period})")
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# LIST — enumerate entities grouped by reason
# ---------------------------------------------------------------------------

def _render_list(frame: CanonicalFrame, entity_scope: Optional[str] = None) -> str | None:
    if isinstance(frame, EntitySetFrame):
        return _render_entity_set_list(frame, entity_scope)
    if isinstance(frame, ObservationFrame):
        return _render_observation_as_list(frame)
    return None


def _render_entity_set_list(frame: EntitySetFrame, entity_scope: Optional[str] = None) -> str | None:
    if frame.is_empty():
        return None

    # Group by membership_reason
    groups: dict[str, list[str]] = {}
    for row in frame.rows:
        reason = row.get("membership_reason", "other")
        label = row.get("entity_label", row.get("entity_id", "?"))
        groups.setdefault(reason, []).append(label)

    lines = []
    if entity_scope:
        lines.append(f"**{entity_scope.replace('_', ' ').title()}:**")

    for reason, labels in groups.items():
        if len(groups) > 1:
            lines.append(f"\n**{reason.replace('_', ' ').title()}:**")
        for label in labels:
            lines.append(f"- {label}")

    return "\n".join(lines)


def _render_observation_as_list(frame: ObservationFrame) -> str | None:
    """Render unique entities from an ObservationFrame as a list."""
    if frame.is_empty():
        return None

    seen: dict[str, str] = {}
    for row in frame.rows:
        eid = row.get("entity_id")
        if eid and eid not in seen:
            seen[eid] = row.get("entity_label", eid)

    if not seen:
        return None

    lines = [f"- {label}" for label in seen.values()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TIMESERIES — period-indexed table
# ---------------------------------------------------------------------------

def _render_timeseries(frame: CanonicalFrame, grouping: Optional[Grouping] = None) -> str | None:
    if not isinstance(frame, ObservationFrame) or frame.is_empty():
        return None

    periods = frame.periods
    metrics = frame.metrics
    entities = frame.entities

    # Build a lookup: (period, entity_id, metric) -> value
    lookup: dict[tuple, dict] = {}
    for row in frame.rows:
        key = (row.get("period"), row.get("entity_id"), row.get("metric"))
        lookup[key] = row

    # Determine columns based on grouping
    if grouping == Grouping.BY_ENTITY and len(entities) > 1:
        return _render_timeseries_by_entity(frame, periods, entities, metrics, lookup)
    if grouping == Grouping.BY_METRIC and len(metrics) > 1:
        return _render_timeseries_by_metric(frame, periods, entities, metrics, lookup)

    # Default: one column per metric×entity combination
    return _render_timeseries_flat(frame, periods, entities, metrics, lookup)


def _render_timeseries_flat(
    frame: ObservationFrame,
    periods: list[str],
    entities: list[str],
    metrics: list[str],
    lookup: dict,
) -> str:
    # Discover actual (entity, metric) combinations that exist in the data.
    col_keys: list[tuple[str | None, str]] = []
    seen_keys: set[tuple] = set()
    for row in frame.rows:
        key = (row.get("entity_id"), row.get("metric"))
        if key not in seen_keys:
            seen_keys.add(key)
            col_keys.append(key)

    # Build column labels
    col_labels = []
    for e, m in col_keys:
        sample = lookup.get((periods[0], e, m)) or next(
            (lookup[k] for k in lookup if k[1] == e and k[2] == m), {}
        )
        lbl = sample.get("entity_label") or m
        unit = sample.get("unit", "")
        col_labels.append(f"{lbl} ({unit})" if unit else lbl)

    # Build markdown table
    header = "| Period | " + " | ".join(col_labels) + " |"
    sep = "|---|" + "|".join("---" for _ in col_labels) + "|"
    lines = [header, sep]

    for p in periods:
        cells = [p]
        for e, m in col_keys:
            row = lookup.get((p, e, m))
            cells.append(_fmt_number(row["value"]) if row and row.get("value") is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _render_timeseries_by_entity(frame, periods, entities, metrics, lookup) -> str:
    """Render with entities as columns, one table per metric."""
    tables = []
    for m in metrics:
        sample = next((lookup[k] for k in lookup if k[2] == m), {})
        unit = sample.get("unit", "")
        title = f"**{m}** ({unit})" if unit else f"**{m}**"

        entity_labels = {}
        for e in entities:
            row = next((lookup[k] for k in lookup if k[1] == e and k[2] == m), {})
            entity_labels[e] = row.get("entity_label", e)

        header = "| Period | " + " | ".join(entity_labels[e] for e in entities) + " |"
        sep = "|---|" + "|".join("---" for _ in entities) + "|"
        rows_lines = [title, header, sep]

        for p in periods:
            cells = [p]
            for e in entities:
                row = lookup.get((p, e, m))
                cells.append(_fmt_number(row["value"]) if row and row.get("value") is not None else "—")
            rows_lines.append("| " + " | ".join(cells) + " |")

        tables.append("\n".join(rows_lines))

    return "\n\n".join(tables)


def _render_timeseries_by_metric(frame, periods, entities, metrics, lookup) -> str:
    """Render with metrics as columns, one table per entity (or one if single entity)."""
    target_entities = entities or [None]
    tables = []

    for e in target_entities:
        if e and len(target_entities) > 1:
            sample = next((lookup[k] for k in lookup if k[1] == e), {})
            title = f"**{sample.get('entity_label', e)}**"
        else:
            title = ""

        metric_labels = {}
        for m in metrics:
            row = next((lookup[k] for k in lookup if k[2] == m), {})
            unit = row.get("unit", "")
            metric_labels[m] = f"{m} ({unit})" if unit else m

        header = "| Period | " + " | ".join(metric_labels[m] for m in metrics) + " |"
        sep = "|---|" + "|".join("---" for _ in metrics) + "|"
        rows_lines = ([title] if title else []) + [header, sep]

        for p in periods:
            cells = [p]
            for m in metrics:
                row = lookup.get((p, e, m))
                cells.append(_fmt_number(row["value"]) if row and row.get("value") is not None else "—")
            rows_lines.append("| " + " | ".join(cells) + " |")

        tables.append("\n".join(rows_lines))

    return "\n\n".join(tables)


# ---------------------------------------------------------------------------
# COMPARISON — subject vs baseline + delta
# ---------------------------------------------------------------------------

def _render_comparison(frame: CanonicalFrame) -> str | None:
    if not isinstance(frame, ComparisonFrame) or frame.is_empty():
        return None

    header = "| Metric | Subject | Baseline | Delta | Change % | Unit |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]

    for row in frame.rows:
        metric = row.get("metric", "")
        subj_label = row.get("subject_label", "")
        subj_val = _fmt_number(row.get("subject_value"))
        base_label = row.get("baseline_label", "")
        base_val = _fmt_number(row.get("baseline_value"))
        delta = _fmt_number(row.get("delta"))
        delta_pct = row.get("delta_percent")
        delta_pct_str = f"{delta_pct:+.1f}%" if delta_pct is not None else "—"
        unit = row.get("unit", "")

        lines.append(
            f"| {metric} | {subj_label}: {subj_val} | {base_label}: {base_val} "
            f"| {delta} | {delta_pct_str} | {unit} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_number(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if abs(val) >= 1000:
            return f"{val:,.0f}"
        if abs(val) >= 1:
            return f"{val:.2f}"
        return f"{val:.4f}"
    return str(val)
