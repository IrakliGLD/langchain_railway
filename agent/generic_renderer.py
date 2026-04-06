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
    metric_hint: Optional[str] = None,
) -> str | None:
    """Format a canonical frame into a user-facing text answer.

    Returns None if the frame/answer_kind combination is not handled by the
    generic renderer (caller should fall back to LLM narrative or specialized
    formatter).
    """
    if answer_kind == AnswerKind.SCALAR:
        return _render_scalar(frame, metric_hint=metric_hint)
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

def _render_scalar(
    frame: CanonicalFrame,
    metric_hint: Optional[str] = None,
) -> str | None:
    if not isinstance(frame, ObservationFrame) or frame.is_empty():
        return None

    # Select the row matching the requested metric when a hint is available.
    # Adapters emit the primary metric first, so rows[0] is the safest fallback
    # (rows[-1] could be an unrelated metric like xrate for balancing queries).
    row = None
    if metric_hint:
        hint_lower = metric_hint.lower()
        for r in frame.rows:
            if hint_lower in (r.get("metric") or "").lower():
                row = r
                break
    if row is None:
        row = frame.rows[0]

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
    if isinstance(frame, ComparisonFrame) and not frame.is_empty():
        return _render_comparison_native(frame)
    if isinstance(frame, ObservationFrame) and not frame.is_empty():
        return _render_comparison_from_observations(frame)
    return None


def _render_comparison_native(frame: ComparisonFrame) -> str:
    """Render a pre-built ComparisonFrame."""
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


def _render_comparison_from_observations(frame: ObservationFrame) -> str | None:
    """Pivot an ObservationFrame into a comparison table.

    Strategy: if 2+ periods exist, compare last two periods across all metrics.
    Otherwise if 2+ entities exist, compare first two entities.
    Returns None if neither condition is met (not enough data for comparison).
    """
    periods = frame.periods
    entities = frame.entities
    metrics = frame.metrics

    # Build lookup: (period, entity_id, metric) -> row
    lookup: dict[tuple, dict] = {}
    for row in frame.rows:
        key = (row.get("period"), row.get("entity_id"), row.get("metric"))
        lookup[key] = row

    if len(periods) >= 2:
        # Compare last two periods (baseline = second-to-last, subject = last)
        baseline_period = periods[-2]
        subject_period = periods[-1]
        comparison_rows = []
        for m in metrics:
            for e in (entities or [None]):
                base_row = lookup.get((baseline_period, e, m))
                subj_row = lookup.get((subject_period, e, m))
                if base_row and subj_row:
                    comparison_rows.append(_build_comparison_row(
                        metric=m,
                        subj_label=subject_period,
                        subj_value=subj_row.get("value"),
                        base_label=baseline_period,
                        base_value=base_row.get("value"),
                        unit=subj_row.get("unit", ""),
                    ))
        if comparison_rows:
            return _format_comparison_table(comparison_rows)

    if len(entities) >= 2:
        # Compare first two entities across all metrics and periods
        ent_a, ent_b = entities[0], entities[1]
        comparison_rows = []
        for m in metrics:
            for p in (periods or [None]):
                row_a = lookup.get((p, ent_a, m))
                row_b = lookup.get((p, ent_b, m))
                if row_a and row_b:
                    label_a = row_a.get("entity_label", ent_a)
                    label_b = row_b.get("entity_label", ent_b)
                    comparison_rows.append(_build_comparison_row(
                        metric=m,
                        subj_label=label_a,
                        subj_value=row_a.get("value"),
                        base_label=label_b,
                        base_value=row_b.get("value"),
                        unit=row_a.get("unit", ""),
                    ))
        if comparison_rows:
            return _format_comparison_table(comparison_rows)

    return None


def _build_comparison_row(
    metric: str,
    subj_label: str,
    subj_value,
    base_label: str,
    base_value,
    unit: str,
) -> dict:
    delta = None
    delta_percent = None
    if subj_value is not None and base_value is not None:
        try:
            delta = float(subj_value) - float(base_value)
            if float(base_value) != 0:
                delta_percent = (delta / float(base_value)) * 100
        except (TypeError, ValueError):
            pass
    return {
        "metric": metric,
        "subject_label": subj_label,
        "subject_value": subj_value,
        "baseline_label": base_label,
        "baseline_value": base_value,
        "delta": delta,
        "delta_percent": delta_percent,
        "unit": unit,
    }


def _format_comparison_table(rows: list[dict]) -> str:
    header = "| Metric | Subject | Baseline | Delta | Change % | Unit |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]

    for row in rows:
        subj_val = _fmt_number(row.get("subject_value"))
        base_val = _fmt_number(row.get("baseline_value"))
        delta = _fmt_number(row.get("delta"))
        delta_pct = row.get("delta_percent")
        delta_pct_str = f"{delta_pct:+.1f}%" if delta_pct is not None else "—"

        lines.append(
            f"| {row['metric']} | {row['subject_label']}: {subj_val} "
            f"| {row['baseline_label']}: {base_val} "
            f"| {delta} | {delta_pct_str} | {row.get('unit', '')} |"
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
