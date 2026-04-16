"""Generic tabular renderer — one function that formats any canonical frame.

Replaces per-question bespoke formatters in summarizer.py for SCALAR, LIST,
TIMESERIES, and COMPARISON answer kinds.  Scenario and forecast remain
specialized.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Optional

from contracts.evidence_frames import (
    CanonicalFrame,
    ComparisonFrame,
    EntitySetFrame,
    ForecastFrame,
    ObservationFrame,
    ScenarioFrame,
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
    if answer_kind == AnswerKind.SCENARIO:
        return _render_scenario(frame)
    if answer_kind == AnswerKind.FORECAST:
        return _render_forecast(frame)
    # EXPLANATION, KNOWLEDGE, CLARIFY — not handled here (require LLM narrative).
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

    if any((row.get("attributes") or {}).get("values") for row in frame.rows):
        return _render_entity_set_snapshot_list(frame, entity_scope)

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


def _render_entity_set_snapshot_list(frame: EntitySetFrame, entity_scope: Optional[str] = None) -> str | None:
    lines = []
    if entity_scope:
        lines.append(f"**{entity_scope.replace('_', ' ').title()}:**")

    periods = []
    for row in frame.rows:
        attrs = row.get("attributes") or {}
        period = attrs.get("period")
        if period and period not in periods:
            periods.append(period)
    if len(periods) == 1:
        lines.append(f"**Period:** {_format_period_label(periods[0])}")

    for row in frame.rows:
        label = row.get("entity_label", row.get("entity_id", "?"))
        attrs = row.get("attributes") or {}
        values = attrs.get("values") or []
        if values:
            lines.append(f"- {label}: {_format_entity_values(values)}")
        else:
            lines.append(f"- {label}")

    return "\n".join(lines)


def _format_entity_values(values: list[dict]) -> str:
    formatted = []
    distinct_labels = {str(v.get("label") or "").strip() for v in values if v.get("label")}
    show_metric_label = len(distinct_labels) > 1
    for entry in values:
        val = _fmt_number(entry.get("value"))
        unit = str(entry.get("unit") or "").strip()
        label = str(entry.get("label") or "").strip().replace("_", " ")
        sub_period = _format_period_label(str(entry.get("sub_period") or "").strip())
        value_text = " ".join(part for part in (val, unit) if part).strip()
        if sub_period:
            value_text = f"{sub_period}: {value_text}" if value_text else sub_period
        if label and show_metric_label:
            formatted.append(f"{label.title()}: {value_text}")
        else:
            formatted.append(value_text)
    return "; ".join(part for part in formatted if part)


def _format_period_label(raw_period: str) -> str:
    period = str(raw_period or "").strip()
    if not period:
        return ""
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(period, fmt)
            if fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
                return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
            if fmt == "%Y":
                return dt.strftime("%Y")
            return dt.strftime("%B %Y")
        except ValueError:
            continue
    return period


def _render_observation_as_list(frame: ObservationFrame) -> str | None:
    """Render unique entities from an ObservationFrame as a list or snapshot."""
    if frame.is_empty():
        return None

    if len(frame.periods) == 1:
        return _render_observation_snapshot_list(frame)

    seen: dict[str, str] = {}
    for row in frame.rows:
        eid = row.get("entity_id")
        if eid and eid not in seen:
            seen[eid] = row.get("entity_label", eid)

    if not seen:
        return None

    lines = [f"- {label}" for label in seen.values()]
    return "\n".join(lines)


def _render_observation_snapshot_list(frame: ObservationFrame) -> str | None:
    if frame.is_empty():
        return None

    shared_metric = frame.metrics[0] if len(frame.metrics) == 1 else ""
    period = frame.periods[0] if frame.periods else ""
    grouped: dict[str, dict[str, object]] = {}
    for row in frame.rows:
        entity_id = row.get("entity_id")
        if not entity_id:
            continue
        group = grouped.setdefault(
            entity_id,
            {
                "label": row.get("entity_label", entity_id),
                "values": [],
            },
        )
        value_text = " ".join(
            part for part in (_fmt_number(row.get("value")), str(row.get("unit") or "").strip()) if part
        ).strip()
        metric = str(row.get("metric") or "").strip().replace("_", " ")
        if metric and shared_metric == "":
            value_text = f"{metric.title()}: {value_text}"
        if value_text:
            group["values"].append(value_text)

    if not grouped:
        return None

    lines = []
    if period:
        lines.append(f"**Period:** {_format_period_label(period)}")
    for entity in grouped.values():
        joined_values = "; ".join(entity["values"])
        lines.append(f"- {entity['label']}: {joined_values}" if joined_values else f"- {entity['label']}")
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
# SCENARIO — payoff / scale / offset analysis
# ---------------------------------------------------------------------------

def _render_scenario(frame: CanonicalFrame) -> str | None:
    if not isinstance(frame, ScenarioFrame) or frame.is_empty():
        return None

    rec = frame.rows[0]
    metric_name = rec.get("metric_name", "")
    factor = rec.get("scenario_factor")
    volume = rec.get("scenario_volume")
    agg_result = rec.get("aggregate_result")
    if agg_result is None:
        return None

    row_count = rec.get("row_count")
    period_range = rec.get("period_range", "")
    min_val = rec.get("min_period_value")
    max_val = rec.get("max_period_value")
    mean_val = rec.get("mean_period_value")
    formula = rec.get("formula", "")

    parts: list[str] = []

    if metric_name == "scenario_payoff":
        positive_sum = rec.get("positive_sum", 0.0)
        negative_sum = rec.get("negative_sum", 0.0)
        positive_count = rec.get("positive_count", 0)
        negative_count = rec.get("negative_count", 0)
        market_component = rec.get("market_component_aggregate")
        combined_total = rec.get("combined_total_aggregate")
        metric_key = str(rec.get("metric") or "").strip()
        market_label = (
            "Balancing market sales income"
            if metric_key in {"p_bal_usd", "p_bal_gel"}
            else "Market sales income at observed prices"
        )
        # Include income breakdown when the data is available.
        include_income_breakdown = (
            market_component is not None and combined_total is not None
        )

        parts.append(
            f"**{'CfD Payoff and Income Analysis' if include_income_breakdown else 'CfD Payoff Analysis'}** "
            f"(strike: {factor} USD/MWh"
            + (f", volume: {volume} MW" if volume is not None else "")
            + ")"
        )
        if period_range:
            parts.append(f"**Period:** {period_range} ({row_count} months)")
        parts.append(f"**Formula:** {formula}")
        if include_income_breakdown:
            parts.append(f"**{market_label}:** {market_component} USD")
            parts.append(f"**CfD financial compensation:** {agg_result} USD")
            parts.append(f"**Total combined income:** {combined_total} USD")
        parts.append(f"**Net total payoff:** {agg_result} USD")
        if positive_sum and positive_sum != 0:
            parts.append(
                f"**Income from favorable periods** (market price below strike): "
                f"{positive_sum} USD across {positive_count} months"
            )
        if negative_sum and negative_sum != 0:
            parts.append(
                f"**Compensation cost in unfavorable periods** (market price above strike): "
                f"{negative_sum} USD across {negative_count} months"
            )
        parts.append(
            f"**Per-period range:** min {min_val} to max {max_val} USD "
            f"(average {mean_val} USD/month)"
        )

    elif metric_name in ("scenario_scale", "scenario_offset"):
        baseline = rec.get("baseline_aggregate")
        delta = rec.get("delta_aggregate")
        delta_pct = rec.get("delta_percent")
        op = "\u00d7" if metric_name == "scenario_scale" else "+"
        parts.append(f"**Scenario Analysis** ({op} {factor})")
        if period_range:
            parts.append(f"**Period:** {period_range} ({row_count} periods)")
        parts.append(f"**Result:** {agg_result}")
        if baseline is not None:
            parts.append(f"**Baseline:** {baseline}")
        if delta is not None:
            parts.append(
                f"**Change:** {delta}"
                + (f" ({delta_pct}%)" if delta_pct is not None else "")
            )
        parts.append(f"**Range:** {min_val} to {max_val} (mean {mean_val})")

    else:
        return None

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# FORECAST — trendline + R² caveat + seasonal breakdown
# ---------------------------------------------------------------------------

_FORECAST_METRIC_LABELS = {
    "p_bal_gel": "Balancing electricity price",
    "p_bal_usd": "Balancing electricity price",
    "p_dereg_gel": "Deregulated electricity price",
    "p_dereg_usd": "Deregulated electricity price",
    "p_gcap_gel": "Guaranteed capacity price",
    "p_gcap_usd": "Guaranteed capacity price",
    "xrate": "Exchange rate",
}

_BALANCING_PRICE_METRICS = {"p_bal_gel", "p_bal_usd"}
_REGIME_BREAK_DATE = "2027-07"
_REGIME_BREAK_WARNING = (
    "\n\n**Important:** Georgia's planned target electricity market model (~July 2027) "
    "would shift balancing price formation from monthly weighted-average settlement to hourly marginal pricing "
    "under self-dispatch. This forecast extends into or beyond that horizon, so the structural break "
    "may fundamentally change price dynamics. Past policy changes (e.g., the January 2024 gas price increase "
    "for regulated thermals, and temporary deregulated hydro pricing rule changes in Jan\u2013Mar 2024) "
    "illustrate how regulatory decisions can materially shift balancing prices."
)


def _price_unit_for_metric(metric: str) -> str:
    metric_lower = str(metric or "").lower()
    if metric_lower.endswith("_usd"):
        return "USD/MWh"
    if metric_lower.endswith("_gel"):
        return "GEL/MWh"
    if metric_lower == "xrate":
        return "GEL per USD"
    return "Value"


def _forecast_caveat_for_r_squared(r_squared: float | None) -> str:
    if r_squared is None:
        return (
            "This forecast is based on historical trendline extrapolation and should be treated cautiously "
            "because balancing prices can change with tariffs, new PPA/CfD capacity, market rules, and import prices."
        )
    if r_squared < 0.5:
        return (
            f"This forecast has moderate-to-low reliability (R\u00b2={r_squared:.3f}) due to variability in historical prices. "
            "Actual prices may differ significantly because of regulatory decisions, new PPA/CfD capacity, market rule changes, or import price shifts."
        )
    if r_squared < 0.7:
        return (
            f"This forecast assumes current market structure, PPA contracts, and regulatory framework remain unchanged (R\u00b2={r_squared:.3f}). "
            "Actual prices may differ because of gas-price negotiations, new PPA/CfD capacity additions, GNERC tariff decisions, or changes in neighbouring electricity markets."
        )
    return (
        f"While this trend is statistically strong (R\u00b2={r_squared:.3f}), it still reflects past patterns and assumes unchanged regulatory and contractual conditions. "
        "Key uncertainties remain PPA/CfD capacity growth, gas-price negotiations, market-rule changes, and import-price dynamics."
    )


def _regime_break_warning_if_needed(target_date: str | None, metrics: set[str]) -> str:
    if not target_date or not (metrics & _BALANCING_PRICE_METRICS):
        return ""
    if target_date >= _REGIME_BREAK_DATE:
        return _REGIME_BREAK_WARNING
    return ""


def _render_forecast(frame: CanonicalFrame) -> str | None:
    if not isinstance(frame, ForecastFrame) or frame.is_empty():
        return None

    target_date = frame.target_date
    entries = frame.rows

    # Determine primary metric and label.
    primary_entry = next(
        (e for e in entries if str(e.get("metric", "")).endswith("_gel")),
        entries[0],
    )
    primary_metric = str(primary_entry.get("metric", "")).strip()
    metric_label = _FORECAST_METRIC_LABELS.get(
        primary_metric, primary_metric.replace("_", " ").title()
    )

    # Format target date for display.
    target_label = target_date or "the requested forecast horizon"
    try:
        import pandas as pd
        ts = pd.to_datetime(target_date, errors="raise")
        target_label = ts.strftime("%B %Y")
    except Exception:
        pass

    # Check regime-break warning.
    entry_metrics = {
        str(e.get("metric", "")).split("_summer")[0].split("_winter")[0]
        for e in entries
    }
    regime_warning = _regime_break_warning_if_needed(target_date, entry_metrics)

    # Separate seasonal and overall entries.
    season_entries = [e for e in entries if e.get("season")]
    overall_entries = [e for e in entries if not e.get("season")]
    gel_entry = next(
        (e for e in overall_entries if str(e.get("metric", "")).endswith("_gel")), None
    )
    usd_entry = next(
        (e for e in overall_entries if str(e.get("metric", "")).endswith("_usd")), None
    )

    if season_entries:
        lines = [
            f"**{metric_label} Forecast**",
            "",
            f"Based on linear regression to {target_label}:",
        ]
        if gel_entry is not None:
            lines.append(
                f"- Overall (GEL): {float(gel_entry['forecast_value']):.2f} "
                f"{_price_unit_for_metric(str(gel_entry.get('metric', '')))} "
                f"(R\u00b2={float(gel_entry.get('r_squared') or 0.0):.3f})"
            )
        if usd_entry is not None:
            lines.append(
                f"- Overall (USD): {float(usd_entry['forecast_value']):.2f} "
                f"{_price_unit_for_metric(str(usd_entry.get('metric', '')))} "
                f"(R\u00b2={float(usd_entry.get('r_squared') or 0.0):.3f})"
            )
        for entry in sorted(
            season_entries,
            key=lambda item: (
                {"summer": 0, "winter": 1}.get(
                    str(item.get("season", "")).strip().lower(), 99
                ),
                0
                if str(item.get("metric", "")).strip().lower().endswith("_gel")
                else 1
                if str(item.get("metric", "")).strip().lower().endswith("_usd")
                else 2,
            ),
        ):
            season = str(entry.get("season", "")).title()
            metric = str(entry.get("metric", "")).strip().lower()
            currency_label = (
                "GEL"
                if metric.endswith("_gel")
                else "USD"
                if metric.endswith("_usd")
                else metric.upper()
            )
            season = f"{season} ({currency_label})"
            unit = _price_unit_for_metric(str(entry.get("metric", "")))
            lines.append(
                f"- {season}: {float(entry['forecast_value']):.2f} {unit} "
                f"(R\u00b2={float(entry.get('r_squared') or 0.0):.3f})"
            )
        lines.append("")
        lines.append(
            _forecast_caveat_for_r_squared(
                (gel_entry or primary_entry).get("r_squared")
            )
        )
        if regime_warning:
            lines.append(regime_warning)
        return "\n".join(lines)

    # Non-seasonal path.
    if gel_entry is not None:
        answer = (
            f"Based on linear regression, **{metric_label}** is forecast to reach "
            f"**{float(gel_entry['forecast_value']):.2f} {_price_unit_for_metric(str(gel_entry.get('metric', '')))}** "
            f"by **{target_label}** (R\u00b2={float(gel_entry.get('r_squared') or 0.0):.3f})."
        )
        if usd_entry is not None:
            answer += (
                f" The parallel USD series points to **{float(usd_entry['forecast_value']):.2f} "
                f"{_price_unit_for_metric(str(usd_entry.get('metric', '')))}** "
                f"(R\u00b2={float(usd_entry.get('r_squared') or 0.0):.3f})."
            )
        answer += f"\n\n{_forecast_caveat_for_r_squared(gel_entry.get('r_squared'))}"
        answer += regime_warning
        return answer

    first_entry = entries[0]
    return (
        f"Based on linear regression, **{metric_label}** is forecast to reach "
        f"**{float(first_entry['forecast_value']):.2f} {_price_unit_for_metric(str(first_entry.get('metric', '')))}** "
        f"by **{target_label}** (R\u00b2={float(first_entry.get('r_squared') or 0.0):.3f}).\n\n"
        f"{_forecast_caveat_for_r_squared(first_entry.get('r_squared'))}"
        f"{regime_warning}"
    )


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
