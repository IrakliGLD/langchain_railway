"""
Pipeline Stage 3: Analysis & Enrichment

Handles share resolution, correlation analysis, forecast mode (CAGR),
"why" causal reasoning, trendline pre-calculation, and seasonal stats.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from contracts.question_analysis import ChartIntent, SemanticRole
from models import QueryContext
from core.query_executor import ENGINE
from analysis.stats import quick_stats, rows_to_preview
from analysis.seasonal_stats import (
    detect_monthly_timeseries,
    calculate_seasonal_stats,
    format_seasonal_stats,
)
from analysis.shares import build_balancing_correlation_df
from agent.provenance import sql_query_hash, stamp_provenance
from agent.sql_executor import BALANCING_SHARE_PIVOT_SQL, ensure_share_dataframe, fetch_balancing_share_panel
from utils.trace_logging import trace_detail

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# Constants — canonical definitions live in config.metric_config;
# re-exported here for backward compatibility with existing imports.
# ---------------------------------------------------------------------------

from config_metrics.metric_config import (
    BALANCING_SHARE_METADATA,
    METRIC_VALUE_ALIASES,
    DERIVED_METRIC_DEFAULTS,
    SUMMER_MONTHS,
)
from agent.metric_registry import MetricContext, dispatch_metric

MONTH_NAME_TO_NUMBER = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_SCENARIO_TOTAL_INCOME_QUERY_SIGNALS = (
    "total income",
    "total revenue",
    "combined income",
    "combined revenue",
    "overall income",
    "overall revenue",
    "market sale",
    "market sales",
    "sell",
    "sales",
)
_SCENARIO_COMPONENT_QUERY_SIGNALS = (
    "cfd",
    "compensation",
    "payoff",
)
_SCENARIO_REFERENCE_QUERY_SIGNALS = (
    "strike",
    "reference",
    "benchmark",
    "threshold",
    "cap",
)


# ---------------------------------------------------------------------------
# Share summary helpers (moved from main.py)
# ---------------------------------------------------------------------------

def build_share_shift_notes(
    cur_shares: dict[str, float],
    prev_shares: dict[str, float],
) -> List[str]:
    """Generate textual notes describing month-over-month share changes."""
    notes = []
    if not cur_shares or not prev_shares:
        return notes

    deltas = []
    for key in cur_shares:
        cur_val = cur_shares.get(key, 0)
        prev_val = prev_shares.get(key, 0)
        delta = cur_val - prev_val
        meta = BALANCING_SHARE_METADATA.get(key, {})
        label = meta.get("label", key.replace("_", " "))
        cost = meta.get("cost", "unknown")
        usd = meta.get("usd_linked", False)
        deltas.append((label, delta, cost, usd, key))

    deltas.sort(key=lambda x: abs(x[1]), reverse=True)

    significant = [(l, d, c, u, k) for l, d, c, u, k in deltas if abs(d) >= 0.005]
    if not significant:
        return notes

    parts = []
    for label, delta, cost, usd, key in significant[:5]:
        direction = "↑" if delta > 0 else "↓"
        parts.append(f"{label} {direction}{abs(delta)*100:.1f}pp")

    notes.append(f"Share shifts month-over-month: {', '.join(parts)}.")

    cheap_delta = sum(d for _, d, c, _, _ in significant if c == "cheap")
    moderate_delta = sum(d for _, d, c, _, _ in significant if c == "moderate")
    expensive_delta = sum(d for _, d, c, _, _ in significant if c == "expensive")

    if cheap_delta < -0.01:
        notes.append("Cheaper balancing supply contracted — upward price pressure.")
    if cheap_delta > 0.01:
        notes.append("Cheaper balancing supply expanded — downward price pressure.")
        
    if moderate_delta > 0.01:
        notes.append("Moderate-cost groups expanded.")
    if moderate_delta < -0.01:
        notes.append("Moderate-cost groups contracted.")

    if expensive_delta > 0.01:
        notes.append("Expensive balancing supply expanded — upward price pressure.")
    if expensive_delta < -0.01:
        notes.append("Expensive balancing supply contracted — downward price pressure.")

    usd_delta = sum(d for _, d, _, u, _ in significant if u)
    if abs(usd_delta) >= 0.01:
        direction = "expanded" if usd_delta > 0 else "contracted"
        notes.append(f"USD-denominated sellers {direction} by {abs(usd_delta)*100:.1f}pp — xrate sensitivity {'increased' if usd_delta > 0 else 'decreased'}.")

    return notes



def _is_balancing_price_query(query_text: str) -> bool:
    query_lower = query_text.lower()
    has_price = any(term in query_lower for term in ["price", "p_bal"])
    has_balancing_context = any(
        term in query_lower
        for term in [
            "balancing price",
            "balancing electricity price",
            "balancing electricity",
            "balancing market",
            " p_bal",
        ]
    ) or query_lower.startswith("p_bal")
    return has_price and has_balancing_context


def _metric_aliases(metric: str) -> list[str]:
    metric_name = str(metric or "").strip()
    if not metric_name:
        return []
    return METRIC_VALUE_ALIASES.get(metric_name, [metric_name])


_SCENARIO_FALLBACK_PATTERNS: list[tuple[str, str, str]] = [
    # (regex, metric_name, factor_group_meaning)
    # "X% higher/lower" → scenario_scale
    (r"(\d+(?:\.\d+)?)\s*%\s*(?:higher|more|increase)", "scenario_scale", "pct_higher"),
    (r"(\d+(?:\.\d+)?)\s*%\s*(?:lower|less|decrease)", "scenario_scale", "pct_lower"),
    # "double/twice" → scenario_scale factor=2.0
    (r"\b(?:double|twice)\b", "scenario_scale", "double"),
    (r"\b(?:half)\b", "scenario_scale", "half"),
    # "strike X" or "strike price X" (adjacent) → scenario_payoff
    (r"strike\s*(?:price)?\s*(\d+(?:\.\d+)?)", "scenario_payoff", "strike"),
    # CfD/PPA contract with a price in USD/GEL/EUR (may be far from "strike")
    (r"(?:cfd|ppa|contract for difference).{0,120}?(\d+(?:\.\d+)?)\s*(?:usd|gel|eur)(?:/mwh)?", "scenario_payoff", "strike"),
    # "strike" anywhere ... number with currency unit later
    (r"strike.{0,120}?(\d+(?:\.\d+)?)\s*(?:usd|gel|eur)(?:/mwh)?", "scenario_payoff", "strike"),
    # "X USD/GEL higher/more" → scenario_offset
    (r"(\d+(?:\.\d+)?)\s*(?:usd|gel|eur)\s*(?:higher|more)", "scenario_offset", "offset"),
]

_DEFAULT_SCENARIO_METRIC = "p_bal_usd"
_VOLUME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mw(?:\b|/)", re.IGNORECASE)


def _scenario_fallback_requests(query: str) -> list[dict[str, Any]]:
    """Try to extract a scenario request from the raw query text.

    This is a best-effort heuristic for when the LLM question-analyzer
    fails to produce a QuestionAnalysis with scenario-type derived metrics.
    Returns an empty list if no scenario pattern is found.
    """
    q = query.lower()
    for pattern, metric_name, meaning in _SCENARIO_FALLBACK_PATTERNS:
        m = re.search(pattern, q)
        if not m:
            continue

        req: dict[str, Any] = {
            "metric_name": metric_name,
            "metric": _DEFAULT_SCENARIO_METRIC,
            "scenario_aggregation": "sum",
        }

        if meaning == "pct_higher":
            req["scenario_factor"] = 1.0 + float(m.group(1)) / 100.0
        elif meaning == "pct_lower":
            req["scenario_factor"] = 1.0 - float(m.group(1)) / 100.0
        elif meaning == "double":
            req["scenario_factor"] = 2.0
        elif meaning == "half":
            req["scenario_factor"] = 0.5
        elif meaning == "strike":
            req["scenario_factor"] = float(m.group(1))
            req["scenario_volume"] = 1.0
        elif meaning == "offset":
            req["scenario_factor"] = float(m.group(1))
        else:
            continue

        # Try to detect GEL metric from query
        if "gel" in q and "usd" not in q:
            req["metric"] = "p_bal_gel"

        # Extract volume from "X mw" or "X mw/month" if present
        vol_match = _VOLUME_RE.search(q)
        if vol_match and metric_name == "scenario_payoff":
            req["scenario_volume"] = float(vol_match.group(1))

        log.info("Scenario fallback: extracted %s from query (factor=%.2f)", metric_name, req["scenario_factor"])
        return [req]

    return []


def _active_analysis_requests(ctx: QueryContext) -> list[dict[str, Any]]:
    if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active":
        requests = [
            req.model_dump(mode="json")
            for req in ctx.question_analysis.analysis_requirements.derived_metrics
        ]
        if requests:
            return requests
        log.info(
            "QA derived_metrics empty, trying scenario fallback for: %.80s",
            ctx.query,
        )

    # When the LLM question-analyzer didn't produce scenario requests,
    # try heuristic extraction from the raw query text.
    scenario_reqs = _scenario_fallback_requests(ctx.query)
    if scenario_reqs:
        return scenario_reqs

    if ctx.question_analysis is not None:
        log.info(
            "Scenario fallback also empty, using defaults for: %.80s",
            ctx.query,
        )
    return [dict(item) for item in DERIVED_METRIC_DEFAULTS]


def _find_yoy_row(df: pd.DataFrame, time_col: str, ts: Optional[pd.Timestamp]) -> pd.DataFrame:
    if ts is None or pd.isna(ts):
        return pd.DataFrame()
    ts = pd.to_datetime(ts)
    yoy_match = df[df[time_col].dt.to_period("M") == (ts - pd.DateOffset(years=1)).to_period("M")]
    if not yoy_match.empty:
        return yoy_match.tail(1)
    return pd.DataFrame()


def _find_historical_month_rows(
    df: pd.DataFrame, time_col: str, ts: Optional[pd.Timestamp], lookback_years: int = 5
) -> pd.DataFrame:
    """Return rows for the same calendar month from up to ``lookback_years`` previous years."""
    if ts is None or pd.isna(ts):
        return pd.DataFrame()
    ts = pd.to_datetime(ts)
    rows: list[pd.DataFrame] = []
    for offset in range(1, lookback_years + 1):
        candidate_period = (ts - pd.DateOffset(years=offset)).to_period("M")
        match = df[df[time_col].dt.to_period("M") == candidate_period]
        if not match.empty:
            rows.append(match.tail(1))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_historical_month_context(
    historical_rows: pd.DataFrame,
    current_row: pd.DataFrame,
    time_col: str,
    _get_val,
) -> dict[str, Any]:
    """Build 5-year historical context for the same calendar month.

    Returns a dict with price_gel / price_usd stats (min, max, avg, trend)
    and a cross_currency comparison separating real price movement from
    exchange-rate effects.
    """
    if historical_rows.empty:
        return {}

    result: dict[str, Any] = {
        "years_found": len(historical_rows),
        "periods": [str(historical_rows[time_col].iloc[i]) for i in range(len(historical_rows))],
    }

    for metric_key, label in [("p_bal_gel", "gel"), ("p_bal_usd", "usd")]:
        aliases = _metric_aliases(metric_key)
        values: list[float] = []
        periods: list[str] = []
        for i in range(len(historical_rows)):
            row_slice = historical_rows.iloc[[i]]
            val = None
            for alias in aliases:
                if alias in row_slice.columns:
                    v = row_slice[alias].iloc[0]
                    if v is not None and pd.notna(v):
                        try:
                            val = float(v)
                        except (ValueError, TypeError):
                            continue
                        break
            if val is not None:
                values.append(val)
                periods.append(str(row_slice[time_col].iloc[0]))

        if not values:
            continue

        cur_val = _get_val(current_row, aliases)
        stats: dict[str, Any] = {
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "avg": round(sum(values) / len(values), 2),
            "observations": len(values),
            "values_by_year": dict(zip(periods, [round(v, 2) for v in values])),
        }

        # Trend: linear slope over historical observations (oldest → newest)
        if len(values) >= 3:
            x = np.arange(len(values), dtype=float)
            slope = float(np.polyfit(x, values, deg=1)[0])
            stats["trend_slope_per_year"] = round(slope, 2)
            # Use relative threshold (2% of average) so the label works for
            # both GEL (~5-15) and USD (~2-5) price scales.
            threshold = stats["avg"] * 0.02 if stats["avg"] > 0 else 0.5
            stats["trend_direction"] = "rising" if slope > threshold else ("falling" if slope < -threshold else "stable")

        # Current month's position in historical range
        if cur_val is not None:
            stats["current_value"] = round(cur_val, 2)
            range_size = stats["max"] - stats["min"]
            if range_size > 0:
                percentile = (cur_val - stats["min"]) / range_size * 100
                stats["current_percentile_in_range"] = round(percentile, 1)
            if cur_val > stats["max"]:
                stats["current_vs_history"] = "above_historical_max"
            elif cur_val < stats["min"]:
                stats["current_vs_history"] = "below_historical_min"
            else:
                stats["current_vs_history"] = "within_range"

        result[f"price_{label}"] = stats

    # Cross-currency comparison: separate real price movement from xrate effect
    gel_stats = result.get("price_gel")
    usd_stats = result.get("price_usd")
    if (
        gel_stats and usd_stats
        and "current_value" in gel_stats
        and "current_value" in usd_stats
    ):
        gel_avg, usd_avg = gel_stats["avg"], usd_stats["avg"]
        gel_cur, usd_cur = gel_stats["current_value"], usd_stats["current_value"]
        if gel_avg > 0 and usd_avg > 0:
            gel_change_pct = (gel_cur - gel_avg) / gel_avg * 100
            usd_change_pct = (usd_cur - usd_avg) / usd_avg * 100
            result["cross_currency"] = {
                "gel_vs_5yr_avg_pct": round(gel_change_pct, 1),
                "usd_vs_5yr_avg_pct": round(usd_change_pct, 1),
                "currency_effect_pct": round(gel_change_pct - usd_change_pct, 1),
                "note": (
                    "If GEL change >> USD change, the difference is currency depreciation effect. "
                    "If both change similarly, the price movement is real (not currency-driven)."
                ),
            }

    return result


def _build_requested_analysis_evidence(
    ctx: QueryContext,
    df: pd.DataFrame,
    time_col: str,
    current_ts: Optional[pd.Timestamp],
    current_row: pd.DataFrame,
    previous_ts: Optional[pd.Timestamp],
    previous_row: pd.DataFrame,
    cur_shares: dict[str, float],
    prev_shares: dict[str, float],
) -> pd.DataFrame:
    requests = _active_analysis_requests(ctx)
    if not requests:
        return pd.DataFrame()

    yoy_row = _find_yoy_row(df, time_col, current_ts)
    yoy_ts = pd.to_datetime(yoy_row[time_col].iloc[0], errors="coerce") if not yoy_row.empty else None
    yoy_shares: dict[str, float] = {}
    if not yoy_row.empty:
        for col in [c for c in yoy_row.columns if c.startswith("share_")]:
            val = yoy_row[col].iloc[0]
            if pd.notna(val):
                try:
                    yoy_shares[col] = float(val)
                except (ValueError, TypeError):
                    pass

    mctx = MetricContext(
        df=df,
        time_col=time_col,
        current_ts=current_ts,
        current_row=current_row,
        previous_ts=previous_ts,
        previous_row=previous_row,
        cur_shares=cur_shares,
        prev_shares=prev_shares,
        yoy_row=yoy_row,
        yoy_ts=yoy_ts,
        yoy_shares=yoy_shares,
        correlation_results=ctx.correlation_results,
    )

    evidence_rows: list[dict[str, Any]] = []
    for request in requests:
        metric_name = str(request.get("metric_name", "")).strip()
        metric = str(request.get("metric", "")).strip()
        target_metric = str(request.get("target_metric", "")).strip() or None
        if not metric_name or not metric:
            continue

        record: dict[str, Any] = {
            "record_type": "derived",
            "derived_metric_name": metric_name,
            "metric": metric,
            "target_metric": target_metric,
            "period": str(current_ts) if current_ts is not None and not pd.isna(current_ts) else None,
            "comparison_period": None,
            "current_value": None,
            "previous_value": None,
            "absolute_change": None,
            "percent_change": None,
            "correlation_value": None,
            "trend_slope": None,
            "formula": "",
        }

        result = dispatch_metric(request, record, mctx)
        if result is not None:
            evidence_rows.append(result)

    return pd.DataFrame(evidence_rows)


def _find_chart_time_column(df: pd.DataFrame) -> Optional[str]:
    """Return the most likely time column for semantic chart overrides."""
    if df.empty:
        return None
    return next(
        (
            col
            for col in df.columns
            if any(
                hint in col.lower()
                for hint in ["date", "year", "month", "period", "თვე", "წელი", "თარიღი"]
            )
        ),
        None,
    )


def _format_chart_time_values(values: pd.Series) -> list[str]:
    """Format time values consistently with Stage 5 chart output."""
    if values.empty:
        return []

    raw_values = values.reset_index(drop=True)
    first_val = raw_values.iloc[0]
    dt_values: pd.Series
    try:
        if isinstance(first_val, (int, float)) or str(type(first_val).__name__) == "Decimal":
            first_num = float(first_val)
            if 1900 <= first_num <= 2100:
                dt_values = pd.to_datetime(raw_values.astype(int), format="%Y", errors="coerce")
            else:
                dt_values = pd.to_datetime(raw_values, errors="coerce")
        else:
            dt_values = pd.to_datetime(raw_values, errors="coerce")
    except Exception:
        return raw_values.astype(str).tolist()

    if dt_values.isna().all():
        return raw_values.astype(str).tolist()

    all_first_of_month = dt_values.dt.day.eq(1).all()
    all_january = all_first_of_month and dt_values.dt.month.eq(1).all()
    if all_january and len(dt_values) > 1:
        return dt_values.dt.strftime("%Y").tolist()
    if all_first_of_month:
        return dt_values.dt.strftime("%Y-%m").tolist()
    return dt_values.dt.strftime("%Y-%m-%d").tolist()


def _price_unit_for_metric(metric: str) -> str:
    metric_lower = metric.lower()
    if "_usd" in metric_lower:
        return "USD/MWh"
    if "_gel" in metric_lower:
        return "GEL/MWh"
    if metric_lower == "xrate":
        return "GEL per USD"
    if metric_lower.startswith("share_"):
        return "Share (0-1)"
    return "Value"


def _amount_unit_for_metric(metric: str) -> str:
    metric_lower = metric.lower()
    if "_usd" in metric_lower:
        return "USD"
    if "_gel" in metric_lower:
        return "GEL"
    return "Value"


def _metric_label(metric: str) -> str:
    from context import COLUMN_LABELS

    return COLUMN_LABELS.get(metric, metric.replace("_", " ").title())


def _scenario_chart_request(
    ctx: QueryContext,
) -> Optional[Tuple[ChartIntent, list[SemanticRole], dict[str, Any], dict[str, Any]]]:
    """Return the single supported scenario request/evidence pair for chart overrides."""
    qa = ctx.question_analysis
    scenario_rows = [
        row
        for row in (ctx.analysis_evidence or [])
        if row.get("record_type") == "scenario"
    ]
    if len(scenario_rows) != 1:
        return None
    scenario_row = scenario_rows[0]

    request = {
        "metric_name": scenario_row.get("derived_metric_name"),
        "metric": scenario_row.get("metric"),
        "scenario_factor": scenario_row.get("scenario_factor"),
        "scenario_volume": scenario_row.get("scenario_volume"),
        "scenario_aggregation": scenario_row.get("scenario_aggregation"),
    }

    chart_intent: Optional[ChartIntent] = None
    target_roles: list[SemanticRole] = []
    if qa is not None:
        vis = qa.visualization
        if vis.chart_intent is not None and len(vis.target_series) >= 2:
            chart_intent = vis.chart_intent
            target_roles = list(vis.target_series)

    if chart_intent is None:
        chart_intent, target_roles = _default_scenario_chart_hint(ctx, request, scenario_row)
        if chart_intent is None or len(target_roles) < 2:
            return None

    return chart_intent, target_roles, request, scenario_row


def _default_scenario_chart_hint(
    ctx: QueryContext,
    request: dict[str, Any],
    scenario_row: dict[str, Any],
) -> Tuple[Optional[ChartIntent], list[SemanticRole]]:
    """Infer a deterministic default chart hint for supported scenario queries."""
    metric_name = str(request.get("metric_name", "")).strip()
    query_lower = (ctx.query or "").lower()

    if metric_name in {"scenario_scale", "scenario_offset"}:
        return ChartIntent.TREND_COMPARE, [SemanticRole.OBSERVED, SemanticRole.DERIVED]

    if metric_name != "scenario_payoff":
        return None, []

    has_total_income_signals = any(signal in query_lower for signal in _SCENARIO_TOTAL_INCOME_QUERY_SIGNALS)
    has_component_signals = any(signal in query_lower for signal in _SCENARIO_COMPONENT_QUERY_SIGNALS)
    has_reference_signals = any(signal in query_lower for signal in _SCENARIO_REFERENCE_QUERY_SIGNALS)
    has_components = (
        scenario_row.get("market_component_aggregate") is not None
        and scenario_row.get("combined_total_aggregate") is not None
    )

    if has_components and has_total_income_signals and has_component_signals:
        return ChartIntent.DECOMPOSITION, [
            SemanticRole.COMPONENT_PRIMARY,
            SemanticRole.COMPONENT_SECONDARY,
        ]

    if has_reference_signals:
        return ChartIntent.TREND_COMPARE, [SemanticRole.OBSERVED, SemanticRole.REFERENCE]

    return ChartIntent.TREND_COMPARE, [SemanticRole.OBSERVED, SemanticRole.REFERENCE]


def _resolve_chart_roles(ctx: QueryContext) -> Optional[Dict[str, Any]]:
    """Resolve semantic chart roles into chart-ready deterministic series."""
    resolved_request = _scenario_chart_request(ctx)
    if resolved_request is None or ctx.df.empty:
        return None

    chart_intent, target_roles, request, evidence_row = resolved_request
    time_col = _find_chart_time_column(ctx.df)
    if not time_col or time_col not in ctx.df.columns:
        return None

    metric = str(request.get("metric", "")).strip()
    value_col = next((alias for alias in _metric_aliases(metric) if alias in ctx.df.columns), None)
    if value_col is None:
        return None

    raw = pd.to_numeric(ctx.df[value_col], errors="coerce")
    valid_mask = raw.notna()
    if valid_mask.sum() == 0:
        return None

    base_series = raw[valid_mask].astype(float).reset_index(drop=True)
    time_values = _format_chart_time_values(ctx.df.loc[valid_mask, time_col])
    if not time_values or len(time_values) != len(base_series):
        return None

    metric_name = str(request.get("metric_name", "")).strip()
    factor = float(request.get("scenario_factor") or evidence_row.get("scenario_factor") or 0.0)
    volume = float(request.get("scenario_volume") or evidence_row.get("scenario_volume") or 1.0)
    base_label = _metric_label(value_col)

    if chart_intent == ChartIntent.TREND_COMPARE:
        resolved_series: list[dict[str, Any]] = []
        units: list[str] = []

        for role in target_roles:
            if role == SemanticRole.OBSERVED:
                resolved_series.append(
                    {
                        "role": role.value,
                        "label": base_label,
                        "values": base_series.round(6).tolist(),
                        "unit": _price_unit_for_metric(value_col),
                    }
                )
                units.append(_price_unit_for_metric(value_col))
            elif role == SemanticRole.REFERENCE:
                if metric_name != "scenario_payoff":
                    return None
                resolved_series.append(
                    {
                        "role": role.value,
                        "label": "Strike Price",
                        "values": [round(factor, 6)] * len(base_series),
                        "unit": _price_unit_for_metric(value_col),
                    }
                )
                units.append(_price_unit_for_metric(value_col))
            elif role == SemanticRole.DERIVED:
                if metric_name == "scenario_scale":
                    values = (base_series * factor).round(6).tolist()
                    label = f"Scaled {base_label}"
                    unit = _price_unit_for_metric(value_col)
                elif metric_name == "scenario_offset":
                    values = (base_series + factor).round(6).tolist()
                    label = f"Adjusted {base_label}"
                    unit = _price_unit_for_metric(value_col)
                elif metric_name == "scenario_payoff":
                    values = ((factor - base_series) * volume).round(6).tolist()
                    label = "CfD Financial Compensation"
                    unit = _amount_unit_for_metric(value_col)
                else:
                    return None
                resolved_series.append(
                    {
                        "role": role.value,
                        "label": label,
                        "values": values,
                        "unit": unit,
                    }
                )
                units.append(unit)
            else:
                return None

        if not resolved_series or len(set(units)) != 1:
            return None

        return {
            "intent": chart_intent.value,
            "time_field": "date",
            "x_axis_title": time_col,
            "time_values": time_values,
            "series": resolved_series,
            "unit": units[0],
        }

    if chart_intent == ChartIntent.DECOMPOSITION:
        if metric_name != "scenario_payoff":
            return None

        component_map = {
            SemanticRole.COMPONENT_PRIMARY: {
                "label": "Balancing Market Sales Income",
                "values": (base_series * volume).round(6).tolist(),
            },
            SemanticRole.COMPONENT_SECONDARY: {
                "label": "CfD Financial Compensation",
                "values": ((factor - base_series) * volume).round(6).tolist(),
            },
        }

        resolved_series = []
        for role in target_roles:
            component = component_map.get(role)
            if component is None:
                return None
            resolved_series.append(
                {
                    "role": role.value,
                    "label": component["label"],
                    "values": component["values"],
                    "unit": _amount_unit_for_metric(value_col),
                }
            )

        return {
            "intent": chart_intent.value,
            "time_field": "date",
            "x_axis_title": "date",
            "time_values": time_values,
            "series": resolved_series,
            "unit": _amount_unit_for_metric(value_col),
        }

    return None


def _build_chart_override(ctx: QueryContext, resolved_roles: Dict[str, Any]) -> Optional[Tuple[list[dict[str, Any]], str, dict[str, Any]]]:
    """Build chart override payloads from resolved semantic roles."""
    time_field = resolved_roles["time_field"]
    time_values = resolved_roles["time_values"]
    series = resolved_roles["series"]
    labels = [item["label"] for item in series]

    if resolved_roles["intent"] == ChartIntent.TREND_COMPARE.value:
        chart_rows: list[dict[str, Any]] = []
        for idx, time_value in enumerate(time_values):
            row: dict[str, Any] = {time_field: time_value}
            for item in series:
                row[item["label"]] = item["values"][idx]
            chart_rows.append(row)

        title = " vs ".join(labels) if len(labels) <= 2 else f"Trend Comparison: {' vs '.join(labels)}"
        meta = {
            "xAxisTitle": resolved_roles["x_axis_title"],
            "yAxisTitle": resolved_roles["unit"],
            "title": title,
            "axisMode": "single",
            "labels": labels,
        }
        return chart_rows, "line", meta

    if resolved_roles["intent"] == ChartIntent.DECOMPOSITION.value:
        chart_rows = []
        for idx, time_value in enumerate(time_values):
            for item in series:
                chart_rows.append(
                    {
                        "date": time_value,
                        "category": item["label"],
                        "value": item["values"][idx],
                    }
                )

        meta = {
            "xAxisTitle": resolved_roles["x_axis_title"],
            "yAxisTitle": resolved_roles["unit"],
            "title": "Derived Component Breakdown",
            "axisMode": "single",
            "labels": labels,
        }
        return chart_rows, "stackedbar", meta

    return None


def _materialize_chart_override(ctx: QueryContext) -> None:
    """Build a deterministic chart override when semantic chart hints are fully satisfiable."""
    ctx.chart_override_data = None
    ctx.chart_override_type = None
    ctx.chart_override_meta = None

    resolved_roles = _resolve_chart_roles(ctx)
    if resolved_roles is None:
        return

    override = _build_chart_override(ctx, resolved_roles)
    if override is None:
        return

    chart_data, chart_type, chart_meta = override
    ctx.chart_override_data = chart_data
    ctx.chart_override_type = chart_type
    ctx.chart_override_meta = chart_meta
    log.info(
        "📊 Derived chart override prepared (intent=%s, type=%s, rows=%d)",
        resolved_roles["intent"],
        chart_type,
        len(chart_data),
    )


def _build_why_provenance_snapshot(
    current_ts: Optional[pd.Timestamp],
    previous_ts: Optional[pd.Timestamp],
    *,
    cur_gel: Optional[float],
    prev_gel: Optional[float],
    cur_usd: Optional[float],
    prev_usd: Optional[float],
    cur_xrate: Optional[float],
    prev_xrate: Optional[float],
    cur_shares: dict[str, float],
    prev_shares: dict[str, float],
) -> pd.DataFrame:
    share_cols = sorted(set(cur_shares) | set(prev_shares))
    cols = ["date", "p_bal_gel", "p_bal_usd", "xrate"] + share_cols
    records: List[Dict[str, Any]] = []

    def _build_record(
        ts: Optional[pd.Timestamp],
        gel: Optional[float],
        usd: Optional[float],
        xrate: Optional[float],
        shares: dict[str, float],
    ) -> None:
        if ts is None or pd.isna(ts):
            return
        record: Dict[str, Any] = {
            "date": pd.to_datetime(ts),
            "p_bal_gel": gel,
            "p_bal_usd": usd,
            "xrate": xrate,
        }
        for key in share_cols:
            record[key] = shares.get(key)
        records.append(record)

    _build_record(previous_ts, prev_gel, prev_usd, prev_xrate, prev_shares)
    _build_record(current_ts, cur_gel, cur_usd, cur_xrate, cur_shares)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)[cols]


def _parse_period_hint(period_hint: str, user_query: str) -> Optional[pd.Period]:
    """Derive a pandas Period (monthly or yearly) from the LLM plan or the raw query."""
    if not period_hint:
        return None
    period_hint = str(period_hint).strip()

    # Try YYYY-MM format
    m = re.match(r"^(\d{4})-(\d{1,2})$", period_hint)
    if m:
        try:
            return pd.Period(f"{m.group(1)}-{m.group(2)}", freq="M")
        except Exception:
            pass

    # Try YYYY format
    m = re.match(r"^(\d{4})$", period_hint)
    if m:
        try:
            return pd.Period(m.group(1), freq="Y")
        except Exception:
            pass

    # Try month name + year from query
    for month_name, month_num in MONTH_NAME_TO_NUMBER.items():
        if month_name in user_query.lower():
            years = re.findall(r"(20\d{2})", user_query)
            if years:
                try:
                    return pd.Period(f"{years[0]}-{month_num:02d}", freq="M")
                except Exception:
                    pass
    return None


def _select_share_column(share_cols: list[str], target_text: str) -> Optional[str]:
    """Choose the most relevant share column based on the user's target description."""
    target_lower = target_text.lower()

    # Direct entity matches
    priority_map = {
        "import": "share_import",
        "renewable_ppa": "share_renewable_ppa",
        "thermal_ppa": "share_thermal_ppa",
        "deregulated_hydro": "share_deregulated_hydro",
        "regulated_hpp": "share_regulated_hpp",
        "all_ppa": "share_all_ppa",
        "ppa": "share_all_ppa",
        "all_renewables": "share_all_renewables",
        "renewable": "share_all_renewables",
        "total_hpp": "share_total_hpp",
        "hydro": "share_total_hpp",
    }

    for keyword, col_name in priority_map.items():
        if keyword in target_lower and col_name in share_cols:
            return col_name

    return share_cols[0] if share_cols else None


def generate_share_summary(df: pd.DataFrame, plan: Dict[str, Any], user_query: str) -> Optional[str]:
    """Produce a deterministic textual answer for share queries to avoid LLM hallucinations."""
    if df is None or df.empty:
        return None

    share_cols = [c for c in df.columns if c.startswith("share_")]
    if not share_cols:
        return None

    target_text = str(plan.get("target", "")) + " " + user_query
    period_hint = str(plan.get("period", ""))
    period = _parse_period_hint(period_hint, user_query)

    selected_col = _select_share_column(share_cols, target_text)
    if not selected_col:
        return None

    # Find the row matching the period
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time_month"])]
    if date_cols:
        date_col = date_cols[0]
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        if period:
            if period.freq and str(period.freq) == "M":
                mask = (df[date_col].dt.year == period.year) & (df[date_col].dt.month == period.month)
                filtered = df[mask]
            else:
                mask = df[date_col].dt.year == period.year
                filtered = df[mask]
        else:
            filtered = df.tail(1)
    else:
        filtered = df.tail(1)

    if filtered.empty:
        filtered = df.tail(1)

    row = filtered.iloc[-1]
    value = row.get(selected_col)
    if value is None or pd.isna(value):
        return None

    value_pct = float(value)
    if value_pct < 1:
        value_pct *= 100

    meta = BALANCING_SHARE_METADATA.get(selected_col, {})
    label = meta.get("label", selected_col.replace("share_", "").replace("_", " "))

    # Format period
    if date_cols and date_cols[0] in filtered.columns:
        ts = pd.to_datetime(filtered.iloc[-1][date_cols[0]])
        period_str = ts.strftime("%B %Y")
    elif period:
        period_str = str(period)
    else:
        period_str = "latest available period"

    summary_parts = [f"**{label.title()}** accounted for **{value_pct:.1f}%** of balancing electricity in {period_str}."]

    # Add breakdown for aggregate columns
    if selected_col == "share_all_ppa":
        renewable = row.get("share_renewable_ppa")
        thermal = row.get("share_thermal_ppa")
        if renewable is not None and pd.notna(renewable) and thermal is not None and pd.notna(thermal):
            r_pct = float(renewable) * 100 if float(renewable) < 1 else float(renewable)
            t_pct = float(thermal) * 100 if float(thermal) < 1 else float(thermal)
            summary_parts.append(f"  - Renewable PPA: {r_pct:.1f}%")
            summary_parts.append(f"  - Thermal PPA: {t_pct:.1f}%")

    return "\n".join(summary_parts)


# ---------------------------------------------------------------------------
# Forecast helpers (moved from ask_post inner functions)
# ---------------------------------------------------------------------------

def _detect_forecast_mode(text_: str) -> bool:
    keys = ["forecast", "predict", "projection", "project", "future", "next year", "estimate", "estimation", "outlook"]
    t = text_.lower()
    return any(k in t for k in keys)


def _detect_why_mode(text_: str) -> bool:
    """Detect if the query requires causal/driver analysis."""
    keys = [
        # English
        "why", "reason", "cause", "factor", "explain", "due to", "behind",
        "what caused", "what influenced", "driver", "impact", "influence",
        "relationship", "correlation", "depend", "determinant",
        # Georgian
        "რატომ", "მიზეზი", "ფაქტორი", "ახსენი", "გავლენა", "დრაივერი",
        # Russian
        "почему", "причина", "фактор", "объясни", "влияние", "драйвер"
    ]
    t = text_.lower()
    return any(k in t for k in keys)


def _extract_forecast_horizon(query: str) -> int:
    """Extract forecast duration (in years) from user query."""
    q = query.lower()
    # Handle patterns like "10 year", "10-year", "10 years"
    match = re.search(r"(\d+)\s*-?year", q)
    if match:
        val = int(match.group(1))
        return min(max(val, 1), 20)  # Clamp between 1 and 20 years

    # Handle "next decade"
    if "decade" in q:
        return 10

    # Default to 3 years if not specified
    return 3


def _month_from_text(s: str) -> Optional[int]:
    months = {
        # English
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        # Georgian (prefix-matched)
        "იანვ": 1, "თებ": 2, "მარტ": 3, "აპრ": 4, "მაის": 5, "ივნ": 6,
        "ივლ": 7, "აგვ": 8, "სექტ": 9, "ოქტ": 10, "ნოემ": 11, "დეკ": 12,
        # Russian (prefix-matched)
        "янв": 1, "фев": 2, "мар": 3, "апр": 4, "мая": 5, "май": 5, "июн": 6,
        "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
    }
    s_lower = s.lower()
    for k, v in months.items():
        if k in s_lower:
            return v
    return None


def _choose_target_for_forecast(df_in: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Return (time_col, value_col) for forecasting."""
    time_candidates = [c for c in df_in.columns if any(k in c.lower() for k in ["date", "year", "month"])]
    time_col = time_candidates[0] if time_candidates else None
    for c in df_in.columns:
        if c.lower() in ["p_bal_usd", "p_bal_gel"]:
            return time_col, c
    for c in df_in.columns:
        if any(k in c.lower() for k in ["price", "tariff", "p_bal"]):
            return time_col, c
    for c in df_in.columns:
        if any(k in c.lower() for k in ["quantity_tech", "quantity", "volume_tj", "generation", "demand"]):
            return time_col, c
    for c in df_in.columns:
        if pd.api.types.is_numeric_dtype(df_in[c]):
            return time_col, c
    return time_col, None


def _detect_data_type(value_col: str) -> str:
    """Classify column into 'price', 'quantity', or 'other'."""
    c = value_col.lower()
    if any(k in c for k in ["p_bal", "price", "tariff"]):
        return "price"
    if any(k in c for k in ["quantity", "volume_tj", "demand", "generation"]):
        return "quantity"
    return "other"


def _generate_cagr_forecast(df_in: pd.DataFrame, user_query: str) -> Tuple[pd.DataFrame, str]:
    """Generate CAGR-based forecast for price or quantity data."""
    df = df_in.copy()
    time_col, value_col = _choose_target_for_forecast(df)
    if not time_col or not value_col:
        return df_in, "Forecast skipped: no clear time/value columns."

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return df_in, "Forecast skipped: no numeric data."

    data_type = _detect_data_type(value_col)
    note_parts = []

    if data_type == "quantity":
        df["year"] = df[time_col].dt.year
        df_y = df.groupby("year")[value_col].sum().reset_index()
        if len(df_y) < 2:
            return df_in, "Forecast skipped: insufficient yearly data."
        first, last = df_y.iloc[0], df_y.iloc[-1]
        span = last["year"] - first["year"]
        if span <= 0 or first[value_col] <= 0:
            return df_in, "Invalid data for CAGR."
        cagr = (last[value_col] / first[value_col]) ** (1 / span) - 1
        note_parts.append(f"Yearly CAGR={cagr*100:.2f}% ({int(first['year'])}→{int(last['year'])}).")
        horizon = _extract_forecast_horizon(user_query)
        yrs_in_q = re.findall(r"(20\d{2})", user_query)
        target_years = sorted({int(y) for y in yrs_in_q if int(y) > last["year"]}) or [int(last["year"]) + i for i in range(1, horizon + 1)]
        f_rows = []
        for y in target_years:
            val = last[value_col] * ((1 + cagr) ** (y - last["year"]))
            # Force year to int to avoid 2026.0-01-01 errors
            f_rows.append({time_col: pd.to_datetime(f"{int(y)}-01-01"), value_col: val, "is_forecast": True})
        if "is_forecast" not in df.columns:
            df["is_forecast"] = False
        df_f = pd.concat([df, pd.DataFrame(f_rows)], ignore_index=True)
        note_parts.append(f"Forecast years: {', '.join(map(str, target_years))}.")
        return df_f, " ".join(note_parts)

    elif data_type == "price":
        df["year"] = df[time_col].dt.year
        df["month"] = df[time_col].dt.month
        df["season"] = np.where(df["month"].isin(SUMMER_MONTHS), "summer", "winter")

        df_y = df.groupby("year")[value_col].mean().reset_index()
        first, last = df_y.iloc[0], df_y.iloc[-1]
        span = last["year"] - first["year"]

        if span > 0 and first[value_col] > 0 and last[value_col] > 0:
            cagr_y = (last[value_col] / first[value_col]) ** (1 / span) - 1
        else:
            cagr_y = 0

        df_s = df.groupby(["year", "season"])[value_col].mean().reset_index()
        summer = df_s[df_s["season"] == "summer"]
        winter = df_s[df_s["season"] == "winter"]

        if len(summer) >= 2:
            s_first, s_last = summer[value_col].iloc[0], summer[value_col].iloc[-1]
            s_span = summer["year"].iloc[-1] - summer["year"].iloc[0]
            cagr_s = (s_last / s_first) ** (1 / s_span) - 1 if s_span > 0 and s_first > 0 and s_last > 0 else np.nan
        else:
            cagr_s = np.nan

        if len(winter) >= 2:
            w_first, w_last = winter[value_col].iloc[0], winter[value_col].iloc[-1]
            w_span = winter["year"].iloc[-1] - winter["year"].iloc[0]
            cagr_w = (w_last / w_first) ** (1 / w_span) - 1 if w_span > 0 and w_first > 0 and w_last > 0 else np.nan
        else:
            cagr_w = np.nan

        def format_cagr(cagr_val):
            return f"{cagr_val*100:.2f}" if not np.isnan(cagr_val) else "N/A"

        note_parts.append(f"Yearly CAGR={format_cagr(cagr_y)}%, Summer={format_cagr(cagr_s)}%, Winter={format_cagr(cagr_w)}%.")

        horizon = _extract_forecast_horizon(user_query)
        yrs_in_q = re.findall(r"(20\d{2})", user_query)
        target_years = sorted({int(y) for y in yrs_in_q if int(y) > last["year"]}) or [int(last["year"]) + i for i in range(1, horizon + 1)]

        f_rows = []
        for y in target_years:
            val_y = last[value_col] * ((1 + cagr_y) ** (y - last["year"]))
            val_s = last[value_col] * ((1 + cagr_s) ** (y - last["year"])) if not np.isnan(cagr_s) else val_y
            val_w = last[value_col] * ((1 + cagr_w) ** (y - last["year"])) if not np.isnan(cagr_w) else val_y
            # Force year to int
            f_rows.append({time_col: pd.to_datetime(f"{int(y)}-04-01"), "season": "summer", value_col: val_s, "is_forecast": True})
            f_rows.append({time_col: pd.to_datetime(f"{int(y)}-12-01"), "season": "winter", value_col: val_w, "is_forecast": True})

        if "is_forecast" not in df.columns:
            df["is_forecast"] = False
        df_f = pd.concat([df, pd.DataFrame(f_rows)], ignore_index=True)
        note_parts.append(f"Forecast years: {', '.join(map(str, target_years))}.")
        return df_f, " ".join(note_parts)

    else:
        return df_in, "Forecast skipped: unrecognized data type."


# ---------------------------------------------------------------------------
# Main pipeline stage
# ---------------------------------------------------------------------------

def enrich(ctx: QueryContext) -> QueryContext:
    """Stage 3: Enrich query results with statistics, shares, correlations, forecasts.

    Reads: ctx.df, ctx.rows, ctx.cols, ctx.plan, ctx.query
    Writes: ctx.preview, ctx.stats_hint, ctx.share_summary_override,
            ctx.correlation_results, ctx.df (possibly enriched), ctx.add_trendlines,
            ctx.trendline_extend_to
    """
    # --- Share resolution ---
    share_intent = str(ctx.plan.get("intent", "")).lower()
    if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active":
        # When the semantic contract is available, trust only structured analyzer
        # signals for share/composition intent. Free-form intent text is not
        # authoritative enough for downstream execution.
        analyzer_share_signal = ctx.analyzer_indicates_share_intent
        share_query_detected = (
            share_intent in {"calculate_share", "share"} or analyzer_share_signal
        )
    else:
        # No analyzer available: fall back to keyword matching (legacy behavior)
        share_query_detected = share_intent in {"calculate_share", "share"} or "share" in ctx.query.lower()
    share_df_for_summary = ctx.df

    if share_query_detected:
        try:
            with ENGINE.connect() as conn:
                conn.execute(text("SET TRANSACTION READ ONLY"))
                resolved_df, used_fallback = ensure_share_dataframe(ctx.df, conn)
            if used_fallback:
                log.warning("🔄 Share query lacked usable rows — using deterministic balancing share pivot.")
                ctx.df = resolved_df
                share_df_for_summary = resolved_df
                ctx.cols = list(resolved_df.columns)
                ctx.rows = [tuple(r) for r in resolved_df.itertuples(index=False, name=None)]
                stamp_provenance(
                    ctx,
                    ctx.cols,
                    ctx.rows,
                    source="sql",
                    query_hash=sql_query_hash(BALANCING_SHARE_PIVOT_SQL),
                )
            else:
                share_df_for_summary = resolved_df
        except Exception as fallback_err:
            log.warning(f"Share pivot resolution failed: {fallback_err}")

    # --- Apply labels and compute preview/stats ---
    from context import COLUMN_LABELS, DERIVED_LABELS
    _all_labels = {**COLUMN_LABELS, **DERIVED_LABELS}
    cols_labeled = [_all_labels.get(c, c) for c in ctx.cols]
    ctx.preview = rows_to_preview(ctx.rows, cols_labeled)
    ctx.stats_hint = quick_stats(ctx.rows, cols_labeled)
    _append_column_aggregates(ctx)

    # --- Seasonal stats ---
    timeseries_info = detect_monthly_timeseries(ctx.df)
    if timeseries_info:
        time_col, value_col = timeseries_info
        try:
            seasonal_stats = calculate_seasonal_stats(ctx.df, time_col, value_col)
            seasonal_text = format_seasonal_stats(seasonal_stats)
            ctx.stats_hint += f"\n\n{seasonal_text}"
            log.info("✅ Added seasonal-adjusted statistics to stats_hint")
        except Exception as e:
            log.warning(f"⚠️ Seasonal stats calculation failed: {e}")

    # --- Share summary override ---
    # Evidence precedence: do not generate share override if a non-share tool result
    # already exists. A get_prices result aligned with a price-comparison semantic
    # target should not be displaced by a share override.
    if share_query_detected and ctx.semantic_locked:
        has_non_share_result = (
            not ctx.df.empty
            and ctx.used_tool
            and ctx.tool_name != "get_balancing_composition"
        )
        if has_non_share_result:
            log.info(
                "Skipping share_summary_override: non-share tool %s result "
                "already exists (semantic_locked)",
                ctx.tool_name,
            )
            share_query_detected = False

    if share_query_detected:
        try:
            ctx.share_summary_override = generate_share_summary(share_df_for_summary, ctx.plan, ctx.query)
            if ctx.share_summary_override:
                log.info("✅ Generated deterministic share summary override.")
        except Exception as share_err:
            log.warning(f"Share summary override failed: {share_err}")

    # --- Correlation analysis ---
    if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active":
        # Use structured analyzer signals for correlation detection
        qa_reqs = ctx.question_analysis.analysis_requirements
        if qa_reqs.needs_driver_analysis or qa_reqs.needs_correlation_context:
            log.info("🧮 Semantic intent → correlation (analyzer: needs_driver=%s needs_correlation=%s).",
                     qa_reqs.needs_driver_analysis, qa_reqs.needs_correlation_context)
            ctx.plan["intent"] = "correlation"
    else:
        # Legacy keyword-based correlation detection (no analyzer available)
        user_text = ctx.query.lower().strip()
        intent_text = str(ctx.plan.get("intent", "")).lower()
        combined_text = f"{intent_text} {user_text}"

        driver_keywords = [
            "driver", "cause", "effect", "factor", "reason", "impact", "influence",
            "relationship", "correlation", "depend", "why", "behind", "due to",
            "explain", "determinant", "driven by", "lead to", "affect", "because",
            "based on", "results in", "responsible for"
        ]
        causal_patterns = [
            r"what.*cause", r"what.*affect", r"why.*change", r"why.*increase",
            r"factors?.*behind", r"factors?.*influenc", r"reason.*for",
            r"cause.*of", r"impact.*on", r"driv.*price", r"lead.*to"
        ]

        text_hit = any(k in combined_text for k in driver_keywords)
        pattern_hit = any(re.search(p, combined_text) for p in causal_patterns)

        if text_hit or pattern_hit:
            log.info("🧮 Semantic intent → correlation (detected cause/effect phrasing).")
            ctx.plan["intent"] = "correlation"

    if ctx.plan.get("intent") == "correlation":
        log.info("🔍 Building comprehensive balancing-price correlation analysis")
        try:
            with ENGINE.connect() as conn:
                conn.execute(text("SET TRANSACTION READ ONLY"))
                corr_df = build_balancing_correlation_df(conn)

            allowed_targets = ["p_bal_gel", "p_bal_usd"]
            allowed_drivers = [
                "xrate", "share_import", "share_deregulated_hydro",
                "share_regulated_hpp", "share_renewable_ppa",
                "enguri_tariff_gel", "gardabani_tpp_tariff_gel",
                "grouped_old_tpp_tariff_gel"
            ]
            corr_df = corr_df[[c for c in corr_df.columns if c in (["date"] + allowed_targets + allowed_drivers)]].copy()

            # Overall correlations
            numeric_df = corr_df.drop(columns=["date"], errors="ignore").apply(pd.to_numeric, errors="coerce")
            for target in allowed_targets:
                if target not in numeric_df.columns:
                    continue
                series = numeric_df.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
                if series.notna().any():
                    ctx.correlation_results[target] = series.sort_values(ascending=False).round(3).to_dict()

            # Seasonal correlations
            if "date" in corr_df.columns:
                corr_df["date"] = pd.to_datetime(corr_df["date"], errors="coerce")
                corr_df["month"] = corr_df["date"].dt.month
                summer_df = corr_df[corr_df["month"].isin(SUMMER_MONTHS)].drop(columns=["date", "month"], errors="ignore")
                winter_df = corr_df[~corr_df["month"].isin(SUMMER_MONTHS)].drop(columns=["date", "month"], errors="ignore")

                for label, seasonal_df in {"summer": summer_df, "winter": winter_df}.items():
                    seasonal_numeric = seasonal_df.apply(pd.to_numeric, errors="coerce")
                    for target in allowed_targets:
                        if target in seasonal_numeric.columns and len(seasonal_numeric) > 2:
                            seasonal_corr = seasonal_numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
                            if seasonal_corr.notna().any():
                                ctx.correlation_results[f"{target}_{label}"] = seasonal_corr.sort_values(ascending=False).round(3).to_dict()

            if ctx.correlation_results:
                ctx.stats_hint += "\n\n--- CORRELATION MATRIX (vs Balancing Price) ---\n" + json.dumps(ctx.correlation_results, indent=2)
                log.info(f"✅ Consolidated correlations computed: {list(ctx.correlation_results.keys())}")
            else:
                log.info("⚠️ No valid correlations found")

        except Exception as e:
            log.warning(f"⚠️ Correlation analysis failed: {e}")

    # --- Forecast mode (CAGR) ---
    _forecast_detected = (
        ctx.question_analysis.classification.query_type.value == "forecast"
        if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active"
        else _detect_forecast_mode(ctx.query)
    )
    if _forecast_detected and not ctx.df.empty:
        try:
            ctx.df, _forecast_note = _generate_cagr_forecast(ctx.df, ctx.query)
            ctx.stats_hint += f"\n\n--- FORECAST NOTE ---\n{_forecast_note}"
            log.info(_forecast_note)
        except Exception as _e:
            log.warning(f"Forecast generation failed: {_e}")

    # --- Why mode (causal reasoning) ---
    _why_detected = False
    if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active":
        qa_reqs = ctx.question_analysis.analysis_requirements
        _why_detected = (
            qa_reqs.needs_driver_analysis
            or ctx.question_analysis.classification.query_type.value == "data_explanation"
        )
    else:
        _why_detected = _detect_why_mode(ctx.query)
    if _why_detected and not ctx.df.empty:
        try:
            _build_why_context(ctx)
        except Exception as _e:
            log.warning(f"'Why' reasoning context build failed: {_e}")

    # --- Analyst-mode derived metrics (MoM/YoY without full causal context) ---
    elif not ctx.df.empty and _needs_standalone_analysis(ctx):
        try:
            _build_standalone_analysis_evidence(ctx)
        except Exception as _e:
            log.warning(f"Standalone analysis evidence build failed: {_e}")

    # --- Semantic chart override materialization ---
    if not ctx.df.empty:
        try:
            _materialize_chart_override(ctx)
        except Exception as _e:
            log.warning(f"Chart override materialization failed: {_e}")

    # --- Trendline detection ---
    if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active":
        ctx.add_trendlines = ctx.question_analysis.classification.query_type.value == "forecast"
    else:
        trend_keywords = [
            "trend", "ტრენდი", "тренд", "trending", "forecast", "პროგნოზი", "прогноз",
            "projection", "predict", "future", "მომავალი", "continue", "extrapolate"
        ]
        ctx.add_trendlines = any(keyword in ctx.query.lower() for keyword in trend_keywords)

    if ctx.add_trendlines:
        year_matches = re.findall(r'\b(20[2-9][0-9])\b', ctx.query)
        if year_matches:
            future_year = max(int(year) for year in year_matches)
            ctx.trendline_extend_to = f"{future_year}-12-01"
        else:
            from datetime import datetime
            current_year = datetime.now().year
            horizon = _extract_forecast_horizon(ctx.query)
            ctx.trendline_extend_to = f"{current_year + horizon}-12-01"
        log.info(f"📈 Trendline requested: extending to {ctx.trendline_extend_to}")

        # Pre-calculate trendlines for forecast answer generation
        _precalculate_trendlines(ctx, cols_labeled)

    trace_detail(
        log,
        ctx,
        "stage_3_analyzer_enrich",
        "enrichment_ready",
        preview_len=len(ctx.preview or ""),
        stats_hint_len=len(ctx.stats_hint or ""),
        share_override=bool(ctx.share_summary_override),
        analysis_evidence_count=len(ctx.analysis_evidence or []),
        correlation_keys=list(ctx.correlation_results.keys()),
        add_trendlines=bool(ctx.add_trendlines),
        trendline_extend_to=ctx.trendline_extend_to or "",
        semantic_locked=ctx.semantic_locked,
    )
    return ctx


def _prepare_timeseries_rows(
    ctx: QueryContext,
) -> Optional[Tuple[pd.DataFrame, str, Optional[pd.Timestamp], pd.DataFrame, Optional[pd.Timestamp], pd.DataFrame]]:
    """Extract current/previous rows from a time-series DataFrame.

    Returns ``(df, time_col, current_ts, current_row, previous_ts, previous_row)``
    or ``None`` if the data lacks a usable time column or rows.
    """
    t_series_col = next(
        (c for c in ctx.df.columns if any(k in c.lower() for k in ["date", "year", "month"])),
        None,
    )
    if not t_series_col:
        return None

    df = ctx.df.copy()
    df[t_series_col] = pd.to_datetime(df[t_series_col], errors="coerce")
    df = df.dropna(subset=[t_series_col]).sort_values(t_series_col)
    if df.empty:
        return None

    # Resolve target period from the query text.
    years = [int(y) for y in re.findall(r"(20\d{2})", ctx.query)]
    mon = _month_from_text(ctx.query.lower())
    if mon is None and ctx.question_analysis is not None:
        canonical = getattr(ctx.question_analysis, "canonical_query_en", "") or ""
        if canonical.strip():
            mon = _month_from_text(canonical.lower())
    target_period = pd.Timestamp(years[0], mon or 1, 1) if years else df[t_series_col].iloc[-1]

    cur_row = df.loc[df[t_series_col] == target_period]
    if cur_row.empty:
        cur_row = df[df[t_series_col] <= target_period].tail(1)
    if cur_row.empty:
        return None

    current_ts = pd.to_datetime(cur_row[t_series_col].iloc[0], errors="coerce")
    prev_row = df[df[t_series_col] < cur_row[t_series_col].iloc[0]].tail(1)
    previous_ts = pd.to_datetime(prev_row[t_series_col].iloc[0], errors="coerce") if not prev_row.empty else None

    return df, t_series_col, current_ts, cur_row, previous_ts, prev_row


def _append_column_aggregates(ctx: QueryContext) -> None:
    """Append numeric column aggregates to stats_hint.

    Uses ctx.df directly (original column names, not labels) so we bypass
    the quick_stats labeling issue where "date" → "Period (Year-Month-Day)"
    causes date-column detection to fail.

    Gives the LLM pre-computed sums/means/ranges so it can reference them
    instead of doing row-by-row arithmetic — reducing both response time
    and grounding false positives (aggregates enter the grounding corpus
    via stats_hint).
    """
    if ctx.df is None or ctx.df.empty:
        return
    numeric_cols = ctx.df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return

    from context import COLUMN_LABELS
    lines = [f"\n--- Column Aggregates ({len(ctx.df)} rows) ---"]
    for col in numeric_cols:
        series = ctx.df[col].dropna()
        if series.empty:
            continue
        label = COLUMN_LABELS.get(col, col)
        lines.append(
            f"{label}: sum={series.sum():.4f}, mean={series.mean():.4f}, "
            f"min={series.min():.4f}, max={series.max():.4f}, count={len(series)}"
        )
    if len(lines) > 1:
        ctx.stats_hint += "\n".join(lines)
        log.info("Added column aggregates to stats_hint for %d numeric columns", len(numeric_cols))


def _needs_standalone_analysis(ctx: QueryContext) -> bool:
    """Return True when derived metrics should be computed outside why-mode."""
    qa = ctx.question_analysis
    if qa is not None:
        if qa.analysis_requirements.derived_metrics:
            return True
        if qa.classification.analysis_mode.value == "analyst":
            return True
    # Fallback: heuristic mode detection still triggers analysis
    # even when question_analysis failed validation.
    if ctx.mode == "analyst":
        return True
    return False


def _build_standalone_analysis_evidence(ctx: QueryContext) -> None:
    """Compute derived metrics (MoM/YoY) for non-why analyst-mode queries.

    Populates ``ctx.analysis_evidence`` and appends the evidence block to
    ``ctx.stats_hint`` so that computed values enter the grounding corpus.
    """
    setup = _prepare_timeseries_rows(ctx)
    if setup is None:
        return
    df, time_col, current_ts, current_row, previous_ts, previous_row = setup
    evidence_df = _build_requested_analysis_evidence(
        ctx, df, time_col, current_ts, current_row,
        previous_ts, previous_row,
        cur_shares={}, prev_shares={},
    )
    if evidence_df.empty:
        return
    ctx.analysis_evidence = evidence_df.to_dict(orient="records")
    ctx.stats_hint += (
        "\n\n--- DERIVED ANALYSIS EVIDENCE (TOP 12) ---\n"
        + json.dumps(ctx.analysis_evidence[:12], default=str, indent=2)
    )
    # Stamp provenance so the grounding corpus includes derived values.
    stamp_provenance(
        ctx,
        list(evidence_df.columns),
        [tuple(r) for r in evidence_df.itertuples(index=False, name=None)],
        source=str(ctx.provenance_source or "sql"),
        query_hash=sql_query_hash(f"{ctx.query}|analysis_evidence"),
    )
    log.info(
        "Standalone analysis evidence (%d records) attached to stats_hint.",
        len(ctx.analysis_evidence),
    )


def _build_why_context(ctx: QueryContext) -> None:
    """Build causal context for 'why' queries. Modifies ctx.stats_hint."""
    why_ctx: Dict[str, Any] = {"notes": [], "signals": {}}

    t_series_col = next((c for c in ctx.df.columns if any(k in c.lower() for k in ["date", "year", "month"])), None)
    if not t_series_col:
        return

    df = ctx.df.copy()
    df[t_series_col] = pd.to_datetime(df[t_series_col], errors="coerce")
    df = df.dropna(subset=[t_series_col]).sort_values(t_series_col)

    years = [int(y) for y in re.findall(r"(20\d{2})", ctx.query)]
    mon = _month_from_text(ctx.query.lower())
    # Fallback: try English canonical query from analyzer
    if mon is None and ctx.question_analysis is not None:
        canonical = getattr(ctx.question_analysis, "canonical_query_en", "") or ""
        if canonical.strip():
            mon = _month_from_text(canonical.lower())
    target_period = pd.Timestamp(years[0], mon or 1, 1) if years else df[t_series_col].iloc[-1]

    cur_row = df.loc[df[t_series_col] == target_period]
    if cur_row.empty:
        cur_row = df[df[t_series_col] <= target_period].tail(1)

    if cur_row.empty:
        log.warning("No data found for target period in 'why' analysis")
        return

    prev_row = df[df[t_series_col] < cur_row[t_series_col].iloc[0]].tail(1)
    prev_ts = pd.to_datetime(prev_row[t_series_col].iloc[0], errors="coerce") if not prev_row.empty else None

    def _get_val(row, cols_):
        if row.empty:
            return None
        for c in cols_:
            if c in row.columns:
                val = row[c].iloc[0] if len(row) > 0 else None
                if val is not None and pd.notna(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
        return None

    cur_gel = _get_val(cur_row, _metric_aliases("p_bal_gel"))
    prev_gel = _get_val(prev_row, _metric_aliases("p_bal_gel")) if not prev_row.empty else None
    cur_usd = _get_val(cur_row, _metric_aliases("p_bal_usd"))
    prev_usd = _get_val(prev_row, _metric_aliases("p_bal_usd")) if not prev_row.empty else None
    cur_xrate = _get_val(cur_row, _metric_aliases("xrate"))
    prev_xrate = _get_val(prev_row, _metric_aliases("xrate")) if not prev_row.empty else None

    target_ts = pd.to_datetime(cur_row[t_series_col].iloc[0], errors="coerce") if not cur_row.empty else None

    # YoY: same month from the previous year
    yoy_row = _find_yoy_row(df, t_series_col, target_ts)
    yoy_gel = _get_val(yoy_row, _metric_aliases("p_bal_gel")) if not yoy_row.empty else None
    yoy_usd = _get_val(yoy_row, _metric_aliases("p_bal_usd")) if not yoy_row.empty else None
    yoy_xrate = _get_val(yoy_row, _metric_aliases("xrate")) if not yoy_row.empty else None

    # 5-year historical month context
    historical_rows = _find_historical_month_rows(df, t_series_col, target_ts, lookback_years=5)

    share_cols = [c for c in df.columns if c.startswith("share_")]
    cur_shares: dict[str, float] = {}
    prev_shares: dict[str, float] = {}
    yoy_shares: dict[str, float] = {}

    def _populate_from_frame(frame, dest):
        if frame is None or frame.empty:
            return
        for col in share_cols:
            if col in frame.columns and not frame[col].empty:
                val = frame[col].iloc[0]
                if pd.notna(val):
                    try:
                        dest[col] = float(val)
                    except (ValueError, TypeError):
                        continue

    if share_cols:
        _populate_from_frame(cur_row, cur_shares)
        if not prev_row.empty:
            _populate_from_frame(prev_row, prev_shares)
        if not yoy_row.empty:
            _populate_from_frame(yoy_row, yoy_shares)
    else:
        # Fall back to deterministic panel
        try:
            with ENGINE.connect() as conn:
                conn.execute(text("SET TRANSACTION READ ONLY"))
                share_panel = fetch_balancing_share_panel(conn)
        except Exception:
            share_panel = pd.DataFrame()

        if not share_panel.empty:
            share_panel = share_panel.copy()
            if "segment" in share_panel.columns:
                share_panel = share_panel[share_panel["segment"] == "balancing"]
            share_panel["date"] = pd.to_datetime(share_panel["date"], errors="coerce")
            share_panel = share_panel.dropna(subset=["date"]).sort_values("date")
            share_cols = [c for c in share_panel.columns if c.startswith("share_")]

            def _match_share_row(ts):
                if ts is None or pd.isna(ts):
                    return pd.DataFrame()
                ts = pd.to_datetime(ts)
                exact = share_panel[share_panel["date"] == ts]
                if not exact.empty:
                    return exact.tail(1)
                monthly = share_panel[share_panel["date"].dt.to_period("M") == ts.to_period("M")]
                if not monthly.empty:
                    return monthly.tail(1)
                earlier = share_panel[share_panel["date"] <= ts]
                if not earlier.empty:
                    return earlier.tail(1)
                return pd.DataFrame()

            share_cur = _match_share_row(target_ts)
            if share_cur.empty and not share_panel.empty:
                share_cur = share_panel.tail(1)
            if not share_cur.empty:
                for col in share_cols:
                    val = share_cur[col].iloc[0]
                    if pd.notna(val):
                        try:
                            cur_shares[col] = float(val)
                        except (ValueError, TypeError):
                            continue
                prev_cutoff = share_cur["date"].iloc[0]
                share_prev = share_panel[share_panel["date"] < prev_cutoff].tail(1)
                if share_prev.empty and target_ts is not None:
                    share_prev = share_panel[share_panel["date"] < target_ts].tail(1)
                if not share_prev.empty:
                    for col in share_cols:
                        val = share_prev[col].iloc[0]
                        if pd.notna(val):
                            try:
                                prev_shares[col] = float(val)
                            except (ValueError, TypeError):
                                continue
                # YoY shares from panel
                if target_ts is not None:
                    yoy_share_row = _match_share_row(target_ts - pd.DateOffset(years=1))
                    if not yoy_share_row.empty:
                        for col in share_cols:
                            val = yoy_share_row[col].iloc[0]
                            if pd.notna(val):
                                try:
                                    yoy_shares[col] = float(val)
                                except (ValueError, TypeError):
                                    continue

    # Track whether any share evidence was found at all
    share_data_available = bool(cur_shares or prev_shares)

    deltas = {k: round(cur_shares.get(k, 0) - prev_shares.get(k, 0), 4) for k in cur_shares}

    why_ctx["signals"] = {
        "period": str(cur_row[t_series_col].iloc[0]) if not cur_row.empty else None,
        "p_bal_gel": {"cur": cur_gel, "prev": prev_gel, "yoy": yoy_gel},
        "p_bal_usd": {"cur": cur_usd, "prev": prev_usd, "yoy": yoy_usd},
        "xrate": {"cur": cur_xrate, "prev": prev_xrate, "yoy": yoy_xrate},
        "share_deltas": deltas,
    }

    if cur_shares:
        why_ctx["signals"]["share_snapshot"] = {k: round(v, 4) for k, v in cur_shares.items()}
    if prev_shares:
        why_ctx["signals"]["share_prev_snapshot"] = {k: round(v, 4) for k, v in prev_shares.items()}
    if yoy_shares:
        why_ctx["signals"]["share_yoy_snapshot"] = {k: round(v, 4) for k, v in yoy_shares.items()}

    # 5-year historical month stats
    historical_ctx = _build_historical_month_context(
        historical_rows, cur_row, t_series_col, _get_val,
    )
    if historical_ctx:
        why_ctx["signals"]["historical_month"] = historical_ctx

    if cur_shares:
        sorted_mix = sorted(cur_shares.items(), key=lambda kv: kv[1], reverse=True)
        mix_parts = []
        for key, value in sorted_mix[:5]:
            label = BALANCING_SHARE_METADATA.get(key, {}).get("label", key.replace("_", " "))
            mix_parts.append(f"{label} {value * 100:.1f}%")
        if mix_parts:
            why_ctx["notes"].append("Current balancing mix composition: " + ", ".join(mix_parts) + ".")

    share_notes = build_share_shift_notes(cur_shares, prev_shares)
    why_ctx["notes"].extend(share_notes)

    # YoY comparison note
    if yoy_gel is not None and cur_gel is not None:
        yoy_delta = cur_gel - yoy_gel
        direction = "higher" if yoy_delta > 0 else ("lower" if yoy_delta < 0 else "unchanged from")
        yoy_period = str(yoy_row[t_series_col].iloc[0]) if not yoy_row.empty else "same month last year"
        why_ctx["notes"].append(
            f"Year-over-year: balancing price is {direction} than {yoy_period} "
            f"({yoy_gel:.1f} \u2192 {cur_gel:.1f} GEL/MWh, {yoy_delta:+.1f})."
        )

    # 5-year historical month notes
    if historical_ctx:
        month_name = target_ts.strftime("%B") if target_ts else "this month"
        n_years = historical_ctx.get("years_found", 0)

        for hist_label, metric_label in [("price_gel", "GEL/MWh"), ("price_usd", "USD/MWh")]:
            hist_stats = historical_ctx.get(hist_label)
            if not hist_stats:
                continue
            note = (
                f"Historical {month_name} ({n_years}-year window): "
                f"{metric_label} ranged {hist_stats['min']:.1f}\u2013{hist_stats['max']:.1f}, "
                f"avg {hist_stats['avg']:.1f}"
            )
            if "trend_direction" in hist_stats:
                note += f", trend: {hist_stats['trend_direction']}"
            if "current_vs_history" in hist_stats:
                note += f". Current is {hist_stats['current_vs_history'].replace('_', ' ')}"
            why_ctx["notes"].append(note + ".")

        cc = historical_ctx.get("cross_currency")
        if cc:
            why_ctx["notes"].append(
                f"GEL price is {cc['gel_vs_5yr_avg_pct']:+.1f}% vs 5-year {month_name} avg, "
                f"USD price is {cc['usd_vs_5yr_avg_pct']:+.1f}% \u2014 "
                f"~{cc['currency_effect_pct']:+.1f}pp of the GEL change is associated with currency movement."
            )

    why_ctx["notes"].append("Balancing price is a weighted average of electricity sold as balancing energy.")
    why_ctx["notes"].append("Regulated and deregulated hydro depend weakly on xrate; thermal PPAs and imports depend strongly on xrate.")
    why_ctx["notes"].append("When GEL depreciates, electricity prices generally rise because USD-denominated constituents (like imported gas) become more expensive.")
    why_ctx["notes"].append("If GEL depreciates, GEL-denominated balancing price rises due to USD-linked gas/import costs.")
    why_ctx["notes"].append("Composition shift toward thermal or import increases price; more hydro or renewable lowers it.")

    analysis_evidence_df = _build_requested_analysis_evidence(
        ctx,
        df,
        t_series_col,
        target_ts,
        cur_row,
        prev_ts,
        prev_row,
        cur_shares,
        prev_shares,
    )
    ctx.analysis_evidence = (
        analysis_evidence_df.replace({np.nan: None}).to_dict(orient="records")
        if not analysis_evidence_df.empty
        else []
    )
    if ctx.analysis_evidence:
        why_ctx["derived_evidence"] = ctx.analysis_evidence

    why_prov_df = _build_why_provenance_snapshot(
        target_ts,
        prev_ts,
        cur_gel=cur_gel,
        prev_gel=prev_gel,
        cur_usd=cur_usd,
        prev_usd=prev_usd,
        cur_xrate=cur_xrate,
        prev_xrate=prev_xrate,
        cur_shares=cur_shares,
        prev_shares=prev_shares,
    )
    combined_prov_df = why_prov_df
    # Add YoY row to provenance so the grounding gate can verify YoY claims.
    if not yoy_row.empty:
        yoy_prov_cols = [c for c in combined_prov_df.columns if c in yoy_row.columns]
        combined_prov_df = pd.concat(
            [combined_prov_df, yoy_row[yoy_prov_cols] if yoy_prov_cols else yoy_row],
            ignore_index=True, sort=False,
        )
    # Add 5-year historical month rows to provenance for grounding gate.
    if not historical_rows.empty:
        hist_prov_cols = [c for c in combined_prov_df.columns if c in historical_rows.columns]
        combined_prov_df = pd.concat(
            [combined_prov_df, historical_rows[hist_prov_cols] if hist_prov_cols else historical_rows],
            ignore_index=True, sort=False,
        )
    if not analysis_evidence_df.empty:
        combined_prov_df = pd.concat([combined_prov_df, analysis_evidence_df], ignore_index=True, sort=False)
    # Add share-delta values as a provenance row so the provenance gate can
    # ground LLM claims like "6.66 pp" via _tokenize_cell_value(0.0666) → 6.66.
    if cur_shares and prev_shares:
        delta_record: dict[str, Any] = {}
        for key in sorted(set(cur_shares) | set(prev_shares)):
            delta = cur_shares.get(key, 0) - prev_shares.get(key, 0)
            if abs(delta) >= 0.005:
                delta_record[key] = round(delta, 6)
        if delta_record:
            delta_df = pd.DataFrame([delta_record])
            combined_prov_df = pd.concat(
                [combined_prov_df, delta_df], ignore_index=True, sort=False,
            )
    if not combined_prov_df.empty:
        base_hash = str(ctx.provenance_query_hash or "")
        if cur_shares or prev_shares:
            why_hash = sql_query_hash(f"{base_hash}|why_summary|{BALANCING_SHARE_PIVOT_SQL}")
        else:
            why_hash = base_hash or sql_query_hash(f"{ctx.query}|why_summary")
        stamp_provenance(
            ctx,
            list(combined_prov_df.columns),
            [tuple(r) for r in combined_prov_df.itertuples(index=False, name=None)],
            source=str(ctx.provenance_source or "sql"),
            query_hash=why_hash,
        )

    # Why-queries are handled by the LLM summarizer using the enriched
    # stats_hint (composition data, correlations, why-context JSON attached
    # below). The deterministic override is removed — the LLM produces
    # more nuanced multi-factor causal analysis.

    trace_detail(
        log,
        ctx,
        "stage_3_analyzer_enrich",
        "why_context",
        why_override=False,
        analysis_evidence_count=len(ctx.analysis_evidence or []),
        signals=why_ctx.get("signals", {}),
    )
    trace_detail(
        log,
        ctx,
        "stage_3_analyzer_enrich",
        "artifact",
        debug=True,
        why_context=why_ctx,
        analysis_evidence=ctx.analysis_evidence,
    )

    # PRIORITY 1: Causal Context (High value, small size)
    # Put this first so it survives prompt truncation.
    ctx.stats_hint += "\n\n--- CAUSAL CONTEXT ---\n" + json.dumps(why_ctx, default=str, indent=2)

    # PRIORITY 2: Detailed Evidence (Lower value, large size)
    # Prune to top 12 to reduce prompt bloat and truncation risk.
    if ctx.analysis_evidence:
        evidence_subset = ctx.analysis_evidence[:12]
        ctx.stats_hint += "\n\n--- DERIVED ANALYSIS EVIDENCE (TOP 12) ---\n" + json.dumps(evidence_subset, default=str, indent=2)

    log.info("Why-context (prioritized) attached to stats_hint.")


def _precalculate_trendlines(ctx: QueryContext, cols_labeled: list) -> None:
    """Pre-calculate trendlines for forecast answer generation."""
    try:
        from visualization.chart_builder import calculate_trendline

        time_key = next((c for c in ctx.cols if any(k in c.lower() for k in ["date", "year", "month", "თვე", "წელი", "თარიღი"])), None)
        season_col = next((c for c in ctx.cols if c.lower() in ["season", "სეზონი"]), None)

        # Fix year-only columns
        if time_key and time_key in ctx.df.columns:
            try:
                first_val = ctx.df[time_key].iloc[0]
                if isinstance(first_val, (int, float)) or str(type(first_val).__name__) == "Decimal":
                    if 1900 <= float(first_val) <= 2100:
                        ctx.df[time_key] = pd.to_datetime(ctx.df[time_key].astype(int), format="%Y")
                        log.info(f"📅 Converted year-only column '{time_key}' to datetime format")
            except Exception:
                pass

        num_cols = [c for c in ctx.cols if c != time_key and c != season_col]
        df_calc = ctx.df.copy()
        for c in num_cols:
            try:
                df_calc[c] = pd.to_numeric(df_calc[c], errors="coerce")
            except Exception:
                pass

        if not time_key or time_key not in df_calc.columns or not num_cols:
            return

        trendline_forecasts = {}

        if season_col and season_col in df_calc.columns:
            log.info("📈 Seasonal forecast detected - calculating separate trendlines")
            seasons = df_calc[season_col].dropna().unique()
            for season in seasons:
                season_df = df_calc[df_calc[season_col] == season].copy()
                for col in num_cols:
                    td = calculate_trendline(season_df, time_key, col, extend_to_date=ctx.trendline_extend_to)
                    if td and td["dates"] and td["values"]:
                        forecast_key = f"{col}_{season}"
                        trendline_forecasts[forecast_key] = {
                            "target_date": td["dates"][-1],
                            "forecast_value": round(td["values"][-1], 2),
                            "equation": td["equation"],
                            "r_squared": round(td["r_squared"], 3),
                            "season": season,
                        }
        else:
            for col in num_cols:
                td = calculate_trendline(df_calc, time_key, col, extend_to_date=ctx.trendline_extend_to)
                if td and td["dates"] and td["values"]:
                    trendline_forecasts[col] = {
                        "target_date": td["dates"][-1],
                        "forecast_value": round(td["values"][-1], 2),
                        "equation": td["equation"],
                        "r_squared": round(td["r_squared"], 3),
                    }

        if trendline_forecasts:
            forecast_summary = f"\n\n--- TRENDLINE FORECASTS (Linear Regression) ---\nTarget date: {ctx.trendline_extend_to}\n"
            for col, fi in trendline_forecasts.items():
                if "season" in fi:
                    forecast_summary += f"\n{col.replace('_' + fi['season'], '')} ({fi['season']}):\n"
                else:
                    forecast_summary += f"\n{col}:\n"
                forecast_summary += f"  - Forecast value: {fi['forecast_value']}\n"
                forecast_summary += f"  - Equation: {fi['equation']}\n"
                forecast_summary += f"  - R² (goodness of fit): {fi['r_squared']}\n"
            ctx.stats_hint += forecast_summary
            log.info(f"📊 Added {len(trendline_forecasts)} forecast values to stats_hint")

    except Exception as e:
        log.warning(f"Trendline pre-calculation failed: {e}")
