"""
Pipeline Stage 3: Analysis & Enrichment

Handles share resolution, correlation analysis, forecast mode (CAGR),
"why" causal reasoning, trendline pre-calculation, and seasonal stats.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from contracts.question_analysis import AnswerKind, ChartIntent, SemanticRole
from models import QueryContext
from core.query_executor import ENGINE
from analysis.stats import quick_stats, rows_to_preview
from analysis.seasonal_stats import (
    detect_monthly_timeseries,
    calculate_seasonal_stats,
    format_seasonal_stats,
)
from analysis.system_quantities import (
    normalize_period_series,
    normalize_period_series_with_granularity,
)
from analysis.shares import build_balancing_correlation_df, compute_regulated_plant_sales
from agent.provenance import sql_query_hash, stamp_provenance
from agent.sql_executor import BALANCING_SHARE_PIVOT_SQL, ensure_share_dataframe, fetch_balancing_share_panel
from agent.router import extract_balancing_entities
from utils.forecasting import extract_excluded_years, extract_forecast_horizon_years
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
    SEMANTIC_TO_COLUMNS,
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

# Minimum R² required to keep a linear-regression trendline in the forecast
# summary / chart.  Below this the fit explains too little variance to be
# useful (common case: USD prices driven by FX noise rather than a time trend).
_TRENDLINE_MIN_R_SQUARED = 0.30

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


_COMPONENT_PRESSURE_SPECS = (
    {
        "component": "deregulated_hydro",
        "share_col": "share_deregulated_hydro",
        "label": "deregulated hydro",
        "price_cols": {
            "gel": "price_deregulated_hydro_gel",
            "usd": "price_deregulated_hydro_usd",
        },
        "contribution_cols": {
            "gel": "contribution_deregulated_hydro_gel",
            "usd": "contribution_deregulated_hydro_usd",
        },
        "mechanism_note": "seasonal hydro layer; summer is often below balancing price, winter must use observed price.",
    },
    {
        "component": "regulated_hpp",
        "share_col": "share_regulated_hpp",
        "label": "regulated HPP",
        "price_cols": {
            "gel": "price_regulated_hpp_gel",
            "usd": "price_regulated_hpp_usd",
        },
        "contribution_cols": {
            "gel": "contribution_regulated_hpp_gel",
            "usd": "contribution_regulated_hpp_usd",
        },
        "mechanism_note": "regulated hydro tariff layer; usually below balancing price when its share is material.",
    },
    {
        "component": "regulated_new_tpp",
        "share_col": "share_regulated_new_tpp",
        "label": "regulated new TPP",
        "price_cols": {
            "gel": "price_regulated_new_tpp_gel",
            "usd": "price_regulated_new_tpp_usd",
        },
        "contribution_cols": {
            "gel": "contribution_regulated_new_tpp_gel",
            "usd": "contribution_regulated_new_tpp_usd",
        },
        "mechanism_note": "regulated thermal tariff layer; gas-price and xrate linkage can matter.",
    },
    {
        "component": "regulated_old_tpp",
        "share_col": "share_regulated_old_tpp",
        "label": "regulated old TPP",
        "price_cols": {
            "gel": "price_regulated_old_tpp_gel",
            "usd": "price_regulated_old_tpp_usd",
        },
        "contribution_cols": {
            "gel": "contribution_regulated_old_tpp_gel",
            "usd": "contribution_regulated_old_tpp_usd",
        },
        "mechanism_note": "regulated thermal tariff layer; gas-price and xrate linkage can matter.",
    },
    {
        "component": "residual_ppa_import",
        "share_col": "share_ppa_import_total",
        "label": "residual PPA/CfD/import layer",
        "price_cols": {},
        "contribution_cols": {
            "gel": "residual_contribution_ppa_import_gel",
            "usd": "residual_contribution_ppa_import_usd",
        },
        "mechanism_note": "residual PPA/CfD/import layer; contribution is observed, but direct source prices are not exposed (confidential).",
    },
)


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return round(float(value), digits)


def _relative_price_position(price_value: float | None, balancing_value: float | None) -> str | None:
    if price_value is None or balancing_value is None:
        return None
    delta = float(price_value) - float(balancing_value)
    if abs(delta) <= 0.05:
        return "near_balancing"
    return "above_balancing" if delta > 0 else "below_balancing"


def _pressure_from_share_shift(relative_position: str | None, share_delta: float | None) -> str | None:
    if relative_position is None or share_delta is None:
        return None
    if abs(share_delta) < 0.0005:
        return "share_stable"
    if relative_position == "above_balancing":
        return "upward_pressure" if share_delta > 0 else "removed_upward_pressure"
    if relative_position == "below_balancing":
        return "downward_pressure" if share_delta > 0 else "removed_downward_pressure"
    return "neutral_pressure"


def _residual_pressure_from_contribution(contribution_delta: float | None) -> str | None:
    if contribution_delta is None:
        return None
    if abs(contribution_delta) < 0.05:
        return "contribution_stable"
    return "higher_residual_contribution" if contribution_delta > 0 else "lower_residual_contribution"


def _build_component_pressure_summary(
    cur_row: pd.DataFrame,
    prev_row: pd.DataFrame,
    *,
    cur_gel: float | None,
    prev_gel: float | None,
    cur_usd: float | None,
    prev_usd: float | None,
    value_getter: Callable[[pd.DataFrame, list[str] | tuple[str, ...]], float | None],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []

    for spec in _COMPONENT_PRESSURE_SPECS:
        share_col = spec["share_col"]
        share_cur = value_getter(cur_row, [share_col])
        share_prev = value_getter(prev_row, [share_col]) if not prev_row.empty else None
        share_delta = None
        if share_cur is not None or share_prev is not None:
            share_delta = (share_cur or 0.0) - (share_prev or 0.0)

        contrib_prev_gel = value_getter(prev_row, [spec["contribution_cols"].get("gel", "")]) if not prev_row.empty else None
        contrib_cur_gel = value_getter(cur_row, [spec["contribution_cols"].get("gel", "")])
        contrib_prev_usd = value_getter(prev_row, [spec["contribution_cols"].get("usd", "")]) if not prev_row.empty else None
        contrib_cur_usd = value_getter(cur_row, [spec["contribution_cols"].get("usd", "")])

        material_share = any(v is not None and abs(v) >= 0.0005 for v in (share_prev, share_cur))
        material_contribution = any(
            v is not None and abs(v) >= 0.05
            for v in (contrib_prev_gel, contrib_cur_gel, contrib_prev_usd, contrib_cur_usd)
        )
        if not (material_share or material_contribution):
            continue

        record: dict[str, Any] = {
            "component": spec["component"],
            "label": spec["label"],
            "share_prev": _round_or_none(share_prev, 4),
            "share_cur": _round_or_none(share_cur, 4),
            "share_delta_pp": _round_or_none((share_delta * 100.0) if share_delta is not None else None, 2),
            "mechanism_note": spec["mechanism_note"],
        }

        for currency, balancing_prev, balancing_cur in (
            ("gel", prev_gel, cur_gel),
            ("usd", prev_usd, cur_usd),
        ):
            price_col = spec["price_cols"].get(currency)
            contribution_col = spec["contribution_cols"].get(currency)
            price_prev = value_getter(prev_row, [price_col]) if price_col and not prev_row.empty else None
            price_cur = value_getter(cur_row, [price_col]) if price_col else None
            contrib_prev = value_getter(prev_row, [contribution_col]) if contribution_col and not prev_row.empty else None
            contrib_cur = value_getter(cur_row, [contribution_col]) if contribution_col else None

            record[f"price_prev_{currency}"] = _round_or_none(price_prev, 2)
            record[f"price_cur_{currency}"] = _round_or_none(price_cur, 2)
            record[f"price_delta_{currency}"] = _round_or_none(
                (price_cur - price_prev) if price_cur is not None and price_prev is not None else None,
                2,
            )
            record[f"price_vs_balancing_prev_{currency}"] = _round_or_none(
                (price_prev - balancing_prev)
                if price_prev is not None and balancing_prev is not None
                else None,
                2,
            )
            record[f"price_vs_balancing_cur_{currency}"] = _round_or_none(
                (price_cur - balancing_cur)
                if price_cur is not None and balancing_cur is not None
                else None,
                2,
            )
            record[f"relative_price_prev_{currency}"] = _relative_price_position(price_prev, balancing_prev)
            record[f"relative_price_cur_{currency}"] = _relative_price_position(price_cur, balancing_cur)
            record[f"contribution_prev_{currency}"] = _round_or_none(contrib_prev, 2)
            record[f"contribution_cur_{currency}"] = _round_or_none(contrib_cur, 2)
            contribution_delta = (
                (contrib_cur - contrib_prev)
                if contrib_cur is not None and contrib_prev is not None
                else None
            )
            record[f"contribution_delta_{currency}"] = _round_or_none(contribution_delta, 2)
            observed_pressure = _pressure_from_share_shift(record[f"relative_price_cur_{currency}"], share_delta)
            if observed_pressure is None and spec["component"] == "residual_ppa_import":
                observed_pressure = _residual_pressure_from_contribution(contribution_delta)
            record[f"observed_pressure_{currency}"] = observed_pressure

        summary.append({k: v for k, v in record.items() if v is not None})

    return summary



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
    if metric_name in METRIC_VALUE_ALIASES:
        return METRIC_VALUE_ALIASES[metric_name]
    if metric_name in SEMANTIC_TO_COLUMNS:
        return SEMANTIC_TO_COLUMNS[metric_name]
    return [metric_name]


def _find_time_series_column(df: pd.DataFrame) -> Optional[str]:
    """Return the most likely time column from an analysis frame."""
    return next(
        (
            c for c in df.columns
            if any(k in c.lower() for k in ["date", "year", "month", "period"])
        ),
        None,
    )


def _normalize_query_target_metric(query: str, target_metric: Optional[str]) -> Optional[str]:
    """Correct common analyzer target drift for technical correlation questions."""
    query_lower = str(query or "").lower()
    target = str(target_metric or "").strip().lower() or None
    if any(token in query_lower for token in ("demand", "consumption")):
        return "demand"
    if any(token in query_lower for token in ("import dependency", "import dependence", "energy security", "self-sufficiency")):
        return "import_dependency"
    if "generation" in query_lower:
        return "generation"
    return target


def _normalize_correlation_request(
    query: str,
    metric: Optional[str],
    target_metric: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Correct common analyzer drift for two-series technical correlations."""

    query_lower = str(query or "").lower()
    normalized_metric = str(metric or "").strip().lower() or None
    normalized_target = _normalize_query_target_metric(query, target_metric)

    balancing_terms = (
        "balancing electricity price",
        "balancing price",
        "balancing electricity cost",
        "balancing cost",
        "p_bal",
    )
    has_balancing = any(term in query_lower for term in balancing_terms)

    comparison_metric = None
    if any(token in query_lower for token in ("demand", "consumption")):
        comparison_metric = "demand"
    elif any(
        token in query_lower
        for token in ("import dependency", "import dependence", "energy security", "self-sufficiency")
    ):
        comparison_metric = "import_dependency"
    elif "generation" in query_lower:
        comparison_metric = "generation"

    if has_balancing and comparison_metric:
        return comparison_metric, "balancing"

    return normalized_metric, normalized_target


def _build_correlation_matrix_from_frame(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute pairwise correlations from the current analysis frame."""
    if df is None or df.empty:
        return {}

    working = df.copy()
    time_col = _find_time_series_column(working)
    if time_col and time_col in working.columns:
        working[time_col] = normalize_period_series(working[time_col])
        working = working.dropna(subset=[time_col]).sort_values(time_col)

    candidate_cols = [col for col in working.columns if col != time_col]
    numeric_df = working[candidate_cols].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if len(numeric_df) < 3 or numeric_df.shape[1] < 2:
        return {}

    preferred_targets = [
        "p_bal_gel",
        "p_bal_usd",
        "total_demand",
        "total_domestic_generation",
        "local_generation",
        "import_dependency_ratio",
        "import_dependent_supply",
    ]
    targets = [col for col in preferred_targets if col in numeric_df.columns]
    if not targets:
        return {}

    corr_results: dict[str, dict[str, float]] = {}
    corr_matrix = numeric_df.corr(numeric_only=True)
    for target in targets:
        if target not in corr_matrix.columns:
            continue
        series = corr_matrix[target].drop(labels=[target], errors="ignore").dropna()
        if not series.empty:
            corr_results[target] = series.sort_values(ascending=False).round(3).to_dict()
    return corr_results


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
    # Prefer structured analyzer requests, then fall back to heuristic scenario extraction.
    if ctx.has_authoritative_question_analysis:
        requests = []
        for req in ctx.question_analysis.analysis_requirements.derived_metrics:
            payload = req.model_dump(mode="json")
            if payload.get("metric_name") == "correlation_to_target":
                payload["metric"], payload["target_metric"] = _normalize_correlation_request(
                    ctx.query,
                    payload.get("metric"),
                    payload.get("target_metric"),
                )
            requests.append(payload)
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

    # Collect per-currency history so Stage 4 can compare the target month with its seasonal peers.
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

        # Position the current observation inside the historical min/max envelope.
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

    # Cross-currency comparison helps separate real price movement from exchange-rate effects.
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

    # Build one shared metric context so each registry function sees the same resolved periods and snapshots.
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

    # Dispatch each requested metric to the registry and keep only successful materializations.
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


# Semantic chart override helpers translate derived evidence into ready-to-render series.
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

    return COLUMN_LABELS.get(metric, metric.replace("_", " "))


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


# Share-target resolution helpers map user phrasing onto the share_* columns already in evidence.
def _select_share_column(share_cols: list[str], target_text: str) -> Optional[str]:
    """Choose the most relevant share column based on the user's target description."""
    target_lower = target_text.lower()

    # Direct entity matches
    priority_map = {
        "cfd_scheme": "share_cfd_scheme",
        "cfd": "share_cfd_scheme",
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


def _share_label_from_column(col_name: str) -> str:
    meta = BALANCING_SHARE_METADATA.get(col_name, {})
    return meta.get("label", col_name.replace("share_", "").replace("_", " "))


_GROUP_SHARE_ALIAS_SIGNALS = (
    "regulated thermal",
    "regulated thermals",
    "all regulated thermal",
    "all regulated thermals",
    "regulated tpp",
    "regulated tpps",
    "remaining electricity",
    "remaining energy",
    "residual ppa/cfd/import",
    "ppa cfd import residual",
)


_EXPLICIT_SHARE_SEGMENT_SPLIT_RE = re.compile(r"\s*(?:,|\band\b|\+)\s*")


def _share_fraction_to_pct(value: Any) -> float:
    numeric = float(value)
    return numeric * 100 if numeric <= 1 else numeric


def _extract_share_component_clause(target_lower: str) -> str:
    patterns = (
        r"share\s+(?:of|or)\s+(?P<clause>.+?)\s+in\s+balancing(?:\s+electricity)?",
        r"share\s+(?:of|or)\s+(?P<clause>.+?)(?:\s+(?:more than|above|over|greater than|at least|below|under|less than|exceed(?:ed|s|ing)?)\b|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, target_lower)
        if match:
            clause = str(match.group("clause") or "").strip()
            if clause:
                return clause
    return target_lower


def _extract_explicit_share_segments(target_lower: str) -> list[str]:
    clause = _extract_share_component_clause(target_lower)
    segments = [
        re.sub(r"^[\s:;.-]+|[\s:;?.-]+$", "", segment).strip()
        for segment in _EXPLICIT_SHARE_SEGMENT_SPLIT_RE.split(clause)
    ]
    return [segment for segment in segments if segment]


def _extract_combined_share_components(
    target_text: str,
    share_cols: list[str],
) -> tuple[list[str], bool]:
    """Resolve explicit multi-component share requests into component columns."""
    target_lower = str(target_text or "").lower()
    if not target_lower:
        return [], False
    clause = _extract_share_component_clause(target_lower)

    # Only treat this as an aggregate when the query explicitly asks for a total/
    # combined/summed share or enumerates multiple components via commas/and.
    explicit_combine = any(
        token in clause
        for token in ("total share", "combined share", "sum of", ",", " and ")
    )
    if not explicit_combine and not any(signal in clause for signal in _GROUP_SHARE_ALIAS_SIGNALS):
        return [], False

    # Evidence frames expose share_* columns in lowercase snake_case
    # (for example ``share_cfd_scheme``), while routing uses canonical
    # balancing entity IDs such as ``CfD_scheme``. Normalize both sides so
    # deterministic combined-share answers don't miss live columns due to case.
    entity_to_col = {col.removeprefix("share_").lower(): col for col in share_cols}
    components: list[str] = []

    if explicit_combine:
        unresolved_segment = False
        for segment in _extract_explicit_share_segments(clause):
            segment_entities = extract_balancing_entities(segment)
            if not segment_entities:
                unresolved_segment = True
                break
            segment_component_count = 0
            for entity in segment_entities:
                col_name = entity_to_col.get(str(entity).lower())
                if not col_name:
                    continue
                segment_component_count += 1
                if col_name not in components:
                    components.append(col_name)
            if segment_component_count == 0:
                unresolved_segment = True
                break
        requested_combination = True
        if unresolved_segment or len(components) <= 1:
            return [], True
        return components, requested_combination

    extracted_entities = extract_balancing_entities(clause)
    for entity in extracted_entities:
        col_name = entity_to_col.get(str(entity).lower())
        if col_name and col_name not in components:
            components.append(col_name)

    requested_combination = len(components) > 1
    if requested_combination and len(components) <= 1:
        return [], True
    return components, requested_combination


def _build_combined_share_label(target_text: str, components: list[str]) -> str:
    labels = [_share_label_from_column(col_name) for col_name in components]
    return " + ".join(labels)


def _resolve_share_target(
    df: pd.DataFrame,
    share_cols: list[str],
    target_text: str,
) -> tuple[pd.DataFrame, Optional[str], Optional[str], list[str]]:
    """Return dataframe + selected share target, supporting explicit combinations."""
    combined_components, requested_combination = _extract_combined_share_components(target_text, share_cols)
    if combined_components:
        working = df.copy()
        synthetic_col = "share_combined_target"
        running = pd.Series(0.0, index=working.index, dtype="float64")
        for col_name in combined_components:
            running = running.add(pd.to_numeric(working[col_name], errors="coerce").fillna(0.0), fill_value=0.0)
        working[synthetic_col] = running
        return working, synthetic_col, _build_combined_share_label(target_text, combined_components), combined_components
    if requested_combination:
        return df, None, None, []

    selected_col = _select_share_column(share_cols, target_text)
    if not selected_col:
        return df, None, None, []
    return df, selected_col, _share_label_from_column(selected_col), [selected_col]


def _build_share_summary_grounding_hint(
    matched: pd.DataFrame,
    *,
    date_col: str,
    selected_col: str,
    label: str,
    threshold_pct: Optional[float] = None,
    component_cols: Optional[list[str]] = None,
    gel_col: Optional[str] = None,
    usd_col: Optional[str] = None,
) -> str:
    """Emit deterministic numeric evidence so provenance can ground computed share answers."""
    component_cols = component_cols or []
    lines = ["DETERMINISTIC SHARE SUMMARY EVIDENCE:"]
    if threshold_pct is not None:
        lines.append(f"requested_threshold_pct={threshold_pct:.1f}")
    if len(component_cols) > 1:
        lines.append(f"combined_components={','.join(component_cols)}")

    evidence_rows = matched.copy()
    evidence_rows[date_col] = pd.to_datetime(evidence_rows[date_col], errors="coerce")
    evidence_rows[selected_col] = pd.to_numeric(evidence_rows[selected_col], errors="coerce")
    evidence_rows = evidence_rows.dropna(subset=[date_col, selected_col]).sort_values(date_col)

    for _, row in evidence_rows.iterrows():
        share_value = float(row[selected_col])
        share_pct = _share_fraction_to_pct(share_value)
        period_str = pd.to_datetime(row[date_col]).strftime("%B %Y")
        line = f"{period_str}: {label}={share_pct:.1f}%"
        if len(component_cols) > 1:
            component_bits = []
            for component_col in component_cols:
                component_val = pd.to_numeric(pd.Series([row.get(component_col)]), errors="coerce").iloc[0]
                if pd.notna(component_val):
                    component_pct = _share_fraction_to_pct(component_val)
                    component_bits.append(f"{_share_label_from_column(component_col)}={component_pct:.1f}%")
            if component_bits:
                line += f"; components: {', '.join(component_bits)}"
        if gel_col and pd.notna(row.get(gel_col)):
            line += f"; balancing_price_gel={float(row[gel_col]):.1f}"
        if usd_col and pd.notna(row.get(usd_col)):
            line += f"; balancing_price_usd={float(row[usd_col]):.1f}"
        lines.append(line)

    return "\n".join(lines)


_SHARE_THRESHOLD_RULES: list[tuple[str, str, str]] = [
    (r"(more than|above|over|exceed(?:ed|s|ing)?|greater than)\s+(\d+(?:\.\d+)?)\s*%?", "gt", "exceeded"),
    (r"(at least|not less than|minimum of)\s+(\d+(?:\.\d+)?)\s*%?", "ge", "was at least"),
    (r"(less than|below|under|fewer than)\s+(\d+(?:\.\d+)?)\s*%?", "lt", "was below"),
    (r"(at most|no more than|maximum of)\s+(\d+(?:\.\d+)?)\s*%?", "le", "was at most"),
]


def _extract_share_threshold(user_query: str) -> Optional[tuple[str, float, str]]:
    """Parse threshold operators like '> 99%' from a share query."""
    query_lower = str(user_query or "").lower()
    if "share" not in query_lower:
        return None

    for pattern, operator, phrase in _SHARE_THRESHOLD_RULES:
        match = re.search(pattern, query_lower)
        if not match:
            continue
        try:
            raw_value = float(match.group(2))
        except (TypeError, ValueError):
            continue
        threshold = raw_value / 100.0 if raw_value > 1 else raw_value
        return operator, threshold, phrase
    return None


def _share_query_requests_prices(user_query: str) -> bool:
    query_lower = str(user_query or "").lower()
    return any(token in query_lower for token in ("price", "gel", "usd"))


def _find_share_answer_price_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    gel_col = next((c for c in ["balancing_price_gel", "p_bal_gel"] if c in df.columns), None)
    usd_col = next((c for c in ["balancing_price_usd", "p_bal_usd"] if c in df.columns), None)
    return gel_col, usd_col


def _build_share_summary_artifact(
    df: pd.DataFrame,
    plan: Dict[str, Any],
    user_query: str,
) -> tuple[Optional[str], str]:
    """Produce deterministic share summary text plus grounding support text."""
    if df is None or df.empty:
        return None, ""

    share_cols = [c for c in df.columns if c.startswith("share_")]
    if not share_cols:
        return None, ""

    target_text = str(plan.get("target", "")) + " " + user_query
    period_hint = str(plan.get("period", ""))
    period = _parse_period_hint(period_hint, user_query)

    working_df, selected_col, label, component_cols = _resolve_share_target(df, share_cols, target_text)
    if not selected_col or not label:
        return None, ""

    date_cols = [c for c in working_df.columns if any(k in c.lower() for k in ["date", "time_month"])]
    threshold_rule = _extract_share_threshold(user_query)

    if threshold_rule and date_cols:
        date_col = date_cols[0]
        working = working_df.copy()
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
        working[selected_col] = pd.to_numeric(working[selected_col], errors="coerce")
        working = working.dropna(subset=[date_col, selected_col]).sort_values(date_col)
        if working.empty:
            return None, ""

        operator, threshold, phrase = threshold_rule
        if operator == "gt":
            matched = working[working[selected_col] > threshold]
        elif operator == "ge":
            matched = working[working[selected_col] >= threshold]
        elif operator == "lt":
            matched = working[working[selected_col] < threshold]
        else:
            matched = working[working[selected_col] <= threshold]

        threshold_pct = threshold * 100 if threshold <= 1 else threshold
        gel_col, usd_col = _find_share_answer_price_columns(matched)
        grounding_hint = _build_share_summary_grounding_hint(
            matched if not matched.empty else working.head(0),
            date_col=date_col,
            selected_col=selected_col,
            label=label,
            threshold_pct=threshold_pct,
            component_cols=component_cols,
            gel_col=gel_col,
            usd_col=usd_col,
        )
        if matched.empty:
            return (
                f"No months were found where **{label.title()}** {phrase} "
                f"**{threshold_pct:.1f}%** of balancing electricity in the available data."
            ), grounding_hint

        include_prices = _share_query_requests_prices(user_query)
        summary_parts = [
            f"Months where **{label.title()}** {phrase} **{threshold_pct:.1f}%** of balancing electricity:"
        ]
        for _, row in matched.iterrows():
            share_value = float(row[selected_col])
            share_pct = _share_fraction_to_pct(share_value)
            period_str = pd.to_datetime(row[date_col]).strftime("%B %Y")
            line = f"- {period_str}: {share_pct:.1f}%"
            if include_prices:
                price_bits = []
                if gel_col and pd.notna(row.get(gel_col)):
                    price_bits.append(f"{float(row[gel_col]):.1f} GEL/MWh")
                if usd_col and pd.notna(row.get(usd_col)):
                    price_bits.append(f"{float(row[usd_col]):.1f} USD/MWh")
                if price_bits:
                    line += f"; balancing price {', '.join(price_bits)}"
            summary_parts.append(line)
            if include_prices and not (gel_col or usd_col):
                summary_parts.append("- Balancing price columns were not available in the retrieved evidence for these months.")
        return "\n".join(summary_parts), grounding_hint

    # Find the row matching the period
    if date_cols:
        date_col = date_cols[0]
        df = working_df.copy()
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
        return None, ""

    value_pct = _share_fraction_to_pct(value)

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
            r_pct = _share_fraction_to_pct(renewable)
            t_pct = _share_fraction_to_pct(thermal)
            summary_parts.append(f"  - Renewable PPA: {r_pct:.1f}%")
            summary_parts.append(f"  - Thermal PPA: {t_pct:.1f}%")

    grounding_hint = _build_share_summary_grounding_hint(
        filtered.tail(1),
        date_col=date_cols[0] if date_cols else "date",
        selected_col=selected_col,
        label=label,
        component_cols=component_cols,
    ) if date_cols else ""
    return "\n".join(summary_parts), grounding_hint


def generate_share_summary(df: pd.DataFrame, plan: Dict[str, Any], user_query: str) -> Optional[str]:
    """Produce a deterministic textual answer for share queries to avoid LLM hallucinations."""
    summary, _grounding_hint = _build_share_summary_artifact(df, plan, user_query)
    return summary


# ---------------------------------------------------------------------------
# Forecast helpers (moved from ask_post inner functions)
# ---------------------------------------------------------------------------

def _extract_forecast_horizon(query: str) -> int:
    """Extract forecast duration (in years) from user query."""
    return extract_forecast_horizon_years(query)


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


def _robust_endpoint_value(series: "pd.Series", *, window: int = 3, which: str) -> float:
    """Return an endpoint value damped against single-year noise.

    Instead of using the raw first or last observation, average the first or
    last ``window`` observations.  Used by :func:`_generate_cagr_forecast` so
    that a single anomalous edge year does not distort a multi-decade CAGR
    projection.

    Important short-series behaviour
    --------------------------------
    When the series has fewer than ``2 * window`` observations, the leading
    and trailing windows would overlap and collapse to the same mean, which
    forces any CAGR computed from them to zero (flat forecast).  In that
    regime we intentionally fall back to the raw first/last value so the
    forecast still reflects the observed endpoint movement.  Damping only
    kicks in once the series is long enough for the two windows to be
    disjoint.

    Parameters
    ----------
    series:
        Numeric pandas Series, ordered chronologically (oldest first).
    window:
        Maximum number of observations to average at the chosen endpoint.
    which:
        ``"first"`` for the leading window, ``"last"`` for the trailing window.
    """
    if series is None or len(series) == 0:
        return float("nan")
    n = len(series)
    w = max(1, int(window))
    # Short-series guard: if leading+trailing windows would overlap, fall
    # back to the raw endpoint to avoid a degenerate first_val == last_val
    # collapse that would zero out the CAGR.
    if n < 2 * w:
        if which == "first":
            return float(series.iloc[0])
        if which == "last":
            return float(series.iloc[-1])
        raise ValueError(f"Unknown endpoint selector: {which!r}")
    if which == "first":
        return float(series.iloc[:w].mean())
    if which == "last":
        return float(series.iloc[-w:].mean())
    raise ValueError(f"Unknown endpoint selector: {which!r}")


def _generate_cagr_forecast(df_in: pd.DataFrame, user_query: str) -> Tuple[pd.DataFrame, str]:
    """Generate CAGR-based forecast for price or quantity data."""
    df = df_in.copy()
    time_col, value_col = _choose_target_for_forecast(df)
    if not time_col or not value_col:
        return df_in, "Forecast skipped: no clear time/value columns."

    df[time_col], time_granularity = normalize_period_series_with_granularity(df[time_col])
    df = df.dropna(subset=[time_col, value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return df_in, "Forecast skipped: no numeric data."

    data_type = _detect_data_type(value_col)
    note_parts = []

    def _strip_scratch(df_out: pd.DataFrame) -> pd.DataFrame:
        """Drop internal ``__forecast_*`` scratch columns before returning.

        ``is_forecast`` (marker) and ``season`` (user-visible dimension) are
        preserved. Defence-in-depth against these columns leaking downstream
        into chart builders / num-col resolvers.
        """
        drop_cols = [c for c in df_out.columns if str(c).startswith("__forecast_")]
        if drop_cols:
            return df_out.drop(columns=drop_cols)
        return df_out

    def _usable_yearly_points_message(yearly_count: int) -> str:
        noun = "point" if yearly_count == 1 else "points"
        return f"Forecast skipped: only {yearly_count} usable yearly {noun} after normalization/filtering."

    def _resolve_target_years(last_year: int) -> list[int]:
        horizon = _extract_forecast_horizon(user_query)
        yrs_in_q = re.findall(r"(20\d{2})", user_query)
        return sorted({int(y) for y in yrs_in_q if int(y) > last_year}) or [
            int(last_year) + i for i in range(1, horizon + 1)
        ]

    excluded_years = extract_excluded_years(user_query)
    if excluded_years:
        observed_years = set(df[time_col].dt.year.dropna().astype(int).tolist())
        applied_exclusions = sorted(observed_years & excluded_years)
        if applied_exclusions:
            note_parts.append(
                f"Excluded years from model fit: {', '.join(map(str, applied_exclusions))}."
            )
            df = df[~df[time_col].dt.year.isin(applied_exclusions)].copy()

    if data_type == "quantity":
        df["__forecast_year"] = df[time_col].dt.year
        df_y = (
            df.groupby("__forecast_year")[value_col]
            .sum()
            .dropna()
            .sort_index()
            .reset_index()
            .rename(columns={"__forecast_year": "year"})
        )
        if len(df_y) < 2:
            return df_in, " ".join(note_parts + [_usable_yearly_points_message(len(df_y))]).strip()
        first, last = df_y.iloc[0], df_y.iloc[-1]
        span = last["year"] - first["year"]
        if span <= 0 or first[value_col] <= 0:
            return df_in, "Invalid data for CAGR."
        # Fix 5: damp endpoint noise with a 3-year trailing average at both ends.
        first_val = _robust_endpoint_value(df_y[value_col], window=3, which="first")
        last_val = _robust_endpoint_value(df_y[value_col], window=3, which="last")
        if first_val <= 0 or last_val <= 0 or np.isnan(first_val) or np.isnan(last_val):
            return df_in, "Invalid data for CAGR."
        cagr = (last_val / first_val) ** (1 / span) - 1
        note_parts.append(f"Yearly CAGR={cagr*100:.2f}% ({int(first['year'])}→{int(last['year'])}).")
        target_years = _resolve_target_years(int(last["year"]))
        f_rows = []
        for y in target_years:
            # Project from the damped last value to match the CAGR formula above.
            val = last_val * ((1 + cagr) ** (y - last["year"]))
            # Force year to int to avoid 2026.0-01-01 errors
            f_rows.append({time_col: pd.to_datetime(f"{int(y)}-01-01"), value_col: val, "is_forecast": True})
        if "is_forecast" not in df.columns:
            df["is_forecast"] = False
        df_f = pd.concat([df, pd.DataFrame(f_rows)], ignore_index=True)
        note_parts.append(f"Forecast years: {', '.join(map(str, target_years))}.")
        return _strip_scratch(df_f), " ".join(note_parts)

    elif data_type == "price":
        df["__forecast_year"] = df[time_col].dt.year
        df["__forecast_month"] = df[time_col].dt.month
        df["__forecast_season"] = np.where(df["__forecast_month"].isin(SUMMER_MONTHS), "summer", "winter")

        df_y = (
            df.groupby("__forecast_year")[value_col]
            .mean()
            .dropna()
            .sort_index()
            .reset_index()
            .rename(columns={"__forecast_year": "year"})
        )
        if len(df_y) < 2:
            return df_in, " ".join(note_parts + [_usable_yearly_points_message(len(df_y))]).strip()
        first, last = df_y.iloc[0], df_y.iloc[-1]
        span = last["year"] - first["year"]

        # Fix 5: damped endpoints (3-year trailing averages).
        first_val = _robust_endpoint_value(df_y[value_col], window=3, which="first")
        last_val = _robust_endpoint_value(df_y[value_col], window=3, which="last")

        if span > 0 and first_val > 0 and last_val > 0 and not np.isnan(first_val) and not np.isnan(last_val):
            cagr_y = (last_val / first_val) ** (1 / span) - 1
        else:
            cagr_y = 0

        def format_cagr(cagr_val):
            return f"{cagr_val*100:.2f}" if not np.isnan(cagr_val) else "N/A"

        target_years = _resolve_target_years(int(last["year"]))

        if time_granularity == "year":
            note_parts.append(f"Yearly CAGR={format_cagr(cagr_y)}% using yearly averages.")
            f_rows = []
            for y in target_years:
                val_y = last_val * ((1 + cagr_y) ** (y - last["year"]))
                f_rows.append({time_col: pd.to_datetime(f"{int(y)}-01-01"), value_col: val_y, "is_forecast": True})

            if "is_forecast" not in df.columns:
                df["is_forecast"] = False
            df_f = pd.concat([df, pd.DataFrame(f_rows)], ignore_index=True)
            note_parts.append(f"Forecast years: {', '.join(map(str, target_years))}.")
            return _strip_scratch(df_f), " ".join(note_parts)

        df_s = (
            df.groupby(["__forecast_year", "__forecast_season"])[value_col]
            .mean()
            .dropna()
            .reset_index()
            .rename(columns={"__forecast_year": "year", "__forecast_season": "season"})
        )
        summer = df_s[df_s["season"] == "summer"].sort_values("year")
        winter = df_s[df_s["season"] == "winter"].sort_values("year")

        # Fix 5: damped endpoints for seasonal CAGR too.
        if len(summer) >= 2:
            s_first = _robust_endpoint_value(summer[value_col], window=3, which="first")
            s_last = _robust_endpoint_value(summer[value_col], window=3, which="last")
            s_span = summer["year"].iloc[-1] - summer["year"].iloc[0]
            cagr_s = (
                (s_last / s_first) ** (1 / s_span) - 1
                if s_span > 0 and s_first > 0 and s_last > 0 and not np.isnan(s_first) and not np.isnan(s_last)
                else np.nan
            )
        else:
            s_last = float("nan")
            cagr_s = np.nan

        if len(winter) >= 2:
            w_first = _robust_endpoint_value(winter[value_col], window=3, which="first")
            w_last = _robust_endpoint_value(winter[value_col], window=3, which="last")
            w_span = winter["year"].iloc[-1] - winter["year"].iloc[0]
            cagr_w = (
                (w_last / w_first) ** (1 / w_span) - 1
                if w_span > 0 and w_first > 0 and w_last > 0 and not np.isnan(w_first) and not np.isnan(w_last)
                else np.nan
            )
        else:
            w_last = float("nan")
            cagr_w = np.nan

        note_parts.append(f"Yearly CAGR={format_cagr(cagr_y)}%, Summer={format_cagr(cagr_s)}%, Winter={format_cagr(cagr_w)}%.")

        f_rows = []
        for y in target_years:
            val_y = last_val * ((1 + cagr_y) ** (y - last["year"]))
            # Fix 4: compound summer from last summer base, winter from last winter base.
            summer_base = s_last if not np.isnan(s_last) and s_last > 0 else last_val
            winter_base = w_last if not np.isnan(w_last) and w_last > 0 else last_val
            val_s = summer_base * ((1 + cagr_s) ** (y - last["year"])) if not np.isnan(cagr_s) else val_y
            val_w = winter_base * ((1 + cagr_w) ** (y - last["year"])) if not np.isnan(cagr_w) else val_y
            # Force year to int
            f_rows.append({time_col: pd.to_datetime(f"{int(y)}-04-01"), "__forecast_season": "summer", value_col: val_s, "is_forecast": True})
            f_rows.append({time_col: pd.to_datetime(f"{int(y)}-12-01"), "__forecast_season": "winter", value_col: val_w, "is_forecast": True})

        if "is_forecast" not in df.columns:
            df["is_forecast"] = False
        if "season" not in df.columns and "__forecast_season" in df.columns:
            df["season"] = df["__forecast_season"]
        df_f = pd.concat([df, pd.DataFrame(f_rows)], ignore_index=True)
        if "__forecast_season" in df_f.columns:
            df_f["season"] = df_f.get("season").fillna(df_f["__forecast_season"])
        note_parts.append(f"Forecast years: {', '.join(map(str, target_years))}.")
        return _strip_scratch(df_f), " ".join(note_parts)

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
    # Defensive backfill of `effective_answer_kind` so enrich() stays robust
    # when invoked outside the pipeline orchestrator (tests, tool-runner
    # entry points, etc.).  Pipeline.py populates this authoritatively after
    # Stage 0.2; here we fall back to the active analyzer's emitted value if
    # it's still unset.
    if ctx.effective_answer_kind is None and ctx.has_authoritative_question_analysis:
        ctx.effective_answer_kind = ctx.question_analysis.answer_kind

    # --- Share resolution ---
    # Structural signal: analyzer indicates share/composition intent or tool is
    # get_balancing_composition.  `analyzer_indicates_share_intent` internally
    # gates on authoritative QA, so the tool_name branch remains the analyzer-
    # absent fallback path (router / evidence planner may still pick the
    # composition tool via keyword heuristics — F1).
    share_query_detected = (
        ctx.analyzer_indicates_share_intent
        or ctx.tool_name == "get_balancing_composition"
    )
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
    # NOTE: `share_summary_override` is a *permanent* specialized LIST/SCALAR
    # formatter for share-intent queries, analogous to how §3.4 treats
    # SCENARIO and FORECAST as specialized formatters for domain-specific
    # decomposition. It is NOT legacy regex-based dispatch — the decision
    # to build it is gated on the structured analyzer signal
    # `ctx.analyzer_indicates_share_intent` (see §3.4 "Specialized formatters
    # — only for domain-specific decomposition"). The artifact decomposes
    # `share_all_ppa` into its renewable/thermal components and joins
    # per-period prices, which the domain-agnostic generic renderer
    # intentionally does not know about.
    #
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
            summary_override, grounding_hint = _build_share_summary_artifact(share_df_for_summary, ctx.plan, ctx.query)
            ctx.share_summary_override = summary_override
            if grounding_hint:
                ctx.stats_hint += f"\n\n{grounding_hint}"
            if ctx.share_summary_override:
                log.info("✅ Generated deterministic share summary override.")
        except Exception as share_err:
            log.warning(f"Share summary override failed: {share_err}")

    # --- Correlation analysis ---
    needs_correlation_analysis = False
    if ctx.has_authoritative_question_analysis:
        qa_reqs = ctx.question_analysis.analysis_requirements
        requested_metric_names = {
            getattr(metric.metric_name, "value", str(metric.metric_name or "")).strip()
            for metric in (qa_reqs.derived_metrics or [])
        }
        if "correlation_to_target" in requested_metric_names:
            log.info("Semantic intent → correlation (derived_metrics include correlation_to_target).")
            needs_correlation_analysis = True
        elif qa_reqs.needs_correlation_context:
            log.info("Semantic intent → correlation (needs_correlation_context=True).")
            needs_correlation_analysis = True
        elif qa_reqs.needs_driver_analysis:
            log.info("Semantic intent → correlation (needs_driver_analysis=True).")
            needs_correlation_analysis = True

    if needs_correlation_analysis:
        log.info("🔍 Building correlation analysis from aligned evidence")
        try:
            current_frame_results = _build_correlation_matrix_from_frame(ctx.df)
            if current_frame_results:
                ctx.correlation_results.update(current_frame_results)

            if not ctx.correlation_results:
                with ENGINE.connect() as conn:
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                    corr_df = build_balancing_correlation_df(conn)

                allowed_targets = ["p_bal_gel", "p_bal_usd"]
                allowed_drivers = [
                    "xrate", "share_import", "share_deregulated_hydro",
                    "share_regulated_hpp", "share_renewable_ppa",
                    "share_all_ppa", "share_all_renewables", "share_all_hydro",
                    "enguri_tariff_gel", "gardabani_tpp_tariff_gel",
                    "grouped_old_tpp_tariff_gel"
                ]
                corr_df = corr_df[[c for c in corr_df.columns if c in (["date"] + allowed_targets + allowed_drivers)]].copy()

                numeric_df = corr_df.drop(columns=["date"], errors="ignore").apply(pd.to_numeric, errors="coerce")
                for target in allowed_targets:
                    if target not in numeric_df.columns:
                        continue
                    series = numeric_df.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
                    if series.notna().any():
                        ctx.correlation_results[target] = series.sort_values(ascending=False).round(3).to_dict()

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
                ctx.stats_hint += "\n\n--- CORRELATION MATRIX ---\n" + json.dumps(ctx.correlation_results, indent=2)
                log.info(f"✅ Consolidated correlations computed: {list(ctx.correlation_results.keys())}")
            else:
                log.info("⚠️ No valid correlations found")

        except Exception as e:
            log.warning(f"⚠️ Correlation analysis failed: {e}")

    # --- Stage 3 per-answer_kind enrichment switch (§3.4 alignment) ---
    # All per-kind enrichment dispatches from the single source of truth
    # `ctx.effective_answer_kind` (populated by pipeline.py after Stage 0.2;
    # falls back to analyzer's emitted value for shadow/failed runs — F1).
    #
    # Structure:
    #   * FORECAST — CAGR enrichment runs independently of the analytical
    #     branch below, so "forecast + MoM/YoY" queries get both forecast
    #     projection AND derived-metric evidence.
    #   * EXPLANATION vs standalone analysis are mutually exclusive: causal
    #     reasoning (_build_why_context) owns the full enrichment path for
    #     why/how queries; standalone analysis computes MoM/YoY/scenario for
    #     every other analytical kind that still requests derived metrics.
    #   * LIST / KNOWLEDGE / CLARIFY / SCALAR-without-derived-metrics: no
    #     Stage 3 enrichment — they route straight to the generic renderer.
    answer_kind = ctx.effective_answer_kind

    if answer_kind == AnswerKind.FORECAST and not ctx.df.empty:
        try:
            ctx.df, _forecast_note = _generate_cagr_forecast(ctx.df, ctx.query)
            ctx.stats_hint += f"\n\n--- FORECAST NOTE ---\n{_forecast_note}"
            log.info(_forecast_note)
        except Exception as _e:
            log.warning(f"Forecast generation failed: {_e}")

    if answer_kind == AnswerKind.EXPLANATION and not ctx.df.empty:
        try:
            _build_why_context(ctx)
        except Exception as _e:
            log.warning(f"'Why' reasoning context build failed: {_e}")
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

    # --- Phase 15 (§16.4.5): generalized derived-chart overrides ---
    # Only fires when Stage 0.2 is authoritative and the contract indicates a
    # derived view (MoM/YoY, indexed growth, decomposition, seasonal, forecast).
    # Writes to ``ctx.chart_override_specs`` which takes precedence over the
    # legacy single-spec path in ``_apply_chart_override`` (Phase 10).
    if not ctx.df.empty and ctx.has_authoritative_question_analysis:
        try:
            from agent.derived_chart_builder import dispatch_derived_chart
            derived_specs = dispatch_derived_chart(ctx)
            if derived_specs:
                ctx.chart_override_specs = derived_specs
                log.info(
                    "📊 Derived chart override assigned (%d specs) via dispatch_derived_chart",
                    len(derived_specs),
                )
        except Exception as _e:
            log.warning("Derived chart dispatch failed: %s", _e)

    # --- Trendline detection ---
    # Mirrors the FORECAST branch of the switch above — reuses the local
    # `answer_kind` so keyword-derived forecast queries also get trendline
    # extension on analyzer failure (F1).
    #
    # Fix 3: if `_generate_cagr_forecast` already produced `is_forecast=True`
    # projection rows, skip trendline calculation entirely — those projection
    # rows already drive the chart's forecast line and overlaying linear
    # trendlines on top creates redundant / conflicting visuals.  Trendlines
    # remain as a fallback when CAGR produced nothing.
    _cagr_produced_rows = bool(
        answer_kind == AnswerKind.FORECAST
        and "is_forecast" in getattr(ctx.df, "columns", [])
        and bool(ctx.df["is_forecast"].fillna(False).any())
    )
    ctx.add_trendlines = (answer_kind == AnswerKind.FORECAST) and not _cagr_produced_rows
    if answer_kind == AnswerKind.FORECAST and _cagr_produced_rows:
        log.info(
            "📈 Skipping trendline extension: CAGR forecast rows already present "
            "(%d is_forecast=True rows).",
            int(ctx.df["is_forecast"].fillna(False).sum()),
        )

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
    # Reuse one normalized time-series snapshot for standalone MoM/YoY analysis.
    t_series_col = _find_time_series_column(ctx.df)
    if not t_series_col:
        return None

    df = ctx.df.copy()
    df[t_series_col] = normalize_period_series(df[t_series_col])
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


# Scenario metric registry keys (kept as strings to mirror METRIC_REGISTRY).
# Scenario math operates on the whole series, so these requests do not need
# the MoM/YoY current/previous row prep and must survive when
# `_prepare_timeseries_rows()` fails to resolve a target period.
_SCENARIO_METRIC_NAME_STRINGS = frozenset({
    "scenario_scale",
    "scenario_offset",
    "scenario_payoff",
})


def _has_active_scenario_request(ctx: QueryContext) -> bool:
    """Return True when any active derived-metric request is a scenario metric."""
    try:
        return any(
            str(req.get("metric_name", "")) in _SCENARIO_METRIC_NAME_STRINGS
            for req in _active_analysis_requests(ctx)
        )
    except Exception:
        return False


def _needs_standalone_analysis(ctx: QueryContext) -> bool:
    """Return True when derived metrics should be computed outside why-mode."""
    qa = ctx.question_analysis if ctx.has_authoritative_question_analysis else None
    if qa is not None:
        if qa.analysis_requirements.derived_metrics:
            return True
        if qa.classification.analysis_mode.value == "analyst":
            return True
    # Fallback: heuristic mode detection still triggers analysis
    # even when question_analysis failed validation.
    if ctx.mode == "analyst":
        return True
    # Scenario fallback requests (heuristic scenario extraction) must also
    # trigger standalone enrichment so scenario answers survive when the
    # analyzer produced no structured derived_metrics.
    if _has_active_scenario_request(ctx):
        return True
    return False


def _build_standalone_analysis_evidence(ctx: QueryContext) -> None:
    """Compute derived metrics (MoM/YoY/scenario) for non-why analyst-mode queries.

    Populates ``ctx.analysis_evidence`` and appends the evidence block to
    ``ctx.stats_hint`` so that computed values enter the grounding corpus.

    Scenario metrics operate on the whole series and do not require
    MoM/YoY current/previous row prep. When ``_prepare_timeseries_rows()``
    cannot resolve a target period (e.g. no explicit year in the query,
    or the DataFrame lacks a recognised time column), we still run scenario
    dispatch against the full frame so payoff / scale / offset answers
    do not silently drop.
    """
    setup = _prepare_timeseries_rows(ctx)
    if setup is not None:
        df, time_col, current_ts, current_row, previous_ts, previous_row = setup
    elif _has_active_scenario_request(ctx) and ctx.df is not None and not ctx.df.empty:
        # Degraded fallback for scenario-only enrichment: no current/previous
        # rows, but the scenario compute function only needs `df` and
        # `time_col`. MoM/YoY requests in this mode will return None
        # gracefully and be dropped from the evidence rows.
        df = ctx.df.copy()
        time_col = _find_time_series_column(df) or ""
        if time_col and time_col in df.columns:
            df[time_col] = normalize_period_series(df[time_col])
            df = df.dropna(subset=[time_col]).sort_values(time_col)
        current_ts = None
        current_row = pd.DataFrame()
        previous_ts = None
        previous_row = pd.DataFrame()
    else:
        return
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


def _format_regulated_plant_sales_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Normalize regulated plant-sales rows for compact prompt inclusion."""
    if df is None or df.empty:
        return []

    working = df.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
    sort_cols = [c for c in ["date", "regulated_group", "balancing_quantity", "plant"] if c in working.columns]
    if sort_cols:
        ascending = [False if c == "balancing_quantity" else True for c in sort_cols]
        working = working.sort_values(sort_cols, ascending=ascending)

    records: list[dict[str, Any]] = []
    for row in working.to_dict(orient="records"):
        record = {
            "regulated_group": row.get("regulated_group"),
            "plant": row.get("plant"),
            "entity_code": row.get("entity_code"),
        }
        if row.get("date") is not None and not pd.isna(row.get("date")):
            record["date"] = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        for field in (
            "balancing_quantity",
            "tariff_gel",
            "tariff_usd",
            "share_of_total_balancing",
            "share_within_group",
        ):
            value = row.get(field)
            if value is None or pd.isna(value):
                continue
            record[field] = round(float(value), 6)
        records.append(record)
    return records


# Why-context assembly adds comparison drivers, historical bounds, and plant-level evidence to stats_hint.
def _build_why_context(ctx: QueryContext) -> None:
    """Build causal context for 'why' queries. Modifies ctx.stats_hint."""
    why_ctx: Dict[str, Any] = {"notes": [], "signals": {}}

    t_series_col = _find_time_series_column(ctx.df)
    if not t_series_col:
        return

    df = ctx.df.copy()
    df[t_series_col] = normalize_period_series(df[t_series_col])
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
    regulated_plant_sales_df = pd.DataFrame()
    regulated_plant_sales_block: dict[str, Any] = {}

    if target_ts is not None:
        plant_window = [ts for ts in [prev_ts, target_ts] if ts is not None]
        if plant_window:
            try:
                with ENGINE.connect() as conn:
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                    regulated_plant_sales_df = compute_regulated_plant_sales(
                        conn,
                        start_date=min(plant_window).strftime("%Y-%m-%d"),
                        end_date=max(plant_window).strftime("%Y-%m-%d"),
                    )
            except Exception as plant_err:
                log.warning("Regulated plant sales evidence build failed: %s", plant_err)

    if not regulated_plant_sales_df.empty:
        regulated_plant_sales_df = regulated_plant_sales_df.copy()
        regulated_plant_sales_df["date"] = pd.to_datetime(regulated_plant_sales_df["date"], errors="coerce")
        current_plants = regulated_plant_sales_df[
            regulated_plant_sales_df["date"].dt.to_period("M") == target_ts.to_period("M")
        ] if target_ts is not None else pd.DataFrame()
        previous_plants = regulated_plant_sales_df[
            regulated_plant_sales_df["date"].dt.to_period("M") == prev_ts.to_period("M")
        ] if prev_ts is not None else pd.DataFrame()
        if not current_plants.empty:
            regulated_plant_sales_block["current_period"] = _format_regulated_plant_sales_records(current_plants)
        if not previous_plants.empty:
            regulated_plant_sales_block["previous_period"] = _format_regulated_plant_sales_records(previous_plants)
        if regulated_plant_sales_block:
            why_ctx["regulated_plant_sales"] = regulated_plant_sales_block

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
    if regulated_plant_sales_block.get("current_period"):
        why_ctx["notes"].append(
            "Regulated plant sales detail is available for the focal month; use the named HPP/TPP sellers to explain the regulated source layer."
        )

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
    component_pressure_summary = _build_component_pressure_summary(
        cur_row,
        prev_row,
        cur_gel=cur_gel,
        prev_gel=prev_gel,
        cur_usd=cur_usd,
        prev_usd=prev_usd,
        value_getter=_get_val,
    )
    component_pressure_df = (
        pd.DataFrame(component_pressure_summary) if component_pressure_summary else pd.DataFrame()
    )

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
    if not component_pressure_df.empty:
        combined_prov_df = pd.concat([combined_prov_df, component_pressure_df], ignore_index=True, sort=False)
    if not regulated_plant_sales_df.empty:
        combined_prov_df = pd.concat([combined_prov_df, regulated_plant_sales_df], ignore_index=True, sort=False)
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
        component_pressure_count=len(component_pressure_summary),
        regulated_plant_rows=len(regulated_plant_sales_df),
        signals=why_ctx.get("signals", {}),
    )
    trace_detail(
        log,
        ctx,
        "stage_3_analyzer_enrich",
        "artifact",
        debug=True,
        why_context=why_ctx,
        component_pressure_summary=component_pressure_summary,
        analysis_evidence=ctx.analysis_evidence,
    )

    # PRIORITY 1: Causal Context (High value, small size)
    # Put this first so it survives prompt truncation.
    ctx.stats_hint += "\n\n--- CAUSAL CONTEXT ---\n" + json.dumps(why_ctx, default=str, indent=2)

    # PRIORITY 2: Deterministic component pressure summary (High value, compact)
    if component_pressure_summary:
        ctx.stats_hint += (
            "\n\n--- COMPONENT PRESSURE SUMMARY ---\n"
            + json.dumps(component_pressure_summary, default=str, indent=2)
        )

    # PRIORITY 3: Regulated plant-level seller evidence (compact but valuable)
    if regulated_plant_sales_block:
        ctx.stats_hint += (
            "\n\n--- REGULATED PLANT SALES ---\n"
            + json.dumps(regulated_plant_sales_block, default=str, indent=2)
        )

    # PRIORITY 4: Detailed Evidence (Lower value, large size)
    # Prune to top 12 to reduce prompt bloat and truncation risk.
    if ctx.analysis_evidence:
        evidence_subset = ctx.analysis_evidence[:12]
        ctx.stats_hint += "\n\n--- DERIVED ANALYSIS EVIDENCE (TOP 12) ---\n" + json.dumps(evidence_subset, default=str, indent=2)

    log.info("Why-context (prioritized) attached to stats_hint.")


def _precalculate_trendlines(ctx: QueryContext, cols_labeled: list) -> None:
    """Pre-calculate trendlines for forecast answer generation."""
    try:
        from visualization.chart_builder import calculate_trendline

        time_key = next((c for c in ctx.cols if any(k in c.lower() for k in ["date", "year", "month", "period", "თვე", "წელი", "თარიღი"])), None)
        season_col = next((c for c in ctx.cols if c.lower() in ["season", "სეზონი"]), None)

        if time_key and time_key in ctx.df.columns:
            ctx.df[time_key] = normalize_period_series(ctx.df[time_key])

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

        def _accept_trendline(td: Optional[Dict[str, Any]]) -> bool:
            """R² gate — reject trendlines whose fit explains < 30% of variance."""
            if not td or not td.get("dates") or not td.get("values"):
                return False
            return float(td.get("r_squared", 0.0)) >= _TRENDLINE_MIN_R_SQUARED

        if season_col and season_col in df_calc.columns:
            log.info("📈 Seasonal forecast detected - calculating separate trendlines")
            seasons = df_calc[season_col].dropna().unique()
            for season in seasons:
                season_df = df_calc[df_calc[season_col] == season].copy()
                for col in num_cols:
                    td = calculate_trendline(season_df, time_key, col, extend_to_date=ctx.trendline_extend_to)
                    if not _accept_trendline(td):
                        if td is not None:
                            log.info(
                                "📈 Skipping trendline %s (%s): R²=%.3f < %.2f threshold",
                                col, season, float(td.get("r_squared", 0.0)), _TRENDLINE_MIN_R_SQUARED,
                            )
                        continue
                    forecast_key = f"{col}_{season}"
                    trendline_forecasts[forecast_key] = {
                        "target_date": td["dates"][-1],
                        "forecast_value": round(td["values"][-1], 2),
                        "equation": td["equation"],
                        "r_squared": round(td["r_squared"], 3),
                        "season": season,
                    }
        else:
            # Overall (non-seasonal) trendlines
            for col in num_cols:
                td = calculate_trendline(df_calc, time_key, col, extend_to_date=ctx.trendline_extend_to)
                if not _accept_trendline(td):
                    if td is not None:
                        log.info(
                            "📈 Skipping trendline %s (overall): R²=%.3f < %.2f threshold",
                            col, float(td.get("r_squared", 0.0)), _TRENDLINE_MIN_R_SQUARED,
                        )
                    continue
                trendline_forecasts[col] = {
                    "target_date": td["dates"][-1],
                    "forecast_value": round(td["values"][-1], 2),
                    "equation": td["equation"],
                    "r_squared": round(td["r_squared"], 3),
                }

            # Derive seasonal split from month when no explicit season column
            try:
                dates = pd.to_datetime(df_calc[time_key], errors="coerce")
                months = dates.dt.month
                if months.notna().sum() > 6:
                    summer_df = df_calc[months.isin(SUMMER_MONTHS)].copy()
                    winter_df = df_calc[~months.isin(SUMMER_MONTHS)].copy()
                    for season_label, s_df in [("summer", summer_df), ("winter", winter_df)]:
                        if len(s_df) < 3:
                            continue
                        for col in num_cols:
                            td = calculate_trendline(s_df, time_key, col, extend_to_date=ctx.trendline_extend_to)
                            if not _accept_trendline(td):
                                if td is not None:
                                    log.info(
                                        "📈 Skipping trendline %s (%s): R²=%.3f < %.2f threshold",
                                        col, season_label, float(td.get("r_squared", 0.0)),
                                        _TRENDLINE_MIN_R_SQUARED,
                                    )
                                continue
                            forecast_key = f"{col}_{season_label}"
                            trendline_forecasts[forecast_key] = {
                                "target_date": td["dates"][-1],
                                "forecast_value": round(td["values"][-1], 2),
                                "equation": td["equation"],
                                "r_squared": round(td["r_squared"], 3),
                                "season": season_label,
                            }
                    if any(fi.get("season") for fi in trendline_forecasts.values()):
                        log.info("📈 Derived seasonal trendlines from month-based split")
            except Exception as seasonal_err:
                log.warning(f"Seasonal split derivation failed: {seasonal_err}")

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
