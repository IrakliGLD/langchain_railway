"""Metric computation registry for derived-metric evidence records.

Each metric type is implemented as a standalone function and registered in
``METRIC_REGISTRY``.  ``_build_requested_analysis_evidence()`` in
``analyzer.py`` dispatches to these functions instead of a monolithic
if/elif chain.
"""
from __future__ import annotations

import dataclasses as dc
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from config_metrics.metric_config import METRIC_VALUE_ALIASES, SEMANTIC_TO_COLUMNS, SUMMER_MONTHS

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# Shared context passed to every metric compute function
# ---------------------------------------------------------------------------

@dc.dataclass
class MetricContext:
    """Bundle of pre-resolved data needed by metric compute functions."""

    df: pd.DataFrame
    time_col: str
    current_ts: Optional[pd.Timestamp]
    current_row: pd.DataFrame
    previous_ts: Optional[pd.Timestamp]
    previous_row: pd.DataFrame
    cur_shares: dict[str, float]
    prev_shares: dict[str, float]
    yoy_row: pd.DataFrame
    yoy_ts: Optional[pd.Timestamp]
    yoy_shares: dict[str, float]
    correlation_results: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# Helpers (promoted from closures in _build_requested_analysis_evidence)
# ---------------------------------------------------------------------------

def _metric_aliases(metric: str) -> list[str]:
    metric_name = str(metric or "").strip()
    if not metric_name:
        return []
    # 1. Exact column-level alias (e.g. "p_bal_gel" → ["p_bal_gel", "balancing_price_gel"])
    if metric_name in METRIC_VALUE_ALIASES:
        return METRIC_VALUE_ALIASES[metric_name]
    # 2. Semantic tool name (e.g. "balancing" → ["p_bal_gel", "p_bal_usd"])
    if metric_name in SEMANTIC_TO_COLUMNS:
        return SEMANTIC_TO_COLUMNS[metric_name]
    # 3. Fallback: treat as literal column name
    return [metric_name]


def row_value(frame: pd.DataFrame, metric_name: str) -> Optional[float]:
    """Extract a float value from a single-row DataFrame using metric aliases."""
    if frame is None or frame.empty:
        return None
    for candidate in _metric_aliases(metric_name):
        if candidate not in frame.columns:
            continue
        value = frame[candidate].iloc[0]
        if value is None or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def share_value(snapshot: dict[str, float], metric_name: str) -> Optional[float]:
    """Extract a share value from a snapshot dict."""
    value = snapshot.get(metric_name)
    return None if value is None else float(value)


def _period_str(ts: Optional[pd.Timestamp]) -> Optional[str]:
    """Safely convert a timestamp to a string, returning None for NaT/None."""
    if ts is None or pd.isna(ts):
        return None
    return str(ts)


def _scenario_formula(metric_name: str, metric: str, factor: float, volume: float, agg: str, n: int) -> str:
    """Build a human-readable formula string for scenario evidence records."""
    if metric_name == "scenario_scale":
        return f"{metric} * {factor}, {agg} over {n} periods"
    elif metric_name == "scenario_offset":
        return f"{metric} + {factor}, {agg} over {n} periods"
    else:
        return f"({factor} - {metric}) * {volume}, {agg} over {n} periods"


def _apply_season_filter(df: pd.DataFrame, time_col: str, season: Optional[str]) -> pd.DataFrame:
    """Filter a DataFrame to summer or winter months only."""
    if not season or season == "full":
        return df
    if time_col not in df.columns:
        return df
    dt = pd.to_datetime(df[time_col], errors="coerce")
    if season == "summer":
        return df[dt.dt.month.isin(SUMMER_MONTHS)]
    elif season == "winter":
        return df[~dt.dt.month.isin(SUMMER_MONTHS)]
    return df


# ---------------------------------------------------------------------------
# Type alias for metric compute functions
# ---------------------------------------------------------------------------

# Each function receives (request_dict, base_record_dict, MetricContext)
# and returns the updated record dict, or None to skip.
MetricComputeFn = Callable[[Dict[str, Any], Dict[str, Any], MetricContext], Optional[Dict[str, Any]]]


# ---------------------------------------------------------------------------
# Metric compute functions
# ---------------------------------------------------------------------------

def compute_mom(
    request: Dict[str, Any],
    record: Dict[str, Any],
    mctx: MetricContext,
) -> Optional[Dict[str, Any]]:
    """Month-over-month absolute or percent change."""
    metric = record["metric"]
    is_share = metric.startswith("share_")
    cur_val = share_value(mctx.cur_shares, metric) if is_share else row_value(mctx.current_row, metric)
    prev_val = share_value(mctx.prev_shares, metric) if is_share else row_value(mctx.previous_row, metric)
    if cur_val is None or prev_val is None:
        return None
    # Compare the latest resolved observation against the immediately preceding one.
    delta = cur_val - prev_val
    pct = None if abs(prev_val) < 1e-12 else (delta / prev_val) * 100.0
    _cur_period = _period_str(mctx.current_ts)
    _prev_period = _period_str(mctx.previous_ts)
    record.update({
        "comparison_period": _prev_period,
        "current_value": round(cur_val, 6),
        "previous_value": round(prev_val, 6),
        "absolute_change": round(delta, 6),
        "percent_change": None if pct is None else round(pct, 4),
        "formula": "current_value - previous_value",
        "source_cells": [
            {"column": metric, "period": _cur_period, "value": round(cur_val, 6), "role": "current"},
            {"column": metric, "period": _prev_period, "value": round(prev_val, 6), "role": "previous"},
        ],
    })
    return record


def compute_yoy(
    request: Dict[str, Any],
    record: Dict[str, Any],
    mctx: MetricContext,
) -> Optional[Dict[str, Any]]:
    """Year-over-year absolute or percent change."""
    metric = record["metric"]
    is_share = metric.startswith("share_")
    cur_val = share_value(mctx.cur_shares, metric) if is_share else row_value(mctx.current_row, metric)
    yoy_val = share_value(mctx.yoy_shares, metric) if is_share else row_value(mctx.yoy_row, metric)
    if cur_val is None or yoy_val is None:
        return None
    # YoY uses the same calendar period from the prior year as the reference point.
    delta = cur_val - yoy_val
    pct = None if abs(yoy_val) < 1e-12 else (delta / yoy_val) * 100.0
    _cur_period = _period_str(mctx.current_ts)
    _yoy_period = _period_str(mctx.yoy_ts)
    record.update({
        "comparison_period": _yoy_period,
        "current_value": round(cur_val, 6),
        "previous_value": round(yoy_val, 6),
        "absolute_change": round(delta, 6),
        "percent_change": None if pct is None else round(pct, 4),
        "formula": "current_value - previous_value_same_period_last_year",
        "source_cells": [
            {"column": metric, "period": _cur_period, "value": round(cur_val, 6), "role": "current"},
            {"column": metric, "period": _yoy_period, "value": round(yoy_val, 6), "role": "yoy_previous"},
        ],
    })
    return record


def compute_share_delta_mom(
    request: Dict[str, Any],
    record: Dict[str, Any],
    mctx: MetricContext,
) -> Optional[Dict[str, Any]]:
    """Month-over-month share delta (always share-based)."""
    metric = record["metric"]
    cur_val = share_value(mctx.cur_shares, metric)
    prev_val = share_value(mctx.prev_shares, metric)
    if cur_val is None or prev_val is None:
        return None
    delta = cur_val - prev_val
    _cur_period = _period_str(mctx.current_ts)
    _prev_period = _period_str(mctx.previous_ts)
    record.update({
        "comparison_period": _prev_period,
        "current_value": round(cur_val, 6),
        "previous_value": round(prev_val, 6),
        "absolute_change": round(delta, 6),
        "formula": "current_share - previous_share",
        "source_cells": [
            {"column": metric, "period": _cur_period, "value": round(cur_val, 6), "role": "current"},
            {"column": metric, "period": _prev_period, "value": round(prev_val, 6), "role": "previous"},
        ],
    })
    return record


def compute_correlation(
    request: Dict[str, Any],
    record: Dict[str, Any],
    mctx: MetricContext,
) -> Optional[Dict[str, Any]]:
    """Correlation of a metric to a target metric."""
    metric = record["metric"]
    corr_target = record.get("target_metric") or "p_bal_gel"

    # Resolve target: try semantic aliases then exact match
    target_candidates = SEMANTIC_TO_COLUMNS.get(corr_target, [corr_target])
    corr_map: Dict[str, Any] = {}
    resolved_target = corr_target
    for t in target_candidates:
        if t in mctx.correlation_results:
            corr_map = mctx.correlation_results[t]
            resolved_target = t
            break
    if not corr_map:
        corr_map = mctx.correlation_results.get(corr_target, {})

    # Resolve metric: try exact match first, then semantic aliases
    corr_value = corr_map.get(metric)
    resolved_metric = metric
    if corr_value is None:
        for candidate in SEMANTIC_TO_COLUMNS.get(metric, []):
            if candidate in corr_map:
                corr_value = corr_map[candidate]
                resolved_metric = candidate
                break
    if corr_value is None:
        return None

    record.update({
        "target_metric": resolved_target,
        "correlation_value": round(float(corr_value), 6),
        "formula": f"corr({resolved_metric}, {resolved_target}) over available series",
        "source_column": resolved_metric,
        "source_row_count": len(mctx.df),
        "source_cells": [
            {"column": resolved_metric, "role": "source_series", "row_count": len(mctx.df)},
            {"column": resolved_target, "role": "target_series", "row_count": len(mctx.df)},
        ],
    })
    return record


def compute_trend_slope(
    request: Dict[str, Any],
    record: Dict[str, Any],
    mctx: MetricContext,
) -> Optional[Dict[str, Any]]:
    """Linear trend slope over ordered observations."""
    metric = record["metric"]
    candidates = _metric_aliases(metric)
    season = request.get("season")
    df = _apply_season_filter(mctx.df, mctx.time_col, season)
    value_col = next((c for c in candidates if c in df.columns), None)
    if value_col is None or len(df) < 2:
        return None
    numeric_series = pd.to_numeric(df[value_col], errors="coerce")
    valid = numeric_series.notna()
    if valid.sum() < 2:
        return None
    # Fit a simple linear trend over ordered observations to summarize direction and pace.
    x = np.arange(valid.sum(), dtype=float)
    y = numeric_series[valid].astype(float).to_numpy()
    slope = np.polyfit(x, y, deg=1)[0]
    season_label = f" ({season})" if season and season != "full" else ""
    record.update({
        "trend_slope": round(float(slope), 6),
        "season": season if season and season != "full" else None,
        "formula": f"linear_slope({metric}){season_label} over ordered observations",
        "source_column": value_col,
        "source_row_count": int(valid.sum()),
        "source_cells": [
            {
                "column": value_col,
                "role": "trend_series",
                "row_count": int(valid.sum()),
                "min_value": round(float(y.min()), 6),
                "max_value": round(float(y.max()), 6),
            },
        ],
    })
    return record


def compute_scenario(
    request: Dict[str, Any],
    record: Dict[str, Any],
    mctx: MetricContext,
) -> Optional[Dict[str, Any]]:
    """Scenario scale, offset, or payoff computation."""
    metric = record["metric"]
    metric_name = record["derived_metric_name"]
    candidates = _metric_aliases(metric)
    season = request.get("season")
    df = _apply_season_filter(mctx.df, mctx.time_col, season)
    value_col = next((c for c in candidates if c in df.columns), None)
    if value_col is None:
        return None
    raw = pd.to_numeric(df[value_col], errors="coerce")
    valid_mask = raw.notna()
    if valid_mask.sum() == 0:
        return None
    series = raw[valid_mask]

    factor = float(request.get("scenario_factor", 0) or 0)
    volume = float(request.get("scenario_volume", 1) or 1)
    agg_name = str(request.get("scenario_aggregation", "sum") or "sum")

    # Skip identity transforms
    if metric_name == "scenario_scale" and abs(factor - 1.0) < 1e-9:
        return None
    if metric_name == "scenario_offset" and abs(factor) < 1e-9:
        return None

    if metric_name == "scenario_scale":
        # Scale scenarios multiply every observation by the requested factor.
        scenario_series = series * factor
    elif metric_name == "scenario_offset":
        # Offset scenarios add or subtract a fixed amount per observation.
        scenario_series = series + factor
    else:  # scenario_payoff
        # Payoff scenarios compare the observed market price against a strike/reference price.
        scenario_series = (factor - series) * volume

    agg_result = float(getattr(scenario_series, agg_name)())

    # Split payoff results into positive and negative buckets so answers can explain both sides.
    if metric_name == "scenario_payoff":
        _pos_mask = scenario_series > 0
        _neg_mask = scenario_series < 0
        positive_sum = round(float(scenario_series[_pos_mask].sum()), 2) if _pos_mask.any() else 0.0
        negative_sum = round(float(scenario_series[_neg_mask].sum()), 2) if _neg_mask.any() else 0.0
        positive_count = int(_pos_mask.sum())
        negative_count = int(_neg_mask.sum())
        market_component_result = float(getattr(series * volume, agg_name)())
        combined_total_result = float(getattr((series * volume) + scenario_series, agg_name)())
    else:
        positive_sum = None
        negative_sum = None
        positive_count = None
        negative_count = None
        market_component_result = None
        combined_total_result = None

    # Baseline deltas only make sense when the scenario modifies the observed series directly.
    if metric_name in ("scenario_scale", "scenario_offset"):
        baseline_result = float(getattr(series, agg_name)())
        delta = agg_result - baseline_result
        delta_pct = (delta / baseline_result * 100) if abs(baseline_result) > 1e-12 else None
    else:
        baseline_result = None
        delta = None
        delta_pct = None

    period_range = ""
    if mctx.time_col in df.columns:
        time_vals = df.loc[valid_mask[valid_mask].index, mctx.time_col]
        if not time_vals.empty:
            period_range = f"{time_vals.iloc[0]} to {time_vals.iloc[-1]}"

    record.update({
        "record_type": "scenario",
        "source_column": value_col,
        "source_row_count": int(valid_mask.sum()),
        "scenario_factor": factor,
        "scenario_volume": volume if metric_name == "scenario_payoff" else None,
        "scenario_aggregation": agg_name,
        "aggregate_result": round(agg_result, 2),
        "baseline_aggregate": round(baseline_result, 2) if baseline_result is not None else None,
        "delta_aggregate": round(delta, 2) if delta is not None else None,
        "delta_percent": round(delta_pct, 2) if delta_pct is not None else None,
        "row_count": int(valid_mask.sum()),
        "period_range": period_range,
        "formula": _scenario_formula(metric_name, metric, factor, volume, agg_name, int(valid_mask.sum())),
        "min_period_value": round(float(scenario_series.min()), 2),
        "max_period_value": round(float(scenario_series.max()), 2),
        "mean_period_value": round(float(scenario_series.mean()), 2),
        "positive_sum": positive_sum,
        "negative_sum": negative_sum,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "market_component_aggregate": (
            round(market_component_result, 2) if market_component_result is not None else None
        ),
        "combined_total_aggregate": (
            round(combined_total_result, 2) if combined_total_result is not None else None
        ),
        "source_cells": [
            {
                "column": value_col,
                "role": "scenario_series",
                "row_count": int(valid_mask.sum()),
                "min_value": round(float(series.min()), 6),
                "max_value": round(float(series.max()), 6),
            },
        ],
    })
    return record


# ---------------------------------------------------------------------------
# Registry: metric_name → compute function
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, MetricComputeFn] = {
    "mom_absolute_change": compute_mom,
    "mom_percent_change": compute_mom,
    "yoy_absolute_change": compute_yoy,
    "yoy_percent_change": compute_yoy,
    "share_delta_mom": compute_share_delta_mom,
    "correlation_to_target": compute_correlation,
    "trend_slope": compute_trend_slope,
    "scenario_scale": compute_scenario,
    "scenario_offset": compute_scenario,
    "scenario_payoff": compute_scenario,
}


def dispatch_metric(
    request: Dict[str, Any],
    record: Dict[str, Any],
    mctx: MetricContext,
) -> Optional[Dict[str, Any]]:
    """Look up and invoke the registered compute function for a metric request.

    Returns the populated record dict on success, or None if the metric
    name is unknown or the computation cannot proceed (missing data).
    """
    metric_name = record.get("derived_metric_name", "")
    fn = METRIC_REGISTRY.get(metric_name)
    if fn is None:
        return None
    return fn(request, record, mctx)
