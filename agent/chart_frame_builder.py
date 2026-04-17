"""Deterministic transforms for chart-specific frames."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import SUMMER_MONTHS


def build_chart_frame(
    df: pd.DataFrame,
    *,
    time_key: Optional[str],
    category_cols: List[str],
    num_cols: List[str],
    dim_map: Dict[str, str],
    time_grain: Optional[str] = None,
    measure_transform: str = "raw",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return a chart-specific frame plus metadata about applied transforms."""

    if df.empty or not num_cols:
        return df.copy(), {}

    working = df.copy()
    meta: Dict[str, Any] = {}

    if time_key and time_key in working.columns:
        working[time_key] = pd.to_datetime(working[time_key], errors="coerce")

    if time_grain and time_grain != "raw":
        aggregation_mode = measure_transform if measure_transform in {"sum", "avg"} else None
        working, grain_meta = _apply_time_grain(
            working,
            time_key=time_key,
            category_cols=category_cols,
            num_cols=num_cols,
            dim_map=dim_map,
            time_grain=time_grain,
            aggregation_mode=aggregation_mode,
        )
        meta.update(grain_meta)

    if measure_transform in {"sum", "avg"}:
        meta["measureTransform"] = measure_transform
        return working, meta

    if measure_transform and measure_transform != "raw":
        working, transform_meta = _apply_measure_transform(
            working,
            time_key=time_key,
            category_cols=category_cols,
            num_cols=num_cols,
            measure_transform=measure_transform,
        )
        meta.update(transform_meta)

    return working, meta


def _apply_time_grain(
    df: pd.DataFrame,
    *,
    time_key: Optional[str],
    category_cols: List[str],
    num_cols: List[str],
    dim_map: Dict[str, str],
    time_grain: str,
    aggregation_mode: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not time_key or time_key not in df.columns:
        return df, {}

    working = df.copy()
    valid_time = working[time_key].notna()
    if not valid_time.any():
        return df, {}

    bucket = "_chart_bucket"

    if time_grain == "year":
        working[bucket] = working[time_key].dt.to_period("Y").dt.to_timestamp()
    elif time_grain == "quarter":
        working[bucket] = working[time_key].dt.to_period("Q").dt.to_timestamp()
    elif time_grain == "month":
        working[bucket] = working[time_key].dt.to_period("M").dt.to_timestamp()
    elif time_grain == "day":
        working[bucket] = working[time_key].dt.normalize()
    elif time_grain == "season":
        years = working[time_key].dt.year.dropna().unique()
        season_label = working[time_key].dt.month.map(
            lambda month: "summer" if month in SUMMER_MONTHS else "winter"
        )
        if len(years) > 1:
            working[bucket] = working[time_key].dt.year.astype("Int64").astype(str) + "-" + season_label
        else:
            working[bucket] = season_label
    else:
        return df, {}

    group_keys = [bucket] + [col for col in category_cols if col in working.columns]
    agg_map = {
        col: (
            "sum"
            if aggregation_mode == "sum"
            else "mean"
            if aggregation_mode == "avg"
            else "sum"
            if dim_map.get(col) == "energy_qty"
            else "mean"
        )
        for col in num_cols
        if col in working.columns
    }
    if not agg_map:
        return df, {}

    grouped = (
        working.groupby(group_keys, dropna=False, as_index=False)
        .agg(agg_map)
        .rename(columns={bucket: time_key})
    )

    if time_grain != "season":
        grouped = grouped.sort_values(time_key)

    return grouped.reset_index(drop=True), {"aggregation": time_grain}


def _apply_measure_transform(
    df: pd.DataFrame,
    *,
    time_key: Optional[str],
    category_cols: List[str],
    num_cols: List[str],
    measure_transform: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not time_key or time_key not in df.columns:
        return df, {}

    working = df.copy()
    group_cols = [col for col in category_cols if col in working.columns]
    sort_cols = group_cols + [time_key]
    working = working.sort_values(sort_cols)

    if measure_transform == "index_100":
        for col in num_cols:
            working[col] = _group_transform(
                working,
                group_cols,
                col,
                lambda series: _index_base_100(series),
            )
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "Index (base=100)",
        }

    if measure_transform == "cagr":
        cagr_meta: Dict[str, float] = {}
        for col in num_cols:
            working[col] = _group_transform(
                working,
                group_cols,
                col,
                lambda series: _index_base_100(series),
            )
            cagr_value = _estimate_cagr(working[col])
            if cagr_value is not None:
                cagr_meta[col] = cagr_value
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "Index (base=100)",
            "cagrBySeries": cagr_meta,
        }

    lag = _infer_period_lag(working[time_key])
    if measure_transform == "share_of_total":
        if len(num_cols) == 1 and group_cols:
            value_col = num_cols[0]
            partition_cols = [time_key] if time_key else []
            totals = working.groupby(partition_cols, dropna=False)[value_col].transform("sum")
            totals = totals.replace(0, pd.NA)
            working[value_col] = (pd.to_numeric(working[value_col], errors="coerce") / totals) * 100.0
        else:
            totals = working[num_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            totals = totals.replace(0, pd.NA)
            for col in num_cols:
                working[col] = (pd.to_numeric(working[col], errors="coerce") / totals) * 100.0
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "Share of total (%)",
        }

    if measure_transform == "mom_delta":
        for col in num_cols:
            working[col] = _group_transform(working, group_cols, col, lambda series: series.diff(1))
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "MoM change",
        }

    if measure_transform == "mom_pct":
        for col in num_cols:
            working[col] = _group_transform(working, group_cols, col, lambda series: series.pct_change(1) * 100.0)
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "MoM change (%)",
        }

    if measure_transform == "yoy_delta":
        for col in num_cols:
            working[col] = _group_transform(working, group_cols, col, lambda series: series.diff(lag))
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "YoY change",
        }

    if measure_transform == "yoy_pct":
        for col in num_cols:
            working[col] = _group_transform(working, group_cols, col, lambda series: series.pct_change(lag) * 100.0)
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "YoY change (%)",
        }

    return df, {}


def _group_transform(
    df: pd.DataFrame,
    group_cols: List[str],
    column: str,
    func,
) -> pd.Series:
    if group_cols:
        return df.groupby(group_cols, dropna=False)[column].transform(func)
    return func(df[column])


def _index_base_100(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    first_valid = numeric.dropna()
    if first_valid.empty:
        return pd.Series([None] * len(series), index=series.index, dtype="float64")
    base = float(first_valid.iloc[0])
    if base == 0.0:
        return pd.Series([None] * len(series), index=series.index, dtype="float64")
    return (numeric / base) * 100.0


def _estimate_cagr(series: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return None
    start = float(numeric.iloc[0])
    end = float(numeric.iloc[-1])
    periods = len(numeric) - 1
    if start <= 0.0 or end <= 0.0 or periods <= 0:
        return None
    return ((end / start) ** (1.0 / periods) - 1.0) * 100.0


def _infer_period_lag(series: pd.Series) -> int:
    dt = pd.to_datetime(series, errors="coerce").dropna().sort_values()
    if len(dt) < 3:
        return 1
    diffs = dt.diff().dropna().dt.days
    if diffs.empty:
        return 1
    median_days = float(diffs.median())
    if median_days <= 35:
        return 12
    if median_days <= 110:
        return 4
    return 1
