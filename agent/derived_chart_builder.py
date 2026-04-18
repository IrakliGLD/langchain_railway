"""Phase 15 (§16.4.5): Generalized derived-chart override builders.

Each builder accepts a prepared ``pd.DataFrame`` plus context helpers and
returns a list of wire-format chart specs (``List[Dict[str, Any]]``) in the
same shape as ``ctx.chart_override_specs`` entries:

    {"data": [...], "type": str, "metadata": {...}}

``dispatch_derived_chart(ctx)`` is the single entry point called from
``analyzer.py:enrich()`` after the legacy ``_materialize_chart_override``
path. It produces overrides only when the Stage 0.2 contract is authoritative
and the contract's ``measure_transform`` or ``answer_kind`` indicates a
derived view that the standard chart pipeline cannot produce alone.

Shadow mode: overrides are written to ``ctx.chart_override_specs``; the
legacy single-spec path (``ctx.chart_override_data/type/meta``) is preserved
as a fallback so rendering is always safe to roll back via env flag.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agent.chart_frame_builder import build_chart_frame_long, from_wide
from contracts.question_analysis import AnswerKind, MeasureTransform, SemanticRole
from context import COLUMN_LABELS

log = logging.getLogger("Enai")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Transforms for which a "dual-panel" override is useful: original + delta.
_MOM_YOY_TRANSFORMS = {
    MeasureTransform.MOM_PCT.value,
    MeasureTransform.MOM_DELTA.value,
    MeasureTransform.YOY_PCT.value,
    MeasureTransform.YOY_DELTA.value,
}

# Transforms for which an indexed-growth single-panel override is useful.
_INDEX_TRANSFORMS = {
    MeasureTransform.INDEX_100.value,
    MeasureTransform.CAGR.value,
}

# ---------------------------------------------------------------------------
# Derived-metrics → MeasureTransform fallback mapping
#
# When visualization.measure_transform is unset (common for EXPLANATION and
# plain data queries), the analyzer LLM may still populate
# analysis_requirements.derived_metrics.  This mapping bridges those
# DerivedMetricName values to the MeasureTransform strings understood by the
# existing builders so that the dispatcher can fire the correct builder
# without requiring the analyzer to explicitly set visualization fields.
# ---------------------------------------------------------------------------

# Maps DerivedMetricName.value → MeasureTransform.value
_DERIVED_METRIC_TO_TRANSFORM: Dict[str, str] = {
    "yoy_percent_change": MeasureTransform.YOY_PCT.value,
    "yoy_absolute_change": MeasureTransform.YOY_DELTA.value,
    "mom_percent_change": MeasureTransform.MOM_PCT.value,
    "mom_absolute_change": MeasureTransform.MOM_DELTA.value,
    # share_delta_mom is a MoM absolute change applied to share columns;
    # reuse the mom_delta builder which handles all numeric columns including shares.
    "share_delta_mom": MeasureTransform.MOM_DELTA.value,
}

# When multiple derived metrics are present, prefer YoY over MoM, pct over delta.
_TRANSFORM_PRIORITY: Tuple[str, ...] = (
    MeasureTransform.YOY_PCT.value,
    MeasureTransform.YOY_DELTA.value,
    MeasureTransform.MOM_PCT.value,
    MeasureTransform.MOM_DELTA.value,
)


def _infer_transform_from_derived_metrics(dm_requests) -> Optional[str]:
    """Inspect a list of DerivedMetricRequest objects and return the
    highest-priority MeasureTransform string, or None when none match.

    Tolerates both real ``DerivedMetricRequest`` instances and SimpleNamespace
    stubs used in unit tests.
    """
    found: set = set()
    for req in dm_requests:
        mn = getattr(req, "metric_name", None)
        if mn is None:
            continue
        # metric_name may be a DerivedMetricName enum or a plain string.
        mn_str = mn.value if hasattr(mn, "value") else str(mn)
        t = _DERIVED_METRIC_TO_TRANSFORM.get(mn_str)
        if t:
            found.add(t)
    for t in _TRANSFORM_PRIORITY:
        if t in found:
            return t
    return None


def _label(col: str) -> str:
    return COLUMN_LABELS.get(col, col.replace("_", " ").title())


def _resolve_time_key(df: pd.DataFrame) -> Optional[str]:
    """Return the first date/year/month/time column in ``df``, or None."""
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "year", "month", "time"]):
            return col
    return None


def _resolve_num_cols(df: pd.DataFrame, time_key: Optional[str]) -> List[str]:
    """Return numeric columns in ``df``, excluding the time key and
    bare year/month integer columns."""
    return [
        col
        for col in df.columns
        if col != time_key
        and pd.api.types.is_numeric_dtype(df[col])
        and not re.search(r"\b(month|year)\b", col.lower())
    ]


def _prep_df(df: pd.DataFrame, time_key: Optional[str]) -> pd.DataFrame:
    """Coerce dtypes; normalise the time column to datetime."""
    out = df.copy()
    if time_key and time_key in out.columns:
        out[time_key] = pd.to_datetime(out[time_key], errors="coerce")
    for col in out.columns:
        if col == time_key:
            continue
        try:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        except Exception:
            pass
    return out


def _infer_dim_map(num_cols: List[str]) -> Dict[str, str]:
    """Infer semantic dimension for each metric column (price/share/etc.)."""
    from agent.chart_pipeline import infer_dimension  # local import to avoid circular
    return {col: infer_dimension(col) for col in num_cols}


# ---------------------------------------------------------------------------
# Forecast-chart series selection (Fix 1 + Fix 6)
# ---------------------------------------------------------------------------

# Known metric aliases.  When both alias members appear in ``num_cols`` the
# canonical form wins so the chart does not double-plot the same series under
# two names.  Extend this map as new aliases surface.
_METRIC_ALIAS_CANONICAL: Dict[str, str] = {
    "balancing_price_gel": "p_bal_gel",
    "balancing_price_usd": "p_bal_usd",
}

# Tokens in the user query that flip the default currency from GEL to USD.
_USD_QUERY_TOKENS: Tuple[str, ...] = ("usd", "dollar", "dollars", "us dollar", "$")


def _prefers_usd(user_query: Optional[str]) -> bool:
    """Return True when the user's query mentions USD/dollar/$ — otherwise the
    forecast chart defaults to the local currency (GEL for Georgia)."""
    if not user_query:
        return False
    q = user_query.lower()
    return any(tok in q for tok in _USD_QUERY_TOKENS)


def _select_forecast_series(
    df: pd.DataFrame,
    num_cols: List[str],
    user_query: Optional[str],
    *,
    max_series: int = 2,
) -> List[str]:
    """Filter the candidate forecast columns to a readable set.

    Applies three passes:

    1. **Alias dedup.**  If a canonical name and its alias both appear
       (e.g. ``p_bal_gel`` and ``balancing_price_gel``), drop the alias.
    2. **Currency preference.**  If columns exist for multiple currencies,
       keep only the currency requested by the query (USD if the query
       mentions USD/dollar/$; otherwise GEL as the local default).  The
       filter is applied per-base-metric so unrelated non-currency columns
       (e.g. shares, volumes) are unaffected.
    3. **Series cap.**  If more than ``max_series`` columns remain, keep the
       top-``max_series`` by non-null numeric variance — this is a safety
       net for unusual schemas.

    Returns the filtered, ordered column list.  Empty input → empty output.
    """
    if not num_cols:
        return []

    # --- Pass 1: alias dedup ---
    names = set(num_cols)
    deduped: List[str] = []
    for col in num_cols:
        canonical = _METRIC_ALIAS_CANONICAL.get(col)
        if canonical and canonical in names and canonical != col:
            # This column is an alias whose canonical form is also present — skip it.
            continue
        deduped.append(col)

    # --- Pass 2: currency preference ---
    prefer_usd = _prefers_usd(user_query)
    gel_cols = [c for c in deduped if c.lower().endswith("_gel")]
    usd_cols = [c for c in deduped if c.lower().endswith("_usd")]
    if gel_cols and usd_cols:
        keep_suffix = "_usd" if prefer_usd else "_gel"
        filtered = [
            c for c in deduped
            if (c.lower().endswith(keep_suffix))
            or (not c.lower().endswith("_gel") and not c.lower().endswith("_usd"))
        ]
    else:
        filtered = deduped

    # --- Pass 3: series cap by variance ---
    if len(filtered) <= max_series:
        return filtered

    variances: List[Tuple[str, float]] = []
    for col in filtered:
        try:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            var = float(series.var()) if len(series) > 1 else 0.0
        except Exception:
            var = 0.0
        variances.append((col, var))
    variances.sort(key=lambda kv: kv[1], reverse=True)
    return [c for c, _ in variances[:max_series]]


def _spec_from_wide(
    wide_df: pd.DataFrame,
    *,
    time_key: Optional[str],
    num_cols: List[str],
    label_map: Dict[str, str],
    chart_type: str,
    title: str,
    y_axis_title: str = "",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a wide DataFrame to a wire chart-spec dict."""
    labeled = wide_df.rename(columns=label_map)
    # Time labels: format as ISO string
    if time_key and time_key in labeled.columns:
        labeled[time_key] = labeled[time_key].apply(
            lambda v: v.isoformat()[:7] if pd.notna(v) else None
        )
    data = labeled.to_dict("records")
    labels = [label_map.get(col, col) for col in num_cols if col in wide_df.columns]
    meta: Dict[str, Any] = {
        "xAxisTitle": time_key or "",
        "yAxisTitle": y_axis_title,
        "title": title,
        "axisMode": "single",
        "labels": labels,
    }
    if extra_meta:
        meta.update(extra_meta)
    return {"data": data, "type": chart_type, "metadata": meta}


# ---------------------------------------------------------------------------
# Builder 1: MoM/YoY dual-panel
# ---------------------------------------------------------------------------


def _build_mom_yoy_specs(
    df: pd.DataFrame,
    time_key: Optional[str],
    num_cols: List[str],
    label_map: Dict[str, str],
    measure_transform: str,
) -> Optional[List[Dict[str, Any]]]:
    """Emit two chart specs — the raw observed data and the period-over-period
    delta — so the renderer can show both panels simultaneously.

    Spec 0: raw ``line`` chart of the original series.
    Spec 1: derived ``bar`` chart of the period-over-period change.
    """
    if not num_cols or not time_key:
        return None

    dim_map = _infer_dim_map(num_cols)

    # --- Spec 0: raw observed ---
    raw_frame = build_chart_frame_long(
        df,
        time_key=time_key,
        category_cols=[],
        num_cols=num_cols,
        dim_map=dim_map,
        time_grain=None,
        measure_transform="raw",
        label_map=label_map,
    )
    if raw_frame.is_empty():
        return None
    wide_raw, _ = raw_frame.long_df, {}
    # Reconstruct wide from long for _spec_from_wide.
    from agent.chart_frame_builder import to_wide
    wide_raw_df = to_wide(raw_frame, time_key=time_key)
    if wide_raw_df.empty:
        return None

    specs: List[Dict[str, Any]] = []

    # Observed spec: use the pivot-back wide frame.
    observed_num = [col for col in num_cols if col in df.columns]
    obs_wide = df[[time_key] + [c for c in observed_num if c in df.columns]].copy()
    obs_wide[time_key] = pd.to_datetime(obs_wide[time_key], errors="coerce")
    obs_wide = obs_wide.dropna(subset=[time_key]).sort_values(time_key)
    for col in observed_num:
        obs_wide[col] = pd.to_numeric(obs_wide[col], errors="coerce")
    specs.append(
        _spec_from_wide(
            obs_wide,
            time_key=time_key,
            num_cols=observed_num,
            label_map=label_map,
            chart_type="line",
            title="Observed Data",
            y_axis_title="",
            extra_meta={"role": SemanticRole.OBSERVED.value},
        )
    )

    # --- Spec 1: derived delta ---
    delta_frame = build_chart_frame_long(
        df,
        time_key=time_key,
        category_cols=[],
        num_cols=num_cols,
        dim_map=dim_map,
        time_grain=None,
        measure_transform=measure_transform,
        label_map=label_map,
        derived_cols=set(num_cols),
    )
    if delta_frame.is_empty():
        return specs  # return at least the observed panel

    from agent.chart_frame_builder import to_wide
    wide_delta = to_wide(delta_frame, time_key=time_key)
    if wide_delta.empty:
        return specs

    delta_labels_in_df = [
        c for c in wide_delta.columns
        if c != time_key and c != "period"
    ]
    if not delta_labels_in_df:
        return specs

    pct_suffixes = {"pct", "percent"}
    is_pct = any(p in measure_transform for p in pct_suffixes)
    y_axis = "Change (%)" if is_pct else "Change"
    period_type = "MoM" if measure_transform.startswith("mom") else "YoY"
    delta_title = f"{period_type} Change (%) " if is_pct else f"{period_type} Change"

    # Build spec from wide_delta
    wide_delta_renamed = wide_delta.copy()
    if "period" in wide_delta_renamed.columns and time_key not in wide_delta_renamed.columns:
        wide_delta_renamed = wide_delta_renamed.rename(columns={"period": time_key})
    if time_key in wide_delta_renamed.columns:
        wide_delta_renamed[time_key] = wide_delta_renamed[time_key].apply(
            lambda v: v.isoformat()[:7] if pd.notna(v) else None
        )
    wide_delta_renamed = wide_delta_renamed.dropna(
        subset=[c for c in delta_labels_in_df if c in wide_delta_renamed.columns],
        how="all",
    )
    data = wide_delta_renamed.to_dict("records")
    meta = {
        "xAxisTitle": time_key or "",
        "yAxisTitle": y_axis,
        "title": delta_title,
        "axisMode": "single",
        "labels": delta_labels_in_df,
        "role": SemanticRole.DERIVED.value,
        "measureTransform": measure_transform,
    }
    specs.append({"data": data, "type": "bar", "metadata": meta})
    return specs


# ---------------------------------------------------------------------------
# Builder 2: Indexed growth
# ---------------------------------------------------------------------------


def _build_index_growth_spec(
    df: pd.DataFrame,
    time_key: Optional[str],
    num_cols: List[str],
    label_map: Dict[str, str],
) -> Optional[List[Dict[str, Any]]]:
    """Emit a single chart spec with all series normalized to index=100 at
    the first data point — ideal for multi-series growth comparison."""
    if not num_cols or not time_key:
        return None

    dim_map = _infer_dim_map(num_cols)
    idx_frame = build_chart_frame_long(
        df,
        time_key=time_key,
        category_cols=[],
        num_cols=num_cols,
        dim_map=dim_map,
        time_grain=None,
        measure_transform="index_100",
        label_map=label_map,
        derived_cols=set(num_cols),
    )
    if idx_frame.is_empty():
        return None

    from agent.chart_frame_builder import to_wide
    wide_idx = to_wide(idx_frame, time_key=time_key)
    if wide_idx.empty:
        return None

    series_labels = [
        c for c in wide_idx.columns if c != time_key and c != "period"
    ]
    if not series_labels:
        return None

    title_parts = [label_map.get(col, col) for col in num_cols[:2]]
    title = "Indexed Growth (%s + ...)" % ", ".join(title_parts) if len(num_cols) > 2 else "Indexed Growth: " + " vs ".join(title_parts)

    if "period" in wide_idx.columns and time_key not in wide_idx.columns:
        wide_idx = wide_idx.rename(columns={"period": time_key})
    if time_key in wide_idx.columns:
        wide_idx[time_key] = wide_idx[time_key].apply(
            lambda v: v.isoformat()[:7] if pd.notna(v) else None
        )
    wide_idx = wide_idx.dropna(subset=series_labels, how="all")
    data = wide_idx.to_dict("records")
    meta = {
        "xAxisTitle": time_key or "",
        "yAxisTitle": "Index (base=100)",
        "title": title,
        "axisMode": "single",
        "labels": series_labels,
        "measureTransform": "index_100",
        "role": SemanticRole.DERIVED.value,
    }
    return [{"data": data, "type": "line", "metadata": meta}]


# ---------------------------------------------------------------------------
# Builder 3: Decomposition (share-based stacked bar)
# ---------------------------------------------------------------------------


def _build_decomposition_spec(
    df: pd.DataFrame,
    time_key: Optional[str],
    num_cols: List[str],
    label_map: Dict[str, str],
) -> Optional[List[Dict[str, Any]]]:
    """Generalised decomposition: stacked bar of share/component columns over
    time. Complements the scenario-specific DECOMPOSITION override in the
    legacy ``_build_chart_override`` path.
    """
    share_cols = [c for c in num_cols if "share" in c.lower() or "component" in c.lower()]
    if not share_cols or not time_key:
        return None

    prepared = df[[time_key] + share_cols].copy()
    prepared[time_key] = pd.to_datetime(prepared[time_key], errors="coerce")
    prepared = prepared.dropna(subset=[time_key]).sort_values(time_key)
    for col in share_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    # Melt to long category-value form expected by stacked bar.
    melted_rows: List[Dict[str, Any]] = []
    for _, row in prepared.iterrows():
        ts = row[time_key]
        ts_str = ts.isoformat()[:7] if pd.notna(ts) else None
        for col in share_cols:
            val = row[col]
            if pd.notna(val):
                melted_rows.append(
                    {
                        "date": ts_str,
                        "category": label_map.get(col, col),
                        "value": float(val),
                    }
                )

    if not melted_rows:
        return None

    labels = [label_map.get(col, col) for col in share_cols]
    meta = {
        "xAxisTitle": time_key or "",
        "yAxisTitle": "Share",
        "title": "Composition Breakdown",
        "axisMode": "single",
        "labels": labels,
        "role": SemanticRole.COMPONENT_PRIMARY.value,
    }
    return [{"data": melted_rows, "type": "stackedbar", "metadata": meta}]


# ---------------------------------------------------------------------------
# Builder 4: Forecast observed vs projected
# ---------------------------------------------------------------------------


def _build_forecast_spec(
    df: pd.DataFrame,
    time_key: Optional[str],
    num_cols: List[str],
    label_map: Dict[str, str],
    user_query: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Emit a chart spec that distinguishes observed (historical) rows from
    projected (forecast) rows. The ``_generate_cagr_forecast`` function in
    ``analyzer.py`` marks projected rows with a ``is_projected`` boolean
    column; we respect that when present and fall back to a date cutoff
    (projection = any row after the last observed row with a non-null value).

    ``user_query`` steers currency preference: queries mentioning USD/dollar/$
    keep USD columns, otherwise the local currency (GEL) wins.
    """
    if not num_cols or not time_key or df.empty:
        return None

    # Fix 1 + 6: collapse alias duplicates, prefer the relevant currency, and
    # cap total series so the chart stays readable.
    selected_cols = _select_forecast_series(df, num_cols, user_query)
    if not selected_cols:
        return None
    if selected_cols != num_cols:
        log.info(
            "dispatch_derived_chart: forecast series filtered %s → %s (query=%r)",
            num_cols, selected_cols, (user_query or "")[:80],
        )
    num_cols = selected_cols

    prepared = df.copy()
    prepared[time_key] = pd.to_datetime(prepared[time_key], errors="coerce")
    prepared = prepared.dropna(subset=[time_key]).sort_values(time_key)

    # ``_generate_cagr_forecast`` marks projected rows with ``is_forecast=True``.
    # Fall back to ``is_projected`` for external callers, then to a period
    # cutoff heuristic only when neither marker column is present.
    if "is_forecast" in prepared.columns:
        prepared["_projected"] = prepared["is_forecast"].fillna(False).astype(bool)
    elif "is_projected" in prepared.columns:
        prepared["_projected"] = prepared["is_projected"].fillna(False).astype(bool)
    else:
        # Without a marker, we cannot reliably distinguish observed from projected;
        # skip the override so the standard trendline path stays in charge.
        return None

    rows: List[Dict[str, Any]] = []
    for _, row in prepared.iterrows():
        ts = row[time_key]
        ts_str = ts.isoformat()[:7] if pd.notna(ts) else None
        is_proj = bool(row.get("_projected", False))
        record: Dict[str, Any] = {time_key: ts_str}
        for col in num_cols:
            if col in row.index:
                val = row[col]
                key_observed = label_map.get(col, col)
                key_projected = f"{label_map.get(col, col)} (Projected)"
                if is_proj:
                    record[key_projected] = None if pd.isna(val) else float(val)
                    record[key_observed] = None
                else:
                    record[key_observed] = None if pd.isna(val) else float(val)
                    record[key_projected] = None
        rows.append(record)

    if not rows:
        return None

    all_cols = [k for k in (rows[0].keys() if rows else {}) if k != time_key]
    meta = {
        "xAxisTitle": time_key or "",
        "yAxisTitle": "",
        "title": "Observed vs Projected",
        "axisMode": "single",
        "labels": all_cols,
        "role": SemanticRole.DERIVED.value,
        "measureTransform": "forecast",
    }
    return [{"data": rows, "type": "line", "metadata": meta}]


# ---------------------------------------------------------------------------
# Builder 5: Seasonal bucket comparison
# ---------------------------------------------------------------------------


def _build_seasonal_spec(
    df: pd.DataFrame,
    time_key: Optional[str],
    num_cols: List[str],
    label_map: Dict[str, str],
) -> Optional[List[Dict[str, Any]]]:
    """Emit a bar chart of summer/winter bucket averages using Phase 12's
    season datetime bucket logic from ``build_chart_frame_long``."""
    if not num_cols or not time_key:
        return None

    # Need at least 12 months to form meaningful buckets.
    prepared = df.copy()
    prepared[time_key] = pd.to_datetime(prepared[time_key], errors="coerce")
    prepared = prepared.dropna(subset=[time_key])
    if len(prepared) < 6:
        return None

    dim_map = _infer_dim_map(num_cols)
    seasonal_frame = build_chart_frame_long(
        prepared,
        time_key=time_key,
        category_cols=[],
        num_cols=num_cols,
        dim_map=dim_map,
        time_grain="season",
        measure_transform="avg",
        label_map=label_map,
        derived_cols=set(),
    )
    if seasonal_frame.is_empty():
        return None

    from agent.chart_frame_builder import to_wide
    wide_seasonal = to_wide(seasonal_frame, time_key=time_key)
    if wide_seasonal.empty:
        return None

    series_labels = [c for c in wide_seasonal.columns if c != time_key and c != "period"]
    if not series_labels:
        return None

    # Label the period column using the seasonLabels map so the renderer
    # can display "2023-summer" / "2023-winter" instead of ISO timestamps.
    season_labels_map: Dict[str, str] = seasonal_frame.meta.get("seasonLabels", {})
    if "period" in wide_seasonal.columns and time_key not in wide_seasonal.columns:
        wide_seasonal = wide_seasonal.rename(columns={"period": time_key})
    if time_key in wide_seasonal.columns:
        wide_seasonal[time_key] = wide_seasonal[time_key].apply(
            lambda v: season_labels_map.get(v.isoformat() if pd.notna(v) else "", v.isoformat()[:7] if pd.notna(v) else None)
        )

    wide_seasonal = wide_seasonal.dropna(subset=series_labels, how="all")
    data = wide_seasonal.to_dict("records")
    meta = {
        "xAxisTitle": time_key or "",
        "yAxisTitle": "",
        "title": "Seasonal Breakdown (Summer vs Winter)",
        "axisMode": "single",
        "labels": series_labels,
        "timeGrain": "season",
        "seasonLabels": season_labels_map,
    }
    return [{"data": data, "type": "bar", "metadata": meta}]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def dispatch_derived_chart(ctx: "QueryContext") -> Optional[List[Dict[str, Any]]]:  # type: ignore[name-defined]
    """Main entry point called from ``analyzer.py:enrich()``.

    Inspects the authoritative ``question_analysis.visualization`` contract
    and dispatches to the appropriate derived-chart builder. Returns a list
    of wire-format specs suitable for ``ctx.chart_override_specs``, or
    ``None`` when no derived view is warranted.

    Only fires when:
    - ``ctx.has_authoritative_question_analysis`` is ``True`` (Stage 0.2
      succeeded), AND
    - The contract's ``measure_transform`` or ``answer_kind`` indicates
      a view that the standard chart pipeline cannot produce alone.

    Shadow mode: the existing ``_materialize_chart_override`` path is NOT
    replaced — this dispatcher runs alongside it. When the new specs are
    non-empty, ``_apply_chart_override`` prefers ``chart_override_specs``
    over the legacy single-spec path (per Phase 10 wiring).
    """
    if not ctx.has_authoritative_question_analysis:
        return None
    if ctx.df is None or ctx.df.empty:
        return None

    visualization = ctx.question_analysis.visualization
    measure_transform = getattr(
        getattr(visualization, "measure_transform", None), "value", None
    )
    answer_kind = ctx.effective_answer_kind
    time_grain = getattr(
        getattr(visualization, "time_grain", None), "value", None
    )

    # Prepare the common inputs once.
    df = ctx.df.copy()
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

    time_key = _resolve_time_key(df)
    if time_key:
        df[time_key] = pd.to_datetime(ctx.df[time_key], errors="coerce")
        # Reset to string after pd.to_numeric would have broken it.
        df[time_key] = pd.to_datetime(ctx.df[time_key], errors="coerce")
    num_cols = _resolve_num_cols(df, time_key)
    if not num_cols:
        return None
    label_map = {col: _label(col) for col in df.columns if col != time_key}

    # ---- MoM/YoY dual-panel ----
    if measure_transform in _MOM_YOY_TRANSFORMS:
        log.info(
            "dispatch_derived_chart: MoM/YoY override for measure_transform=%s",
            measure_transform,
        )
        return _build_mom_yoy_specs(df, time_key, num_cols, label_map, measure_transform)

    # ---- Indexed growth ----
    if measure_transform in _INDEX_TRANSFORMS:
        log.info(
            "dispatch_derived_chart: indexed growth override for measure_transform=%s",
            measure_transform,
        )
        return _build_index_growth_spec(df, time_key, num_cols, label_map)

    # ---- Decomposition (explicit intent, non-scenario) ----
    chart_intent = getattr(
        getattr(visualization, "chart_intent", None), "value", None
    )
    if chart_intent == "decomposition" and any("share" in c.lower() for c in num_cols):
        log.info("dispatch_derived_chart: decomposition override")
        return _build_decomposition_spec(df, time_key, num_cols, label_map)

    # ---- Seasonal bucket ----
    if time_grain == "season":
        log.info("dispatch_derived_chart: seasonal bucket override")
        return _build_seasonal_spec(df, time_key, num_cols, label_map)

    # ---- Forecast observed vs projected ----
    if answer_kind == AnswerKind.FORECAST:
        log.info("dispatch_derived_chart: forecast observed-vs-projected override")
        return _build_forecast_spec(
            df, time_key, num_cols, label_map, user_query=getattr(ctx, "query", None)
        )

    # ---- Derived-metrics fallback ----
    # When visualization.measure_transform is unset or raw (the common case for
    # EXPLANATION and plain data queries), the analyzer still populates
    # analysis_requirements.derived_metrics.  Inspect those entries and derive
    # the best transform so the correct builder fires without requiring the LLM
    # to explicitly set visualization.measure_transform.
    if not measure_transform or measure_transform == MeasureTransform.RAW.value:
        dm_requests = (
            getattr(
                getattr(ctx.question_analysis, "analysis_requirements", None),
                "derived_metrics",
                [],
            )
            or []
        )
        inferred = _infer_transform_from_derived_metrics(dm_requests)
        if inferred:
            log.info(
                "dispatch_derived_chart: derived_metrics fallback → inferred transform=%s "
                "(analysis_requirements.derived_metrics had %d entries)",
                inferred,
                len(dm_requests),
            )
            if inferred in _MOM_YOY_TRANSFORMS:
                return _build_mom_yoy_specs(df, time_key, num_cols, label_map, inferred)
            if inferred in _INDEX_TRANSFORMS:
                return _build_index_growth_spec(df, time_key, num_cols, label_map)

    return None
