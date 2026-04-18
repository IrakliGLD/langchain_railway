"""
Pipeline Stage 5: Chart Building

Handles chart gating, chart-specific data transforms, semantic chart selection,
and rendering a backward-compatible single-chart payload plus a chart collection.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agent.analyzer import METRIC_VALUE_ALIASES
from agent.chart_frame_builder import build_chart_frame, from_wide
from config import ENAI_CHART_LONGFORM
from context import COLUMN_LABELS
from models import QueryContext
from utils.trace_logging import trace_detail
from visualization.chart_selector import select_chart_type, should_generate_chart

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# Chart metric alias map: any name -> set of equivalent names.
# Built from METRIC_VALUE_ALIASES so plan names like "balancing_price_gel"
# resolve to the DB column "p_bal_gel" and vice versa.
# ---------------------------------------------------------------------------
_CHART_METRIC_ALIASES: Dict[str, List[str]] = {}
for _canonical, _aliases in METRIC_VALUE_ALIASES.items():
    _all = list(set(_aliases) | {_canonical})
    for _name in _all:
        _CHART_METRIC_ALIASES[_name] = _all


_DEFAULT_MAX_SERIES = 3
_AUTO_PANEL_DIMENSION_ORDER = [
    "price_tariff",
    "xrate",
    "energy_qty",
    "share",
    "index",
    "other",
]
_NORMALIZED_CHART_TYPES = {
    "line": "line",
    "bar": "bar",
    "pie": "pie",
    "dualaxis": "dualaxis",
    "stacked": "stackedbar",
    "stackedbar": "stackedbar",
    "stacked_bar": "stackedbar",
    "stackedarea": "stackedbar",
    "stacked_area": "stackedbar",
}


# ---------------------------------------------------------------------------
# Dimension/unit inference helpers
# ---------------------------------------------------------------------------

def infer_dimension(col: str) -> str:
    """Infer semantic dimension of a column for chart axis decisions."""
    col_l = col.lower()
    if any(x in col_l for x in ["xrate", "exchange", "rate"]):
        return "xrate"
    if any(x in col_l for x in ["share_", "proportion", "percent"]):
        return "share"
    if any(x in col_l for x in ["cpi", "index", "inflation"]):
        return "index"
    if any(x in col_l for x in ["quantity", "generation", "volume_tj", "volume", "mw", "tj"]):
        return "energy_qty"
    if any(x in col_l for x in ["price", "tariff", "_gel", "_usd", "p_bal", "p_dereg", "p_gcap"]):
        return "price_tariff"
    return "other"


def unit_for_price(cols_: List[str]) -> str:
    has_gel = any("_gel" in c.lower() for c in cols_)
    has_usd = any("_usd" in c.lower() for c in cols_)
    if has_gel and has_usd:
        return "currency/MWh"
    if has_gel:
        return "GEL/MWh"
    if has_usd:
        return "USD/MWh"
    return "currency/MWh"


def unit_for_qty(cols_: List[str]) -> str:
    has_tj = any("tj" in c.lower() for c in cols_)
    has_thousand_mwh = any("quantity" in c.lower() for c in cols_)
    if has_tj and not has_thousand_mwh:
        return "TJ"
    if has_thousand_mwh and not has_tj:
        return "thousand MWh"
    return "Energy Quantity"


def unit_for_index(_: List[str]) -> str:
    return "Index (2015=100)"


def unit_for_xrate(_: List[str]) -> str:
    return "GEL per USD"


def unit_for_share(_: List[str]) -> str:
    return "Share (0-1)"


def relevance_score(col: str, query_lower: str) -> int:
    """Score a column by keyword relevance to the user's query."""
    score = 0
    col_lower = col.lower()

    if any(k in query_lower for k in ["price"]) and any(k in col_lower for k in ["price", "p_bal"]):
        score += 10
    if any(k in query_lower for k in ["xrate", "exchange"]) and "xrate" in col_lower:
        score += 10
    if any(k in query_lower for k in ["share", "composition"]) and "share" in col_lower:
        score += 5
    if "tariff" in query_lower and "tariff" in col_lower:
        score += 5
    if "p_bal" in col_lower:
        score += 3
    if "xrate" in col_lower:
        score += 2

    return score


_DIM_PRIORITY = {
    "price_tariff": 6,
    "energy_qty": 5,
    "xrate": 4,
    "share": 3,
    "index": 2,
    "other": 1,
}


# ---------------------------------------------------------------------------
# Main pipeline stage
# ---------------------------------------------------------------------------

def build_chart(ctx: QueryContext) -> QueryContext:
    """Stage 5: Build chart data from query results."""
    _reset_chart_outputs(ctx)

    if ctx.chart_override_specs or ctx.chart_override_data:
        if ctx.chart_override_specs:
            log.info(
                "Using derived chart override (multi-spec, count=%d)",
                len(ctx.chart_override_specs),
            )
        else:
            log.info(
                "Using derived chart override (type=%s, rows=%d)",
                ctx.chart_override_type,
                len(ctx.chart_override_data or []),
            )
        return _apply_chart_override(ctx)

    if not ctx.rows or not ctx.cols:
        return ctx

    df, time_key, label_map_all, categorical_cols, num_cols = _prepare_chart_source(ctx)
    if df.empty or not num_cols:
        return ctx

    qa_for_chart = ctx.question_analysis if ctx.question_analysis_source == "llm_active" else None
    vis = getattr(qa_for_chart, "visualization", None)

    generate_chart = should_generate_chart(
        ctx.query,
        len(df),
        response_mode=ctx.response_mode,
        question_analysis=qa_for_chart,
    )
    generate_chart = _apply_legacy_chart_suppression(
        ctx,
        generate_chart=generate_chart,
        row_count=len(df),
        num_cols=num_cols,
        visualization=vis,
    )
    if not generate_chart:
        log.info("Skipping chart generation (rows=%d).", len(df))
        return ctx

    chart_groups = _resolve_chart_groups(ctx, num_cols, vis)
    if not chart_groups:
        log.info("Skipping chart generation: no eligible chart groups.")
        return ctx

    charts: List[Dict[str, Any]] = []
    for group_index, group in enumerate(chart_groups):
        chart_spec = _build_chart_spec(
            ctx=ctx,
            source_df=df,
            time_key=time_key,
            label_map_all=label_map_all,
            categorical_cols=categorical_cols,
            group=group,
            visualization=vis,
            group_index=group_index,
        )
        if chart_spec is not None:
            charts.append(chart_spec)

    if not charts:
        log.info("Skipping chart generation: chart specs could not be materialized.")
        return ctx

    _apply_chart_collection(ctx, charts)

    # Phase 12: when the analyzer asks for chart_plus_table, attach a
    # companion table so the renderer can show the underlying observed
    # values next to any derived chart. Uses the pre-transform frame (df)
    # so exact numeric lookups are preserved regardless of measure_transform.
    presentation = getattr(getattr(vis, "primary_presentation", None), "value", None)
    if presentation == "chart_plus_table":
        ctx.companion_table = _build_companion_table(df, label_map_all)
        # Surface on primary chart metadata so a single-payload consumer
        # sees the companion without reaching into a separate field.
        if ctx.chart_meta is not None:
            ctx.chart_meta["companionTable"] = ctx.companion_table

    trace_detail(
        log,
        ctx,
        "stage_5_chart_build",
        "selection",
        chart_count=len(charts),
        chart_type=ctx.chart_type or "",
        row_count=len(ctx.chart_data or []),
        reason="chart_collection_built",
    )
    log.info(
        "Chart collection built | charts=%d | primary_type=%s",
        len(charts),
        ctx.chart_type,
    )
    return ctx


def _reset_chart_outputs(ctx: QueryContext) -> None:
    ctx.charts = []
    ctx.chart_data = None
    ctx.chart_type = None
    ctx.chart_meta = None


def _apply_chart_override(ctx: QueryContext) -> QueryContext:
    # Multi-spec override takes precedence so derived-chart builders (scenario
    # multi-panel, MoM/YoY observed+delta, seasonal, forecast observed-vs-projected)
    # can surface multiple coordinated charts. Fall back to the legacy
    # single-spec shape (chart_override_data/_type/_meta) when no list is set.
    if ctx.chart_override_specs:
        charts = [dict(spec) for spec in ctx.chart_override_specs if spec]
        if charts:
            _apply_chart_collection(ctx, charts)
            return ctx

    metadata = dict(ctx.chart_override_meta or {})
    chart_spec = {
        "data": list(ctx.chart_override_data or []),
        "type": ctx.chart_override_type,
        "metadata": metadata,
    }
    _apply_chart_collection(ctx, [chart_spec])
    return ctx


def _prepare_chart_source(
    ctx: QueryContext,
) -> Tuple[pd.DataFrame, Optional[str], Dict[str, str], List[str], List[str]]:
    df = ctx.df.copy()
    cols = list(ctx.cols or df.columns)

    time_key = next(
        (
            c
            for c in cols
            if any(k in c.lower() for k in ["date", "year", "month", "time"])
        ),
        None,
    )

    if time_key and time_key in df.columns:
        try:
            first_val = df[time_key].iloc[0]
            if isinstance(first_val, (int, float)) or str(type(first_val).__name__) == "Decimal":
                first_val_num = float(first_val)
                if 1900 <= first_val_num <= 2100:
                    df[time_key] = pd.to_datetime(df[time_key].astype(int), format="%Y")
                    log.info("Converted year-only column '%s' to datetime", time_key)
        except Exception as exc:
            log.warning("Year column conversion check failed in chart builder: %s", exc)

    categorical_hints = [
        "type",
        "tech",
        "entity",
        "sector",
        "source",
        "segment",
        "region",
        "category",
        "ownership",
        "market",
        "trade",
        "fuel",
    ]
    for col in cols:
        if col == time_key or col not in df.columns:
            continue
        if any(hint in col.lower() for hint in categorical_hints):
            df[col] = df[col].astype(str).replace("nan", None)
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

    categorical_cols = [
        col
        for col in df.columns
        if col != time_key and not pd.api.types.is_numeric_dtype(df[col])
    ]
    label_map_all = {
        col: COLUMN_LABELS.get(col, col.replace("_", " ").title())
        for col in df.columns
        if col != time_key
    }
    num_cols = [
        col
        for col in df.columns
        if col != time_key
        and pd.api.types.is_numeric_dtype(df[col])
        and not re.search(r"\b(month|year)\b", col.lower())
    ]
    return df, time_key, label_map_all, categorical_cols, num_cols


def _apply_legacy_chart_suppression(
    ctx: QueryContext,
    *,
    generate_chart: bool,
    row_count: int,
    num_cols: List[str],
    visualization: Optional[Any],
) -> bool:
    if not generate_chart:
        return False

    query_text = ctx.query.lower()
    chart_requested = bool(getattr(visualization, "chart_requested_by_user", False))
    chart_contract_prefers_chart = getattr(
        getattr(visualization, "primary_presentation", None),
        "value",
        None,
    ) in {"chart", "chart_plus_table"}

    if ctx.skip_chart_due_to_relevance:
        log.info("Skipping chart generation: SQL query not relevant to user question")
        return False

    if any(word in query_text for word in ["define", "meaning of"]) and not chart_requested:
        return False

    if any(word in query_text for word in ["why", "how", "reason", "explain", "because"]):
        has_shares = any("share" in col.lower() for col in num_cols)
        if (
            not chart_requested
            and not chart_contract_prefers_chart
            and (row_count < 2 or (row_count < 5 and not has_shares))
        ):
            log.info(
                "Skipping chart: explanatory query with small non-compositional result (rows=%d, has_shares=%s)",
                row_count,
                has_shares,
            )
            return False

    return True


def _resolve_chart_groups(
    ctx: QueryContext,
    num_cols: List[str],
    visualization: Optional[Any],
) -> List[Dict[str, Any]]:
    planned_groups = ctx.plan.get("chart_groups", [])
    groups: List[Dict[str, Any]] = []

    if isinstance(planned_groups, list):
        for group in planned_groups:
            if not isinstance(group, dict):
                continue
            metrics = _resolve_group_metrics(group.get("metrics", []), num_cols)
            if not metrics:
                log.warning(
                    "Chart plan metrics %s matched none of data columns %s",
                    group.get("metrics", []),
                    num_cols,
                )
                continue
            groups.append(
                {
                    "metrics": metrics,
                    "type": group.get("type"),
                    "title": group.get("title"),
                    "y_axis_label": group.get("y_axis_label"),
                    "source": "plan",
                }
            )

    if not groups:
        strategy = str(ctx.plan.get("chart_strategy", "")).lower()
        split_mode = getattr(getattr(visualization, "series_split_mode", None), "value", None)
        should_split = split_mode == "multi_panel" or strategy in {"grouped", "multiple"}
        if should_split:
            groups.extend(_auto_split_groups(num_cols))
        else:
            groups.append({"metrics": list(num_cols), "source": "auto"})

    max_series = getattr(visualization, "max_series", None) or _DEFAULT_MAX_SERIES
    sort_rule = getattr(getattr(visualization, "sort_rule", None), "value", None)
    top_n = getattr(visualization, "top_n", None)
    query_lower = ctx.query.lower()
    normalized_groups: List[Dict[str, Any]] = []
    # Dedup key includes title and type so planner-authored groups that share
    # the same metric list but render as distinct panels (e.g., two bars
    # differing only in title/type) are not silently collapsed. Previously
    # the key was `tuple(metrics)` alone, which dropped legitimate planner
    # groups whose `title`/`type` were the differentiator.
    seen_signatures: set[Tuple[str, str, Tuple[str, ...]]] = set()
    for group in groups:
        metrics = list(group.get("metrics", []))
        preserve_order = group.get("source") == "plan"
        metrics = _limit_series(
            metrics,
            query_lower,
            max_series,
            preserve_order=preserve_order,
            sort_rule=sort_rule,
            top_n=top_n,
        )
        if not metrics:
            continue
        signature = (
            str(group.get("title") or ""),
            str(group.get("type") or ""),
            tuple(metrics),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        normalized = dict(group)
        normalized["metrics"] = metrics
        normalized_groups.append(normalized)
    return normalized_groups


def _resolve_group_metrics(requested_metrics: Any, available_num_cols: List[str]) -> List[str]:
    if not isinstance(requested_metrics, list):
        return []

    resolved: List[str] = []
    seen: set[str] = set()
    for metric in requested_metrics:
        aliases = set(_CHART_METRIC_ALIASES.get(str(metric), [str(metric)]))
        for col in available_num_cols:
            if col in aliases and col not in seen:
                resolved.append(col)
                seen.add(col)
    return resolved


def _auto_split_groups(num_cols: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[str]] = {}
    for col in num_cols:
        bucket = infer_dimension(col)
        buckets.setdefault(bucket, []).append(col)

    groups: List[Dict[str, Any]] = []
    for bucket in _AUTO_PANEL_DIMENSION_ORDER:
        metrics = buckets.get(bucket, [])
        if not metrics:
            continue
        groups.append(
            {
                "metrics": metrics,
                "title": _default_group_title(bucket),
                "source": "auto_panel",
            }
        )
    return groups


def _default_group_title(bucket: str) -> str:
    title_map = {
        "price_tariff": "Price Trend",
        "xrate": "Exchange Rate Trend",
        "energy_qty": "Quantity Trend",
        "share": "Composition Trend",
        "index": "Index Trend",
        "other": "Indicator Trend",
    }
    return title_map.get(bucket, "Indicator Trend")


def _limit_series(
    metrics: List[str],
    query_lower: str,
    max_series: int,
    *,
    preserve_order: bool,
    sort_rule: Optional[str] = None,
    top_n: Optional[int] = None,
) -> List[str]:
    """Cap a chart group's metric list to an effective series budget.

    Phase 14 (§16.4.1 consumer): consumes ``sort_rule`` / ``top_n`` from the
    visualization plan when present. The effective cap is
    ``min(top_n, max_series)`` when ``top_n`` is set, otherwise ``max_series``.

    Pre-transform context — we only have column *names* here, not values.
    Value-based sort rules (``value_desc``/``value_asc``) therefore fall
    through to the preserve-order path at this stage; the actual value-aware
    reorder happens post-transform in :func:`_reorder_metrics_by_value`.
    """

    effective_cap = max_series
    if top_n is not None and top_n > 0:
        effective_cap = min(int(top_n), max_series)

    if effective_cap <= 0 or len(metrics) <= effective_cap:
        return list(metrics)

    # Alphabetic ordering is purely name-based — safe to apply pre-transform.
    if sort_rule == "category_alpha":
        trimmed = sorted(metrics)[:effective_cap]
        log.info("Limited to %d series via category_alpha: %s", effective_cap, trimmed)
        return trimmed

    # Time-based sort rules describe the x-axis, not series selection. When
    # they are active we want to preserve the caller's column order so the
    # time axis is not reshuffled by a name-based fallback.
    if sort_rule in {"time_asc", "time_desc"} or preserve_order:
        trimmed = list(metrics[:effective_cap])
        log.info("Trimmed chart group to %d series (preserve_order): %s", effective_cap, trimmed)
        return trimmed

    # Value-based sort rules require transformed data — reorder post-transform.
    # For the pre-transform cap we keep the explicit column order so the
    # post-transform step has a deterministic starting set.
    if sort_rule in {"value_desc", "value_asc"}:
        trimmed = list(metrics[:effective_cap])
        log.info(
            "Pre-transform cap for %s keeps first %d metrics; value-aware reorder "
            "happens post-transform: %s",
            sort_rule,
            effective_cap,
            trimmed,
        )
        return trimmed

    # Default ("relevance" or unspecified) → keyword heuristic against the
    # user query. This is the legacy path, retained for backward compat when
    # analyzer has not emitted a sort_rule.
    scored_cols = [(col, relevance_score(col, query_lower)) for col in metrics]
    scored_cols.sort(key=lambda item: item[1], reverse=True)
    trimmed = [col for col, _ in scored_cols[:effective_cap]]
    log.info("Limited to %d series via relevance heuristic: %s", effective_cap, trimmed)
    return trimmed


def _reorder_metrics_by_value(
    transformed_df: pd.DataFrame,
    metrics: List[str],
    *,
    sort_rule: Optional[str],
    top_n: Optional[int],
    max_series: int,
) -> List[str]:
    """Reorder (and optionally re-cap) ``metrics`` using aggregated values.

    Only active when ``sort_rule`` ∈ ``{value_desc, value_asc}`` and the
    transformed frame actually contains the columns. Pure function — does
    not mutate ``transformed_df``.
    """

    if sort_rule not in {"value_desc", "value_asc"}:
        return metrics
    if transformed_df is None or transformed_df.empty:
        return metrics

    effective_cap = max_series
    if top_n is not None and top_n > 0:
        effective_cap = min(int(top_n), max_series)
    if effective_cap <= 0:
        return metrics

    present = [col for col in metrics if col in transformed_df.columns]
    if not present:
        return metrics

    aggregates: Dict[str, float] = {}
    for col in present:
        series = pd.to_numeric(transformed_df[col], errors="coerce")
        if series.notna().any():
            aggregates[col] = float(series.sum(skipna=True))
        else:
            aggregates[col] = float("nan")

    reverse = sort_rule == "value_desc"
    # NaN aggregates sort to the end regardless of direction.
    def _sort_key(col: str) -> Tuple[int, float]:
        value = aggregates.get(col, float("nan"))
        if pd.isna(value):
            return (1, 0.0)
        return (0, -value if reverse else value)

    ordered = sorted(present, key=_sort_key)
    trimmed = ordered[:effective_cap]
    log.info(
        "Post-transform reorder via %s (top %d): %s",
        sort_rule,
        effective_cap,
        trimmed,
    )
    return trimmed


def _build_chart_spec(
    *,
    ctx: QueryContext,
    source_df: pd.DataFrame,
    time_key: Optional[str],
    label_map_all: Dict[str, str],
    categorical_cols: List[str],
    group: Dict[str, Any],
    visualization: Optional[Any],
    group_index: int,
) -> Optional[Dict[str, Any]]:
    metrics = [metric for metric in group.get("metrics", []) if metric in source_df.columns]
    if not metrics:
        return None

    dim_map = {col: infer_dimension(col) for col in metrics}
    dims = set(dim_map.values())
    metrics, dim_map, dims = _cap_dimensions(
        metrics,
        dim_map,
        dims,
        ctx.query.lower(),
        source=group.get("source"),
    )

    columns_to_keep = []
    if time_key and time_key in source_df.columns:
        columns_to_keep.append(time_key)
    columns_to_keep.extend(col for col in categorical_cols if col in source_df.columns)
    columns_to_keep.extend(metrics)
    working = source_df[[col for col in columns_to_keep if col in source_df.columns]].copy()
    if working.empty:
        return None

    effective_time_grain = getattr(getattr(visualization, "time_grain", None), "value", None)
    effective_measure_transform = getattr(
        getattr(visualization, "measure_transform", None),
        "value",
        "raw",
    )
    # Phase 12: the auto-yearly-rollup heuristic (share + >24 rows → year)
    # is deprecated in favor of analyzer-emitted `time_grain`. During
    # rollout we still apply the rollup to protect legacy payloads, but
    # log a deprecation warning so we can confirm analyzer coverage via
    # shadow telemetry. Remove the auto-rollup entirely once the warning
    # rate drops to near zero.
    auto_yearly_rollup = False
    if (
        effective_time_grain in {None, "", "raw"}
        and time_key
        and time_key in working.columns
        and "share" in dims
        and len(working) > 24
    ):
        log.warning(
            "Deprecated auto_yearly_rollup fired: share dim + %d rows with no explicit "
            "time_grain. Analyzer should emit visualization.time_grain='year'. "
            "This heuristic will be removed after analyzer coverage reaches 100%%.",
            len(working),
        )
        effective_time_grain = "year"
        auto_yearly_rollup = True

    transformed_df, transform_meta = build_chart_frame(
        working,
        time_key=time_key,
        category_cols=[col for col in categorical_cols if col in working.columns],
        num_cols=metrics,
        dim_map=dim_map,
        time_grain=effective_time_grain,
        measure_transform=effective_measure_transform,
    )
    if transformed_df.empty:
        return None

    # Phase 14 (§16.4.1 runtime): value-aware reorder using the contract's
    # ``sort_rule`` / ``top_n`` now that values exist on the transformed frame.
    # No-op when sort_rule is not a value rule or top_n is unset.
    sort_rule_value = getattr(getattr(visualization, "sort_rule", None), "value", None)
    top_n_value = getattr(visualization, "top_n", None)
    max_series_cap = getattr(visualization, "max_series", None) or _DEFAULT_MAX_SERIES
    metrics = _reorder_metrics_by_value(
        transformed_df,
        metrics,
        sort_rule=sort_rule_value,
        top_n=top_n_value,
        max_series=max_series_cap,
    )
    if not metrics:
        return None
    # Drop any columns that the reorder cut so the rendered frame only
    # carries the kept series.
    keep_columns = [time_key] if time_key and time_key in transformed_df.columns else []
    keep_columns += [col for col in categorical_cols if col in transformed_df.columns]
    keep_columns += [col for col in metrics if col in transformed_df.columns]
    transformed_df = transformed_df[[c for c in keep_columns if c in transformed_df.columns]].copy()

    has_time = bool(time_key and time_key in transformed_df.columns)
    category_cols_in_df = [col for col in categorical_cols if col in transformed_df.columns]
    has_categories = bool(category_cols_in_df)
    category_count = 0
    if has_categories:
        category_count = int(transformed_df[category_cols_in_df[0]].nunique(dropna=False))

    chart_type = _choose_chart_type(
        group=group,
        visualization=visualization,
        has_time=has_time,
        has_categories=has_categories,
        dimensions=dims,
        category_count=category_count,
    )

    chart_labels = [label_map_all.get(col, col) for col in metrics]
    labeled_df = transformed_df.rename(columns=label_map_all)
    _format_time_labels(labeled_df, time_key)

    chart_data = labeled_df.to_dict("records")
    chart_meta = _build_chart_metadata(dims, chart_type, metrics, chart_labels, time_key)

    transform_meta = dict(transform_meta or {})
    y_axis_override = transform_meta.pop("yAxisTitle", None)
    if y_axis_override:
        if chart_meta.get("axisMode") == "dual":
            chart_meta["yAxisLeft"] = y_axis_override
        else:
            chart_meta["yAxisTitle"] = y_axis_override

    aggregation_value = transform_meta.pop("aggregation", None)
    if auto_yearly_rollup:
        chart_meta["aggregation"] = "yearly"
    elif aggregation_value is not None:
        chart_meta["aggregation"] = aggregation_value

    chart_meta.update(transform_meta)

    if group.get("title"):
        chart_meta["title"] = group["title"]
    if group.get("y_axis_label"):
        if chart_meta.get("axisMode") == "dual":
            chart_meta["yAxisLeft"] = group["y_axis_label"]
        else:
            chart_meta["yAxisTitle"] = group["y_axis_label"]

    chart_meta["groupIndex"] = group_index
    chart_meta["sourceMetrics"] = list(metrics)
    chart_meta["groupSource"] = group.get("source", "auto")

    visual_goal = getattr(getattr(visualization, "visual_goal", None), "value", None)
    if visual_goal:
        chart_meta["visualGoal"] = visual_goal
    measure_transform = getattr(getattr(visualization, "measure_transform", None), "value", None)
    if measure_transform:
        chart_meta["measureTransform"] = chart_meta.get("measureTransform", measure_transform)
    time_grain = getattr(getattr(visualization, "time_grain", None), "value", None)
    if time_grain:
        chart_meta["timeGrain"] = time_grain

    if ctx.add_trendlines and has_time and time_key and chart_type in {"line", "dualaxis"}:
        _add_trendlines_to_chart(
            transformed_df,
            time_key,
            metrics,
            chart_labels,
            ctx.trendline_extend_to,
            chart_meta,
        )

    if chart_meta.get("axisMode") == "dual":
        if chart_type != "line" or "share" in dims:
            chart_type = "dualaxis"

    # Phase 13 (§16.4.2): shadow-mode long-form ChartFrame. The wire payload
    # stays wide for backward compatibility; when ``ENAI_CHART_LONGFORM`` is
    # enabled the long-form records are attached under ``chart_meta["longFrame"]``
    # so frontend / override builders can opt in without a breaking change.
    if ENAI_CHART_LONGFORM:
        try:
            chart_frame = from_wide(
                transformed_df,
                time_key=time_key,
                num_cols=list(metrics),
                dim_map=dim_map,
                label_map=label_map_all,
                measure_transform=effective_measure_transform,
                meta={},
                unit_map={col: dim_map.get(col, "") for col in metrics},
            )
            if not chart_frame.is_empty():
                # Serialise period as ISO string so the payload is JSON-safe.
                long_payload = chart_frame.long_df.copy()
                long_payload["period"] = long_payload["period"].apply(
                    lambda v: v.isoformat() if pd.notna(v) else None
                )
                chart_meta["longFrame"] = long_payload.to_dict("records")
                chart_meta["longFrameColumns"] = list(chart_frame.long_df.columns)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("ENAI_CHART_LONGFORM payload failed: %s", exc)

    return {
        "data": chart_data,
        "type": chart_type,
        "metadata": chart_meta,
    }


def _cap_dimensions(
    metrics: List[str],
    dim_map: Dict[str, str],
    dims: set[str],
    query_lower: str,
    *,
    source: Optional[str] = None,
) -> Tuple[List[str], Dict[str, str], set[str]]:
    if len(dims) <= 2:
        return metrics, dim_map, dims

    # Planner-authored chart groups express explicit multi-dimension intent
    # (e.g., price + share + xrate panels). Silently trimming them would
    # discard upstream reasoning that already identified the right grouping.
    # Log a warning so the drift is visible but preserve all dimensions.
    if source == "plan":
        log.warning(
            "Dimension cap skipped for planner-authored group: kept all %d dims %s "
            "(would have trimmed to top 2 by relevance)",
            len(dims),
            dims,
        )
        return metrics, dim_map, dims

    dim_best: Dict[str, int] = {}
    for col in metrics:
        dim = dim_map[col]
        dim_best[dim] = max(dim_best.get(dim, 0), relevance_score(col, query_lower))
    ranked = sorted(
        dim_best,
        key=lambda dim: (dim_best[dim], _DIM_PRIORITY.get(dim, 0)),
        reverse=True,
    )
    top_dims = set(ranked[:2])
    dropped = set(ranked[2:])
    capped_metrics = [col for col in metrics if dim_map[col] in top_dims]
    capped_dim_map = {col: dim_map[col] for col in capped_metrics}
    capped_dims = set(capped_dim_map.values())
    log.info("Dimension cap: kept %s, dropped %s", top_dims, dropped)
    return capped_metrics, capped_dim_map, capped_dims


def _choose_chart_type(
    *,
    group: Dict[str, Any],
    visualization: Optional[Any],
    has_time: bool,
    has_categories: bool,
    dimensions: set[str],
    category_count: int,
) -> str:
    explicit_group_type = _normalize_chart_type(group.get("type"))
    if explicit_group_type:
        chart_type = explicit_group_type
    else:
        preferred_family = _normalize_chart_type(
            getattr(getattr(visualization, "preferred_chart_family", None), "value", None)
        )
        if preferred_family:
            chart_type = preferred_family
        else:
            visual_goal = getattr(getattr(visualization, "visual_goal", None), "value", None)
            chart_type = _chart_type_for_visual_goal(
                visual_goal=visual_goal,
                has_time=has_time,
                has_categories=has_categories,
                dimensions=dimensions,
                category_count=category_count,
            )
            if chart_type is None:
                chart_type = select_chart_type(
                    has_time=has_time,
                    has_categories=has_categories,
                    dimensions=dimensions,
                    category_count=category_count,
                )

    if dimensions == {"share"} and has_time:
        chart_type = "stackedbar"

    if ("price_tariff" in dimensions or "xrate" in dimensions) and "share" not in dimensions:
        if chart_type in {"bar", "stackedbar", "pie"}:
            log.info("Forced chart_type to 'line': price/xrate must be line (was %s)", chart_type)
            chart_type = "line"

    return chart_type


def _chart_type_for_visual_goal(
    *,
    visual_goal: Optional[str],
    has_time: bool,
    has_categories: bool,
    dimensions: set[str],
    category_count: int,
) -> Optional[str]:
    if visual_goal in {"trend", "relationship"}:
        return "line"
    if visual_goal in {"composition", "decomposition"}:
        if has_time:
            return "stackedbar"
        if has_categories and category_count <= 8 and dimensions == {"share"}:
            return "pie"
        return "bar"
    if visual_goal == "ranking":
        # Ranking is always a bar (or horizontal bar) view — even when a time
        # column is present the user intent is to compare series magnitudes,
        # not to trace a trend over time. Previously mapped to "line" when
        # has_time, which produced semantically wrong charts for queries like
        # "rank generators by average output in 2024".
        return "bar"
    if visual_goal == "compare":
        return "line" if has_time else "bar"
    if visual_goal == "threshold_scan":
        # Threshold scans inspect whether/when a series crosses a level — a
        # line with a reference marker is the natural representation over
        # time; without time, a sorted bar works better than a line segment.
        return "line" if has_time else "bar"
    return None


def _normalize_chart_type(raw_type: Any) -> Optional[str]:
    if raw_type is None:
        return None
    normalized = str(raw_type).strip().lower().replace("-", "_")
    return _NORMALIZED_CHART_TYPES.get(normalized)


def _format_time_labels(df: pd.DataFrame, time_key: Optional[str]) -> None:
    if not time_key or time_key not in df.columns:
        return
    try:
        dt_col = pd.to_datetime(df[time_key])
        all_first_of_month = dt_col.dt.day.eq(1).all()
        all_january = all_first_of_month and dt_col.dt.month.eq(1).all()
        if all_january and len(dt_col) > 1:
            df[time_key] = dt_col.dt.strftime("%Y")
        elif all_first_of_month:
            df[time_key] = dt_col.dt.strftime("%Y-%m")
        else:
            df[time_key] = dt_col.dt.strftime("%Y-%m-%d")
    except Exception:
        pass


def _apply_chart_collection(ctx: QueryContext, charts: List[Dict[str, Any]]) -> None:
    ctx.charts = charts
    primary = charts[0]
    ctx.chart_data = primary.get("data")
    ctx.chart_type = primary.get("type")
    ctx.chart_meta = primary.get("metadata")


def _build_companion_table(
    df: pd.DataFrame,
    label_map: Dict[str, str],
) -> Dict[str, Any]:
    """Phase 12: build a compact table payload to accompany chart_plus_table
    presentations. Uses the *pre-transform* frame so the companion shows the
    raw observed values even when the chart visualizes a derived measure
    (e.g., MoM %, index_100, share_of_total).
    """
    labeled = df.rename(columns=label_map) if label_map else df
    columns = list(labeled.columns)
    return {
        "columns": columns,
        "rows": labeled.to_dict("records"),
    }


def _build_series_config(
    num_cols: List[str],
    chart_labels: List[str],
    dims: set[str],
) -> Dict[str, Dict[str, str]]:
    """Build per-series render config based on dimension semantics."""
    config: Dict[str, Dict[str, str]] = {}
    for col, label in zip(num_cols, chart_labels):
        dim = infer_dimension(col)
        if dim == "share":
            config[label] = {"type": "bar", "stack": "shares", "yAxis": "right"}
        elif dim == "xrate":
            y_axis = "right" if "price_tariff" in dims else "left"
            config[label] = {"type": "line", "yAxis": y_axis, "dashStyle": "dash"}
        elif dim == "price_tariff" and "energy_qty" in dims:
            config[label] = {"type": "line", "yAxis": "right"}
        elif dim == "energy_qty" and "price_tariff" in dims:
            config[label] = {"type": "line", "yAxis": "left"}
        else:
            config[label] = {"type": "line", "yAxis": "left"}
    return config


def _build_chart_metadata(
    dims: set[str],
    chart_type: str,
    num_cols: List[str],
    chart_labels: List[str],
    time_key: Optional[str],
) -> Dict[str, Any]:
    """Build chart metadata based on dimension semantics."""
    series_config = _build_series_config(num_cols, chart_labels, dims)

    dual_axis_combos = [
        (
            {"price_tariff", "xrate"},
            lambda d: "price_tariff" in d and "xrate" in d,
            "Price vs Exchange Rate",
            unit_for_price,
            unit_for_xrate,
        ),
        (
            {"price_tariff", "share"},
            lambda d: "price_tariff" in d and "share" in d,
            "Price vs Composition Shares",
            unit_for_price,
            unit_for_share,
        ),
        (
            {"price_tariff", "energy_qty"},
            lambda d: "price_tariff" in d and "energy_qty" in d,
            "Quantity vs Price/Tariff",
            unit_for_qty,
            unit_for_price,
        ),
        (
            {"xrate", "share"},
            lambda d: "xrate" in d and "share" in d,
            "Exchange Rate vs Composition",
            unit_for_xrate,
            unit_for_share,
        ),
        (
            {"index"},
            lambda d: "index" in d and len(d) > 1,
            "Index vs Other Indicator",
            lambda nc: unit_for_price(nc) if "price_tariff" in dims else unit_for_qty(nc),
            unit_for_index,
        ),
    ]

    for _, condition, title, left_fn, right_fn in dual_axis_combos:
        if condition(dims):
            return {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": left_fn(num_cols),
                "yAxisRight": right_fn(num_cols),
                "title": title,
                "axisMode": "dual",
                "labels": chart_labels,
                "seriesConfig": series_config,
            }

    if dims == {"price_tariff"}:
        y_unit = unit_for_price(num_cols)
    elif dims == {"energy_qty"}:
        y_unit = unit_for_qty(num_cols)
    elif dims == {"index"}:
        y_unit = unit_for_index(num_cols)
    elif dims == {"xrate"}:
        y_unit = unit_for_xrate(num_cols)
    elif dims == {"share"}:
        y_unit = unit_for_share(num_cols)
    else:
        y_unit = "Value"

    return {
        "xAxisTitle": time_key or "time",
        "yAxisTitle": y_unit,
        "title": "Indicator Comparison (same dimension)",
        "axisMode": "single",
        "labels": chart_labels,
        "seriesConfig": series_config,
    }


def _add_trendlines_to_chart(
    df: pd.DataFrame,
    time_key: str,
    num_cols: List[str],
    chart_labels: List[str],
    trendline_extend_to: Optional[str],
    chart_meta: Dict[str, Any],
) -> None:
    """Calculate and add trendlines to chart metadata."""
    from visualization.chart_builder import calculate_trendline

    log.info("Calculating trendlines for %d series", len(num_cols))
    trendlines = []

    for col in num_cols:
        trendline_data = calculate_trendline(
            df,
            time_key,
            col,
            extend_to_date=trendline_extend_to,
        )
        if trendline_data:
            label_idx = num_cols.index(col)
            label = chart_labels[label_idx] if label_idx < len(chart_labels) else col
            trendlines.append(
                {
                    "column": col,
                    "label": f"{label} (Trend)",
                    "data": trendline_data,
                    "original_label": label,
                }
            )
            log.info("Trendline added for %s", label)

    if trendlines:
        chart_meta["trendlines"] = trendlines
        chart_meta["has_projection"] = bool(trendline_extend_to)
        chart_meta["projection_to"] = trendline_extend_to
