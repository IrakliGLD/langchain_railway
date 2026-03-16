"""
Pipeline Stage 5: Chart Building

Handles data coercion, column labeling, chart type selection (semantic-aware),
dimension inference, unit inference, dual-axis logic, trendline calculation,
and series limiting.
"""
import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from models import QueryContext
from visualization.chart_selector import should_generate_chart
from context import COLUMN_LABELS
from utils.trace_logging import trace_detail

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# Dimension/unit inference helpers (moved from ask_post inner functions)
# ---------------------------------------------------------------------------

def infer_dimension(col: str) -> str:
    """Infer semantic dimension of a column for chart axis decisions."""
    col_l = col.lower()
    if any(x in col_l for x in ["xrate", "exchange", "rate", "კურსი"]):
        return "xrate"
    if any(x in col_l for x in ["share_", "წილი_", "proportion", "percent", "პროცენტ"]):
        return "share"
    if any(x in col_l for x in ["cpi", "index", "inflation", "ინდექსი"]):
        return "index"
    if any(x in col_l for x in ["quantity", "generation", "volume_tj", "volume", "mw", "tj", "რაოდენობა", "მოცულობა", "გენერაცია"]):
        return "energy_qty"
    if any(x in col_l for x in ["price", "tariff", "_gel", "_usd", "p_bal", "p_dereg", "p_gcap", "ფასი", "ტარიფი"]):
        return "price_tariff"
    return "other"


def unit_for_price(cols_: list[str]) -> str:
    has_gel = any("_gel" in c.lower() for c in cols_)
    has_usd = any("_usd" in c.lower() for c in cols_)
    if has_gel and has_usd:
        return "per MWh"
    if has_gel:
        return "GEL/MWh"
    if has_usd:
        return "USD/MWh"
    return "per MWh"


def unit_for_qty(cols_: list[str]) -> str:
    has_tj = any("tj" in c.lower() for c in cols_)
    has_thousand_mwh = any("quantity" in c.lower() for c in cols_)
    if has_tj and not has_thousand_mwh:
        return "TJ"
    if has_thousand_mwh and not has_tj:
        return "thousand MWh"
    return "Energy Quantity"


def unit_for_index(_: list[str]) -> str:
    return "Index (2015=100)"


def unit_for_xrate(_: list[str]) -> str:
    return "GEL per USD"


def unit_for_share(_: list[str]) -> str:
    return "Share (0-1)"


def relevance_score(col: str, query_lower: str) -> int:
    """Score a column by keyword relevance to the user's query."""
    score = 0
    col_lower = col.lower()

    if any(k in query_lower for k in ["price", "ფასი", "цена"]) and any(k in col_lower for k in ["price", "p_bal", "ფასი"]):
        score += 10
    if any(k in query_lower for k in ["xrate", "exchange", "კურსი", "курс"]) and "xrate" in col_lower:
        score += 10
    if any(k in query_lower for k in ["share", "წილი", "доля", "composition"]) and "share" in col_lower:
        score += 5
    if any(k in query_lower for k in ["tariff", "ტარიფი", "тариф"]) and "tariff" in col_lower:
        score += 5
    if "p_bal" in col_lower:
        score += 3
    if "xrate" in col_lower:
        score += 2

    return score


# ---------------------------------------------------------------------------
# Main pipeline stage
# ---------------------------------------------------------------------------

def build_chart(ctx: QueryContext) -> QueryContext:
    """Stage 5: Build chart data from query results.

    Reads: ctx.df, ctx.rows, ctx.cols, ctx.plan, ctx.query,
           ctx.skip_chart_due_to_relevance, ctx.add_trendlines, ctx.trendline_extend_to
    Writes: ctx.chart_data, ctx.chart_type, ctx.chart_meta
    """
    if not ctx.rows or not ctx.cols:
        return ctx

    df = ctx.df.copy()
    cols = ctx.cols

    # --- Detect time column ---
    time_key = next((c for c in cols if any(k in c.lower() for k in ["date", "year", "month", "თვე", "წელი", "თარიღი"])), None)

    # --- Fix year-only columns ---
    if time_key and time_key in df.columns:
        try:
            first_val = df[time_key].iloc[0]
            if isinstance(first_val, (int, float)) or str(type(first_val).__name__) == "Decimal":
                first_val_num = float(first_val)
                if 1900 <= first_val_num <= 2100:
                    df[time_key] = pd.to_datetime(df[time_key].astype(int), format="%Y")
                    log.info(f"📅 Converted year-only column '{time_key}' to datetime format for chart building")
        except Exception as e:
            log.warning(f"Year column conversion check failed in chart builder: {e}")

    # --- Categorical detection & coercion ---
    categorical_hints = [
        "type", "tech", "entity", "sector", "source", "segment",
        "region", "category", "ownership", "market", "trade", "fuel",
        "ტიპი", "სექტორი", "წყარო"
    ]
    for c in cols:
        if c != time_key:
            if any(h in c.lower() for h in categorical_hints):
                df[c] = df[c].astype(str).replace("nan", None)
            else:
                try:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                except Exception:
                    pass

    categorical_cols = [
        c for c in df.columns
        if c != time_key and not pd.api.types.is_numeric_dtype(df[c])
    ]

    # --- Apply column labels ---
    label_map_all = {c: COLUMN_LABELS.get(c, c.replace("_", " ").title()) for c in cols if c != time_key}

    for c in categorical_cols:
        new_name = label_map_all.get(c, c.replace("_", " ").title())
        if new_name != c:
            df.rename(columns={c: new_name}, inplace=True)

    # --- Reorder columns ---
    ordered_cols = []
    if time_key:
        ordered_cols.append(time_key)
    ordered_cols += categorical_cols
    for c in df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols]

    # --- Numeric columns ---
    num_cols = [
        c for c in df.columns
        if c != time_key
        and pd.api.types.is_numeric_dtype(df[c])
        and not re.search(r"\b(month|year)\b", c.lower())
    ]

    # --- Decide whether to generate chart ---
    generate_chart = should_generate_chart(ctx.query, len(df))

    intent = str(ctx.plan.get("intent", "")).lower()
    query_text = ctx.query.lower()

    # Chart group filtering from plan
    chart_strategy = ctx.plan.get("chart_strategy", "single")
    chart_groups = ctx.plan.get("chart_groups", [])

    if chart_groups and len(chart_groups) > 0:
        first_group = chart_groups[0]
        chart_metrics = first_group.get("metrics", [])
        if chart_metrics:
            original_num_cols = num_cols.copy()
            num_cols = [col for col in num_cols if col in chart_metrics]
            if num_cols:
                log.info(f"📊 Filtered chart metrics: {len(original_num_cols)} → {len(num_cols)} columns")
                cols_to_keep = [time_key] + num_cols if time_key and time_key in df.columns else num_cols
                cols_to_keep = [c for c in cols_to_keep if c in df.columns]
                df = df[cols_to_keep]
            else:
                num_cols = original_num_cols

    # Override: disable chart for explanatory questions with small results
    if any(word in query_text for word in ["why", "how", "reason", "explain", "because", "cause", "რატომ", "როგორ", "почему"]):
        # Allow chart even for 2 rows if it's a compositional comparison (shares available)
        has_shares = any("share" in c.lower() for c in num_cols)
        if len(df) < 2 or (len(df) < 5 and not has_shares):
            generate_chart = False
            log.info(f"🧭 Skipping chart: explanatory query with small non-compositional result (rows={len(df)}, has_shares={has_shares})")

    if any(word in query_text for word in ["define", "meaning of", "განმარტება"]):
        generate_chart = False

    if ctx.skip_chart_due_to_relevance:
        generate_chart = False
        log.info("🧭 Skipping chart generation: SQL query not relevant to user question")

    if not generate_chart:
        log.info(f"🧭 Skipping chart generation (rows={len(df)}).")
        return ctx

    log.info("🎨 Proceeding with chart generation.")

    # --- Chart type selection ---
    time_cols = [c for c in df.columns if re.search(r"(year|month|date|წელი|თვე|თარიღი)", c.lower())]
    category_cols = [c for c in df.columns if re.search(r"(type|sector|entity|source|segment|ownership|technology|region|area|category|ტიპი|სექტორი)", c.lower())]

    dim_map = {c: infer_dimension(c) for c in num_cols}
    dims = set(dim_map.values())
    log.info(f"📐 Detected dimensions: {dim_map} → {dims}")

    chart_type = "line"
    has_time = len(time_cols) >= 1
    has_categories = len(category_cols) >= 1

    chart_reason = "default_line"
    if has_time and has_categories:
        if "share" in dims:
            chart_type = "stackedbar"
            chart_reason = "time+category+share_dim"
        else:
            chart_type = "line"
            chart_reason = "time+category"
    elif has_time and not has_categories:
        chart_type = "line"
        chart_reason = "time_series"
    elif not has_time and has_categories:
        if "share" in dims and len(category_cols) == 1:
            unique_cats = df[category_cols[0]].nunique()
            chart_type = "pie" if unique_cats <= 8 else "bar"
            chart_reason = f"categorical_share(cats={unique_cats})"
        else:
            chart_type = "bar"
            chart_reason = "categorical"
    else:
        chart_type = "line"
        chart_reason = "no_time_no_category_fallback"

    log.info(f"🧠 Chart type: {chart_type}")

    trace_detail(
        log, ctx, "stage_5_chart_build", "selection",
        chart_type=chart_type,
        row_count=len(df),
        dimension_count=len(dims),
        series_count=len(num_cols),
        reason=chart_reason,
        has_time=has_time,
        has_categories=has_categories,
    )

    # --- Limit series ---
    MAX_SERIES = 3
    if len(num_cols) > MAX_SERIES:
        query_lower = ctx.query.lower()
        scored_cols = [(col, relevance_score(col, query_lower)) for col in num_cols]
        scored_cols.sort(key=lambda x: x[1], reverse=True)
        num_cols = [col for col, _ in scored_cols[:MAX_SERIES]]
        log.info(f"📊 Limited to {MAX_SERIES} series: {num_cols}")
        dim_map = {c: infer_dimension(c) for c in num_cols}
        dims = set(dim_map.values())

    # --- Label map ---
    chart_labels = [label_map_all.get(c, c) for c in num_cols]
    df_labeled = df.rename(columns=label_map_all)

    # --- Axis mode & metadata ---
    chart_data = df_labeled.to_dict("records")
    chart_meta = _build_chart_metadata(
        dims, chart_type, num_cols, chart_labels, time_key
    )
    if chart_meta.get("axisMode") == "dual":
        chart_type = "dualaxis"

    log.info(f"✅ Chart built | type={chart_type} | axisMode={chart_meta.get('axisMode')}")

    # --- Trendlines ---
    if ctx.add_trendlines and time_key and time_key in df.columns:
        _add_trendlines_to_chart(df, time_key, num_cols, chart_labels, ctx.trendline_extend_to, chart_meta)

    ctx.chart_data = chart_data
    ctx.chart_type = chart_type
    ctx.chart_meta = chart_meta
    return ctx


def _build_chart_metadata(
    dims: set, chart_type: str, num_cols: list, chart_labels: list, time_key: Optional[str]
) -> Dict[str, Any]:
    """Build chart metadata based on dimension semantics."""

    # Dual-axis combinations
    dual_axis_combos = [
        ({"price_tariff", "xrate"}, lambda d: "price_tariff" in d and "xrate" in d,
         "Price vs Exchange Rate", unit_for_price, unit_for_xrate),
        ({"price_tariff", "share"}, lambda d: "price_tariff" in d and "share" in d,
         "Price vs Composition Shares", unit_for_price, unit_for_share),
        ({"price_tariff", "energy_qty"}, lambda d: "price_tariff" in d and "energy_qty" in d,
         "Quantity vs Price/Tariff", unit_for_qty, unit_for_price),
        ({"xrate", "share"}, lambda d: "xrate" in d and "share" in d,
         "Exchange Rate vs Composition", unit_for_xrate, unit_for_share),
        ({"index"}, lambda d: "index" in d and len(d) > 1, "Index vs Other Indicator",
         lambda nc: unit_for_price(nc) if "price_tariff" in dims else unit_for_qty(nc),
         unit_for_index),
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
            }

    # Single axis
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
    }


def _add_trendlines_to_chart(
    df: pd.DataFrame, time_key: str, num_cols: list,
    chart_labels: list, trendline_extend_to: Optional[str],
    chart_meta: Dict[str, Any],
) -> None:
    """Calculate and add trendlines to chart metadata."""
    from visualization.chart_builder import calculate_trendline

    log.info(f"📈 Calculating trendlines for {len(num_cols)} series")
    trendlines = []

    for col in num_cols:
        trendline_data = calculate_trendline(
            df, time_key, col, extend_to_date=trendline_extend_to
        )
        if trendline_data:
            label_idx = num_cols.index(col)
            label = chart_labels[label_idx] if label_idx < len(chart_labels) else col
            trendlines.append({
                "column": col,
                "label": f"{label} (Trend)",
                "data": trendline_data,
                "original_label": label
            })
            log.info(f"  ✅ {label}: R²={trendline_data['r_squared']:.3f}")

    if trendlines:
        chart_meta["trendlines"] = trendlines
        chart_meta["has_projection"] = bool(trendline_extend_to)
        chart_meta["projection_to"] = trendline_extend_to
        log.info(f"📊 Added {len(trendlines)} trendlines to chart metadata")
