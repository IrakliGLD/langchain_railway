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

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# Constants (moved from main.py)
# ---------------------------------------------------------------------------

BALANCING_SHARE_METADATA: dict[str, dict[str, Any]] = {
    "share_regulated_hpp": {"label": "regulated HPP", "cost": "cheap", "usd_linked": False},
    "share_deregulated_hydro": {"label": "deregulated hydro", "cost": "cheap", "usd_linked": False},
    "share_renewable_ppa": {"label": "renewable PPA", "cost": "moderate", "usd_linked": True},
    "share_thermal_ppa": {"label": "thermal PPA", "cost": "expensive", "usd_linked": True},
    "share_import": {"label": "imports", "cost": "expensive", "usd_linked": True},
    "share_regulated_new_tpp": {"label": "new regulated TPP", "cost": "expensive", "usd_linked": True},
    "share_regulated_old_tpp": {"label": "old regulated TPP", "cost": "moderate", "usd_linked": True},
    "share_all_ppa": {"label": "all PPAs", "cost": "expensive", "usd_linked": True},
    "share_all_renewables": {"label": "all renewables", "cost": "mixed", "usd_linked": True},
}

MONTH_NAME_TO_NUMBER = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


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
    expensive_delta = sum(d for _, d, c, _, _ in significant if c in ("expensive", "moderate"))

    if cheap_delta < -0.01:
        notes.append("Cheaper balancing supply contracted — upward price pressure.")
    if expensive_delta > 0.01:
        notes.append("Higher-cost groups expanded — upward price pressure.")
    if expensive_delta < -0.01:
        notes.append("Higher-cost groups contracted — downward price pressure.")

    usd_delta = sum(d for _, d, _, u, _ in significant if u)
    if abs(usd_delta) >= 0.01:
        direction = "expanded" if usd_delta > 0 else "contracted"
        notes.append(f"USD-denominated sellers {direction} by {abs(usd_delta)*100:.1f}pp — xrate sensitivity {'increased' if usd_delta > 0 else 'decreased'}.")

    return notes


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
    keys = ["why", "reason", "cause", "factor", "explain", "due to", "behind", "what caused", "what influenced"]
    t = text_.lower()
    return any(k in t for k in keys)


def _month_from_text(s: str) -> Optional[int]:
    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    for k, v in months.items():
        if k in s:
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
        yrs_in_q = re.findall(r"(20\d{2})", user_query)
        target_years = sorted({int(y) for y in yrs_in_q if int(y) > last["year"]}) or [last["year"] + i for i in range(1, 4)]
        f_rows = []
        for y in target_years:
            val = last[value_col] * ((1 + cagr) ** (y - last["year"]))
            f_rows.append({time_col: pd.to_datetime(f"{y}-01-01"), value_col: val, "is_forecast": True})
        if "is_forecast" not in df.columns:
            df["is_forecast"] = False
        df_f = pd.concat([df, pd.DataFrame(f_rows)], ignore_index=True)
        note_parts.append(f"Forecast years: {', '.join(map(str, target_years))}.")
        return df_f, " ".join(note_parts)

    elif data_type == "price":
        df["year"] = df[time_col].dt.year
        df["month"] = df[time_col].dt.month
        df["season"] = np.where(df["month"].isin([4, 5, 6, 7]), "summer", "winter")

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

        yrs_in_q = re.findall(r"(20\d{2})", user_query)
        target_years = sorted({int(y) for y in yrs_in_q if int(y) > last["year"]}) or [last["year"] + i for i in range(1, 4)]

        f_rows = []
        for y in target_years:
            val_y = last[value_col] * ((1 + cagr_y) ** (y - last["year"]))
            val_s = last[value_col] * ((1 + cagr_s) ** (y - last["year"])) if not np.isnan(cagr_s) else val_y
            val_w = last[value_col] * ((1 + cagr_w) ** (y - last["year"])) if not np.isnan(cagr_w) else val_y
            f_rows.append({time_col: pd.to_datetime(f"{y}-04-01"), "season": "summer", value_col: val_s, "is_forecast": True})
            f_rows.append({time_col: pd.to_datetime(f"{y}-12-01"), "season": "winter", value_col: val_w, "is_forecast": True})

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
    if share_query_detected:
        try:
            ctx.share_summary_override = generate_share_summary(share_df_for_summary, ctx.plan, ctx.query)
            if ctx.share_summary_override:
                log.info("✅ Generated deterministic share summary override.")
        except Exception as share_err:
            log.warning(f"Share summary override failed: {share_err}")

    # --- Correlation analysis ---
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
            corr_df = corr_df[[c for c in corr_df.columns if c in (["date"] + allowed_targets + allowed_drivers)]]

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
                summer_df = corr_df[corr_df["month"].isin([4, 5, 6, 7])].drop(columns=["date", "month"], errors="ignore")
                winter_df = corr_df[~corr_df["month"].isin([4, 5, 6, 7])].drop(columns=["date", "month"], errors="ignore")

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
    if _detect_forecast_mode(ctx.query) and not ctx.df.empty:
        try:
            ctx.df, _forecast_note = _generate_cagr_forecast(ctx.df, ctx.query)
            ctx.stats_hint += f"\n\n--- FORECAST NOTE ---\n{_forecast_note}"
            log.info(_forecast_note)
        except Exception as _e:
            log.warning(f"Forecast generation failed: {_e}")

    # --- Why mode (causal reasoning) ---
    if _detect_why_mode(ctx.query) and not ctx.df.empty:
        try:
            _build_why_context(ctx)
        except Exception as _e:
            log.warning(f"'Why' reasoning context build failed: {_e}")

    # --- Trendline detection ---
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
            ctx.trendline_extend_to = f"{current_year + 2}-12-01"
        log.info(f"📈 Trendline requested: extending to {ctx.trendline_extend_to}")

        # Pre-calculate trendlines for forecast answer generation
        _precalculate_trendlines(ctx, cols_labeled)

    return ctx


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
    target_period = pd.Timestamp(years[0], mon or 1, 1) if years else df[t_series_col].iloc[-1]

    cur_row = df.loc[df[t_series_col] == target_period]
    if cur_row.empty:
        cur_row = df[df[t_series_col] <= target_period].tail(1)

    if cur_row.empty:
        log.warning("No data found for target period in 'why' analysis")
        return

    prev_row = df[df[t_series_col] < cur_row[t_series_col].iloc[0]].tail(1)

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

    cur_gel = _get_val(cur_row, ["p_bal_gel"])
    prev_gel = _get_val(prev_row, ["p_bal_gel"]) if not prev_row.empty else None
    cur_usd = _get_val(cur_row, ["p_bal_usd"])
    prev_usd = _get_val(prev_row, ["p_bal_usd"]) if not prev_row.empty else None
    cur_xrate = _get_val(cur_row, ["xrate"])
    prev_xrate = _get_val(prev_row, ["xrate"]) if not prev_row.empty else None

    share_cols = [c for c in df.columns if c.startswith("share_")]
    cur_shares: dict[str, float] = {}
    prev_shares: dict[str, float] = {}

    target_ts = pd.to_datetime(cur_row[t_series_col].iloc[0], errors="coerce") if not cur_row.empty else None

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

    deltas = {k: round(cur_shares.get(k, 0) - prev_shares.get(k, 0), 4) for k in cur_shares}

    why_ctx["signals"] = {
        "period": str(cur_row[t_series_col].iloc[0]) if not cur_row.empty else None,
        "p_bal_gel": {"cur": cur_gel, "prev": prev_gel},
        "p_bal_usd": {"cur": cur_usd, "prev": prev_usd},
        "xrate": {"cur": cur_xrate, "prev": prev_xrate},
        "share_deltas": deltas,
    }

    if cur_shares:
        why_ctx["signals"]["share_snapshot"] = {k: round(v, 4) for k, v in cur_shares.items()}
    if prev_shares:
        why_ctx["signals"]["share_prev_snapshot"] = {k: round(v, 4) for k, v in prev_shares.items()}

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

    why_ctx["notes"].append("Balancing price is a weighted average of electricity sold as balancing energy.")
    why_ctx["notes"].append("Regulated and deregulated hydro depend weakly on xrate; thermal PPAs and imports depend strongly on xrate.")
    why_ctx["notes"].append("When GEL depreciates, electricity prices generally rise because USD-denominated constituents (like imported gas) become more expensive.")
    why_ctx["notes"].append("If GEL depreciates, GEL-denominated balancing price rises due to USD-linked gas/import costs.")
    why_ctx["notes"].append("Composition shift toward thermal or import increases price; more hydro or renewable lowers it.")

    ctx.stats_hint += "\n\n--- CAUSAL CONTEXT ---\n" + json.dumps(why_ctx, default=str, indent=2)
    log.info("Why-context attached to stats_hint.")


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
