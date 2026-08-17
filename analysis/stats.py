"""
Statistical analysis and data preview generation.

Handles:
- Quick statistics generation for query results
- Trend analysis (yearly, seasonal)
- CAGR calculations for summer/winter periods
- Data preview formatting
"""
import logging
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from analysis.system_quantities import normalize_period_series_with_granularity
from config import PREVIEW_MAX_CHARS, PREVIEW_MAX_ROWS, SUMMER_MONTHS

log = logging.getLogger("Enai")


# Intensive metrics (per-unit values like prices/rates) must be AVERAGED across
# periods, never summed — summing a per-MWh price over 12 months yields a
# meaningless inflated level (e.g. 12×123 ≈ 1,482 GEL/MWh). Extensive metrics
# (quantities/volumes) sum to a meaningful annual total.
# NOTE: keep tokens that won't substring-match an extensive word. "ratio" is
# deliberately excluded — it is a substring of "gene[ratio]n".
_INTENSIVE_TOKENS = (
    "price", "tariff", "xrate", "rate", "p_bal", "cost",
    "share", "pct", "percent", "cagr",
)
_EXTENSIVE_TOKENS = (
    "quantity", "volume", "generation", "demand",
    "consumption", "export", "import", "supply",
)
# A currency-per-energy unit in the name is DEFINITIVE: the column holds a price
# per unit of energy, so it is intensive whatever else the name says. This wins
# over the word heuristic below, because an extensive word can appear as a
# QUALIFIER on a per-unit price without making it a volume —
# "supply_vs_wholesale_spread_gel_kwh" is a per-kWh spread, and the word "supply"
# had it summed across 66 months into the grounding corpus (2026-08-16 trace).
# NOTE: bare energy units ("_mwh", "_gwh") are deliberately absent — those really
# do name quantities, so "demand_mwh" must stay extensive.
_CURRENCY_PER_ENERGY_TOKENS = (
    "gel_kwh", "gel_mwh", "usd_kwh", "usd_mwh",
)


def is_intensive_metric(col: str) -> bool:
    """Whether *col* holds an intensive (per-unit) value that must be averaged
    rather than summed across periods.

    A currency-per-energy unit in the name settles it outright; otherwise
    intensive tokens take precedence (a price-of-X is still a price); unknown
    columns default to intensive — averaging never inflates a level, and a
    period-over-period % trend is identical either way, so this is the safe bias.

    Separators are normalised first so a raw name and its display label classify
    the SAME way: agent/analyzer.py labels before calling quick_stats, where the
    unit reads "(GEL/kWh)" rather than "_gel_kwh". Matching only the underscore
    form fixed the column-aggregates block and left the yearly-trend line still
    summing a per-kWh spread.
    """
    c = re.sub(r"[^a-z0-9]+", "_", (col or "").lower())
    if any(tok in c for tok in _CURRENCY_PER_ENERGY_TOKENS):
        return True
    if any(tok in c for tok in _INTENSIVE_TOKENS):
        return True
    if any(tok in c for tok in _EXTENSIVE_TOKENS):
        return False
    return True


#: Says rows were dropped. Without it the model reads the last kept row and the
#: first kept tail row as consecutive periods -- on the 2026-08-16 trace that was
#: 2026-06 directly followed by 2021-09, with nothing in between and no sign
#: anything was missing.
PREVIEW_OMISSION_MARKER = "...[rows omitted from preview]..."

#: Column names that look like a time axis. Same tuple the trend analysis uses.
_TIME_KEYWORDS = ("date", "year", "month", "period")

#: A text column is worth moving into a legend only if it is this wide on
#: average. Short codes cost less inline than they would as legend entries.
_LEGEND_MIN_AVG_WIDTH = 12

#: ...and only if it takes few enough distinct values that the legend is far
#: smaller than the repetition it replaces.
_LEGEND_MAX_DISTINCT = 32


def _preview_time_column(df):
    """The frame's time axis by name, or None."""
    for col in df.columns:
        if any(kw in str(col).lower() for kw in _TIME_KEYWORDS):
            return col
    return None


def _extract_repeated_labels(df):
    """Move long, key-determined text columns out of the rows into a legend.

    On the 2026-08-16 retail frame three columns -- ``supply_company`` (43
    chars), ``series_label`` (38) and ``category_label`` (up to 37) -- carried
    roughly 110 of each 197-char row and were repeated across all 528 rows,
    while each is already implied by a short key on the same row. Stating the
    mapping once and keeping the key roughly doubles how much of the frame fits
    in the same budget.

    Deliberately expressed as a property of the data -- a wide text column that
    is constant within some narrower column's groups -- rather than as a list of
    retail column names, so it does not need editing when a tool adds a column.

    Returns ``(df, legend_text)``; ``legend_text`` is "" when nothing qualified.
    """
    candidates = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    widths = {col: df[col].astype(str).map(len).mean() for col in candidates}

    dropped: list[str] = []
    legend_lines: list[str] = []
    for col in candidates:
        if widths[col] < _LEGEND_MIN_AVG_WIDTH:
            continue
        values = df[col].astype(str)
        distinct = values.nunique()
        # One value per row is an identifier, not a label: there is nothing to
        # factor out and the legend would be as long as the column.
        if distinct > _LEGEND_MAX_DISTINCT or distinct >= len(df):
            continue

        # The narrowest column that determines this one. Narrower matters: the
        # key stays on every row, so a key wider than the label saves nothing.
        key = None
        for candidate_key in candidates:
            if candidate_key == col or candidate_key in dropped:
                continue
            if widths[candidate_key] >= widths[col]:
                continue
            if df.groupby(df[candidate_key].astype(str), dropna=False)[col].nunique(
                dropna=False
            ).max() != 1:
                continue
            if key is None or df[candidate_key].nunique() < df[key].nunique():
                key = candidate_key
        if key is None:
            continue

        dropped.append(col)
        mapping = dict(zip(df[key].astype(str), values))
        for key_value, label in sorted(mapping.items()):
            legend_lines.append(f"  {key}={key_value} -> {col}={label}")

    if not dropped:
        return df, ""

    legend = (
        "LEGEND (constant per key, stated once instead of on every row):\n"
        + "\n".join(legend_lines)
        + "\n"
    )
    return df.drop(columns=dropped), legend


def _sample_whole_periods(df, max_chars: int):
    """Keep every series at evenly spaced dates across the full span.

    Head-and-tail truncation on a date-ordered frame keeps only the ends: on the
    2026-08-16 trace it left the five most recent months and the three oldest,
    so 2022, 2023, 2024 and 2025 -- including both years the question named --
    reached the model not at all.

    Sampling by DATE rather than by row position is what keeps a cross-series
    comparison possible: every retained date carries all of its series, so the
    categories can still be set against each other at any period shown. Sampling
    rows would leave different categories present on different dates.

    Returns the CSV, or None when the frame has no usable time axis or spans too
    little for sampling to beat the existing head/tail behaviour.
    """
    time_col = _preview_time_column(df)
    if time_col is None:
        return None
    periods = pd.to_datetime(df[time_col], errors="coerce")
    if periods.isna().any():
        return None
    distinct = sorted(periods.unique())
    if len(distinct) < 4:
        return None
    span_days = (distinct[-1] - distinct[0]) / pd.Timedelta(days=1)
    if span_days < 365:
        return None

    csv_len = len(df.to_csv(index=False))
    chars_per_row = max(1.0, csv_len / max(1, len(df)))
    affordable_rows = max(1, int(max_chars / chars_per_row))
    rows_per_date = max(1, round(len(df) / len(distinct)))
    keep_count = max(2, min(len(distinct), affordable_rows // rows_per_date))

    if keep_count >= len(distinct):
        return None

    # Evenly spaced, endpoints always included: the newest and oldest periods
    # anchor the range the answer is allowed to describe.
    step = (len(distinct) - 1) / (keep_count - 1)
    indices = sorted({int(round(position * step)) for position in range(keep_count)})
    keep = {distinct[index] for index in indices}

    sampled = df[periods.isin(keep)]
    note = (
        f"{PREVIEW_OMISSION_MARKER} sampled to {len(keep)} of {len(distinct)} periods, "
        "evenly spaced across the full range; every period shown carries all series"
    )
    return note + "\n" + sampled.to_csv(index=False)


def rows_to_preview(
    rows: List[Tuple],
    cols: List[str],
    max_rows: int | None = None,
    max_preview_chars: int | None = None,
) -> str:
    """
    Convert query results to compact CSV preview for LLM consumption.

    Args:
        rows: List of tuples containing query results
        cols: List of column names
        max_rows: Maximum rows before the character budget applies. Defaults to
            ``PREVIEW_MAX_ROWS``. This is a head slice with no tail
            preservation, so it should stay high enough that the character cap
            below is what actually binds.
        max_preview_chars: Soft cap on output size; if exceeded, middle rows
            are progressively dropped while preserving the first and last rows
            so the LLM sees the full date range. Defaults to
            ``PREVIEW_MAX_CHARS``.

    Returns:
        CSV-formatted string (header + data rows)
    """
    if not rows:
        return "No rows returned."

    max_rows = PREVIEW_MAX_ROWS if max_rows is None else max_rows
    max_preview_chars = PREVIEW_MAX_CHARS if max_preview_chars is None else max_preview_chars

    df = pd.DataFrame(rows[:max_rows], columns=cols)

    # Round numeric columns to 3 decimal places
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float).round(3)

    preview = df.to_csv(index=False)
    if len(preview) <= max_preview_chars:
        return preview

    # Everything below applies only because the frame did NOT fit. A preview
    # within budget is returned above, untouched.

    # 1. Width first: repeated labels cost more than the rows they describe.
    df, legend = _extract_repeated_labels(df)
    preview = df.to_csv(index=False)
    if len(legend) + len(preview) <= max_preview_chars:
        return legend + preview

    # 2. Then depth, by whole periods, so every year stays represented.
    sampled = _sample_whole_periods(df, max_preview_chars - len(legend))
    if sampled is not None:
        return legend + sampled

    # 3. Last resort: keep both ends and say what went missing. The marker is
    #    the difference between a gap and an apparently continuous series.
    while len(preview) > max_preview_chars and len(df) > 20:
        keep_head = max(10, len(df) // 2)
        keep_tail = max(5, len(df) // 4)
        head, tail = df.head(keep_head), df.tail(keep_tail)
        df = pd.concat([head, tail])
        preview = df.to_csv(index=False)

    if len(df) < len(rows[:max_rows]):
        lines = preview.splitlines(keepends=True)
        boundary = 1 + len(df.head(len(df) // 2 if len(df) > 20 else len(df)))
        preview = "".join(lines[:boundary]) + (
            f"{PREVIEW_OMISSION_MARKER} middle rows dropped; both ends retained\n"
        ) + "".join(lines[boundary:])

    return legend + preview


def quick_stats(rows: List[Tuple], cols: List[str]) -> str:
    """
    Generate quick statistics for query results.

    Provides:
    - Row count
    - Yearly trend analysis (first full year → last full year)
    - Seasonal trends with CAGR (summer vs winter)
    - Period range
    - Numeric summary statistics

    Args:
        rows: List of tuples containing query results
        cols: List of column names

    Returns:
        String summary of statistics and trends

    Examples:
        >>> rows = [('2023-01-01', 100), ('2023-06-01', 120), ('2024-01-01', 130)]
        >>> cols = ['date', 'price']
        >>> stats = quick_stats(rows, cols)
        >>> print(stats)
        Rows: 3
        Trend (Yearly Avg, 2023→2024): increasing (18.2%)
        ...
    """
    if not rows:
        return "0 rows."

    df = pd.DataFrame(rows, columns=cols).copy()  # Protect original data
    out = [f"Rows: {len(df)}"]

    # 1. Detect date/year column
    _time_kws = ("date", "year", "month", "period")
    date_cols = [c for c in df.columns if any(kw in c.lower() for kw in _time_kws)]
    if not date_cols:
        # Fallback to simple stats if no date or numeric data
        return "\n".join(out)

    time_col = date_cols[0]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != time_col]
    if not numeric_cols:
        return "\n".join(out)
    time_granularity = None

    # --- Trend Calculation: Compare First Full Year vs Last Full Year ---
    try:
        df[time_col], time_granularity = normalize_period_series_with_granularity(df[time_col])

        # Verify conversion worked before using .dt accessors
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            # Conversion failed, still object dtype - skip trend calculation
            log.warning(f"⚠️ Column {time_col} could not be converted to datetime, skipping trend")
            return "\n".join(out)

        df = df.dropna(subset=[time_col]).drop_duplicates().sort_values(time_col)
        df['__year'] = df[time_col].dt.year

        if time_granularity != "year":
            try:
                months_per_year = (
                    df.assign(_year_month=df[time_col].dt.to_period("M"))
                    .groupby("__year")["_year_month"]
                    .nunique()
                    .sort_index()
                )
                incomplete_years = months_per_year[months_per_year < 12].index.tolist()
                if incomplete_years:
                    log.info(
                        "Excluding incomplete years from trend calculation (granularity=%s): %s",
                        time_granularity or "unknown",
                        incomplete_years,
                    )
                    df = df[~df['__year'].isin(incomplete_years)]
            except Exception as e:
                log.warning(f"⚠️ Failed to filter incomplete years: {e}")

        valid_years = df['__year'].dropna().unique()
        if len(valid_years) >= 2:
            first_full_year = int(valid_years.min())
            last_full_year = int(valid_years.max())

            # Ensure we are comparing two different years
            if first_full_year != last_full_year:

                # Compare each metric independently. Combining prices,
                # quantities, shares, and rates into one mean is dimensionally
                # meaningless and can reverse the apparent trend.
                for col in numeric_cols:
                    intensive = is_intensive_metric(col)
                    aggregate_kind = "Avg" if intensive else "Total"
                    df_first = df.loc[df['__year'] == first_full_year, col]
                    df_last = df.loc[df['__year'] == last_full_year, col]
                    first_value = df_first.mean() if intensive else df_first.sum(min_count=1)
                    last_value = df_last.mean() if intensive else df_last.sum(min_count=1)
                    if pd.isna(first_value) or pd.isna(last_value):
                        continue
                    change = (
                        (last_value - first_value) / first_value * 100
                        if first_value != 0 else np.nan
                    )
                    if np.isclose(last_value, first_value, rtol=1e-9, atol=1e-12):
                        trend = "stable"
                    else:
                        trend = "increasing" if last_value > first_value else "decreasing"
                    change_text = f"{change:.1f}%" if np.isfinite(change) else "undefined from zero baseline"
                    metric_part = f", {col}" if len(numeric_cols) > 1 else ""
                    out.append(
                        f"Trend (Yearly {aggregate_kind}{metric_part}, "
                        f"{first_full_year}→{last_full_year}): {trend} ({change_text})"
                    )

                # --- Seasonal split (Summer vs Winter) with CAGR ---
                try:
                    df['month'] = df[time_col].dt.month
                    summer_mask = df['month'].isin(SUMMER_MONTHS)
                    winter_mask = ~summer_mask

                    def seasonal_avg(df_season, col, year):
                        """Calculate seasonal average for a specific year."""
                        return df_season.loc[df_season['__year'] == year, col].mean()

                    def seasonal_cagr(df_season, col):
                        """Compute CAGR (Compound Annual Growth Rate) for a column across years within a seasonal subset."""
                        df_y = df_season.groupby('__year')[col].mean().dropna()
                        if len(df_y) >= 2:
                            first, last = df_y.iloc[0], df_y.iloc[-1]
                            n = int(df_y.index[-1]) - int(df_y.index[0])
                            return ((last / first) ** (1 / n) - 1) * 100 if first > 0 else np.nan
                        return np.nan

                    for col in numeric_cols:
                        if 'p_bal' in col.lower() or 'price' in col.lower():
                            summer_first = seasonal_avg(df.loc[summer_mask], col, first_full_year)
                            summer_last = seasonal_avg(df.loc[summer_mask], col, last_full_year)
                            winter_first = seasonal_avg(df.loc[winter_mask], col, first_full_year)
                            winter_last = seasonal_avg(df.loc[winter_mask], col, last_full_year)

                            cagr_summer = seasonal_cagr(df.loc[summer_mask], col)
                            cagr_winter = seasonal_cagr(df.loc[winter_mask], col)

                            # Report both the absolute seasonal shift and the compounded annual pace.
                            out.append(
                                f"Seasonal Trend ({col}): "
                                f"Summer {first_full_year}→{last_full_year}: "
                                f"{(summer_last - summer_first):.1f} Δ, CAGR {cagr_summer:.2f}%; "
                                f"Winter {first_full_year}→{last_full_year}: "
                                f"{(winter_last - winter_first):.1f} Δ, CAGR {cagr_winter:.2f}%."
                            )
                except Exception as e:
                    log.warning(f'⚠️ Seasonal trend calculation failed: {e}')

            else:
                out.append("Trend: Less than one full year of data for comparison.")

        else:
            out.append("Trend: Insufficient data for yearly comparison.")

    except Exception as e:
        log.warning(f"⚠️ Yearly trend calculation failed: {e}")
        # Fallback: skip trend calculation

    # Date range display
    first = df[time_col].min()
    last = df[time_col].max()
    if pd.isna(first) or pd.isna(last):
        out.append("Period: unavailable")
    elif time_granularity == "year":
        out.append(f"Period: {int(first.year)} → {int(last.year)}")
    else:
        out.append(f"Period: {first.date()} → {last.date()}")

    # Numeric summary
    if numeric_cols:
        desc = df[numeric_cols].describe().round(3)
        out.append("Numeric summary:")
        out.append(desc.to_string())

    return "\n".join(out)
