"""Helpers for normalizing system quantity datasets into analysis-ready frames."""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from context import DEMAND_TECH_TYPES, SUPPLY_TECH_TYPES

_DOMESTIC_GENERATION_TYPES = ("hydro", "thermal", "wind", "solar")
_LOCAL_GENERATION_TYPES = ("hydro", "wind", "solar")
_IMPORT_DEPENDENT_TYPES = ("thermal", "import")

# A per-technology generation share is only meaningful when the period observes
# at least this many generation techs. The long-form source records a row per
# (period, tech) only when that tech has data — an ABSENT tech is unknown, not
# zero (real zeros are stored as explicit 0 rows, e.g. import=0). So a period
# that observes a single generation tech has an incomplete denominator, and
# ``hydro / hydro = 100%`` is a data-coverage artifact, not a fact
# (2026-07-08 report: share_hydro read as constant 100%). Such shares are set
# to NA. Real-zero techs (0 rows) count as observed, so a genuine 100%-hydro
# month with explicit thermal=0/wind=0 rows is preserved.
_MIN_GENERATION_TECHS_FOR_SHARE = 2


def infer_period_granularity(series: pd.Series) -> Optional[str]:
    """Infer whether a normalized timestamp series is yearly, monthly, or finer."""

    if series is None:
        return None

    timestamps = pd.to_datetime(series, errors="coerce")
    non_null = timestamps.dropna()
    if non_null.empty:
        return None

    if non_null.dt.month.eq(1).all() and non_null.dt.day.eq(1).all():
        rows_per_year = non_null.dt.year.value_counts(dropna=True)
        if not rows_per_year.empty and int(rows_per_year.max()) == 1:
            return "year"

    if non_null.dt.day.eq(1).all():
        rows_per_month = non_null.dt.to_period("M").value_counts()
        if not rows_per_month.empty and int(rows_per_month.max()) == 1:
            return "month"

    return "day"


def normalize_period_series_with_granularity(series: pd.Series) -> Tuple[pd.Series, Optional[str]]:
    """Coerce year-like and date-like series into timestamps plus inferred granularity."""

    if series is None:
        return series, None

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        non_null = numeric.dropna()
        if not non_null.empty and non_null.between(1900, 2100).all():
            year_strings = numeric.round().astype("Int64").astype(str)
            return pd.to_datetime(year_strings, format="%Y", errors="coerce"), "year"

    text = series.astype(str).str.strip()
    cleaned = text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    non_null_text = cleaned.dropna()
    if not non_null_text.empty and non_null_text.str.fullmatch(r"(19|20)\d{2}").all():
        return pd.to_datetime(cleaned, format="%Y", errors="coerce"), "year"

    timestamps = pd.to_datetime(series, errors="coerce")
    return timestamps, infer_period_granularity(timestamps)


def normalize_period_series(series: pd.Series) -> pd.Series:
    """Coerce year-like and date-like series into timestamps."""

    normalized, _granularity = normalize_period_series_with_granularity(series)
    return normalized


def canonicalize_generation_mix_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot raw generation-mix rows into one analysis-ready row per period.

    The raw tool output is long-form: one ``type_tech`` row per period.
    For analytics, correlation, and grounding we need one row per period with
    canonical demand/supply aggregates.
    """

    if df is None or df.empty or "type_tech" not in df.columns:
        return df

    period_col = next((col for col in ("period", "date", "month", "year") if col in df.columns), None)
    if not period_col:
        return df

    working = df.copy()
    working["type_tech"] = working["type_tech"].astype(str).str.strip().str.lower()
    working = working[working["type_tech"] != ""]
    working[period_col] = normalize_period_series(working[period_col])
    working = working.dropna(subset=[period_col])
    if working.empty:
        return df

    blocks: list[pd.DataFrame] = []
    if "quantity_tech" in working.columns:
        quantity = (
            working.pivot_table(
                index=period_col,
                columns="type_tech",
                values="quantity_tech",
                aggfunc="sum",
            )
            .sort_index(axis=1)
        )
        quantity.columns = [f"quantity_{col}" for col in quantity.columns]
        blocks.append(quantity)

    has_raw_share_tech = "share_tech" in working.columns
    if has_raw_share_tech:
        share = (
            working.pivot_table(
                index=period_col,
                columns="type_tech",
                values="share_tech",
                aggfunc="sum",
            )
            .sort_index(axis=1)
        )
        share.columns = [f"share_{col}" for col in share.columns]
        blocks.append(share)

    if not blocks:
        return df

    result = pd.concat(blocks, axis=1).sort_index()

    quantity_cols = [col for col in result.columns if col.startswith("quantity_")]
    computed_share_cols: set[str] = set()
    if quantity_cols:
        supply_quantity_cols = [f"quantity_{tech}" for tech in SUPPLY_TECH_TYPES if f"quantity_{tech}" in result.columns]
        demand_quantity_cols = [f"quantity_{tech}" for tech in DEMAND_TECH_TYPES if f"quantity_{tech}" in result.columns]
        has_mixed_market_sides = bool(supply_quantity_cols and demand_quantity_cols)
        overwrite_raw_mixed_shares = has_raw_share_tech and has_mixed_market_sides

        def _assign_side_shares(tech_types: Iterable[str]) -> None:
            cols = [f"quantity_{tech}" for tech in tech_types if f"quantity_{tech}" in result.columns]
            if not cols:
                return
            denominator = result[cols].sum(axis=1, min_count=1).replace(0, pd.NA)
            for col in cols:
                tech = col[len("quantity_"):]
                share_col = f"share_{tech}"
                if share_col not in result.columns or overwrite_raw_mixed_shares:
                    result[share_col] = result[col] / denominator
                    computed_share_cols.add(share_col)

        _assign_side_shares(SUPPLY_TECH_TYPES)
        _assign_side_shares(DEMAND_TECH_TYPES)

        known_side_techs = set(SUPPLY_TECH_TYPES) | set(DEMAND_TECH_TYPES)
        remaining_quantity_cols = [
            col for col in quantity_cols
            if col[len("quantity_"):] not in known_side_techs
        ]
        if remaining_quantity_cols:
            total_remaining = result[remaining_quantity_cols].sum(axis=1, min_count=1).replace(0, pd.NA)
            for col in remaining_quantity_cols:
                tech = col[len("quantity_"):]
                share_col = f"share_{tech}"
                if share_col not in result.columns:
                    result[share_col] = result[col] / total_remaining
                    computed_share_cols.add(share_col)

    # Coverage guard: NA any generation-tech share for periods that observe
    # fewer than _MIN_GENERATION_TECHS_FOR_SHARE generation techs, so an
    # incomplete-denominator "100%" is reported as unknown, not asserted as
    # fact. See the _MIN_GENERATION_TECHS_FOR_SHARE note above.
    gen_quantity_cols = [
        f"quantity_{tech}" for tech in _DOMESTIC_GENERATION_TYPES
        if f"quantity_{tech}" in result.columns
    ]
    gen_share_cols = [
        f"share_{tech}" for tech in _DOMESTIC_GENERATION_TYPES
        if f"share_{tech}" in result.columns
    ]
    guarded_gen_share_cols = [col for col in gen_share_cols if col in computed_share_cols]
    if gen_quantity_cols and guarded_gen_share_cols:
        observed_gen_techs = result[gen_quantity_cols].notna().sum(axis=1)
        incomplete = observed_gen_techs < _MIN_GENERATION_TECHS_FOR_SHARE
        if incomplete.any():
            result.loc[incomplete, guarded_gen_share_cols] = pd.NA

    def _sum_columns(tech_types: Iterable[str], output_col: str) -> None:
        cols = [f"quantity_{tech}" for tech in tech_types if f"quantity_{tech}" in result.columns]
        if cols:
            result[output_col] = result[cols].sum(axis=1, min_count=1)

    _sum_columns(DEMAND_TECH_TYPES, "total_demand")
    _sum_columns(_DOMESTIC_GENERATION_TYPES, "total_domestic_generation")
    _sum_columns(_LOCAL_GENERATION_TYPES, "local_generation")
    _sum_columns(_IMPORT_DEPENDENT_TYPES, "import_dependent_supply")
    _sum_columns(SUPPLY_TECH_TYPES, "total_supply")

    if "import_dependent_supply" in result.columns and "total_supply" in result.columns:
        denom = result["total_supply"].replace(0, pd.NA)
        result["import_dependency_ratio"] = result["import_dependent_supply"] / denom

    result.index.name = "period"
    return result.reset_index()


def coerce_decimal_columns_to_float(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Coerce object-dtype columns that hold ``Decimal`` values to float64.

    Root-cause fix (2026-05-17) — Q2 production trace 5b117786. PostgreSQL
    ``numeric`` columns arrive in pandas as Python ``Decimal`` objects with
    ``object`` dtype. ``pd.api.types.is_numeric_dtype`` returns False for
    object dtype, so downstream code that relies on ``select_dtypes(include=
    "number")`` (``_append_column_aggregates`` in ``agent/analyzer.py``,
    the numeric-summary branch of ``quick_stats`` in ``analysis/stats.py``)
    silently skips every column. Result: ``stats_hint`` is just "Rows: N"
    and the LLM, given no pre-computed statistics, hallucinates plausible
    averages from the raw 12-row preview. Those hallucinated numbers don't
    exist in any data corpus, so the grounding gate correctly rejects them
    and the conservative fallback replaces the substantive answer.

    Returns the (possibly-mutated) DataFrame and the list of column names
    that were coerced, for caller logging/observability.
    """
    if df is None or df.empty:
        return df, []
    converted: List[str] = []
    for col in df.columns:
        if df[col].dtype != object:
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        # Type-test the first non-null sample. A column is "Decimal-typed"
        # only if its sample is a Decimal; mixed/string columns are left alone.
        sample = non_null.iloc[0]
        if not isinstance(sample, Decimal):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted.append(col)
        except Exception:
            # Defensive: keep column unchanged on any conversion error.
            continue
    return df, converted


def normalize_tool_dataframe(tool_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply tool-specific normalization before analytics consume a result frame.

    Always runs the Decimal→float64 coercion so that PostgreSQL ``numeric``
    columns become first-class numeric dtypes for ``select_dtypes(include=
    "number")`` consumers (per-column aggregates, grouping logic, chart
    builders, grounding-token extraction).
    """

    if df is None or df.empty:
        return df
    df, _converted = coerce_decimal_columns_to_float(df)
    if tool_name == "get_generation_mix":
        return canonicalize_generation_mix_df(df)
    return df
