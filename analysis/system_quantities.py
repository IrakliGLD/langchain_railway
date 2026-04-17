"""Helpers for normalizing system quantity datasets into analysis-ready frames."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import pandas as pd

from context import DEMAND_TECH_TYPES

_DOMESTIC_GENERATION_TYPES = ("hydro", "thermal", "wind", "solar")
_LOCAL_GENERATION_TYPES = ("hydro", "wind", "solar")
_IMPORT_DEPENDENT_TYPES = ("thermal", "import")


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

    if "share_tech" in working.columns:
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
    if quantity_cols:
        total_observed = result[quantity_cols].sum(axis=1).replace(0, pd.NA)
        for col in quantity_cols:
            tech = col[len("quantity_"):]
            share_col = f"share_{tech}"
            if share_col not in result.columns:
                result[share_col] = result[col] / total_observed

    def _sum_columns(tech_types: Iterable[str], output_col: str) -> None:
        cols = [f"quantity_{tech}" for tech in tech_types if f"quantity_{tech}" in result.columns]
        if cols:
            result[output_col] = result[cols].sum(axis=1)

    _sum_columns(DEMAND_TECH_TYPES, "total_demand")
    _sum_columns(_DOMESTIC_GENERATION_TYPES, "total_domestic_generation")
    _sum_columns(_LOCAL_GENERATION_TYPES, "local_generation")
    _sum_columns(_IMPORT_DEPENDENT_TYPES, "import_dependent_supply")

    if "total_domestic_generation" in result.columns and "quantity_import" in result.columns:
        result["total_supply"] = result["total_domestic_generation"] + result["quantity_import"]
    if "import_dependent_supply" in result.columns and "total_demand" in result.columns:
        denom = result["total_demand"].replace(0, pd.NA)
        result["import_dependency_ratio"] = result["import_dependent_supply"] / denom

    result.index.name = "period"
    return result.reset_index()


def normalize_tool_dataframe(tool_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply tool-specific normalization before analytics consume a result frame."""

    if df is None or df.empty:
        return df
    if tool_name == "get_generation_mix":
        return canonicalize_generation_mix_df(df)
    return df
