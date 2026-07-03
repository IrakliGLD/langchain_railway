"""
Seasonal analysis for energy market data.

Handles:
- Summer vs Winter seasonal aggregation
- Seasonal averages and sums

Note: Summer is defined as months [4,5,6,7] (April-July),
which is specific to Georgian hydro generation patterns.
"""
import logging
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger("Enai")


def compute_seasonal_average(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    agg_func: Literal["avg", "mean", "sum"] = "avg"
) -> pd.DataFrame:
    """
    Compute seasonal (Summer vs Winter) average or sum for a given value column.

    Summer months: [4, 5, 6, 7] (April-July) - High hydro generation period
    Winter months: [1, 2, 3, 8, 9, 10, 11, 12] - Lower hydro, higher thermal/import period

    Args:
        df: DataFrame with date and value columns
        date_col: Name of the date column (will be converted to datetime)
        value_col: Name of the value column to aggregate
        agg_func: Aggregation function - "avg"/"mean" or "sum"

    Returns:
        DataFrame with two rows (Summer, Winter) and aggregated values

    Raises:
        ValueError: If agg_func is not 'avg', 'mean', or 'sum'

    Examples:
        >>> df = pd.DataFrame({
        ...     'date': ['2023-04-01', '2023-08-01', '2024-04-01'],
        ...     'price': [100, 150, 110]
        ... })
        >>> result = compute_seasonal_average(df, 'date', 'price', 'avg')
        >>> print(result)
          season  avg_price
        0  Summer      105.0
        1  Winter      150.0
    """
    # Validate columns exist
    if date_col not in df.columns or value_col not in df.columns:
        log.warning(f"⚠️ Missing columns: {date_col} or {value_col} not in DataFrame")
        return df

    df = df.copy()

    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Define seasons (vectorized operation - faster than .apply())
    summer_months = [4, 5, 6, 7]
    df["season"] = np.where(
        df[date_col].dt.month.isin(summer_months),
        "Summer",
        "Winter"
    )

    # Aggregate by season
    if agg_func.lower() in ("avg", "mean"):
        grouped = df.groupby("season")[value_col].mean().reset_index(name=f"avg_{value_col}")
    elif agg_func.lower() == "sum":
        grouped = df.groupby("season")[value_col].sum().reset_index(name=f"sum_{value_col}")
    else:
        raise ValueError("agg_func must be 'avg', 'mean', or 'sum'")

    return grouped
