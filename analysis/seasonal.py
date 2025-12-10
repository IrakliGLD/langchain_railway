"""
Seasonal analysis for energy market data.

Handles:
- Summer vs Winter seasonal aggregation
- Seasonal averages and sums
- CAGR (Compound Annual Growth Rate) calculations

Note: Summer is defined as months [4,5,6,7] (April-July),
which is specific to Georgian hydro generation patterns.
"""
import logging
from typing import Literal

import pandas as pd
import numpy as np

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


def compute_seasonal_cagr(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    season: Literal["summer", "winter"]
) -> float:
    """
    Compute CAGR (Compound Annual Growth Rate) for a specific season across years.

    Args:
        df: DataFrame with date and value columns
        date_col: Name of the date column
        value_col: Name of the value column
        season: "summer" or "winter"

    Returns:
        CAGR as percentage (e.g., 5.2 means 5.2% annual growth)
        Returns np.nan if insufficient data

    Examples:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=48, freq='M'),
        ...     'price': range(100, 148)
        ... })
        >>> cagr = compute_seasonal_cagr(df, 'date', 'price', 'summer')
        >>> print(f"Summer CAGR: {cagr:.2f}%")
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month

    # Filter by season
    summer_months = [4, 5, 6, 7]
    if season.lower() == "summer":
        df_season = df[df['month'].isin(summer_months)]
    else:
        df_season = df[~df['month'].isin(summer_months)]

    # Group by year and calculate mean
    yearly_avg = df_season.groupby('year')[value_col].mean().dropna()

    if len(yearly_avg) < 2:
        log.warning(f"⚠️ Insufficient data for {season} CAGR calculation (need ≥2 years)")
        return np.nan

    first_value = yearly_avg.iloc[0]
    last_value = yearly_avg.iloc[-1]
    n_years = len(yearly_avg) - 1

    if first_value <= 0:
        log.warning(f"⚠️ Cannot calculate CAGR: first value is {first_value} (must be > 0)")
        return np.nan

    # CAGR formula: ((end_value / start_value) ^ (1 / n_years)) - 1
    cagr = ((last_value / first_value) ** (1 / n_years) - 1) * 100

    return cagr


def compute_seasonal_comparison(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    first_year: int,
    last_year: int
) -> dict:
    """
    Compare seasonal metrics between first and last year.

    Returns summer/winter averages, changes, and CAGRs.

    Args:
        df: DataFrame with date and value columns
        date_col: Name of the date column
        value_col: Name of the value column to analyze
        first_year: Starting year for comparison
        last_year: Ending year for comparison

    Returns:
        Dictionary with seasonal metrics:
        {
            'summer_first': float,
            'summer_last': float,
            'summer_change': float,
            'summer_cagr': float,
            'winter_first': float,
            'winter_last': float,
            'winter_change': float,
            'winter_cagr': float
        }

    Examples:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=60, freq='M'),
        ...     'price': np.linspace(100, 150, 60)
        ... })
        >>> comparison = compute_seasonal_comparison(df, 'date', 'price', 2020, 2024)
        >>> print(f"Summer change: {comparison['summer_change']:.1f}")
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month

    # Define seasons
    summer_months = [4, 5, 6, 7]
    df['is_summer'] = df['month'].isin(summer_months)

    def seasonal_avg(year: int, is_summer: bool) -> float:
        """Get average for specific year and season."""
        mask = (df['year'] == year) & (df['is_summer'] == is_summer)
        return df.loc[mask, value_col].mean()

    # Calculate metrics
    summer_first = seasonal_avg(first_year, True)
    summer_last = seasonal_avg(last_year, True)
    winter_first = seasonal_avg(first_year, False)
    winter_last = seasonal_avg(last_year, False)

    summer_cagr = compute_seasonal_cagr(df, date_col, value_col, "summer")
    winter_cagr = compute_seasonal_cagr(df, date_col, value_col, "winter")

    return {
        'summer_first': summer_first,
        'summer_last': summer_last,
        'summer_change': summer_last - summer_first,
        'summer_cagr': summer_cagr,
        'winter_first': winter_first,
        'winter_last': winter_last,
        'winter_change': winter_last - winter_first,
        'winter_cagr': winter_cagr
    }
