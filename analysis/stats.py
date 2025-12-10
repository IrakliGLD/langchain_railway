"""
Statistical analysis and data preview generation.

Handles:
- Quick statistics generation for query results
- Trend analysis (yearly, seasonal)
- CAGR calculations for summer/winter periods
- Data preview formatting
"""
import logging
from typing import List, Tuple

import pandas as pd
import numpy as np

log = logging.getLogger("Enai")


def rows_to_preview(rows: List[Tuple], cols: List[str], max_rows: int = 200) -> str:
    """
    Convert query results to formatted string preview.

    Args:
        rows: List of tuples containing query results
        cols: List of column names
        max_rows: Maximum number of rows to include in preview

    Returns:
        Formatted string representation of data

    Examples:
        >>> rows = [(1, 'A', 10.5), (2, 'B', 20.3)]
        >>> cols = ['id', 'name', 'value']
        >>> print(rows_to_preview(rows, cols))
        id name  value
         1    A  10.5
         2    B  20.3
    """
    if not rows:
        return "No rows returned."

    df = pd.DataFrame(rows[:max_rows], columns=cols)

    # Round numeric columns to 3 decimal places
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float).round(3)

    return df.to_string(index=False)


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
    numeric = df.select_dtypes(include=[np.number])
    out = [f"Rows: {len(df)}"]

    # 1. Detect date/year column
    date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower() or "month" in c.lower()]
    if not date_cols or numeric.empty:
        # Fallback to simple stats if no date or numeric data
        return "\n".join(out)

    time_col = date_cols[0]

    # --- Trend Calculation: Compare First Full Year vs Last Full Year ---
    try:
        # Ensure the time column is datetime, then extract year/month
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            # Attempt to coerce strings/objects to datetime
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        # Verify conversion worked before using .dt accessors
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            # Conversion failed, still object dtype - skip trend calculation
            log.warning(f"⚠️ Column {time_col} could not be converted to datetime, skipping trend")
            return "\n".join(out)

        df['__year'] = df[time_col].dt.year

        # --- Detect and exclude incomplete final year ---
        # Applies to all time columns, even if already datetime
        try:
            df['_year_month'] = df[time_col].dt.to_period('M')
            # Count how many records (months) exist for each year
            months_per_year = df.groupby(df['_year_month'].dt.year).size().sort_index()
            # Mark as incomplete if less than 10 months (not a full year of data)
            incomplete_years = months_per_year[months_per_year < 10].index.tolist()
            if incomplete_years:
                log.info(f"🧩 Excluding incomplete years from trend calculation: {incomplete_years}")
                df = df[~df['__year'].isin(incomplete_years)]
        except Exception as e:
            log.warning(f"⚠️ Failed to filter incomplete years: {e}")

        valid_years = df['__year'].dropna().unique()
        if len(valid_years) >= 2:
            first_full_year = int(valid_years.min())
            last_full_year = int(valid_years.max())

            # Ensure we are comparing two different years
            if first_full_year != last_full_year:

                # Filter data for the first and last full years
                df_first = df[df['__year'] == first_full_year]
                df_last = df[df['__year'] == last_full_year]

                # Get the mean of all numeric values for these years
                # Using .values.mean() to get single average across all values
                mean_first_year = df_first[numeric.columns].values.mean()
                mean_last_year = df_last[numeric.columns].values.mean()

                change = ((mean_last_year - mean_first_year) / mean_first_year * 100) if mean_first_year != 0 else 0
                trend = "increasing" if mean_last_year > mean_first_year else "decreasing"
                out.append(f"Trend (Yearly Avg, {first_full_year}→{last_full_year}): {trend} ({change:.1f}%)")

                # --- Seasonal split (Summer vs Winter) with CAGR ---
                try:
                    df['month'] = df[time_col].dt.month
                    summer_mask = df['month'].isin([4, 5, 6, 7])
                    winter_mask = ~summer_mask

                    def seasonal_avg(df_season, col, year):
                        """Calculate seasonal average for a specific year."""
                        return df_season.loc[df_season['__year'] == year, col].mean()

                    def seasonal_cagr(df_season, col):
                        """Compute CAGR (Compound Annual Growth Rate) for a column across years within a seasonal subset."""
                        df_y = df_season.groupby('__year')[col].mean().dropna()
                        if len(df_y) >= 2:
                            first, last = df_y.iloc[0], df_y.iloc[-1]
                            n = len(df_y) - 1
                            return ((last / first) ** (1 / n) - 1) * 100 if first > 0 else np.nan
                        return np.nan

                    for col in numeric.columns:
                        if 'p_bal' in col.lower() or 'price' in col.lower():
                            summer_first = seasonal_avg(df.loc[summer_mask], col, first_full_year)
                            summer_last = seasonal_avg(df.loc[summer_mask], col, last_full_year)
                            winter_first = seasonal_avg(df.loc[winter_mask], col, first_full_year)
                            winter_last = seasonal_avg(df.loc[winter_mask], col, last_full_year)

                            cagr_summer = seasonal_cagr(df.loc[summer_mask], col)
                            cagr_winter = seasonal_cagr(df.loc[winter_mask], col)

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
    out.append(f"Period: {first} → {last}")

    # Numeric summary
    if not numeric.empty:
        desc = numeric.describe().round(3)
        out.append("Numeric summary:")
        out.append(desc.to_string())

    return "\n".join(out)
