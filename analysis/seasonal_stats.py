"""
Seasonal-adjusted statistics for time series analysis.

Handles:
- Year-over-year growth calculations
- Yearly totals with incomplete year detection
- Seasonal pattern analysis
- Trend extraction accounting for seasonality
"""
import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

log = logging.getLogger("Enai")


def detect_monthly_timeseries(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Detect if DataFrame contains monthly time series data.

    Args:
        df: DataFrame to analyze

    Returns:
        Tuple of (time_column, value_column) if monthly series detected, None otherwise

    Examples:
        >>> df = pd.DataFrame({'month': ['2023-01', '2023-02'], 'demand': [100, 120]})
        >>> detect_monthly_timeseries(df)
        ('month', 'demand')
    """
    # Look for time column (YYYY-MM format)
    time_cols = [c for c in df.columns if 'month' in c.lower() or 'date' in c.lower() or 'year' in c.lower()]

    if not time_cols:
        return None

    time_col = time_cols[0]

    # Check if values look like YYYY-MM or dates
    sample_val = str(df[time_col].iloc[0])
    if not (len(sample_val) >= 7 and '-' in sample_val):
        return None

    # Look for numeric value column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return None

    value_col = numeric_cols[0]  # Take first numeric column

    log.info(f"📅 Detected monthly time series: time={time_col}, value={value_col}")
    return (time_col, value_col)


def calculate_seasonal_stats(
    df: pd.DataFrame,
    time_col: str,
    value_col: str
) -> Dict[str, any]:
    """
    Calculate seasonal-adjusted statistics for monthly time series.

    Handles:
    - Yearly totals (with incomplete year detection)
    - Year-over-year growth rates
    - Seasonal patterns
    - Trend analysis accounting for seasonality

    Args:
        df: DataFrame with monthly time series
        time_col: Name of time column (YYYY-MM format)
        value_col: Name of value column

    Returns:
        Dictionary with seasonal statistics

    Examples:
        >>> df = pd.DataFrame({
        ...     'month': ['2023-01', '2023-02', '2024-01'],
        ...     'demand': [100, 110, 120]
        ... })
        >>> stats = calculate_seasonal_stats(df, 'month', 'demand')
        >>> stats['yoy_growth_pct']
        20.0
    """
    stats = {}

    # Parse year and month
    df = df.copy()
    df['_year'] = df[time_col].astype(str).str[:4].astype(int)
    df['_month'] = df[time_col].astype(str).str[5:7].astype(int)

    # 1. Yearly totals
    yearly_totals = df.groupby('_year')[value_col].sum().sort_index()

    # 2. Detect incomplete years
    months_per_year = df.groupby('_year').size()
    last_year = yearly_totals.index[-1]
    last_year_months = months_per_year[last_year]

    is_incomplete_last_year = last_year_months < 12
    stats['incomplete_last_year'] = is_incomplete_last_year
    stats['last_year_months'] = int(last_year_months)

    # 3. Calculate yearly totals (excluding incomplete last year for trends)
    complete_years = yearly_totals.iloc[:-1] if is_incomplete_last_year else yearly_totals

    if len(complete_years) >= 2:
        first_year_total = complete_years.iloc[0]
        last_year_total = complete_years.iloc[-1]
        years_span = len(complete_years) - 1

        # Overall growth (complete years only)
        overall_growth_pct = ((last_year_total - first_year_total) / first_year_total * 100) if first_year_total > 0 else 0
        stats['overall_growth_pct'] = round(overall_growth_pct, 1)
        stats['first_year'] = int(complete_years.index[0])
        stats['last_year'] = int(complete_years.index[-1])
        stats['first_year_total'] = round(first_year_total, 1)
        stats['last_year_total'] = round(last_year_total, 1)
        stats['years_span'] = years_span

        # Average annual growth rate (CAGR)
        if years_span > 0 and first_year_total > 0:
            cagr = (pow(last_year_total / first_year_total, 1 / years_span) - 1) * 100
            stats['cagr'] = round(cagr, 1)

    # 4. Year-over-year growth (same months comparison)
    # Compare each month to the same month in previous year
    yoy_growth_rates = []

    for year in df['_year'].unique()[1:]:  # Skip first year (no previous year)
        for month in df[df['_year'] == year]['_month'].unique():
            current = df[(df['_year'] == year) & (df['_month'] == month)][value_col]
            previous = df[(df['_year'] == year - 1) & (df['_month'] == month)][value_col]

            if len(current) > 0 and len(previous) > 0:
                current_val = current.iloc[0]
                previous_val = previous.iloc[0]

                if previous_val > 0:
                    yoy_growth = ((current_val - previous_val) / previous_val) * 100
                    yoy_growth_rates.append(yoy_growth)

    if yoy_growth_rates:
        stats['yoy_growth_avg'] = round(np.mean(yoy_growth_rates), 1)
        stats['yoy_growth_median'] = round(np.median(yoy_growth_rates), 1)
        stats['yoy_growth_std'] = round(np.std(yoy_growth_rates), 1)

    # 5. Seasonal pattern (average value per month across all years)
    monthly_avg = df.groupby('_month')[value_col].mean().sort_index()

    if len(monthly_avg) >= 12:
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        seasonality_range = monthly_avg.max() - monthly_avg.min()
        seasonality_pct = (seasonality_range / monthly_avg.mean()) * 100 if monthly_avg.mean() > 0 else 0

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        stats['peak_month'] = month_names[peak_month - 1] if 1 <= peak_month <= 12 else str(peak_month)
        stats['low_month'] = month_names[low_month - 1] if 1 <= low_month <= 12 else str(low_month)
        stats['seasonality_pct'] = round(seasonality_pct, 1)

    # 6. Recent trend (last 12 months vs previous 12 months, if available)
    if len(df) >= 24:
        recent_12 = df.tail(12)[value_col].sum()
        previous_12 = df.iloc[-24:-12][value_col].sum()

        if previous_12 > 0:
            recent_trend_pct = ((recent_12 - previous_12) / previous_12) * 100
            stats['recent_12m_growth'] = round(recent_trend_pct, 1)

    log.info(f"📊 Seasonal statistics calculated: {len(stats)} metrics")
    return stats


def format_seasonal_stats(stats: Dict[str, any]) -> str:
    """
    Format seasonal statistics into human-readable text for LLM.

    Args:
        stats: Dictionary of seasonal statistics

    Returns:
        Formatted string for inclusion in stats_hint

    Examples:
        >>> stats = {'overall_growth_pct': 25.5, 'first_year': 2020, 'last_year': 2023}
        >>> text = format_seasonal_stats(stats)
        >>> 'Overall growth: +25.5%' in text
        True
    """
    lines = []

    lines.append("SEASONAL-ADJUSTED TREND ANALYSIS:")

    # Overall trend
    if 'overall_growth_pct' in stats:
        lines.append(f"- Overall growth ({stats['first_year']}-{stats['last_year']}): {stats['overall_growth_pct']:+.1f}% "
                    f"({stats['first_year_total']:.0f} → {stats['last_year_total']:.0f})")

    if 'cagr' in stats:
        lines.append(f"- Average annual growth rate (CAGR): {stats['cagr']:+.1f}%/year")

    # Year-over-year
    if 'yoy_growth_avg' in stats:
        lines.append(f"- Year-over-year growth (same months): avg {stats['yoy_growth_avg']:+.1f}%, "
                    f"median {stats['yoy_growth_median']:+.1f}%")

    # Recent trend
    if 'recent_12m_growth' in stats:
        lines.append(f"- Recent 12-month trend: {stats['recent_12m_growth']:+.1f}%")

    # Seasonality
    if 'peak_month' in stats:
        lines.append(f"- Seasonal pattern: Peak in {stats['peak_month']}, Low in {stats['low_month']} "
                    f"(range: {stats['seasonality_pct']:.0f}% of average)")

    # Incomplete year warning
    if stats.get('incomplete_last_year', False):
        lines.append(f"- ⚠️ IMPORTANT: Last year has only {stats['last_year_months']} months of data "
                    f"(incomplete year - excluded from trend calculations)")

    return "\n".join(lines)
