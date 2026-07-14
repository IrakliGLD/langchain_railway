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

import numpy as np
import pandas as pd

from analysis.system_quantities import normalize_period_series_with_granularity

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
    if df is None or df.empty:
        return None

    # Look for time column (YYYY-MM format)
    time_cols = [c for c in df.columns if 'month' in c.lower() or 'date' in c.lower() or 'year' in c.lower()]

    if not time_cols:
        return None

    time_col = time_cols[0]

    # Check if values look like YYYY-MM or dates
    non_null_times = df[time_col].dropna()
    if non_null_times.empty:
        return None
    sample_val = str(non_null_times.iloc[0])
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

    # Intensive metrics (prices/rates) must be AVERAGED per year, not summed —
    # summing monthly prices inflates the level 12× (the 1,482-GEL/MWh bug).
    # Extensive metrics (quantities) sum to a meaningful annual total.
    from analysis.stats import is_intensive_metric
    intensive = is_intensive_metric(value_col)
    agg = "mean" if intensive else "sum"
    stats['aggregate_kind'] = "average" if intensive else "total"

    # Parse, sort, and collapse to one analytical value per unique period.
    # Exact duplicate source rows are removed first; multiple entity rows in an
    # extensive series are then summed, while intensive levels are averaged.
    df = df.copy().drop_duplicates()
    df[time_col], granularity = normalize_period_series_with_granularity(df[time_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[time_col, value_col]).sort_values(time_col)
    if df.empty:
        return stats

    period_freq = "Y" if granularity == "year" else "M"
    df['_period'] = df[time_col].dt.to_period(period_freq)
    period_values = df.groupby('_period', sort=True)[value_col].agg(agg)
    series = period_values.rename(value_col).reset_index()
    series['_date'] = series['_period'].dt.to_timestamp()
    series['_year'] = series['_date'].dt.year
    series['_month'] = series['_date'].dt.month

    # 1. Yearly values (mean for intensive prices, sum for extensive quantities)
    yearly_totals = series.groupby('_year')[value_col].agg(agg).sort_index()

    # 2. Detect incomplete/missing periods using unique calendar periods, not rows.
    months_per_year = series.groupby('_year')['_period'].nunique().sort_index()
    last_year = int(yearly_totals.index[-1])
    last_year_months = int(months_per_year[last_year])

    expected_per_year = 1 if granularity == "year" else 12
    is_incomplete_last_year = last_year_months < expected_per_year
    stats['incomplete_last_year'] = is_incomplete_last_year
    stats['last_year_months'] = last_year_months

    if granularity == "year":
        expected_periods = pd.period_range(series['_period'].min(), series['_period'].max(), freq="Y")
    else:
        first_expected = pd.Period(f"{int(series['_year'].min())}-01", freq="M")
        last_expected = pd.Period(f"{int(series['_year'].max())}-12", freq="M")
        expected_periods = pd.period_range(first_expected, last_expected, freq="M")
    observed_periods = set(series['_period'].tolist())
    missing_periods = [str(period) for period in expected_periods if period not in observed_periods]
    stats['missing_period_count'] = len(missing_periods)
    stats['missing_periods'] = missing_periods

    complete_year_ids = months_per_year[months_per_year >= expected_per_year].index
    complete_years = yearly_totals.loc[yearly_totals.index.isin(complete_year_ids)]

    if len(complete_years) >= 2:
        first_year_total = complete_years.iloc[0]
        last_year_total = complete_years.iloc[-1]
        first_year = int(complete_years.index[0])
        last_complete_year = int(complete_years.index[-1])
        years_span = last_complete_year - first_year

        # Overall growth (complete years only)
        overall_growth_pct = ((last_year_total - first_year_total) / first_year_total * 100) if first_year_total > 0 else 0
        stats['overall_growth_pct'] = round(overall_growth_pct, 1)
        stats['first_year'] = first_year
        stats['last_year'] = last_complete_year
        stats['first_year_total'] = round(first_year_total, 1)
        stats['last_year_total'] = round(last_year_total, 1)
        stats['years_span'] = years_span
        observed_years = set(int(year) for year in yearly_totals.index)
        stats['missing_years'] = [
            year for year in range(first_year, last_complete_year + 1)
            if year not in observed_years
        ]

        # Average annual growth rate (CAGR)
        if years_span > 0 and first_year_total > 0:
            cagr = (pow(last_year_total / first_year_total, 1 / years_span) - 1) * 100
            stats['cagr'] = round(cagr, 1)

    # Compare each observed month only against the same month in the prior year.
    yoy_growth_rates = []

    for year in sorted(series['_year'].unique())[1:]:  # Skip first year (no previous year)
        for month in sorted(series[series['_year'] == year]['_month'].unique()):
            current = series[(series['_year'] == year) & (series['_month'] == month)][value_col]
            previous = series[(series['_year'] == year - 1) & (series['_month'] == month)][value_col]

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

    # Summarize the recurring seasonal shape by averaging each calendar month.
    monthly_avg = series.groupby('_month')[value_col].mean().sort_index()

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
    if granularity != "year" and len(series) >= 24:
        recent_12 = series.tail(12)[value_col].agg(agg)
        previous_12 = series.iloc[-24:-12][value_col].agg(agg)

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

    # Overall trend. Label the endpoints by aggregate kind so an averaged price
    # level (e.g. 123 → 158) is never misread as an annual total.
    if 'overall_growth_pct' in stats:
        _level_word = "avg" if stats.get('aggregate_kind') == "average" else "annual total"
        lines.append(f"- Overall growth ({_level_word}, {stats['first_year']}-{stats['last_year']}): "
                    f"{stats['overall_growth_pct']:+.1f}% "
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

    if stats.get('missing_period_count', 0):
        sample = ", ".join(stats.get('missing_periods', [])[:6])
        suffix = "..." if stats['missing_period_count'] > 6 else ""
        lines.append(
            f"- ⚠️ Missing calendar periods: {stats['missing_period_count']} "
            f"({sample}{suffix}); growth uses complete observed years only."
        )

    if stats.get('missing_years'):
        lines.append(
            "- ⚠️ Missing calendar years inside the CAGR span: "
            + ", ".join(str(year) for year in stats['missing_years'])
            + ". CAGR uses actual elapsed calendar years."
        )

    return "\n".join(lines)
