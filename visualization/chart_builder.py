"""
Chart data building and formatting.

Handles:
- Series filtering based on query relevance
- Label mapping from context
- Chart metadata generation
- Axis title determination
- Trendline calculation and projection
"""
import logging
from typing import Dict, List, Set, Tuple, Optional
import re

import pandas as pd
import numpy as np
from scipy import stats

from visualization.chart_selector import (
    infer_dimension,
    unit_for_price,
    unit_for_qty,
    unit_for_index,
    unit_for_xrate,
    unit_for_share,
    determine_axis_mode
)

log = logging.getLogger("Enai")


# -----------------------------
# Dimension Safety Check (Option 2)
# -----------------------------

def detect_mixed_dimensions(dimensions: Set[str]) -> Tuple[bool, str]:
    """
    Detect if dimensions are mixed and incompatible for a single chart.

    This is a safety net to catch cases where LLM didn't properly separate dimensions.

    Incompatible dimension pairs:
    - 'price_tariff' + 'share' (GEL/MWh vs %)
    - 'price_tariff' + 'energy_qty' (GEL/MWh vs MWh)
    - 'price_tariff' + 'xrate' (GEL/MWh vs GEL/USD)
    - 'share' + 'energy_qty' (% vs MWh)
    - 'share' + 'xrate' (% vs GEL/USD)
    - 'xrate' + 'energy_qty' (GEL/USD vs MWh)
    - Any dimension + 'index' (CPI is separate)

    Args:
        dimensions: Set of semantic dimensions from columns

    Returns:
        Tuple of (is_mixed, reason)
        - is_mixed: True if dimensions should not be on same chart
        - reason: Human-readable explanation

    Examples:
        >>> detect_mixed_dimensions({'price_tariff', 'share'})
        (True, 'Cannot mix price (GEL/MWh) with shares (%)')

        >>> detect_mixed_dimensions({'price_tariff'})
        (False, 'Single dimension')

        >>> detect_mixed_dimensions({'share'})
        (False, 'Single dimension')
    """
    if len(dimensions) == 1:
        return False, "Single dimension"

    # Define incompatible pairs
    incompatible_pairs = [
        ({'price_tariff', 'share'}, 'Cannot mix price (GEL/MWh) with shares (%)'),
        ({'price_tariff', 'energy_qty'}, 'Cannot mix price (GEL/MWh) with quantities (MWh)'),
        ({'price_tariff', 'xrate'}, 'Cannot mix price (GEL/MWh) with exchange rate (GEL/USD)'),
        ({'share', 'energy_qty'}, 'Cannot mix shares (%) with quantities (MWh)'),
        ({'share', 'xrate'}, 'Cannot mix shares (%) with exchange rate (GEL/USD)'),
        ({'xrate', 'energy_qty'}, 'Cannot mix exchange rate (GEL/USD) with quantities (MWh)'),
    ]

    # Check each incompatible pair
    for pair, reason in incompatible_pairs:
        if pair.issubset(dimensions):
            return True, reason

    # Special case: index should always be separate
    if 'index' in dimensions and len(dimensions) > 1:
        return True, 'CPI/index must be on separate chart from other metrics'

    return False, "Dimensions are compatible"


def calculate_trendline(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    extend_to_date: Optional[str] = None
) -> Optional[Dict]:
    """
    Calculate linear trendline for time series data.

    Args:
        df: DataFrame with time series data
        date_col: Name of date column
        value_col: Name of value column to trend
        extend_to_date: Optional future date to extend trendline (e.g., "2035-12-01")

    Returns:
        Dictionary with trendline data:
        {
            "dates": List of dates (including extended dates if requested),
            "values": List of trendline values,
            "equation": "y = mx + b",
            "r_squared": R² coefficient,
            "slope": slope value,
            "intercept": intercept value
        }
        Returns None if insufficient data points

    Examples:
        >>> df = pd.DataFrame({'date': ['2020-01-01', '2021-01-01'], 'price': [50, 60]})
        >>> trendline = calculate_trendline(df, 'date', 'price')
        >>> print(trendline['equation'])
        'y = 10.0x + 50.0'
    """
    try:
        # Convert dates to numeric (days since first date)
        df_sorted = df[[date_col, value_col]].copy().sort_values(date_col)
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])

        # Remove NaN values
        df_clean = df_sorted.dropna(subset=[value_col])

        if len(df_clean) < 2:
            log.warning(f"⚠️ Trendline: insufficient data points ({len(df_clean)} < 2)")
            return None

        dates = df_clean[date_col]
        values = df_clean[value_col].values

        # Convert dates to days since first date
        days = (dates - dates.min()).dt.days.values

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(days, values)
        r_squared = r_value ** 2

        # Calculate trendline for existing dates
        trendline_dates = dates.tolist()
        trendline_values = (slope * days + intercept).tolist()

        # Extend to future date if requested
        if extend_to_date:
            try:
                future_date = pd.to_datetime(extend_to_date)
                if future_date > dates.max():
                    # Add monthly points from last date to future date
                    extended_dates = pd.date_range(
                        start=dates.max() + pd.DateOffset(months=1),
                        end=future_date,
                        freq='MS'  # Month start
                    )
                    extended_days = (extended_dates - dates.min()).days.values
                    extended_values = slope * extended_days + intercept

                    trendline_dates.extend(extended_dates.tolist())
                    trendline_values.extend(extended_values.tolist())

                    log.info(f"📈 Trendline extended to {future_date.strftime('%Y-%m')}")
            except Exception as e:
                log.warning(f"⚠️ Could not extend trendline to {extend_to_date}: {e}")

        # Format dates as strings
        trendline_dates_str = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in trendline_dates]

        return {
            "dates": trendline_dates_str,
            "values": [float(v) for v in trendline_values],
            "equation": f"y = {slope:.4f}x + {intercept:.2f}",
            "r_squared": float(r_squared),
            "slope": float(slope),
            "intercept": float(intercept)
        }
    except Exception as e:
        log.error(f"❌ Trendline calculation failed: {e}")
        return None


def filter_series_by_relevance(
    num_cols: List[str],
    user_query: str,
    max_series: int = 3
) -> List[str]:
    """
    Filter numeric columns to most relevant series based on query keywords.

    Uses keyword matching to score columns by relevance to the user's query.
    Keeps only the top max_series most relevant columns.

    Args:
        num_cols: List of numeric column names
        user_query: User's natural language query
        max_series: Maximum number of series to keep

    Returns:
        Filtered list of column names (up to max_series)

    Examples:
        >>> cols = ['p_bal_gel', 'p_bal_usd', 'xrate', 'share_import', 'tariff_gel']
        >>> filter_series_by_relevance(cols, 'show me price and xrate', max_series=3)
        ['p_bal_gel', 'xrate', 'p_bal_usd']  # Ordered by relevance
    """
    if len(num_cols) <= max_series:
        return num_cols

    log.info(f"⚠️ Too many series ({len(num_cols)}), limiting to {max_series} most relevant")

    query_lower = user_query.lower()

    def relevance_score(col: str) -> int:
        """Score column by keyword relevance to query."""
        score = 0
        col_lower = col.lower()

        # High priority keywords (exact matches)
        if any(k in query_lower for k in ["price", "ფასი", "цена"]) and any(k in col_lower for k in ["price", "p_bal", "ფასი"]):
            score += 10
        if any(k in query_lower for k in ["xrate", "exchange", "კურსი", "курс"]) and "xrate" in col_lower:
            score += 10
        if any(k in query_lower for k in ["share", "წილი", "доля", "composition"]) and "share" in col_lower:
            score += 5
        if any(k in query_lower for k in ["tariff", "ტარიფი", "тариф"]) and "tariff" in col_lower:
            score += 5

        # Prefer primary metrics (even if not in query)
        if "p_bal" in col_lower:
            score += 3
        if "xrate" in col_lower:
            score += 2

        return score

    # Sort by relevance and take top max_series
    scored_cols = [(col, relevance_score(col)) for col in num_cols]
    scored_cols.sort(key=lambda x: x[1], reverse=True)
    filtered_cols = [col for col, _ in scored_cols[:max_series]]

    log.info(f"📊 Selected series: {filtered_cols}")
    return filtered_cols


def apply_labels(
    df: pd.DataFrame,
    columns: List[str],
    time_key: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Apply human-readable labels to DataFrame columns.

    Uses custom label_map if provided, otherwise uses context.COLUMN_LABELS,
    falling back to title-cased column names.

    Args:
        df: DataFrame to label
        columns: List of column names
        time_key: Time column to exclude from labeling
        label_map: Optional custom label mapping

    Returns:
        Tuple of (labeled_df, label_map_used)

    Examples:
        >>> df = pd.DataFrame({'p_bal_gel': [100, 110], 'xrate': [2.5, 2.6]})
        >>> labeled_df, labels = apply_labels(df, ['p_bal_gel', 'xrate'])
        >>> 'Balancing Price (GEL)' in labeled_df.columns
        True
    """
    # Try to import COLUMN_LABELS from context
    if label_map is None:
        try:
            from context import COLUMN_LABELS
            label_map = COLUMN_LABELS
        except ImportError:
            log.warning("⚠️ Could not import COLUMN_LABELS from context, using fallback")
            label_map = {}

    # Build label map for all columns except time axis
    cols_to_label = [c for c in columns if c != time_key]
    final_label_map = {
        c: label_map.get(c, c.replace("_", " ").title())
        for c in cols_to_label
    }

    # Apply renaming
    labeled_df = df.rename(columns=final_label_map)

    return labeled_df, final_label_map


def build_chart_metadata(
    dimensions: Set[str],
    num_cols: List[str],
    chart_labels: List[str],
    time_key: Optional[str] = None,
    chart_type: str = "line"
) -> Dict[str, any]:
    """
    Build chart metadata including axis titles and labels.

    Determines axis mode (single vs dual), titles, and units based on
    dimensions present in the data.

    Args:
        dimensions: Set of semantic dimensions detected
        num_cols: List of numeric column names
        chart_labels: List of human-readable labels for series
        time_key: Time column name (for x-axis title)
        chart_type: Chart type ('line', 'bar', 'stackedbar', etc.)

    Returns:
        Dictionary with chart metadata:
        {
            'xAxisTitle': str,
            'yAxisTitle': str (single axis) OR
            'yAxisLeft': str, 'yAxisRight': str (dual axis),
            'title': str,
            'axisMode': 'single' or 'dual',
            'labels': List[str]
        }

    Examples:
        >>> dims = {'price_tariff', 'xrate'}
        >>> meta = build_chart_metadata(dims, ['p_bal_gel', 'xrate'],
        ...                              ['Price', 'Exchange Rate'], 'date')
        >>> meta['axisMode']
        'dual'
    """
    axis_mode = determine_axis_mode(dimensions)

    # Dual-axis cases
    if axis_mode == "dual":
        if "index" in dimensions and len(dimensions) > 1:
            # CPI mixed with any other → dual axes (index is always right)
            log.info("📊 Mixed index + other dimension → dual-axis chart.")
            if "price_tariff" in dimensions:
                left_unit = unit_for_price(num_cols)
            else:
                left_unit = unit_for_qty(num_cols)

            return {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": left_unit,
                "yAxisRight": unit_for_index(num_cols),
                "title": "Index vs Other Indicator",
                "axisMode": "dual",
                "labels": chart_labels,
            }

        elif "price_tariff" in dimensions and "xrate" in dimensions:
            # Price + Exchange Rate → dual axes (different units!)
            log.info("📊 Mixed price and xrate → dual-axis chart.")
            return {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": unit_for_price(num_cols),
                "yAxisRight": unit_for_xrate(num_cols),
                "title": "Price vs Exchange Rate",
                "axisMode": "dual",
                "labels": chart_labels,
            }

        elif "price_tariff" in dimensions and "share" in dimensions:
            # Price + Share → dual axes (different scales: 0-200 vs 0-1)
            log.info("📊 Mixed price and share → dual-axis chart.")
            return {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": unit_for_price(num_cols),
                "yAxisRight": unit_for_share(num_cols),
                "title": "Price vs Composition Shares",
                "axisMode": "dual",
                "labels": chart_labels,
            }

        elif "price_tariff" in dimensions and "energy_qty" in dimensions:
            # Price/Tariff + Quantity → dual axes
            log.info("📊 Mixed price/tariff and quantity → dual-axis chart.")
            return {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": unit_for_qty(num_cols),
                "yAxisRight": unit_for_price(num_cols),
                "title": "Quantity vs Price/Tariff",
                "axisMode": "dual",
                "labels": chart_labels,
            }

        elif "xrate" in dimensions and "share" in dimensions:
            # Exchange Rate + Share → dual axes
            log.info("📊 Mixed xrate and share → dual-axis chart.")
            return {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": unit_for_xrate(num_cols),
                "yAxisRight": unit_for_share(num_cols),
                "title": "Exchange Rate vs Composition",
                "axisMode": "dual",
                "labels": chart_labels,
            }

    # Single-axis case
    log.info("📊 Uniform dimension → single-axis chart.")

    # Decide unit by the dimension present
    if dimensions == {"price_tariff"}:
        y_unit = unit_for_price(num_cols)
    elif dimensions == {"energy_qty"}:
        y_unit = unit_for_qty(num_cols)
    elif dimensions == {"index"}:
        y_unit = unit_for_index(num_cols)
    elif dimensions == {"xrate"}:
        y_unit = unit_for_xrate(num_cols)
    elif dimensions == {"share"}:
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


def prepare_chart_data(
    df: pd.DataFrame,
    num_cols: List[str],
    user_query: str,
    time_key: Optional[str] = None,
    max_series: int = 3,
    add_trendlines: bool = False,
    trendline_extend_to: Optional[str] = None
) -> Tuple[List[Dict], Dict[str, any], List[str], Set[str]]:
    """
    Complete chart data preparation pipeline with optional trendline support.

    Performs:
    1. Series filtering (if needed)
    2. Dimension inference
    3. Label application
    4. Metadata generation
    5. Trendline calculation (if requested)

    Args:
        df: DataFrame with query results
        num_cols: List of numeric column names
        user_query: User's query for relevance filtering
        time_key: Time column name
        max_series: Maximum number of series
        add_trendlines: If True, calculate trendlines for time series data
        trendline_extend_to: Optional future date to extend trendlines (e.g., "2035-12-01")

    Returns:
        Tuple of:
        - chart_data: List of dicts (records format)
        - chart_metadata: Dict with axis info, labels, and trendline data
        - filtered_num_cols: List of selected numeric columns
        - dimensions: Set of semantic dimensions

    Examples:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2023-01-01', periods=12, freq='M'),
        ...     'p_bal_gel': range(100, 112),
        ...     'xrate': [2.5] * 12
        ... })
        >>> data, meta, cols, dims = prepare_chart_data(
        ...     df, ['p_bal_gel', 'xrate'], 'show price and xrate', 'date'
        ... )
        >>> meta['axisMode']
        'dual'

        >>> # With trendlines
        >>> data, meta, cols, dims = prepare_chart_data(
        ...     df, ['p_bal_gel'], 'show trend', 'date', add_trendlines=True
        ... )
        >>> 'trendlines' in meta
        True
    """
    # Step 1: Filter series if needed
    filtered_num_cols = filter_series_by_relevance(num_cols, user_query, max_series)

    # Step 2: Infer dimensions
    dim_map = {c: infer_dimension(c) for c in filtered_num_cols}
    dimensions = set(dim_map.values())
    log.info(f"📐 Detected dimensions: {dim_map} → {dimensions}")

    # Step 2.5: Safety check for mixed dimensions (Option 2)
    is_mixed, mix_reason = detect_mixed_dimensions(dimensions)
    if is_mixed:
        log.warning(f"⚠️ CHART DIMENSION WARNING: {mix_reason}")
        log.warning(f"⚠️ Detected dimensions: {dimensions}")
        log.warning(f"⚠️ Columns: {filtered_num_cols}")
        log.warning(f"⚠️ LLM should have created separate chart groups for these metrics!")
        # Note: We still proceed with chart generation, but log the warning
        # In future, could auto-split into multiple charts here

    # Step 3: Apply labels
    labeled_df, label_map = apply_labels(df, df.columns.tolist(), time_key)

    # Step 4: Build chart labels for series
    chart_labels = [label_map.get(c, c) for c in filtered_num_cols]

    # Step 5: Build metadata
    chart_metadata = build_chart_metadata(
        dimensions,
        filtered_num_cols,
        chart_labels,
        time_key
    )

    # Step 6: Convert to records format
    chart_data = labeled_df.to_dict("records")

    # Step 7: Calculate trendlines if requested and time_key exists
    trendlines = []
    if add_trendlines and time_key and time_key in df.columns:
        log.info(f"📈 Calculating trendlines for {len(filtered_num_cols)} series")
        for col in filtered_num_cols:
            trendline_data = calculate_trendline(
                df, time_key, col, extend_to_date=trendline_extend_to
            )
            if trendline_data:
                # Get the label for this series
                label = label_map.get(col, col)
                trendlines.append({
                    "column": col,
                    "label": f"{label} (Trend)",
                    "data": trendline_data,
                    "original_label": label
                })
                log.info(f"  ✅ {label}: R²={trendline_data['r_squared']:.3f}, equation={trendline_data['equation']}")

    # Add trendline info to metadata
    if trendlines:
        chart_metadata["trendlines"] = trendlines
        chart_metadata["has_projection"] = bool(trendline_extend_to)
        chart_metadata["projection_to"] = trendline_extend_to if trendline_extend_to else None
        log.info(f"📊 Added {len(trendlines)} trendlines to chart metadata")

    return chart_data, chart_metadata, filtered_num_cols, dimensions
