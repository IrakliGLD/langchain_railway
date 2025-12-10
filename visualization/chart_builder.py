"""
Chart data building and formatting.

Handles:
- Series filtering based on query relevance
- Label mapping from context
- Chart metadata generation
- Axis title determination
"""
import logging
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd

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
    max_series: int = 3
) -> Tuple[List[Dict], Dict[str, any], List[str], Set[str]]:
    """
    Complete chart data preparation pipeline.

    Performs:
    1. Series filtering (if needed)
    2. Dimension inference
    3. Label application
    4. Metadata generation

    Args:
        df: DataFrame with query results
        num_cols: List of numeric column names
        user_query: User's query for relevance filtering
        time_key: Time column name
        max_series: Maximum number of series

    Returns:
        Tuple of:
        - chart_data: List of dicts (records format)
        - chart_metadata: Dict with axis info and labels
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

    return chart_data, chart_metadata, filtered_num_cols, dimensions
