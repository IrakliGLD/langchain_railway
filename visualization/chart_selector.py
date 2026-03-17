"""
Chart type selection and decision logic.

Handles:
- Determining if a chart should be generated
- Inferring semantic dimensions from columns
- Selecting appropriate chart types based on structure and semantics
- Unit inference for axis labels
"""
import logging
import re
from typing import List, Set, Tuple

from core.llm import classify_query_type

log = logging.getLogger("Enai")


def should_generate_chart(user_query: str, row_count: int) -> bool:
    """
    Determine if a chart would be helpful for answering the query.

    Uses query type classification for better decisions.
    Returns False if user wants table/list/single value.
    Returns True if user wants trend/comparison or explicitly requests visualization.

    Args:
        user_query: User's natural language query
        row_count: Number of rows in query results

    Returns:
        True if chart should be generated, False otherwise

    Examples:
        >>> should_generate_chart("What was the price in June 2024?", 1)
        False  # Single value query

        >>> should_generate_chart("Show me the trend from 2020 to 2024", 48)
        True  # Trend query with sufficient data

        >>> should_generate_chart("List all entities", 15)
        False  # List query
    """
    query_lower = user_query.lower()
    query_type = classify_query_type(user_query)

    # NEVER generate chart for these query types
    if query_type in ["single_value", "list"]:
        log.info(f"🚫 Skipping chart: query type = {query_type}")
        return False

    # ALWAYS generate chart for these types if enough data
    if query_type in ["comparison", "trend"]:
        if row_count >= 3:
            log.info(f"✅ Generating chart: query type = {query_type}")
            return True
        return False

    # Explicit chart request (highest priority)
    if any(k in query_lower for k in [
        "chart", "graph", "plot", "visualize", "show chart", "draw",
        "დიაგრამა", "გრაფიკი", "график", "визуализ"
    ]):
        if row_count >= 2:
            log.info("✅ Generating chart: explicit request")
            return True
        return False

    # Explicit table request (suppress chart)
    if any(k in query_lower for k in [
        "table", "show table", "tabular", "give me table",
        "ცხრილი", "таблица"
    ]):
        log.info("🚫 Skipping chart: explicit table request")
        return False

    # Don't generate chart for simple fact queries
    no_chart_indicators = [
        "what is the", "what was the", "how much", "how many",
        "give me the value", "tell me the", "რა არის", "რამდენი"
    ]
    for indicator in no_chart_indicators:
        if indicator in query_lower and row_count <= 3:
            log.info(f"🚫 Skipping chart: simple fact query with {row_count} rows")
            return False

    # For unknown/table query types with significant time series data
    # Use conservative threshold (10 instead of 5)
    if query_type in ["unknown", "table"]:
        if row_count >= 10:
            log.info(f"✅ Generating chart: {row_count} rows (time series assumed)")
            return True
        log.info(f"🚫 Skipping chart: only {row_count} rows for {query_type} type")
        return False

    # Default: don't generate chart
    log.info(f"🚫 Skipping chart: no clear indicators (type={query_type}, rows={row_count})")
    return False


def infer_dimension(col: str) -> str:
    """
    Infer semantic dimension from column name.

    Checks in order of specificity to avoid false matches.

    Args:
        col: Column name

    Returns:
        Dimension type: 'xrate', 'share', 'index', 'energy_qty', 'price_tariff', or 'other'

    Examples:
        >>> infer_dimension("xrate")
        'xrate'

        >>> infer_dimension("share_import")
        'share'

        >>> infer_dimension("p_bal_gel")
        'price_tariff'

        >>> infer_dimension("quantity_tech")
        'energy_qty'
    """
    col_l = col.lower()

    # Exchange rate - check FIRST before price (has _gel/_usd but is not a price)
    if any(x in col_l for x in ["xrate", "exchange", "rate", "კურსი"]):
        return "xrate"

    # Shares/proportions - check BEFORE other
    if any(x in col_l for x in ["share_", "წილი_", "proportion", "percent", "პროცენტ"]):
        return "share"

    # Index
    if any(x in col_l for x in ["cpi", "index", "inflation", "ინდექსი"]):
        return "index"

    # Quantity
    if any(x in col_l for x in ["quantity", "generation", "volume_tj", "volume", "mw", "tj", "რაოდენობა", "მოცულობა", "გენერაცია"]):
        return "energy_qty"

    # Price/Tariff - check AFTER xrate
    if any(x in col_l for x in ["price", "tariff", "_gel", "_usd", "p_bal", "p_dereg", "p_gcap", "ფასი", "ტარიფი"]):
        return "price_tariff"

    return "other"


def detect_column_types(columns: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Detect time, category, and value columns from column names.

    Args:
        columns: List of column names

    Returns:
        Tuple of (time_cols, category_cols, value_cols)

    Examples:
        >>> cols = ['date', 'entity', 'price_gel', 'quantity']
        >>> time_cols, cat_cols, val_cols = detect_column_types(cols)
        >>> 'date' in time_cols
        True
        >>> 'entity' in cat_cols
        True
    """
    time_cols = [c for c in columns if re.search(r"(year|month|date|წელი|თვე|თარიღი)", c.lower())]
    category_cols = [c for c in columns if re.search(
        r"(type|sector|entity|source|segment|ownership|technology|region|area|category|ტიპი|სექტორი)",
        c.lower()
    )]
    value_cols = [c for c in columns if re.search(
        r"(quantity|volume|value|amount|price|tariff|cpi|index|mwh|tj|usd|gel|რაოდენობა|მოცულობა|ფასი|ტარიფი|საშუალო|სულ)",
        c.lower()
    )]

    return time_cols, category_cols, value_cols


def select_chart_type(
    has_time: bool,
    has_categories: bool,
    dimensions: Set[str],
    category_count: int = 0
) -> str:
    """
    Select chart type based on structure and semantic dimensions.

    Decision matrix:
    - Time + Categories + Share → stackedbar (composition over time)
    - Time + Categories + Other → line (trend comparison)
    - Time only → line (single time series)
    - Categories + Share (few) → pie (composition snapshot)
    - Categories + Share (many) → bar (composition snapshot)
    - Categories + Other → bar (categorical comparison)

    Args:
        has_time: True if time columns present
        has_categories: True if category columns present
        dimensions: Set of semantic dimensions detected
        category_count: Number of unique categories (for pie vs bar decision)

    Returns:
        Chart type: 'line', 'bar', 'stackedbar', 'pie', or 'dualaxis'

    Examples:
        >>> select_chart_type(True, True, {'share'}, 5)
        'stackedbar'

        >>> select_chart_type(True, False, {'price_tariff'}, 0)
        'line'

        >>> select_chart_type(False, True, {'share'}, 5)
        'pie'
    """
    if has_time and has_categories:
        # Time series with categories: decision depends on dimension
        if "share" in dimensions:
            # Shares over time → stacked bar (part-to-whole composition)
            log.info("📊 Chart type: stackedbar (time + categories + share = composition over time)")
            return "stackedbar"
        elif any(d in dimensions for d in ["price_tariff", "energy_qty", "index", "xrate"]):
            # Prices, quantities, indices, exchange rate → line (trend comparison)
            log.info(f"📊 Chart type: line (time + categories + {dimensions} = trend comparison)")
            return "line"
        else:
            # Mixed or unknown dimensions → default to line for time series
            log.info("📊 Chart type: line (time + categories + mixed/unknown dimensions)")
            return "line"

    elif has_time and not has_categories:
        # Single time series → always line
        log.info("📊 Chart type: line (time series without categories)")
        return "line"

    elif not has_time and has_categories:
        # Categorical comparison (no time): decision depends on dimension
        if "share" in dimensions:
            # Single-period composition: pie if few categories, bar if many
            if category_count <= 8:
                log.info(f"📊 Chart type: pie (composition snapshot with {category_count} categories)")
                return "pie"
            else:
                log.info(f"📊 Chart type: bar (composition snapshot with {category_count} categories, too many for pie)")
                return "bar"
        else:
            # Categorical comparison (prices, quantities, etc.) → bar
            log.info("📊 Chart type: bar (categorical comparison, no time)")
            return "bar"

    else:
        # Fallback: no clear structure
        log.info("📊 Chart type: line (fallback)")
        return "line"


def unit_for_price(cols: List[str]) -> str:
    """
    Determine unit for price/tariff axis.

    Args:
        cols: List of column names

    Returns:
        Unit string: 'GEL/MWh', 'USD/MWh', or 'per MWh'

    Examples:
        >>> unit_for_price(['p_bal_gel', 'tariff_gel'])
        'GEL/MWh'

        >>> unit_for_price(['p_bal_usd'])
        'USD/MWh'

        >>> unit_for_price(['p_bal_gel', 'p_bal_usd'])
        'currency/MWh'  # Mixed currencies
    """
    has_gel = any("_gel" in c.lower() for c in cols)
    has_usd = any("_usd" in c.lower() for c in cols)

    # Mixed currencies share the same physical unit
    if has_gel and has_usd:
        return "currency/MWh"
    if has_gel:
        return "GEL/MWh"
    if has_usd:
        return "USD/MWh"

    # Fallback for generic price columns
    return "currency/MWh"


def unit_for_qty(cols: List[str]) -> str:
    """
    Determine unit for quantity/energy axis.

    Args:
        cols: List of column names

    Returns:
        Unit string: 'TJ', 'thousand MWh', or 'Energy Quantity'

    Examples:
        >>> unit_for_qty(['volume_tj'])
        'TJ'

        >>> unit_for_qty(['quantity_tech'])
        'thousand MWh'
    """
    has_tj = any("tj" in c.lower() for c in cols) or any("volume_tj" in c.lower() for c in cols)
    has_thousand_mwh = any("quantity" in c.lower() or "quantity_tech" in c.lower() for c in cols)

    if has_tj and not has_thousand_mwh:
        return "TJ"
    if has_thousand_mwh and not has_tj:
        return "thousand MWh"

    # Mixed TJ & thousand MWh → generic quantity unit
    return "Energy Quantity"


def unit_for_index(cols: List[str]) -> str:
    """Return unit for index columns."""
    return "Index (2015=100)"


def unit_for_xrate(cols: List[str]) -> str:
    """Return unit for exchange rate columns."""
    return "GEL per USD"


def unit_for_share(cols: List[str]) -> str:
    """Return unit for share/proportion columns."""
    return "Share (0-1)"


def determine_axis_mode(dimensions: Set[str]) -> str:
    """
    Determine if single or dual axis is needed.

    Dual axis is needed when:
    - Index + any other dimension
    - Price + Exchange rate
    - Price + Share
    - Price + Quantity
    - Exchange rate + Share

    Args:
        dimensions: Set of semantic dimensions

    Returns:
        'single' or 'dual'

    Examples:
        >>> determine_axis_mode({'price_tariff', 'xrate'})
        'dual'

        >>> determine_axis_mode({'price_tariff'})
        'single'
    """
    if "index" in dimensions and len(dimensions) > 1:
        return "dual"
    if "price_tariff" in dimensions and "xrate" in dimensions:
        return "dual"
    if "price_tariff" in dimensions and "share" in dimensions:
        return "dual"
    if "price_tariff" in dimensions and "energy_qty" in dimensions:
        return "dual"
    if "xrate" in dimensions and "share" in dimensions:
        return "dual"

    return "single"
