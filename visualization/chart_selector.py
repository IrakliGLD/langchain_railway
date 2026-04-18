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


def should_generate_chart(
    user_query: str,
    row_count: int,
    *,
    response_mode: str = "",
    question_analysis=None,
) -> bool:
    """
    Determine if a chart would be helpful for answering the query.

    Prefer the structured visualization contract when Stage 0.2 analysis is
    authoritative. Fall back to the older query-type heuristics when that
    contract is unavailable.
    """

    query_lower = user_query.lower()
    explicit_chart_keywords = (
        "chart", "graph", "plot", "visualize", "show chart", "draw",
        "áƒ“áƒ˜áƒáƒ’áƒ áƒáƒ›áƒ", "áƒ’áƒ áƒáƒ¤áƒ˜áƒ™áƒ˜", "Ð³Ñ€Ð°Ñ„Ð¸Ðº", "Ð²Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·",
    )
    explicit_chart_request = any(k in query_lower for k in explicit_chart_keywords)

    if question_analysis is not None:
        vis = getattr(question_analysis, "visualization", None)
        answer_kind = getattr(getattr(question_analysis, "answer_kind", None), "value", None)

        chart_requested = bool(getattr(vis, "chart_requested_by_user", False))
        chart_recommended = bool(getattr(vis, "chart_recommended", False))
        primary_presentation = getattr(getattr(vis, "primary_presentation", None), "value", None)

        if response_mode == "knowledge_primary" and not explicit_chart_request and not chart_requested:
            log.info("Skipping chart: response_mode=knowledge_primary")
            return False

        if primary_presentation in {"text", "table"} and not chart_requested:
            log.info("Skipping chart: visualization plan prefers %s", primary_presentation)
            return False

        if answer_kind in {"knowledge", "clarify"} and not chart_requested:
            log.info("Skipping chart: answer_kind=%s", answer_kind)
            return False

        if answer_kind in {"scalar", "list"} and not chart_requested:
            log.info("Skipping chart: answer_kind=%s", answer_kind)
            return False

        if chart_requested or explicit_chart_request:
            minimum_rows = 1 if answer_kind == "scalar" else 2
            should_draw = row_count >= minimum_rows
            log.info(
                "%s chart: explicit request (answer_kind=%s, rows=%d)",
                "Generating" if should_draw else "Skipping",
                answer_kind,
                row_count,
            )
            return should_draw

        if primary_presentation in {"chart", "chart_plus_table"} or chart_recommended:
            if answer_kind in {"timeseries", "forecast", "scenario", "comparison"}:
                should_draw = row_count >= 2
            elif answer_kind == "explanation":
                should_draw = row_count >= 3
            else:
                should_draw = row_count >= 3
            log.info(
                "%s chart: visualization contract (answer_kind=%s, presentation=%s, recommended=%s, rows=%d)",
                "Generating" if should_draw else "Skipping",
                answer_kind,
                primary_presentation,
                chart_recommended,
                row_count,
            )
            return should_draw

        # Compatibility fallback for older analyzer payloads that do not emit
        # the richer visualization-plan fields yet.
        if answer_kind in {"timeseries", "forecast", "comparison"} and row_count >= 3:
            log.info("Generating chart: answer_kind=%s fallback", answer_kind)
            return True
    else:
        query_type = classify_query_type(user_query)

    if response_mode == "knowledge_primary" and not explicit_chart_request:
        log.info("Skipping chart: response_mode=knowledge_primary")
        return False

    if query_type in ["single_value", "list"]:
        log.info("Skipping chart: query type = %s", query_type)
        return False

    if query_type in ["comparison", "trend"]:
        if row_count >= 3:
            log.info("Generating chart: query type = %s", query_type)
            return True
        return False

    if explicit_chart_request:
        if row_count >= 2:
            log.info("Generating chart: explicit request")
            return True
        return False

    if any(k in query_lower for k in [
        "table", "show table", "tabular", "give me table",
        "áƒªáƒ®áƒ áƒ˜áƒšáƒ˜", "Ñ‚Ð°Ð±Ð»Ð¸Ñ†Ð°",
    ]):
        log.info("Skipping chart: explicit table request")
        return False

    no_chart_indicators = [
        "what is the", "what was the", "how much", "how many",
        "give me the value", "tell me the", "áƒ áƒ áƒáƒ áƒ˜áƒ¡", "áƒ áƒáƒ›áƒ“áƒ”áƒœáƒ˜",
    ]
    for indicator in no_chart_indicators:
        if indicator in query_lower and row_count <= 3:
            log.info("Skipping chart: simple fact query with %d rows", row_count)
            return False

    if query_type in ["unknown", "table"]:
        if row_count >= 10:
            log.info("Generating chart: %d rows (time series assumed)", row_count)
            return True
        log.info("Skipping chart: only %d rows for %s type", row_count, query_type)
        return False

    log.info("Skipping chart: no clear indicators (type=%s, rows=%d)", query_type, row_count)
    return False


def infer_dimension(col: str) -> str:
    """
    Infer semantic dimension from column name.

    Checks in order of specificity to avoid false matches.
    """

    col_l = col.lower()

    if any(x in col_l for x in ["xrate", "exchange", "rate", "áƒ™áƒ£áƒ áƒ¡áƒ˜"]):
        return "xrate"

    if any(x in col_l for x in ["share_", "áƒ¬áƒ˜áƒšáƒ˜_", "proportion", "percent", "áƒžáƒ áƒáƒªáƒ”áƒœáƒ¢"]):
        return "share"

    if any(x in col_l for x in ["cpi", "index", "inflation", "áƒ˜áƒœáƒ“áƒ”áƒ¥áƒ¡áƒ˜"]):
        return "index"

    if any(x in col_l for x in ["quantity", "generation", "volume_tj", "volume", "mw", "tj", "áƒ áƒáƒáƒ“áƒ”áƒœáƒáƒ‘áƒ", "áƒ›áƒáƒªáƒ£áƒšáƒáƒ‘áƒ", "áƒ’áƒ”áƒœáƒ”áƒ áƒáƒªáƒ˜áƒ"]):
        return "energy_qty"

    if any(x in col_l for x in ["price", "tariff", "_gel", "_usd", "p_bal", "p_dereg", "p_gcap", "áƒ¤áƒáƒ¡áƒ˜", "áƒ¢áƒáƒ áƒ˜áƒ¤áƒ˜"]):
        return "price_tariff"

    return "other"


def detect_column_types(columns: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Detect time, category, and value columns from column names.
    """

    time_cols = [c for c in columns if re.search(r"(year|month|date|áƒ¬áƒ”áƒšáƒ˜|áƒ—áƒ•áƒ”|áƒ—áƒáƒ áƒ˜áƒ¦áƒ˜)", c.lower())]
    category_cols = [c for c in columns if re.search(
        r"(type|sector|entity|source|segment|ownership|technology|region|area|category|áƒ¢áƒ˜áƒžáƒ˜|áƒ¡áƒ”áƒ¥áƒ¢áƒáƒ áƒ˜)",
        c.lower(),
    )]
    value_cols = [c for c in columns if re.search(
        r"(quantity|volume|value|amount|price|tariff|cpi|index|mwh|tj|usd|gel|áƒ áƒáƒáƒ“áƒ”áƒœáƒáƒ‘áƒ|áƒ›áƒáƒªáƒ£áƒšáƒáƒ‘áƒ|áƒ¤áƒáƒ¡áƒ˜|áƒ¢áƒáƒ áƒ˜áƒ¤áƒ˜|áƒ¡áƒáƒ¨áƒ£áƒáƒšáƒ|áƒ¡áƒ£áƒš)",
        c.lower(),
    )]

    return time_cols, category_cols, value_cols


def select_chart_type(
    has_time: bool,
    has_categories: bool,
    dimensions: Set[str],
    category_count: int = 0,
) -> str:
    """
    Select chart type based on structure and semantic dimensions.
    """

    if has_time and has_categories:
        if "share" in dimensions:
            log.info("Chart type: stackedbar (time + categories + share = composition over time)")
            return "stackedbar"
        if any(d in dimensions for d in ["price_tariff", "energy_qty", "index", "xrate"]):
            log.info("Chart type: line (time + categories + %s = trend comparison)", dimensions)
            return "line"
        log.info("Chart type: line (time + categories + mixed/unknown dimensions)")
        return "line"

    if has_time and not has_categories:
        log.info("Chart type: line (time series without categories)")
        return "line"

    if not has_time and has_categories:
        if "share" in dimensions:
            if category_count <= 8:
                log.info("Chart type: pie (composition snapshot with %d categories)", category_count)
                return "pie"
            log.info(
                "Chart type: bar (composition snapshot with %d categories, too many for pie)",
                category_count,
            )
            return "bar"
        log.info("Chart type: bar (categorical comparison, no time)")
        return "bar"

    log.info("Chart type: line (fallback)")
    return "line"


def unit_for_price(cols: List[str]) -> str:
    """Determine unit for price/tariff axis."""

    has_gel = any("_gel" in c.lower() for c in cols)
    has_usd = any("_usd" in c.lower() for c in cols)

    if has_gel and has_usd:
        return "currency/MWh"
    if has_gel:
        return "GEL/MWh"
    if has_usd:
        return "USD/MWh"
    return "currency/MWh"


def unit_for_qty(cols: List[str]) -> str:
    """Determine unit for quantity/energy axis."""

    has_tj = any("tj" in c.lower() for c in cols) or any("volume_tj" in c.lower() for c in cols)
    has_thousand_mwh = any("quantity" in c.lower() or "quantity_tech" in c.lower() for c in cols)

    if has_tj and not has_thousand_mwh:
        return "TJ"
    if has_thousand_mwh and not has_tj:
        return "thousand MWh"
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
