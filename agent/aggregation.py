"""
SQL Helper Functions for Aggregation Intent Detection and Validation

This module provides functions to detect what type of aggregation the user wants
and validate that the LLM-generated SQL matches that intent.

Author: AI Engineer Review
Date: 2025-12-10
"""

import re
import logging
from typing import Dict, Tuple

log = logging.getLogger("Enai")


def detect_aggregation_intent(user_query: str) -> Dict[str, bool]:
    """
    Detect aggregation intent from user query.

    Analyzes the user's natural language question to determine:
    - Do they want a total/sum?
    - Do they want an average/mean?
    - Do they want a breakdown (group by)?
    - Do they want shares/percentages?

    Args:
        user_query: The natural language query from the user

    Returns:
        Dictionary with boolean flags:
        {
            "needs_total": True/False,
            "needs_average": True/False,
            "needs_breakdown": True/False,
            "needs_share": True/False,
            "aggregation_type": "total"|"average"|"breakdown"|"share"|"none"
        }

    Examples:
        >>> detect_aggregation_intent("What was total generation in 2023?")
        {"needs_total": True, "needs_breakdown": False, "aggregation_type": "total"}

        >>> detect_aggregation_intent("Show me generation by technology")
        {"needs_total": False, "needs_breakdown": True, "aggregation_type": "breakdown"}

        >>> detect_aggregation_intent("What is share of hydro?")
        {"needs_total": False, "needs_share": True, "aggregation_type": "share"}
    """
    query_lower = user_query.lower()

    # Initialize intent flags
    intent = {
        "needs_total": False,
        "needs_average": False,
        "needs_breakdown": False,
        "needs_share": False,
        "aggregation_type": "none"
    }

    # Total/Sum indicators (English, Georgian, Russian)
    total_indicators = [
        # English
        "total", "sum", "overall", "all", "aggregate", "combined",
        "entire", "whole", "cumulative",
        # Georgian
        "სულ", "ჯამი", "მთლიანი", "ყველა", "ერთობლივი",
        # Russian
        "всего", "сумма", "общий", "итого", "суммарный"
    ]

    # Average indicators
    average_indicators = [
        # English
        "average", "mean", "typical", "avg",
        # Georgian
        "საშუალო", "საშ",
        # Russian
        "средний", "среднее"
    ]

    # Breakdown indicators (implies GROUP BY)
    breakdown_indicators = [
        # English
        "by ", "each ", "per ", "breakdown", "by type", "by entity",
        "by technology", "by sector", "by month", "by year",
        "for each", "grouped by", "broken down",
        # Georgian
        "ტიპის მიხედვით", "ტექნოლოგიის მიხედვით", "თითოეული",
        "ყოველი", "თვეების მიხედვით",
        # Russian
        "по типу", "по технологии", "каждый", "для каждого",
        "в разбивке"
    ]

    # Share/Percentage indicators
    share_indicators = [
        # English
        "share", "percentage", "percent", "proportion", "ratio",
        "composition", "distribution", "portion",
        # Georgian
        "წილი", "პროცენტი", "წილობრივი", "შემადგენლობა",
        # Russian
        "доля", "процент", "доля в", "состав"
    ]

    # Check for each type
    if any(indicator in query_lower for indicator in total_indicators):
        # But if they also say "by", they want total PER category, not grand total
        if not any(ind in query_lower for ind in breakdown_indicators):
            intent["needs_total"] = True
            intent["aggregation_type"] = "total"

    if any(indicator in query_lower for indicator in average_indicators):
        intent["needs_average"] = True
        if intent["aggregation_type"] == "none":
            intent["aggregation_type"] = "average"

    if any(indicator in query_lower for indicator in breakdown_indicators):
        intent["needs_breakdown"] = True
        if intent["aggregation_type"] == "none":
            intent["aggregation_type"] = "breakdown"

    if any(indicator in query_lower for indicator in share_indicators):
        intent["needs_share"] = True
        if intent["aggregation_type"] == "none":
            intent["aggregation_type"] = "share"

    # Special case: "total X by Y" means total PER Y (breakdown with sum)
    if intent["needs_total"] and intent["needs_breakdown"]:
        intent["aggregation_type"] = "breakdown"  # Breakdown takes precedence

    log.info(f"📊 Aggregation intent detected: {intent['aggregation_type']} - {intent}")

    return intent


def validate_aggregation_logic(sql: str, intent: Dict[str, bool]) -> Tuple[bool, str]:
    """
    Validate that SQL matches the detected aggregation intent.

    Checks if the LLM-generated SQL has the correct aggregation logic
    for what the user requested.

    Args:
        sql: The SQL query generated by LLM
        intent: The aggregation intent dict from detect_aggregation_intent()

    Returns:
        Tuple of (is_valid: bool, reason: str)

    Examples:
        >>> sql = "SELECT type_tech, quantity_tech FROM tech_quantity_view WHERE ..."
        >>> intent = {"needs_total": True, "needs_breakdown": False}
        >>> validate_aggregation_logic(sql, intent)
        (False, "Total requested but SQL doesn't use SUM aggregation")

        >>> sql = "SELECT SUM(quantity_tech) FROM tech_quantity_view WHERE ..."
        >>> intent = {"needs_total": True, "needs_breakdown": False}
        >>> validate_aggregation_logic(sql, intent)
        (True, "OK")
    """
    sql_upper = sql.upper()
    sql_lower = sql.lower()

    # 1. Check TOTAL intent
    if intent.get("needs_total") and not intent.get("needs_breakdown"):
        # User wants a single total, should have SUM() and NO GROUP BY
        if "SUM(" not in sql_upper and "sum(" not in sql_lower:
            return False, "Total requested but SQL doesn't use SUM aggregation"

        if "GROUP BY" in sql_upper:
            return False, "Total requested but SQL has GROUP BY (should be single total, not per-category)"

        log.info("✅ Total SQL validation: OK (has SUM, no GROUP BY)")

    # 2. Check AVERAGE intent
    if intent.get("needs_average"):
        if "AVG(" not in sql_upper and "avg(" not in sql_lower:
            return False, "Average requested but SQL doesn't use AVG aggregation"

        log.info("✅ Average SQL validation: OK (has AVG)")

    # 3. Check BREAKDOWN intent
    if intent.get("needs_breakdown"):
        if "GROUP BY" not in sql_upper:
            # Exception: If they want breakdown AND it's a simple entity list, no GROUP BY needed
            # But if aggregation keywords present, GROUP BY is required
            has_aggregation = any(agg in sql_upper for agg in ["SUM(", "AVG(", "COUNT(", "MAX(", "MIN("])

            if has_aggregation:
                return False, "Breakdown with aggregation requested but SQL doesn't have GROUP BY"

        log.info("✅ Breakdown SQL validation: OK")

    # 4. Check SHARE intent
    if intent.get("needs_share"):
        # Share calculations should typically have division (/) or percentage calculation
        if "/" not in sql and "ROUND(" not in sql_upper:
            # Warning only, not a hard failure (LLM might use a share column directly)
            log.warning("⚠️ Share requested but SQL doesn't have division or ROUND (might be using pre-computed share column)")

    # 5. Sanity check: If they want a specific value (not breakdown), result should be simple
    if intent.get("aggregation_type") == "total" or intent.get("aggregation_type") == "average":
        # Should typically return 1 row (single total/average)
        # Can't validate row count here, but can warn if complex query
        if sql.lower().count("union") > 0 or sql.lower().count("join") > 2:
            log.warning("⚠️ Total/Average query looks complex (multiple joins/unions) - verify it returns single value")

    return True, "OK"


def enhance_sql_examples_for_aggregation() -> str:
    """
    Return additional few-shot SQL examples focused on aggregation patterns.

    These examples specifically address the common failures with total calculations
    and group-by logic.

    Returns:
        String with SQL examples to append to few-shot prompt
    """
    return """
-- ============================================================================
-- AGGREGATION EXAMPLES (CRITICAL for Total vs Breakdown disambiguation)
-- ============================================================================

-- Example A1: TOTAL generation (single number, all technologies)
-- User: "What was total generation in 2023?"
-- Intent: Single total across ALL technologies
SELECT
  SUM(quantity_tech) * 1000 AS total_generation_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
  AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
LIMIT 3750;
-- IMPORTANT: NO GROUP BY - returns single row

-- Example A2: TOTAL generation BY TECHNOLOGY (breakdown)
-- User: "What was total generation by technology in 2023?"
-- Intent: Total for EACH technology
SELECT
  type_tech,
  SUM(quantity_tech) * 1000 AS total_generation_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
  AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
GROUP BY type_tech
ORDER BY total_generation_mwh DESC
LIMIT 3750;
-- IMPORTANT: Has GROUP BY - returns multiple rows (one per technology)

-- Example A3: AVERAGE balancing price (single number)
-- User: "What was average balancing price in 2023?"
-- Intent: Single average across entire year
SELECT
  AVG(p_bal_gel) AS average_balancing_price_gel
FROM price_with_usd
WHERE EXTRACT(YEAR FROM date) = 2023
LIMIT 3750;
-- IMPORTANT: NO GROUP BY - returns single average

-- Example A4: SHARE calculation (percentage breakdown)
-- User: "What is share of each technology in total generation for 2023?"
-- Intent: Percentage contribution of each technology
WITH totals AS (
  SELECT
    type_tech,
    SUM(quantity_tech) AS tech_total
  FROM tech_quantity_view
  WHERE EXTRACT(YEAR FROM date) = 2023
    AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
  GROUP BY type_tech
),
grand_total AS (
  SELECT SUM(tech_total) AS overall_total FROM totals
)
SELECT
  t.type_tech,
  t.tech_total * 1000 AS generation_mwh,
  gt.overall_total * 1000 AS total_generation_mwh,
  ROUND((t.tech_total / gt.overall_total) * 100, 2) AS share_percent
FROM totals t, grand_total gt
ORDER BY share_percent DESC
LIMIT 3750;
-- IMPORTANT: Uses CTE to calculate shares properly

-- Example A5: TOTAL vs BREAKDOWN disambiguation
-- User: "Show me hydro generation in 2023"
-- Intent: Unclear - could be total OR monthly breakdown. Default to total if simple.
SELECT
  SUM(quantity_tech) * 1000 AS hydro_generation_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
  AND type_tech = 'hydro'
LIMIT 3750;
-- If they say "show me hydro generation BY MONTH", then use GROUP BY

-- Example A6: Multiple aggregations (total and average)
-- User: "What was total generation and average monthly generation in 2023?"
-- Intent: Both sum and average in same query
SELECT
  SUM(quantity_tech) * 1000 AS total_generation_mwh,
  AVG(quantity_tech) * 1000 AS average_monthly_generation_mwh,
  COUNT(*) AS number_of_months
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
  AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
LIMIT 3750;
-- IMPORTANT: Multiple aggregations, still no GROUP BY (single result row)
"""


def get_aggregation_guidance(intent: Dict[str, bool]) -> str:
    """
    Generate specific SQL guidance based on detected aggregation intent.

    Args:
        intent: Aggregation intent dict from detect_aggregation_intent()

    Returns:
        String with SQL generation guidance for the specific intent
    """
    guidance = []

    if intent.get("needs_total") and not intent.get("needs_breakdown"):
        guidance.append("""
CRITICAL - TOTAL CALCULATION:
- User wants a SINGLE total number
- Use SUM(column_name) AS total
- DO NOT use GROUP BY (should return 1 row)
- Example: SELECT SUM(quantity_tech) FROM tech_quantity_view WHERE type_tech IN (...)
""")

    elif intent.get("needs_breakdown"):
        guidance.append("""
BREAKDOWN CALCULATION:
- User wants totals/values FOR EACH category
- Use SUM(column) or other aggregation
- MUST use GROUP BY category_column
- Example: SELECT type_tech, SUM(quantity_tech) FROM ... GROUP BY type_tech
""")

    if intent.get("needs_average"):
        guidance.append("""
AVERAGE CALCULATION:
- Use AVG(column_name) AS average
- If no breakdown, should return 1 row
- If breakdown requested, use AVG with GROUP BY
""")

    if intent.get("needs_share"):
        guidance.append("""
SHARE/PERCENTAGE CALCULATION:
- Requires: (individual_value / total_value) * 100
- Best practice: Use CTE to calculate total first, then divide
- Example:
  WITH totals AS (SELECT SUM(qty) as total FROM ...)
  SELECT entity, (qty/total)*100 as share FROM ..., totals
""")

    if not guidance:
        # No specific aggregation, simple query
        guidance.append("""
SIMPLE QUERY (no aggregation):
- SELECT columns directly, no SUM/AVG needed
- Use WHERE for filtering
- Add ORDER BY if ordering makes sense
""")

    return "\n".join(guidance)
