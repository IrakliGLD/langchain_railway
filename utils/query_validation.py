"""
Query validation and relevance checking.

Handles:
- Conceptual/definitional query detection
- SQL relevance validation
- Topic extraction from queries and SQL
- Decision logic for when to skip charts or SQL execution
"""
import logging
import re
from typing import Tuple, Set, Optional

log = logging.getLogger("Enai")


def is_conceptual_question(query: str) -> bool:
    """
    Detect if query is conceptual/definitional (doesn't need data).

    These queries should be answered using domain knowledge only,
    without executing SQL or generating charts.

    Args:
        query: User's natural language query

    Returns:
        True if query is conceptual/definitional

    Examples:
        >>> is_conceptual_question("What is CfD?")
        True

        >>> is_conceptual_question("რა არის PPA?")
        True

        >>> is_conceptual_question("Show me demand trends")
        False
    """
    query_lower = query.lower()

    # Definition indicators (strong signals)
    definition_patterns = [
        r"\bwhat is\b", r"\bwhat are\b", r"\bwhat does\b",
        r"\bdefine\b", r"\bexplain\b", r"\bmeaning of\b",
        r"\bრა არის\b", r"\bრა არიან\b", r"\bგანმარტე\b",
        r"\bчто такое\b", r"\bопредели\b", r"\bобъясни\b",
    ]

    for pattern in definition_patterns:
        if re.search(pattern, query_lower):
            # Check if NOT followed by data-specific keywords
            # e.g., "What is the price in June 2024?" is data query, not conceptual
            # e.g., "Explain balancing price between april 2025 and september 2025" → data query
            if not re.search(
                r"(in \d{4}|for \d{4}|last month|this year|\d+-\d+|\b\d{4}\b|between .{0,40}\d{4})",
                query_lower,
            ):
                return True

    # Explanation indicators
    explanation_keywords = [
        "how does", "როგორ მუშაობს", "როგორ ხდება", "как работает",
        "how works", "explain how", "განმარტე როგორ",
    ]

    for keyword in explanation_keywords:
        if keyword in query_lower:
            # Not if followed by specific entity/time
            if not re.search(r"(\d{4}|entity|specific)", query_lower):
                return True

    # Conceptual comparison (e.g., "What's the difference between CfD and PPA?")
    if re.search(r"(difference between|განსხვავება|различие между)", query_lower):
        # If comparing concepts (not data points)
        if not re.search(r"(\d{4}|price|demand|quantity)", query_lower):
            return True

    return False


def extract_query_topics(query: str) -> Set[str]:
    """
    Extract main topics/keywords from user query.

    Args:
        query: User's natural language query

    Returns:
        Set of normalized topic keywords

    Examples:
        >>> extract_query_topics("What is CfD scheme?")
        {'cfd', 'scheme', 'contract'}

        >>> extract_query_topics("Show me balancing price trends")
        {'balancing', 'price', 'trend'}
    """
    query_lower = query.lower()
    topics = set()

    # Topic keywords mapping (concept → normalized terms)
    topic_map = {
        # Market mechanisms
        'cfd': ['cfd', 'contract for difference', 'კონტრაქტი განსხვავებაზე'],
        'ppa': ['ppa', 'power purchase agreement', 'შესყიდვის ხელშეკრულება'],
        'balancing': ['balancing', 'balancing electricity', 'საბალანსო'],
        'bilateral': ['bilateral', 'bilateral contracts', 'ორმხრივი'],
        'deregulated': ['deregulated', 'dereg', 'დერეგულირებული'],

        # Metrics
        'price': ['price', 'tariff', 'ფასი', 'ტარიფი', 'цена'],
        'demand': ['demand', 'consumption', 'მოთხოვნა', 'მოხმარება'],
        'generation': ['generation', 'supply', 'გენერაცია', 'წარმოება'],
        'quantity': ['quantity', 'volume', 'მოცულობა', 'რაოდენობა'],
        'share': ['share', 'composition', 'წილი', 'სტრუქტურა'],

        # Sources
        'hydro': ['hydro', 'hydropower', 'hpp', 'ჰიდრო', 'წყალმიწოდება'],
        'thermal': ['thermal', 'tpp', 'თერმული', 'თბოელექტრო'],
        'import': ['import', 'იმპორტი', 'импорт'],
        'renewable': ['renewable', 'განახლებადი', 'возобновляемая'],

        # Other
        'exchange_rate': ['exchange rate', 'xrate', 'კურსი', 'курс'],
        'cpi': ['cpi', 'inflation', 'ინფლაცია', 'инфляция'],
    }

    # Check each topic
    for topic_key, keywords in topic_map.items():
        for keyword in keywords:
            if keyword in query_lower:
                topics.add(topic_key)
                break

    return topics


def extract_sql_topics(sql: str) -> Set[str]:
    """
    Extract topics from SQL query (tables and columns).

    Args:
        sql: SQL query string

    Returns:
        Set of normalized topic keywords detected in SQL

    Examples:
        >>> sql = "SELECT p_bal_gel FROM price_with_usd"
        >>> extract_sql_topics(sql)
        {'price', 'balancing'}

        >>> sql = "SELECT quantity_tech FROM tech_quantity_view WHERE type_tech = 'hydro'"
        >>> extract_sql_topics(sql)
        {'quantity', 'generation', 'hydro'}
    """
    sql_lower = sql.lower()
    topics = set()

    # Table → topic mapping
    table_topics = {
        'price_with_usd': {'price', 'balancing'},
        'tariff_with_usd': {'price', 'tariff'},
        'tech_quantity_view': {'quantity', 'generation', 'demand'},
        'trade_derived_entities': {'trade', 'balancing', 'bilateral'},
        'monthly_cpi_mv': {'cpi', 'inflation'},
        'entities_mv': {'entity'},
    }

    # Column → topic mapping
    column_topics = {
        'p_bal': {'price', 'balancing'},
        'p_dereg': {'price', 'deregulated'},
        'tariff': {'price', 'tariff'},
        'xrate': {'exchange_rate'},
        'quantity': {'quantity'},
        'share': {'share', 'composition'},
        'hydro': {'hydro'},
        'thermal': {'thermal'},
        'import': {'import'},
        'cpi': {'cpi'},
    }

    # Value → topic mapping (for WHERE clauses)
    value_topics = {
        'balancing': {'balancing'},
        'bilateral': {'bilateral'},
        'renewable_ppa': {'ppa', 'renewable'},
        'thermal_ppa': {'ppa', 'thermal'},
        'deregulated_hydro': {'hydro', 'deregulated'},
    }

    # Check tables
    for table, table_topic_set in table_topics.items():
        if table in sql_lower:
            topics.update(table_topic_set)

    # Check columns
    for col, col_topic_set in column_topics.items():
        if col in sql_lower:
            topics.update(col_topic_set)

    # Check values (in WHERE clauses)
    for value, value_topic_set in value_topics.items():
        if value in sql_lower:
            topics.update(value_topic_set)

    return topics


def validate_sql_relevance(
    query: str,
    sql: str,
    plan: dict,
    min_overlap: float = 0.3
) -> Tuple[bool, str, bool]:
    """
    Validate if SQL query is relevant to user's question.

    Args:
        query: User's natural language query
        sql: Generated SQL query
        plan: LLM plan dictionary
        min_overlap: Minimum topic overlap ratio (0.0-1.0)

    Returns:
        Tuple of (is_relevant, reason, should_skip_chart)
        - is_relevant: True if SQL matches query topics
        - reason: Human-readable explanation
        - should_skip_chart: True if chart should be skipped

    Examples:
        >>> validate_sql_relevance("What is CfD?", "SELECT quantity FROM tech_quantity_view", {})
        (False, "Conceptual question doesn't need data", True)

        >>> validate_sql_relevance("Show demand trends", "SELECT quantity FROM tech_quantity_view", {})
        (True, "Topics match: demand, quantity", False)
    """
    # Check if conceptual question
    if is_conceptual_question(query):
        log.info("🎓 Detected conceptual question, SQL not needed")
        return False, "Conceptual question doesn't need data query", True

    # Extract topics
    query_topics = extract_query_topics(query)
    sql_topics = extract_sql_topics(sql)

    if not query_topics:
        # If we can't extract topics, assume relevant (benefit of doubt)
        log.info("📋 No clear topics in query, assuming SQL is relevant")
        return True, "No specific topics detected, proceeding", False

    # Calculate overlap
    common_topics = query_topics & sql_topics
    overlap_ratio = len(common_topics) / len(query_topics) if query_topics else 0

    log.info(f"📊 Topic analysis: Query={query_topics}, SQL={sql_topics}, Overlap={common_topics}")

    if overlap_ratio >= min_overlap:
        reason = f"Topics match: {', '.join(common_topics)}"
        log.info(f"✅ SQL relevance validated: {reason} (overlap: {overlap_ratio:.1%})")
        return True, reason, False
    else:
        reason = f"Topic mismatch: Query asked about {query_topics}, SQL queries {sql_topics}"
        log.warning(f"⚠️ SQL may not be relevant: {reason}")

        # If significant mismatch, skip chart but still execute SQL
        # (SQL might still provide some context for the answer)
        should_skip_chart = overlap_ratio < 0.2

        return False, reason, should_skip_chart


def should_skip_sql_execution(
    query: str,
    plan: dict
) -> Tuple[bool, str]:
    """
    Decide if SQL execution should be skipped entirely.

    For purely conceptual questions, we can skip database queries
    and answer using domain knowledge only.

    Args:
        query: User's natural language query
        plan: LLM plan dictionary

    Returns:
        Tuple of (should_skip, reason)

    Examples:
        >>> should_skip_sql_execution("What is CfD?", {})
        (True, "Conceptual question - domain knowledge sufficient")

        >>> should_skip_sql_execution("Show me prices", {})
        (False, "Data query - SQL needed")
    """
    # Check if conceptual
    if is_conceptual_question(query):
        return True, "Conceptual question - domain knowledge sufficient"

    # Check plan intent
    intent = str(plan.get("intent", "")).lower()
    if any(keyword in intent for keyword in ["განმარტება", "explanation", "definition"]):
        return True, f"Plan intent is explanatory: {intent}"

    return False, "Data query - SQL needed"
