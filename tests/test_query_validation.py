"""
Tests for utils/query_validation.py — is_conceptual_question() and related helpers.

Regression coverage for:
  - GH bug: queries like "explain balancing price between april 2025 and september 2025"
    were mis-classified as conceptual because the temporal guard regex only matched
    "in YYYY" / "for YYYY" patterns, not bare four-digit years.
"""
import pytest

from utils.query_validation import is_conceptual_question


# ---------------------------------------------------------------------------
# Queries that SHOULD be classified as data-driven (not conceptual)
# ---------------------------------------------------------------------------
DATA_QUERIES = [
    # The exact failing query from the bug report
    "Could you explain balancing price formation and variation between april 2025 and september 2025?",
    # Other "explain + year range" variants
    "explain balancing price trends from january 2024 to june 2024",
    "explain what happened to prices in 2023",
    "explain price variation in 2025",
    "what is the balancing price in June 2024?",
    "what were prices between 2022 and 2023?",
    "why did prices spike in 2024?",
    # Already covered by original code
    "Show me demand trends",
    "What is the price in June 2024?",
]

# ---------------------------------------------------------------------------
# Queries that SHOULD be classified as purely conceptual
# ---------------------------------------------------------------------------
CONCEPTUAL_QUERIES = [
    "What is CfD?",
    "What are PPAs?",
    "Explain how balancing electricity works",
    "Define renewable energy",
    "What is the difference between CfD and PPA?",
    "explain the concept of balancing electricity",
]


@pytest.mark.parametrize("query", DATA_QUERIES)
def test_data_queries_not_conceptual(query):
    """Queries containing specific time periods must NOT be classified as conceptual."""
    assert is_conceptual_question(query) is False, (
        f"Expected data query, but was classified as conceptual: {query!r}"
    )


@pytest.mark.parametrize("query", CONCEPTUAL_QUERIES)
def test_conceptual_queries_are_conceptual(query):
    """Queries that are genuinely definitional SHOULD be classified as conceptual."""
    assert is_conceptual_question(query) is True, (
        f"Expected conceptual, but was classified as data query: {query!r}"
    )


# -----------------------------------------------------------------------
# Regression: time-bound "why" questions must never skip SQL
# -----------------------------------------------------------------------
from utils.query_validation import should_skip_sql_execution

TIME_BOUND_WHY_CASES = [
    ("why did balancing electricity price change in November 2022?", {"intent": "price_explanation"}),
    ("why did balancing electricity price change in 2022?", {"intent": "explanation"}),
    ("explain the price spike in November 2022", {"intent": "price_explanation"}),
    ("what caused the tariff change in 2023-03?", {"intent": "definition"}),
]

@pytest.mark.parametrize("query,plan", TIME_BOUND_WHY_CASES)
def test_time_bound_explanation_must_not_skip_sql(query, plan):
    """A 'why' question that names a specific period+metric must run SQL."""
    skip, reason = should_skip_sql_execution(query, plan)
    assert skip is False, (
        f"Expected SQL to run, but got skip=True for: {query!r}\nReason: {reason}"
    )


# -----------------------------------------------------------------------
# Regression: pure conceptual explanations (no time, no metric) still skip
# -----------------------------------------------------------------------
PURE_CONCEPTUAL_CASES = [
    ("explain how balancing electricity works", {"intent": "explanation"}),
    ("what is the definition of a PPA?", {"intent": "definition"}),
    ("explain the concept of deregulated market", {"intent": "explanation"}),
]

@pytest.mark.parametrize("query,plan", PURE_CONCEPTUAL_CASES)
def test_pure_conceptual_with_explanation_intent_skips_sql(query, plan):
    """Pure conceptual questions with explanation intent should still skip SQL."""
    skip, reason = should_skip_sql_execution(query, plan)
    assert skip is True, (
        f"Expected SQL to be skipped, but got skip=False for: {query!r}\nReason: {reason}"
    )
