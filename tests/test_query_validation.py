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
