"""Unit tests for ``core.query_classifier`` heuristics (audit P0-1 leaf).

Both functions are pure, rule-based, and dependency-free (stdlib ``re`` only).
These cover the priority ordering that the rules deliberately encode — the part
most likely to regress when a new keyword is added to the wrong branch — plus
false-positive and multilingual cases.
"""
import pytest

from core.query_classifier import classify_query_type, get_query_focus


@pytest.mark.parametrize(
    "query,expected",
    [
        # single_value: value phrase + a period token
        ("What was the balancing price in December 2023?", "single_value"),
        # regulatory_procedure wins over the broad "list" fallback...
        ("Who is eligible to participate in the market?", "regulatory_procedure"),
        # ...but a data-intent query is NOT a regulatory procedure
        ("How many participants are registered?", "unknown"),
        ("List all licensed entities", "list"),
        ("Compare tariffs vs balancing prices", "comparison"),
        ("Show the trend over time", "trend"),
        ("monthly generation 2024", "trend"),
        ("Show me all detailed data in a table", "table"),
        ("Random unrelated sentence", "unknown"),
    ],
)
def test_classify_query_type(query, expected):
    assert classify_query_type(query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is the CPI in 2024?", "cpi"),
        # tariff is more specific than balancing and checked first
        ("Show tariff schedule for 2024", "tariff"),
        # balancing is the catch-all last check
        ("balancing price dynamics", "balancing"),
        # data-intent registration query must NOT get the regulation focus
        ("How many participants are registered?", "general"),
        # eligibility/registration without data-intent -> regulation
        ("What are the requirements for registration?", "regulation"),
        # import dependence is energy_security, checked before trade
        ("What is our import dependence?", "energy_security"),
        ("electricity imports and exports by year", "trade"),
        # Georgian CPI keyword
        ("ინფლაცია 2024", "cpi"),
    ],
)
def test_get_query_focus(query, expected):
    assert get_query_focus(query) == expected
