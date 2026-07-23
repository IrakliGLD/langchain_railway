"""
Tests for utils/query_validation.py — is_conceptual_question() and related helpers.

Regression coverage for:
  - GH bug: queries like "explain balancing price between april 2025 and september 2025"
    were mis-classified as conceptual because the temporal guard regex only matched
    "in YYYY" / "for YYYY" patterns, not bare four-digit years.
"""
from types import SimpleNamespace

import pytest

from utils.query_validation import (
    extract_query_topics,
    is_conceptual_question,
    validate_sql_relevance,
    validate_tool_relevance,
)

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
    # Analytical queries with definitional prefixes (greedy short-circuit fix)
    "What is a trend of balancing electricity price?",
    "What are the main drivers of the trend?",
    "What is the evolution of demand?",
    "Explain the dynamics of exchange rate",
    "What is the growth of hydro generation?",
    "Explain the decline in thermal generation",
    # ---------------------------------------------------------------
    # Fix #2 (2026-05-16): multi-clause data intent demotes conceptual
    # ---------------------------------------------------------------
    # Q7 production trace 4e9b17da — was wrongly classified conceptual
    # because of "Define" prefix despite "list" + "show" + "last quarter".
    "Define guaranteed source, list the three most recent guaranteed-source generators by name, and show their average sale price to ESCO in the last quarter.",
    # Stripped-down variants that hit each data-intent token individually.
    "What is balancing electricity and show its monthly average",
    "Define CfD and list the largest active contracts",
    "Explain capacity reserves and show the top 5 plants",
    "What is a deregulated plant, and what is the breakdown by entity?",
    "Define net metering and count the registered micro-generators",
    "What is the import dependency and how many TWh did we import last quarter?",
    "Explain balancing settlement and display this week's results",
    # Rolling-window anchors that the per-branch year regex missed
    "What is the balancing price over the last 30 days?",
    "Show me prices in the last 6 months",
    "What were tariffs over the last 2 years?",
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
    # Fix #2 invariants — these MUST remain conceptual after the
    # data-intent demotion is added.  Years and regulatory phrasing alone
    # don't trigger the new pattern; only explicit data verbs do.
    "What is a guaranteed-source generator?",
    "Explain the role of ESCO",
    "What is the purpose of the balancing market?",
    "Define micro-generator under Georgian regulation",
    "What is the situation with power plant liberalization? Are many plants still regulated?",
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


def test_liberalization_query_exposes_market_structure_capability():
    topics = extract_query_topics(
        "What is the situation with power plant liberalization? Are many plants still regulated?"
    )
    assert "market_structure" in topics


def test_raw_liberalization_intent_blocks_analyzer_derived_tariff_overlap():
    """A wrong analyzer metric must not override an incompatible raw-query topic."""
    question_analysis = SimpleNamespace(
        sql_hints=SimpleNamespace(metric="tariff"),
        analysis_requirements=SimpleNamespace(derived_metrics=[]),
        tooling=SimpleNamespace(candidate_tools=[]),
    )

    relevant, reason = validate_tool_relevance(
        "What is the situation with power plant liberalization? Are many plants still regulated?",
        "get_tariffs",
        question_analysis=question_analysis,
    )

    assert relevant is False
    assert "raw query" in reason.lower()


def test_raw_liberalization_intent_survives_analyzer_canonical_rewrite():
    question_analysis = SimpleNamespace(
        sql_hints=SimpleNamespace(metric="tariff"),
        analysis_requirements=SimpleNamespace(derived_metrics=[]),
        tooling=SimpleNamespace(candidate_tools=[]),
    )

    relevant, reason = validate_tool_relevance(
        "Show regulated tariff values",
        "get_tariffs",
        question_analysis=question_analysis,
        raw_query=(
            "What is the situation with power plant liberalization? "
            "Are many plants still regulated?"
        ),
    )

    assert relevant is False
    assert "raw query" in reason.lower()


def test_energy_security_accepts_import_and_system_quantity_sql():
    relevant, reason, skip_chart = validate_sql_relevance(
        "Review the energy security situation in Georgia.",
        """
        SELECT date, type_tech, SUM(quantity_tech)
        FROM tech_quantity_view
        WHERE type_tech IN ('hydro', 'thermal', 'import', 'demand')
        GROUP BY date, type_tech
        """,
        {"intent": "energy_security_review"},
    )

    assert relevant is True
    assert skip_chart is False
    assert "energy_security" in reason


def test_energy_security_still_rejects_unrelated_price_sql():
    relevant, _, skip_chart = validate_sql_relevance(
        "Review the energy security situation in Georgia.",
        "SELECT date, p_bal_gel FROM price_with_usd",
        {"intent": "energy_security_review"},
    )

    assert relevant is False
    assert skip_chart is True


def test_energy_security_rejects_import_only_quantity_sql():
    relevant, _, skip_chart = validate_sql_relevance(
        "Review the energy security situation in Georgia.",
        """
        SELECT date, SUM(quantity_tech)
        FROM tech_quantity_view
        WHERE type_tech = 'import'
        GROUP BY date
        """,
        {"intent": "energy_security_review"},
    )

    assert relevant is False
    assert skip_chart is True


def test_energy_security_accepts_named_import_and_domestic_generation_columns():
    relevant, reason, skip_chart = validate_sql_relevance(
        "Review the energy security situation in Georgia.",
        """
        SELECT date, share_import, total_domestic_generation
        FROM energy_security_mv
        """,
        {"intent": "energy_security_review"},
    )

    assert relevant is True
    assert "energy_security" in reason
    assert skip_chart is False
