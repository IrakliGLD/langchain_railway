"""A pasted profile states inputs; it does not request a topic.

2026-08-17, spans 5514de3c and c3e08d37. With the analyzer lost, the turn fell
back to legacy plan+SQL. The SQL was right -- price_with_usd joined to
demand_tariff_mv, end-user tariff against wholesale. The relevance guard blocked
it anyway:

    Query asked about {'demand'}, SQL queries
    {'end_user_tariff', 'price', 'balancing', 'tariff'}

'demand' came from the Georgian word for consumption, in the user's own line
"my consumption by months is as follows" above twelve pasted figures. The user
was stating their input, not asking for demand data. Overlap 0/1 hard-blocked
execution to zero rows, and the model then answered about missing data.

Same rule as the report-track guardrails: what a question ASKS FOR is decided
by the question, not by data quoted underneath it.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from utils.query_validation import (  # noqa: E402
    extract_query_topics,
    validate_sql_relevance,
)

PASTED_PROFILE_QUERY = (
    "ვარ 35-100 კვ თელმიკოს მომხმარებელი. ჩემი მოხმარება თვეების ხედვით არის შემდეგი:\n"
    "იანვარი 412000\n"
    "თებერვალი 389500\n"
    "მარტი 401200\n"
    "აპრილი 355800\n"
    "მაისი 341000\n"
    "ივნისი 327400\n"
    "ივლისი 366900\n"
    "აგვისტო 372300\n"
    "სექტემბერი 358100\n"
    "ოქტომბერი 384600\n"
    "ნოემბერი 407700\n"
    "დეკემბერი 431500"
)

INCIDENT_SQL = """
SELECT d.date, d.end_user_tariff_gel, p.p_bal_gel
FROM demand_tariff_mv d
JOIN price_with_usd p ON p.date = d.date
ORDER BY d.date
"""


def test_consumption_stated_in_a_pasted_profile_is_not_a_demand_request():
    topics = extract_query_topics(PASTED_PROFILE_QUERY)

    assert "demand" not in topics


def test_a_real_demand_question_still_reads_as_demand():
    """The guard must keep working for questions that do ask about demand."""
    assert "demand" in extract_query_topics("Show me electricity demand for 2025")
    assert "demand" in extract_query_topics("როგორია მოხმარება 2025 წელს?")


def test_the_incident_sql_is_no_longer_blocked():
    is_relevant, reason, skip_chart = validate_sql_relevance(
        PASTED_PROFILE_QUERY, INCIDENT_SQL, {}
    )

    assert is_relevant, f"still blocked: {reason}"
    assert not skip_chart


def test_topics_asked_for_in_the_question_line_still_count():
    """Stripping the pasted rows must not strip the question above them."""
    topics = extract_query_topics(
        "რა იყო საბალანსო ფასი?\nიანვარი 412000\nთებერვალი 389500\nმარტი 401200"
    )

    assert "balancing" in topics
    assert "price" in topics


def test_an_unrelated_sql_is_still_blocked():
    """The guard's real job survives: this must not become permissive."""
    is_relevant, _reason, _skip = validate_sql_relevance(
        "Show me hydro generation in 2025",
        "SELECT date, exchange_rate FROM fx_rates ORDER BY date",
        {},
    )

    assert not is_relevant
