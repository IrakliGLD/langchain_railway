"""Comparing a retail price to the wholesale benchmark needs a named scope.

2026-08-15, from the domain owner: "the main question is which category to
compare, you cannot average and mix company/category". A comparison against
the balancing price is only meaningful for ONE customer category on ONE
supply company's network -- there are eight categories and two company pairs,
and their final prices differ. So the comparison asks first.

A plain trend question is different: it shows every category side by side
(see tests/test_grouped_column_aggregates.py) rather than asking.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest  # noqa: E402

from contracts.question_analysis import QuestionAnalysis  # noqa: E402
from models import QueryContext, ResolutionPolicy  # noqa: E402


def _ctx(canonical: str, *, topics=("network_supply_tariffs",), query_type="ambiguous",
         preferred_path="knowledge") -> QueryContext:
    payload = {
        "version": "question_analysis_v1",
        "raw_query": canonical,
        "canonical_query_en": canonical,
        "language": {"input_language": "ka", "answer_language": "ka"},
        "classification": {
            "query_type": query_type,
            "analysis_mode": "light",
            "intent": "test intent",
            "needs_clarification": False,
            "confidence": 0.9,
        },
        "routing": {
            "preferred_path": preferred_path,
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [
                {"name": t, "score": 0.9} for t in topics
            ]
        },
        "tooling": {"candidate_tools": []},
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    ctx = QueryContext(query=canonical)
    ctx.question_analysis = QuestionAnalysis(**payload)
    ctx.question_analysis_source = "llm_active"
    ctx.resolved_query = canonical
    return ctx


_COMPARISON = (
    "How does the final end-user electricity price compare with the balancing price?"
)


def test_scope_free_benchmark_comparison_asks_first():
    from agent.pipeline import _derive_resolution_policy

    ctx = _ctx(_COMPARISON)
    assert _derive_resolution_policy(ctx) == ResolutionPolicy.CLARIFY
    assert ctx.clarify_reason == "end_user_scope_unspecified"


@pytest.mark.parametrize(
    "canonical",
    [
        # Category named but no company -- still ambiguous, the two networks
        # carry different distribution tariffs.
        "Compare the final end-user price for household cat2 with the balancing price",
        # Company named but no category -- eight categories on that network.
        "Compare Telmico end-user prices with the balancing price",
    ],
)
def test_half_a_scope_is_still_not_a_scope(canonical):
    from agent.pipeline import _derive_resolution_policy

    ctx = _ctx(canonical)
    assert _derive_resolution_policy(ctx) == ResolutionPolicy.CLARIFY, (
        "both company and category are required before comparing"
    )


def test_a_fully_scoped_comparison_is_answered():
    from agent.pipeline import _derive_resolution_policy

    ctx = _ctx(
        "Compare the Telmico household cat2 end-user price with the balancing price"
    )
    assert _derive_resolution_policy(ctx) == ResolutionPolicy.ANSWER
    assert ctx.clarify_reason == ""


def test_a_plain_trend_question_shows_every_category_instead_of_asking():
    """The domain owner asked for all categories on broad questions."""
    from agent.pipeline import _derive_resolution_policy

    ctx = _ctx(
        "What is the trend in end-user electricity prices?",
        query_type="data_retrieval",
        preferred_path="tool",
    )
    assert _derive_resolution_policy(ctx) == ResolutionPolicy.ANSWER


def test_a_non_retail_comparison_is_untouched():
    from agent.pipeline import _derive_resolution_policy

    ctx = _ctx(
        "Compare the Enguri plant tariff with the balancing price",
        topics=("tariffs",),
    )
    assert _derive_resolution_policy(ctx) == ResolutionPolicy.ANSWER


def test_a_wholesale_question_carrying_the_retail_topic_is_not_asked_to_clarify():
    """The analyzer offers several candidate topics, so the retail topic can
    ride along on a purely wholesale question. Asking that user to pick a
    household consumption band would be nonsense."""
    from agent.pipeline import _derive_resolution_policy

    ctx = _ctx("What drives the balancing price?")
    assert _derive_resolution_policy(ctx) == ResolutionPolicy.ANSWER


def test_a_distribution_company_name_pins_its_supply_pair():
    """Naming Telasi means the Tbilisi stack, whose supply component is
    Telmico's. Only the supply code is emitted -- the tool rejects "telasi"
    as a supplier by design."""
    from agent.tools.end_user_price_tools import SUPPLIER_TO_DISTRIBUTOR, resolve_scope

    for text, expected in (
        ("telasi network tariff for cat2", "telmico"),
        ("energo-pro georgia distribution tariff cat2", "eps"),
    ):
        supplier, _ = resolve_scope(text)
        assert supplier == expected
        assert supplier in SUPPLIER_TO_DISTRIBUTOR, "must emit a supply code, not a network"


def test_an_already_chosen_branch_is_not_re_asked():
    """Answering the clarification must not loop back into the same question."""
    from agent.pipeline import _derive_resolution_policy

    ctx = _ctx(_COMPARISON)
    ctx.clarify_selection_override = True
    assert _derive_resolution_policy(ctx) == ResolutionPolicy.ANSWER


def test_the_clarification_names_both_companies_and_the_categories():
    from agent.summarizer import _build_clarification_options

    ctx = _ctx(_COMPARISON)
    ctx.clarify_reason = "end_user_scope_unspecified"
    options = "\n".join(_build_clarification_options(ctx)).lower()

    # Both supply companies, under names the user would recognise.
    assert "telmico" in options and "telasi" in options
    assert "ep georgia" in options or "eps" in options
    assert "energo-pro" in options or "epg" in options
    # And the category axis, not just the company axis.
    assert "household" in options
    assert "commercial" in options
