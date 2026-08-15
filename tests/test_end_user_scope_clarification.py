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
         preferred_path="knowledge", scores=None) -> QueryContext:
    scores = scores or tuple(0.9 for _ in topics)
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
                {"name": t, "score": s} for t, s in zip(topics, scores)
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


class TestRetailQuestionsAnswerFromDataThenOfferToNarrow:
    """2026-08-15, from the domain owner after seeing both behaviours:

    "i prefer general answer and then ask to clarify to provide targeted
    information and assessment", and make all three questions behave the same.

    Withholding the general answer to ask which category was meant is
    therefore wrong. The same three questions had previously taken three
    different paths in one session: one clarified, one fetched data, and one
    wrote an essay.
    """

    @pytest.mark.parametrize(
        "topics,scores,query_type",
        [
            # Q1/Q3 shape: analyzer says ambiguous, prefers knowledge.
            (("network_supply_tariffs",), (0.9,), "ambiguous"),
            # Q3 exactly: the retail topic ranked THIRD, which used to
            # suppress the retail path entirely.
            (
                ("tariffs", "market_structure", "network_supply_tariffs"),
                (0.9, 0.8, 0.7),
                "ambiguous",
            ),
            # Q2 shape: already a data question.
            (("network_supply_tariffs",), (0.9,), "data_retrieval"),
        ],
    )
    def test_every_retail_question_reaches_the_data(self, topics, scores, query_type):
        from agent.pipeline import _derive_response_mode

        ctx = _ctx(
            "how have end-user prices moved",
            topics=topics,
            scores=scores,
            query_type=query_type,
        )
        assert _derive_response_mode(ctx) == "data_primary"

    @pytest.mark.parametrize(
        "query_type", ["ambiguous", "data_retrieval", "comparison"]
    )
    def test_no_retail_question_is_blocked_on_a_clarification(self, query_type):
        from agent.pipeline import _derive_resolution_policy

        ctx = _ctx(_COMPARISON, query_type=query_type)
        assert _derive_resolution_policy(ctx) == ResolutionPolicy.ANSWER, (
            "the general answer must not be withheld to ask which category"
        )

    def test_an_unscoped_comparison_also_answers(self):
        """Previously the one case that always asked."""
        from agent.pipeline import _derive_resolution_policy, _derive_response_mode

        ctx = _ctx(_COMPARISON)
        assert _derive_resolution_policy(ctx) == ResolutionPolicy.ANSWER
        assert _derive_response_mode(ctx) == "data_primary"

    def test_a_non_retail_ambiguous_question_is_untouched(self):
        from agent.pipeline import _derive_response_mode

        ctx = _ctx("what is happening in the market", topics=("market_structure",))
        assert _derive_response_mode(ctx) == "knowledge_primary"

    def test_a_retail_definition_question_stays_on_knowledge(self):
        """"What is a supply tariff" needs prose, not sixteen price series."""
        from agent.pipeline import _derive_response_mode

        ctx = _ctx(
            "what is a supply tariff", query_type="conceptual_definition"
        )
        assert _derive_response_mode(ctx) == "knowledge_primary"


class TestScopeResolution:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Telmico, 3.3-6-10 kV, commercial (public supply)", "3.3-6-10|com|other"),
            ("Telmico; small commercial at 220/380 V", "220/380|com|small"),
            ("commercial at 35-110 kV", "35-110|com|other"),
            ("commercial 35-100 category", "35-110|com|other"),
            ("household at 3.3-6-10 kV", "3.3-6-10|hh|"),
            ("household 101-301 kWh at 220/380", "220/380|hh|cat2"),
        ],
    )
    def test_voltage_and_class_compose_into_a_category(self, text, expected):
        from agent.tools.end_user_price_tools import resolve_scope

        _, category = resolve_scope(text)
        assert category == expected

    @pytest.mark.parametrize(
        "text,expected",
        [("EPS, 220/380 V, household 101-301 kWh", "eps"), ("Telasi network", "telmico")],
    )
    def test_a_company_is_recognised_next_to_punctuation(self, text, expected):
        """"EPS," is not " eps " -- the old space-padded alias missed it."""
        from agent.tools.end_user_price_tools import resolve_scope

        supplier, _ = resolve_scope(text)
        assert supplier == expected

    @pytest.mark.parametrize(
        "text", ["household at 35-110 kV", "tariffs in general", "commercial customers"]
    )
    def test_an_unpublished_or_vague_combination_resolves_to_nothing(self, text):
        from agent.tools.end_user_price_tools import resolve_scope

        _, category = resolve_scope(text)
        assert category is None


def test_the_guidance_requires_answering_before_offering_to_narrow():
    from skills.loader import load_reference

    rules = load_reference("energy-analyst", "retail-tariff-rules.md").lower()

    assert "never withhold the general answer" in rules
    # The offer must name real, selectable options.
    assert "telmico" in rules and "eps" in rules
    assert "3.3–6–10" in rules or "3.3-6-10" in rules
    assert "35–110" in rules or "35-110" in rules
    assert "101" in rules and "301" in rules
