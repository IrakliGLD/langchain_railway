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


def test_the_analyzer_own_clarify_signal_is_honoured():
    """The keyword match failed in production; the analyzer had already decided.

    2026-08-15 15:54: query_type=ambiguous, preferred_path=knowledge,
    answer_kind=clarify, confidence 0.95, network_supply_tariffs among the
    candidate topics -- and resolution_policy still came out ANSWER, so a
    theoretical essay shipped instead of a question. When the analyzer has
    read the Georgian question and concluded it is ambiguous, matching English
    keywords against its canonicalisation is a worse signal than the
    conclusion itself.
    """
    from agent.pipeline import _derive_resolution_policy

    # Wording deliberately free of retail and comparison markers: only the
    # analyzer's own verdict is available to act on.
    ctx = _ctx(
        "Assess how the regulated position compares against market outcomes",
        query_type="ambiguous",
        preferred_path="knowledge",
    )
    assert _derive_resolution_policy(ctx) == ResolutionPolicy.CLARIFY
    assert ctx.clarify_reason == "end_user_scope_unspecified"


def test_the_analyzer_clarify_signal_needs_the_retail_topic():
    """An ambiguous question about something else keeps its existing path."""
    from agent.pipeline import _derive_resolution_policy

    ctx = _ctx(
        "Assess how the regulated position compares against market outcomes",
        topics=("market_structure",),
        query_type="ambiguous",
        preferred_path="knowledge",
    )
    assert _derive_resolution_policy(ctx) == ResolutionPolicy.ANSWER


def test_a_fully_scoped_ambiguous_retail_question_is_still_answered():
    """Scope present means there is nothing left to ask."""
    from agent.pipeline import _derive_resolution_policy

    ctx = _ctx(
        "Telmico household cat2 regulated position against market outcomes",
        query_type="ambiguous",
        preferred_path="knowledge",
    )
    assert _derive_resolution_policy(ctx) == ResolutionPolicy.ANSWER


class TestAnsweringTheClarificationEndsIt:
    """"does not work even after category clarification" -- 2026-08-15.

    The user answered twice with a company and a category and was asked the
    same question both times. Two causes, both fixed here: the gate read only
    the query and ignored ``entity_scope``, which is where the analyzer puts
    the scope it extracted; and the alias table could not express a commercial
    category at a named voltage -- including the very example the clarification
    offers, so following the instructions exactly looped forever.
    """

    @pytest.mark.parametrize(
        "entity_scope",
        [
            # Verbatim from the production traces.
            "Telmico; commercial 35-100 category",
            "Telmico; small commercial at 220/380 V",
            # The example the clarification itself gives the user.
            "Telmico, 3.3-6-10 kV, commercial (public supply)",
            "EPS, 220/380 V, household 101-301 kWh",
        ],
    )
    def test_a_scoped_reply_is_not_asked_again(self, entity_scope):
        from agent.pipeline import _derive_resolution_policy

        ctx = _ctx("follow-up naming the scope")
        ctx.question_analysis.entity_scope = entity_scope

        assert _derive_resolution_policy(ctx) == ResolutionPolicy.ANSWER, (
            f"re-asked a question already answered by scope {entity_scope!r}"
        )

    @pytest.mark.parametrize(
        "entity_scope",
        [
            "Telmico; commercial 35-100 category",
            "Telmico, 3.3-6-10 kV, commercial (public supply)",
        ],
    )
    def test_a_scoped_reply_reaches_the_data_path(self, entity_scope):
        """Not being re-asked is not enough: the analyzer still calls these
        follow-ups ambiguous with preferred_path=knowledge, which blocks the
        tool. The user who answered exactly as asked would get an essay."""
        from agent.pipeline import _derive_response_mode

        ctx = _ctx("follow-up naming the scope")
        ctx.question_analysis.entity_scope = entity_scope

        assert _derive_response_mode(ctx) == "data_primary", (
            "a fully scoped retail question must fetch data, not narrate"
        )

    def test_an_unscoped_retail_question_still_goes_to_knowledge(self):
        from agent.pipeline import _derive_response_mode

        ctx = _ctx("how do retail prices compare with the balancing price")
        assert _derive_response_mode(ctx) == "knowledge_primary"


class TestScopeResolution:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Telmico, 3.3-6-10 kV, commercial (public supply)", "3.3-6-10|com|other"),
            ("Telmico; small commercial at 220/380 V", "220/380|com|small"),
            ("commercial at 35-110 kV", "35-110|com|other"),
            # The user's own typo for the voltage band.
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
        "text",
        [
            "household at 35-110 kV",  # not published
            "tariffs in general",
            "commercial customers",  # no voltage named
        ],
    )
    def test_an_unpublished_or_vague_combination_resolves_to_nothing(self, text):
        """Guessing a neighbouring category is worse than widening."""
        from agent.tools.end_user_price_tools import resolve_scope

        _, category = resolve_scope(text)
        assert category is None


def test_the_options_name_selectable_values_not_prose():
    """"the user does not know the categories you are suggesting to compared
    by, tell what are options. e.g. telmico, 3-6-10 kw, public supply"."""
    from agent.summarizer import _build_clarification_options

    ctx = _ctx(_COMPARISON)
    ctx.clarify_reason = "end_user_scope_unspecified"
    options = "\n".join(_build_clarification_options(ctx))
    lowered = options.lower()

    # The user's own example must be expressible from what is offered.
    assert "telmico" in lowered
    assert "3.3-6-10" in lowered, "voltage levels must appear as the view stores them"
    assert "public" in lowered and "universal" in lowered, "supply activity must be named"
    # Consumption bands, so a household customer can identify themselves.
    assert "101" in lowered and "301" in lowered
    # And the high-voltage commercial band.
    assert "35-110" in lowered


def test_a_wholesale_question_carrying_the_retail_topic_is_not_asked_to_clarify():
    """The analyzer offers several candidate topics, so the retail topic can
    ride along on a purely wholesale question. Asking that user to pick a
    household consumption band would be nonsense.

    The discriminator is the analyzer's own ranking: here balancing_price
    outranks the trailing retail topic, whereas the production comparison put
    network_supply_tariffs first.
    """
    from agent.pipeline import _derive_resolution_policy

    ctx = _ctx(
        "What drives the balancing price?",
        topics=("balancing_price", "network_supply_tariffs"),
        scores=(0.95, 0.4),
    )
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
