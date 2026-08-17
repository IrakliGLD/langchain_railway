"""What makes a retail question a make-or-buy comparison.

2026-08-17, post-deploy. A Georgian question -- "for Telmico's 6-10 kV
commercial customer, is the price better on the retail market or the wholesale
one" -- came back with five numeric columns instead of seven: no
``wholesale_benchmark_gel_kwh``, no spread, and therefore no annual comparison
block, because there was nothing to compare.

``asks_for_wholesale_comparison`` reads four English keywords out of
``qa.canonical_query_en`` -- the analyzer's TRANSLATION. The 2026-08-16
canonical (190 chars) happened to contain "wholesale"; the 2026-08-17 one (144
chars) did not, and the whole comparison silently did not happen. The analyzer
had meanwhile nominated ``balancing_price`` among its candidate topics.

This is the fourth time wording markers have failed on this path;
``is_retail_data_question`` already carries the fix pattern in its docstring:
key on the analyzer's structured nomination AND on wording, never wording alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.planner import build_end_user_price_params
from agent.tools.end_user_price_tools import asks_for_wholesale_comparison

# The 2026-08-17 question, and a canonical translation that carries none of the
# English markers -- which is what actually shipped.
GEORGIAN_QUERY = (
    "თელმიკოს 6-10 კვ კომერციული მომხმარებლისთვის, "
    "საცალო ბაზარზე უკეთესი ფასია თუ საბითუმო ბაზარზე?"
)
CANONICAL_WITHOUT_MARKERS = (
    "For Telmico commercial customers connected at 6-10 kV, is the regulated "
    "retail price better than buying on the exchange?"
)


class TestTheTriggerDoesNotDependOnATranslation:
    def test_english_wording_still_triggers(self):
        assert asks_for_wholesale_comparison("compare the supply tariff with wholesale")
        assert asks_for_wholesale_comparison("versus the balancing price")

    def test_georgian_wording_triggers_without_an_english_translation(self):
        """The raw question is authoritative; the canonical is a paraphrase."""
        assert asks_for_wholesale_comparison(GEORGIAN_QUERY)

    def test_the_analyzer_topic_nomination_triggers_it(self):
        """``balancing_price`` among candidate topics IS the question saying so.

        On the failing trace the analyzer emitted
        ["network_supply_tariffs", "market_structure", "balancing_price"] --
        a structured signal that the question reaches the wholesale side, which
        no keyword list has to be lucky enough to catch.
        """
        assert asks_for_wholesale_comparison(
            CANONICAL_WITHOUT_MARKERS,
            topics=("network_supply_tariffs", "market_structure", "balancing_price"),
        )

    def test_a_plain_retail_question_is_not_a_comparison(self):
        """False-positive check: not every retail question compares anything."""
        assert not asks_for_wholesale_comparison(
            "what is the household tariff for Telmico",
            topics=("network_supply_tariffs",),
        )
        assert not asks_for_wholesale_comparison("რა არის მიწოდების ტარიფი")


class TestTheBenchmarkReachesTheToolParams:
    def test_the_georgian_question_asks_for_the_benchmark(self):
        params = build_end_user_price_params(
            CANONICAL_WITHOUT_MARKERS.lower(),
            raw_query=GEORGIAN_QUERY,
            topics=("network_supply_tariffs", "market_structure", "balancing_price"),
        )

        assert params["include_wholesale_benchmark"] is True

    def test_a_plain_retail_question_does_not(self):
        params = build_end_user_price_params(
            "telmico household tariff", topics=("network_supply_tariffs",)
        )

        assert params["include_wholesale_benchmark"] is False


def test_the_failing_trace_now_reaches_the_tool_with_a_benchmark_and_a_window():
    """End to end through the planner, on the 2026-08-17 shape.

    Wiring is the part that fails silently: a correct predicate the caller never
    passes its inputs to changes nothing. This asserts what the tool would
    actually be called with, not what the helpers return in isolation.
    """
    from agent.planner import resolve_tool_params
    from tests.test_end_user_scope_clarification import _ctx

    ctx = _ctx(
        CANONICAL_WITHOUT_MARKERS,
        topics=("network_supply_tariffs", "market_structure", "balancing_price"),
        query_type="ambiguous",
    )
    qa = ctx.question_analysis
    qa.entity_scope = "Telmico commercial customers connected at 6–10 kV"
    qa.sql_hints.period = type(
        "P", (), {"start_date": "2026-06-01", "end_date": "2026-06-30", "granularity": None}
    )()

    params = resolve_tool_params(qa, "get_end_user_prices", GEORGIAN_QUERY)

    assert params is not None
    assert params["include_wholesale_benchmark"] is True, "no benchmark -> nothing to compare"
    assert params["supplier"] == "telmico"
    # A single analyzer month is widened so the annual block has years to report.
    assert params["end_date"] == "2026-06-30"
    assert params["start_date"] < "2022-01-01"


class TestAMakeOrBuyWindowCoversYearsNotOneMonth:
    """rows=1 on the failing trace: a single month cannot answer an irreversible
    choice, and the annual block needs more than one year to show anything."""

    def test_a_single_month_window_is_widened(self):
        from agent.planner import expand_make_or_buy_window

        start, end = expand_make_or_buy_window(
            "2026-06-01", "2026-06-30", is_make_or_buy=True
        )

        assert end == "2026-06-30"
        assert start is not None and start < "2022-01-01"

    def test_an_explicit_multi_year_window_is_left_alone(self):
        from agent.planner import expand_make_or_buy_window

        assert expand_make_or_buy_window(
            "2021-07-01", "2026-06-30", is_make_or_buy=True
        ) == ("2021-07-01", "2026-06-30")

    def test_a_non_comparison_window_is_left_alone(self):
        from agent.planner import expand_make_or_buy_window

        assert expand_make_or_buy_window(
            "2026-06-01", "2026-06-30", is_make_or_buy=False
        ) == ("2026-06-01", "2026-06-30")

    def test_an_open_window_is_left_open(self):
        """No dates means the tool already fetches the full range."""
        from agent.planner import expand_make_or_buy_window

        assert expand_make_or_buy_window(None, None, is_make_or_buy=True) == (None, None)
