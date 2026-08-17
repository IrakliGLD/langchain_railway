"""A dateless make-or-buy question must not be answered about one month.

2026-08-17, second attempt. The widener I added fetched 61 months but left the
analyzer's declared period at August 2026, and the summarizer's question IS the
analyzer's canonical query -- the census proves it, user_question_chars equals
canonical_query_en length in every trace. So the model answered "is retail
cheaper in August 2026" while holding six years of evidence, and titled its
result accordingly.

The third request in that same session is the control: it had NO period, no
widening, 66 rows, and covered everything. Unbounded beats widened.

The rule: a period the USER stated is honoured; a period the ANALYZER invented on
a dateless question is cleared, because a make-or-buy decision is irreversible
and a single month cannot inform it.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.planner import resolve_make_or_buy_window


class TestAnInventedPeriodIsCleared:
    def test_the_analyzer_pinned_month_is_dropped_when_the_user_named_no_date(self):
        """The exact trace: analyzer set 2026-08-01..2026-08-31 on a dateless question."""
        assert resolve_make_or_buy_window(
            "2026-08-01", "2026-08-31",
            is_make_or_buy=True,
            user_stated_dates=False,
        ) == (None, None)

    def test_an_already_open_window_stays_open(self):
        assert resolve_make_or_buy_window(
            None, None, is_make_or_buy=True, user_stated_dates=False
        ) == (None, None)

    def test_even_a_wide_invented_window_is_cleared(self):
        """An invented window is not evidence of intent at any width.

        Clearing gives the full published range, which is what the control
        request produced -- and it recovers the months a 5-year window loses
        (61 rows against 66).
        """
        assert resolve_make_or_buy_window(
            "2024-01-01", "2026-08-31",
            is_make_or_buy=True,
            user_stated_dates=False,
        ) == (None, None)


class TestAUserStatedPeriodIsHonoured:
    def test_a_month_the_user_named_is_kept(self):
        """Overriding an explicit request answers a question nobody asked."""
        assert resolve_make_or_buy_window(
            "2024-06-01", "2024-06-30",
            is_make_or_buy=True,
            user_stated_dates=True,
        ) == ("2024-06-01", "2024-06-30")

    def test_a_range_the_user_named_is_kept(self):
        assert resolve_make_or_buy_window(
            "2022-01-01", "2024-12-31",
            is_make_or_buy=True,
            user_stated_dates=True,
        ) == ("2022-01-01", "2024-12-31")


class TestNonComparisonQuestionsAreUntouched:
    def test_a_non_make_or_buy_window_is_left_exactly_as_it_was(self):
        assert resolve_make_or_buy_window(
            "2026-08-01", "2026-08-31",
            is_make_or_buy=False,
            user_stated_dates=False,
        ) == ("2026-08-01", "2026-08-31")


def test_the_planner_clears_the_contract_period_too():
    """Otherwise the contract still declares the month.

    sql_hints.period feeds the continuity snapshot, so an invented month left in
    place is inherited by the NEXT turn as well.
    """
    from agent.planner import resolve_tool_params
    from tests.test_end_user_scope_clarification import _ctx

    ctx = _ctx(
        "For Telmico commercial customers at 6-10 kV, is the retail price cheaper "
        "than the wholesale market?",
        topics=("network_supply_tariffs", "market_structure", "balancing_price"),
        query_type="comparison",
    )
    qa = ctx.question_analysis
    qa.entity_scope = "Telmico commercial customers connected at 6–10 kV"
    qa.sql_hints.period = type(
        "P", (), {"start_date": "2026-08-01", "end_date": "2026-08-31", "granularity": None}
    )()

    params = resolve_tool_params(qa, "get_end_user_prices", "თელმიკოს 6-10 კვ კომერციული")

    assert params is not None
    assert params["include_wholesale_benchmark"] is True
    assert "start_date" not in params, "an invented month still bounds the fetch"
    assert "end_date" not in params
    assert qa.sql_hints.period is None, "the contract still declares the month"


def test_the_annual_block_states_the_range_it_covers():
    """The model titled its answer for a single month while holding six years.

    Naming the covered span in the evidence gives it a grounded basis for saying
    what the comparison actually spans.
    """
    import pandas as pd

    from agent.analyzer import _append_annual_comparison
    from models import QueryContext
    from tests.test_annual_comparison_block import _rows

    records = _rows(2022, range(1, 13), 0.145, 0.147) + _rows(2024, range(1, 13), 0.170, 0.168)
    df = pd.DataFrame(records)
    ctx = QueryContext(query="q", df=df, cols=list(df.columns))
    _append_annual_comparison(ctx)

    assert "covers 2022-01 to 2024-12" in ctx.stats_hint
