"""A consumption-weighted wholesale benchmark, when the consumer states a load shape.

Domain owner, 2026-08-17: show the simple average and remark on it, and when the
user provides monthly consumption, provide the weighted average.

It has to be computed in CODE, not by the model. A weighted mean the model works
out appears in no row of the corpus, so the provenance gate strips it -- the same
mechanism that cut fourteen figures on 2026-08-15. So the analyzer extracts the
profile into the contract, and the annual block does the arithmetic.

Only relative weights matter to a weighted mean, so the profile may be given in
kWh or in shares; the result is identical either way.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.analyzer import _append_annual_comparison
from models import QueryContext
from tests.test_annual_comparison_block import _rows

# Summer (Apr-Jul) cheap, winter dear -- the real shape of the Georgian year.
SEASONAL_BENCHMARK = [0.170, 0.170, 0.170, 0.120, 0.120, 0.120, 0.120,
                      0.170, 0.170, 0.170, 0.170, 0.170]
# Unweighted mean of the above = 0.15333.
SUMMER_HEAVY = {4: 300.0, 5: 300.0, 6: 300.0, 7: 300.0, 1: 25.0, 2: 25.0, 3: 25.0,
                8: 25.0, 9: 25.0, 10: 25.0, 11: 25.0, 12: 25.0}


class TestTheContractCarriesTheProfile:
    def test_a_month_to_consumption_map_is_accepted(self):
        from contracts.question_analysis import AnalysisRequirementsInfo

        req = AnalysisRequirementsInfo(monthly_consumption={1: 100.0, 7: 250.5})

        assert req.monthly_consumption == {1: 100.0, 7: 250.5}

    def test_it_defaults_to_absent(self):
        from contracts.question_analysis import AnalysisRequirementsInfo

        assert AnalysisRequirementsInfo().monthly_consumption is None

    def test_months_outside_the_calendar_are_rejected(self):
        from contracts.question_analysis import AnalysisRequirementsInfo

        with pytest.raises(ValueError):
            AnalysisRequirementsInfo(monthly_consumption={0: 10.0})
        with pytest.raises(ValueError):
            AnalysisRequirementsInfo(monthly_consumption={13: 10.0})

    def test_negative_consumption_is_rejected(self):
        from contracts.question_analysis import AnalysisRequirementsInfo

        with pytest.raises(ValueError):
            AnalysisRequirementsInfo(monthly_consumption={5: -1.0})

    def test_a_profile_with_no_consumption_at_all_is_rejected(self):
        """All-zero weights would make the weighted mean undefined."""
        from contracts.question_analysis import AnalysisRequirementsInfo

        with pytest.raises(ValueError):
            AnalysisRequirementsInfo(monthly_consumption={1: 0.0, 2: 0.0})


def _ctx(records, *, profile=None) -> QueryContext:
    df = pd.DataFrame(records)
    ctx = QueryContext(query="regulated vs wholesale", df=df, cols=list(df.columns))
    if profile is not None:
        from tests.test_end_user_scope_clarification import _ctx as _analysis_ctx

        analysis = _analysis_ctx("q", topics=("network_supply_tariffs",)).question_analysis
        analysis.analysis_requirements.monthly_consumption = profile
        ctx.question_analysis = analysis
        ctx.question_analysis_source = "llm_active"
    return ctx


def _run(records, *, profile=None) -> str:
    ctx = _ctx(records, profile=profile)
    _append_annual_comparison(ctx)
    return ctx.stats_hint


def _year_line(stats: str, year: str = "2022") -> str:
    return next(line for line in stats.splitlines() if line.strip().startswith(year))


class TestTheWeightedFigureSitsBesideTheUnweightedOne:
    def test_without_a_profile_nothing_weighted_is_emitted(self):
        stats = _run(_rows(2022, range(1, 13), 0.145, SEASONAL_BENCHMARK))

        assert "weighted" in stats.lower(), "the unweighted label must still be there"
        assert "consumption-weighted" not in stats.lower()

    def test_a_summer_heavy_profile_lands_below_the_unweighted_mean(self):
        """The whole point: this consumer's real wholesale cost is far lower."""
        stats = _run(
            _rows(2022, range(1, 13), 0.145, SEASONAL_BENCHMARK), profile=SUMMER_HEAVY
        )
        line = _year_line(stats)

        assert "consumption-weighted" in line.lower()
        # Unweighted 0.1533 stays the headline; weighted is far lower.
        assert "0.1533" in line
        assert "0.1271" in line, line

    def test_shares_and_kwh_give_the_same_weighted_figure(self):
        """Only relative weights matter, so the unit the user chose is irrelevant."""
        as_kwh = _run(
            _rows(2022, range(1, 13), 0.145, SEASONAL_BENCHMARK), profile=SUMMER_HEAVY
        )
        as_shares = _run(
            _rows(2022, range(1, 13), 0.145, SEASONAL_BENCHMARK),
            profile={month: value / 1400.0 for month, value in SUMMER_HEAVY.items()},
        )

        def weighted(text):
            token = text.split("consumption-weighted=")[1]
            return token.split()[0]

        assert weighted(as_kwh) == weighted(as_shares)

    def test_the_weighted_spread_is_reported_against_the_same_tariff(self):
        stats = _run(
            _rows(2022, range(1, 13), 0.145, SEASONAL_BENCHMARK), profile=SUMMER_HEAVY
        )
        line = _year_line(stats)

        # Regulated 0.1450 against a weighted 0.1271 -> regulated is DEARER for
        # this consumer. The unweighted spread is -0.0083 and says the opposite,
        # so the load shape does not merely shade the answer, it reverses it.
        assert "+0.0179" in line, line

    def test_stage_3_enrichment_produces_the_weighted_figure_end_to_end(self):
        """The app must actually do this, not just the helper in isolation.

        ``enrich`` is the Stage 3 entry point that builds stats_hint, so a
        question carrying a load shape has to come out of THERE with the weighted
        figure -- otherwise the contract field is populated and nothing uses it.
        """
        from agent.analyzer import enrich

        ctx = _ctx(
            _rows(2022, range(1, 13), 0.145, SEASONAL_BENCHMARK), profile=SUMMER_HEAVY
        )
        ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False, name=None)]

        enrich(ctx)

        assert "consumption-weighted=0.1271" in ctx.stats_hint
        assert "+0.0179" in ctx.stats_hint
        # The unweighted basis survives beside it.
        assert "benchmark=0.1533" in ctx.stats_hint

    def test_a_profile_covering_only_some_months_says_so(self):
        """A partial profile weights only the months it names, which is not the year."""
        stats = _run(
            _rows(2022, range(1, 13), 0.145, SEASONAL_BENCHMARK),
            profile={6: 100.0, 7: 100.0},
        )
        line = _year_line(stats)

        assert "consumption-weighted" in line.lower()
        assert "2/12" in line, "the covered-month count has to be visible"
