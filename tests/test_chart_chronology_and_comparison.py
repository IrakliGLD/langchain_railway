"""Charts for a regulated-versus-wholesale question.

Two defects visible in the 2026-08-17 report screenshots:

1. Both charts read 2026-12 on the LEFT and 2021 on the right. The retail tool
   sorts DESC when no date filter is given (so a LIMIT captures recent months)
   and nothing in the chart path sorts chronologically -- the only
   ``sort_values(date_col)`` is inside ``calculate_trendline``, for line fitting.
   Series are therefore plotted in frame order, backwards.

2. Neither chart carried wholesale data. The frame holds
   ``wholesale_benchmark_gel_kwh`` beside ``supply_tariff_gel_kwh`` -- same unit,
   same time axis -- and the comparison the question asked for was drawn nowhere.
   The final-price line and the component stack both answer a different question,
   and the domain rule forbids comparing the FINAL price against wholesale at
   all: transmission and distribution are paid either way, roughly half the bar.
"""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.chart_pipeline import _prepare_chart_source, _retail_chart_groups
from models import QueryContext

SUPPLY = "supply_tariff_gel_kwh"
BENCHMARK = "wholesale_benchmark_gel_kwh"
SPREAD = "supply_vs_wholesale_spread_gel_kwh"


def _retail_frame(*, months: int = 30, with_benchmark: bool = True, newest_first: bool = True):
    """One category, monthly, ordered the way the tool returns it."""
    dates = list(pd.date_range("2024-01-01", periods=months, freq="MS"))
    if newest_first:
        dates = dates[::-1]
    rows = []
    for index, date in enumerate(dates):
        supply = 0.145 + 0.0005 * index
        row = {
            "date": date.strftime("%Y-%m-%d"),
            "supplier": "telmico",
            "category": "3.3-6-10|com|other",
            "series_label": "Telmico - Commercial - other (3.3-6-10)",
            "transmission_tariff_gel_kwh": 0.0067,
            "distribution_tariff_gel_kwh": 0.0812,
            SUPPLY: supply,
            "final_price_net_gel_kwh": 0.0067 + 0.0812 + supply,
        }
        if with_benchmark:
            row[BENCHMARK] = 0.147
            row[SPREAD] = supply - 0.147
        rows.append(row)
    return pd.DataFrame(rows)


def _ctx(df) -> QueryContext:
    return QueryContext(
        query="is the regulated supply tariff cheaper than the wholesale market",
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
    )


class TestTheTimeAxisReadsForwards:
    def test_a_newest_first_frame_is_charted_oldest_first(self):
        """The screenshots read 2026-12 -> 2021, left to right."""
        df, time_key, _labels, _cats, _nums = _prepare_chart_source(
            _ctx(_retail_frame(newest_first=True))
        )

        assert time_key == "date"
        periods = pd.to_datetime(df[time_key])
        assert periods.is_monotonic_increasing, (
            f"chart source still runs backwards: {periods.iloc[0]} -> {periods.iloc[-1]}"
        )

    def test_an_already_ascending_frame_is_unchanged(self):
        df, time_key, _labels, _cats, _nums = _prepare_chart_source(
            _ctx(_retail_frame(newest_first=False))
        )

        periods = pd.to_datetime(df[time_key])
        assert periods.is_monotonic_increasing
        assert len(periods) == 30


class TestTheComparisonIsActuallyDrawn:
    def test_a_make_or_buy_frame_leads_with_supply_against_the_benchmark(self):
        """The comparison chart comes FIRST: it is the question that was asked."""
        groups = _retail_chart_groups(_ctx(_retail_frame()))

        assert groups
        first = groups[0]
        assert set(first["metrics"]) == {SUPPLY, BENCHMARK}
        assert first["type"] == "line"

    def test_the_comparison_never_uses_the_final_price(self):
        """Comparing the final price overstates the gap by the network stack.

        Both the knowledge file and the retail rules forbid it, and the stacked
        bar in the report shows why: transmission plus distribution is about half
        the bar.
        """
        groups = _retail_chart_groups(_ctx(_retail_frame()))

        comparison = groups[0]
        assert "final_price_net_gel_kwh" not in comparison["metrics"]
        assert "total_gross_gel_kwh" not in comparison["metrics"]

    def test_the_spread_gets_its_own_panel(self):
        """0.145 against 0.147 on a zero-baseline axis is one line.

        The gap is under 1.5% of the level -- the report's own axis runs 0 to
        0.30 with the values at 0.22-0.26 -- so the sign change has to be
        readable somewhere, and that is a panel of its own.
        """
        groups = _retail_chart_groups(_ctx(_retail_frame()))

        spread_groups = [g for g in groups if SPREAD in g.get("metrics", [])]
        assert len(spread_groups) == 1
        assert spread_groups[0]["metrics"] == [SPREAD]

    def test_the_comparison_asks_the_renderer_not_to_start_at_zero(self):
        """A zero baseline hides the quantity the chart exists to show.

        The gap is a percent or two of the level, so on a 0-0.30 axis the two
        lines nearly coincide. ``MyChartComponent`` hard-codes
        ``beginAtZero: true`` on its y-scales, so the backend has to ask for the
        zoom explicitly -- and the flag has to reach ``chart_meta``, which is the
        only thing that component reads. A flag that stops at the group dict
        would be read by nothing.
        """
        groups = _retail_chart_groups(_ctx(_retail_frame()))

        assert groups[0]["_y_begin_at_zero"] is False
        # The spread panel crosses zero, so zero must stay on that axis.
        spread = next(g for g in groups if SPREAD in g["metrics"])
        assert spread.get("_y_begin_at_zero") is not False

    def test_the_flag_reaches_chart_meta(self):
        """Not just the group dict: chart_meta is what the renderer consumes."""
        from agent.chart_pipeline import build_chart

        ctx = build_chart(_ctx(_retail_frame()))

        assert ctx.chart_meta is not None
        charts = (ctx.chart_meta or {}).get("charts") or []
        metas = [c.get("chartMeta", c) for c in charts] if charts else [ctx.chart_meta]
        assert any(m.get("yBeginAtZero") is False for m in metas), (
            f"no chart asked for a non-zero baseline: {[sorted(m) for m in metas]}"
        )

    def test_the_redundant_final_price_line_is_dropped_on_a_comparison(self):
        """The component stack already carries the total, and the question is
        not about the total."""
        groups = _retail_chart_groups(_ctx(_retail_frame()))

        headline_only = [
            g for g in groups if g.get("metrics") == ["final_price_net_gel_kwh"]
        ]
        assert not headline_only, "final-price line is redundant here"
        # The component stack stays: it shows why only supply is comparable.
        assert any(g.get("_component_composition") for g in groups)

class TestAClarifyShapedAnswerFromDataStillCharts:
    """The same stale premise Phase 5 fixed for vector retrieval, in the chart path.

    ``answer_kind=clarify`` suppresses charts entirely. But retail routing answers
    clarify-shaped retail questions FROM DATA -- the domain owner's "answer
    generally, then offer to narrow" rule -- and the analyzer emitted clarify on
    both traces on record. So the comparison chart can vanish on a rerun for a
    reason that has nothing to do with charting.
    """

    @staticmethod
    def _analysis(answer_kind: str):
        from tests.test_guardrails import _make_chart_stage_question_analysis

        return _make_chart_stage_question_analysis(
            answer_kind=answer_kind,
            primary_presentation=None,
            chart_recommended=True,
        )

    def test_clarify_still_charts_when_the_answer_came_from_data(self):
        from visualization.chart_selector import should_generate_chart

        assert should_generate_chart(
            "is the regulated supply tariff cheaper than wholesale",
            66,
            response_mode="data_primary",
            question_analysis=self._analysis("clarify"),
        ) is True

    def test_clarify_still_suppresses_a_chart_outside_the_data_path(self):
        from visualization.chart_selector import should_generate_chart

        assert should_generate_chart(
            "which category did you mean",
            0,
            question_analysis=self._analysis("clarify"),
        ) is False

    def test_a_knowledge_answer_is_not_rescued(self):
        """A knowledge answer really has no data to chart; only clarify is stale."""
        from visualization.chart_selector import should_generate_chart

        assert should_generate_chart(
            "what is a supply tariff",
            10,
            response_mode="data_primary",
            question_analysis=self._analysis("knowledge"),
        ) is False


class TestFramesWithoutABenchmark:
    def test_a_multi_series_frame_keeps_the_widened_headline_chart(self):
        """Documented limitation, pinned so it stays deliberate.

        The multi-series path widens the frame by pivoting on the headline column
        alone, which leaves no benchmark column to overlay. A per-series
        comparison would need that pivot to carry two metrics, so for now a
        multi-series frame keeps its existing chart even when the benchmark is
        present.
        """
        first = _retail_frame()
        second = _retail_frame()
        second["category"] = "35-110|com|other"
        second["series_label"] = "Telmico - Commercial - other (35-110)"
        groups = _retail_chart_groups(_ctx(pd.concat([first, second], ignore_index=True)))

        assert groups
        assert not any(BENCHMARK in g.get("metrics", []) for g in groups)
        assert not any(SPREAD in g.get("metrics", []) for g in groups)

    def test_the_chart_column_names_are_the_ones_the_tool_emits(self, monkeypatch):
        """Drift guard: a rename elsewhere silently stops the chart being drawn.

        Three consumers share these spellings now -- the annual block, these
        charts, and the frame adapter -- so bind them to the tool's real output
        rather than trusting three copies to stay equal.
        """
        import agent.tools.end_user_price_tools as tool_module
        from tests.test_end_user_price_tool import _stub_rows

        monkeypatch.setattr(
            tool_module,
            "run_text_query",
            _stub_rows(
                wholesale_benchmark_gel_kwh=0.1470,
                supply_vs_wholesale_spread_gel_kwh=-0.0366,
            ),
        )
        _, cols, _ = tool_module.get_end_user_prices(include_wholesale_benchmark=True)

        assert tool_module.MAKE_OR_BUY_TARIFF_COLUMN in cols
        assert tool_module.MAKE_OR_BUY_BENCHMARK_COLUMN in cols
        assert tool_module.MAKE_OR_BUY_SPREAD_COLUMN in cols

    def test_a_frame_without_a_benchmark_keeps_the_old_charts(self):
        """No comparison to draw -> unchanged behaviour."""
        groups = _retail_chart_groups(_ctx(_retail_frame(with_benchmark=False)))

        assert groups
        assert groups[0]["metrics"] == ["final_price_net_gel_kwh"]
        assert not any(SPREAD in g.get("metrics", []) for g in groups)
