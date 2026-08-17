"""The annual make-or-buy comparison block.

A retail question that sets the regulated supply tariff against the wholesale
benchmark got, on the 2026-08-16 trace, exactly one comparison figure per
category: a mean over the whole 2021-2026 span. That average *by construction*
hides the year-to-year sign flips the question is about -- the domain owner's
own reading is 2022 regulated-cheaper turning into 2024 wholesale-cheaper.

These tests pin the per-year block that replaces it, and the truncation
priority that keeps it in the prompt.
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

from agent.analyzer import _append_annual_comparison
from models import QueryContext

BLOCK_HEADER = "--- ANNUAL MAKE-OR-BUY COMPARISON ---"


def _rows(year: int, months, supply: float, benchmark, series=("telmico", "3.3-6-10|com|other")):
    """One row per month for a single (supplier, category) series.

    ``benchmark`` may be a float or a per-month sequence; ``None`` in the
    sequence models the LEFT JOIN on price_with_usd finding no row.
    """
    supplier, category = series
    out = []
    for index, month in enumerate(months):
        bench = benchmark[index] if isinstance(benchmark, (list, tuple)) else benchmark
        out.append(
            {
                "date": pd.Timestamp(year=year, month=month, day=1),
                "supplier": supplier,
                "category": category,
                "supply_tariff_gel_kwh": supply,
                "wholesale_benchmark_gel_kwh": bench,
            }
        )
    return out


def _ctx(records) -> QueryContext:
    df = pd.DataFrame(records)
    return QueryContext(query="regulated vs wholesale", df=df, cols=list(df.columns))


def _run(records) -> str:
    ctx = _ctx(records)
    _append_annual_comparison(ctx)
    return ctx.stats_hint


def test_emits_one_line_per_year_in_both_unit_renderings():
    """GEL/kWh is the stored unit; GEL/MWh is the unit the comparison is discussed in.

    Both must be literal in the corpus. ``_add_rounded_source_variants`` rounds a
    token but never derives a x1000 form, so a corpus holding only 0.1450 does
    not make "145" quotable -- and 145 is the figure the domain owner uses.
    """
    stats = _run(
        _rows(2022, range(1, 13), 0.145, 0.147)
        + _rows(2023, range(1, 13), 0.160, 0.158)
    )

    assert BLOCK_HEADER in stats
    assert "2022" in stats and "2023" in stats
    # GEL/kWh, 4 dp.
    assert "0.1450" in stats and "0.1470" in stats
    # GEL/MWh, 1 dp.
    assert "145.0" in stats and "147.0" in stats


def test_verdict_names_the_cheaper_side_in_both_directions():
    """Spread = tariff - benchmark, so NEGATIVE means the regulated side is cheaper.

    An inverted verdict is the highest-consequence failure this block can have:
    it would tell someone facing a one-way, irreversible switch to move for a
    saving that does not exist. Both directions are pinned against the domain
    owner's own figures -- 2022 regulated 145 vs wholesale 147, 2024 regulated
    170 vs wholesale 168.
    """
    stats = _run(
        _rows(2022, range(1, 13), 0.145, 0.147)
        + _rows(2024, range(1, 13), 0.170, 0.168)
    )

    lines = {line.split()[0]: line for line in stats.splitlines() if line.strip()[:4].isdigit()}
    assert "regulated cheaper" in lines["2022"], lines["2022"]
    assert "wholesale cheaper" in lines["2024"], lines["2024"]
    # And the signed spread agrees with the verdict.
    assert "-0.0020" in lines["2022"] and "-2.0" in lines["2022"]
    assert "+0.0020" in lines["2024"] and "+2.0" in lines["2024"]


def test_a_year_with_no_gap_between_the_sides_is_reported_as_level():
    """Ambiguous case: neither side is cheaper. Do not round it into a verdict."""
    stats = _run(_rows(2022, range(1, 13), 0.147, 0.147))

    assert "level" in stats
    assert "cheaper" not in stats.split(BLOCK_HEADER)[1].split("Full years")[0]


def test_partial_year_is_marked_named_and_kept_out_of_the_tally():
    """Domain owner, 2026-08-17: use whatever is available, but report it.

    The month span is named, not just counted, because the wholesale side is
    seasonal and the tariff is not -- a Jul-Dec stub and a Jan-Jun stub are
    biased in opposite directions. The tally counts full years only, so a
    six-month stub cannot swing the headline.
    """
    stats = _run(
        _rows(2022, range(1, 13), 0.145, 0.147)
        + _rows(2023, range(1, 7), 0.160, 0.158)
    )

    lines = {line.split()[0]: line for line in stats.splitlines() if line.strip()[:4].isdigit()}
    assert "FULL" in lines["2022"]
    assert "PARTIAL" in lines["2023"]
    assert "6/12" in lines["2023"]
    assert "Jan-Jun" in lines["2023"]
    # 2023 is wholesale-cheaper but partial, so the tally sees one full year only.
    assert "Full years: 1" in stats
    assert "regulated cheaper in 1" in stats
    assert "wholesale cheaper in 0" in stats


def test_months_without_a_benchmark_leave_both_the_mean_and_the_coverage():
    """The benchmark arrives via LEFT JOIN, so a month can carry a tariff and no benchmark.

    Coverage is counted on PAIRED months. Averaging the tariff over 12 months
    against a benchmark over 6 would compare two different periods.
    """
    benchmark = [0.147] * 6 + [None] * 6
    stats = _run(_rows(2022, range(1, 13), 0.145, benchmark))

    line = next(line for line in stats.splitlines() if line.strip().startswith("2022"))
    assert "PARTIAL" in line
    assert "6/12" in line
    assert "Jan-Jun" in line
    # The tariff mean is over the paired months only, not all twelve.
    assert "0.1450" in line and "0.1470" in line


class TestTheWholesaleSideIsSplitBySeason:
    """A simple annual mean of the wholesale side is not what a consumer pays.

    The regulated stack is flat, so its mean IS its per-kWh price. The wholesale
    side is seasonal -- summer is hydro-dominant and cheap, winter is
    thermal/import-dominant and dear -- so an UNWEIGHTED annual mean is only what
    a consumer would pay if consumption were flat across the year. A summer-heavy
    consumer pays less than the mean suggests, a winter-heavy one more.

    ``seasonal_patterns.md`` already forbids the shortcut outright: "ALWAYS
    mention summer and winter averages separately when comparing prices -- never
    use annual averages only." The block reported annual means only.
    """

    @staticmethod
    def _seasonal_year():
        # Summer (Apr-Jul) cheap, winter dear. Months 1..12.
        benchmark = [0.170, 0.170, 0.170, 0.120, 0.120, 0.120, 0.120,
                     0.170, 0.170, 0.170, 0.170, 0.170]
        return _rows(2022, range(1, 13), 0.145, benchmark)

    def test_each_year_reports_the_benchmark_by_season(self):
        stats = _run(self._seasonal_year())

        line = next(line for line in stats.splitlines() if line.strip().startswith("2022"))
        assert "summer" in line and "winter" in line, line
        # Uses the repo's season definition: summer is months 4-7.
        assert "0.1200" in line, "summer mean of the four cheap months"
        assert "0.1700" in line, "winter mean of the eight dear months"

    def test_the_annual_benchmark_mean_is_labelled_unweighted(self):
        """Otherwise the model quotes it as what the consumer would have paid."""
        stats = _run(self._seasonal_year())

        assert "unweighted" in stats.lower()

    def test_the_regulated_side_is_never_split_by_season(self):
        """Administered prices have no season, and the rules forbid saying they do.

        Splitting the tariff invites exactly the summer-versus-winter comparison
        ``retail-tariff-rules.md`` bans for administered prices.
        """
        stats = _run(self._seasonal_year())

        line = next(line for line in stats.splitlines() if line.strip().startswith("2022"))
        # One seasonal pair only, on the benchmark.
        assert line.count("summer=") == 1
        assert line.count("winter=") == 1

    def test_a_year_with_no_summer_months_reports_no_summer_mean(self):
        """A Jan-Mar stub has no summer to average; inventing one would be a lie."""
        stats = _run(_rows(2023, range(1, 4), 0.145, 0.170))

        line = next(line for line in stats.splitlines() if line.strip().startswith("2023"))
        assert "PARTIAL" in line
        assert "summer=" not in line
        assert "winter=" in line


def test_no_block_at_all_without_the_benchmark_column():
    """This is the make-or-buy shape, not a general per-year facility."""
    records = _rows(2022, range(1, 13), 0.145, 0.147)
    for record in records:
        del record["wholesale_benchmark_gel_kwh"]

    assert _run(records) == ""


def test_multi_series_frames_are_labelled_and_never_pooled():
    stats = _run(
        _rows(2022, range(1, 13), 0.145, 0.147, series=("telmico", "3.3-6-10|com|other"))
        + _rows(2022, range(1, 13), 0.190, 0.147, series=("eps", "35-110|com|other"))
    )

    assert "telmico" in stats and "eps" in stats
    assert "0.1450" in stats and "0.1900" in stats
    # The two series disagree, so a pooled 0.1675 must appear nowhere.
    assert "0.1675" not in stats


def test_above_the_row_budget_it_degrades_to_a_cross_series_tally():
    """Too many series to enumerate is not a licence to pool them either."""
    records = []
    for index in range(14):
        for year in (2021, 2022, 2023, 2024, 2025):
            records += _rows(
                year, range(1, 13), 0.145, 0.147,
                series=(f"supplier{index}", f"category{index}"),
            )

    stats = _run(records)

    assert BLOCK_HEADER in stats
    # No per-series enumeration above the budget.
    assert "[supplier=supplier0" not in stats
    # But the per-year picture survives.
    assert "2021" in stats and "2025" in stats
    assert "70 series" in stats or "series" in stats


def test_the_trigger_columns_are_the_names_the_tool_actually_emits(monkeypatch):
    """Regression guard, not a TDD cycle: this passes today and must keep passing.

    The block is triggered by two hardcoded column names. If the tool ever
    renames either, the block stops emitting SILENTLY -- no error, no log, just
    the multi-year average back again. Binding the constants to the tool's real
    output turns that into a test failure instead.
    """
    import agent.tools.end_user_price_tools as tool_module
    from agent.analyzer import (
        _MAKE_OR_BUY_BENCHMARK_COLUMN,
        _MAKE_OR_BUY_TARIFF_COLUMN,
    )
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

    assert _MAKE_OR_BUY_TARIFF_COLUMN in cols
    assert _MAKE_OR_BUY_BENCHMARK_COLUMN in cols


def test_a_benchmark_column_that_is_entirely_null_emits_nothing():
    """Regression guard: the LEFT JOIN can miss every month.

    An empty comparison must produce no block at all rather than a block of
    zeroes or NaNs, which would enter the grounding corpus as quotable numbers.
    """
    stats = _run(_rows(2022, range(1, 13), 0.145, [None] * 12))

    assert stats == ""


def test_stage_3_enrichment_actually_emits_the_block():
    """Wiring: a correct function nobody calls fixes nothing.

    ``enrich`` is the Stage 3 entry point that builds stats_hint, so the block
    has to appear from there, not only when called directly.
    """
    from agent.analyzer import enrich

    records = (
        _rows(2022, range(1, 13), 0.145, 0.147)
        + _rows(2024, range(1, 13), 0.170, 0.168)
    )
    df = pd.DataFrame(records)
    ctx = QueryContext(
        query="should I move to the wholesale market",
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
    )

    enrich(ctx)

    assert BLOCK_HEADER in ctx.stats_hint
    assert "regulated cheaper" in ctx.stats_hint
    assert "wholesale cheaper" in ctx.stats_hint
    assert "145.0" in ctx.stats_hint and "168.0" in ctx.stats_hint


def test_the_block_outranks_column_aggregates_when_statistics_are_compacted():
    """Without a priority entry the section defaults to 10 -- below COLUMN
    AGGREGATES at 25 -- so it would be the first block shed, reproducing the
    very defect it exists to fix.
    """
    from core.llm import (
        _SUMMARIZER_STATS_SECTION_PRIORITY,
        _compact_summarizer_statistics,
        _summarizer_stats_priority,
    )

    assert _SUMMARIZER_STATS_SECTION_PRIORITY.get("ANNUAL MAKE-OR-BUY COMPARISON") == 88
    # The header the block actually emits must resolve to that priority.
    assert _summarizer_stats_priority(BLOCK_HEADER.strip("- ")) == 88

    annual = BLOCK_HEADER + "\n" + ("annual line\n" * 40)
    aggregates = "--- COLUMN AGGREGATES ---\n" + ("aggregate line\n" * 40)
    compacted = _compact_summarizer_statistics(
        annual + "\n" + aggregates, max_chars=len(annual) + 40
    )

    assert BLOCK_HEADER in compacted
    assert "COLUMN AGGREGATES" not in compacted
