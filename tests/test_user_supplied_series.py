"""A pasted profile is evidence, not decoration.

Incident 2026-08-17: a user pasted twelve months of consumption and asked
whether retail or wholesale was cheaper for them. Nothing in the pipeline could
carry those figures -- ``derived_metrics`` came back empty, the scenario
fallback came back empty, and the numbers reached neither the statistics nor
the grounding corpus. The answer could only ignore them.

These tests pin a deterministic extractor: the figures the user typed become a
typed series in ``stats_hint`` and in the provenance corpus, so an answer may
cite them and the grounding gate accepts the citation.
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

from agent.user_supplied_series import (  # noqa: E402
    UserSuppliedSeries,
    extract_user_supplied_series,
)
from models import QueryContext  # noqa: E402

GEORGIAN_PROFILE = (
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


def test_extracts_every_month_of_a_georgian_profile():
    series = extract_user_supplied_series(GEORGIAN_PROFILE)

    assert series is not None
    assert len(series.points) == 12
    assert series.points[0].period == "01"
    assert series.points[0].value == 412000.0
    assert series.points[11].period == "12"
    assert series.points[11].value == 431500.0


def test_total_is_computed_deterministically():
    """The model must never have to add twelve numbers itself."""
    series = extract_user_supplied_series(GEORGIAN_PROFILE)

    assert series.total == 4_548_000.0
    assert series.point_count == 12


def test_extracts_english_month_names():
    series = extract_user_supplied_series(
        "My monthly consumption:\n"
        "January 1200\nFebruary 1100\nMarch 1350\nApril 990\nMay 1010\nJune 875"
    )

    assert [p.period for p in series.points] == ["01", "02", "03", "04", "05", "06"]
    assert series.points[2].value == 1350.0


def test_extracts_iso_periods_and_thousands_separators():
    series = extract_user_supplied_series(
        "2025-01: 412,000 kWh\n2025-02: 389 500 kWh\n2025-03: 401200 kWh\n2025-04: 355800"
    )

    assert [p.period for p in series.points] == ["2025-01", "2025-02", "2025-03", "2025-04"]
    assert [p.value for p in series.points] == [412000.0, 389500.0, 401200.0, 355800.0]


def test_decimal_values_survive():
    series = extract_user_supplied_series(
        "jan 10.5\nfeb 11.25\nmar 9.75\napr 12.0"
    )

    assert [p.value for p in series.points] == [10.5, 11.25, 9.75, 12.0]


def test_ordinary_prose_is_not_mistaken_for_a_series():
    """Two stray numbers in a sentence must not become an evidence record."""
    assert extract_user_supplied_series(
        "Why did the balancing price change between 2024 and 2025?"
    ) is None
    assert extract_user_supplied_series(
        "Is the retail price cheaper than wholesale for a 6-10 kV customer?"
    ) is None
    assert extract_user_supplied_series("") is None


def test_multi_line_prose_about_months_is_not_a_series():
    """A report track query carries months and figures in its coverage bullets.

    Those bullets describe the surrounding work; reading them as a profile the
    user supplied would invent evidence out of a research brief.
    """
    assert extract_user_supplied_series(
        "How did the balancing price move through 2025?\n"
        "- In January 2025 the price rose sharply against a cold snap of 1200 GWh demand\n"
        "- By March 2025 imports had recovered to roughly 340 GWh across the region\n"
        "- Through June 2025 hydro output returned to its 900 GWh seasonal average\n"
        "- December 2025 closed the year near the 1100 GWh mark once again\n"
    ) is None


def test_a_year_beside_a_month_name_is_not_the_measurement():
    """"January 2025 412000" states a period of Jan-2025 and a value of 412000."""
    series = extract_user_supplied_series(
        "January 2025 412000\nFebruary 2025 389500\nMarch 2025 401200"
    )

    assert [p.value for p in series.points] == [412000.0, 389500.0, 401200.0]


def test_a_two_point_list_is_below_the_series_threshold():
    """Guard the false-positive edge: a series needs enough points to be one."""
    assert extract_user_supplied_series("January 100\nFebruary 200") is None


def test_duplicate_periods_keep_the_first_reading():
    series = extract_user_supplied_series(
        "jan 100\nfeb 200\njan 999\nmar 300\napr 400"
    )

    assert [p.period for p in series.points] == ["01", "02", "03", "04"]
    assert series.points[0].value == 100.0


def _ctx_with_query(query: str) -> QueryContext:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-07-01"]),
            "final_price_net_gel_kwh": [0.1987, 0.2013],
        }
    )
    return QueryContext(
        query=query,
        trace_id="user-series",
        session_id="user-series",
        preview=df.to_string(index=False),
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        used_tool=True,
        tool_name="get_end_user_prices",
    )


def test_series_is_attached_to_stats_hint_as_labelled_evidence():
    from agent.user_supplied_series import attach_user_supplied_series

    ctx = _ctx_with_query(GEORGIAN_PROFILE)
    attached = attach_user_supplied_series(ctx)

    assert attached is True
    assert "USER-SUPPLIED SERIES" in ctx.stats_hint
    for value in ("412000", "431500", "327400"):
        assert value in ctx.stats_hint
    assert "4548000" in ctx.stats_hint.replace(",", "")


def test_attached_series_enters_the_grounding_corpus():
    """A cited user figure must pass the grounding gate, not be repaired away."""
    from agent.summary_grounding import _build_grounding_corpus
    from agent.user_supplied_series import attach_user_supplied_series

    ctx = _ctx_with_query(GEORGIAN_PROFILE)
    attach_user_supplied_series(ctx)

    corpus = _build_grounding_corpus(ctx)
    for value in ("412000", "431500", "4548000"):
        assert value in corpus.replace(",", "")


def test_attaching_a_series_does_not_clobber_measured_provenance():
    """The pasted figures are extra evidence, never a replacement for the frame."""
    from agent.provenance import stamp_provenance
    from agent.user_supplied_series import attach_user_supplied_series

    ctx = _ctx_with_query(GEORGIAN_PROFILE)
    stamp_provenance(
        ctx,
        ctx.cols,
        ctx.rows,
        source="sql",
        query_hash="measured-frame",
    )

    attach_user_supplied_series(ctx)

    assert ctx.provenance_source == "sql"
    assert ctx.provenance_query_hash == "measured-frame"
    assert ctx.provenance_cols == ["date", "final_price_net_gel_kwh"]
    assert len(ctx.provenance_rows) == 2


def test_nothing_is_attached_when_there_is_no_series():
    from agent.user_supplied_series import attach_user_supplied_series

    ctx = _ctx_with_query("What was the balancing price in May 2024?")
    before = ctx.stats_hint

    assert attach_user_supplied_series(ctx) is False
    assert ctx.stats_hint == before


def test_stage_3_enrichment_attaches_a_pasted_profile():
    """The extractor is worthless unless Stage 3 actually runs it."""
    from agent import analyzer

    ctx = _ctx_with_query(GEORGIAN_PROFILE)
    enriched = analyzer.enrich(ctx)

    assert "USER-SUPPLIED SERIES" in enriched.stats_hint
    assert "431500" in enriched.stats_hint


def test_stage_3_enrichment_is_silent_without_a_pasted_profile():
    from agent import analyzer

    ctx = _ctx_with_query("What was the retail price in June 2026?")
    enriched = analyzer.enrich(ctx)

    assert "USER-SUPPLIED SERIES" not in (enriched.stats_hint or "")


def test_user_series_survives_statistics_compaction():
    """Compaction must not drop the one block the user typed themselves."""
    import core.llm as llm_core
    from agent.user_supplied_series import attach_user_supplied_series

    ctx = _ctx_with_query(GEORGIAN_PROFILE)
    # Enough filler to push the corpus past the 36k compaction cap, so the
    # priority ordering actually decides what survives.
    ctx.stats_hint = "\n\n".join(
        f"--- COLUMN AGGREGATES FILLER {n} ---\n" + ("x" * 2000) for n in range(25)
    )
    attach_user_supplied_series(ctx)
    assert len(ctx.stats_hint) > 36000

    compacted = llm_core._compact_summarizer_statistics(ctx.stats_hint)

    assert "USER-SUPPLIED SERIES" in compacted
    assert "431500" in compacted


def test_series_contract_rejects_an_empty_point_list():
    with pytest.raises(ValueError):
        UserSuppliedSeries(points=[], unit="")
