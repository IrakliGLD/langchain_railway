"""Even spacing must not cost the model the recent months.

``_compact_preview_by_period`` drops whole periods evenly rather than cutting
the middle out of the range -- that was the right call on 2026-08-16, when
head-and-tail truncation meant the years the question named never reached the
model at all. But even spacing has no notion of which end matters: on the
2026-08-17 make-or-buy turns a 66-period retail frame was thinned to fit 12,000
chars, and the months dropped were spread across the range including the recent
ones a twelve-month comparison depends on.

Both properties are wanted at once: the range stays represented, AND the recent
tail survives whole.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import core.llm as llm_core  # noqa: E402


def _monthly_preview(months: int, *, width: int = 90) -> str:
    """A period-led CSV preview spanning ``months`` consecutive months."""
    header = "date,final_price_net_gel_kwh,p_bal_gel"
    lines = []
    for index in range(months):
        year = 2021 + index // 12
        month = index % 12 + 1
        padding = "9" * max(0, width - 30)
        lines.append(f"{year}-{month:02d}-01,0.2013,183.8,{padding}")
    return "\n".join([header, *lines])


def _periods_in(preview: str) -> list[str]:
    return [
        line.split(",", 1)[0]
        for line in preview.splitlines()
        if llm_core._PREVIEW_PERIOD_RE.match(line.split(",", 1)[0])
    ]


def test_the_recent_twelve_periods_all_survive_compaction():
    """A twelve-month question is answered from the twelve most recent months."""
    preview = _monthly_preview(66)
    compacted = llm_core._compact_summarizer_preview(preview, max_chars=3_000)

    kept = _periods_in(compacted)
    expected_recent = _periods_in(preview)[-12:]

    assert len(kept) < len(_periods_in(preview)), "test must exercise compaction"
    for period in expected_recent:
        assert period in kept, f"recent period {period} was dropped"


def test_the_start_of_the_range_still_survives():
    """The 2026-08-16 property: the far end of the range stays represented."""
    preview = _monthly_preview(66)
    compacted = llm_core._compact_summarizer_preview(preview, max_chars=3_000)

    kept = _periods_in(compacted)
    assert kept[0] == _periods_in(preview)[0]


def test_kept_periods_stay_in_chronological_order():
    preview = _monthly_preview(66)
    compacted = llm_core._compact_summarizer_preview(preview, max_chars=3_000)

    kept = _periods_in(compacted)
    assert kept == sorted(kept)


def test_a_short_frame_is_untouched():
    preview = _monthly_preview(6)
    assert llm_core._compact_summarizer_preview(preview, max_chars=100_000) == preview


def test_a_budget_too_small_for_the_tail_still_returns_something():
    """Degrade, never explode, when even the recent tail will not fit."""
    preview = _monthly_preview(66)
    compacted = llm_core._compact_summarizer_preview(preview, max_chars=400)

    assert compacted
    assert len(compacted) <= 400
