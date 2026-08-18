"""A slow analyzer must not spend the summarizer's budget.

Incident 2026-08-17, requests 4 and 5. Budget 115,000 ms. Stage 0.2 took 41.5 s
and 36.1 s, leaving 72.2 s and 77.5 s; the per-call timeout is clamped to
``remaining - REQUEST_CLEANUP_ALLOWANCE_MS`` by
``RequestDeadline.bounded_timeout_seconds``, so Stage 4 got 69.2 s and 74.5 s
for a 22.4k-token prompt at reasoning_effort=high. Both timed out at exactly
those figures and shipped a 77-character "temporarily unavailable" string after
a 112-second wait. Nothing overspent -- the split was simply never governed.

Two properties are pinned here:

1. Stage 4 keeps a reserve, so an analyzer overrun is bounded and visible
   rather than silently charged to it. (Expressed as the summarizer's reserve
   rather than the analyzer's share -- the share form regressed production;
   tests/test_analyzer_deadline_floor.py records why.)
2. When Stage 4 starts with less time than its prompt plausibly needs, it sheds
   prompt rather than starting a call that cannot finish. A shorter answer
   beats a 112-second failure.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import config  # noqa: E402
import core.llm as llm_core  # noqa: E402
from utils.request_deadline import (  # noqa: E402
    RequestDeadline,
    bind_request_execution_scope,
)


def _deadline(budget_ms: int) -> RequestDeadline:
    return RequestDeadline.from_budget_ms(budget_ms=budget_ms, source="test")


def test_the_split_between_the_stages_is_configured():
    """An unstated split is what let Stage 0.2 take 41s of a 115s budget."""
    assert config.SUMMARIZER_DEADLINE_RESERVE_MS > 0
    assert 0.0 < config.ANALYZER_MINIMUM_DEADLINE_SHARE < 1.0


def test_analyzer_call_is_bounded_by_stage_4s_reserve():
    """A 115s budget must not hand Stage 0.2 the whole thing.

    Bounded by what Stage 4 needs, not by a fraction of the budget -- see
    tests/test_analyzer_deadline_floor.py for why the fraction was wrong.
    """
    with bind_request_execution_scope(deadline=_deadline(115_000)):
        seconds = llm_core._analyzer_timeout_seconds(configured_seconds=120.0)

    assert seconds > 0
    assert seconds < 115.0
    assert 115_000 - seconds * 1000 >= config.SUMMARIZER_DEADLINE_RESERVE_MS * 0.95


def test_analyzer_cap_never_exceeds_what_the_request_has_left():
    """Late in a request the share is irrelevant; the remaining budget rules."""
    with bind_request_execution_scope(deadline=_deadline(4_000)):
        seconds = llm_core._analyzer_timeout_seconds(configured_seconds=120.0)

    assert seconds <= 4.0


def test_analyzer_cap_is_inert_without_a_deadline():
    """Offline and test entry points have no request scope; keep them working."""
    assert llm_core._analyzer_timeout_seconds(configured_seconds=120.0) == 120.0


def test_summarizer_budget_shrinks_when_little_time_is_left():
    """The shed is the difference between a short answer and no answer."""
    roomy = llm_core._deadline_aware_summarizer_budget(
        configured_budget=90_000, remaining_ms=110_000
    )
    cramped = llm_core._deadline_aware_summarizer_budget(
        configured_budget=90_000, remaining_ms=30_000
    )

    assert roomy == 90_000
    assert cramped < 90_000
    assert cramped >= 1_500


def test_calibration_against_the_incident_requests():
    """The constant has to shed the two that failed and spare the three that did not.

    Measured on 2026-08-17, (prompt chars, ms remaining when Stage 4 started):
    answered  62,830 / 72,586 · 73,495 / 89,279 · 75,639 / 87,977
    timed out 88,364 / 72,231 · 89,664 / 77,517

    A budget that leaves the failures untouched fixes nothing; one that trims
    the successes costs depth on turns that were fine.
    """
    configured = 98_500  # the effective production ceiling at the time

    for prompt_chars, remaining_ms in ((62_830, 72_586), (73_495, 89_279), (75_639, 87_977)):
        budget = llm_core._deadline_aware_summarizer_budget(
            configured_budget=configured, remaining_ms=remaining_ms
        )
        assert budget >= prompt_chars, f"{prompt_chars} would have been trimmed needlessly"

    for prompt_chars, remaining_ms in ((88_364, 72_231), (89_664, 77_517)):
        budget = llm_core._deadline_aware_summarizer_budget(
            configured_budget=configured, remaining_ms=remaining_ms
        )
        assert budget < prompt_chars, f"{prompt_chars} would still have timed out"


def test_summarizer_budget_is_never_raised_above_what_was_configured():
    generous = llm_core._deadline_aware_summarizer_budget(
        configured_budget=45_000, remaining_ms=300_000
    )

    assert generous == 45_000


def test_summarizer_budget_is_unchanged_without_a_deadline():
    """remaining_ms=-1 is the no-scope sentinel used by the caller."""
    assert (
        llm_core._deadline_aware_summarizer_budget(
            configured_budget=45_000, remaining_ms=-1
        )
        == 45_000
    )


class _DummyCache:
    def get(self, _key):
        return None

    def set(self, _key, _value):
        return None


class _DummyMessage:
    content = '{"answer":"ok","claims":[],"citations":[],"confidence":0.9}'
    response_metadata: dict = {}


def test_a_cramped_stage_4_actually_ships_a_smaller_prompt(monkeypatch):
    """End to end: the incident's remaining budget must shrink the prompt."""
    captured: dict = {}

    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_a, **_k: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_a, **_k: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_a, **_k: None)

    def _capture(_llm, messages, *_args, **_kwargs):
        captured.setdefault("prompts", []).append(messages[1][1])
        return _DummyMessage()

    monkeypatch.setattr(llm_core, "_invoke_at_stage", _capture)

    big_preview = "\n".join(
        f"2026-{month:02d}-01,{0.19 + month / 1000:.4f},{180 + month}" for month in range(1, 13)
    ) * 40
    kwargs = dict(
        user_query="Is retail cheaper than wholesale for a 35-100 kV customer?",
        data_preview="date,final_price_net_gel_kwh,p_bal_gel\n" + big_preview,
        stats_hint="--- ANNUAL MAKE-OR-BUY COMPARISON ---\n" + ("stat line\n" * 400),
        domain_knowledge="domain " * 2000,
        vector_knowledge="passage " * 2000,
    )

    with bind_request_execution_scope(deadline=_deadline(300_000)):
        llm_core.llm_summarize_structured(**kwargs)
    with bind_request_execution_scope(deadline=_deadline(25_000)):
        llm_core.llm_summarize_structured(**kwargs)

    roomy_prompt, cramped_prompt = captured["prompts"]
    assert len(cramped_prompt) < len(roomy_prompt)
