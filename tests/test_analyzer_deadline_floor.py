"""Bounding Stage 0.2 must not make Stage 0.2 fail.

Regression from the 2026-08-17 deadline work, caught in production within
minutes of deploy (spans 5514de3c and c3e08d37). ANALYZER_DEADLINE_SHARE=0.35
against a 115,000 ms budget capped the analyzer at 40,250 ms -- but analyzer
calls that COMPLETED before the change took 41,017 ms and 41,473 ms. The cap
sat inside the distribution it was meant to bound, so both post-deploy calls
died at it (40,514 ms and 40,294 ms, disposition=timed_out).

Losing the analyzer is not a degraded answer, it is a different pipeline: no
contract, no evidence plan, fallback to legacy plan+SQL, and on these turns the
SQL relevance guard then blocked execution to zero rows. The user saw the model
argue about missing data.

The bound is therefore expressed the way the requirement actually reads --
Stage 4 keeps a reserve, the analyzer may use everything else -- never as a
fraction that can land below what the analyzer needs.
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


def test_stage_4_reserve_is_configured():
    assert config.SUMMARIZER_DEADLINE_RESERVE_MS > 0


def test_every_analyzer_call_seen_in_production_would_have_been_allowed():
    """The five real durations, against the real 115s budget."""
    observed_ms = [24_430, 25_832, 36_146, 41_017, 41_473]

    with bind_request_execution_scope(deadline=_deadline(115_000)):
        allowed = llm_core._analyzer_timeout_seconds(configured_seconds=120.0)

    for duration_ms in observed_ms:
        assert allowed * 1000 > duration_ms, (
            f"a {duration_ms} ms analyzer call would be cut off at {allowed * 1000:.0f} ms"
        )


def test_the_analyzer_is_still_bounded():
    """Still not permitted to spend the whole request -- that was the point."""
    with bind_request_execution_scope(deadline=_deadline(115_000)):
        allowed = llm_core._analyzer_timeout_seconds(configured_seconds=120.0)

    assert allowed < 115.0


def test_stage_4_keeps_its_reserve():
    budget_ms = 115_000
    with bind_request_execution_scope(deadline=_deadline(budget_ms)):
        allowed_ms = llm_core._analyzer_timeout_seconds(configured_seconds=120.0) * 1000

    assert budget_ms - allowed_ms >= config.SUMMARIZER_DEADLINE_RESERVE_MS * 0.95


def test_a_budget_too_small_to_reserve_never_starves_the_analyzer():
    """When no split is good, halving beats guaranteeing the analyzer dies."""
    with bind_request_execution_scope(deadline=_deadline(50_000)):
        allowed = llm_core._analyzer_timeout_seconds(configured_seconds=120.0)

    assert allowed >= 20.0


def test_a_generous_report_budget_is_left_to_the_configured_timeout():
    with bind_request_execution_scope(deadline=_deadline(600_000)):
        allowed = llm_core._analyzer_timeout_seconds(configured_seconds=120.0)

    assert allowed == 120.0


def test_cap_is_inert_without_a_request_scope():
    assert llm_core._analyzer_timeout_seconds(configured_seconds=120.0) == 120.0
