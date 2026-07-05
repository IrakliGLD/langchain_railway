"""Unit tests for ``utils.resilience.CircuitBreaker`` recovery semantics.

Focus: the half-open probe must be consumable exactly once, an advisory
``would_allow()`` peek must never consume it, and ``record_success`` /
``record_failure`` must clear it. This is the combination that a prior unpaired
advisory ``allow_request()`` call in the pipeline broke — it claimed the probe
without ever recording an outcome, wedging recovery in ``half_open`` until the
process restarted.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import time  # noqa: E402

from utils.resilience import CircuitBreaker  # noqa: E402


def _tripped_breaker(reset_timeout: int = 30) -> CircuitBreaker:
    """A breaker driven straight to ``open`` (failure_threshold=1)."""
    cb = CircuitBreaker("test", failure_threshold=1, reset_timeout_seconds=reset_timeout)
    cb.record_failure()
    assert cb.state == "open"
    return cb


def _elapse_reset_window(cb: CircuitBreaker) -> None:
    """Simulate the reset timeout elapsing without sleeping."""
    cb._opened_at = time.time() - (cb.reset_timeout_seconds + 5)


def test_closed_allows_without_engaging_probe():
    cb = CircuitBreaker("t", failure_threshold=3, reset_timeout_seconds=30)
    assert cb.state == "closed"
    assert cb.allow_request() == (True, "closed")
    assert cb.would_allow() == (True, "closed")
    assert cb.snapshot()["half_open_probe_in_flight"] is False


def test_open_blocks_within_reset_window():
    cb = _tripped_breaker(reset_timeout=30)
    assert cb.allow_request() == (False, "open")
    assert cb.would_allow() == (False, "open")
    assert cb.state == "open"


def test_would_allow_does_not_consume_half_open_probe():
    cb = _tripped_breaker(reset_timeout=1)
    _elapse_reset_window(cb)
    # Advisory peek reports the probe would be granted, but must NOT claim it
    # and must NOT transition state — repeatable and side-effect free.
    assert cb.would_allow() == (True, "half_open_probe")
    assert cb.would_allow() == (True, "half_open_probe")
    assert cb.snapshot()["half_open_probe_in_flight"] is False
    assert cb.state == "open"
    # The real guarded caller can therefore still acquire the sole probe.
    assert cb.allow_request() == (True, "half_open_probe")
    assert cb.state == "half_open"
    assert cb.snapshot()["half_open_probe_in_flight"] is True


def test_half_open_probe_is_single_use_until_recorded():
    cb = _tripped_breaker(reset_timeout=1)
    _elapse_reset_window(cb)
    assert cb.allow_request() == (True, "half_open_probe")   # claims the probe
    assert cb.allow_request() == (False, "half_open_busy")   # no second probe
    assert cb.would_allow() == (False, "half_open_busy")     # advisory agrees


def test_record_success_closes_and_frees_probe():
    cb = _tripped_breaker(reset_timeout=1)
    _elapse_reset_window(cb)
    cb.allow_request()
    cb.record_success()
    assert cb.state == "closed"
    assert cb.snapshot()["half_open_probe_in_flight"] is False
    assert cb.allow_request() == (True, "closed")


def test_record_failure_reopens_from_half_open():
    cb = _tripped_breaker(reset_timeout=1)
    _elapse_reset_window(cb)
    cb.allow_request()
    cb.record_failure()
    assert cb.state == "open"
    assert cb.snapshot()["half_open_probe_in_flight"] is False


def test_advisory_peek_then_guarded_call_recovers_after_outage():
    """Regression: pipeline advisory-check + executor guarded-call sequence.

    Previously the advisory ``allow_request()`` consumed the probe and the
    executor's own ``allow_request()`` then saw ``half_open_busy`` forever. With
    ``would_allow()`` the executor still gets the probe and recovery completes.
    """
    cb = _tripped_breaker(reset_timeout=1)
    _elapse_reset_window(cb)
    advisory_allowed, _ = cb.would_allow()          # pipeline advisory peek
    assert advisory_allowed is True
    guarded_allowed, reason = cb.allow_request()    # executor guarded probe
    assert (guarded_allowed, reason) == (True, "half_open_probe")
    cb.record_success()                             # probe succeeds
    assert cb.state == "closed"
