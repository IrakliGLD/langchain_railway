"""Concurrency regressions for the LLM response singleflight cache."""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import core.llm_runtime as llm_runtime
from core.llm_runtime import LLMResponseCache
from utils.request_deadline import RequestDeadline, bind_request_execution_scope


def test_follower_reuses_leader_result():
    cache = LLMResponseCache(coalesce_timeout=0.2)
    value, leader_token = cache.get_or_reserve("prompt")
    assert value is None
    assert leader_token is not None

    observed = []

    def follower():
        observed.append(cache.get_or_reserve("prompt"))

    thread = threading.Thread(target=follower)
    thread.start()
    time.sleep(0.02)
    assert cache.set("prompt", "answer", token=leader_token)
    thread.join(timeout=1)

    assert observed == [("answer", None)]
    assert cache.stats()["in_flight"] == 0


def test_stale_leader_cannot_overwrite_replacement_owner():
    cache = LLMResponseCache(coalesce_timeout=0.02)
    _value, stale_token = cache.get_or_reserve("prompt")
    time.sleep(0.03)

    value, replacement_token = cache.get_or_reserve("prompt")

    assert value is None
    assert replacement_token is not None
    assert replacement_token is not stale_token
    assert cache.set("prompt", "stale", token=stale_token) is False
    assert cache.set("prompt", "fresh", token=replacement_token) is True
    assert cache.get("prompt") == "fresh"


def test_stale_cancel_does_not_cancel_replacement_owner():
    cache = LLMResponseCache(coalesce_timeout=0.02)
    _value, stale_token = cache.get_or_reserve("prompt")
    time.sleep(0.03)
    _value, replacement_token = cache.get_or_reserve("prompt")

    assert cache.cancel_in_flight("prompt", token=stale_token) is False
    assert cache.stats()["in_flight"] == 1
    assert cache.cancel_in_flight("prompt", token=replacement_token) is True
    assert cache.stats()["in_flight"] == 0


def test_staleness_is_measured_from_the_leader_reservation():
    """A late follower waits only for the unused part of the owner's lease."""
    cache = LLMResponseCache(coalesce_timeout=0.2)
    _value, stale_token = cache.get_or_reserve("prompt")
    time.sleep(0.15)

    started = time.monotonic()
    value, replacement_token = cache.get_or_reserve("prompt")
    elapsed = time.monotonic() - started

    assert value is None
    assert replacement_token is not None
    assert replacement_token is not stale_token
    assert elapsed < 0.12
    assert cache.cancel_in_flight("prompt", token=replacement_token)


def test_legacy_get_wait_is_measured_from_the_leader_reservation():
    cache = LLMResponseCache(coalesce_timeout=0.2)
    assert cache.mark_in_flight("prompt")
    time.sleep(0.15)

    started = time.monotonic()
    assert cache.get("prompt") is None
    elapsed = time.monotonic() - started

    assert elapsed < 0.12
    assert cache.cancel_in_flight("prompt")


def test_wait_preserves_the_callers_execution_budget(monkeypatch):
    """Coalescing must not consume time explicitly reserved for the next stage."""
    monkeypatch.setattr(llm_runtime, "REQUEST_CLEANUP_ALLOWANCE_MS", 10)
    monkeypatch.setattr(llm_runtime, "PROVIDER_MINIMUM_START_BUDGET_MS", 10)
    cache = LLMResponseCache(coalesce_timeout=1.0)
    _value, stale_token = cache.get_or_reserve("prompt")
    deadline = RequestDeadline.from_budget_ms(budget_ms=250, source="test")

    started = time.monotonic()
    with bind_request_execution_scope(deadline=deadline):
        value, replacement_token = cache.get_or_reserve(
            "prompt", minimum_remaining_ms=180
        )
    elapsed = time.monotonic() - started

    assert value is None
    assert replacement_token is not None
    assert replacement_token is not stale_token
    assert elapsed < 0.14
    assert deadline.remaining_ms() >= 140
    assert cache.cancel_in_flight("prompt", token=replacement_token)
