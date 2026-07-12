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

from core.llm_runtime import LLMResponseCache


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
