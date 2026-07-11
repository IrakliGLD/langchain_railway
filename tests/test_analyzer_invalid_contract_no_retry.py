"""Invalid analyzer JSON must not trigger another long model call."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from core import llm


class _DummyCache:
    def get(self, _key):
        return None

    def mark_in_flight(self, _key):
        return None

    def cancel_in_flight(self, _key, token=None):
        return True


def test_invalid_json_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "llm_cache", _DummyCache())

    def invalid_response(*_args, **_kwargs):
        calls.append(1)
        return SimpleNamespace(content='{"broken":')

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invalid_response)

    with pytest.raises(ValueError):
        llm.llm_analyze_question("What was the balancing price in January 2024?")

    assert len(calls) == 1
