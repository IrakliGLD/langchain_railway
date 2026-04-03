"""Prompt-guidance regression tests for balancing summaries."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import core.llm as llm_core


def test_balancing_focus_loads_full_balancing_template(monkeypatch):
    captured = {}

    class _DummyCache:
        def get(self, _key):
            return None

        def set(self, _key, _value):
            return None

    class _DummyMessage:
        content = '{"answer":"ok","claims":[],"citations":[],"confidence":0.9}'
        response_metadata = {}

    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    def _capture_invoke(_llm, messages, _model_name):
        captured["prompt"] = messages[1][1]
        return _DummyMessage()

    monkeypatch.setattr(llm_core, "_invoke_with_resilience", _capture_invoke)

    llm_core.llm_summarize_structured(
        user_query="Why did balancing electricity price change in February 2022?",
        data_preview="date,p_bal_gel,price_deregulated_hydro_gel\n2022-01-01,183.8,45.0",
        stats_hint="Month-over-month change available.",
    )

    prompt = captured["prompt"]
    assert "Source Price / Tariff Layer" in prompt
    assert "price_deregulated_hydro_gel" in prompt
    assert "residual_contribution_ppa_import_*" in prompt
