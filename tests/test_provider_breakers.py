"""Tests for P5.4 provider configuration and breaker ownership (finding M11).

Three defects closed: NVIDIA aliased to the OpenAI circuit breaker (shared
open/reset state), OpenAI-primary startup skipped credential validation, and a
Gemini stage-model override silently swapped a non-Gemini deployment onto
Gemini.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from config import validate_runtime_settings
from core import llm as core_llm
from utils.resilience import CircuitBreaker, _llm_breakers, get_llm_breaker

_VALIDATE_BASE = dict(
    supabase_db_url="postgresql://user:pass@localhost/db",
    gateway_shared_secret="gateway",
    session_signing_secret="session",
    evaluate_admin_secret="evaluate",
    auth_mode="gateway_only",
    deployment_env="production",
    supabase_jwt_secret=None,
    enable_evaluate_endpoint=False,
    allow_evaluate_endpoint=False,
)


# ---------------------------------------------------------------------------
# Independent breaker per provider
# ---------------------------------------------------------------------------


class TestBreakerOwnership:
    def test_every_registry_provider_has_its_own_breaker(self):
        # Registry/breaker alignment: a provider added to _PROVIDERS must get
        # its own breaker instead of falling through to OpenAI's.
        assert set(_llm_breakers) >= set(core_llm._PROVIDERS)

    def test_nvidia_breaker_is_distinct_from_openai(self):
        nvidia = get_llm_breaker("nvidia")
        openai = get_llm_breaker("openai")
        gemini = get_llm_breaker("gemini")
        assert nvidia is not openai
        assert nvidia is not gemini
        assert nvidia.name == "llm_nvidia"

    def test_unknown_provider_still_defaults_to_openai(self):
        assert get_llm_breaker("something-else") is get_llm_breaker("openai")

    def test_breaker_state_is_independent(self):
        # Fresh instances (not the module singletons) so no global pollution.
        a = CircuitBreaker(name="llm_a", failure_threshold=2, reset_timeout_seconds=60)
        b = CircuitBreaker(name="llm_b", failure_threshold=2, reset_timeout_seconds=60)
        a.record_failure()
        a.record_failure()
        assert a.state == "open"
        allowed, _reason = b.allow_request()
        assert allowed is True
        assert b.state == "closed"


# ---------------------------------------------------------------------------
# Startup credential validation for every selected provider
# ---------------------------------------------------------------------------


class TestStartupValidation:
    def test_openai_primary_requires_key(self):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            validate_runtime_settings(
                model_type="openai",
                google_api_key=None,
                openai_api_key=None,
                **_VALIDATE_BASE,
            )

    def test_openai_primary_passes_with_key(self):
        validate_runtime_settings(
            model_type="openai",
            google_api_key=None,
            openai_api_key="k",
            **_VALIDATE_BASE,
        )

    def test_gemini_primary_does_not_require_openai_key(self):
        validate_runtime_settings(
            model_type="gemini",
            google_api_key="g",
            openai_api_key=None,
            **_VALIDATE_BASE,
        )


# ---------------------------------------------------------------------------
# Stage-model overrides never swap the provider
# ---------------------------------------------------------------------------


class _Sentinel:
    pass


class TestStageOverrideProviderHonesty:
    def test_non_gemini_primary_ignores_gemini_stage_override(self, monkeypatch):
        primary = _Sentinel()
        monkeypatch.setattr(core_llm, "MODEL_TYPE", "openai")
        monkeypatch.setattr(core_llm, "GOOGLE_API_KEY", "present-but-irrelevant")
        monkeypatch.setattr(core_llm, "get_primary_llm", lambda: primary)

        def _boom(**kwargs):
            raise AssertionError("Gemini client must not be constructed for a non-Gemini primary")
        monkeypatch.setattr(core_llm, "ChatGoogleGenerativeAI", _boom)

        result = core_llm.get_llm_for_stage(stage_model="gemini-2.5-pro")
        assert result is primary

    def test_gemini_primary_still_honors_stage_override(self, monkeypatch):
        constructed = {}

        class _FakeGemini:
            def __init__(self, **kwargs):
                constructed.update(kwargs)

        monkeypatch.setattr(core_llm, "MODEL_TYPE", "gemini")
        monkeypatch.setattr(core_llm, "GOOGLE_API_KEY", "g-key")
        monkeypatch.setattr(core_llm, "ChatGoogleGenerativeAI", _FakeGemini)
        monkeypatch.setattr(core_llm, "_stage_model_cache", {})

        result = core_llm.get_llm_for_stage(stage_model="gemini-2.5-pro")
        assert isinstance(result, _FakeGemini)
        assert constructed["model"] == "gemini-2.5-pro"

    def test_dedicated_override_branch_still_guards_non_gemini(self, monkeypatch):
        primary = _Sentinel()
        monkeypatch.setattr(core_llm, "MODEL_TYPE", "nvidia")
        monkeypatch.setattr(core_llm, "GOOGLE_API_KEY", "present")
        monkeypatch.setattr(core_llm, "get_primary_llm", lambda: primary)

        result = core_llm.get_llm_for_stage(stage_model="gemini-2.5-pro", thinking_budget=512)
        assert result is primary


# ---------------------------------------------------------------------------
# P5.1 (H13): OpenAI now has an explicit, bounded per-call timeout
# ---------------------------------------------------------------------------


class TestOpenAITimeout:
    def _patch_client(self, monkeypatch):
        from core import llm_runtime

        captured = {}

        class _FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(llm_runtime, "ChatOpenAI", _FakeChatOpenAI)
        monkeypatch.setattr(llm_runtime, "OPENAI_API_KEY", "k")
        monkeypatch.setattr(llm_runtime, "_openai_llm", None)
        return llm_runtime, captured

    def test_timeout_bounds_call_and_drops_retries(self, monkeypatch):
        llm_runtime, captured = self._patch_client(monkeypatch)
        monkeypatch.setattr(llm_runtime, "OPENAI_TIMEOUT_SECONDS", 120.0)

        llm_runtime.get_openai()

        assert captured["request_timeout"] == 120.0
        assert captured["max_retries"] == 0

    def test_unbounded_when_timeout_disabled(self, monkeypatch):
        llm_runtime, captured = self._patch_client(monkeypatch)
        monkeypatch.setattr(llm_runtime, "OPENAI_TIMEOUT_SECONDS", None)

        llm_runtime.get_openai()

        assert "request_timeout" not in captured
        assert captured["max_retries"] == 0

    def test_default_timeout_is_bounded(self):
        import config

        assert config.OPENAI_TIMEOUT_SECONDS == 120.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
