"""Tests for env-controlled LLM provider selection (gemini / openai / nvidia).

Covers the generalized resolver added when NVIDIA (build.nvidia.com) was wired
in alongside Gemini and OpenAI: the provider classifier, per-provider cost
attribution, the MODEL_TYPE-driven resolvers, the OpenAI-fallback guard, and the
config validation for the new provider.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("NVIDIA_API_KEY", "test-nvidia-key")

import pytest

import config
from core import llm, llm_runtime


def test_provider_from_model_name_classifies_three_providers():
    assert llm._provider_from_model_name("openai/gpt-oss-120b") == "nvidia"
    assert llm._provider_from_model_name("meta/llama-3.1-8b") == "nvidia"
    assert llm._provider_from_model_name("google/gemma-4-31b-it") == "nvidia"
    assert llm._provider_from_model_name("gpt-4o-mini") == "openai"
    assert llm._provider_from_model_name("o1-preview") == "openai"
    assert llm._provider_from_model_name("gemini-2.5-flash") == "gemini"


def test_is_openai_model_name_excludes_namespaced_nvidia_id():
    # The NVIDIA model id starts with "openai/" but must NOT be treated as OpenAI.
    assert llm._is_openai_model_name("gpt-4o-mini") is True
    assert llm._is_openai_model_name("openai/gpt-oss-120b") is False


def test_provider_from_empty_name_uses_active_model_type(monkeypatch):
    monkeypatch.setattr(llm, "MODEL_TYPE", "nvidia")
    assert llm._provider_from_model_name("") == "nvidia"
    monkeypatch.setattr(llm, "MODEL_TYPE", "gemini")
    assert llm._provider_from_model_name("") == "gemini"


def test_estimate_cost_uses_nvidia_rates_for_nvidia_model(monkeypatch):
    monkeypatch.setattr(llm, "NVIDIA_INPUT_COST_PER_1K_USD", 2.0)
    monkeypatch.setattr(llm, "NVIDIA_OUTPUT_COST_PER_1K_USD", 4.0)
    cost = llm._estimate_cost_usd(1000, 1000, model_name="openai/gpt-oss-120b")
    assert abs(cost - 6.0) < 1e-9


def test_get_primary_model_name_follows_model_type(monkeypatch):
    monkeypatch.setattr(llm, "MODEL_TYPE", "nvidia")
    assert llm.get_primary_model_name() == config.NVIDIA_MODEL
    monkeypatch.setattr(llm, "MODEL_TYPE", "openai")
    assert llm.get_primary_model_name() == config.OPENAI_MODEL
    monkeypatch.setattr(llm, "MODEL_TYPE", "gemini")
    assert llm.get_primary_model_name() == config.GEMINI_MODEL


def test_get_primary_llm_dispatches_to_active_provider(monkeypatch):
    sentinel = {"nvidia": object(), "openai": object(), "gemini": object()}
    monkeypatch.setattr(llm, "make_nvidia", lambda: sentinel["nvidia"])
    monkeypatch.setattr(llm, "make_openai", lambda: sentinel["openai"])
    monkeypatch.setattr(llm, "make_gemini", lambda: sentinel["gemini"])

    monkeypatch.setattr(llm, "MODEL_TYPE", "nvidia")
    assert llm.get_primary_llm() is sentinel["nvidia"]
    monkeypatch.setattr(llm, "MODEL_TYPE", "openai")
    assert llm.get_primary_llm() is sentinel["openai"]
    monkeypatch.setattr(llm, "MODEL_TYPE", "gemini")
    assert llm.get_primary_llm() is sentinel["gemini"]


def test_nvidia_factory_builds_chatopenai_with_base_url(monkeypatch):
    # Make the test robust to suite import order: config may have been imported
    # by an earlier test file before NVIDIA_API_KEY was set, so pin the key on
    # the runtime module and reset the cached singleton.
    monkeypatch.setattr(llm_runtime, "NVIDIA_API_KEY", "test-nvidia-key")
    monkeypatch.setattr(llm_runtime, "_nvidia_llm", None)

    client = llm_runtime.get_nvidia()

    model = getattr(client, "model_name", None) or getattr(client, "model", None)
    assert model == config.NVIDIA_MODEL
    base = str(getattr(client, "openai_api_base", "") or getattr(client, "base_url", ""))
    assert "integrate.api.nvidia.com" in base
    # Env-configurable output cap + sampling temperature are applied.
    assert client.temperature == config.NVIDIA_TEMPERATURE
    assert client.max_tokens == config.NVIDIA_MAX_TOKENS


def test_nvidia_model_option_resolves_gemma_alias_with_defaults():
    model = config._resolve_nvidia_model("gemma-4-31b-it")
    assert model == "google/gemma-4-31b-it"
    assert config._nvidia_model_defaults(model) == {
        "max_tokens": 16384,
        "temperature": 1.0,
        "top_p": 0.95,
        "enable_thinking": True,
    }


def test_nvidia_api_key_resolver_prefers_model_family_specific_keys():
    assert config._resolve_nvidia_api_key(
        "google/gemma-4-31b-it",
        default_api_key="generic-key",
        openai_api_key="nvidia-openai-key",
        gemma_api_key="nvidia-gemma-key",
    ) == "nvidia-gemma-key"

    assert config._resolve_nvidia_api_key(
        "openai/gpt-oss-120b",
        default_api_key="generic-key",
        openai_api_key="nvidia-openai-key",
        gemma_api_key="nvidia-gemma-key",
    ) == "nvidia-openai-key"


def test_nvidia_api_key_resolver_keeps_generic_fallback():
    assert config._resolve_nvidia_api_key(
        "google/gemma-4-31b-it",
        default_api_key="generic-key",
        openai_api_key=None,
        gemma_api_key=None,
    ) == "generic-key"
    assert config._resolve_nvidia_api_key(
        "meta/llama-3.1-8b",
        default_api_key="generic-key",
        openai_api_key="nvidia-openai-key",
        gemma_api_key="nvidia-gemma-key",
    ) == "generic-key"


def test_nvidia_factory_applies_gemma_request_options(monkeypatch):
    monkeypatch.setattr(llm_runtime, "NVIDIA_API_KEY", "test-nvidia-key")
    monkeypatch.setattr(llm_runtime, "NVIDIA_MODEL", "google/gemma-4-31b-it")
    monkeypatch.setattr(llm_runtime, "NVIDIA_MAX_TOKENS", 16384)
    monkeypatch.setattr(llm_runtime, "NVIDIA_TEMPERATURE", 1.0)
    monkeypatch.setattr(llm_runtime, "NVIDIA_TOP_P", 0.95)
    monkeypatch.setattr(llm_runtime, "NVIDIA_CHAT_TEMPLATE_KWARGS", {"enable_thinking": True})
    monkeypatch.setattr(llm_runtime, "_nvidia_llm", None)

    client = llm_runtime.get_nvidia()

    model = getattr(client, "model_name", None) or getattr(client, "model", None)
    assert model == "google/gemma-4-31b-it"
    assert client.max_tokens == 16384
    assert client.temperature == 1.0

    model_kwargs = getattr(client, "model_kwargs", {}) or {}
    top_p = getattr(client, "top_p", None)
    assert (top_p if top_p is not None else model_kwargs.get("top_p")) == 0.95

    extra_body = getattr(client, "extra_body", None) or model_kwargs.get("extra_body")
    assert extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_nvidia_factory_requires_key(monkeypatch):
    monkeypatch.setattr(llm_runtime, "NVIDIA_API_KEY", None)
    monkeypatch.setattr(llm_runtime, "_nvidia_llm", None)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        llm_runtime.get_nvidia()


def test_should_fallback_to_openai_semantics(monkeypatch):
    # Non-OpenAI primary + OpenAI key configured -> fallback allowed.
    monkeypatch.setattr(llm, "MODEL_TYPE", "nvidia")
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "k")
    assert llm._should_fallback_to_openai() is True
    # Non-OpenAI primary + no OpenAI key -> no keyless fallback crash.
    monkeypatch.setattr(llm, "OPENAI_API_KEY", None)
    assert llm._should_fallback_to_openai() is False
    # OpenAI is already primary -> never self-fallback.
    monkeypatch.setattr(llm, "MODEL_TYPE", "openai")
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "k")
    assert llm._should_fallback_to_openai() is False


_VALIDATE_BASE = dict(
    supabase_db_url="postgresql://u:p@localhost/db",
    gateway_shared_secret="g",
    session_signing_secret="s",
    evaluate_admin_secret="e",
    auth_mode="gateway_only",
    deployment_env="development",
    supabase_jwt_secret=None,
    enable_evaluate_endpoint=False,
    allow_evaluate_endpoint=False,
    google_api_key=None,
)


def test_validate_runtime_settings_nvidia_requires_key():
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        config.validate_runtime_settings(
            model_type="nvidia", nvidia_api_key=None, **_VALIDATE_BASE
        )
    # With a key present it must pass.
    config.validate_runtime_settings(
        model_type="nvidia", nvidia_api_key="k", **_VALIDATE_BASE
    )


def test_validate_runtime_settings_rejects_unknown_model_type():
    with pytest.raises(RuntimeError, match="Invalid MODEL_TYPE"):
        config.validate_runtime_settings(model_type="bogus", **_VALIDATE_BASE)
