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


def test_report_stage_accessor_uses_dedicated_profile_without_changing_primary(
    monkeypatch,
):
    primary = object()
    report = object()
    monkeypatch.setattr(llm, "REPORT_MODEL_TYPE", "openai", raising=False)
    monkeypatch.setattr(llm, "REPORT_MODEL", "gpt-5.6-terra", raising=False)
    monkeypatch.setattr(llm, "make_report", lambda: report, raising=False)
    monkeypatch.setattr(llm, "get_primary_llm", lambda: primary)

    assert llm.get_report_model_name("gemini-stage-model") == "gpt-5.6-terra"
    assert (
        llm.get_llm_for_stage(
            "gemini-stage-model",
            report_profile=True,
            max_retries=1,
        )
        is report
    )
    assert llm.get_llm_for_stage() is primary


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


def test_nvidia_factory_requires_key(monkeypatch):
    monkeypatch.setattr(llm_runtime, "NVIDIA_API_KEY", None)
    monkeypatch.setattr(llm_runtime, "_nvidia_llm", None)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        llm_runtime.get_nvidia()


def test_report_openai_factory_uses_dedicated_worker_profile(monkeypatch):
    captured = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_runtime, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(llm_runtime, "REPORT_MODEL_TYPE", "openai")
    monkeypatch.setattr(llm_runtime, "REPORT_MODEL", "gpt-5.6-terra")
    monkeypatch.setattr(llm_runtime, "REPORT_MAX_OUTPUT_TOKENS", 8192)
    monkeypatch.setattr(llm_runtime, "REPORT_TIMEOUT_SECONDS", 300)
    monkeypatch.setattr(llm_runtime, "REPORT_REASONING_EFFORT", "medium")
    monkeypatch.setattr(llm_runtime, "OPENAI_API_KEY", "report-openai-key")
    monkeypatch.setattr(llm_runtime, "_report_llm", None, raising=False)

    client = llm_runtime.get_report()

    assert isinstance(client, _FakeChatOpenAI)
    assert captured == {
        "model": "gpt-5.6-terra",
        "openai_api_key": "report-openai-key",
        "max_tokens": 8192,
        "request_timeout": 300,
        "max_retries": 0,
        "use_responses_api": True,
        "reasoning_effort": "medium",
    }


def test_report_gemini_factory_uses_same_report_contract(monkeypatch):
    captured = {}

    class _FakeGemini:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        llm_runtime,
        "ChatGoogleGenerativeAI",
        _FakeGemini,
    )
    monkeypatch.setattr(llm_runtime, "REPORT_MODEL_TYPE", "gemini")
    monkeypatch.setattr(llm_runtime, "REPORT_MODEL", "gemini-3.6-flash")
    monkeypatch.setattr(llm_runtime, "REPORT_MAX_OUTPUT_TOKENS", 8192)
    monkeypatch.setattr(llm_runtime, "REPORT_TIMEOUT_SECONDS", 300)
    monkeypatch.setattr(llm_runtime, "REPORT_REASONING_EFFORT", "high")
    monkeypatch.setattr(llm_runtime, "GOOGLE_API_KEY", "report-google-key")
    monkeypatch.setattr(llm_runtime, "_report_llm", None, raising=False)

    client = llm_runtime.get_report()

    assert isinstance(client, _FakeGemini)
    assert captured == {
        "model": "gemini-3.6-flash",
        "google_api_key": "report-google-key",
        "convert_system_message_to_human": True,
        "max_output_tokens": 8192,
        "timeout": 300,
        "max_retries": 1,
        "thinking_level": "high",
    }


def test_report_nvidia_factory_keeps_nvidia_endpoint_and_key(monkeypatch):
    captured = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_runtime, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(llm_runtime, "REPORT_MODEL_TYPE", "nvidia")
    monkeypatch.setattr(
        llm_runtime,
        "REPORT_MODEL",
        "nvidia/nemotron-3-super-120b-a12b",
    )
    monkeypatch.setattr(llm_runtime, "REPORT_MAX_OUTPUT_TOKENS", 8192)
    monkeypatch.setattr(llm_runtime, "REPORT_TIMEOUT_SECONDS", 300)
    monkeypatch.setattr(llm_runtime, "REPORT_REASONING_EFFORT", None)
    monkeypatch.setattr(llm_runtime, "NVIDIA_API_KEY", "report-nvidia-key")
    monkeypatch.setattr(
        llm_runtime,
        "NVIDIA_BASE_URL",
        "https://integrate.api.nvidia.com/v1",
    )
    monkeypatch.setattr(llm_runtime, "_report_llm", None, raising=False)

    client = llm_runtime.get_report()

    assert isinstance(client, _FakeChatOpenAI)
    assert captured == {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "openai_api_key": "report-nvidia-key",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "max_tokens": 8192,
        "request_timeout": 300,
        "max_retries": 0,
    }


def test_should_fallback_to_openai_semantics(monkeypatch):
    # Merely adding an OpenAI key never enables fallback.
    monkeypatch.setattr(llm, "MODEL_TYPE", "nvidia")
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "k")
    monkeypatch.setattr(llm, "ENABLE_OPENAI_FALLBACK", False)
    assert llm._should_fallback_to_openai() is False

    # An explicit deployment opt-in permits a non-OpenAI primary to fall back.
    monkeypatch.setattr(llm, "ENABLE_OPENAI_FALLBACK", True)
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


def test_qwen_is_a_first_class_provider_not_an_nvidia_alias():
    """Qwen must attribute cost and trip its breaker under its own key.

    Routing qwencloud through the NVIDIA slot works, because both are
    OpenAI-compatible endpoints reached with a custom base_url, but it merges
    the two providers' cost attribution and circuit-breaker state.
    """

    assert "qwen" in llm._PROVIDERS
    assert llm._provider_from_model_name("qwen3.7-max") == "qwen"
    assert llm._provider_from_model_name("qwen-plus") == "qwen"
    # Prefix classification must not steal the other providers' ids.
    assert llm._provider_from_model_name("gpt-5.6-terra") == "openai"
    assert llm._provider_from_model_name("openai/gpt-oss-120b") == "nvidia"


def test_qwen_cost_attribution_uses_qwen_rates(monkeypatch):
    monkeypatch.setattr(llm, "QWEN_INPUT_COST_PER_1K_USD", 0.002)
    monkeypatch.setattr(llm, "QWEN_OUTPUT_COST_PER_1K_USD", 0.006)

    cost = llm._estimate_cost_usd(1000, 1000, model_name="qwen3.7-max")
    assert cost == pytest.approx(0.008)


def test_qwen_report_structured_output_prefers_tool_calling(monkeypatch):
    """auto must not assume strict json_schema on a compatible endpoint.

    qwencloud documents Structured Outputs and Function Calling; tool calling is
    the more consistently implemented of the two, and the report writers depend
    on typed output, so auto takes the safer one. Strict schemas stay available
    through REPORT_STRUCTURED_OUTPUT_METHOD.
    """

    monkeypatch.setattr(llm, "REPORT_STRUCTURED_OUTPUT_METHOD", "auto")
    assert llm._report_structured_output_method("qwen") == "function_calling"
    assert llm._report_structured_output_method("openai") == "json_schema"
    assert llm._report_structured_output_method("nvidia") is None

    # An operator can still force it, but qwencloud documents only
    # response_format {"type": "json_object"}, so this is not a supported path.
    monkeypatch.setattr(llm, "REPORT_STRUCTURED_OUTPUT_METHOD", "json_schema")
    assert llm._report_structured_output_method("qwen") == "json_schema"


def test_qwen_client_requires_key_and_compatible_base_url(monkeypatch):
    """A missing base URL must fail as configuration, not as an opaque 401."""

    monkeypatch.setattr(llm_runtime, "_qwen_llm", None)
    monkeypatch.setattr(llm_runtime, "QWEN_API_KEY", "test-qwen-key")
    monkeypatch.setattr(llm_runtime, "QWEN_BASE_URL", "")
    with pytest.raises(RuntimeError, match="QWEN_BASE_URL"):
        llm_runtime.get_qwen()

    monkeypatch.setattr(llm_runtime, "QWEN_API_KEY", None)
    monkeypatch.setattr(llm_runtime, "QWEN_BASE_URL", "https://example.invalid/v1")
    with pytest.raises(RuntimeError, match="QWEN_API_KEY"):
        llm_runtime.get_qwen()


def test_qwen_client_is_built_from_env_without_reasoning_effort(monkeypatch):
    """No reasoning_effort, and no output cap unless one is configured.

    reasoning_effort is OpenAI-specific. max_tokens is omitted because
    qwencloud's structured-output guide warns that truncated output produces
    invalid JSON, and every report stage returns JSON.
    """

    captured = {}

    class _Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_runtime, "_qwen_llm", None)
    monkeypatch.setattr(llm_runtime, "ChatOpenAI", _Client)
    monkeypatch.setattr(llm_runtime, "QWEN_API_KEY", "test-qwen-key")
    monkeypatch.setattr(
        llm_runtime, "QWEN_BASE_URL", "https://example.invalid/compatible/v1"
    )
    monkeypatch.setattr(llm_runtime, "QWEN_MODEL", "qwen3.7-max")

    llm_runtime.get_qwen()

    assert captured["model"] == "qwen3.7-max"
    assert captured["base_url"] == "https://example.invalid/compatible/v1"
    assert captured["openai_api_key"] == "test-qwen-key"
    assert captured["max_retries"] == 0
    assert "reasoning_effort" not in captured
    assert "max_tokens" not in captured

    monkeypatch.setattr(llm_runtime, "_qwen_llm", None)
    monkeypatch.setattr(llm_runtime, "QWEN_MAX_TOKENS", 8192)
    llm_runtime.get_qwen()
    assert captured["max_tokens"] == 8192


def test_qwen_default_base_url_is_the_documented_compatible_endpoint():
    """The default comes from qwencloud's getting-started guide, not a guess."""

    assert config.QWEN_BASE_URL == (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )


def test_config_accepts_qwen_and_reports_missing_qwen_settings():
    base = dict(
        supabase_db_url="postgresql://user:pass@localhost/db",
        gateway_shared_secret="gateway",
        session_signing_secret="session",
        evaluate_admin_secret="evaluate",
        auth_mode="gateway_only",
        deployment_env="development",
        supabase_jwt_secret=None,
        enable_evaluate_endpoint=False,
        allow_evaluate_endpoint=False,
        google_api_key=None,
        model_type="qwen",
    )

    config.validate_runtime_settings(
        **base,
        qwen_api_key="test-qwen-key",
        qwen_base_url="https://example.invalid/compatible/v1",
    )

    with pytest.raises(RuntimeError, match="QWEN_BASE_URL"):
        config.validate_runtime_settings(
            **base,
            qwen_api_key="test-qwen-key",
            qwen_base_url="",
        )

    with pytest.raises(RuntimeError, match="QWEN_API_KEY"):
        config.validate_runtime_settings(
            **base,
            qwen_api_key=None,
            qwen_base_url="https://example.invalid/compatible/v1",
        )
