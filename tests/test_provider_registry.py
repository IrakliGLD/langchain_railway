"""P0-2 disagreement review: the ``_PROVIDERS`` registry must reproduce the exact
provider-selection and cost-attribution decisions of the pre-refactor if/elif code
(architecture-audit 2026-06-30).

These are the "routing change" evaluation cases required by developer-phased-audit:
clean ids, namespaced NIM ids, the o1/o3/o4 families, uppercase, empty, unknown, and
each MODEL_TYPE — asserted against the behavior locked before the registry landed.
Complements tests/test_provider_selection.py (which pins the public contract).
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from core import llm


# MODEL_TYPE-independent cases (exact configured-model match, name-prefix, or
# namespaced "/" rule all classify regardless of the active provider).
@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("gemini-2.5-flash", "gemini"),      # gemini exact + prefix
        ("gemini-1.5-pro", "gemini"),        # gemini prefix
        ("GEMINI-2.5-FLASH", "gemini"),      # case-insensitive
        ("gpt-4o-mini", "openai"),           # openai exact + prefix
        ("gpt-4o", "openai"),                # gpt- prefix
        ("o1-preview", "openai"),            # o1 family
        ("o3-mini", "openai"),               # o3 family
        ("o4-mega", "openai"),               # o4 family
        ("openai/gpt-oss-120b", "nvidia"),   # NVIDIA_MODEL default; namespaced, NOT openai
        ("meta/llama-3.1-8b", "nvidia"),     # namespaced NIM id
        ("nvidia/nemotron-4", "nvidia"),     # namespaced NIM id
    ],
)
def test_provider_from_model_name_matches_prerefactor(model_name, expected):
    assert llm._provider_from_model_name(model_name) == expected


def test_namespaced_nvidia_id_is_never_openai():
    # Regression guard: "openai/..." starts with "openai" but must route to nvidia.
    assert llm._provider_from_model_name("openai/gpt-oss-120b") == "nvidia"
    assert llm._is_openai_model_name("openai/gpt-oss-120b") is False


def test_empty_and_unknown_follow_active_model_type(monkeypatch):
    # No exact/prefix/namespaced match -> fall back to the active MODEL_TYPE.
    monkeypatch.setattr(llm, "MODEL_TYPE", "nvidia")
    assert llm._provider_from_model_name("") == "nvidia"
    assert llm._provider_from_model_name("totally-unknown-model") == "nvidia"
    monkeypatch.setattr(llm, "MODEL_TYPE", "gemini")
    assert llm._provider_from_model_name("") == "gemini"
    # An invalid MODEL_TYPE falls back to the documented gemini default.
    monkeypatch.setattr(llm, "MODEL_TYPE", "bogus")
    assert llm._provider_from_model_name("") == "gemini"


@pytest.mark.parametrize(
    "model_name,in_attr,out_attr",
    [
        ("gpt-4o-mini", "OPENAI_INPUT_COST_PER_1K_USD", "OPENAI_OUTPUT_COST_PER_1K_USD"),
        ("gemini-2.5-flash", "GEMINI_INPUT_COST_PER_1K_USD", "GEMINI_OUTPUT_COST_PER_1K_USD"),
        ("openai/gpt-oss-120b", "NVIDIA_INPUT_COST_PER_1K_USD", "NVIDIA_OUTPUT_COST_PER_1K_USD"),
    ],
)
def test_cost_attribution_uses_matched_provider_rates(monkeypatch, model_name, in_attr, out_attr):
    monkeypatch.setattr(llm, in_attr, 1.5)
    monkeypatch.setattr(llm, out_attr, 3.0)
    # 2000 prompt + 1000 completion -> 2*1.5 + 1*3.0 = 6.0
    assert abs(llm._estimate_cost_usd(2000, 1000, model_name=model_name) - 6.0) < 1e-9


def test_registry_is_single_source_of_truth():
    assert set(llm._PROVIDERS) == {"gemini", "openai", "nvidia"}
    for key, prov in llm._PROVIDERS.items():
        assert prov.key == key
        assert callable(prov.make_client)
        assert isinstance(prov.model_name(), str)
        assert isinstance(prov.input_rate(), float)
        assert isinstance(prov.output_rate(), float)


def test_nvidia_timeout_env_bounds_client_and_drops_retries(monkeypatch):
    """2026-07-08 glm-5.2 incident: NVIDIA_TIMEOUT_SECONDS must bound the call
    and force max_retries=1 (retrying a timeout on a slow model multiplies the
    wait before the OpenAI fallback can fire). Unset keeps prior behavior."""
    from core import llm_runtime

    monkeypatch.setattr(llm_runtime, "_nvidia_llm", None)
    monkeypatch.setattr(llm_runtime, "NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(llm_runtime, "NVIDIA_TIMEOUT_SECONDS", 90.0)
    bounded = llm_runtime.get_nvidia()
    assert bounded.request_timeout == 90.0
    assert bounded.max_retries == 1

    monkeypatch.setattr(llm_runtime, "_nvidia_llm", None)
    monkeypatch.setattr(llm_runtime, "NVIDIA_TIMEOUT_SECONDS", None)
    unbounded = llm_runtime.get_nvidia()
    assert unbounded.request_timeout is None
    assert unbounded.max_retries == 2
