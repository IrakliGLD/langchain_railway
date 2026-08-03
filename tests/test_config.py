"""Configuration contract tests for runtime auth and deployment modes."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


from config import (  # noqa: E402
    DB_APPLICATION_CONCURRENCY,
    DB_CONTROL_RESERVED_SLOTS,
    DB_MAX_CONCURRENCY,
    DB_POOL_CONNECTION_CEILING,
    DB_SECONDARY_WORKERS,
    HTTP_SERVER_PORT,
    HTTP_SERVER_WORKERS,
    MAX_REQUEST_BODY_BYTES,
    REPORT_JOB_TIMEOUT_SECONDS,
    REPORT_MAX_GENERATIVE_CALLS,
    REPORT_RESEARCH_MAX_TRACKS,
    REPORT_RESEARCH_MAX_WORKERS,
    REPORT_SECTION_MAX_WORKERS,
    REPORT_TRACK_ANALYSIS_MODE,
    SCHEMA_READINESS_CACHE_TTL_SECONDS,
    SCHEMA_READINESS_RETRY_INTERVAL_SECONDS,
    SESSION_HISTORY_MAX_ITEM_CHARS,
    SESSION_MAX_ENTRIES,
    SESSION_TURN_WAIT_TIMEOUT_MS,
    STATIC_ALLOWED_TABLES,
    _read_bounded_int_env,
    _read_single_worker_count,
    validate_runtime_settings,
)

_READONLY_ROLE_SQL = Path(__file__).resolve().parents[1] / "scripts" / "least_privilege_api_role.sql"


@pytest.mark.parametrize("raw_value", ["not-an-integer", "262143", "1048577"])
def test_request_body_limit_configuration_fails_closed(monkeypatch, raw_value):
    monkeypatch.setenv("TEST_BODY_LIMIT", raw_value)

    with pytest.raises(RuntimeError, match="TEST_BODY_LIMIT"):
        _read_bounded_int_env("TEST_BODY_LIMIT", 262144, 262144, 1048576)


def test_request_body_limit_default_is_within_the_enforced_bounds():
    assert 262144 <= MAX_REQUEST_BODY_BYTES <= 1048576


def test_http_server_port_defaults_to_fixed_railway_target():
    assert HTTP_SERVER_PORT == 3000


def test_http_runtime_is_pinned_to_one_worker():
    assert HTTP_SERVER_WORKERS == 1


def test_database_work_budget_reserves_control_capacity():
    assert DB_MAX_CONCURRENCY <= DB_POOL_CONNECTION_CEILING
    assert DB_CONTROL_RESERVED_SLOTS >= 1
    assert DB_APPLICATION_CONCURRENCY >= 1
    assert DB_APPLICATION_CONCURRENCY + DB_CONTROL_RESERVED_SLOTS == DB_MAX_CONCURRENCY
    assert DB_SECONDARY_WORKERS <= DB_APPLICATION_CONCURRENCY


def test_report_concurrency_and_deadline_controls_are_bounded():
    assert 1 <= REPORT_SECTION_MAX_WORKERS <= 8
    assert 60 <= REPORT_JOB_TIMEOUT_SECONDS <= 3600
    assert 2 <= REPORT_MAX_GENERATIVE_CALLS <= 6
    assert 1 <= REPORT_RESEARCH_MAX_TRACKS <= 8
    assert 1 <= REPORT_RESEARCH_MAX_WORKERS <= REPORT_RESEARCH_MAX_TRACKS


def _report_pipeline_v2_with_env(**overrides) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for name in (
        "REPORT_PIPELINE_V2_MODE",
        "REPORT_MAX_GENERATIVE_CALLS",
        "REPORT_RESEARCH_MAX_TRACKS",
        "REPORT_RESEARCH_MAX_WORKERS",
        "REPORT_TRACK_ANALYSIS_MODE",
    ):
        env.pop(name, None)
    env.update(overrides)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import config; "
                "print(config.REPORT_PIPELINE_V2_MODE, "
                "config.REPORT_MAX_GENERATIVE_CALLS, "
                "config.REPORT_RESEARCH_MAX_TRACKS, "
                "config.REPORT_RESEARCH_MAX_WORKERS, "
                "config.REPORT_TRACK_ANALYSIS_MODE)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_report_pipeline_v2_defaults_are_disabled_and_bounded():
    result = _report_pipeline_v2_with_env()

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["disabled", "3", "4", "3", "disabled"]


@pytest.mark.parametrize("mode", ["disabled", "shadow", "enabled"])
def test_report_pipeline_v2_accepts_only_declared_rollout_modes(mode):
    result = _report_pipeline_v2_with_env(REPORT_PIPELINE_V2_MODE=mode)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[0] == mode


def test_report_pipeline_v2_rejects_unknown_rollout_mode():
    result = _report_pipeline_v2_with_env(REPORT_PIPELINE_V2_MODE="maybe")

    assert result.returncode != 0
    assert "REPORT_PIPELINE_V2_MODE" in result.stderr


@pytest.mark.parametrize("mode", ["disabled", "shadow", "enabled"])
def test_report_track_analysis_accepts_only_declared_rollout_modes(mode):
    result = _report_pipeline_v2_with_env(REPORT_TRACK_ANALYSIS_MODE=mode)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-1] == mode


def test_report_track_analysis_rejects_unknown_rollout_mode():
    result = _report_pipeline_v2_with_env(REPORT_TRACK_ANALYSIS_MODE="maybe")

    assert result.returncode != 0
    assert "REPORT_TRACK_ANALYSIS_MODE" in result.stderr


def test_report_track_analysis_defaults_disabled():
    assert REPORT_TRACK_ANALYSIS_MODE == "disabled"


def test_report_pipeline_v2_rejects_worker_count_above_track_count():
    result = _report_pipeline_v2_with_env(
        REPORT_RESEARCH_MAX_TRACKS="2",
        REPORT_RESEARCH_MAX_WORKERS="3",
    )

    assert result.returncode != 0
    assert "REPORT_RESEARCH_MAX_WORKERS" in result.stderr


@pytest.mark.parametrize(
    "method",
    ["auto", "json_schema", "function_calling", "prompt"],
)
def test_report_structured_output_accepts_declared_methods(method):
    env = os.environ.copy()
    env["REPORT_STRUCTURED_OUTPUT_METHOD"] = method

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import config; print(config.REPORT_STRUCTURED_OUTPUT_METHOD)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == method


def test_report_structured_output_rejects_unknown_method():
    env = os.environ.copy()
    env["REPORT_STRUCTURED_OUTPUT_METHOD"] = "best_effort"

    result = subprocess.run(
        [sys.executable, "-c", "import config"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "REPORT_STRUCTURED_OUTPUT_METHOD" in result.stderr


def test_report_section_concurrency_defaults_to_one_eight_section_wave():
    env = os.environ.copy()
    env.pop("ENAI_REPORT_SECTION_MAX_WORKERS", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import config; print(config.REPORT_SECTION_MAX_WORKERS)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "8"


def _nvidia_token_limits_with_env(**overrides) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(overrides)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import config; "
                "print(config.NVIDIA_CONFIGURED_MAX_TOKENS, config.NVIDIA_MAX_TOKENS)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_nvidia_hosted_gpt_oss_20b_output_limit_is_capped_to_provider_contract():
    result = _nvidia_token_limits_with_env(
        NVIDIA_MODEL="openai/gpt-oss-20b",
        NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1",
        NVIDIA_MAX_TOKENS="16000",
    )

    assert result.returncode == 0, result.stderr
    configured, effective = (int(value) for value in result.stdout.split())
    assert configured == 16000
    assert effective == 4096


def test_custom_nvidia_compatible_endpoint_keeps_its_configured_output_limit():
    result = _nvidia_token_limits_with_env(
        NVIDIA_MODEL="openai/gpt-oss-20b",
        NVIDIA_BASE_URL="https://nim.internal.example/v1",
        NVIDIA_MAX_TOKENS="16000",
    )

    assert result.returncode == 0, result.stderr
    configured, effective = (int(value) for value in result.stdout.split())
    assert configured == 16000
    assert effective == 16000


def _report_profile_with_env(**overrides) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for name in (
        "REPORT_MODEL_TYPE",
        "REPORT_MODEL",
        "REPORT_MAX_OUTPUT_TOKENS",
        "REPORT_TIMEOUT_SECONDS",
        "REPORT_REASONING_EFFORT",
        "ENABLE_OPENAI_FALLBACK",
    ):
        env.pop(name, None)
    env.update(overrides)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import config; "
                "print(config.REPORT_MODEL_TYPE, config.REPORT_MODEL, "
                "config.REPORT_MAX_OUTPUT_TOKENS, config.REPORT_TIMEOUT_SECONDS, "
                "config.REPORT_REASONING_EFFORT, config.ENABLE_OPENAI_FALLBACK)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_report_profile_is_disabled_and_openai_fallback_is_off_by_default():
    result = _report_profile_with_env()

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["None", "None", "8192", "240", "None", "False"]


def test_report_profile_reads_dedicated_openai_worker_settings():
    result = _report_profile_with_env(
        MODEL_TYPE="nvidia",
        NVIDIA_API_KEY="test-nvidia-key",
        OPENAI_API_KEY="test-openai-key",
        REPORT_MODEL_TYPE="openai",
        REPORT_MODEL="gpt-5.6-terra",
        REPORT_MAX_OUTPUT_TOKENS="8192",
        REPORT_TIMEOUT_SECONDS="300",
        REPORT_REASONING_EFFORT="medium",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "openai",
        "gpt-5.6-terra",
        "8192",
        "300",
        "medium",
        "False",
    ]


def test_report_openai_profile_requires_openai_key_even_when_primary_is_nvidia():
    result = _report_profile_with_env(
        MODEL_TYPE="nvidia",
        NVIDIA_API_KEY="test-nvidia-key",
        OPENAI_API_KEY="",
        REPORT_MODEL_TYPE="openai",
        REPORT_MODEL="gpt-5.6-terra",
    )

    assert result.returncode != 0
    assert "REPORT_MODEL_TYPE=openai but OPENAI_API_KEY is missing" in result.stderr


def test_report_model_without_report_provider_fails_closed():
    result = _report_profile_with_env(
        REPORT_MODEL="gpt-5.6-terra",
    )

    assert result.returncode != 0
    assert "REPORT_MODEL requires REPORT_MODEL_TYPE" in result.stderr


def _config_with_env(**overrides) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(overrides)
    return subprocess.run(
        [sys.executable, "-c", "import config; print(config.REPORT_JOB_TIMEOUT_SECONDS, config.REPORT_WORKER_LEASE_SECONDS)"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_every_accepted_job_timeout_admits_a_valid_worker_lease():
    """The lease ceiling must clear the timeout ceiling by the safety margin.

    Enforced by widening the lease bound rather than narrowing the shared
    timeout bound: config.py is imported by both services, so narrowing a
    worker-only limit there can refuse a configuration the web service was
    already running with.
    """

    accepted = _config_with_env(
        ENAI_REPORT_JOB_TIMEOUT_SECONDS="3600",
        ENAI_REPORT_WORKER_LEASE_SECONDS="3630",
    )
    assert accepted.returncode == 0, accepted.stderr
    timeout, lease = (int(value) for value in accepted.stdout.split())
    assert timeout == 3600
    assert lease >= timeout + 30

    rejected = _config_with_env(ENAI_REPORT_JOB_TIMEOUT_SECONDS="3601")
    assert rejected.returncode != 0
    assert "ENAI_REPORT_JOB_TIMEOUT_SECONDS" in rejected.stderr


def test_database_pool_rejects_capacity_without_application_and_control_slots():
    env = os.environ.copy()
    env.update({"ENAI_DB_POOL_SIZE": "1", "ENAI_DB_MAX_OVERFLOW": "0"})
    env.pop("ENAI_DB_MAX_CONCURRENCY", None)
    result = subprocess.run(
        [sys.executable, "-c", "import config"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "one application and one control connection" in result.stderr


def test_lower_pool_uses_a_safe_dynamic_concurrency_default():
    env = os.environ.copy()
    env.update({"ENAI_DB_POOL_SIZE": "2", "ENAI_DB_MAX_OVERFLOW": "0"})
    env.pop("ENAI_DB_MAX_CONCURRENCY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import config; print(config.DB_MAX_CONCURRENCY)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2"

@pytest.mark.parametrize("raw_value", ["0", "2", "not-an-integer"])
def test_http_runtime_rejects_unsupported_worker_settings(monkeypatch, raw_value):
    monkeypatch.setenv("TEST_HTTP_WORKERS", raw_value)

    with pytest.raises(RuntimeError, match="TEST_HTTP_WORKERS"):
        _read_single_worker_count("TEST_HTTP_WORKERS")


def test_session_capacity_and_turn_wait_are_bounded():
    assert 256 <= SESSION_HISTORY_MAX_ITEM_CHARS <= 20000
    assert 1 <= SESSION_MAX_ENTRIES <= 100000
    assert 0 <= SESSION_TURN_WAIT_TIMEOUT_MS <= 30000


def test_schema_readiness_cache_ttl_is_bounded():
    assert 5 <= SCHEMA_READINESS_CACHE_TTL_SECONDS <= 3600
    assert 1 <= SCHEMA_READINESS_RETRY_INTERVAL_SECONDS <= 300


def test_readonly_role_grants_match_whitelist():
    """The least-privilege role migration (audit S7) must grant SELECT on exactly
    the relations in config.STATIC_ALLOWED_TABLES — no more, no fewer. Catches a
    new whitelisted view being added without a matching GRANT (or vice versa)."""
    sql = _READONLY_ROLE_SQL.read_text(encoding="utf-8")

    # Isolate the public GRANT SELECT statement (the one that lists public.* tables).
    match = re.search(
        r"GRANT SELECT ON\s+(?P<body>.*?)\s+TO enai_api_readonly;",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert match, "public GRANT SELECT block not found in migration"
    granted = set(re.findall(r"public\.(\w+)", match.group("body")))

    assert granted == set(STATIC_ALLOWED_TABLES), (
        "Drift between least_privilege_api_role.sql and STATIC_ALLOWED_TABLES: "
        f"only-in-SQL={granted - set(STATIC_ALLOWED_TABLES)}, "
        f"only-in-whitelist={set(STATIC_ALLOWED_TABLES) - granted}"
    )


def test_validate_runtime_settings_requires_jwt_secret_for_bearer_mode():
    with pytest.raises(RuntimeError, match="SUPABASE_JWT_SECRET"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_and_bearer",
            deployment_env="development",
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=False,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
        )


def test_validate_runtime_settings_blocks_evaluate_outside_dev_or_test():
    with pytest.raises(RuntimeError, match="development or test"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_only",
            deployment_env="production",
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=True,
            allow_evaluate_endpoint=True,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
        )


def test_validate_runtime_settings_accepts_gateway_only_without_jwt_secret():
    validate_runtime_settings(
        supabase_db_url="postgresql://user:pass@localhost/db",
        gateway_shared_secret="gateway",
        session_signing_secret="session",
        evaluate_admin_secret="evaluate",
        auth_mode="gateway_only",
        deployment_env="production",
        release_sha="a" * 40,
        supabase_jwt_secret=None,
        enable_evaluate_endpoint=False,
        allow_evaluate_endpoint=False,
        model_type="openai",
        openai_api_key="test-openai-key",
        google_api_key=None,
    )


def test_validate_runtime_settings_rejects_unknown_actor_assertion_mode():
    with pytest.raises(RuntimeError, match="ENAI_GATEWAY_ACTOR_ASSERTION_MODE"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_only",
            deployment_env="production",
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=False,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
            gateway_actor_assertion_mode="disabled",
        )


def test_validate_runtime_settings_rejects_unknown_evidence_finalization_mode():
    with pytest.raises(RuntimeError, match="ENAI_EVIDENCE_FINALIZATION_MODE"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_only",
            deployment_env="production",
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=False,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
            evidence_finalization_mode="on",
        )


def test_validate_runtime_settings_accepts_valid_evidence_finalization_modes():
    for mode in ("off", "shadow", "enforce"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_only",
            deployment_env="production",
            release_sha="a" * 40,
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=False,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
            evidence_finalization_mode=mode,
        )


def test_validate_runtime_settings_rejects_unknown_plan_validation_mode():
    with pytest.raises(RuntimeError, match="ENAI_PLAN_VALIDATION_MODE"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_only",
            deployment_env="production",
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=False,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
            plan_validation_mode="strict",
        )


def test_validate_runtime_settings_accepts_valid_plan_validation_modes():
    for mode in ("warn", "enforce"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_only",
            deployment_env="production",
            release_sha="a" * 40,
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=False,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
            plan_validation_mode=mode,
        )


def test_validate_runtime_settings_blocks_direct_bearer_in_production():
    with pytest.raises(RuntimeError, match="server-owned entitlement path"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_and_bearer",
            deployment_env="production",
            supabase_jwt_secret="jwt-secret",
            enable_evaluate_endpoint=False,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
        )


def test_validate_runtime_settings_allows_direct_bearer_only_in_test():
    validate_runtime_settings(
        supabase_db_url="postgresql://user:pass@localhost/db",
        gateway_shared_secret="gateway",
        session_signing_secret="session",
        evaluate_admin_secret="evaluate",
        auth_mode="gateway_and_bearer",
        deployment_env="test",
        supabase_jwt_secret="jwt-secret",
        enable_evaluate_endpoint=False,
        allow_evaluate_endpoint=False,
        model_type="openai",
        openai_api_key="test-openai-key",
        google_api_key=None,
    )


def test_validate_runtime_settings_rejects_implicit_auto_auth_mode():
    with pytest.raises(RuntimeError, match="gateway_only, gateway_and_bearer"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="auto",
            deployment_env="development",
            supabase_jwt_secret="jwt-secret",
            enable_evaluate_endpoint=False,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
        )


def test_validate_runtime_settings_requires_explicit_opt_in_for_evaluate():
    with pytest.raises(RuntimeError, match="ALLOW_EVALUATE_ENDPOINT"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_only",
            deployment_env="development",
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=True,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
        )


def test_validate_runtime_settings_allows_evaluate_with_explicit_opt_in_in_test():
    validate_runtime_settings(
        supabase_db_url="postgresql://user:pass@localhost/db",
        gateway_shared_secret="gateway",
        session_signing_secret="session",
        evaluate_admin_secret="evaluate",
        auth_mode="gateway_only",
        deployment_env="test",
        supabase_jwt_secret=None,
        enable_evaluate_endpoint=True,
        allow_evaluate_endpoint=True,
        model_type="openai",
        openai_api_key="test-openai-key",
        google_api_key=None,
    )


def test_gemini_vector_provider_requires_dedicated_embedding_key():
    with pytest.raises(RuntimeError, match="GEMINI_EMBEDDING_API_KEY"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_only",
            deployment_env="test",
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=False,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key="legacy-key",
            vector_embedding_provider="gemini",
            gemini_embedding_api_key=None,
        )


def test_legacy_google_embedding_flag_is_removed_from_runtime_sources():
    root = Path(__file__).resolve().parents[1]
    runtime_source = "\n".join(
        (root / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "config.py",
            "knowledge/embedding_service.py",
        )
    )

    assert "ALLOW_LEGACY_GOOGLE_EMBEDDING_KEY" not in runtime_source


def test_gemini_vector_provider_rejects_non_developer_api_mode():
    with pytest.raises(
        RuntimeError,
        match="VECTOR_KNOWLEDGE_EMBEDDING_API_MODE",
    ):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_only",
            deployment_env="test",
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=False,
            allow_evaluate_endpoint=False,
            model_type="openai",
            openai_api_key="test-openai-key",
            google_api_key=None,
            vector_embedding_provider="gemini",
            gemini_embedding_api_key="embedding-key",
            vector_embedding_api_mode="vertex",
        )


def test_embedding_task_profile_rejects_unsupported_provider_or_model():
    common = {
        "supabase_db_url": "postgresql://user:pass@localhost/db",
        "gateway_shared_secret": "gateway",
        "session_signing_secret": "session",
        "evaluate_admin_secret": "evaluate",
        "auth_mode": "gateway_only",
        "deployment_env": "test",
        "supabase_jwt_secret": None,
        "enable_evaluate_endpoint": False,
        "allow_evaluate_endpoint": False,
        "model_type": "openai",
        "openai_api_key": "test-openai-key",
        "google_api_key": None,
    }

    with pytest.raises(
        RuntimeError,
        match="VECTOR_KNOWLEDGE_EMBEDDING_TASK_PROFILE",
    ):
        validate_runtime_settings(
            **common,
            vector_embedding_provider="openai",
            vector_embedding_model="text-embedding-3-small",
            vector_embedding_task_profile=(
                "retrieval_document_query_v1"
            ),
        )

    with pytest.raises(
        RuntimeError,
        match="gemini-embedding-001",
    ):
        validate_runtime_settings(
            **common,
            vector_embedding_provider="gemini",
            vector_embedding_model="gemini-embedding-2-preview",
            vector_embedding_task_profile=(
                "retrieval_document_query_v1"
            ),
            gemini_embedding_api_key="embedding-key",
        )
