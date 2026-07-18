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
