"""Configuration contract tests for runtime auth and deployment modes."""

import os
import re
from pathlib import Path

import pytest


os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


from config import STATIC_ALLOWED_TABLES, validate_runtime_settings  # noqa: E402


_READONLY_ROLE_SQL = Path(__file__).resolve().parents[1] / "scripts" / "least_privilege_api_role.sql"


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
        supabase_jwt_secret=None,
        enable_evaluate_endpoint=False,
        allow_evaluate_endpoint=False,
        model_type="openai",
        google_api_key=None,
    )


def test_validate_runtime_settings_accepts_auto_mode_with_jwt_secret():
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
        google_api_key=None,
    )
