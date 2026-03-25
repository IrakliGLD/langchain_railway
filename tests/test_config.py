"""Configuration contract tests for runtime auth and deployment modes."""

import os

import pytest


os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


from config import validate_runtime_settings  # noqa: E402


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
            model_type="openai",
            google_api_key=None,
        )


def test_validate_runtime_settings_blocks_evaluate_in_production():
    with pytest.raises(RuntimeError, match="ENABLE_EVALUATE_ENDPOINT"):
        validate_runtime_settings(
            supabase_db_url="postgresql://user:pass@localhost/db",
            gateway_shared_secret="gateway",
            session_signing_secret="session",
            evaluate_admin_secret="evaluate",
            auth_mode="gateway_only",
            deployment_env="production",
            supabase_jwt_secret=None,
            enable_evaluate_endpoint=True,
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
        model_type="openai",
        google_api_key=None,
    )
