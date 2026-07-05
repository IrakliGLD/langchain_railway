"""Unit tests for ``core.db`` — the DB-URL coercion leaf (audit P1).

``coerce_to_psycopg_url`` is pure and dependency-light, so it is directly
testable without a database. These pin the scheme-normalisation rules the rest
of the stack depends on (engine unification with ``knowledge.vector_store``).
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest  # noqa: E402

from core.db import DB_URL, coerce_to_psycopg_url  # noqa: E402


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("postgres://user:pass@host/db", "postgresql+psycopg://user:pass@host/db"),
        ("postgresql://user:pass@host/db", "postgresql+psycopg://user:pass@host/db"),
        # Already-correct scheme is returned unchanged.
        ("postgresql+psycopg://user:pass@host/db", "postgresql+psycopg://user:pass@host/db"),
    ],
)
def test_coerce_scheme_normalisation(raw, expected):
    assert coerce_to_psycopg_url(raw) == expected


def test_coerce_preserves_port_and_query():
    raw = "postgres://u:p@h:5432/db?sslmode=require"
    assert coerce_to_psycopg_url(raw) == "postgresql+psycopg://u:p@h:5432/db?sslmode=require"


def test_coerce_leaves_other_psycopg_family_scheme_untouched():
    # Any explicit postgresql+<driver> scheme is respected, not rewritten.
    raw = "postgresql+asyncpg://u:p@h/db"
    assert coerce_to_psycopg_url(raw) == raw


def test_module_db_url_uses_psycopg_scheme():
    # The engine is built from the coerced URL, so the configured SUPABASE_DB_URL
    # (plain postgresql://) must surface as the psycopg driver scheme.
    assert DB_URL.startswith("postgresql+psycopg://")
