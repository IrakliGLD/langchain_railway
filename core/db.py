"""Database engine - single source of truth.

Extracted from core/query_executor.py (audit P1) as a dependency-light leaf so that
both core.query_executor and knowledge.vector_store can import ENGINE downward without
the core<->knowledge lazy-import dance. Imports only stdlib + sqlalchemy + config.
"""
import logging
import urllib.parse

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from config import (
    DB_CONNECT_TIMEOUT_SECONDS,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT_SECONDS,
    DB_STATEMENT_TIMEOUT_MS,
    SUPABASE_DB_URL,
)

log = logging.getLogger("Enai")


def coerce_to_psycopg_url(url: str) -> str:
    """
    Convert database URL to use psycopg driver.

    Ensures the URL uses postgresql+psycopg:// scheme for SQLAlchemy.

    Args:
        url: Database URL (can be postgres://, postgresql://, or postgresql+psycopg://)

    Returns:
        URL with postgresql+psycopg:// scheme

    Examples:
        >>> coerce_to_psycopg_url("postgres://user:pass@host/db")
        'postgresql+psycopg://user:pass@host/db'
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("postgres", "postgresql"):
        return url.replace(parsed.scheme, "postgresql+psycopg", 1)
    if not parsed.scheme.startswith("postgresql+"):
        return "postgresql+psycopg://" + url.split("://", 1)[-1]
    return url


# Database URL with psycopg driver
DB_URL = coerce_to_psycopg_url(SUPABASE_DB_URL)

# SQLAlchemy engine with connection pooling
# Pool budget: pool_size + max_overflow = 5 max connections from this engine.
# Combined with vector_store (which reuses this engine after Fix 3), total
# stays well within Supabase PgBouncer limits (typically 10-20 slots).
ENGINE = create_engine(
    DB_URL,
    poolclass=QueuePool,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT_SECONDS,
    pool_pre_ping=True,
    pool_recycle=300,      # 5 min: recycle before PgBouncer kills idle connections
    connect_args={
        "connect_timeout": DB_CONNECT_TIMEOUT_SECONDS,
        # Default; request calls replace it transaction-locally from remaining budget.
        "options": f"-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}",
        # PgBouncer compatibility: disable psycopg auto-prepared statements
        # to avoid "prepared statement already exists" errors in transaction-pooled mode.
        # Required for engine unification with vector_store (Fix 3).
        "prepare_threshold": None,
    },
)
