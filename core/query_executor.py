"""
Database query execution with connection pooling and security.

Handles:
- SQLAlchemy engine creation with connection pooling
- Read-only transaction enforcement
- Query timeout configuration
- Safe SQL execution with pandas DataFrame output
"""
import time
import logging
import urllib.parse
from typing import Tuple, List, Any

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError
import pandas as pd
from fastapi import HTTPException

from config import SUPABASE_DB_URL, MAX_RESULT_SIZE_MB, MAX_ROWS
from utils.metrics import metrics
from utils.resilience import db_circuit_breaker

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
    pool_size=3,           # Conservative: avoids PgBouncer saturation under concurrent load
    max_overflow=2,        # Total max: 5 connections from this engine
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=300,      # 5 min: recycle before PgBouncer kills idle connections
    connect_args={
        "connect_timeout": 10,  # Fail fast: 10s is sufficient for Supabase TCP handshake
        # Phase 1D Security: Database-level query timeout (30 seconds max)
        "options": "-c statement_timeout=30000",  # 30s in milliseconds
        # PgBouncer compatibility: disable psycopg auto-prepared statements
        # to avoid "prepared statement already exists" errors in transaction-pooled mode.
        # Required for engine unification with vector_store (Fix 3).
        "prepare_threshold": None,
    },
)


def check_dataframe_memory(df: pd.DataFrame, max_mb: int = None) -> None:
    """
    Check if DataFrame exceeds memory limit and raise error if so.

    PRODUCTION SAFETY: Prevents Out-Of-Memory (OOM) errors from large query results.

    Args:
        df: DataFrame to check
        max_mb: Maximum allowed memory in megabytes (uses MAX_RESULT_SIZE_MB from config if None)

    Raises:
        HTTPException 413: If DataFrame exceeds memory limit

    Examples:
        >>> df = pd.DataFrame({'col': range(1000000)})
        >>> check_dataframe_memory(df, max_mb=50)
        # Raises HTTPException if > 50MB
    """
    if max_mb is None:
        max_mb = MAX_RESULT_SIZE_MB

    # Calculate memory usage (including object overhead)
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)

    if memory_mb > max_mb:
        log.error(f"❌ Result set too large: {memory_mb:.2f} MB (limit: {max_mb} MB)")
        log.error(f"   Rows: {len(df):,}, Columns: {len(df.columns)}")
        raise HTTPException(
            status_code=413,
            detail=f"Query result too large ({memory_mb:.0f} MB exceeds {max_mb} MB limit). Please add filters to reduce result size (e.g., add date range, LIMIT clause, or specific conditions)."
        )

    log.info(f"✅ Memory check passed: {memory_mb:.2f} MB / {max_mb} MB limit")


def is_database_available() -> bool:
    """Best-effort database readiness check.

    This must never raise during startup paths; readiness endpoints can call it
    and return degraded status instead of crashing process initialization.
    """
    allowed, reason = db_circuit_breaker.allow_request()
    if not allowed:
        metrics.log_circuit_open("db")
        log.warning("Database availability check skipped due to open circuit: %s", reason)
        return False

    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_circuit_breaker.record_success()
        return True
    except SQLAlchemyError as exc:
        db_circuit_breaker.record_failure()
        log.warning("Database readiness check failed: %s", exc)
        return False


def execute_sql_safely(sql: str, timeout_seconds: int = 30) -> Tuple[pd.DataFrame, List[str], List[Any], float]:
    """
    Execute SQL with read-only transaction enforcement.

    Phase 1D Security Enhancement:
    - Enforces READ ONLY transaction mode to prevent data modification
    - Uses database-level timeout (already configured in ENGINE)
    - Returns pandas DataFrame for consistency with existing code

    Args:
        sql: The validated SQL query to execute
        timeout_seconds: Maximum execution time (already set at connection level)

    Returns:
        tuple: (DataFrame, column_names, rows, execution_time)
        - DataFrame: pandas DataFrame with query results
        - column_names: List of column names
        - rows: List of row tuples
        - execution_time: Query execution time in seconds

    Raises:
        DatabaseError: If query attempts to modify data or exceeds timeout

    Examples:
        >>> df, cols, rows, elapsed = execute_sql_safely("SELECT * FROM dates_mv LIMIT 5")
        >>> print(f"Returned {len(rows)} rows in {elapsed:.2f}s")
    """
    start = time.time()
    allowed, reason = db_circuit_breaker.allow_request()
    if not allowed:
        metrics.log_circuit_open("db")
        raise HTTPException(
            status_code=503,
            detail=f"Database temporarily unavailable (circuit breaker: {reason}). Please retry shortly.",
        )

    try:
        with ENGINE.connect() as conn:
            # Phase 1D: Enforce read-only mode
            conn.execute(text("SET TRANSACTION READ ONLY"))

            # Execute query with incremental fetch to limit memory pressure.
            # LIMIT is already enforced upstream by plan_validate_repair(),
            # but fetchmany provides a defense-in-depth safety net.
            result = conn.execute(text(sql))
            cols = list(result.keys())
            rows = []
            _BATCH = 1000
            while True:
                batch = result.fetchmany(_BATCH)
                if not batch:
                    break
                rows.extend(batch)
                if len(rows) >= MAX_ROWS:
                    log.warning(
                        "Row safety cap hit (%d rows), truncating result", len(rows),
                    )
                    rows = rows[:MAX_ROWS]
                    break

            # Convert to DataFrame for compatibility
            df = pd.DataFrame(rows, columns=cols)
    except (OperationalError, DatabaseError, SQLAlchemyError):
        db_circuit_breaker.record_failure()
        raise
    else:
        db_circuit_breaker.record_success()

    elapsed = time.time() - start

    # PRODUCTION SAFETY: Check memory limits to prevent OOM errors
    check_dataframe_memory(df)

    log.info(f"⚡ SQL executed safely in {elapsed:.2f}s, returned {len(rows)} rows")

    return df, cols, rows, elapsed

