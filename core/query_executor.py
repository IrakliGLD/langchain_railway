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

from config import SUPABASE_DB_URL, MAX_RESULT_SIZE_MB

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
ENGINE = create_engine(
    DB_URL,
    poolclass=QueuePool,
    pool_size=10,  # Increased from 5 for better concurrency
    max_overflow=5,  # Increased from 2 to handle traffic spikes
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=1800,  # Increased from 300 (30 min) for Supabase
    connect_args={
        "connect_timeout": 30,
        # Phase 1D Security: Database-level query timeout (30 seconds max)
        "options": "-c statement_timeout=30000"  # 30s in milliseconds
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


def test_connection() -> bool:
    """
    Test database connectivity.

    Returns:
        True if connection successful, False otherwise

    Raises:
        SQLAlchemyError: If connection test fails
    """
    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
            log.info("✅ Database connectivity verified")
            return True
    except SQLAlchemyError as e:
        log.error(f"❌ Database connection failed: {e}")
        raise


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

    with ENGINE.connect() as conn:
        # Phase 1D: Enforce read-only mode
        conn.execute(text("SET TRANSACTION READ ONLY"))

        # Execute query
        result = conn.execute(text(sql))
        rows = result.fetchall()
        cols = list(result.keys())

        # Convert to DataFrame for compatibility
        df = pd.DataFrame(rows, columns=cols)

    elapsed = time.time() - start

    # PRODUCTION SAFETY: Check memory limits to prevent OOM errors
    check_dataframe_memory(df)

    log.info(f"⚡ SQL executed safely in {elapsed:.2f}s, returned {len(rows)} rows")

    return df, cols, rows, elapsed


# Test connection on module import
test_connection()
