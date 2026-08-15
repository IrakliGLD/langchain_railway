"""
Database query execution with connection pooling and security.

Handles:
- SQLAlchemy engine creation with connection pooling
- Read-only transaction enforcement
- Query timeout configuration
- Safe SQL execution with pandas DataFrame output
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, List, Tuple

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import text

from config import DATABASE_RUNTIME_ROLE, MAX_RESULT_SIZE_MB, MAX_ROWS

# ENGINE + DB_URL live in core/db.py (leaf) now; re-exported here for back-compat so
# every `from core.query_executor import ENGINE` (and its test monkeypatches) keeps working.
from core.db import DB_URL, ENGINE, coerce_to_psycopg_url  # noqa: F401
from core.db_gateway import database_connection

log = logging.getLogger("Enai")


@dataclass(frozen=True, slots=True)
class DatabaseRuntimeIdentity:
    """Protected result of the runtime-role/read-only readiness probe."""

    current_user: str
    expected_user: str
    default_transaction_read_only: bool

    @property
    def ready(self) -> bool:
        if not self.expected_user:
            return bool(self.current_user)
        return (
            self.current_user == self.expected_user
            and self.default_transaction_read_only
        )

    def protected_metadata(self) -> dict[str, Any]:
        return {
            "current_user": self.current_user,
            "expected_user": self.expected_user,
            "role_matches": self.current_user == self.expected_user,
            "default_transaction_read_only": self.default_transaction_read_only,
            "ready": self.ready,
        }


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


def get_database_runtime_identity() -> DatabaseRuntimeIdentity:
    """Probe connection identity without exposing it on public endpoints."""

    try:
        with database_connection(
            ENGINE,
            operation="runtime_identity_probe",
            begin=True,
            read_only=True,
            priority="control",
        ) as conn:
            row = conn.execute(
                text(
                    "select current_user as current_user, "
                    "current_setting('default_transaction_read_only') as default_read_only"
                )
            ).mappings().one()
        return DatabaseRuntimeIdentity(
            current_user=str(row["current_user"]),
            expected_user=DATABASE_RUNTIME_ROLE,
            default_transaction_read_only=str(row["default_read_only"]).lower() == "on",
        )
    except Exception as exc:
        log.warning(
            "Database runtime identity probe failed. error_class=%s",
            type(exc).__name__,
        )
        return DatabaseRuntimeIdentity(
            current_user="",
            expected_user=DATABASE_RUNTIME_ROLE,
            default_transaction_read_only=False,
        )


def is_database_available() -> bool:
    """Best-effort database readiness check.

    This must never raise during startup paths; readiness endpoints can call it
    and return degraded status instead of crashing process initialization.
    """
    identity = get_database_runtime_identity()
    if not identity.ready:
        log.warning(
            "Database readiness identity mismatch. role_matches=%s read_only_default=%s",
            identity.current_user == identity.expected_user,
            identity.default_transaction_read_only,
        )
    return identity.ready


def coerce_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Make PostgreSQL ``numeric`` columns visible to numeric consumers.

    psycopg returns ``numeric`` as ``Decimal``, which pandas stores as object
    dtype.  ``select_dtypes(include="number")`` then skips those columns, so
    per-column aggregates (``agent/analyzer.py``), grounding-token extraction
    (``agent/summary_grounding.py``), seasonal statistics
    (``analysis/seasonal_stats.py``) and chart builders all silently see zero
    numeric columns.  ``stats_hint`` collapses to just "Rows: N", the model is
    left to compute its own figures, and the strict grounding gate correctly
    rejects them -- shipping a truncated answer.

    Coerce once here, at the single boundary where SQL rows become a frame,
    rather than at each consumer.  Mirrors the existing treatment of
    driver-enrichment frames in ``agent/pipeline.py``; see
    ``coerce_decimal_columns_to_float`` for the full root-cause note.
    """
    if df is None or df.empty:
        return df
    # Local import: analysis.* is imported lazily at the other call site too,
    # to keep core/ free of a hard dependency on analysis/.
    from analysis.system_quantities import coerce_decimal_columns_to_float

    coerced, changed_columns = coerce_decimal_columns_to_float(df)
    if changed_columns:
        log.info(
            "Coerced %d Decimal column(s) to float on the SQL result frame",
            len(changed_columns),
        )
    return coerced


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
    with database_connection(ENGINE, operation="fallback_sql", read_only=True) as conn:

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

        # Convert to DataFrame for compatibility. Coerce Decimal columns here so
        # every downstream numeric consumer sees real float dtypes.
        df = coerce_result_frame(pd.DataFrame(rows, columns=cols))

    elapsed = time.time() - start

    # PRODUCTION SAFETY: Check memory limits to prevent OOM errors
    check_dataframe_memory(df)

    log.info(f"⚡ SQL executed safely in {elapsed:.2f}s, returned {len(rows)} rows")

    return df, cols, rows, elapsed

