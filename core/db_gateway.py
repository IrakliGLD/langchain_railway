"""Single guarded entry point for database connections.

All runtime database paths must acquire connections through this module so the
database circuit breaker observes the same failure classification everywhere.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import (
    DBAPIError,
    DisconnectionError,
    OperationalError,
    SQLAlchemyError,
)
from sqlalchemy.exc import (
    TimeoutError as SQLAlchemyTimeoutError,
)

from config import DB_STATEMENT_TIMEOUT_MS, REQUEST_CLEANUP_ALLOWANCE_MS
from utils.metrics import metrics
from utils.request_deadline import current_request_execution_scope
from utils.resilience import db_circuit_breaker

log = logging.getLogger("Enai")

_TRANSIENT_SQLSTATE_PREFIXES = ("08", "53", "58")
_TRANSIENT_SQLSTATES = {
    "40001",  # serialization_failure
    "40P01",  # deadlock_detected
    "57014",  # query_canceled / statement timeout
    "57P01",  # admin_shutdown
    "57P02",  # crash_shutdown
    "57P03",  # cannot_connect_now
}


def _sqlstate(error: BaseException) -> str | None:
    """Return a driver SQLSTATE without relying on one psycopg version."""
    candidates: list[object] = [error]
    original = getattr(error, "orig", None)
    if original is not None:
        candidates.append(original)
    cause = getattr(error, "__cause__", None)
    if cause is not None:
        candidates.append(cause)

    for candidate in candidates:
        value = getattr(candidate, "sqlstate", None) or getattr(candidate, "pgcode", None)
        if isinstance(value, str) and value:
            return value.upper()
    return None


def is_transient_database_error(error: BaseException) -> bool:
    """Classify only infrastructure/retryable failures as breaker failures."""
    state = _sqlstate(error)
    if state is not None:
        return state in _TRANSIENT_SQLSTATES or state.startswith(_TRANSIENT_SQLSTATE_PREFIXES)

    if isinstance(error, (SQLAlchemyTimeoutError, DisconnectionError, OperationalError)):
        return True
    if isinstance(error, DBAPIError):
        return bool(getattr(error, "connection_invalidated", False))
    return False


def _circuit_open_error(reason: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            "Database temporarily unavailable "
            f"(circuit breaker: {reason}). Please retry shortly."
        ),
    )


@contextmanager
def database_connection(
    engine: Engine,
    *,
    operation: str,
    begin: bool = False,
    read_only: bool = False,
) -> Iterator[Connection]:
    """Yield one breaker-guarded connection or transaction.

    Non-transient SQL/schema/content errors prove the database is reachable, so
    they release a half-open probe as a success but never increment failures.
    """
    allowed, reason = db_circuit_breaker.allow_request()
    if not allowed:
        metrics.log_circuit_open("db")
        log.warning("Database operation blocked by open circuit: %s", operation)
        raise _circuit_open_error(reason)

    try:
        scope = current_request_execution_scope()
        deadline = scope.deadline if scope is not None else None
        if deadline is not None:
            deadline.ensure_remaining(
                f"db_{operation}_checkout",
                minimum_ms=REQUEST_CLEANUP_ALLOWANCE_MS + 1,
            )
        manager = engine.begin() if begin else engine.connect()
        with manager as connection:
            # PostgreSQL requires SET TRANSACTION to be the first statement in
            # a transaction. Apply it before the request-local timeout.
            if read_only:
                connection.execute(text("SET TRANSACTION READ ONLY"), {})
            statement_timeout_ms = DB_STATEMENT_TIMEOUT_MS
            if deadline is not None:
                statement_timeout_ms = deadline.bounded_timeout_ms(
                    f"db_{operation}_statement",
                    configured_timeout_ms=DB_STATEMENT_TIMEOUT_MS,
                    cleanup_allowance_ms=REQUEST_CLEANUP_ALLOWANCE_MS,
                )
            # Bind-safe and transaction-local: a pooled connection cannot leak
            # one request's timeout to the next request.
            execute = getattr(connection, "execute", None)
            if callable(execute):
                execute(
                    text("select set_config('statement_timeout', :timeout_ms, true)"),
                    {"timeout_ms": str(statement_timeout_ms)},
                )
            yield connection
    except BaseException as error:
        if is_transient_database_error(error):
            db_circuit_breaker.record_failure()
            log.warning(
                "Transient database operation failure: operation=%s type=%s",
                operation,
                type(error).__name__,
            )
        else:
            db_circuit_breaker.record_success()
            if isinstance(error, SQLAlchemyError):
                log.info(
                    "Non-transient database operation error: operation=%s type=%s",
                    operation,
                    type(error).__name__,
                )
        raise
    else:
        db_circuit_breaker.record_success()
