"""Application composition and database-schema readiness ownership.

This module keeps lifecycle and readiness state behind one stable boundary so
``main`` can remain the HTTP composition root.  It intentionally does not own
request handlers, middleware, or the application's public response schemas.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, ContextManager

from fastapi import FastAPI
from sqlalchemy import text


@dataclass(frozen=True)
class ReadinessSnapshot:
    """Typed internal result for the existing ``/readyz`` contract."""

    database_ready: bool
    schema_ready: bool

    @property
    def ready(self) -> bool:
        return self.database_ready and self.schema_ready

    @property
    def status_code(self) -> int:
        return 200 if self.ready else 503

    def public_payload(self) -> dict[str, bool | str]:
        return {
            "status": "ready" if self.ready else "degraded",
            "database_ready": self.database_ready,
            "schema_ready": self.schema_ready,
        }


class ApplicationRuntime:
    """Own startup work, schema reflection state, and readiness evaluation."""

    _SCHEMA_REFLECTION_SQL = text(
        """
        SELECT m.matviewname AS view_name, a.attname AS column_name
        FROM pg_matviews m
        JOIN pg_attribute a ON m.matviewname::regclass = a.attrelid
        WHERE a.attnum > 0 AND NOT a.attisdropped
        AND m.schemaname = 'public';
        """
    )

    def __init__(
        self,
        *,
        engine: Any,
        database_connection: Callable[..., ContextManager[Any]],
        database_available: Callable[[], bool],
        required_schema_columns: Mapping[str, Sequence[str]],
        cache_ttl_seconds: float,
        retry_interval_seconds: float,
        logger: logging.Logger,
        monotonic: Callable[[], float] = time.monotonic,
        process_initializers: Sequence[Callable[[], Any]] = (),
        lifespan_tasks: Sequence[Callable[[], Any]] = (),
    ) -> None:
        self._engine = engine
        self._database_connection = database_connection
        self._database_available = database_available
        self._required_schema_columns = {
            table_name: frozenset(columns) for table_name, columns in required_schema_columns.items()
        }
        self._cache_ttl_seconds = cache_ttl_seconds
        self._retry_interval_seconds = retry_interval_seconds
        self._logger = logger
        self._monotonic = monotonic
        self._process_initializers = tuple(process_initializers)
        self._lifespan_tasks = tuple(lifespan_tasks)

        self._schema_map: dict[str, set[str]] = {}
        self._schema_refreshed_at: float | None = None
        self._schema_refresh_failed_at: float | None = None
        self._schema_refresh_lock = threading.Lock()
        self._process_initialization_lock = threading.Lock()
        self._process_initialized = False

    @property
    def schema_map(self) -> dict[str, set[str]]:
        """Return diagnostic reflection state without exposing mutable internals."""

        return {name: set(columns) for name, columns in self._schema_map.items()}

    @property
    def required_schema_columns(self) -> dict[str, frozenset[str]]:
        return dict(self._required_schema_columns)

    def schema_gaps(
        self,
        schema_map: Mapping[str, set[str]] | None = None,
    ) -> dict[str, list[str]]:
        reflected = self._schema_map if schema_map is None else schema_map
        gaps: dict[str, list[str]] = {}
        for table_name, required_columns in self._required_schema_columns.items():
            missing = sorted(required_columns - set(reflected.get(table_name, set())))
            if missing:
                gaps[table_name] = missing
        return gaps

    def _cache_is_fresh(self, now: float) -> bool:
        if self._schema_refreshed_at is None:
            return False
        age_seconds = now - self._schema_refreshed_at
        return 0 <= age_seconds <= self._cache_ttl_seconds

    def _retry_is_due(self, now: float) -> bool:
        if self._schema_refresh_failed_at is None:
            return True
        age_seconds = now - self._schema_refresh_failed_at
        return age_seconds < 0 or age_seconds >= self._retry_interval_seconds

    def _ensure_schema_current(self) -> bool:
        now = self._monotonic()
        if self._cache_is_fresh(now):
            return True
        if not self._retry_is_due(now):
            return False

        with self._schema_refresh_lock:
            now = self._monotonic()
            if self._cache_is_fresh(now):
                return True
            if not self._retry_is_due(now):
                return False
            return self.refresh_schema()

    def refresh_schema(self) -> bool:
        """Atomically replace reflection state after one successful DB query."""

        try:
            with self._database_connection(
                self._engine,
                operation="schema_reflection",
                priority="control",
            ) as connection:
                rows = connection.execute(self._SCHEMA_REFLECTION_SQL).fetchall()

            reflected: dict[str, set[str]] = {}
            for view_name, column_name in rows:
                reflected.setdefault(str(view_name).lower(), set()).add(str(column_name).lower())

            self._schema_map = reflected
            self._schema_refreshed_at = self._monotonic()
            self._schema_refresh_failed_at = None
            self._logger.info("Schema reflection complete: %s views", len(reflected))
            gaps = self.schema_gaps(reflected)
            if gaps:
                gap_summary = "; ".join(
                    f"{table_name}({','.join(columns)})" for table_name, columns in sorted(gaps.items())
                )
                self._logger.warning("Schema readiness missing required columns: %s", gap_summary)
            return True
        except Exception as exc:
            self._schema_refresh_failed_at = self._monotonic()
            self._logger.warning("Schema reflection unavailable: %s", exc)
            return False

    def readiness(self) -> ReadinessSnapshot:
        database_ready = bool(self._database_available())
        schema_refreshed = self._ensure_schema_current() if database_ready else False
        schema_ready = bool(database_ready and schema_refreshed and not self.schema_gaps())
        return ReadinessSnapshot(database_ready, schema_ready)

    def initialize_process(self) -> None:
        """Run import-time initializers once, retrying only after a failure."""

        if self._process_initialized:
            return
        with self._process_initialization_lock:
            if self._process_initialized:
                return
            for initializer in self._process_initializers:
                initializer()
            self._process_initialized = True

    @asynccontextmanager
    async def lifespan(self, _app: FastAPI):
        self.refresh_schema()
        for task in self._lifespan_tasks:
            task()
        yield

    def create_application(self, *, title: str, version: str) -> FastAPI:
        return FastAPI(title=title, version=version, lifespan=self.lifespan)
