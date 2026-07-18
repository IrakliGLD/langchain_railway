"""Characterization tests for startup composition and readiness ownership."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from fastapi.testclient import TestClient

from core.application_runtime import ApplicationRuntime, ReadinessSnapshot

REQUIRED_SCHEMA = {
    "required_view": {"date", "value"},
    "second_view": {"entity"},
}
COMPLETE_ROWS = [
    ("required_view", "date"),
    ("required_view", "value"),
    ("second_view", "entity"),
]


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _Connection:
    def __init__(self, rows):
        self._rows = rows
        self.statements: list[str] = []

    def execute(self, statement):
        self.statements.append(str(statement))
        return _Result(self._rows)


def _runtime(
    *,
    database_ready=lambda: True,
    connection_factory=None,
    clock=lambda: 100.0,
    cache_ttl=30.0,
    retry_interval=5.0,
    process_initializers=(),
    lifespan_tasks=(),
):
    if connection_factory is None:

        @contextmanager
        def connection_factory(_engine, **_kwargs):
            yield _Connection(COMPLETE_ROWS)

    return ApplicationRuntime(
        engine=object(),
        database_connection=connection_factory,
        database_available=database_ready,
        required_schema_columns=REQUIRED_SCHEMA,
        cache_ttl_seconds=cache_ttl,
        retry_interval_seconds=retry_interval,
        logger=logging.getLogger("test.application_runtime"),
        monotonic=clock,
        process_initializers=process_initializers,
        lifespan_tasks=lifespan_tasks,
    )


def test_readiness_snapshot_preserves_public_http_contract():
    ready = ReadinessSnapshot(database_ready=True, schema_ready=True)
    degraded = ReadinessSnapshot(database_ready=True, schema_ready=False)

    assert ready.ready is True
    assert ready.status_code == 200
    assert ready.public_payload() == {
        "status": "ready",
        "database_ready": True,
        "schema_ready": True,
    }
    assert degraded.ready is False
    assert degraded.status_code == 503
    assert degraded.public_payload() == {
        "status": "degraded",
        "database_ready": True,
        "schema_ready": False,
    }


def test_schema_gaps_require_every_declared_column():
    runtime = _runtime()

    assert runtime.schema_gaps({}) == {
        "required_view": ["date", "value"],
        "second_view": ["entity"],
    }
    assert (
        runtime.schema_gaps(
            {
                "required_view": {"date", "value", "extra"},
                "second_view": {"entity"},
            }
        )
        == {}
    )


def test_database_failure_skips_reflection_and_fails_closed():
    connection_calls = 0

    @contextmanager
    def connection_factory(_engine, **_kwargs):
        nonlocal connection_calls
        connection_calls += 1
        yield _Connection(COMPLETE_ROWS)

    runtime = _runtime(
        database_ready=lambda: False,
        connection_factory=connection_factory,
    )

    assert runtime.readiness() == ReadinessSnapshot(False, False)
    assert connection_calls == 0


def test_successful_reflection_is_cached_without_expanding_external_allowlists():
    connection_calls = 0
    rows = [*COMPLETE_ROWS, ("private_unlisted_view", "secret")]

    @contextmanager
    def connection_factory(_engine, *, operation, priority):
        nonlocal connection_calls
        assert operation == "schema_reflection"
        assert priority == "control"
        connection_calls += 1
        yield _Connection(rows)

    runtime = _runtime(connection_factory=connection_factory)

    assert runtime.readiness().ready is True
    assert runtime.readiness().ready is True
    assert connection_calls == 1
    assert runtime.schema_map["private_unlisted_view"] == {"secret"}
    assert set(runtime.required_schema_columns) == set(REQUIRED_SCHEMA)


def test_incomplete_but_fresh_reflection_is_reused_and_remains_degraded():
    connection_calls = 0

    @contextmanager
    def connection_factory(_engine, **_kwargs):
        nonlocal connection_calls
        connection_calls += 1
        yield _Connection([("required_view", "date")])

    runtime = _runtime(connection_factory=connection_factory)

    assert runtime.readiness() == ReadinessSnapshot(True, False)
    assert runtime.readiness() == ReadinessSnapshot(True, False)
    assert connection_calls == 1


def test_failed_stale_refresh_keeps_diagnostics_but_is_retry_throttled():
    now = [100.0]
    connection_calls = 0
    should_fail = False

    @contextmanager
    def connection_factory(_engine, **_kwargs):
        nonlocal connection_calls
        connection_calls += 1
        if should_fail:
            raise RuntimeError("database unavailable")
        yield _Connection(COMPLETE_ROWS)

    runtime = _runtime(
        connection_factory=connection_factory,
        clock=lambda: now[0],
        cache_ttl=10.0,
        retry_interval=5.0,
    )
    assert runtime.readiness().ready is True
    prior_schema = runtime.schema_map

    now[0] = 111.0
    should_fail = True
    assert runtime.readiness() == ReadinessSnapshot(True, False)
    assert runtime.readiness() == ReadinessSnapshot(True, False)
    assert connection_calls == 2
    assert runtime.schema_map == prior_schema

    now[0] = 116.0
    assert runtime.readiness() == ReadinessSnapshot(True, False)
    assert connection_calls == 3


def test_schema_refresh_is_singleflight_across_concurrent_probes():
    connection_calls = 0

    @contextmanager
    def connection_factory(_engine, **_kwargs):
        nonlocal connection_calls
        connection_calls += 1
        time.sleep(0.02)
        yield _Connection(COMPLETE_ROWS)

    runtime = _runtime(connection_factory=connection_factory)

    with ThreadPoolExecutor(max_workers=8) as executor:
        snapshots = list(executor.map(lambda _: runtime.readiness(), range(8)))

    assert snapshots == [ReadinessSnapshot(True, True)] * 8
    assert connection_calls == 1


def test_process_initializers_run_once_in_declared_order():
    calls: list[str] = []
    runtime = _runtime(
        process_initializers=(
            lambda: calls.append("knowledge"),
            lambda: calls.append("logging"),
        )
    )

    runtime.initialize_process()
    runtime.initialize_process()

    assert calls == ["knowledge", "logging"]


def test_lifespan_refreshes_before_tasks_and_propagates_task_failure():
    calls: list[str] = []
    runtime = _runtime(
        lifespan_tasks=(
            lambda: calls.append("validate"),
            lambda: calls.append("warm"),
        )
    )
    runtime.refresh_schema = lambda: calls.append("reflect") or True
    app = runtime.create_application(title="Test analyst", version="1.2.3")

    with TestClient(app):
        assert app.title == "Test analyst"
        assert app.version == "1.2.3"
    assert calls == ["reflect", "validate", "warm"]

    failing_runtime = _runtime(lifespan_tasks=(lambda: (_ for _ in ()).throw(RuntimeError("invalid skill")),))
    failing_runtime.refresh_schema = lambda: True
    failing_app = failing_runtime.create_application(title="Test", version="1")

    async def enter_lifespan():
        async with failing_runtime.lifespan(failing_app):
            pass

    try:
        asyncio.run(enter_lifespan())
    except RuntimeError as exc:
        assert str(exc) == "invalid skill"
    else:
        raise AssertionError("startup task failures must abort application startup")
