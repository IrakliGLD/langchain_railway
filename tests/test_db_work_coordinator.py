from __future__ import annotations

import os
import threading
import time
from pathlib import Path

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from core.db_work_coordinator import (
    DatabaseWorkCapacityExceeded,
    DatabaseWorkCoordinator,
)


def _coordinator(
    *,
    application_capacity: int = 2,
    control_capacity: int = 1,
    secondary_workers: int = 1,
    secondary_pending_limit: int = 2,
) -> DatabaseWorkCoordinator:
    return DatabaseWorkCoordinator(
        application_capacity=application_capacity,
        control_capacity=control_capacity,
        queue_timeout_seconds=0.01,
        secondary_workers=secondary_workers,
        secondary_pending_limit=secondary_pending_limit,
    )


def test_application_work_never_exceeds_its_capacity():
    coordinator = _coordinator(application_capacity=2)
    try:
        with coordinator.admission(operation="one", timeout_seconds=0):
            with coordinator.admission(operation="two", timeout_seconds=0):
                with pytest.raises(DatabaseWorkCapacityExceeded):
                    with coordinator.admission(operation="three", timeout_seconds=0):
                        pass
        snapshot = coordinator.snapshot()
        assert snapshot["peak_application"] == 2
        assert snapshot["active_application"] == 0
        assert snapshot["rejected_application"] == 1
    finally:
        coordinator.shutdown_for_tests()


def test_control_capacity_remains_available_during_application_saturation():
    coordinator = _coordinator(application_capacity=1, control_capacity=1)
    try:
        with coordinator.admission(operation="application", timeout_seconds=0):
            with coordinator.admission(
                operation="readiness", priority="control", timeout_seconds=0
            ):
                snapshot = coordinator.snapshot()
                assert snapshot["active_application"] == 1
                assert snapshot["active_control"] == 1
        assert coordinator.snapshot()["peak_control"] == 1
    finally:
        coordinator.shutdown_for_tests()


def test_queue_wait_is_bounded_and_capacity_is_released_after_error():
    coordinator = _coordinator(application_capacity=1)
    try:
        with coordinator.admission(operation="holder", timeout_seconds=0):
            started = time.monotonic()
            with pytest.raises(DatabaseWorkCapacityExceeded):
                with coordinator.admission(operation="queued", timeout_seconds=0.02):
                    pass
            assert time.monotonic() - started < 0.1

        with pytest.raises(RuntimeError):
            with coordinator.admission(operation="error", timeout_seconds=0):
                raise RuntimeError("boom")
        with coordinator.admission(operation="after_error", timeout_seconds=0):
            pass
        assert coordinator.snapshot()["active_application"] == 0
    finally:
        coordinator.shutdown_for_tests()


def test_secondary_submission_queue_is_bounded():
    coordinator = _coordinator(
        application_capacity=1,
        secondary_workers=1,
        secondary_pending_limit=1,
    )
    release = threading.Event()
    try:
        first = coordinator.submit_secondary(
            "request-one", lambda: release.wait(1.0), timeout_seconds=0
        )
        with pytest.raises(DatabaseWorkCapacityExceeded):
            coordinator.submit_secondary(
                "request-two", lambda: None, timeout_seconds=0.01
            )
        release.set()
        assert first.result(timeout=1.0) is True
        snapshot = coordinator.snapshot()
        assert snapshot["secondary_rejected"] == 1
        assert snapshot["secondary_peak_active"] == 1
    finally:
        release.set()
        coordinator.shutdown_for_tests()


def test_drain_cancels_queued_work_and_waits_for_running_work():
    coordinator = _coordinator(
        application_capacity=1,
        secondary_workers=1,
        secondary_pending_limit=2,
    )
    release = threading.Event()
    timer = threading.Timer(0.03, release.set)
    try:
        coordinator.submit_secondary(
            "request-drain", lambda: release.wait(1.0), timeout_seconds=0
        )
        queued = coordinator.submit_secondary(
            "request-drain", lambda: "must-not-run", timeout_seconds=0
        )
        timer.start()
        result = coordinator.drain_secondary(
            "request-drain", timeout_seconds=0.3
        )
        assert result.remaining == 0
        assert result.cancelled == 1
        assert queued.cancelled() is True
        assert coordinator.snapshot()["secondary_outstanding"] == 0
    finally:
        release.set()
        timer.cancel()
        coordinator.shutdown_for_tests()


def test_uncooperative_secondary_work_is_detected_not_hidden():
    coordinator = _coordinator(
        application_capacity=1,
        secondary_workers=1,
        secondary_pending_limit=1,
    )
    try:
        future = coordinator.submit_secondary(
            "request-orphan", lambda: time.sleep(0.05), timeout_seconds=0
        )
        result = coordinator.drain_secondary(
            "request-orphan", timeout_seconds=0
        )
        assert result.remaining == 1
        assert coordinator.snapshot()["secondary_orphaned"] == 1
        future.result(timeout=1.0)
        assert coordinator.snapshot()["secondary_outstanding"] == 0
    finally:
        coordinator.shutdown_for_tests()


def test_pipeline_has_no_per_request_thread_pool():
    pipeline_source = (
        Path(__file__).resolve().parents[1] / "agent" / "pipeline.py"
    ).read_text(encoding="utf-8")
    assert "ThreadPoolExecutor(" not in pipeline_source


def test_primary_and_secondary_work_share_the_application_budget():
    coordinator = _coordinator(
        application_capacity=2,
        secondary_workers=1,
        secondary_pending_limit=1,
    )
    secondary_started = threading.Event()
    release = threading.Event()

    def _secondary_call():
        with coordinator.admission(operation="secondary", timeout_seconds=0):
            secondary_started.set()
            return release.wait(1.0)

    try:
        with coordinator.admission(operation="primary", timeout_seconds=0):
            future = coordinator.submit_secondary(
                "request-shared-budget",
                _secondary_call,
                timeout_seconds=0,
            )
            assert secondary_started.wait(0.5)
            with pytest.raises(DatabaseWorkCapacityExceeded):
                with coordinator.admission(operation="third", timeout_seconds=0):
                    pass
            assert coordinator.snapshot()["active_application"] == 2
            release.set()
            assert future.result(timeout=1.0) is True
        assert coordinator.snapshot()["peak_application"] == 2
        assert coordinator.snapshot()["active_application"] == 0
    finally:
        release.set()
        coordinator.shutdown_for_tests()
