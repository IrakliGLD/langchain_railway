"""Lease-safe report worker lifecycle tests."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from contracts.report_jobs import ReportJobLease, ReportJobPhase
from core.report_job_worker import (
    ReportJobFailure,
    ReportJobWorker,
)


def _lease(*, cancel_requested: bool = False) -> ReportJobLease:
    return ReportJobLease.model_validate(
        {
            "contract_version": "report-job-v1",
            "job_id": str(uuid4()),
            "request_id": "report:req-1",
            "actor_user_id": str(uuid4()),
            "query": "Prepare the report.",
            "attempt_count": 1,
            "max_attempts": 3,
            "lease_owner": "worker-1",
            "lease_expires_at": datetime.now(timezone.utc) + timedelta(minutes=2),
            "phase": "planning",
            "progress_percent": 5,
            "cancel_requested": cancel_requested,
            "checkpoint": None,
        }
    )


class _Repository:
    def __init__(
        self,
        lease: ReportJobLease | None,
        *,
        cancellation_requested: bool = False,
    ):
        self.next_lease = lease
        self.should_cancel = cancellation_requested
        self.calls: list[tuple] = []

    def lease_next(self, *, worker_id: str, lease_seconds: int):
        self.calls.append(("lease", worker_id, lease_seconds))
        lease, self.next_lease = self.next_lease, None
        return lease

    def heartbeat(
        self,
        *,
        job_id,
        worker_id: str,
        phase: ReportJobPhase,
        progress_percent: int,
        checkpoint,
        lease_seconds: int,
    ) -> bool:
        self.calls.append(
            (
                "heartbeat",
                job_id,
                worker_id,
                phase,
                progress_percent,
                checkpoint,
                lease_seconds,
            )
        )
        return True

    def complete(self, *, job_id, worker_id: str, result: dict) -> bool:
        self.calls.append(("complete", job_id, worker_id, result))
        return True

    def fail(
        self,
        *,
        job_id,
        worker_id: str,
        error_code: str,
        retryable: bool,
        retry_delay_seconds: int,
    ) -> bool:
        self.calls.append(
            (
                "fail",
                job_id,
                worker_id,
                error_code,
                retryable,
                retry_delay_seconds,
            )
        )
        return True

    def acknowledge_cancellation(self, *, job_id, worker_id: str) -> bool:
        self.calls.append(("cancel", job_id, worker_id))
        return True

    def cancellation_requested(self, *, job_id, worker_id: str) -> bool:
        self.calls.append(("cancel_requested", job_id, worker_id))
        return self.should_cancel


def _worker(repository: _Repository) -> ReportJobWorker:
    return ReportJobWorker(
        repository=repository,
        worker_id="worker-1",
        lease_seconds=120,
        retry_delay_seconds=30,
        poll_interval_seconds=0.01,
        logger=logging.getLogger("test.report_worker"),
    )


def test_worker_returns_without_processing_when_no_job_is_available():
    repository = _Repository(None)
    handled = []

    assert _worker(repository).run_once(lambda *_: handled.append(True)) is False
    assert handled == []
    assert repository.calls == [("lease", "worker-1", 120)]


def test_worker_exposes_heartbeat_checkpoint_and_completes_with_owned_lease():
    lease = _lease()
    repository = _Repository(lease)

    def handler(received, control):
        assert received == lease
        assert control.heartbeat(
            phase=ReportJobPhase.GENERATING_SECTIONS,
            progress_percent=50,
            checkpoint={"completed_section_ids": ["executive_summary"]},
        )
        return {"contract_version": "report-result-v1", "content": "Complete report."}

    assert _worker(repository).run_once(handler) is True

    assert repository.calls[0] == ("lease", "worker-1", 120)
    assert repository.calls[1][0] == "heartbeat"
    assert repository.calls[1][3:6] == (
        ReportJobPhase.GENERATING_SECTIONS,
        50,
        {"completed_section_ids": ["executive_summary"]},
    )
    assert repository.calls[2][0] == "cancel_requested"
    assert repository.calls[3][0] == "complete"


def test_worker_acknowledges_cancellation_without_calling_handler():
    lease = _lease(cancel_requested=True)
    repository = _Repository(lease)
    handled = []

    assert _worker(repository).run_once(lambda *_: handled.append(True)) is True

    assert handled == []
    assert repository.calls[-1] == ("cancel", lease.job_id, "worker-1")


def test_worker_honours_cancellation_that_arrives_before_completion():
    lease = _lease()
    repository = _Repository(lease, cancellation_requested=True)

    assert _worker(repository).run_once(
        lambda *_: {
            "contract_version": "report-result-v1",
            "content": "Do not complete this cancelled result.",
        }
    )

    assert [call[0] for call in repository.calls] == [
        "lease",
        "cancel_requested",
        "cancel",
    ]


def test_worker_maps_expected_failure_to_bounded_retry_metadata():
    lease = _lease()
    repository = _Repository(lease)

    def handler(*_):
        raise ReportJobFailure("REPORT_EVIDENCE_TEMPORARILY_UNAVAILABLE", retryable=True)

    assert _worker(repository).run_once(handler) is True
    assert repository.calls[-1] == (
        "fail",
        lease.job_id,
        "worker-1",
        "REPORT_EVIDENCE_TEMPORARILY_UNAVAILABLE",
        True,
        30,
    )


def test_worker_redacts_unexpected_exception_content_from_persistence():
    lease = _lease()
    repository = _Repository(lease)

    def handler(*_):
        raise RuntimeError("provider secret must not be persisted")

    assert _worker(repository).run_once(handler) is True
    assert repository.calls[-1] == (
        "fail",
        lease.job_id,
        "worker-1",
        "REPORT_WORKER_UNEXPECTED",
        True,
        30,
    )


def test_run_until_stopped_uses_interruptible_wait():
    repository = _Repository(None)
    worker = _worker(repository)
    stop_event = threading.Event()
    stop_event.set()

    worker.run_until_stopped(lambda *_: {}, stop_event=stop_event)

    assert repository.calls == []
