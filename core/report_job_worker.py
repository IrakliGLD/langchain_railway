"""Lease-safe, persistence-agnostic runtime for durable report jobs."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from typing import Any, Protocol
from uuid import UUID

from contracts.report_jobs import ReportJobLease, ReportJobPhase

_ERROR_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class ReportJobRepository(Protocol):
    def lease_next(self, *, worker_id: str, lease_seconds: int) -> ReportJobLease | None: ...

    def heartbeat(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        phase: ReportJobPhase,
        progress_percent: int,
        checkpoint: dict[str, Any] | None,
        lease_seconds: int,
    ) -> bool: ...

    def complete(self, *, job_id: UUID, worker_id: str, result: dict[str, Any]) -> bool: ...

    def fail(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        error_code: str,
        retryable: bool,
        retry_delay_seconds: int,
    ) -> bool: ...

    def acknowledge_cancellation(self, *, job_id: UUID, worker_id: str) -> bool: ...

    def cancellation_requested(self, *, job_id: UUID, worker_id: str) -> bool: ...


class ReportJobFailure(RuntimeError):
    """Expected bounded failure metadata; never contains provider error text."""

    def __init__(self, error_code: str, *, retryable: bool) -> None:
        if not _ERROR_CODE_PATTERN.fullmatch(error_code):
            raise ValueError("Report job error_code must be a bounded uppercase identifier.")
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


class ReportJobExecutionControl:
    """Lease-bound checkpoint and cancellation operations exposed to a handler."""

    def __init__(
        self,
        *,
        repository: ReportJobRepository,
        lease: ReportJobLease,
        lease_seconds: int,
    ) -> None:
        self._repository = repository
        self._lease = lease
        self._lease_seconds = lease_seconds

    def heartbeat(
        self,
        *,
        phase: ReportJobPhase,
        progress_percent: int,
        checkpoint: dict[str, Any] | None = None,
    ) -> bool:
        return self._repository.heartbeat(
            job_id=self._lease.job_id,
            worker_id=self._lease.lease_owner,
            phase=phase,
            progress_percent=progress_percent,
            checkpoint=checkpoint,
            lease_seconds=self._lease_seconds,
        )

    def cancellation_requested(self) -> bool:
        return self._repository.cancellation_requested(
            job_id=self._lease.job_id,
            worker_id=self._lease.lease_owner,
        )


ReportJobHandler = Callable[
    [ReportJobLease, ReportJobExecutionControl],
    dict[str, Any],
]


class ReportJobWorker:
    """Poll one durable queue without owning threads or application startup."""

    def __init__(
        self,
        *,
        repository: ReportJobRepository,
        worker_id: str,
        lease_seconds: int,
        retry_delay_seconds: int,
        poll_interval_seconds: float,
        logger: logging.Logger,
    ) -> None:
        if not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$", worker_id):
            raise ValueError("Invalid report worker_id.")
        if not 30 <= lease_seconds <= 3600:
            raise ValueError("Report lease_seconds must be between 30 and 3600.")
        if not 1 <= retry_delay_seconds <= 3600:
            raise ValueError("Report retry_delay_seconds must be between 1 and 3600.")
        if not 0.01 <= poll_interval_seconds <= 60:
            raise ValueError("Report poll_interval_seconds must be between 0.01 and 60.")
        self._repository = repository
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._logger = logger

    def run_once(self, handler: ReportJobHandler) -> bool:
        lease = self._repository.lease_next(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if lease is None:
            return False
        if lease.lease_owner != self._worker_id:
            raise RuntimeError("Report repository returned a lease owned by another worker.")

        if lease.cancel_requested:
            self._repository.acknowledge_cancellation(
                job_id=lease.job_id,
                worker_id=self._worker_id,
            )
            return True

        control = ReportJobExecutionControl(
            repository=self._repository,
            lease=lease,
            lease_seconds=self._lease_seconds,
        )
        try:
            result = handler(lease, control)
            if not isinstance(result, dict):
                raise ReportJobFailure("REPORT_RESULT_INVALID", retryable=False)
            if control.cancellation_requested():
                self._repository.acknowledge_cancellation(
                    job_id=lease.job_id,
                    worker_id=self._worker_id,
                )
                return True
            self._repository.complete(
                job_id=lease.job_id,
                worker_id=self._worker_id,
                result=result,
            )
        except ReportJobFailure as exc:
            self._logger.warning(
                "Report job attempt failed: job_id=%s error_code=%s retryable=%s",
                lease.job_id,
                exc.error_code,
                exc.retryable,
            )
            self._repository.fail(
                job_id=lease.job_id,
                worker_id=self._worker_id,
                error_code=exc.error_code,
                retryable=exc.retryable,
                retry_delay_seconds=self._retry_delay_seconds,
            )
        except Exception as exc:
            self._logger.error(
                "Unexpected report worker failure: job_id=%s exception_type=%s",
                lease.job_id,
                type(exc).__name__,
            )
            self._repository.fail(
                job_id=lease.job_id,
                worker_id=self._worker_id,
                error_code="REPORT_WORKER_UNEXPECTED",
                retryable=True,
                retry_delay_seconds=self._retry_delay_seconds,
            )
        return True

    def run_until_stopped(
        self,
        handler: ReportJobHandler,
        *,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            processed = self.run_once(handler)
            if not processed:
                stop_event.wait(self._poll_interval_seconds)
