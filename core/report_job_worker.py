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
        # Upper bound clears the maximum job timeout by the safety margin the
        # worker entrypoint requires, so every accepted timeout has a valid lease.
        if not 30 <= lease_seconds <= 3630:
            raise ValueError("Report lease_seconds must be between 30 and 3630.")
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
        self._active_lease_lock = threading.Lock()
        self._active_lease: ReportJobLease | None = None
        self._shutdown_handoff_in_progress = False

    @staticmethod
    def _stop_requested(stop_event: threading.Event | None) -> bool:
        return stop_event is not None and stop_event.is_set()

    def _set_active_lease(self, lease: ReportJobLease) -> None:
        with self._active_lease_lock:
            self._active_lease = lease

    def _clear_active_lease(self, lease: ReportJobLease) -> None:
        with self._active_lease_lock:
            if self._active_lease is lease:
                self._active_lease = None

    def handoff_active_job_for_shutdown(self) -> bool:
        """Release the owned job for retry while an active handler winds down."""

        with self._active_lease_lock:
            lease = self._active_lease
            if lease is None or self._shutdown_handoff_in_progress:
                return False
            self._shutdown_handoff_in_progress = True

        handed_off = False
        try:
            handed_off = self._repository.fail(
                job_id=lease.job_id,
                worker_id=self._worker_id,
                error_code="REPORT_WORKER_STOPPING",
                retryable=True,
                retry_delay_seconds=self._retry_delay_seconds,
            )
            if handed_off:
                self._logger.info(
                    "Report job handed off during worker shutdown: job_id=%s",
                    lease.job_id,
                )
            else:
                self._logger.error(
                    "Report job shutdown handoff was not persisted: job_id=%s",
                    lease.job_id,
                )
            return handed_off
        except Exception as exc:
            self._logger.error(
                "Report job shutdown handoff failed: job_id=%s exception_type=%s",
                lease.job_id,
                type(exc).__name__,
            )
            return False
        finally:
            with self._active_lease_lock:
                self._shutdown_handoff_in_progress = False
                if handed_off and self._active_lease is lease:
                    self._active_lease = None

    def run_once(
        self,
        handler: ReportJobHandler,
        *,
        stop_event: threading.Event | None = None,
    ) -> bool:
        if self._stop_requested(stop_event):
            return False
        lease = self._repository.lease_next(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if lease is None:
            return False
        if lease.lease_owner != self._worker_id:
            raise RuntimeError("Report repository returned a lease owned by another worker.")

        self._set_active_lease(lease)
        try:
            if lease.cancel_requested:
                self._repository.acknowledge_cancellation(
                    job_id=lease.job_id,
                    worker_id=self._worker_id,
                )
                return True
            if self._stop_requested(stop_event):
                self.handoff_active_job_for_shutdown()
                return True

            control = ReportJobExecutionControl(
                repository=self._repository,
                lease=lease,
                lease_seconds=self._lease_seconds,
            )
            try:
                result = handler(lease, control)
            except ReportJobFailure as exc:
                if self._stop_requested(stop_event):
                    self.handoff_active_job_for_shutdown()
                    return True
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
                return True
            except Exception as exc:
                if self._stop_requested(stop_event):
                    self.handoff_active_job_for_shutdown()
                    return True
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

            if self._stop_requested(stop_event):
                self.handoff_active_job_for_shutdown()
                return True
            if not isinstance(result, dict):
                self._repository.fail(
                    job_id=lease.job_id,
                    worker_id=self._worker_id,
                    error_code="REPORT_RESULT_INVALID",
                    retryable=False,
                    retry_delay_seconds=self._retry_delay_seconds,
                )
                return True
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
            return True
        finally:
            self._clear_active_lease(lease)

    def run_until_stopped(
        self,
        handler: ReportJobHandler,
        *,
        stop_event: threading.Event,
    ) -> None:
        consecutive_failures = 0
        while not stop_event.is_set():
            try:
                processed = self.run_once(
                    handler,
                    stop_event=stop_event,
                )
            except Exception as exc:
                consecutive_failures += 1
                retry_delay_seconds = min(
                    60.0,
                    self._poll_interval_seconds
                    * (2 ** min(consecutive_failures - 1, 10)),
                )
                self._logger.error(
                    "Report worker loop recovered: exception_type=%s "
                    "consecutive_failures=%s retry_delay_seconds=%.2f",
                    type(exc).__name__,
                    consecutive_failures,
                    retry_delay_seconds,
                )
                stop_event.wait(retry_delay_seconds)
                continue
            consecutive_failures = 0
            if not processed:
                stop_event.wait(self._poll_interval_seconds)
