"""Process-wide admission and secondary-work coordination for database I/O."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, TypeVar

from config import (
    DB_APPLICATION_CONCURRENCY,
    DB_CONTROL_RESERVED_SLOTS,
    DB_QUEUE_TIMEOUT_MS,
    DB_SECONDARY_PENDING_LIMIT,
    DB_SECONDARY_WORKERS,
)

DatabaseWorkPriority = Literal["application", "control"]
_ResultT = TypeVar("_ResultT")


class DatabaseWorkCapacityExceeded(TimeoutError):
    """Raised when bounded database admission cannot be acquired in time."""

    def __init__(self, stage: str, priority: str) -> None:
        super().__init__(f"database capacity unavailable before {stage}")
        self.stage = stage
        self.priority = priority


@dataclass(frozen=True, slots=True)
class SecondaryDrainResult:
    cancelled: int
    completed: int
    remaining: int


class DatabaseWorkCoordinator:
    """Own all process-local DB admission and asynchronous secondary work.

    Application and control slots are separate so request saturation cannot
    consume the capacity reserved for readiness and runtime-identity probes.
    The secondary executor is process-wide and its pending queue is bounded.
    """

    def __init__(
        self,
        *,
        application_capacity: int,
        control_capacity: int,
        queue_timeout_seconds: float,
        secondary_workers: int,
        secondary_pending_limit: int,
    ) -> None:
        if application_capacity < 1:
            raise ValueError("application_capacity must be positive")
        if control_capacity < 1:
            raise ValueError("control_capacity must be positive")
        if secondary_workers < 1 or secondary_workers > application_capacity:
            raise ValueError("secondary_workers must fit within application capacity")
        if secondary_pending_limit < secondary_workers:
            raise ValueError("secondary_pending_limit must cover every worker")

        self.application_capacity = int(application_capacity)
        self.control_capacity = int(control_capacity)
        self.queue_timeout_seconds = max(0.0, float(queue_timeout_seconds))
        self.secondary_workers = int(secondary_workers)
        self.secondary_pending_limit = int(secondary_pending_limit)
        self._slots = {
            "application": threading.BoundedSemaphore(self.application_capacity),
            "control": threading.BoundedSemaphore(self.control_capacity),
        }
        self._executor = ThreadPoolExecutor(
            max_workers=self.secondary_workers,
            thread_name_prefix="db-secondary",
        )
        self._secondary_pending_slots = threading.BoundedSemaphore(
            self.secondary_pending_limit
        )
        self._lock = threading.Lock()
        self._active = {"application": 0, "control": 0}
        self._peak_active = {"application": 0, "control": 0}
        self._accepted = {"application": 0, "control": 0}
        self._rejected = {"application": 0, "control": 0}
        self._queue_wait_seconds = {"application": 0.0, "control": 0.0}
        self._queue_wait_max_seconds = {"application": 0.0, "control": 0.0}
        self._secondary_by_request: dict[str, set[Future]] = {}
        self._secondary_active = 0
        self._secondary_peak_active = 0
        self._secondary_submitted = 0
        self._secondary_rejected = 0
        self._secondary_completed = 0
        self._secondary_cancelled = 0
        self._secondary_orphaned = 0
        self._orphan_reported: set[Future] = set()

    @contextmanager
    def admission(
        self,
        *,
        operation: str,
        priority: DatabaseWorkPriority = "application",
        timeout_seconds: float | None = None,
    ) -> Iterator[None]:
        if priority not in self._slots:
            raise ValueError(f"unsupported database priority: {priority}")
        timeout = self.queue_timeout_seconds if timeout_seconds is None else max(
            0.0, float(timeout_seconds)
        )
        started = time.monotonic()
        acquired = self._slots[priority].acquire(timeout=timeout)
        waited = max(0.0, time.monotonic() - started)
        with self._lock:
            self._queue_wait_seconds[priority] += waited
            self._queue_wait_max_seconds[priority] = max(
                self._queue_wait_max_seconds[priority], waited
            )
            if acquired:
                self._accepted[priority] += 1
                self._active[priority] += 1
                self._peak_active[priority] = max(
                    self._peak_active[priority], self._active[priority]
                )
            else:
                self._rejected[priority] += 1
        if not acquired:
            raise DatabaseWorkCapacityExceeded(
                f"db_{operation}_queue", priority
            )
        try:
            yield
        finally:
            with self._lock:
                self._active[priority] -= 1
            self._slots[priority].release()

    def submit_secondary(
        self,
        request_key: str,
        call: Callable[[], _ResultT],
        *,
        timeout_seconds: float,
    ) -> Future[_ResultT]:
        """Submit one bounded secondary task associated with its parent request."""
        if not request_key:
            raise ValueError("request_key is required for secondary database work")
        acquired = self._secondary_pending_slots.acquire(
            timeout=max(0.0, float(timeout_seconds))
        )
        if not acquired:
            with self._lock:
                self._secondary_rejected += 1
            raise DatabaseWorkCapacityExceeded(
                "db_secondary_submission_queue", "application"
            )

        start_gate = threading.Event()

        def _run():
            start_gate.wait()
            with self._lock:
                self._secondary_active += 1
                self._secondary_peak_active = max(
                    self._secondary_peak_active, self._secondary_active
                )
            try:
                return call()
            finally:
                with self._lock:
                    self._secondary_active -= 1

        try:
            future = self._executor.submit(_run)
        except BaseException:
            self._secondary_pending_slots.release()
            raise

        with self._lock:
            self._secondary_submitted += 1
            self._secondary_by_request.setdefault(request_key, set()).add(future)

        def _complete(completed: Future) -> None:
            with self._lock:
                request_futures = self._secondary_by_request.get(request_key)
                if request_futures is not None:
                    request_futures.discard(completed)
                    if not request_futures:
                        self._secondary_by_request.pop(request_key, None)
                self._orphan_reported.discard(completed)
                if completed.cancelled():
                    self._secondary_cancelled += 1
                else:
                    self._secondary_completed += 1
            self._secondary_pending_slots.release()

        future.add_done_callback(_complete)
        start_gate.set()
        return future

    def drain_secondary(
        self,
        request_key: str,
        *,
        timeout_seconds: float,
    ) -> SecondaryDrainResult:
        """Cancel queued work and wait a bounded interval for running work."""
        with self._lock:
            futures = set(self._secondary_by_request.get(request_key, set()))
        cancelled = sum(1 for future in futures if future.cancel())
        done, not_done = wait(futures, timeout=max(0.0, float(timeout_seconds)))
        with self._lock:
            # Future callbacks acquire this same lock. Re-check completion while
            # holding it so a task finishing at the timeout boundary is never
            # falsely reported (or retained) as an orphan.
            still_running = {future for future in not_done if not future.done()}
            new_orphans = still_running - self._orphan_reported
            self._orphan_reported.update(new_orphans)
            self._secondary_orphaned += len(new_orphans)
        return SecondaryDrainResult(
            cancelled=cancelled,
            completed=len(done) + len(not_done - still_running),
            remaining=len(still_running),
        )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            outstanding = sum(len(items) for items in self._secondary_by_request.values())
            return {
                "application_capacity": self.application_capacity,
                "control_capacity": self.control_capacity,
                "queue_timeout_seconds": self.queue_timeout_seconds,
                "active_application": self._active["application"],
                "active_control": self._active["control"],
                "peak_application": self._peak_active["application"],
                "peak_control": self._peak_active["control"],
                "accepted_application": self._accepted["application"],
                "accepted_control": self._accepted["control"],
                "rejected_application": self._rejected["application"],
                "rejected_control": self._rejected["control"],
                "max_queue_wait_application_seconds": round(
                    self._queue_wait_max_seconds["application"], 6
                ),
                "max_queue_wait_control_seconds": round(
                    self._queue_wait_max_seconds["control"], 6
                ),
                "secondary_workers": self.secondary_workers,
                "secondary_pending_limit": self.secondary_pending_limit,
                "secondary_active": self._secondary_active,
                "secondary_peak_active": self._secondary_peak_active,
                "secondary_outstanding": outstanding,
                "secondary_submitted": self._secondary_submitted,
                "secondary_rejected": self._secondary_rejected,
                "secondary_completed": self._secondary_completed,
                "secondary_cancelled": self._secondary_cancelled,
                "secondary_orphaned": self._secondary_orphaned,
            }

    def shutdown_for_tests(self) -> None:
        """Release test-owned executor threads; never used by application code."""
        self._executor.shutdown(wait=True, cancel_futures=True)


db_work_coordinator = DatabaseWorkCoordinator(
    application_capacity=DB_APPLICATION_CONCURRENCY,
    control_capacity=DB_CONTROL_RESERVED_SLOTS,
    queue_timeout_seconds=DB_QUEUE_TIMEOUT_MS / 1000.0,
    secondary_workers=DB_SECONDARY_WORKERS,
    secondary_pending_limit=DB_SECONDARY_PENDING_LIMIT,
)
