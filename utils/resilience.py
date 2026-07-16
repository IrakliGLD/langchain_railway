"""
Reliability primitives: circuit breakers and backpressure gate.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Tuple

from config import (
    ASK_BACKPRESSURE_TIMEOUT_SECONDS,
    ASK_MAX_CONCURRENT_REQUESTS,
    DB_CB_FAILURE_THRESHOLD,
    DB_CB_RESET_TIMEOUT_SECONDS,
    LLM_CB_FAILURE_THRESHOLD,
    LLM_CB_RESET_TIMEOUT_SECONDS,
)


class CircuitBreaker:
    """Simple thread-safe circuit breaker with half-open probe."""

    def __init__(self, name: str, failure_threshold: int, reset_timeout_seconds: int):
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.reset_timeout_seconds = max(1, int(reset_timeout_seconds))
        self._state = "closed"  # closed | open | half_open
        self._failure_count = 0
        self._opened_at = 0.0
        self._half_open_probe_in_flight = False
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Current state (``closed`` | ``open`` | ``half_open``), lock-guarded.

        Prefer this over reading ``_state`` directly so callers never touch the
        private attribute without the lock.
        """
        with self._lock:
            return self._state

    def _evaluate(self, now_ts: float) -> Tuple[bool, str, bool]:
        """Pure decision for the current state; caller must hold ``self._lock``.

        Returns ``(allowed, reason, starts_probe)``. ``starts_probe`` is True when
        *committing* this decision consumes the single half-open probe slot.
        ``allow_request`` commits the decision; ``would_allow`` discards it — so
        both share one code path and cannot drift.
        """
        if self._state == "open":
            if now_ts - self._opened_at >= self.reset_timeout_seconds:
                # Reset window elapsed: the next real acquire probes once.
                return True, "half_open_probe", True
            return False, "open", False
        if self._state == "half_open":
            if self._half_open_probe_in_flight:
                return False, "half_open_busy", False
            return True, "half_open_probe", True
        return True, "closed", False

    def allow_request(self) -> Tuple[bool, str]:
        """Acquire permission to proceed, consuming the half-open probe slot.

        This is the *guarded-call* entry point: a caller granted ``allowed=True``
        in half-open state has claimed the sole probe and MUST report the outcome
        via ``record_success``/``record_failure``. An advisory check that will not
        itself make the guarded call must use ``would_allow`` instead — otherwise
        the probe slot is claimed but never released and recovery wedges until
        process restart.
        """
        now_ts = time.time()
        with self._lock:
            allowed, reason, starts_probe = self._evaluate(now_ts)
            if allowed and starts_probe:
                self._state = "half_open"
                self._half_open_probe_in_flight = True
            return allowed, reason

    def would_allow(self) -> Tuple[bool, str]:
        """Read-only preview of ``allow_request`` that mutates no state.

        Use for advisory pre-checks (e.g. "should I attempt tool execution?")
        that do NOT themselves make the guarded DB call. Reading this never
        consumes the half-open probe, so an advisory caller cannot starve the
        real probe owner.
        """
        now_ts = time.time()
        with self._lock:
            allowed, reason, _starts_probe = self._evaluate(now_ts)
            return allowed, reason

    def record_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._opened_at = 0.0
            self._half_open_probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = time.time()
                self._failure_count = self.failure_threshold
                self._half_open_probe_in_flight = False
                return

            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.time()
                self._half_open_probe_in_flight = False

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            open_for_seconds = 0.0
            if self._state == "open" and self._opened_at > 0:
                open_for_seconds = max(0.0, time.time() - self._opened_at)
            return {
                "name": self.name,
                "state": self._state,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "reset_timeout_seconds": self.reset_timeout_seconds,
                "open_for_seconds": round(open_for_seconds, 3),
                "half_open_probe_in_flight": self._half_open_probe_in_flight,
            }


class RequestBackpressureGate:
    """Concurrency gate with optional queue timeout for load shedding."""

    def __init__(self, max_concurrent: int, wait_timeout_seconds: float):
        self.max_concurrent = max(1, int(max_concurrent))
        self.wait_timeout_seconds = max(0.0, float(wait_timeout_seconds))
        self._semaphore = threading.BoundedSemaphore(self.max_concurrent)
        self._lock = threading.Lock()
        self._in_flight = 0
        self._rejected = 0
        self._accepted = 0

    def try_acquire(self) -> bool:
        if self.wait_timeout_seconds > 0:
            acquired = self._semaphore.acquire(timeout=self.wait_timeout_seconds)
        else:
            acquired = self._semaphore.acquire(blocking=False)

        with self._lock:
            if acquired:
                self._in_flight += 1
                self._accepted += 1
            else:
                self._rejected += 1
        return acquired

    def release(self) -> bool:
        with self._lock:
            if self._in_flight <= 0:
                return False
            self._in_flight -= 1
        self._semaphore.release()
        return True

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "max_concurrent": self.max_concurrent,
                "wait_timeout_seconds": self.wait_timeout_seconds,
                "in_flight": self._in_flight,
                "accepted": self._accepted,
                "rejected": self._rejected,
            }


# One breaker per provider. NVIDIA has its own (P5.4, finding M11): before this
# it fell through get_llm_breaker's unknown-key default to the OpenAI breaker,
# so an NVIDIA outage could open/reset OpenAI's state and vice versa. Keys must
# match _provider_from_model_name()'s provider keys in core/llm.py.
_llm_breakers = {
    "openai": CircuitBreaker(
        name="llm_openai",
        failure_threshold=LLM_CB_FAILURE_THRESHOLD,
        reset_timeout_seconds=LLM_CB_RESET_TIMEOUT_SECONDS,
    ),
    "gemini": CircuitBreaker(
        name="llm_gemini",
        failure_threshold=LLM_CB_FAILURE_THRESHOLD,
        reset_timeout_seconds=LLM_CB_RESET_TIMEOUT_SECONDS,
    ),
    "nvidia": CircuitBreaker(
        name="llm_nvidia",
        failure_threshold=LLM_CB_FAILURE_THRESHOLD,
        reset_timeout_seconds=LLM_CB_RESET_TIMEOUT_SECONDS,
    ),
}

db_circuit_breaker = CircuitBreaker(
    name="database",
    failure_threshold=DB_CB_FAILURE_THRESHOLD,
    reset_timeout_seconds=DB_CB_RESET_TIMEOUT_SECONDS,
)

request_backpressure_gate = RequestBackpressureGate(
    max_concurrent=ASK_MAX_CONCURRENT_REQUESTS,
    wait_timeout_seconds=ASK_BACKPRESSURE_TIMEOUT_SECONDS,
)


def get_llm_breaker(provider: str) -> CircuitBreaker:
    key = (provider or "").strip().lower()
    if key not in _llm_breakers:
        key = "openai"
    return _llm_breakers[key]


def get_resilience_snapshot() -> Dict[str, object]:
    # Import lazily to avoid a config/resilience/coordinator startup cycle.
    from core.db_work_coordinator import db_work_coordinator

    return {
        "llm_breakers": {k: v.snapshot() for k, v in _llm_breakers.items()},
        "db_breaker": db_circuit_breaker.snapshot(),
        "db_work": db_work_coordinator.snapshot(),
        "request_backpressure": request_backpressure_gate.snapshot(),
    }
