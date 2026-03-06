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

    def allow_request(self) -> Tuple[bool, str]:
        """Return whether request can proceed and decision reason."""
        now_ts = time.time()
        with self._lock:
            if self._state == "open":
                if now_ts - self._opened_at >= self.reset_timeout_seconds:
                    self._state = "half_open"
                    self._half_open_probe_in_flight = False
                else:
                    return False, "open"

            if self._state == "half_open":
                if self._half_open_probe_in_flight:
                    return False, "half_open_busy"
                self._half_open_probe_in_flight = True
                return True, "half_open_probe"

            return True, "closed"

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
    return {
        "llm_breakers": {k: v.snapshot() for k, v in _llm_breakers.items()},
        "db_breaker": db_circuit_breaker.snapshot(),
        "request_backpressure": request_backpressure_gate.snapshot(),
    }
