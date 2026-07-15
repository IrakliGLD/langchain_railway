"""Thread-safe in-process rate-limit state behind one stable interface.

The repository is deliberately process-local.  P7/P8 deployment policy keeps
the backend at one worker/replica until this interface has a shared-store
implementation with explicit TTL and failure semantics.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable


class InMemoryRateLimitRepository:
    """Own sliding-window subject state without leaking keys to callers."""

    def __init__(
        self,
        *,
        namespaces: Iterable[str] = ("preauth", "gateway", "user"),
        sweep_interval_seconds: float = 300.0,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        normalized = tuple(sorted({str(name).strip() for name in namespaces if str(name).strip()}))
        if not normalized:
            raise ValueError("At least one rate-limit namespace is required")
        self._buckets: dict[str, dict[str, list[float]]] = {
            name: {} for name in normalized
        }
        self._locks = {name: threading.Lock() for name in normalized}
        self._last_sweep = {name: 0.0 for name in normalized}
        self._sweep_interval_seconds = float(sweep_interval_seconds)
        self._time_fn = time_fn

    def consume(
        self,
        namespace: str,
        subject_id: str,
        *,
        max_requests: int,
        window_seconds: float = 60.0,
    ) -> bool:
        """Consume one request and return whether it fits the subject budget."""

        buckets = self._buckets[namespace]
        now = self._time_fn()
        with self._locks[namespace]:
            if now - self._last_sweep[namespace] > self._sweep_interval_seconds:
                self._last_sweep[namespace] = now
                stale = [
                    key
                    for key, stamps in buckets.items()
                    if not stamps or now - stamps[-1] >= window_seconds
                ]
                for key in stale:
                    del buckets[key]

            timestamps = [
                stamp
                for stamp in buckets.get(subject_id, [])
                if now - stamp < window_seconds
            ]
            if len(timestamps) >= max_requests:
                buckets[subject_id] = timestamps
                return False
            timestamps.append(now)
            buckets[subject_id] = timestamps
            return True

    def clear(self) -> None:
        """Remove all subject state; intended for tests and controlled resets."""

        for namespace, buckets in self._buckets.items():
            with self._locks[namespace]:
                buckets.clear()

    def protected_snapshot(self) -> dict[str, object]:
        """Expose aggregate topology/counts, never subject identifiers."""

        counts: dict[str, int] = {}
        for namespace, buckets in self._buckets.items():
            with self._locks[namespace]:
                counts[namespace] = len(buckets)
        return {
            "backend": "in_memory",
            "shared_across_processes": False,
            "subjects_by_namespace": counts,
        }
