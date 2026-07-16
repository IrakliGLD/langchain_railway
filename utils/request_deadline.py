"""Request-wide deadline primitives for the trusted chat gateway path."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

MINIMUM_START_BUDGET_MS = 250


class InvalidRequestBudget(ValueError):
    """Raised when the gateway budget header is not an integer."""


class RequestDeadlineExceeded(TimeoutError):
    """Raised when a request has no remaining execution budget."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"request deadline exceeded before {stage}")
        self.stage = stage


@dataclass(frozen=True, slots=True)
class RequestDeadline:
    budget_ms: int
    started_monotonic: float
    deadline_monotonic: float
    source: str
    retry_owner: str = "backend"

    @classmethod
    def from_budget_ms(
        cls,
        *,
        budget_ms: int,
        now_monotonic: float | None = None,
        source: str,
    ) -> "RequestDeadline":
        started = time.monotonic() if now_monotonic is None else now_monotonic
        return cls(
            budget_ms=budget_ms,
            started_monotonic=started,
            deadline_monotonic=started + (budget_ms / 1000.0),
            source=source,
        )

    def remaining_seconds(self, now_monotonic: float | None = None) -> float:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return max(0.0, self.deadline_monotonic - now)

    def remaining_ms(self, now_monotonic: float | None = None) -> int:
        return max(0, math.floor(self.remaining_seconds(now_monotonic) * 1000))

    def ensure_remaining(self, stage: str, *, minimum_ms: int = 1) -> None:
        if self.remaining_ms() < minimum_ms:
            raise RequestDeadlineExceeded(stage)

    def bounded_timeout_ms(
        self,
        stage: str,
        *,
        configured_timeout_ms: int,
        cleanup_allowance_ms: int,
        minimum_start_ms: int = MINIMUM_START_BUDGET_MS,
    ) -> int:
        """Return the safe external-call budget or fail before starting it."""
        available_ms = self.remaining_ms() - max(0, cleanup_allowance_ms)
        if available_ms < minimum_start_ms:
            raise RequestDeadlineExceeded(stage)
        return min(max(1, int(configured_timeout_ms)), available_ms)

    def bounded_timeout_seconds(
        self,
        stage: str,
        *,
        configured_timeout_seconds: float,
        cleanup_allowance_ms: int,
        minimum_start_ms: int = MINIMUM_START_BUDGET_MS,
    ) -> float:
        timeout_ms = self.bounded_timeout_ms(
            stage,
            configured_timeout_ms=max(1, math.floor(configured_timeout_seconds * 1000)),
            cleanup_allowance_ms=cleanup_allowance_ms,
            minimum_start_ms=minimum_start_ms,
        )
        return timeout_ms / 1000.0

    def public_metadata(self) -> dict[str, int | str]:
        return {
            "budget_ms": self.budget_ms,
            "remaining_ms": self.remaining_ms(),
            "retry_owner": self.retry_owner,
        }


def build_request_deadline(
    raw_budget_ms: str | None,
    *,
    default_budget_ms: int,
    maximum_budget_ms: int,
    now_monotonic: float | None = None,
) -> RequestDeadline:
    if raw_budget_ms is None:
        budget_ms = default_budget_ms
        source = "backend_default"
    else:
        try:
            budget_ms = int(raw_budget_ms.strip())
        except (AttributeError, ValueError) as exc:
            raise InvalidRequestBudget("request budget must be an integer") from exc
        if budget_ms < 0:
            raise InvalidRequestBudget("request budget must not be negative")
        budget_ms = min(budget_ms, maximum_budget_ms)
        source = "gateway"

    return RequestDeadline.from_budget_ms(
        budget_ms=budget_ms,
        now_monotonic=now_monotonic,
        source=source,
    )

@dataclass(frozen=True, slots=True)
class RequestExecutionScope:
    """Privacy-safe identity and deadline propagated to deep I/O boundaries."""

    deadline: RequestDeadline | None
    request_id: str
    actor_binding: str


_REQUEST_EXECUTION_SCOPE: ContextVar[RequestExecutionScope | None] = ContextVar(
    "enai_request_execution_scope", default=None
)


def current_request_execution_scope() -> RequestExecutionScope | None:
    return _REQUEST_EXECUTION_SCOPE.get()


@contextmanager
def bind_request_execution_scope(
    *,
    deadline: RequestDeadline | None,
    request_id: str = "",
    actor_id: str = "",
) -> Iterator[RequestExecutionScope]:
    # Import lazily so this dependency-light primitive does not create a
    # config/privacy import cycle during application startup.
    from utils.privacy_logging import hash_private_identifier

    scope = RequestExecutionScope(
        deadline=deadline,
        request_id=str(request_id or ""),
        actor_binding=hash_private_identifier(actor_id, namespace="actor"),
    )
    token = _REQUEST_EXECUTION_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _REQUEST_EXECUTION_SCOPE.reset(token)
