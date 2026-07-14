"""Request-wide deadline primitives for the trusted chat gateway path."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

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
