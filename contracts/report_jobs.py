"""Strict contracts for durable, resumable report jobs.

The persistence state is intentionally smaller than the report-generation
workflow. ``state`` owns queue/worker lifecycle, while ``phase`` communicates
the last durable report checkpoint to clients and retrying workers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPORT_JOB_CONTRACT_VERSION = "report-job-v1"
REPORT_JOB_MAX_QUERY_CHARS = 4000

RequestId = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"),
]
WorkerId = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"),
]
ErrorCode = Annotated[
    str,
    Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$"),
]


class ReportJobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReportJobPhase(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    GENERATING_SECTIONS = "generating_sections"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ACTIVE_PHASE_ORDER = {
    ReportJobPhase.QUEUED: 0,
    ReportJobPhase.PLANNING: 1,
    ReportJobPhase.GENERATING_SECTIONS: 2,
    ReportJobPhase.ASSEMBLING: 3,
}
_TERMINAL_STATES = {
    ReportJobState.COMPLETED,
    ReportJobState.FAILED,
    ReportJobState.CANCELLED,
}


class _StrictReportJobModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ReportJobCreateRequest(_StrictReportJobModel):
    contract_version: Literal["report-job-v1"]
    request_id: RequestId
    query: str = Field(min_length=1, max_length=REPORT_JOB_MAX_QUERY_CHARS)

    @field_validator("query")
    @classmethod
    def _query_must_contain_text(cls, value: str) -> str:
        if not value:
            raise ValueError("Report query must contain text.")
        return value


class ReportJobSnapshot(_StrictReportJobModel):
    """Safe status projection returned to an authenticated report owner."""

    contract_version: Literal["report-job-v1"]
    job_id: UUID
    request_id: RequestId
    state: ReportJobState
    phase: ReportJobPhase
    progress_percent: int = Field(ge=0, le=100)
    attempt_count: int = Field(ge=0, le=10)
    max_attempts: int = Field(ge=1, le=10)
    cancel_requested: bool
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    error_code: ErrorCode | None
    result: Dict[str, Any] | None

    @model_validator(mode="after")
    def _validate_state_projection(self) -> "ReportJobSnapshot":
        if self.attempt_count > self.max_attempts:
            raise ValueError("Report job attempt_count cannot exceed max_attempts.")

        if self.state is ReportJobState.COMPLETED:
            if self.phase is not ReportJobPhase.COMPLETED:
                raise ValueError("A completed report job must use the completed phase.")
            if self.progress_percent != 100:
                raise ValueError("A completed report job must be 100 percent complete.")
            if self.completed_at is None or self.result is None:
                raise ValueError("A completed report job requires completed_at and result.")
            if self.error_code is not None:
                raise ValueError("A completed report job cannot contain an error_code.")
            return self

        if self.state is ReportJobState.FAILED:
            if self.phase is not ReportJobPhase.FAILED:
                raise ValueError("A failed report job must use the failed phase.")
            if self.completed_at is None or self.error_code is None:
                raise ValueError("A failed report job requires completed_at and error_code.")
            if self.result is not None:
                raise ValueError("A failed report job cannot contain a result.")
            return self

        if self.state is ReportJobState.CANCELLED:
            if self.phase is not ReportJobPhase.CANCELLED:
                raise ValueError("A cancelled report job must use the cancelled phase.")
            if self.completed_at is None:
                raise ValueError("A cancelled report job requires completed_at.")
            if self.result is not None or self.error_code is not None:
                raise ValueError("A cancelled report job cannot contain a result or error_code.")
            return self

        if self.completed_at is not None or self.result is not None or self.error_code is not None:
            raise ValueError("A non-terminal report job cannot contain terminal fields.")
        if self.phase not in _ACTIVE_PHASE_ORDER:
            raise ValueError("A non-terminal report job must use an active phase.")
        if self.progress_percent == 100:
            raise ValueError("A non-terminal report job cannot be 100 percent complete.")
        if self.state is ReportJobState.RUNNING:
            if self.attempt_count < 1:
                raise ValueError("A running report job requires a positive attempt_count.")
            if self.phase is ReportJobPhase.QUEUED:
                raise ValueError("A running report job cannot remain in the queued phase.")
        return self


class ReportJobLease(_StrictReportJobModel):
    """Internal worker-only payload returned by the lease RPC."""

    contract_version: Literal["report-job-v1"]
    job_id: UUID
    request_id: RequestId
    actor_user_id: UUID
    query: str = Field(min_length=1, max_length=REPORT_JOB_MAX_QUERY_CHARS)
    attempt_count: int = Field(ge=1, le=10)
    max_attempts: int = Field(ge=1, le=10)
    lease_owner: WorkerId
    lease_expires_at: datetime
    phase: ReportJobPhase
    progress_percent: int = Field(ge=0, le=99)
    cancel_requested: bool
    checkpoint: Dict[str, Any] | None

    @model_validator(mode="after")
    def _validate_lease(self) -> "ReportJobLease":
        if self.attempt_count > self.max_attempts:
            raise ValueError("Report job lease attempt_count cannot exceed max_attempts.")
        if self.phase not in {
            ReportJobPhase.PLANNING,
            ReportJobPhase.GENERATING_SECTIONS,
            ReportJobPhase.ASSEMBLING,
        }:
            raise ValueError("A report job lease must use an executable phase.")
        return self


def validate_report_job_transition(
    previous: ReportJobSnapshot,
    current: ReportJobSnapshot,
) -> None:
    """Reject illegal or lossy durable state transitions."""

    if previous.job_id != current.job_id or previous.request_id != current.request_id:
        raise ValueError("Report job identity cannot change.")
    if previous.state in _TERMINAL_STATES:
        raise ValueError("A terminal report job cannot transition.")

    allowed_states = {
        ReportJobState.QUEUED: {
            ReportJobState.RUNNING,
            ReportJobState.CANCELLED,
        },
        ReportJobState.RUNNING: {
            ReportJobState.RUNNING,
            ReportJobState.QUEUED,
            ReportJobState.COMPLETED,
            ReportJobState.FAILED,
            ReportJobState.CANCELLED,
        },
    }[previous.state]
    if current.state not in allowed_states:
        raise ValueError(
            f"Report job transition {previous.state.value} -> {current.state.value} is not allowed."
        )

    if current.progress_percent < previous.progress_percent:
        raise ValueError("Report job progress cannot decrease.")

    if previous.phase in _ACTIVE_PHASE_ORDER and current.phase in _ACTIVE_PHASE_ORDER:
        if _ACTIVE_PHASE_ORDER[current.phase] < _ACTIVE_PHASE_ORDER[previous.phase]:
            raise ValueError("Report job phase cannot move backwards.")

    if previous.state is ReportJobState.QUEUED and current.state is ReportJobState.RUNNING:
        if current.attempt_count != previous.attempt_count + 1:
            raise ValueError("Leasing a report job must increment attempt_count exactly once.")
    elif current.attempt_count != previous.attempt_count:
        raise ValueError("Only a queued-to-running lease may change attempt_count.")
