"""Versioned durable-report job and state-machine contract tests."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from contracts.report_jobs import (
    REPORT_JOB_CONTRACT_VERSION,
    ReportJobCreateRequest,
    ReportJobPhase,
    ReportJobSnapshot,
    ReportJobState,
    validate_report_job_transition,
)


def _snapshot_payload(**overrides) -> dict:
    payload = {
        "contract_version": REPORT_JOB_CONTRACT_VERSION,
        "job_id": str(uuid4()),
        "request_id": "report:req-1",
        "state": "queued",
        "phase": "queued",
        "progress_percent": 0,
        "attempt_count": 0,
        "max_attempts": 3,
        "cancel_requested": False,
        "created_at": datetime(2026, 7, 26, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 7, 26, tzinfo=timezone.utc),
        "completed_at": None,
        "error_code": None,
        "result": None,
    }
    payload.update(overrides)
    return payload


def test_report_job_create_request_is_closed_and_bounded():
    request = ReportJobCreateRequest.model_validate(
        {
            "contract_version": REPORT_JOB_CONTRACT_VERSION,
            "request_id": "report:req-1",
            "query": "Prepare an evidence-grounded electricity market report.",
        }
    )

    assert request.contract_version == REPORT_JOB_CONTRACT_VERSION
    assert request.request_id == "report:req-1"
    assert request.query.startswith("Prepare")
    assert ReportJobCreateRequest.model_json_schema()["additionalProperties"] is False

    with pytest.raises(ValidationError):
        ReportJobCreateRequest.model_validate(
            {
                "contract_version": REPORT_JOB_CONTRACT_VERSION,
                "request_id": "report:req-1",
                "query": "valid",
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError):
        ReportJobCreateRequest.model_validate(
            {
                "contract_version": REPORT_JOB_CONTRACT_VERSION,
                "request_id": "../unsafe",
                "query": "valid",
            }
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "state": "completed",
                "phase": "assembling",
                "progress_percent": 100,
                "completed_at": datetime(2026, 7, 26, tzinfo=timezone.utc),
                "result": {"contract_version": "report-result-v1"},
            },
            "completed phase",
        ),
        (
            {
                "state": "completed",
                "phase": "completed",
                "progress_percent": 99,
                "completed_at": datetime(2026, 7, 26, tzinfo=timezone.utc),
                "result": {"contract_version": "report-result-v1"},
            },
            "100 percent",
        ),
        (
            {
                "state": "failed",
                "phase": "failed",
                "error_code": None,
                "completed_at": datetime(2026, 7, 26, tzinfo=timezone.utc),
            },
            "error_code",
        ),
        (
            {
                "state": "running",
                "phase": "planning",
                "attempt_count": 0,
            },
            "positive attempt_count",
        ),
    ],
)
def test_report_job_snapshot_rejects_inconsistent_state(overrides, message):
    with pytest.raises(ValidationError, match=message):
        ReportJobSnapshot.model_validate(_snapshot_payload(**overrides))


def test_report_job_snapshot_accepts_resume_safe_retry_progress():
    snapshot = ReportJobSnapshot.model_validate(
        _snapshot_payload(
            state="queued",
            phase="generating_sections",
            progress_percent=55,
            attempt_count=1,
        )
    )

    assert snapshot.state is ReportJobState.QUEUED
    assert snapshot.phase is ReportJobPhase.GENERATING_SECTIONS
    assert snapshot.progress_percent == 55


def test_report_job_transition_allows_lease_progress_retry_and_completion():
    queued = ReportJobSnapshot.model_validate(_snapshot_payload())
    planning = ReportJobSnapshot.model_validate(
        _snapshot_payload(
            job_id=queued.job_id,
            state="running",
            phase="planning",
            progress_percent=5,
            attempt_count=1,
        )
    )
    section_progress = ReportJobSnapshot.model_validate(
        _snapshot_payload(
            job_id=queued.job_id,
            state="running",
            phase="generating_sections",
            progress_percent=60,
            attempt_count=1,
        )
    )
    retry = ReportJobSnapshot.model_validate(
        _snapshot_payload(
            job_id=queued.job_id,
            state="queued",
            phase="generating_sections",
            progress_percent=60,
            attempt_count=1,
        )
    )
    resumed = ReportJobSnapshot.model_validate(
        _snapshot_payload(
            job_id=queued.job_id,
            state="running",
            phase="generating_sections",
            progress_percent=60,
            attempt_count=2,
        )
    )
    completed = ReportJobSnapshot.model_validate(
        _snapshot_payload(
            job_id=queued.job_id,
            state="completed",
            phase="completed",
            progress_percent=100,
            attempt_count=2,
            completed_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
            result={"contract_version": "report-result-v1", "content": "Done."},
        )
    )

    for previous, current in (
        (queued, planning),
        (planning, section_progress),
        (section_progress, retry),
        (retry, resumed),
        (resumed, completed),
    ):
        validate_report_job_transition(previous, current)


@pytest.mark.parametrize(
    ("previous_overrides", "current_overrides", "message"),
    [
        (
            {},
            {"state": "completed", "phase": "completed", "progress_percent": 100, "attempt_count": 1,
             "completed_at": datetime(2026, 7, 26, tzinfo=timezone.utc),
             "result": {"contract_version": "report-result-v1"}},
            "not allowed",
        ),
        (
            {"state": "running", "phase": "generating_sections", "progress_percent": 60, "attempt_count": 1},
            {"state": "running", "phase": "planning", "progress_percent": 65, "attempt_count": 1},
            "phase cannot move backwards",
        ),
        (
            {"state": "running", "phase": "generating_sections", "progress_percent": 60, "attempt_count": 1},
            {"state": "running", "phase": "generating_sections", "progress_percent": 50, "attempt_count": 1},
            "progress cannot decrease",
        ),
        (
            {"state": "completed", "phase": "completed", "progress_percent": 100, "attempt_count": 1,
             "completed_at": datetime(2026, 7, 26, tzinfo=timezone.utc),
             "result": {"contract_version": "report-result-v1"}},
            {"state": "running", "phase": "assembling", "progress_percent": 99, "attempt_count": 2},
            "terminal",
        ),
    ],
)
def test_report_job_transition_rejects_invalid_lifecycle_changes(
    previous_overrides,
    current_overrides,
    message,
):
    job_id = uuid4()
    previous = ReportJobSnapshot.model_validate(
        _snapshot_payload(job_id=job_id, **previous_overrides)
    )
    current = ReportJobSnapshot.model_validate(
        _snapshot_payload(job_id=job_id, **current_overrides)
    )

    with pytest.raises(ValueError, match=message):
        validate_report_job_transition(previous, current)
