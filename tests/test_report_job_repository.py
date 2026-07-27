"""PostgreSQL report-job repository adapter tests."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from contracts.report_jobs import ReportJobPhase
from core.report_job_repository import (
    PostgresReportJobRepository,
    ReportJobRepositoryError,
)


class _Result:
    def __init__(self, payload):
        self._payload = payload

    def scalar_one(self):
        return self._payload


class _Connection:
    def __init__(self, payloads):
        self._payloads = payloads
        self.calls = []

    def execute(self, statement, parameters):
        self.calls.append((str(statement), parameters))
        return _Result(self._payloads.pop(0))


def _lease_payload():
    return {
        "ok": True,
        "disposition": "leased",
        "lease": {
            "contract_version": "report-job-v1",
            "job_id": str(uuid4()),
            "request_id": "report:req-1",
            "actor_user_id": str(uuid4()),
            "query": "Prepare the report.",
            "attempt_count": 1,
            "max_attempts": 3,
            "lease_owner": "worker-1",
            "lease_expires_at": (
                datetime.now(timezone.utc) + timedelta(minutes=2)
            ).isoformat(),
            "phase": "planning",
            "progress_percent": 5,
            "cancel_requested": False,
            "checkpoint": None,
        },
    }


def _repository(payloads):
    connection = _Connection(list(payloads))
    calls = []

    @contextmanager
    def database_connection(engine, **kwargs):
        calls.append((engine, kwargs))
        yield connection

    repository = PostgresReportJobRepository(
        engine="engine",
        database_connection=database_connection,
    )
    return repository, connection, calls


def test_repository_leases_through_guarded_transaction_and_validates_contract():
    repository, connection, guard_calls = _repository(
        [json.dumps(_lease_payload())]
    )

    lease = repository.lease_next(worker_id="worker-1", lease_seconds=120)

    assert lease is not None
    assert lease.lease_owner == "worker-1"
    assert guard_calls == [
        (
            "engine",
            {
                "operation": "report_job_lease",
                "begin": True,
                "priority": "application",
            },
        )
    ]
    assert "lease_report_job_v1" in connection.calls[0][0]
    assert connection.calls[0][1] == {
        "worker_id": "worker-1",
        "lease_seconds": 120,
    }


def test_repository_returns_none_for_an_empty_queue():
    repository, _, _ = _repository(
        [{"ok": True, "disposition": "none"}]
    )

    assert repository.lease_next(worker_id="worker-1", lease_seconds=120) is None


def test_repository_rejects_malformed_or_failed_rpc_payloads():
    repository, _, _ = _repository([{"ok": False, "code": "INVALID_WORKER_ID"}])
    with pytest.raises(ReportJobRepositoryError, match="INVALID_WORKER_ID"):
        repository.lease_next(worker_id="worker-1", lease_seconds=120)

    repository, _, _ = _repository(["not-json"])
    with pytest.raises(ReportJobRepositoryError, match="invalid payload"):
        repository.lease_next(worker_id="worker-1", lease_seconds=120)

    repository, _, _ = _repository(
        [{"ok": False, "code": "provider secret must not escape"}]
    )
    with pytest.raises(ReportJobRepositoryError, match="REPORT_JOB_RPC_FAILED") as exc_info:
        repository.lease_next(worker_id="worker-1", lease_seconds=120)
    assert "provider secret" not in str(exc_info.value)


def test_repository_maps_every_worker_mutation_to_versioned_rpc():
    job_id = uuid4()
    repository, connection, guard_calls = _repository(
        [
            {"ok": True, "updated": True},
            {"ok": True},
            {"ok": True},
            {"ok": True},
            {"ok": True, "cancel_requested": True},
        ]
    )

    assert repository.heartbeat(
        job_id=job_id,
        worker_id="worker-1",
        phase=ReportJobPhase.GENERATING_SECTIONS,
        progress_percent=50,
        checkpoint={"completed_section_ids": ["summary"]},
        lease_seconds=120,
    )
    assert repository.complete(
        job_id=job_id,
        worker_id="worker-1",
        result={"contract_version": "report-result-v1", "content": "Done"},
    )
    assert repository.fail(
        job_id=job_id,
        worker_id="worker-1",
        error_code="REPORT_PROVIDER_UNAVAILABLE",
        retryable=True,
        retry_delay_seconds=30,
    )
    assert repository.acknowledge_cancellation(
        job_id=job_id,
        worker_id="worker-1",
    )
    assert repository.cancellation_requested(
        job_id=job_id,
        worker_id="worker-1",
    )

    statements = [statement for statement, _ in connection.calls]
    assert any("heartbeat_report_job_v1" in statement for statement in statements)
    assert any("complete_report_job_v1" in statement for statement in statements)
    assert any("fail_report_job_v1" in statement for statement in statements)
    assert any(
        "acknowledge_report_job_cancellation_v1" in statement
        for statement in statements
    )
    assert any(
        "report_job_cancellation_requested_v1" in statement
        for statement in statements
    )
    assert all(
        call[1]["begin"] is True and call[1]["priority"] == "application"
        for call in guard_calls
    )


def test_repository_heartbeat_surfaces_cooperative_cancellation():
    repository, _, _ = _repository(
        [{"ok": True, "updated": False, "cancel_requested": True}]
    )

    assert repository.heartbeat(
        job_id=uuid4(),
        worker_id="worker-1",
        phase=ReportJobPhase.ASSEMBLING,
        progress_percent=90,
        checkpoint=None,
        lease_seconds=120,
    ) is False


def test_repository_heartbeat_requires_explicit_update_acknowledgement():
    repository, _, _ = _repository([{"ok": True}])

    assert repository.heartbeat(
        job_id=uuid4(),
        worker_id="worker-1",
        phase=ReportJobPhase.GENERATING_SECTIONS,
        progress_percent=50,
        checkpoint=None,
        lease_seconds=120,
    ) is False
