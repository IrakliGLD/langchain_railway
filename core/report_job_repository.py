"""Guarded PostgreSQL adapter for versioned report-job RPCs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import text

from contracts.report_jobs import ReportJobLease, ReportJobPhase


class ReportJobRepositoryError(RuntimeError):
    """A bounded persistence-contract failure without database payload leakage."""


class PostgresReportJobRepository:
    def __init__(
        self,
        *,
        engine: Any,
        database_connection: Callable[..., Any],
    ) -> None:
        self._engine = engine
        self._database_connection = database_connection

    @staticmethod
    def _payload(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ReportJobRepositoryError("Report job RPC returned an invalid payload.") from exc
        if not isinstance(value, dict):
            raise ReportJobRepositoryError("Report job RPC returned an invalid payload.")
        return value

    def _call(
        self,
        *,
        operation: str,
        statement: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        with self._database_connection(
            self._engine,
            operation=operation,
            begin=True,
            priority="application",
        ) as connection:
            raw_payload = connection.execute(text(statement), parameters).scalar_one()
        payload = self._payload(raw_payload)
        if payload.get("ok") is not True:
            code = payload.get("code")
            safe_code = (
                code
                if isinstance(code, str)
                and re.fullmatch(r"^[A-Z][A-Z0-9_]{0,63}$", code)
                else "REPORT_JOB_RPC_FAILED"
            )
            raise ReportJobRepositoryError(safe_code)
        return payload

    def lease_next(self, *, worker_id: str, lease_seconds: int) -> ReportJobLease | None:
        payload = self._call(
            operation="report_job_lease",
            statement=(
                "select public.lease_report_job_v1("
                ":worker_id, :lease_seconds) as payload"
            ),
            parameters={
                "worker_id": worker_id,
                "lease_seconds": lease_seconds,
            },
        )
        if payload.get("disposition") == "none":
            return None
        if payload.get("disposition") != "leased":
            raise ReportJobRepositoryError("Report job lease returned an invalid disposition.")
        try:
            return ReportJobLease.model_validate(payload.get("lease"))
        except ValidationError as exc:
            raise ReportJobRepositoryError("Report job lease returned an invalid payload.") from exc

    def heartbeat(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        phase: ReportJobPhase,
        progress_percent: int,
        checkpoint: dict[str, Any] | None,
        lease_seconds: int,
    ) -> bool:
        payload = self._call(
            operation="report_job_heartbeat",
            statement=(
                "select public.heartbeat_report_job_v1("
                ":job_id, :worker_id, :phase, :progress_percent, "
                "cast(:checkpoint as jsonb), :lease_seconds) as payload"
            ),
            parameters={
                "job_id": job_id,
                "worker_id": worker_id,
                "phase": phase.value,
                "progress_percent": progress_percent,
                "checkpoint": (
                    json.dumps(checkpoint, separators=(",", ":"))
                    if checkpoint is not None
                    else None
                ),
                "lease_seconds": lease_seconds,
            },
        )
        return payload.get("updated") is not False

    def complete(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        result: dict[str, Any],
    ) -> bool:
        self._call(
            operation="report_job_complete",
            statement=(
                "select public.complete_report_job_v1("
                ":job_id, :worker_id, cast(:result as jsonb)) as payload"
            ),
            parameters={
                "job_id": job_id,
                "worker_id": worker_id,
                "result": json.dumps(result, separators=(",", ":")),
            },
        )
        return True

    def fail(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        error_code: str,
        retryable: bool,
        retry_delay_seconds: int,
    ) -> bool:
        self._call(
            operation="report_job_fail",
            statement=(
                "select public.fail_report_job_v1("
                ":job_id, :worker_id, :error_code, :retryable, "
                ":retry_delay_seconds) as payload"
            ),
            parameters={
                "job_id": job_id,
                "worker_id": worker_id,
                "error_code": error_code,
                "retryable": retryable,
                "retry_delay_seconds": retry_delay_seconds,
            },
        )
        return True

    def acknowledge_cancellation(self, *, job_id: UUID, worker_id: str) -> bool:
        self._call(
            operation="report_job_cancel_ack",
            statement=(
                "select public.acknowledge_report_job_cancellation_v1("
                ":job_id, :worker_id) as payload"
            ),
            parameters={"job_id": job_id, "worker_id": worker_id},
        )
        return True

    def cancellation_requested(self, *, job_id: UUID, worker_id: str) -> bool:
        payload = self._call(
            operation="report_job_cancel_check",
            statement=(
                "select public.report_job_cancellation_requested_v1("
                ":job_id, :worker_id) as payload"
            ),
            parameters={"job_id": job_id, "worker_id": worker_id},
        )
        return payload.get("cancel_requested") is True
