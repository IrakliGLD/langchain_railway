"""Checkpointed orchestration for one durable analytical-report job."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from agent.report_assembly import ReportAssemblyError, assemble_report
from agent.report_charts import (
    build_report_charts,
    demote_unbuildable_required_charts,
)
from agent.report_evaluation import evaluate_report_plan
from agent.report_evidence import (
    build_report_evidence_manifest,
)
from agent.report_intent import build_report_planning_context
from agent.report_planner import (
    ReportPlanEvidenceError,
    ReportPlanSemanticError,
    plan_report,
    validate_report_plan_semantics,
)
from agent.report_sections import (
    ReportSectionGenerationError,
    generate_report_sections,
)
from contracts.report import ReportPlan
from contracts.report_evidence import ReportEvidenceKind, ReportEvidenceManifest
from contracts.report_generation import (
    ReportCheckpointTooLargeError,
    ReportGenerationCheckpoint,
)
from contracts.report_jobs import ReportJobLease, ReportJobPhase
from contracts.report_sections import ReportSectionDraft
from core.report_job_worker import (
    ReportJobExecutionControl,
    ReportJobFailure,
)
from utils.provider_attempts import ProviderExecutionError
from utils.request_deadline import (
    RequestDeadlineExceeded,
    bind_request_execution_scope,
    cap_request_deadline,
    current_request_execution_scope,
)

_LOGGER = logging.getLogger("Enai.ReportProcessor")

QueryPipeline = Callable[..., Any]
EvidenceBuilder = Callable[[Any], Any]
Planner = Callable[..., Any]
Evaluator = Callable[..., Any]
ChartBuilder = Callable[[ReportPlan, ReportEvidenceManifest], Any]
SectionGenerator = Callable[..., list[ReportSectionDraft]]
Assembler = Callable[..., Any]

_REPORT_FAILURE_RETRYABILITY = {
    "REPORT_ASSEMBLY_INVALID": False,
    "REPORT_CANCELLED": False,
    "REPORT_CHECKPOINT_INVALID": False,
    "REPORT_CHECKPOINT_TOO_LARGE": False,
    "REPORT_DEADLINE_EXCEEDED": True,
    "REPORT_EVIDENCE_INVALID": False,
    "REPORT_EVIDENCE_UNAVAILABLE": False,
    "REPORT_LEASE_LOST": True,
    "REPORT_PLAN_INVALID": False,
    "REPORT_PLAN_NOT_READY": False,
    "REPORT_PLAN_PROVIDER_FAILED": True,
    "REPORT_SECTION_INVALID": False,
    "REPORT_SECTION_PROVIDER_FAILED": True,
}
_SECTION_PROVIDER_FAILURE_CODES = {
    "SECTION_REPAIR_PROVIDER_FAILED",
    "SECTION_WRITE_PROVIDER_FAILED",
}


def _diagnostic_identifier(value: str | None) -> str:
    candidate = str(value or "")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", candidate):
        return candidate
    return "unknown"


def _diagnostic_error_codes(error_codes: list[str]) -> str:
    safe_codes = [
        code
        for code in error_codes[:16]
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code)
    ]
    return ",".join(safe_codes) or "unknown"


def _report_failure(error_code: str) -> ReportJobFailure:
    try:
        retryable = _REPORT_FAILURE_RETRYABILITY[error_code]
    except KeyError as exc:
        raise ValueError("Unknown report failure policy code.") from exc
    return ReportJobFailure(error_code, retryable=retryable)


class ReportJobProcessor:
    """Run report phases while persisting enough state for retry/resume."""

    def __init__(
        self,
        *,
        query_pipeline: QueryPipeline | None = None,
        evidence_builder: EvidenceBuilder = build_report_evidence_manifest,
        planner: Planner = plan_report,
        evaluator: Evaluator = evaluate_report_plan,
        chart_builder: ChartBuilder = build_report_charts,
        section_generator: SectionGenerator = generate_report_sections,
        assembler: Assembler = assemble_report,
        max_section_workers: int = 4,
        job_timeout_seconds: int = 600,
    ) -> None:
        if not 1 <= max_section_workers <= 8:
            raise ValueError(
                "max_section_workers must be between 1 and 8."
            )
        if not 1 <= job_timeout_seconds <= 3600:
            raise ValueError(
                "job_timeout_seconds must be between 1 and 3600."
            )
        self._query_pipeline = query_pipeline
        self._evidence_builder = evidence_builder
        self._planner = planner
        self._evaluator = evaluator
        self._chart_builder = chart_builder
        self._section_generator = section_generator
        self._assembler = assembler
        self._max_section_workers = max_section_workers
        self._job_timeout_seconds = job_timeout_seconds

    @staticmethod
    def _validate_query_binding(
        query: str,
        manifest: ReportEvidenceManifest,
    ) -> None:
        expected_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if manifest.query_digest != expected_digest:
            raise _report_failure("REPORT_CHECKPOINT_INVALID")

    @staticmethod
    def _checkpoint_payload(
        manifest: ReportEvidenceManifest,
        plan: ReportPlan,
        completed_by_id: dict[str, ReportSectionDraft],
    ) -> dict[str, Any]:
        checkpoint = ReportGenerationCheckpoint(
            contract_version="report-generation-checkpoint-v1",
            manifest=manifest,
            plan=plan,
            completed_sections=[
                completed_by_id[section.section_id]
                for section in plan.sections
                if section.section_id in completed_by_id
            ],
        )
        return checkpoint.model_dump(mode="json")

    @classmethod
    def _safe_checkpoint_payload(
        cls,
        manifest: ReportEvidenceManifest,
        plan: ReportPlan,
        completed_by_id: dict[str, ReportSectionDraft],
    ) -> dict[str, Any]:
        """Build a checkpoint, separating "too big" from "structurally wrong"."""

        try:
            return cls._checkpoint_payload(manifest, plan, completed_by_id)
        except ReportCheckpointTooLargeError as exc:
            raise _report_failure("REPORT_CHECKPOINT_TOO_LARGE") from exc
        except (ValidationError, ValueError) as exc:
            raise _report_failure("REPORT_CHECKPOINT_INVALID") from exc

    @staticmethod
    def _heartbeat(
        control: ReportJobExecutionControl,
        *,
        phase: ReportJobPhase,
        progress_percent: int,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        if control.heartbeat(
            phase=phase,
            progress_percent=progress_percent,
            checkpoint=checkpoint,
        ):
            return
        if control.cancellation_requested():
            raise _report_failure("REPORT_CANCELLED")
        raise _report_failure("REPORT_LEASE_LOST")

    @staticmethod
    def _raise_if_cancelled(
        control: ReportJobExecutionControl,
    ) -> None:
        if control.cancellation_requested():
            raise _report_failure("REPORT_CANCELLED")

    def _run_query_pipeline(self, lease: ReportJobLease) -> Any:
        pipeline = self._query_pipeline
        if pipeline is None:
            from agent.pipeline import process_query

            pipeline = process_query
        execution_scope = current_request_execution_scope()
        request_deadline = (
            execution_scope.deadline
            if execution_scope is not None
            else None
        )
        request_id = (
            execution_scope.request_id
            if execution_scope is not None
            else lease.request_id
        )
        return pipeline(
            lease.query,
            trace_id=str(lease.job_id),
            actor_id=str(lease.actor_user_id),
            request_id=request_id,
            request_deadline=request_deadline,
            answer_mode="report",
        )

    def __call__(
        self,
        lease: ReportJobLease,
        control: ReportJobExecutionControl,
    ) -> dict[str, Any]:
        deadline = cap_request_deadline(
            maximum_seconds=self._job_timeout_seconds,
            source="report_job",
        )
        execution_request_id = (
            f"{lease.request_id}:attempt:{lease.attempt_count}"
        )
        with bind_request_execution_scope(
            deadline=deadline,
            request_id=execution_request_id,
            actor_id=str(lease.actor_user_id),
        ):
            try:
                return self._run_bound_attempt(lease, control)
            except RequestDeadlineExceeded as exc:
                _LOGGER.warning(
                    "Report deadline exceeded: job_id=%s job_attempt=%s "
                    "stage=%s",
                    lease.job_id,
                    lease.attempt_count,
                    _diagnostic_identifier(exc.stage),
                )
                raise _report_failure("REPORT_DEADLINE_EXCEEDED") from exc

    def _run_bound_attempt(
        self,
        lease: ReportJobLease,
        control: ReportJobExecutionControl,
    ) -> dict[str, Any]:
        progress = lease.progress_percent
        checkpoint: ReportGenerationCheckpoint | None = None
        if lease.checkpoint is not None:
            try:
                checkpoint = ReportGenerationCheckpoint.model_validate(
                    lease.checkpoint
                )
            except (ValidationError, ValueError) as exc:
                raise _report_failure("REPORT_CHECKPOINT_INVALID") from exc
            self._validate_query_binding(lease.query, checkpoint.manifest)

        if checkpoint is None:
            if lease.phase is not ReportJobPhase.PLANNING:
                raise _report_failure("REPORT_CHECKPOINT_INVALID")
            progress = max(progress, 10)
            self._heartbeat(
                control,
                phase=ReportJobPhase.PLANNING,
                progress_percent=progress,
                checkpoint=None,
            )
            self._raise_if_cancelled(control)
            try:
                context = self._run_query_pipeline(lease)
                planning_context = build_report_planning_context(context)
                manifest_raw = self._evidence_builder(context)
                manifest = (
                    manifest_raw
                    if isinstance(manifest_raw, ReportEvidenceManifest)
                    else ReportEvidenceManifest.model_validate(manifest_raw)
                )
                expected_digest = hashlib.sha256(
                    lease.query.encode("utf-8")
                ).hexdigest()
                if manifest.query_digest != expected_digest:
                    raise ValueError(
                        "Fresh report evidence does not match the job query."
                    )
                if (
                    planning_context.requires_table
                    and not any(
                        item.kind is ReportEvidenceKind.TABLE
                        for item in manifest.items
                    )
                ):
                    raise _report_failure("REPORT_EVIDENCE_UNAVAILABLE")
            except ReportJobFailure:
                raise
            except (ValidationError, ValueError) as exc:
                raise _report_failure("REPORT_EVIDENCE_INVALID") from exc

            self._raise_if_cancelled(control)
            try:
                raw_plan = self._planner(
                    lease.query,
                    manifest,
                    planning_context=planning_context,
                )
                plan = (
                    raw_plan
                    if isinstance(raw_plan, ReportPlan)
                    else ReportPlan.model_validate(raw_plan)
                )
                validate_report_plan_semantics(plan, planning_context)
                chart_decisions = self._chart_builder(plan, manifest)
                plan, chart_decisions = demote_unbuildable_required_charts(
                    plan,
                    chart_decisions,
                )
                evaluation = self._evaluator(
                    plan,
                    manifest,
                    chart_decisions=chart_decisions,
                )
            except ProviderExecutionError as exc:
                _LOGGER.warning(
                    "Report provider failure: job_id=%s job_attempt=%s "
                    "provider=%s provider_stage=%s provider_disposition=%s",
                    lease.job_id,
                    lease.attempt_count,
                    _diagnostic_identifier(exc.provider),
                    _diagnostic_identifier(exc.stage),
                    exc.disposition.value,
                )
                raise _report_failure("REPORT_PLAN_PROVIDER_FAILED") from exc
            except (ReportPlanEvidenceError, ReportPlanSemanticError) as exc:
                raise _report_failure("REPORT_PLAN_INVALID") from exc
            except (ValidationError, ValueError) as exc:
                raise _report_failure("REPORT_PLAN_INVALID") from exc
            if not evaluation.ready_for_generation:
                raise _report_failure("REPORT_PLAN_NOT_READY")
            completed_by_id: dict[str, ReportSectionDraft] = {}
            progress = max(progress, 25)
            checkpoint_payload = self._safe_checkpoint_payload(
                manifest,
                plan,
                completed_by_id,
            )
            self._heartbeat(
                control,
                phase=ReportJobPhase.GENERATING_SECTIONS,
                progress_percent=progress,
                checkpoint=checkpoint_payload,
            )
        else:
            manifest = checkpoint.manifest
            plan = checkpoint.plan
            completed_by_id = {
                draft.section_id: draft
                for draft in checkpoint.completed_sections
            }
            try:
                chart_decisions = self._chart_builder(plan, manifest)
                # A checkpoint written before chart demotion shipped still marks
                # an unbuildable chart required; resuming must not kill it.
                plan, chart_decisions = demote_unbuildable_required_charts(
                    plan,
                    chart_decisions,
                )
                evaluation = self._evaluator(
                    plan,
                    manifest,
                    chart_decisions=chart_decisions,
                )
            except (ReportPlanEvidenceError, ValidationError, ValueError) as exc:
                raise _report_failure("REPORT_CHECKPOINT_INVALID") from exc
            if not evaluation.ready_for_generation:
                raise _report_failure("REPORT_CHECKPOINT_INVALID")
            if (
                lease.phase is ReportJobPhase.ASSEMBLING
                and len(completed_by_id) != len(plan.sections)
            ):
                raise _report_failure("REPORT_CHECKPOINT_INVALID")

        self._raise_if_cancelled(control)

        total_sections = len(plan.sections)
        if len(completed_by_id) < total_sections:
            progress = max(
                progress,
                25 + math.floor(60 * len(completed_by_id) / total_sections),
            )
            if checkpoint is not None:
                checkpoint_payload = self._safe_checkpoint_payload(
                    manifest,
                    plan,
                    completed_by_id,
                )
                self._heartbeat(
                    control,
                    phase=ReportJobPhase.GENERATING_SECTIONS,
                    progress_percent=progress,
                    checkpoint=checkpoint_payload,
                )

            def persist_section(
                completed: int,
                total: int,
                draft: ReportSectionDraft,
            ) -> None:
                nonlocal progress
                completed_by_id[draft.section_id] = draft
                progress = max(
                    progress,
                    min(85, 25 + math.floor(60 * completed / total)),
                )
                self._heartbeat(
                    control,
                    phase=ReportJobPhase.GENERATING_SECTIONS,
                    progress_percent=progress,
                    checkpoint=self._safe_checkpoint_payload(
                        manifest,
                        plan,
                        completed_by_id,
                    ),
                )

            try:
                drafts = self._section_generator(
                    lease.query,
                    plan,
                    manifest,
                    existing_drafts=completed_by_id,
                    progress_callback=persist_section,
                    max_workers=self._max_section_workers,
                )
            except ReportSectionGenerationError as exc:
                _LOGGER.warning(
                    "Report section phase failed: job_id=%s job_attempt=%s "
                    "section_id=%s error_codes=%s provider=%s "
                    "provider_stage=%s provider_disposition=%s",
                    lease.job_id,
                    lease.attempt_count,
                    exc.section_id,
                    _diagnostic_error_codes(exc.error_codes),
                    _diagnostic_identifier(exc.provider),
                    _diagnostic_identifier(exc.provider_stage),
                    _diagnostic_identifier(exc.provider_disposition),
                )
                error_code = (
                    "REPORT_SECTION_PROVIDER_FAILED"
                    if _SECTION_PROVIDER_FAILURE_CODES.intersection(
                        exc.error_codes
                    )
                    else "REPORT_SECTION_INVALID"
                )
                raise _report_failure(error_code) from exc
            except (ValidationError, ValueError) as exc:
                raise _report_failure("REPORT_SECTION_INVALID") from exc
        else:
            drafts = [
                completed_by_id[section.section_id]
                for section in plan.sections
            ]

        self._raise_if_cancelled(control)
        progress = max(progress, 90)
        self._heartbeat(
            control,
            phase=ReportJobPhase.ASSEMBLING,
            progress_percent=progress,
            checkpoint=self._safe_checkpoint_payload(
                manifest,
                plan,
                {draft.section_id: draft for draft in drafts},
            ),
        )
        try:
            result = self._assembler(
                plan,
                manifest,
                drafts,
                chart_decisions,
            )
        except (ReportAssemblyError, ValidationError, ValueError) as exc:
            raise _report_failure("REPORT_ASSEMBLY_INVALID") from exc
        return result.model_dump(mode="json")
