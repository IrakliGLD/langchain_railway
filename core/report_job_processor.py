"""Checkpointed orchestration for one durable analytical-report job."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from agent.report_assembly import ReportAssemblyError, assemble_report
from agent.report_charts import build_report_charts
from agent.report_evaluation import evaluate_report_plan
from agent.report_evidence import build_report_evidence_manifest
from agent.report_planner import ReportPlanEvidenceError, plan_report
from agent.report_sections import (
    ReportSectionGenerationError,
    generate_report_sections,
)
from contracts.report import ReportPlan
from contracts.report_evidence import ReportEvidenceManifest
from contracts.report_generation import ReportGenerationCheckpoint
from contracts.report_jobs import ReportJobLease, ReportJobPhase
from contracts.report_sections import ReportSectionDraft
from core.report_job_worker import (
    ReportJobExecutionControl,
    ReportJobFailure,
)

QueryPipeline = Callable[..., Any]
EvidenceBuilder = Callable[[Any], Any]
Planner = Callable[[str, ReportEvidenceManifest], Any]
Evaluator = Callable[[ReportPlan, ReportEvidenceManifest], Any]
ChartBuilder = Callable[[ReportPlan, ReportEvidenceManifest], Any]
SectionGenerator = Callable[..., list[ReportSectionDraft]]
Assembler = Callable[..., Any]


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
    ) -> None:
        if not 1 <= max_section_workers <= 8:
            raise ValueError(
                "max_section_workers must be between 1 and 8."
            )
        self._query_pipeline = query_pipeline
        self._evidence_builder = evidence_builder
        self._planner = planner
        self._evaluator = evaluator
        self._chart_builder = chart_builder
        self._section_generator = section_generator
        self._assembler = assembler
        self._max_section_workers = max_section_workers

    @staticmethod
    def _validate_query_binding(
        query: str,
        manifest: ReportEvidenceManifest,
    ) -> None:
        expected_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if manifest.query_digest != expected_digest:
            raise ReportJobFailure(
                "REPORT_CHECKPOINT_INVALID",
                retryable=False,
            )

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
            raise ReportJobFailure("REPORT_CANCELLED", retryable=False)
        raise ReportJobFailure("REPORT_LEASE_LOST", retryable=True)

    @staticmethod
    def _raise_if_cancelled(
        control: ReportJobExecutionControl,
    ) -> None:
        if control.cancellation_requested():
            raise ReportJobFailure("REPORT_CANCELLED", retryable=False)

    def _run_query_pipeline(self, lease: ReportJobLease) -> Any:
        pipeline = self._query_pipeline
        if pipeline is None:
            from agent.pipeline import process_query

            pipeline = process_query
        return pipeline(
            lease.query,
            trace_id=str(lease.job_id),
            actor_id=str(lease.actor_user_id),
            request_id=lease.request_id,
            answer_mode="report",
        )

    def __call__(
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
                raise ReportJobFailure(
                    "REPORT_CHECKPOINT_INVALID",
                    retryable=False,
                ) from exc
            self._validate_query_binding(lease.query, checkpoint.manifest)

        if checkpoint is None:
            if lease.phase is not ReportJobPhase.PLANNING:
                raise ReportJobFailure(
                    "REPORT_CHECKPOINT_INVALID",
                    retryable=False,
                )
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
            except ReportJobFailure:
                raise
            except (ValidationError, ValueError) as exc:
                raise ReportJobFailure(
                    "REPORT_EVIDENCE_INVALID",
                    retryable=False,
                ) from exc

            self._raise_if_cancelled(control)
            try:
                raw_plan = self._planner(lease.query, manifest)
                plan = (
                    raw_plan
                    if isinstance(raw_plan, ReportPlan)
                    else ReportPlan.model_validate(raw_plan)
                )
                evaluation = self._evaluator(plan, manifest)
            except ReportPlanEvidenceError as exc:
                raise ReportJobFailure(
                    "REPORT_PLAN_INVALID",
                    retryable=False,
                ) from exc
            except (ValidationError, ValueError) as exc:
                raise ReportJobFailure(
                    "REPORT_PLAN_INVALID",
                    retryable=True,
                ) from exc
            if not evaluation.ready_for_generation:
                raise ReportJobFailure(
                    "REPORT_PLAN_NOT_READY",
                    retryable=False,
                )
            completed_by_id: dict[str, ReportSectionDraft] = {}
            progress = max(progress, 25)
            checkpoint_payload = self._checkpoint_payload(
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
                evaluation = self._evaluator(plan, manifest)
            except (ReportPlanEvidenceError, ValidationError, ValueError) as exc:
                raise ReportJobFailure(
                    "REPORT_CHECKPOINT_INVALID",
                    retryable=False,
                ) from exc
            if not evaluation.ready_for_generation:
                raise ReportJobFailure(
                    "REPORT_CHECKPOINT_INVALID",
                    retryable=False,
                )
            if (
                lease.phase is ReportJobPhase.ASSEMBLING
                and len(completed_by_id) != len(plan.sections)
            ):
                raise ReportJobFailure(
                    "REPORT_CHECKPOINT_INVALID",
                    retryable=False,
                )

        self._raise_if_cancelled(control)
        chart_decisions = self._chart_builder(plan, manifest)

        total_sections = len(plan.sections)
        if len(completed_by_id) < total_sections:
            progress = max(
                progress,
                25 + math.floor(60 * len(completed_by_id) / total_sections),
            )
            checkpoint_payload = self._checkpoint_payload(
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
                    checkpoint=self._checkpoint_payload(
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
                raise ReportJobFailure(
                    "REPORT_SECTION_INVALID",
                    retryable=True,
                ) from exc
            except (ValidationError, ValueError) as exc:
                raise ReportJobFailure(
                    "REPORT_SECTION_INVALID",
                    retryable=True,
                ) from exc
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
            checkpoint=self._checkpoint_payload(
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
            raise ReportJobFailure(
                "REPORT_ASSEMBLY_INVALID",
                retryable=False,
            ) from exc
        return result.model_dump(mode="json")
