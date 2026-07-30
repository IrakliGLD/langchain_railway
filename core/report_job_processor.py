"""Checkpointed orchestration for one durable analytical-report job."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from agent.report_assembly import ReportAssemblyError, assemble_report
from agent.report_charts import (
    build_report_chart_requests,
    build_report_charts,
    build_report_research_exhibits,
    demote_unbuildable_required_charts,
)
from agent.report_document_assembly import (
    ReportDocumentAssemblyError,
    assemble_report_document,
)
from agent.report_document_generation import (
    ReportDocumentGenerationError,
    generate_report_document,
)
from agent.report_document_planner import build_report_document_plan
from agent.report_evaluation import evaluate_report_plan
from agent.report_evidence import (
    build_report_evidence_manifest,
)
from agent.report_evidence_gate import evaluate_report_evidence
from agent.report_grounding import build_evidence_grounding_index
from agent.report_intent import build_report_planning_context
from agent.report_planner import (
    ReportPlanEvidenceError,
    ReportPlanSemanticError,
    plan_report,
    validate_report_plan_semantics,
)
from agent.report_research_execution import (
    consolidate_report_evidence_packets,
    execute_report_research,
)
from agent.report_research_planner import (
    ReportResearchPlanError,
    plan_report_research,
    validate_report_research_plan,
)
from agent.report_sections import (
    ReportSectionGenerationError,
    generate_report_sections,
)
from config import (
    REPORT_MAX_GENERATIVE_CALLS,
    REPORT_PIPELINE_V2_MODE,
    REPORT_RESEARCH_MAX_TRACKS,
    REPORT_RESEARCH_MAX_WORKERS,
)
from contracts.report import ReportPlan, ReportPlanningContext
from contracts.report_charts import ReportChartBuildDecision
from contracts.report_document import (
    ReportDocumentDraft,
    ReportDocumentPlan,
)
from contracts.report_evidence import ReportEvidenceKind, ReportEvidenceManifest
from contracts.report_generation import (
    REPORT_GENERATION_CHECKPOINT_MAX_BYTES,
    ReportCheckpointTooLargeError,
    ReportGenerationCheckpoint,
)
from contracts.report_jobs import ReportJobLease, ReportJobPhase
from contracts.report_research import (
    ReportEvidenceGate,
    ReportEvidencePacket,
    ReportResearchPlan,
)
from contracts.report_sections import ReportSectionDraft
from core.report_job_worker import (
    ReportJobExecutionControl,
    ReportJobFailure,
)
from utils.metrics import metrics
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
ResearchPlanner = Callable[..., Any]
ResearchExecutor = Callable[..., Any]
ManifestConsolidator = Callable[..., Any]
ResearchExhibitBuilder = Callable[..., Any]
EvidenceGateEvaluator = Callable[..., Any]
DocumentPlanner = Callable[..., Any]
DocumentGenerator = Callable[..., Any]
DocumentAssembler = Callable[..., Any]
DocumentChartBuilder = Callable[..., Any]

# Stages the generative-call budget governs: planning the report, writing it,
# and repairing it. Deliberately excludes report_question_analyzer, which is the
# query pipeline's analyzer running under a report-prefixed stage name during
# narrative enrichment.
_REPORT_GENERATION_STAGES = frozenset(
    {
        "report_research_planner",
        "report_plan_repair",
        "report_document_writer",
        "report_analysis_writer",
        "report_synthesis_writer",
        "report_document_repair",
    }
)
# Provider attempts are claimed once per (actor, request_id, provider, stage).
# Narrative enrichment runs the whole query pipeline inside the report's own
# request identity, so its vector-knowledge stage asked for a second
# gemini|query_embedding and was refused before it could send. Give the nested
# run its own request identity: these are two distinct logical calls, and the
# no-replay guarantee still holds within each.
_NARRATIVE_REQUEST_ID_SUFFIX = ":narrative"


def _is_report_generation_stage(stage: str) -> bool:
    # Legacy per-section stages carry the section id and attempt number.
    return stage in _REPORT_GENERATION_STAGES or stage.startswith(
        "report_section_"
    )


def _report_document_allows_repair(
    *,
    profile: Any,
    generative_calls_used: int,
    maximum_calls: int,
) -> bool:
    profile_value = getattr(profile, "value", profile)
    generation_calls = 1 if profile_value == "compact" else 2
    return (
        generative_calls_used + generation_calls + 1
        <= maximum_calls
    )

_REPORT_FAILURE_RETRYABILITY = {
    "REPORT_ASSEMBLY_INVALID": False,
    "REPORT_CANCELLED": False,
    "REPORT_CHECKPOINT_INVALID": False,
    "REPORT_CHECKPOINT_TOO_LARGE": False,
    "REPORT_DEADLINE_EXCEEDED": True,
    "REPORT_DOCUMENT_INVALID": False,
    "REPORT_DOCUMENT_PROVIDER_FAILED": True,
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
_PROJECTED_SECTION_CHECKPOINT_BYTES = 80_000


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


def _report_failure(
    error_code: str,
    *,
    retryable: bool | None = None,
) -> ReportJobFailure:
    try:
        default_retryable = _REPORT_FAILURE_RETRYABILITY[error_code]
    except KeyError as exc:
        raise ValueError("Unknown report failure policy code.") from exc
    return ReportJobFailure(
        error_code,
        retryable=default_retryable if retryable is None else retryable,
    )


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
        pipeline_v2_mode: str = REPORT_PIPELINE_V2_MODE,
        max_generative_calls: int = REPORT_MAX_GENERATIVE_CALLS,
        research_planner: ResearchPlanner = plan_report_research,
        max_research_tracks: int = REPORT_RESEARCH_MAX_TRACKS,
        research_executor: ResearchExecutor = execute_report_research,
        manifest_consolidator: ManifestConsolidator = (
            consolidate_report_evidence_packets
        ),
        max_research_workers: int = REPORT_RESEARCH_MAX_WORKERS,
        research_exhibit_builder: ResearchExhibitBuilder = (
            build_report_research_exhibits
        ),
        evidence_gate_evaluator: EvidenceGateEvaluator = (
            evaluate_report_evidence
        ),
        document_planner: DocumentPlanner = build_report_document_plan,
        document_generator: DocumentGenerator = generate_report_document,
        document_assembler: DocumentAssembler = (
            assemble_report_document
        ),
        document_chart_builder: DocumentChartBuilder = (
            build_report_chart_requests
        ),
    ) -> None:
        if not 1 <= max_section_workers <= 8:
            raise ValueError(
                "max_section_workers must be between 1 and 8."
            )
        if not 1 <= job_timeout_seconds <= 3600:
            raise ValueError(
                "job_timeout_seconds must be between 1 and 3600."
            )
        if pipeline_v2_mode not in {"disabled", "shadow", "enabled"}:
            raise ValueError(
                "pipeline_v2_mode must be disabled, shadow, or enabled."
            )
        if not 2 <= max_generative_calls <= 6:
            raise ValueError(
                "max_generative_calls must be between 2 and 6."
            )
        if not 1 <= max_research_tracks <= 8:
            raise ValueError(
                "max_research_tracks must be between 1 and 8."
            )
        if not 1 <= max_research_workers <= max_research_tracks:
            raise ValueError(
                "max_research_workers must be between 1 and "
                "max_research_tracks."
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
        self._pipeline_v2_mode = pipeline_v2_mode
        self._max_generative_calls = max_generative_calls
        self._research_planner = research_planner
        self._max_research_tracks = max_research_tracks
        self._research_executor = research_executor
        self._manifest_consolidator = manifest_consolidator
        self._max_research_workers = max_research_workers
        self._research_exhibit_builder = research_exhibit_builder
        self._evidence_gate_evaluator = evidence_gate_evaluator
        self._document_planner = document_planner
        self._document_generator = document_generator
        self._document_assembler = document_assembler
        self._document_chart_builder = document_chart_builder

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
        *,
        planning_context: ReportPlanningContext | None = None,
    ) -> str:
        checkpoint_fields: dict[str, Any] = {
            "contract_version": (
                "report-generation-checkpoint-v2"
                if planning_context is not None
                else "report-generation-checkpoint-v1"
            ),
            "manifest": manifest,
            "plan": plan,
            "completed_sections": [
                completed_by_id[section.section_id]
                for section in plan.sections
                if section.section_id in completed_by_id
            ],
        }
        if planning_context is not None:
            checkpoint_fields.update(
                {
                    "checkpoint_stage": "plan_ready",
                    "planning_context": planning_context,
                }
            )
        checkpoint = ReportGenerationCheckpoint(**checkpoint_fields)
        return checkpoint.durable_json()

    @staticmethod
    def _evidence_checkpoint_payload(
        manifest: ReportEvidenceManifest,
        planning_context: ReportPlanningContext,
    ) -> str:
        checkpoint = ReportGenerationCheckpoint(
            contract_version="report-generation-checkpoint-v2",
            checkpoint_stage="evidence_ready",
            manifest=manifest,
            planning_context=planning_context,
            plan=None,
            completed_sections=[],
        )
        return checkpoint.durable_json()

    @classmethod
    def _safe_checkpoint_payload(
        cls,
        manifest: ReportEvidenceManifest,
        plan: ReportPlan,
        completed_by_id: dict[str, ReportSectionDraft],
        *,
        planning_context: ReportPlanningContext | None = None,
    ) -> str:
        """Build a checkpoint, separating "too big" from "structurally wrong"."""

        try:
            return cls._checkpoint_payload(
                manifest,
                plan,
                completed_by_id,
                planning_context=planning_context,
            )
        except ReportCheckpointTooLargeError as exc:
            raise _report_failure("REPORT_CHECKPOINT_TOO_LARGE") from exc
        except (ValidationError, ValueError) as exc:
            raise _report_failure("REPORT_CHECKPOINT_INVALID") from exc

    @classmethod
    def _safe_evidence_checkpoint_payload(
        cls,
        manifest: ReportEvidenceManifest,
        planning_context: ReportPlanningContext,
    ) -> str:
        try:
            return cls._evidence_checkpoint_payload(
                manifest,
                planning_context,
            )
        except ReportCheckpointTooLargeError as exc:
            raise _report_failure("REPORT_CHECKPOINT_TOO_LARGE") from exc
        except (ValidationError, ValueError) as exc:
            raise _report_failure("REPORT_CHECKPOINT_INVALID") from exc

    @staticmethod
    def _v3_checkpoint_payload(
        *,
        checkpoint_stage: str,
        research_plan: ReportResearchPlan,
        completed_packets: list[ReportEvidencePacket] | None = None,
        manifest: ReportEvidenceManifest | None = None,
        document_plan: ReportDocumentPlan | None = None,
        document_draft: ReportDocumentDraft | None = None,
    ) -> str:
        checkpoint = ReportGenerationCheckpoint(
            contract_version="report-generation-checkpoint-v3",
            checkpoint_stage=checkpoint_stage,
            research_plan=research_plan,
            completed_packets=completed_packets or [],
            manifest=manifest,
            document_plan=document_plan,
            document_draft=document_draft,
        )
        return checkpoint.durable_json()

    @classmethod
    def _safe_v3_checkpoint_payload(
        cls,
        **kwargs: Any,
    ) -> str:
        try:
            return cls._v3_checkpoint_payload(**kwargs)
        except ReportCheckpointTooLargeError as exc:
            raise _report_failure("REPORT_CHECKPOINT_TOO_LARGE") from exc
        except (ValidationError, ValueError) as exc:
            raise _report_failure("REPORT_CHECKPOINT_INVALID") from exc

    @staticmethod
    def _projected_final_checkpoint_size_bytes(
        manifest: ReportEvidenceManifest,
        plan: ReportPlan,
        completed_by_id: dict[str, ReportSectionDraft],
        *,
        planning_context: ReportPlanningContext | None,
    ) -> int:
        current_payload = ReportJobProcessor._checkpoint_payload(
            manifest,
            plan,
            completed_by_id,
            planning_context=planning_context,
        )
        current_bytes = len(current_payload.encode("utf-8"))
        remaining_sections = len(plan.sections) - len(completed_by_id)
        return (
            current_bytes
            + remaining_sections * _PROJECTED_SECTION_CHECKPOINT_BYTES
        )

    @staticmethod
    def _heartbeat(
        control: ReportJobExecutionControl,
        *,
        phase: ReportJobPhase,
        progress_percent: int,
        checkpoint: dict[str, Any] | str | None,
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

    def _finalize_attempt_telemetry(
        self,
        lease: ReportJobLease,
        *,
        outcome: str,
        started_at: float,
    ) -> None:
        """Log one content-free cost and stage summary without masking failures."""

        try:
            usage = metrics.finalize_request_telemetry()
            llm_calls = max(0, int(usage.get("llm_calls", 0)))
            # The generative budget governs the report's own generation stages.
            # Narrative enrichment runs the query pipeline, whose calls land in
            # llm_calls too — and one of them is named report_question_analyzer,
            # so a "report_" prefix test counts enrichment as generation and
            # reports every enriched report as over budget. Match the stages
            # the budget actually governs.
            stage_usage = usage.get("stages", {})
            report_stage_calls = sum(
                max(0, int((stats or {}).get("calls", 0)))
                for stage, stats in (
                    stage_usage.items()
                    if isinstance(stage_usage, dict)
                    else ()
                )
                if _is_report_generation_stage(str(stage))
            )
            payload = {
                "attempt": lease.attempt_count,
                "completion_tokens": max(
                    0,
                    int(usage.get("completion_tokens", 0)),
                ),
                "duration_ms": round(
                    (time.monotonic() - started_at) * 1000,
                    3,
                ),
                "estimated_cost_usd": max(
                    0.0,
                    float(usage.get("estimated_cost_usd", 0.0)),
                ),
                "generative_call_budget": self._max_generative_calls,
                "job_id": str(lease.job_id),
                "llm_calls": llm_calls,
                "models": usage.get("models", {}),
                "outcome": outcome,
                "over_generative_call_budget": (
                    report_stage_calls > self._max_generative_calls
                ),
                "report_stage_calls": report_stage_calls,
                "pipeline_v2_mode": self._pipeline_v2_mode,
                "prompt_tokens": max(
                    0,
                    int(usage.get("prompt_tokens", 0)),
                ),
                # The cached share of the prompt, so prefix-cache behaviour is
                # measurable per attempt rather than inferred from cost.
                "cached_prompt_tokens": max(
                    0,
                    int(usage.get("cached_prompt_tokens", 0)),
                ),
                "stages": usage.get("stages", {}),
                "total_tokens": max(
                    0,
                    int(usage.get("total_tokens", 0)),
                ),
            }
            _LOGGER.info(
                "REPORT_JOB_ATTEMPT_TELEMETRY %s",
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        except Exception:
            _LOGGER.exception(
                "Report attempt telemetry finalization failed: "
                "job_id=%s job_attempt=%s",
                lease.job_id,
                lease.attempt_count,
            )

    def _pipeline_narrative_items(
        self,
        lease: ReportJobLease,
    ) -> list[Any]:
        """Return the standard pipeline's computed statistics and knowledge.

        The adaptive collectors return raw tables only. The query pipeline is
        what computes statistics and selects curated knowledge, and a report
        without them can only restate cells — which is why adaptive reports
        read far weaker than the same question asked in standard mode.

        Enrichment must never fail a report, so every failure here degrades to
        an empty list and is logged.
        """

        from agent.report_evidence import build_report_narrative_items

        try:
            context = self._run_query_pipeline(lease)
            items = build_report_narrative_items(context)
        except Exception as exc:
            _LOGGER.warning(
                "Report narrative enrichment skipped: job_id=%s "
                "job_attempt=%s exception_type=%s",
                lease.job_id,
                lease.attempt_count,
                _diagnostic_identifier(type(exc).__name__),
            )
            return []
        _LOGGER.info(
            "Report narrative enrichment ready: job_id=%s kinds=%s",
            lease.job_id,
            ",".join(item.kind.value for item in items) or "none",
        )
        return items

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
        # process_query binds its own execution scope from this request_id, so
        # deriving it here is what separates the nested run's provider claims
        # from the report's. trace_id stays the job id, so traces still join.
        return pipeline(
            lease.query,
            trace_id=str(lease.job_id),
            actor_id=str(lease.actor_user_id),
            request_id=f"{request_id}{_NARRATIVE_REQUEST_ID_SUFFIX}",
            request_deadline=request_deadline,
            answer_mode="report",
        )

    def _run_shadow_research_planner(self, lease: ReportJobLease) -> None:
        """Evaluate v2 planning without changing durable state or output."""

        started_at = time.monotonic()
        payload: dict[str, Any] = {
            "job_id": str(lease.job_id),
            "mode": "shadow",
            "outcome": "failed",
            "duration_ms": 0,
            "track_count": 0,
            "required_track_count": 0,
            "packet_count": 0,
            "evidence_item_count": 0,
            "numeric_observation_count": 0,
            "coverage_status": "",
            "coverage_findings": [],
            "built_chart_count": 0,
            "omitted_chart_count": 0,
            "document_plan_outcome": "not_run",
            "document_section_count": 0,
            "document_target_words": 0,
            "document_chart_count": 0,
            "collector_ids": [],
            "recognized_requirements": [],
            "finding_codes": [],
        }
        try:
            raw_plan = self._research_planner(
                lease.query,
                max_tracks=self._max_research_tracks,
            )
            plan = (
                raw_plan
                if isinstance(raw_plan, ReportResearchPlan)
                else ReportResearchPlan.model_validate(raw_plan)
            )
            assessment = validate_report_research_plan(
                lease.query,
                plan,
                max_tracks=self._max_research_tracks,
            )
            if not assessment.valid:
                raise ReportResearchPlanError(assessment)
            raw_packets = self._research_executor(
                lease.query,
                plan,
                max_workers=self._max_research_workers,
            )
            packets = [
                packet
                if isinstance(packet, ReportEvidencePacket)
                else ReportEvidencePacket.model_validate(packet)
                for packet in raw_packets
            ]
            raw_manifest = self._manifest_consolidator(
                lease.query,
                packets,
            )
            manifest = (
                raw_manifest
                if isinstance(raw_manifest, ReportEvidenceManifest)
                else ReportEvidenceManifest.model_validate(raw_manifest)
            )
            self._validate_query_binding(lease.query, manifest)
            raw_chart_decisions = self._research_exhibit_builder(
                packets,
                manifest,
            )
            chart_decisions = [
                decision
                if isinstance(decision, ReportChartBuildDecision)
                else ReportChartBuildDecision.model_validate(decision)
                for decision in raw_chart_decisions
            ]
            raw_gate = self._evidence_gate_evaluator(
                plan,
                packets,
                chart_decisions=chart_decisions,
            )
            gate = (
                raw_gate
                if isinstance(raw_gate, ReportEvidenceGate)
                else ReportEvidenceGate.model_validate(raw_gate)
            )
            if gate.query_digest != plan.query_digest:
                raise ValueError("Evidence gate query identity mismatch.")
            payload.update(
                outcome=(
                    "valid"
                    if gate.ready_for_writing
                    else "quality_failed"
                ),
                track_count=len(plan.tracks),
                required_track_count=sum(
                    1 for track in plan.tracks if track.required
                ),
                collector_ids=sorted(
                    {
                        collector.value
                        for track in plan.tracks
                        for collector in track.collector_ids
                    }
                ),
                recognized_requirements=[
                    requirement.value
                    for requirement in assessment.recognized_requirements
                ],
                packet_count=len(packets),
                evidence_item_count=len(manifest.items),
                numeric_observation_count=sum(
                    packet.numeric_observation_count
                    for packet in packets
                ),
                coverage_status=gate.status.value,
                coverage_findings=list(gate.finding_codes),
                built_chart_count=sum(
                    1
                    for decision in chart_decisions
                    if decision.status == "built"
                ),
                omitted_chart_count=sum(
                    1
                    for decision in chart_decisions
                    if decision.status == "omitted"
                ),
            )
            if gate.ready_for_writing:
                try:
                    raw_document_plan = self._document_planner(
                        lease.query,
                        plan,
                        packets,
                        manifest,
                        gate,
                        chart_decisions,
                    )
                    document_plan = (
                        raw_document_plan
                        if isinstance(
                            raw_document_plan,
                            ReportDocumentPlan,
                        )
                        else ReportDocumentPlan.model_validate(
                            raw_document_plan
                        )
                    )
                    payload.update(
                        document_plan_outcome="valid",
                        document_section_count=len(
                            document_plan.sections
                        ),
                        document_target_words=(
                            document_plan.target_words
                        ),
                        document_chart_count=len(
                            document_plan.charts
                        ),
                    )
                except Exception as exc:
                    payload["document_plan_outcome"] = "failed"
                    payload["document_plan_error_type"] = (
                        _diagnostic_identifier(type(exc).__name__)
                    )
        except ReportResearchPlanError as exc:
            payload["finding_codes"] = list(
                exc.assessment.finding_codes
            )
            payload["error_type"] = type(exc).__name__
        except Exception as exc:
            payload["error_type"] = _diagnostic_identifier(
                type(exc).__name__
            )
        finally:
            payload["duration_ms"] = max(
                0,
                int((time.monotonic() - started_at) * 1000),
            )
            _LOGGER.info(
                "REPORT_RESEARCH_PLAN_SHADOW %s",
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

    def __call__(
        self,
        lease: ReportJobLease,
        control: ReportJobExecutionControl,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        outcome = "failed"
        metrics.start_request_telemetry(str(lease.job_id))
        try:
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
                    result = self._run_bound_attempt(lease, control)
                    outcome = "completed"
                    return result
                except RequestDeadlineExceeded as exc:
                    _LOGGER.warning(
                        "Report deadline exceeded: job_id=%s job_attempt=%s "
                        "stage=%s",
                        lease.job_id,
                        lease.attempt_count,
                        _diagnostic_identifier(exc.stage),
                    )
                    raise _report_failure(
                        "REPORT_DEADLINE_EXCEEDED"
                    ) from exc
        finally:
            self._finalize_attempt_telemetry(
                lease,
                outcome=outcome,
                started_at=started_at,
            )

    def _run_v2_bound_attempt(
        self,
        lease: ReportJobLease,
        control: ReportJobExecutionControl,
    ) -> dict[str, Any]:
        """Run the bounded research/document pipeline without legacy routing."""

        progress = lease.progress_percent
        checkpoint: ReportGenerationCheckpoint | None = None
        research_plan: ReportResearchPlan | None = None
        packets: list[ReportEvidencePacket] = []
        manifest: ReportEvidenceManifest | None = None
        document_plan: ReportDocumentPlan | None = None
        document_draft: ReportDocumentDraft | None = None
        chart_decisions: list[ReportChartBuildDecision] = []
        generative_calls_used = 0

        if lease.checkpoint is not None:
            try:
                checkpoint = ReportGenerationCheckpoint.model_validate(
                    lease.checkpoint
                )
            except (ValidationError, ValueError) as exc:
                raise _report_failure("REPORT_CHECKPOINT_INVALID") from exc
            if checkpoint.contract_version != (
                "report-generation-checkpoint-v3"
            ):
                raise _report_failure("REPORT_CHECKPOINT_INVALID")
            expected_digest = hashlib.sha256(
                lease.query.encode("utf-8")
            ).hexdigest()
            if checkpoint.research_plan is None or (
                checkpoint.research_plan.query_digest != expected_digest
            ):
                raise _report_failure("REPORT_CHECKPOINT_INVALID")
            expected_phase_by_stage = {
                "research_plan_ready": ReportJobPhase.PLANNING,
                "evidence_collecting": ReportJobPhase.PLANNING,
                "evidence_ready": ReportJobPhase.PLANNING,
                "document_plan_ready": (
                    ReportJobPhase.GENERATING_SECTIONS
                ),
                "draft_ready": ReportJobPhase.ASSEMBLING,
            }
            if lease.phase is not expected_phase_by_stage[
                checkpoint.checkpoint_stage
            ]:
                raise _report_failure("REPORT_CHECKPOINT_INVALID")
            research_plan = checkpoint.research_plan
            packets = list(checkpoint.completed_packets)
            manifest = checkpoint.manifest
            document_plan = checkpoint.document_plan
            document_draft = checkpoint.document_draft

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
                raw_plan = self._research_planner(
                    lease.query,
                    max_tracks=self._max_research_tracks,
                )
                research_plan = (
                    raw_plan
                    if isinstance(raw_plan, ReportResearchPlan)
                    else ReportResearchPlan.model_validate(raw_plan)
                )
                assessment = validate_report_research_plan(
                    lease.query,
                    research_plan,
                    max_tracks=self._max_research_tracks,
                )
                if not assessment.valid:
                    raise ReportResearchPlanError(assessment)
            except ProviderExecutionError as exc:
                _LOGGER.warning(
                    "Report v2 research-planner provider failure: "
                    "job_id=%s job_attempt=%s provider=%s "
                    "provider_stage=%s provider_disposition=%s",
                    lease.job_id,
                    lease.attempt_count,
                    _diagnostic_identifier(exc.provider),
                    _diagnostic_identifier(exc.stage),
                    exc.disposition.value,
                )
                raise _report_failure(
                    "REPORT_PLAN_PROVIDER_FAILED",
                    retryable=exc.safe_to_retry,
                ) from exc
            except ReportResearchPlanError as exc:
                _LOGGER.warning(
                    "Report v2 research plan rejected: job_id=%s "
                    "job_attempt=%s finding_codes=%s "
                    "schema_error_codes=%s",
                    lease.job_id,
                    lease.attempt_count,
                    _diagnostic_error_codes(
                        list(exc.assessment.finding_codes)
                    ),
                    _diagnostic_error_codes(
                        list(exc.schema_error_codes)
                    ),
                )
                raise _report_failure(
                    "REPORT_PLAN_INVALID",
                    retryable=True,
                ) from exc
            except (ValidationError, ValueError) as exc:
                raise _report_failure("REPORT_PLAN_INVALID") from exc
            generative_calls_used = 1
            progress = max(progress, 20)
            self._heartbeat(
                control,
                phase=ReportJobPhase.PLANNING,
                progress_percent=progress,
                checkpoint=self._safe_v3_checkpoint_payload(
                    checkpoint_stage="research_plan_ready",
                    research_plan=research_plan,
                ),
            )

        if research_plan is None:
            raise _report_failure("REPORT_CHECKPOINT_INVALID")

        if document_plan is None:
            self._raise_if_cancelled(control)
            try:
                # Consolidated checkpoints intentionally avoid duplicating the
                # manifest inside packets. If a retry stopped before document
                # planning, re-run the deterministic collectors; the expensive
                # research-planner model call remains checkpointed.
                if (
                    len(packets) != len(research_plan.tracks)
                    or {
                        packet.track_id for packet in packets
                    }
                    != {
                        track.track_id for track in research_plan.tracks
                    }
                ):
                    raw_packets = self._research_executor(
                        lease.query,
                        research_plan,
                        max_workers=self._max_research_workers,
                    )
                    packets = [
                        packet
                        if isinstance(packet, ReportEvidencePacket)
                        else ReportEvidencePacket.model_validate(packet)
                        for packet in raw_packets
                    ]
                raw_manifest = self._manifest_consolidator(
                    lease.query,
                    packets,
                    extra_items=self._pipeline_narrative_items(lease),
                )
                manifest = (
                    raw_manifest
                    if isinstance(raw_manifest, ReportEvidenceManifest)
                    else ReportEvidenceManifest.model_validate(raw_manifest)
                )
                self._validate_query_binding(lease.query, manifest)
                raw_decisions = self._research_exhibit_builder(
                    packets,
                    manifest,
                )
                chart_decisions = [
                    decision
                    if isinstance(decision, ReportChartBuildDecision)
                    else ReportChartBuildDecision.model_validate(
                        decision
                    )
                    for decision in raw_decisions
                ]
                raw_gate = self._evidence_gate_evaluator(
                    research_plan,
                    packets,
                    chart_decisions=chart_decisions,
                )
                gate = (
                    raw_gate
                    if isinstance(raw_gate, ReportEvidenceGate)
                    else ReportEvidenceGate.model_validate(raw_gate)
                )
                if (
                    gate.query_digest != research_plan.query_digest
                    or not gate.ready_for_writing
                ):
                    raise _report_failure(
                        "REPORT_EVIDENCE_UNAVAILABLE"
                    )
            except ReportJobFailure:
                raise
            except (ValidationError, ValueError) as exc:
                raise _report_failure("REPORT_EVIDENCE_INVALID") from exc
            try:
                raw_document_plan = self._document_planner(
                    lease.query,
                    research_plan,
                    packets,
                    manifest,
                    gate,
                    chart_decisions,
                )
                document_plan = (
                    raw_document_plan
                    if isinstance(
                        raw_document_plan,
                        ReportDocumentPlan,
                    )
                    else ReportDocumentPlan.model_validate(
                        raw_document_plan
                    )
                )
            except ReportJobFailure:
                raise
            except (ValidationError, ValueError) as exc:
                raise _report_failure("REPORT_PLAN_INVALID") from exc
            progress = max(progress, 55)
            self._heartbeat(
                control,
                phase=ReportJobPhase.GENERATING_SECTIONS,
                progress_percent=progress,
                checkpoint=self._safe_v3_checkpoint_payload(
                    checkpoint_stage="document_plan_ready",
                    research_plan=research_plan,
                    manifest=manifest,
                    document_plan=document_plan,
                ),
            )
        else:
            if manifest is None:
                raise _report_failure("REPORT_CHECKPOINT_INVALID")
            self._validate_query_binding(lease.query, manifest)
            try:
                raw_decisions = self._document_chart_builder(
                    list(document_plan.charts),
                    manifest,
                )
                chart_decisions = [
                    decision
                    if isinstance(decision, ReportChartBuildDecision)
                    else ReportChartBuildDecision.model_validate(
                        decision
                    )
                    for decision in raw_decisions
                ]
            except (ValidationError, ValueError) as exc:
                raise _report_failure(
                    "REPORT_CHECKPOINT_INVALID"
                ) from exc

        if manifest is None or document_plan is None:
            raise _report_failure("REPORT_CHECKPOINT_INVALID")

        if document_draft is None:
            self._raise_if_cancelled(control)
            if checkpoint is not None:
                progress = max(progress, 55)
                self._heartbeat(
                    control,
                    phase=ReportJobPhase.GENERATING_SECTIONS,
                    progress_percent=progress,
                    checkpoint=None,
                )
            allow_repair = _report_document_allows_repair(
                profile=document_plan.profile,
                generative_calls_used=generative_calls_used,
                maximum_calls=self._max_generative_calls,
            )
            try:
                raw_draft = self._document_generator(
                    lease.query,
                    document_plan,
                    research_plan,
                    manifest,
                    packets,
                    allow_repair=allow_repair,
                )
                document_draft = (
                    raw_draft
                    if isinstance(raw_draft, ReportDocumentDraft)
                    else ReportDocumentDraft.model_validate(raw_draft)
                )
            except ProviderExecutionError as exc:
                _LOGGER.warning(
                    "Report v2 document provider failure: job_id=%s "
                    "job_attempt=%s provider=%s provider_stage=%s "
                    "provider_disposition=%s",
                    lease.job_id,
                    lease.attempt_count,
                    _diagnostic_identifier(exc.provider),
                    _diagnostic_identifier(exc.stage),
                    exc.disposition.value,
                )
                raise _report_failure(
                    "REPORT_DOCUMENT_PROVIDER_FAILED",
                    retryable=exc.safe_to_retry,
                ) from exc
            except ReportDocumentGenerationError as exc:
                _LOGGER.warning(
                    "Report v2 document rejected: job_id=%s "
                    "job_attempt=%s error_codes=%s",
                    lease.job_id,
                    lease.attempt_count,
                    _diagnostic_error_codes(
                        [
                            *exc.validation.document_errors,
                            *(
                                code
                                for errors in (
                                    exc.validation.section_errors.values()
                                )
                                for code in errors
                            ),
                        ]
                    ),
                )
                raise _report_failure(
                    "REPORT_DOCUMENT_INVALID"
                ) from exc
            except (ValidationError, ValueError) as exc:
                raise _report_failure(
                    "REPORT_DOCUMENT_INVALID"
                ) from exc
            progress = max(progress, 90)
            self._heartbeat(
                control,
                phase=ReportJobPhase.ASSEMBLING,
                progress_percent=progress,
                checkpoint=self._safe_v3_checkpoint_payload(
                    checkpoint_stage="draft_ready",
                    research_plan=research_plan,
                    manifest=manifest,
                    document_plan=document_plan,
                    document_draft=document_draft,
                ),
            )
        else:
            progress = max(progress, 90)
            self._heartbeat(
                control,
                phase=ReportJobPhase.ASSEMBLING,
                progress_percent=progress,
                checkpoint=None,
            )

        self._raise_if_cancelled(control)
        try:
            result = self._document_assembler(
                document_plan,
                research_plan,
                manifest,
                document_draft,
                chart_decisions,
            )
        except (
            ReportDocumentAssemblyError,
            ValidationError,
            ValueError,
        ) as exc:
            raise _report_failure("REPORT_ASSEMBLY_INVALID") from exc
        return result.model_dump(mode="json")

    def _run_bound_attempt(
        self,
        lease: ReportJobLease,
        control: ReportJobExecutionControl,
    ) -> dict[str, Any]:
        if self._pipeline_v2_mode == "enabled":
            checkpoint_version = (
                str(lease.checkpoint.get("contract_version", ""))
                if isinstance(lease.checkpoint, dict)
                else ""
            )
            if not lease.checkpoint or checkpoint_version == (
                "report-generation-checkpoint-v3"
            ):
                return self._run_v2_bound_attempt(lease, control)
            # Finish jobs already checkpointed on the legacy pipeline instead
            # of invalidating in-flight work during the rollout.
            return self._run_legacy_bound_attempt(lease, control)
        if (
            self._pipeline_v2_mode == "shadow"
            and lease.checkpoint is None
        ):
            self._run_shadow_research_planner(lease)
        return self._run_legacy_bound_attempt(lease, control)

    def _run_legacy_bound_attempt(
        self,
        lease: ReportJobLease,
        control: ReportJobExecutionControl,
    ) -> dict[str, Any]:
        progress = lease.progress_percent
        checkpoint: ReportGenerationCheckpoint | None = None
        manifest: ReportEvidenceManifest | None = None
        planning_context: ReportPlanningContext | None = None
        if lease.checkpoint is not None:
            try:
                checkpoint = ReportGenerationCheckpoint.model_validate(
                    lease.checkpoint
                )
            except (ValidationError, ValueError) as exc:
                raise _report_failure("REPORT_CHECKPOINT_INVALID") from exc
            if checkpoint.manifest is None:
                raise _report_failure("REPORT_CHECKPOINT_INVALID")
            self._validate_query_binding(lease.query, checkpoint.manifest)
            manifest = checkpoint.manifest
            planning_context = checkpoint.planning_context
            if (
                checkpoint.plan is None
                and lease.phase is not ReportJobPhase.PLANNING
            ):
                raise _report_failure("REPORT_CHECKPOINT_INVALID")

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

            progress = max(progress, 20)
            self._heartbeat(
                control,
                phase=ReportJobPhase.PLANNING,
                progress_percent=progress,
                checkpoint=self._safe_evidence_checkpoint_payload(
                    manifest,
                    planning_context,
                ),
            )

        if manifest is None:
            raise _report_failure("REPORT_CHECKPOINT_INVALID")
        if checkpoint is not None and checkpoint.plan is None:
            if planning_context is None:
                raise _report_failure("REPORT_CHECKPOINT_INVALID")
            if (
                planning_context.requires_table
                and not any(
                    item.kind is ReportEvidenceKind.TABLE
                    for item in manifest.items
                )
            ):
                raise _report_failure("REPORT_CHECKPOINT_INVALID")

        checkpointed_by_id: dict[str, ReportSectionDraft] = {}
        checkpoint_plan_is_current = False
        if checkpoint is None or checkpoint.plan is None:
            if planning_context is None:
                raise _report_failure("REPORT_CHECKPOINT_INVALID")
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
                raise _report_failure(
                    "REPORT_PLAN_PROVIDER_FAILED",
                    retryable=exc.safe_to_retry,
                ) from exc
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
                planning_context=planning_context,
            )
            self._heartbeat(
                control,
                phase=ReportJobPhase.GENERATING_SECTIONS,
                progress_percent=progress,
                checkpoint=checkpoint_payload,
            )
            checkpoint_plan_is_current = True
            checkpoint = None
        else:
            checkpoint_plan = checkpoint.plan
            plan = checkpoint_plan
            completed_by_id = {
                draft.section_id: draft
                for draft in checkpoint.completed_sections
            }
            try:
                if planning_context is not None:
                    validate_report_plan_semantics(
                        plan,
                        planning_context,
                    )
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
            checkpointed_by_id = dict(completed_by_id)
            checkpoint_plan_is_current = plan == checkpoint_plan

        projected_checkpoint_bytes = (
            self._projected_final_checkpoint_size_bytes(
                manifest,
                plan,
                completed_by_id,
                planning_context=planning_context,
            )
        )
        if (
            projected_checkpoint_bytes
            > REPORT_GENERATION_CHECKPOINT_MAX_BYTES
        ):
            _LOGGER.warning(
                "Projected report checkpoint exceeds the durable limit: "
                "job_id=%s job_attempt=%s projected_bytes=%s "
                "completed_sections=%s total_sections=%s",
                lease.job_id,
                lease.attempt_count,
                projected_checkpoint_bytes,
                len(completed_by_id),
                len(plan.sections),
            )
            raise _report_failure("REPORT_CHECKPOINT_TOO_LARGE")

        item_by_ref = manifest.item_by_ref()
        grounding_index = build_evidence_grounding_index(
            item_by_ref,
            {
                ref
                for section in plan.sections
                for ref in section.required_evidence_refs
            },
        )
        self._raise_if_cancelled(control)

        total_sections = len(plan.sections)
        if len(completed_by_id) < total_sections:
            progress = max(
                progress,
                25 + math.floor(60 * len(completed_by_id) / total_sections),
            )
            if checkpoint is not None:
                checkpoint_payload = (
                    None
                    if checkpoint_plan_is_current
                    else self._safe_checkpoint_payload(
                        manifest,
                        plan,
                        completed_by_id,
                        planning_context=planning_context,
                    )
                )
                self._heartbeat(
                    control,
                    phase=ReportJobPhase.GENERATING_SECTIONS,
                    progress_percent=progress,
                    checkpoint=checkpoint_payload,
                )
                checkpoint_plan_is_current = True

            def persist_section(
                completed: int,
                total: int,
                draft: ReportSectionDraft,
            ) -> None:
                nonlocal checkpointed_by_id, progress
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
                        planning_context=planning_context,
                    ),
                )
                checkpointed_by_id = dict(completed_by_id)

            try:
                drafts = self._section_generator(
                    lease.query,
                    plan,
                    manifest,
                    existing_drafts=completed_by_id,
                    progress_callback=persist_section,
                    max_workers=self._max_section_workers,
                    grounding_index=grounding_index,
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
        draft_by_id = {
            draft.section_id: draft
            for draft in drafts
        }
        assembly_checkpoint = (
            None
            if (
                checkpoint_plan_is_current
                and draft_by_id == checkpointed_by_id
            )
            else self._safe_checkpoint_payload(
                manifest,
                plan,
                draft_by_id,
                planning_context=planning_context,
            )
        )
        self._heartbeat(
            control,
            phase=ReportJobPhase.ASSEMBLING,
            progress_percent=progress,
            checkpoint=assembly_checkpoint,
        )
        try:
            result = self._assembler(
                plan,
                manifest,
                drafts,
                chart_decisions,
                grounding_index=grounding_index,
            )
        except (ReportAssemblyError, ValidationError, ValueError) as exc:
            raise _report_failure("REPORT_ASSEMBLY_INVALID") from exc
        return result.model_dump(mode="json")
