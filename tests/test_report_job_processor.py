"""End-to-end orchestration tests for one durable report job attempt."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from agent import report_grounding
from agent.report_charts import build_report_charts
from agent.report_evaluation import evaluate_report_plan
from agent.report_intent import build_report_planning_context
from agent.report_planner import ReportPlanEvidenceError
from agent.report_research_planner import plan_report_research
from agent.report_sections import (
    ReportSectionGenerationError,
    generate_report_sections,
)
from contracts.report import ReportPlan
from contracts.report_generation import (
    REPORT_GENERATION_CHECKPOINT_MAX_BYTES,
    ReportGenerationCheckpoint,
)
from contracts.report_jobs import ReportJobLease, ReportJobPhase
from contracts.report_result import ReportResult, ReportResultV2
from contracts.report_sections import ReportSectionDraft
from core import report_job_processor
from core.report_job_processor import ReportJobProcessor
from core.report_job_worker import ReportJobFailure
from models import QueryContext
from tests.test_report_document_pipeline_v2 import (
    _QUERY as _V2_QUERY,
)
from tests.test_report_document_pipeline_v2 import (
    _document_components,
    _valid_document_draft,
)
from tests.test_report_planner import _manifest, _plan_payload
from tests.test_report_research_contract import _research_plan_payload
from tests.test_report_sections import _draft
from utils.provider_attempts import (
    ProviderDeliveryDisposition,
    ProviderExecutionError,
)
from utils.request_deadline import (
    RequestDeadlineExceeded,
    current_request_execution_scope,
)


def _lease(
    *,
    checkpoint: dict | None = None,
    phase: str = "planning",
    progress_percent: int = 5,
    query: str = "Explain the price trend.",
) -> ReportJobLease:
    return ReportJobLease.model_validate(
        {
            "contract_version": "report-job-v1",
            "job_id": str(uuid4()),
            "request_id": "report:req-processor",
            "actor_user_id": str(uuid4()),
            "query": query,
            "attempt_count": 1,
            "max_attempts": 3,
            "lease_owner": "worker-processor",
            "lease_expires_at": datetime.now(timezone.utc) + timedelta(minutes=2),
            "phase": phase,
            "progress_percent": progress_percent,
            "cancel_requested": False,
            "checkpoint": checkpoint,
        }
    )


class _Control:
    def __init__(self) -> None:
        self.heartbeats: list[tuple[ReportJobPhase, int, dict | None]] = []
        self.cancelled = False

    def heartbeat(
        self,
        *,
        phase: ReportJobPhase,
        progress_percent: int,
        checkpoint: dict | None = None,
    ) -> bool:
        if isinstance(checkpoint, str):
            checkpoint = json.loads(checkpoint)
        self.heartbeats.append((phase, progress_percent, checkpoint))
        return not self.cancelled

    def cancellation_requested(self) -> bool:
        return self.cancelled


def _pipeline_context(query: str) -> QueryContext:
    return QueryContext(
        query=query,
        provenance_source="tool",
        provenance_cols=["period", "price"],
        provenance_rows=[
            ["2026-01", 120.0],
            ["2026-02", 130.0],
        ],
        provenance_refs=["query:tool:prices"],
        stats_hint="Average price was 125 GEL/MWh.",
        terminal_outcome="data_answer",
        answer_mode="report",
    )


def _manifest_for_query(query: str):
    payload = _manifest().model_dump(mode="json")
    payload["query_digest"] = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return type(_manifest()).model_validate(payload)


def _processor(
    *,
    pipeline_calls: list | None = None,
    generated: list | None = None,
    planning_contexts: list | None = None,
    evaluator=None,
    chart_builder=None,
    execution_scopes: list | None = None,
    job_timeout_seconds: int | None = None,
    research_planner=None,
    research_executor=None,
    manifest_consolidator=None,
    research_exhibit_builder=None,
    evidence_gate_evaluator=None,
    pipeline_v2_mode: str = "disabled",
    max_research_tracks: int = 4,
    max_research_workers: int = 3,
):
    calls = pipeline_calls if pipeline_calls is not None else []
    generated_ids = generated if generated is not None else []
    received_contexts = (
        planning_contexts
        if planning_contexts is not None
        else []
    )

    def pipeline(query, **kwargs):
        calls.append((query, kwargs))
        if execution_scopes is not None:
            execution_scopes.append(current_request_execution_scope())
        return _pipeline_context(query)

    def sections(query, plan, manifest, **kwargs):
        return generate_report_sections(
            query,
            plan,
            manifest,
            existing_drafts=kwargs["existing_drafts"],
            generate_section=lambda _q, _p, section, _m: (
                generated_ids.append(section.section_id) or _draft(section)
            ),
            progress_callback=kwargs["progress_callback"],
            max_workers=kwargs["max_workers"],
            grounding_index=kwargs.get("grounding_index"),
        )

    def planner(_query, _manifest_value, **kwargs):
        received_contexts.append(kwargs["planning_context"])
        return ReportPlan.model_validate(_plan_payload())

    overrides = {}
    if evaluator is not None:
        overrides["evaluator"] = evaluator
    if chart_builder is not None:
        overrides["chart_builder"] = chart_builder
    if job_timeout_seconds is not None:
        overrides["job_timeout_seconds"] = job_timeout_seconds
    if research_planner is not None:
        overrides["research_planner"] = research_planner
    if research_executor is not None:
        overrides["research_executor"] = research_executor
    if manifest_consolidator is not None:
        overrides["manifest_consolidator"] = manifest_consolidator
    if research_exhibit_builder is not None:
        overrides["research_exhibit_builder"] = research_exhibit_builder
    if evidence_gate_evaluator is not None:
        overrides["evidence_gate_evaluator"] = evidence_gate_evaluator
    return ReportJobProcessor(
        query_pipeline=pipeline,
        evidence_builder=lambda ctx: _manifest_for_query(ctx.query),
        planner=planner,
        section_generator=sections,
        max_section_workers=5,
        pipeline_v2_mode=pipeline_v2_mode,
        max_research_tracks=max_research_tracks,
        max_research_workers=max_research_workers,
        **overrides,
    )


def test_checkpoint_contract_binds_manifest_plan_and_completed_sections():
    plan = ReportPlan.model_validate(_plan_payload())
    first = ReportSectionDraft.model_validate(_draft(plan.sections[0]))
    checkpoint = ReportGenerationCheckpoint(
        contract_version="report-generation-checkpoint-v1",
        manifest=_manifest(),
        plan=plan,
        completed_sections=[first],
    )

    assert checkpoint.plan.evidence_manifest_id == checkpoint.manifest.manifest_id

    payload = checkpoint.model_dump(mode="json")
    payload["plan"]["evidence_manifest_id"] = "manifest:" + "9" * 32
    with pytest.raises(ValueError, match="manifest identity"):
        ReportGenerationCheckpoint.model_validate(payload)

    payload = checkpoint.model_dump(mode="json")
    payload["completed_sections"][0]["section_id"] = "unknown_section"
    with pytest.raises(ValueError, match="not present in the report plan"):
        ReportGenerationCheckpoint.model_validate(payload)


def test_legacy_processor_rejects_pre_manifest_v3_checkpoint_cleanly():
    checkpoint = ReportGenerationCheckpoint.model_validate(
        {
            "contract_version": "report-generation-checkpoint-v3",
            "checkpoint_stage": "research_plan_ready",
            "manifest": None,
            "research_plan": _research_plan_payload(),
        }
    )
    lease = _lease(checkpoint=checkpoint.model_dump(mode="json"))

    with pytest.raises(ReportJobFailure) as exc_info:
        _processor()(lease, _Control())

    assert exc_info.value.error_code == "REPORT_CHECKPOINT_INVALID"
    assert exc_info.value.retryable is False


def test_checkpoint_payload_is_serialized_once_before_repository_boundary():
    plan = ReportPlan.model_validate(_plan_payload())
    planning_context = build_report_planning_context(
        _pipeline_context("Explain the price trend.")
    )

    payload = ReportJobProcessor._checkpoint_payload(
        _manifest(),
        plan,
        {},
        planning_context=planning_context,
    )

    assert isinstance(payload, str)
    checkpoint = ReportGenerationCheckpoint.model_validate(
        json.loads(payload)
    )
    assert checkpoint.checkpoint_stage == "plan_ready"


def test_checkpoint_v2_expresses_only_valid_evidence_and_plan_stages():
    manifest = _manifest()
    planning_context = build_report_planning_context(
        _pipeline_context("Explain the price trend.")
    )
    evidence_ready = ReportGenerationCheckpoint(
        contract_version="report-generation-checkpoint-v2",
        checkpoint_stage="evidence_ready",
        manifest=manifest,
        planning_context=planning_context,
        plan=None,
        completed_sections=[],
    )

    assert evidence_ready.plan is None
    assert evidence_ready.planning_context == planning_context

    invalid = evidence_ready.model_dump(mode="json")
    invalid["completed_sections"] = [
        _draft(ReportPlan.model_validate(_plan_payload()).sections[0])
    ]
    with pytest.raises(ValueError, match="evidence_ready"):
        ReportGenerationCheckpoint.model_validate(invalid)

    invalid = evidence_ready.model_dump(mode="json")
    invalid["checkpoint_stage"] = "plan_ready"
    with pytest.raises(ValueError, match="plan_ready"):
        ReportGenerationCheckpoint.model_validate(invalid)

    invalid_context = planning_context.model_copy(
        update={"language_code": "ka"}
    )
    with pytest.raises(ValueError, match="planning context"):
        ReportGenerationCheckpoint(
            contract_version="report-generation-checkpoint-v2",
            checkpoint_stage="plan_ready",
            manifest=manifest,
            planning_context=invalid_context,
            plan=ReportPlan.model_validate(_plan_payload()),
            completed_sections=[],
        )


def test_fresh_job_runs_pipeline_parallel_sections_and_deterministic_assembly():
    pipeline_calls: list = []
    generated: list = []
    planning_contexts: list = []
    lease = _lease()
    control = _Control()

    raw_result = _processor(
        pipeline_calls=pipeline_calls,
        generated=generated,
        planning_contexts=planning_contexts,
    )(lease, control)
    result = ReportResult.model_validate(raw_result)

    assert result.contract_version == "report-result-v1"
    assert [section.section_id for section in result.sections] == [
        section["section_id"] for section in _plan_payload()["sections"]
    ]
    assert set(generated) == {
        section["section_id"] for section in _plan_payload()["sections"]
    }
    assert len(pipeline_calls) == 1
    pipeline_query, pipeline_kwargs = pipeline_calls[0]
    assert pipeline_query == lease.query
    assert pipeline_kwargs["trace_id"] == str(lease.job_id)
    assert pipeline_kwargs["actor_id"] == str(lease.actor_user_id)
    # Derived from the attempt identity, not equal to it: the nested pipeline
    # needs its own provider-attempt namespace (see the narrative-identity
    # tests), while still being traceable back to this attempt.
    assert pipeline_kwargs["request_id"] == (
        f"{lease.request_id}:attempt:{lease.attempt_count}:narrative"
    )
    assert pipeline_kwargs["request_deadline"].source == "report_job"
    assert pipeline_kwargs["answer_mode"] == "report"
    assert len(planning_contexts) == 1
    assert planning_contexts[0].intent.value == "general"
    assert planning_contexts[0].language_code == "en"
    assert planning_contexts[0].requires_table is True
    assert control.heartbeats[0][:2] == (ReportJobPhase.PLANNING, 10)
    assert control.heartbeats[-1][:2] == (ReportJobPhase.ASSEMBLING, 90)
    assert control.heartbeats[-1][2] is None
    assert all(
        earlier[1] <= later[1]
        for earlier, later in zip(control.heartbeats, control.heartbeats[1:])
    )
    empty_generation_checkpoints = [
        checkpoint
        for phase, progress, checkpoint in control.heartbeats
        if (
            phase is ReportJobPhase.GENERATING_SECTIONS
            and progress == 25
            and checkpoint is not None
            and checkpoint["completed_sections"] == []
        )
    ]
    assert len(empty_generation_checkpoints) == 1


def test_report_attempt_binds_identity_and_deadline_for_deep_calls():
    lease = _lease()
    scopes = []
    pipeline_calls = []

    result = _processor(
        pipeline_calls=pipeline_calls,
        execution_scopes=scopes,
        job_timeout_seconds=120,
    )(lease, _Control())

    ReportResult.model_validate(result)
    assert len(scopes) == 1
    scope = scopes[0]
    assert scope is not None
    assert scope.request_id == (
        f"{lease.request_id}:attempt:{lease.attempt_count}"
    )
    assert scope.actor_binding
    assert scope.deadline is not None
    assert scope.deadline.source == "report_job"
    assert 0 < scope.deadline.remaining_seconds() <= 120
    assert pipeline_calls[0][1]["request_deadline"] is scope.deadline
    # The trusted parts of the identity -- actor and deadline -- are shared with
    # the nested run. Only the request id is namespaced, so the pipeline's
    # provider claims cannot collide with the report's own.
    assert pipeline_calls[0][1]["request_id"].startswith(scope.request_id)
    assert pipeline_calls[0][1]["request_id"] != scope.request_id
    assert pipeline_calls[0][1]["actor_id"] == str(lease.actor_user_id)


class _AttemptMetrics:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.started = []
        self.finalized = 0

    def start_request_telemetry(self, trace_id):
        self.started.append(trace_id)

    def finalize_request_telemetry(self):
        self.finalized += 1
        return self.snapshot


def test_report_attempt_logs_stage_aware_budget_telemetry(
    caplog,
    monkeypatch,
):
    lease = _lease()
    metrics = _AttemptMetrics(
        {
            "trace_id": str(lease.job_id),
            "llm_calls": 2,
            "prompt_tokens": 1300,
            "completion_tokens": 400,
            "total_tokens": 1700,
            "estimated_cost_usd": 0.03,
            "models": {"gpt-5.6-luna": {"calls": 2}},
            "stages": {
                "report_research_planner": {"calls": 1},
                "report_document_writer": {"calls": 1},
            },
        }
    )
    processor = ReportJobProcessor(
        pipeline_v2_mode="shadow",
        max_generative_calls=3,
    )
    monkeypatch.setattr(report_job_processor, "metrics", metrics)
    monkeypatch.setattr(
        processor,
        "_run_bound_attempt",
        lambda _lease, _control: {"ok": True},
    )
    caplog.set_level("INFO", logger="Enai.ReportProcessor")

    assert processor(lease, _Control()) == {"ok": True}

    assert metrics.started == [str(lease.job_id)]
    assert metrics.finalized == 1
    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_JOB_ATTEMPT_TELEMETRY ")
    )
    payload = json.loads(record.split(" ", 1)[1])
    assert payload["outcome"] == "completed"
    assert payload["pipeline_v2_mode"] == "shadow"
    assert payload["llm_calls"] == 2
    assert payload["generative_call_budget"] == 3
    assert payload["over_generative_call_budget"] is False
    assert set(payload["stages"]) == {
        "report_research_planner",
        "report_document_writer",
    }


def test_shadow_research_planner_runs_before_legacy_pipeline_without_cutover(
    caplog,
):
    lease = _lease()
    pipeline_calls = []
    research_calls = []
    execution_calls = []
    consolidation_calls = []
    gate_calls = []

    def research_planner(query: str, *, max_tracks: int):
        assert pipeline_calls == []
        research_calls.append((query, max_tracks))
        return _research_plan_payload(
            query_digest=hashlib.sha256(query.encode("utf-8")).hexdigest()
        )

    def research_executor(query, plan, *, max_workers):
        execution_calls.append((query, len(plan.tracks), max_workers))
        return []

    def manifest_consolidator(query, packets):
        consolidation_calls.append((query, packets))
        return _manifest_for_query(query)

    def exhibit_builder(packets, manifest):
        assert packets == []
        assert manifest.query_digest == hashlib.sha256(
            lease.query.encode("utf-8")
        ).hexdigest()
        return []

    def gate_evaluator(plan, packets, *, chart_decisions):
        gate_calls.append((len(plan.tracks), packets, chart_decisions))
        return {
            "contract_version": "report-evidence-gate-v1",
            "query_digest": plan.query_digest,
            "status": "ready",
            "tracks": [
                {
                    "track_id": track.track_id,
                    "required": track.required,
                    "status": "complete",
                    "evidence_item_count": 1,
                    "numeric_observation_count": 0,
                    "chart_candidate_count": 0,
                    "finding_codes": [],
                }
                for track in plan.tracks
            ],
            "finding_codes": [],
        }

    caplog.set_level("INFO", logger="Enai.ReportProcessor")
    result = _processor(
        pipeline_calls=pipeline_calls,
        research_planner=research_planner,
        research_executor=research_executor,
        manifest_consolidator=manifest_consolidator,
        research_exhibit_builder=exhibit_builder,
        evidence_gate_evaluator=gate_evaluator,
        pipeline_v2_mode="shadow",
        max_research_tracks=4,
        max_research_workers=3,
    )(lease, _Control())

    assert result["contract_version"] == "report-result-v1"
    assert research_calls == [(lease.query, 4)]
    assert execution_calls == [(lease.query, 3, 3)]
    assert consolidation_calls == [(lease.query, [])]
    assert gate_calls == [(3, [], [])]
    assert len(pipeline_calls) == 1
    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_RESEARCH_PLAN_SHADOW ")
    )
    payload = json.loads(record.split(" ", 1)[1])
    assert payload["outcome"] == "valid"
    assert payload["track_count"] == 3
    assert payload["packet_count"] == 0
    assert payload["evidence_item_count"] == len(_manifest().items)
    assert payload["coverage_status"] == "ready"
    assert payload["built_chart_count"] == 0
    assert "query" not in payload


def test_shadow_research_failure_is_content_free_and_does_not_fail_legacy(
    caplog,
):
    lease = _lease()

    def unavailable_planner(*_args, **_kwargs):
        raise RuntimeError("private-query-fragment")

    caplog.set_level("INFO", logger="Enai.ReportProcessor")
    result = _processor(
        research_planner=unavailable_planner,
        pipeline_v2_mode="shadow",
    )(lease, _Control())

    assert result["contract_version"] == "report-result-v1"
    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_RESEARCH_PLAN_SHADOW ")
    )
    assert '"outcome":"failed"' in record
    assert "private-query-fragment" not in record


def test_disabled_pipeline_does_not_invoke_research_planner():
    def unexpected_planner(*_args, **_kwargs):
        raise AssertionError("disabled mode must not run v2 planning")

    result = _processor(
        research_planner=unexpected_planner,
        pipeline_v2_mode="disabled",
    )(_lease(), _Control())

    assert result["contract_version"] == "report-result-v1"


def test_enabled_v2_runs_without_legacy_analyzer_and_checkpoints_each_stage():
    """The pipeline enriches evidence; tracks still own planning and tables.

    v2 originally refused the query pipeline outright, which left it with no
    computed statistics and no curated knowledge -- the reason adaptive reports
    read weaker than the same question in standard mode. It now runs once,
    purely to contribute narrative evidence, and must still never drive
    planning, evidence collection or per-section generation.
    """

    (
        research_plan,
        packets,
        manifest,
        decisions,
        gate,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    calls = []
    consolidator_kwargs = {}

    def record(name, value):
        calls.append(name)
        return value

    def document_generator(*_args, **kwargs):
        calls.append("document_generator")
        assert kwargs["allow_repair"] is True
        assert kwargs["max_repair_attempts"] == 2
        return draft

    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: record(
            "query_pipeline",
            SimpleNamespace(
                stats_hint="Observed mean balancing price was 141.0 GEL/MWh.",
                summary_domain_knowledge="The balancing market settles hourly.",
                provenance_refs=["query:prices"],
            ),
        ),
        planner=lambda *_args, **_kwargs: pytest.fail(
            "enabled v2 must not invoke the legacy report planner"
        ),
        section_generator=lambda *_args, **_kwargs: pytest.fail(
            "enabled v2 must not invoke per-section generation"
        ),
        pipeline_v2_mode="enabled",
        research_planner=lambda *_args, **_kwargs: record(
            "research_planner",
            research_plan,
        ),
        research_executor=lambda *_args, **_kwargs: record(
            "research_executor",
            packets,
        ),
        manifest_consolidator=lambda *_args, **kwargs: (
            consolidator_kwargs.update(kwargs)
            or record("manifest_consolidator", manifest)
        ),
        research_exhibit_builder=lambda *_args, **_kwargs: record(
            "research_exhibit_builder",
            decisions,
        ),
        evidence_gate_evaluator=lambda *_args, **_kwargs: record(
            "evidence_gate_evaluator",
            gate,
        ),
        document_planner=lambda *_args, **_kwargs: record(
            "document_planner",
            document_plan,
        ),
        document_generator=document_generator,
    )
    control = _Control()

    result = ReportResultV2.model_validate(
        processor(_lease(query=_V2_QUERY), control)
    )

    assert result.contract_version == "report-result-v2"
    # The pipeline's products reach the manifest as narrative evidence.
    assert [
        item.kind.value for item in consolidator_kwargs["extra_items"]
    ] == ["statistics", "knowledge"]
    assert calls == [
        "research_planner",
        "research_executor",
        "query_pipeline",
        "manifest_consolidator",
        "research_exhibit_builder",
        "evidence_gate_evaluator",
        "document_planner",
        "document_generator",
    ]
    checkpoints = [
        heartbeat[2]
        for heartbeat in control.heartbeats
        if heartbeat[2] is not None
    ]
    assert [
        checkpoint["checkpoint_stage"] for checkpoint in checkpoints
    ] == [
        "research_plan_ready",
        "document_plan_ready",
        "draft_ready",
    ]
    assert all(
        checkpoint["contract_version"]
        == "report-generation-checkpoint-v3"
        for checkpoint in checkpoints
    )


def test_global_narrative_enrichment_rejects_blocked_pipeline_context(caplog):
    blocked = QueryContext(
        query=_V2_QUERY,
        stats_hint="Observed mean balancing price was 141.0 GEL/MWh.",
        summary_domain_knowledge="The balancing market settles hourly.",
        terminal_outcome="clarification_required",
        missing_evidence_for_metrics=["mom_percent_change"],
        answer_mode="report",
    )
    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: blocked,
    )
    caplog.set_level("INFO", logger="Enai.ReportProcessor")

    items = processor._pipeline_narrative_items(
        _lease(query=_V2_QUERY)
    )

    assert items == []
    assert "reason=missing_derived_evidence" in caplog.text


def test_track_analysis_shadow_is_parallel_isolated_and_does_not_change_output(
    caplog,
):
    (
        research_plan,
        packets,
        manifest,
        decisions,
        gate,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    packet_by_track = {packet.track_id: packet for packet in packets}
    barrier = threading.Barrier(len(research_plan.tracks))
    analysis_calls = []
    consolidated_packets = []

    def query_pipeline(query, **_kwargs):
        return _pipeline_context(query)

    def track_analyzer(report_query, track, **kwargs):
        analysis_calls.append((report_query, track.track_id, kwargs))
        barrier.wait(timeout=2)
        if track.track_id == research_plan.tracks[-1].track_id:
            raise RuntimeError("private-track-fragment")
        return packet_by_track[track.track_id]

    def consolidate(query, received_packets, **_kwargs):
        consolidated_packets.extend(received_packets)
        return manifest

    caplog.set_level("INFO", logger="Enai.ReportProcessor")
    processor = ReportJobProcessor(
        query_pipeline=query_pipeline,
        pipeline_v2_mode="enabled",
        track_analysis_mode="shadow",
        track_analyzer=track_analyzer,
        research_planner=lambda *_args, **_kwargs: research_plan,
        research_executor=lambda *_args, **_kwargs: packets,
        manifest_consolidator=consolidate,
        research_exhibit_builder=lambda *_args, **_kwargs: decisions,
        evidence_gate_evaluator=lambda *_args, **_kwargs: gate,
        document_planner=lambda *_args, **_kwargs: document_plan,
        document_generator=lambda *_args, **_kwargs: draft,
    )

    result = ReportResultV2.model_validate(
        processor(_lease(query=_V2_QUERY), _Control())
    )

    assert result.contract_version == "report-result-v2"
    assert [packet.track_id for packet in consolidated_packets] == [
        packet.track_id for packet in packets
    ]
    assert sorted(call[1] for call in analysis_calls) == sorted(
        track.track_id for track in research_plan.tracks
    )
    assert len(
        {
            call[2]["request_id"]
            for call in analysis_calls
        }
    ) == len(research_plan.tracks)
    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_TRACK_ANALYSIS_SHADOW ")
    )
    payload = json.loads(record.split(" ", 1)[1])
    assert payload["mode"] == "shadow"
    assert payload["track_count"] == len(research_plan.tracks)
    assert payload["completed_count"] == len(research_plan.tracks) - 1
    assert payload["failed_count"] == 1
    assert "private-track-fragment" not in record


def test_track_analysis_shadow_infrastructure_failure_cannot_fail_report(
    monkeypatch,
    caplog,
):
    (
        research_plan,
        packets,
        manifest,
        decisions,
        gate,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)

    def fail_executor(*_args, **_kwargs):
        raise RuntimeError("private-shadow-infrastructure-fragment")

    monkeypatch.setattr(
        report_job_processor,
        "ThreadPoolExecutor",
        fail_executor,
    )
    caplog.set_level("INFO", logger="Enai.ReportProcessor")
    processor = ReportJobProcessor(
        query_pipeline=lambda query, **_kwargs: _pipeline_context(query),
        pipeline_v2_mode="enabled",
        track_analysis_mode="shadow",
        research_planner=lambda *_args, **_kwargs: research_plan,
        research_executor=lambda *_args, **_kwargs: packets,
        manifest_consolidator=lambda *_args, **_kwargs: manifest,
        research_exhibit_builder=lambda *_args, **_kwargs: decisions,
        evidence_gate_evaluator=lambda *_args, **_kwargs: gate,
        document_planner=lambda *_args, **_kwargs: document_plan,
        document_generator=lambda *_args, **_kwargs: draft,
    )

    result = ReportResultV2.model_validate(
        processor(_lease(query=_V2_QUERY), _Control())
    )

    assert result.contract_version == "report-result-v2"
    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_TRACK_ANALYSIS_SHADOW ")
    )
    payload = json.loads(record.split(" ", 1)[1])
    assert payload["completed_count"] == 0
    assert payload["failed_count"] == len(research_plan.tracks)
    assert "private-shadow-infrastructure-fragment" not in record


def test_track_analysis_enabled_uses_track_packets_without_global_broadcast(
    caplog,
):
    (
        research_plan,
        packets,
        manifest,
        decisions,
        gate,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    packet_by_track = {packet.track_id: packet for packet in packets}
    analysis_calls = []
    consolidation = {}

    def global_pipeline(*_args, **_kwargs):
        raise AssertionError(
            "enabled track analysis must not run global enrichment"
        )

    def track_analyzer(_query, track, **_kwargs):
        analysis_calls.append(track.track_id)
        return packet_by_track[track.track_id]

    def consolidate(_query, received_packets, **kwargs):
        consolidation["packets"] = list(received_packets)
        consolidation["extra_items"] = list(kwargs["extra_items"])
        return manifest

    caplog.set_level("INFO", logger="Enai.ReportProcessor")
    processor = ReportJobProcessor(
        query_pipeline=global_pipeline,
        pipeline_v2_mode="enabled",
        track_analysis_mode="enabled",
        track_analyzer=track_analyzer,
        research_planner=lambda *_args, **_kwargs: research_plan,
        research_executor=lambda *_args, **_kwargs: packets,
        manifest_consolidator=consolidate,
        research_exhibit_builder=lambda *_args, **_kwargs: decisions,
        evidence_gate_evaluator=lambda *_args, **_kwargs: gate,
        document_planner=lambda *_args, **_kwargs: document_plan,
        document_generator=lambda *_args, **_kwargs: draft,
    )

    result = ReportResultV2.model_validate(
        processor(_lease(query=_V2_QUERY), _Control())
    )

    assert result.contract_version == "report-result-v2"
    assert sorted(analysis_calls) == sorted(packet_by_track)
    assert consolidation["extra_items"] == []
    assert [packet.track_id for packet in consolidation["packets"]] == [
        track.track_id for track in research_plan.tracks
    ]
    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_TRACK_ANALYSIS_ENABLED ")
    )
    payload = json.loads(record.split(" ", 1)[1])
    assert payload["completed_count"] == len(research_plan.tracks)
    assert payload["failed_count"] == 0


def test_track_analysis_enabled_falls_back_per_track_after_pipeline_failure(
    caplog,
):
    (
        research_plan,
        packets,
        manifest,
        decisions,
        gate,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    packet_by_track = {packet.track_id: packet for packet in packets}
    failed_track_id = research_plan.tracks[-1].track_id
    consolidated_packets = []

    def track_analyzer(_query, track, **_kwargs):
        if track.track_id == failed_track_id:
            raise RuntimeError("private-enabled-track-fragment")
        return packet_by_track[track.track_id]

    def consolidate(_query, received_packets, **_kwargs):
        consolidated_packets.extend(received_packets)
        return manifest

    caplog.set_level("INFO", logger="Enai.ReportProcessor")
    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: pytest.fail(
            "enabled track analysis must not run global enrichment"
        ),
        pipeline_v2_mode="enabled",
        track_analysis_mode="enabled",
        track_analyzer=track_analyzer,
        research_planner=lambda *_args, **_kwargs: research_plan,
        research_executor=lambda *_args, **_kwargs: packets,
        manifest_consolidator=consolidate,
        research_exhibit_builder=lambda *_args, **_kwargs: decisions,
        evidence_gate_evaluator=lambda *_args, **_kwargs: gate,
        document_planner=lambda *_args, **_kwargs: document_plan,
        document_generator=lambda *_args, **_kwargs: draft,
    )

    result = ReportResultV2.model_validate(
        processor(_lease(query=_V2_QUERY), _Control())
    )

    assert result.contract_version == "report-result-v2"
    selected_by_track = {
        packet.track_id: packet for packet in consolidated_packets
    }
    assert set(selected_by_track) == set(packet_by_track)
    assert (
        selected_by_track[failed_track_id].model_dump(mode="json")
        == packet_by_track[failed_track_id].model_dump(mode="json")
    )
    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_TRACK_ANALYSIS_ENABLED ")
    )
    payload = json.loads(record.split(" ", 1)[1])
    assert payload["failed_count"] == 1
    assert payload["fallback_count"] == 1
    assert "private-enabled-track-fragment" not in record


def test_enabled_v2_resumes_document_plan_without_research_calls():
    (
        research_plan,
        _,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    checkpoint = ReportGenerationCheckpoint(
        contract_version="report-generation-checkpoint-v3",
        checkpoint_stage="document_plan_ready",
        research_plan=research_plan,
        manifest=manifest,
        document_plan=document_plan,
    )
    generator_calls = []
    processor = ReportJobProcessor(
        pipeline_v2_mode="enabled",
        research_planner=lambda *_args, **_kwargs: pytest.fail(
            "resume must not re-plan research"
        ),
        research_executor=lambda *_args, **_kwargs: pytest.fail(
            "resume must not repeat evidence collection"
        ),
        document_planner=lambda *_args, **_kwargs: pytest.fail(
            "resume must not repeat document planning"
        ),
        document_generator=lambda *_args, **kwargs: (
            generator_calls.append(kwargs["allow_repair"]) or draft
        ),
    )

    result = ReportResultV2.model_validate(
        processor(
            _lease(
                query=_V2_QUERY,
                checkpoint=checkpoint.model_dump(mode="json"),
                phase="generating_sections",
                progress_percent=55,
            ),
            _Control(),
        )
    )

    assert result.contract_version == "report-result-v2"
    assert generator_calls == [True]


def test_enabled_v2_resumes_validated_draft_without_another_model_call():
    (
        research_plan,
        _,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    checkpoint = ReportGenerationCheckpoint(
        contract_version="report-generation-checkpoint-v3",
        checkpoint_stage="draft_ready",
        research_plan=research_plan,
        manifest=manifest,
        document_plan=document_plan,
        document_draft=draft,
    )
    processor = ReportJobProcessor(
        pipeline_v2_mode="enabled",
        research_planner=lambda *_args, **_kwargs: pytest.fail(
            "draft resume must not invoke a model"
        ),
        document_generator=lambda *_args, **_kwargs: pytest.fail(
            "draft resume must not invoke a model"
        ),
    )

    result = ReportResultV2.model_validate(
        processor(
            _lease(
                query=_V2_QUERY,
                checkpoint=checkpoint.model_dump(mode="json"),
                phase="assembling",
                progress_percent=90,
            ),
            _Control(),
        )
    )

    assert result.contract_version == "report-result-v2"


def test_enabled_v2_two_call_budget_disables_document_repair():
    (
        research_plan,
        packets,
        manifest,
        decisions,
        gate,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    repair_flags = []
    processor = ReportJobProcessor(
        # Inert pipeline: narrative enrichment is not this test's subject,
        # and without an injection the processor would run the real one.
        query_pipeline=lambda *_args, **_kwargs: None,
        pipeline_v2_mode="enabled",
        max_generative_calls=2,
        research_planner=lambda *_args, **_kwargs: research_plan,
        research_executor=lambda *_args, **_kwargs: packets,
        manifest_consolidator=lambda *_args, **_kwargs: manifest,
        research_exhibit_builder=lambda *_args, **_kwargs: decisions,
        evidence_gate_evaluator=lambda *_args, **_kwargs: gate,
        document_planner=lambda *_args, **_kwargs: document_plan,
        document_generator=lambda *_args, **kwargs: (
            repair_flags.append(kwargs["allow_repair"]) or draft
        ),
    )

    ReportResultV2.model_validate(
        processor(_lease(query=_V2_QUERY), _Control())
    )

    assert repair_flags == [False]


def test_document_repair_budget_depends_on_document_profile():
    assert (
        report_job_processor._report_document_allows_repair(
            profile="compact",
            generative_calls_used=1,
            maximum_calls=3,
        )
        is True
    )
    assert (
        report_job_processor._report_document_allows_repair(
            profile="focused",
            generative_calls_used=1,
            maximum_calls=3,
        )
        is False
    )
    assert (
        report_job_processor._report_document_allows_repair(
            profile="full",
            generative_calls_used=1,
            maximum_calls=4,
        )
        is True
    )


def test_document_repair_budget_reserves_generation_and_uses_spare_calls():
    assert report_job_processor._report_document_repair_budget(
        profile="compact",
        generative_calls_used=1,
        maximum_calls=4,
    ) == 2
    assert report_job_processor._report_document_repair_budget(
        profile="full",
        generative_calls_used=1,
        maximum_calls=5,
    ) == 2
    assert report_job_processor._report_document_repair_budget(
        profile="full",
        generative_calls_used=1,
        maximum_calls=3,
    ) == 0


def test_enabled_v2_logs_safe_research_plan_schema_diagnostics(caplog):
    private_value = "private-invalid-collector"

    def invalid_planner(query, *, max_tracks):
        payload = _research_plan_payload(
            query_digest=hashlib.sha256(
                query.encode("utf-8")
            ).hexdigest()
        )
        payload["tracks"][0]["collector_ids"] = [private_value]

        def invalid_invoker(*_args, **_kwargs):
            from contracts.report_research import ReportResearchPlan

            return ReportResearchPlan.model_validate(payload)

        return plan_report_research(
            query,
            max_tracks=max_tracks,
            invoke_model=invalid_invoker,
        )

    processor = ReportJobProcessor(
        pipeline_v2_mode="enabled",
        research_planner=invalid_planner,
    )

    with caplog.at_level(logging.WARNING, logger="Enai.ReportProcessor"):
        with pytest.raises(ReportJobFailure) as exc_info:
            processor(_lease(query=_V2_QUERY), _Control())

    assert exc_info.value.error_code == "REPORT_PLAN_INVALID"
    assert exc_info.value.retryable is True
    assert "finding_codes=PLAN_SCHEMA_INVALID" in caplog.text
    assert (
        "schema_error_codes="
        "SCHEMA_TRACKS_ITEM_COLLECTOR_IDS_ITEM_ENUM"
        in caplog.text
    )
    assert private_value not in caplog.text


def test_enabled_v2_does_not_retry_ambiguous_research_planner_delivery():
    processor = ReportJobProcessor(
        pipeline_v2_mode="enabled",
        research_planner=lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(
            ProviderExecutionError(
                "provider response could not be reconciled",
                provider="openai",
                stage="report_research_planner",
                disposition=ProviderDeliveryDisposition.AMBIGUOUS,
            )
        ),
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(_lease(query=_V2_QUERY), _Control())

    assert exc_info.value.error_code == "REPORT_PLAN_PROVIDER_FAILED"
    assert exc_info.value.retryable is False


def test_enabled_v2_document_plan_validation_failure_is_typed_as_plan_invalid():
    (
        research_plan,
        packets,
        manifest,
        decisions,
        gate,
        _,
    ) = _document_components()
    processor = ReportJobProcessor(
        # Inert pipeline: narrative enrichment is not this test's subject,
        # and without an injection the processor would run the real one.
        query_pipeline=lambda *_args, **_kwargs: None,
        pipeline_v2_mode="enabled",
        research_planner=lambda *_args, **_kwargs: research_plan,
        research_executor=lambda *_args, **_kwargs: packets,
        manifest_consolidator=lambda *_args, **_kwargs: manifest,
        research_exhibit_builder=lambda *_args, **_kwargs: decisions,
        evidence_gate_evaluator=lambda *_args, **_kwargs: gate,
        document_planner=lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(ValueError("The document plan is invalid.")),
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(_lease(query=_V2_QUERY), _Control())

    assert exc_info.value.error_code == "REPORT_PLAN_INVALID"
    assert exc_info.value.retryable is False


def test_report_attempt_finalizes_telemetry_when_execution_fails(monkeypatch):
    lease = _lease()
    metrics = _AttemptMetrics(
        {
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
            "estimated_cost_usd": 0.0,
            "models": {},
            "stages": {},
        }
    )
    processor = ReportJobProcessor()
    monkeypatch.setattr(report_job_processor, "metrics", metrics)
    monkeypatch.setattr(
        processor,
        "_run_bound_attempt",
        lambda _lease, _control: (_ for _ in ()).throw(
            ValueError("synthetic failure")
        ),
    )

    with pytest.raises(ValueError, match="synthetic failure"):
        processor(lease, _Control())

    assert metrics.finalized == 1


def test_report_deadline_exhaustion_has_a_typed_retry_policy():
    lease = _lease()
    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RequestDeadlineExceeded("report_pipeline")
        ),
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(lease, _Control())

    assert exc_info.value.error_code == "REPORT_DEADLINE_EXCEEDED"
    assert exc_info.value.retryable is True


def test_chart_decisions_are_built_once_and_shared_with_evaluation(
    monkeypatch,
):
    chart_calls = []
    monkeypatch.setattr(
        "agent.report_evaluation.build_report_charts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "evaluation must reuse the processor's chart decisions"
            )
        ),
    )

    def chart_builder(plan, manifest):
        chart_calls.append((plan.evidence_manifest_id, manifest.manifest_id))
        return build_report_charts(plan, manifest)

    def evaluator(plan, manifest, *, chart_decisions=None):
        assert chart_decisions is not None
        return evaluate_report_plan(
            plan,
            manifest,
            chart_decisions=chart_decisions,
        )

    lease = _lease()
    processor = _processor(
        evaluator=evaluator,
        chart_builder=chart_builder,
    )

    ReportResult.model_validate(processor(lease, _Control()))

    assert chart_calls == [
        (_manifest().manifest_id, _manifest().manifest_id),
    ]


def test_quantitative_report_without_table_evidence_is_not_retried():
    lease = _lease()
    control = _Control()
    context = QueryContext(
        query=lease.query,
        summary_domain_knowledge="Prices are formed under the applicable market rules.",
        answer_mode="report",
    )
    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: context,
        planner=lambda *_args, **_kwargs: pytest.fail(
            "planner must not publish a quantitative report without table evidence"
        ),
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(lease, control)

    assert exc_info.value.error_code == "REPORT_EVIDENCE_UNAVAILABLE"
    assert exc_info.value.retryable is False


def test_retry_resumes_valid_sections_without_repeating_pipeline_or_planner():
    plan = ReportPlan.model_validate(_plan_payload())
    resumed = ReportSectionDraft.model_validate(_draft(plan.sections[0]))
    manifest_payload = _manifest().model_dump(mode="json")
    manifest_payload["query_digest"] = hashlib.sha256(
        b"Explain the price trend."
    ).hexdigest()
    checkpoint = ReportGenerationCheckpoint.model_validate(
        {
            "contract_version": "report-generation-checkpoint-v1",
            "manifest": manifest_payload,
            "plan": {
                **_plan_payload(),
                "evidence_manifest_id": manifest_payload["manifest_id"],
            },
            "completed_sections": [resumed.model_dump(mode="json")],
        }
    )
    pipeline_calls: list = []
    generated: list = []
    lease = _lease(
        checkpoint=checkpoint.model_dump(mode="json"),
        phase="generating_sections",
        progress_percent=37,
    )
    control = _Control()
    processor = _processor(
        pipeline_calls=pipeline_calls,
        generated=generated,
    )

    result = ReportResult.model_validate(processor(lease, control))

    assert result.sections[0].section_id == resumed.section_id
    assert pipeline_calls == []
    assert resumed.section_id not in generated
    assert control.heartbeats[0][1] >= lease.progress_percent
    assert control.heartbeats[0][2] is None


def test_processor_reuses_one_grounding_index_for_generation_and_assembly(
    monkeypatch,
):
    calls = Counter()
    original = report_grounding._evidence_grounding_facts

    def counted(item):
        calls[item.evidence_ref] += 1
        return original(item)

    monkeypatch.setattr(
        report_grounding,
        "_evidence_grounding_facts",
        counted,
    )

    _processor()(_lease(), _Control())

    assert calls == Counter(
        {item.evidence_ref: 1 for item in _manifest().items}
    )


def test_checkpoint_for_a_different_query_fails_closed():
    plan = ReportPlan.model_validate(_plan_payload())
    checkpoint = ReportGenerationCheckpoint(
        contract_version="report-generation-checkpoint-v1",
        manifest=_manifest(),
        plan=plan,
        completed_sections=[],
    )
    lease = _lease(
        checkpoint=checkpoint.model_dump(mode="json"),
        phase="generating_sections",
        progress_percent=25,
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        _processor()(lease, _Control())

    assert exc_info.value.error_code == "REPORT_CHECKPOINT_INVALID"
    assert exc_info.value.retryable is False


def test_irreparable_evidence_bound_plan_is_not_retried():
    lease = _lease()

    def pipeline(query, **_kwargs):
        return _pipeline_context(query)

    def invalid_planner(*_args, **_kwargs):
        raise ReportPlanEvidenceError("Bounded planner validation failed.")

    processor = ReportJobProcessor(
        query_pipeline=pipeline,
        evidence_builder=lambda ctx: _manifest_for_query(ctx.query),
        planner=invalid_planner,
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(lease, _Control())

    assert exc_info.value.error_code == "REPORT_PLAN_INVALID"
    assert exc_info.value.retryable is False


def test_schema_invalid_report_plan_is_not_retried_as_a_whole_job():
    lease = _lease()
    processor = ReportJobProcessor(
        query_pipeline=lambda query, **_kwargs: _pipeline_context(query),
        evidence_builder=lambda ctx: _manifest_for_query(ctx.query),
        planner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("The model returned an invalid plan.")
        ),
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(lease, _Control())

    assert exc_info.value.error_code == "REPORT_PLAN_INVALID"
    assert exc_info.value.retryable is False


def test_report_plan_cannot_override_processor_planning_semantics():
    lease = _lease()
    mismatched_payload = _plan_payload()
    mismatched_payload["language_code"] = "ka"
    processor = ReportJobProcessor(
        query_pipeline=lambda query, **_kwargs: _pipeline_context(query),
        evidence_builder=lambda ctx: _manifest_for_query(ctx.query),
        planner=lambda *_args, **_kwargs: ReportPlan.model_validate(
            mismatched_payload
        ),
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(lease, _Control())

    assert exc_info.value.error_code == "REPORT_PLAN_INVALID"
    assert exc_info.value.retryable is False


def test_report_plan_provider_failure_respects_delivery_and_is_diagnosable(
    caplog,
):
    lease = _lease()

    def unavailable_planner(*_args, **_kwargs):
        raise ProviderExecutionError(
            "provider secret must not be logged",
            provider="nvidia",
            stage="report_planner",
            disposition=ProviderDeliveryDisposition.TIMED_OUT,
        )

    processor = ReportJobProcessor(
        query_pipeline=lambda query, **_kwargs: _pipeline_context(query),
        evidence_builder=lambda ctx: _manifest_for_query(ctx.query),
        planner=unavailable_planner,
    )

    with caplog.at_level(logging.WARNING, logger="Enai.ReportProcessor"):
        with pytest.raises(ReportJobFailure) as exc_info:
            processor(lease, _Control())

    assert exc_info.value.error_code == "REPORT_PLAN_PROVIDER_FAILED"
    assert exc_info.value.retryable is False
    assert f"job_id={lease.job_id}" in caplog.text
    assert f"job_attempt={lease.attempt_count}" in caplog.text
    assert "provider=nvidia" in caplog.text
    assert "provider_stage=report_planner" in caplog.text
    assert "provider_disposition=timed_out" in caplog.text
    assert "provider secret" not in caplog.text


def test_planner_failure_persists_evidence_ready_checkpoint_for_retry():
    lease = _lease()
    control = _Control()

    processor = ReportJobProcessor(
        query_pipeline=lambda query, **_kwargs: _pipeline_context(query),
        evidence_builder=lambda ctx: _manifest_for_query(ctx.query),
        planner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProviderExecutionError(
                "provider unavailable",
                provider="nvidia",
                stage="report_planner",
                disposition=ProviderDeliveryDisposition.TIMED_OUT,
            )
        ),
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(lease, control)

    assert exc_info.value.error_code == "REPORT_PLAN_PROVIDER_FAILED"
    checkpoint = control.heartbeats[-1][2]
    assert checkpoint is not None
    assert checkpoint["contract_version"] == (
        "report-generation-checkpoint-v2"
    )
    assert checkpoint["checkpoint_stage"] == "evidence_ready"
    assert checkpoint["plan"] is None


def test_evidence_ready_retry_resumes_planning_without_pipeline():
    query = "Explain the price trend."
    planning_context = build_report_planning_context(
        _pipeline_context(query)
    )
    checkpoint = ReportGenerationCheckpoint(
        contract_version="report-generation-checkpoint-v2",
        checkpoint_stage="evidence_ready",
        manifest=_manifest_for_query(query),
        planning_context=planning_context,
        plan=None,
        completed_sections=[],
    )
    pipeline_calls = []
    received_contexts = []

    result = _processor(
        pipeline_calls=pipeline_calls,
        planning_contexts=received_contexts,
    )(
        _lease(
            checkpoint=checkpoint.model_dump(mode="json"),
            phase="planning",
            progress_percent=20,
        ),
        _Control(),
    )

    ReportResult.model_validate(result)
    assert pipeline_calls == []
    assert received_contexts == [planning_context]


@pytest.mark.parametrize(
    ("section_error_codes", "expected_code", "expected_retryable"),
    [
        (["UNGROUNDED_NUMERIC_CLAIM"], "REPORT_SECTION_INVALID", False),
        (
            ["SECTION_WRITE_PROVIDER_FAILED"],
            "REPORT_SECTION_PROVIDER_FAILED",
            True,
        ),
        (
            ["SECTION_REPAIR_PROVIDER_FAILED"],
            "REPORT_SECTION_PROVIDER_FAILED",
            True,
        ),
    ],
)
def test_section_failure_retryability_distinguishes_validation_from_provider(
    section_error_codes,
    expected_code,
    expected_retryable,
):
    lease = _lease()

    def fail_sections(*_args, **_kwargs):
        raise ReportSectionGenerationError(
            "key_findings",
            section_error_codes,
        )

    processor = ReportJobProcessor(
        query_pipeline=lambda query, **_kwargs: _pipeline_context(query),
        evidence_builder=lambda ctx: _manifest_for_query(ctx.query),
        planner=lambda *_args, **_kwargs: ReportPlan.model_validate(_plan_payload()),
        section_generator=fail_sections,
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(lease, _Control())

    assert exc_info.value.error_code == expected_code
    assert exc_info.value.retryable is expected_retryable


def test_unbuildable_required_chart_is_demoted_instead_of_killing_the_job():
    def planner_with_unbuildable_required_chart(_query, _manifest_value, **_kwargs):
        payload = _plan_payload()
        payload["charts"][0]["purpose"] = "relationship"
        return ReportPlan.model_validate(payload)

    processor = ReportJobProcessor(
        query_pipeline=lambda query, **_kwargs: _pipeline_context(query),
        evidence_builder=lambda ctx: _manifest_for_query(ctx.query),
        planner=planner_with_unbuildable_required_chart,
        section_generator=lambda query, plan, manifest, **kwargs: (
            generate_report_sections(
                query,
                plan,
                manifest,
                existing_drafts=kwargs["existing_drafts"],
                generate_section=lambda _q, _p, section, _m: _draft(section),
                progress_callback=kwargs["progress_callback"],
                max_workers=kwargs["max_workers"],
            )
        ),
        max_section_workers=5,
    )

    result = processor(_lease(), _Control())

    assert result["charts"] == []
    assert result["omitted_charts"] == [
        {
            "chart_id": "price_trend",
            "title": "Observed electricity price",
            "reason_code": "REPORT_CHART_EXPLICIT_AXES_REQUIRED",
        }
    ]
    assert [section["chart_refs"] for section in result["sections"]] == [
        [] for _ in result["sections"]
    ]


def test_resume_demotes_a_required_chart_left_by_an_older_checkpoint():
    payload = _plan_payload()
    payload["charts"][0]["purpose"] = "relationship"
    plan = ReportPlan.model_validate(payload)
    manifest = _manifest_for_query("Explain the price trend.")
    checkpoint = ReportGenerationCheckpoint(
        contract_version="report-generation-checkpoint-v1",
        manifest=manifest,
        plan=plan,
        completed_sections=[],
    ).model_dump(mode="json")

    processor = ReportJobProcessor(
        query_pipeline=lambda query, **_kwargs: _pipeline_context(query),
        evidence_builder=lambda ctx: _manifest_for_query(ctx.query),
        planner=lambda *_a, **_k: pytest.fail("resume must not re-plan"),
        section_generator=lambda query, plan_value, manifest_value, **kwargs: (
            generate_report_sections(
                query,
                plan_value,
                manifest_value,
                existing_drafts=kwargs["existing_drafts"],
                generate_section=lambda _q, _p, section, _m: _draft(section),
                progress_callback=kwargs["progress_callback"],
                max_workers=kwargs["max_workers"],
            )
        ),
        max_section_workers=5,
    )

    control = _Control()
    result = processor(
        _lease(checkpoint=checkpoint, phase="generating_sections"),
        control,
    )

    assert result["charts"] == []
    assert control.heartbeats[0][2] is not None
    assert [omission["chart_id"] for omission in result["omitted_charts"]] == [
        "price_trend"
    ]


def test_oversized_checkpoint_is_reported_as_its_own_failure(monkeypatch):
    from contracts.report_generation import ReportCheckpointTooLargeError

    def _oversized(*_args, **_kwargs):
        raise ReportCheckpointTooLargeError(
            "Report generation checkpoint exceeds 1 MiB."
        )

    monkeypatch.setattr(
        ReportJobProcessor,
        "_checkpoint_payload",
        staticmethod(_oversized),
    )

    with pytest.raises(ReportJobFailure) as excinfo:
        _processor()(_lease(), _Control())

    assert excinfo.value.error_code == "REPORT_CHECKPOINT_TOO_LARGE"
    assert excinfo.value.retryable is False


def test_projected_checkpoint_overflow_fails_before_section_calls(
    monkeypatch,
):
    generated = []
    monkeypatch.setattr(
        ReportJobProcessor,
        "_projected_final_checkpoint_size_bytes",
        staticmethod(
            lambda *_args, **_kwargs: (
                REPORT_GENERATION_CHECKPOINT_MAX_BYTES + 1
            )
        ),
        raising=False,
    )

    with pytest.raises(ReportJobFailure) as excinfo:
        _processor(generated=generated)(_lease(), _Control())

    assert excinfo.value.error_code == "REPORT_CHECKPOINT_TOO_LARGE"
    assert generated == []


def test_invalid_checkpoint_identity_is_not_reported_as_oversized(monkeypatch):
    def _identity_failure(*_args, **_kwargs):
        raise ValueError("Report checkpoint plan and manifest identity must match.")

    monkeypatch.setattr(
        ReportJobProcessor,
        "_checkpoint_payload",
        staticmethod(_identity_failure),
    )

    with pytest.raises(ReportJobFailure) as excinfo:
        _processor()(_lease(), _Control())

    assert excinfo.value.error_code == "REPORT_CHECKPOINT_INVALID"
    assert excinfo.value.retryable is False


def test_generative_budget_counts_generation_stages_not_enrichment():
    """report_question_analyzer is enrichment, not a budgeted generation call.

    Narrative enrichment runs the query pipeline, whose analyzer reports under a
    report-prefixed stage name. Counting it made an in-budget report look over
    budget (job 22237205: 5 calls against a budget of 4).
    """

    assert report_job_processor._is_report_generation_stage(
        "report_analysis_writer"
    )
    assert report_job_processor._is_report_generation_stage(
        "report_document_repair"
    )
    assert report_job_processor._is_report_generation_stage(
        "report_section_prices_attempt_1"
    )
    assert not report_job_processor._is_report_generation_stage(
        "report_question_analyzer"
    )


# ---------------------------------------------------------------------------
# Narrative enrichment provider identity
# ---------------------------------------------------------------------------


def test_narrative_pipeline_runs_under_a_distinct_request_identity():
    """The nested query pipeline must not share the report's provider claims.

    Provider attempts are claimed once per (actor, request_id, provider,
    stage). The report claims gemini|query_embedding for its own retrieval, so
    reusing its request identity got the nested pipeline's vector-knowledge
    stage refused before it could send -- reported as ProviderExecutionError
    with no HTTP call, and mistaken for a provider outage.
    """

    lease = _lease()
    pipeline_calls: list = []
    scopes: list = []

    result = _processor(
        pipeline_calls=pipeline_calls,
        execution_scopes=scopes,
        job_timeout_seconds=120,
    )(lease, _Control())

    ReportResult.model_validate(result)
    outer_request_id = scopes[0].request_id
    nested_request_id = pipeline_calls[0][1]["request_id"]
    assert outer_request_id
    assert nested_request_id != outer_request_id
    assert nested_request_id.startswith(outer_request_id)
    assert nested_request_id.endswith(":narrative")


def test_report_and_narrative_identities_can_each_claim_one_embedding():
    """Two distinct request identities each get their own single claim.

    This is the mechanism the fix relies on, asserted directly: the same
    provider and stage collides inside one identity and does not collide across
    the report's identity and the derived narrative one. The no-replay
    guarantee still holds within each identity.
    """

    from core.report_job_processor import _NARRATIVE_REQUEST_ID_SUFFIX
    from utils.provider_attempts import (
        ProviderDeliveryDisposition,
        ProviderExecutionError,
        claim_provider_attempt,
        finish_provider_attempt,
        reset_provider_attempts_for_tests,
    )
    from utils.request_deadline import bind_request_execution_scope

    actor_id = str(uuid4())
    report_request_id = "report:req-collision:attempt:1"

    reset_provider_attempts_for_tests()
    try:
        with bind_request_execution_scope(
            deadline=None, request_id=report_request_id, actor_id=actor_id
        ):
            token = claim_provider_attempt("gemini", "query_embedding")
            finish_provider_attempt(
                token, ProviderDeliveryDisposition.COMPLETED
            )
            # Same identity, same provider and stage: refused, as designed.
            with pytest.raises(ProviderExecutionError):
                claim_provider_attempt("gemini", "query_embedding")

        with bind_request_execution_scope(
            deadline=None,
            request_id=f"{report_request_id}{_NARRATIVE_REQUEST_ID_SUFFIX}",
            actor_id=actor_id,
        ):
            nested = claim_provider_attempt("gemini", "query_embedding")
            assert nested.bound
            finish_provider_attempt(
                nested, ProviderDeliveryDisposition.COMPLETED
            )
    finally:
        reset_provider_attempts_for_tests()
