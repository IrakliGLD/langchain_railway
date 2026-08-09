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
from agent.report_intent import build_report_planning_context
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
from core.report_job_processor import (
    ReportJobProcessor,
    _diagnostic_error_locations,
)
from core.report_job_worker import ReportJobFailure
from models import QueryContext
from tests.fixtures_report_manifest import _manifest, _plan_payload
from tests.test_report_document_pipeline_v2 import (
    _QUERY as _V2_QUERY,
)
from tests.test_report_document_pipeline_v2 import (
    _document_components,
    _valid_document_draft,
)
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


def _v2_processor(
    *,
    pipeline_calls: list | None = None,
    execution_scopes: list | None = None,
    job_timeout_seconds: int | None = None,
    **overrides,
):
    """A document-pipeline processor with the legacy factory's observability.

    ``_processor`` above builds the retired path. These hooks — the nested
    pipeline's call kwargs and the execution scope bound around an attempt —
    guard behaviour that is shared, not legacy, so they follow the tests that
    need them onto the surviving path rather than being deleted with it.
    """

    (
        research_plan,
        packets,
        manifest,
        decisions,
        gate,
        document_plan,
    ) = _document_components()
    calls = pipeline_calls if pipeline_calls is not None else []

    def pipeline(query, **kwargs):
        calls.append((query, kwargs))
        if execution_scopes is not None:
            execution_scopes.append(current_request_execution_scope())
        return _pipeline_context(query)

    arguments = {
        "query_pipeline": pipeline,
        # Deliberately disabled: this is the mode that still runs the nested
        # pipeline for report-wide narrative evidence, which is what the
        # identity and deadline assertions observe.
        "track_analysis_mode": "disabled",
        "research_planner": lambda *_args, **_kwargs: research_plan,
        "research_executor": lambda *_args, **_kwargs: packets,
        "manifest_consolidator": lambda *_args, **_kwargs: manifest,
        "research_exhibit_builder": lambda *_args, **_kwargs: decisions,
        "evidence_gate_evaluator": lambda *_args, **_kwargs: gate,
        "document_planner": lambda *_args, **_kwargs: document_plan,
        "document_generator": lambda *_args, **_kwargs: (
            _valid_document_draft(document_plan, manifest)
        ),
    }
    if job_timeout_seconds is not None:
        arguments["job_timeout_seconds"] = job_timeout_seconds
    arguments.update(overrides)
    return ReportJobProcessor(**arguments)


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


def test_a_pre_manifest_checkpoint_is_rejected_cleanly():
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
        _v2_processor()(lease, _Control())

    assert exc_info.value.error_code == "REPORT_CHECKPOINT_INVALID"
    assert exc_info.value.retryable is False


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


def test_report_attempt_binds_identity_and_deadline_for_deep_calls():
    lease = _lease(query=_V2_QUERY)
    scopes = []
    pipeline_calls = []

    result = _v2_processor(
        pipeline_calls=pipeline_calls,
        execution_scopes=scopes,
        job_timeout_seconds=120,
    )(lease, _Control())

    ReportResultV2.model_validate(result)
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
    assert payload["llm_calls"] == 2
    assert payload["generative_call_budget"] == 3
    assert payload["over_generative_call_budget"] is False
    assert set(payload["stages"]) == {
        "report_research_planner",
        "report_document_writer",
    }


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
        track_analysis_mode="disabled",
        query_pipeline=lambda *_args, **_kwargs: record(
            "query_pipeline",
            SimpleNamespace(
                stats_hint="Observed mean balancing price was 141.0 GEL/MWh.",
                summary_domain_knowledge="The balancing market settles hourly.",
                provenance_refs=["query:prices"],
            ),
        ),
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


def _blocked_pipeline_context() -> QueryContext:
    return QueryContext(
        query=_V2_QUERY,
        stats_hint="Observed mean balancing price was 141.0 GEL/MWh.",
        summary_domain_knowledge="The balancing market settles hourly.",
        terminal_outcome="clarification_required",
        missing_evidence_for_metrics=["mom_percent_change"],
        answer_mode="report",
    )


def test_global_narrative_enrichment_rejects_blocked_pipeline_context(
    caplog,
    monkeypatch,
):
    from agent import report_evidence

    monkeypatch.setattr(
        report_evidence,
        "ENABLE_REPORT_PARTIAL_TRACK_EVIDENCE",
        False,
    )
    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: _blocked_pipeline_context(),
    )
    caplog.set_level("INFO", logger="Enai.ReportProcessor")

    items = processor._pipeline_narrative_items(
        _lease(query=_V2_QUERY)
    )

    assert items == []
    assert "reason=missing_derived_evidence" in caplog.text


def test_a_clarified_context_stays_blocked_under_partial_track_evidence(
    caplog,
    monkeypatch,
):
    """The flag reclassifies the reason, never the rejection.

    An underived metric stops being a blocking reason, but a context that
    clarified produced no answer at all and remains unusable.
    """
    from agent import report_evidence

    monkeypatch.setattr(
        report_evidence,
        "ENABLE_REPORT_PARTIAL_TRACK_EVIDENCE",
        True,
    )
    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: _blocked_pipeline_context(),
    )
    caplog.set_level("INFO", logger="Enai.ReportProcessor")

    items = processor._pipeline_narrative_items(
        _lease(query=_V2_QUERY)
    )

    assert items == []
    assert "reason=terminal_clarification_required" in caplog.text


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
    # market_model is a planner-declared knowledge track: its deterministic
    # packet is the right evidence, so the nested pipeline is not run for it.
    assert sorted(analysis_calls) == sorted(
        track.track_id
        for track in research_plan.tracks
        if track.analysis_preferred_path.value != "knowledge"
    )
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
    assert payload["completed_count"] == len(research_plan.tracks) - 1
    assert payload["planner_knowledge_track_ids"] == ["market_model"]
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
    # tracks[-1] is market_model, a planner-declared knowledge track that is
    # deliberately not analysed; this test needs one that is.
    failed_track_id = research_plan.tracks[0].track_id
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
    # Which track degraded, not just how many: a report running on fallback
    # evidence for two of four tracks is otherwise indistinguishable from a
    # healthy one at the document gate (job cf47a2f6).
    assert payload["failed_tracks"] == [
        {
            "track_id": failed_track_id,
            "error_type": "RuntimeError",
            "reason": "unknown",
            "invalid_fields": [],
        }
    ]


def test_track_analysis_failure_reports_the_block_reason(caplog):
    """An unusable pipeline context must say what made it unusable."""

    from agent.report_research_execution import ReportTrackAnalysisUnusable

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
    # tracks[-1] is market_model, a planner-declared knowledge track that is
    # deliberately not analysed; this test needs one that is.
    failed_track_id = research_plan.tracks[0].track_id

    def track_analyzer(_query, track, **_kwargs):
        if track.track_id == failed_track_id:
            raise ReportTrackAnalysisUnusable(
                track.track_id,
                "missing_derived_evidence",
            )
        return packet_by_track[track.track_id]

    caplog.set_level("INFO", logger="Enai.ReportProcessor")
    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: pytest.fail(
            "enabled track analysis must not run global enrichment"
        ),
        track_analysis_mode="enabled",
        track_analyzer=track_analyzer,
        research_planner=lambda *_args, **_kwargs: research_plan,
        research_executor=lambda *_args, **_kwargs: packets,
        manifest_consolidator=lambda *_args, **_kwargs: manifest,
        research_exhibit_builder=lambda *_args, **_kwargs: decisions,
        evidence_gate_evaluator=lambda *_args, **_kwargs: gate,
        document_planner=lambda *_args, **_kwargs: document_plan,
        document_generator=lambda *_args, **_kwargs: draft,
    )

    processor(_lease(query=_V2_QUERY), _Control())

    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_TRACK_ANALYSIS_ENABLED ")
    )
    payload = json.loads(record.split(" ", 1)[1])
    assert payload["failed_tracks"] == [
        {
            "track_id": failed_track_id,
            "error_type": "ReportTrackAnalysisUnusable",
            "reason": "missing_derived_evidence",
            "invalid_fields": [],
        }
    ]


def test_a_rejected_contract_names_the_field_that_rejected_the_track():
    """"ValidationError, reason=unknown" leaves the cause to be guessed.

    Job 5e6b0cf3 discarded supply_mix_and_flows that way, and the packet
    contract that rejected it had to be inferred from the report's prose.
    """
    from pydantic import ValidationError

    from contracts.report_research import ReportEvidencePacket

    try:
        ReportEvidencePacket(
            contract_version="report-evidence-packet-v1",
            track_id="prices",
            status="complete",
            gaps=["duplicate", "duplicate"],
        )
    except ValidationError as exc:
        located = _diagnostic_error_locations(exc)
    else:  # pragma: no cover - the contract must reject duplicates
        raise AssertionError("duplicate gaps must not validate")

    assert located == ["gaps"]
    assert _diagnostic_error_locations(RuntimeError("boom")) == []


def test_track_analysis_telemetry_names_the_gaps_a_kept_track_declares(caplog):
    """A report running on declared gaps must not look like a complete one.

    ENABLE_REPORT_PARTIAL_TRACK_EVIDENCE deliberately keeps tracks that could
    not supply everything; an operator who cannot see which, and what they
    owe, cannot tell the flag worked from the flag never being read.
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
    packet_by_track = {packet.track_id: packet for packet in packets}
    gapped_track_id = research_plan.tracks[0].track_id

    def track_analyzer(_query, track, **_kwargs):
        from contracts.report_research import ReportTrackStatus

        packet = packet_by_track[track.track_id]
        if track.track_id != gapped_track_id:
            return packet
        return packet.model_copy(
            update={
                "gaps": ["MISSING_DERIVED_METRIC_MOM_PERCENT_CHANGE"],
                "status": ReportTrackStatus.PARTIAL,
            }
        )

    caplog.set_level("INFO", logger="Enai.ReportProcessor")
    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: pytest.fail(
            "enabled track analysis must not run global enrichment"
        ),
        track_analysis_mode="enabled",
        track_analyzer=track_analyzer,
        research_planner=lambda *_args, **_kwargs: research_plan,
        research_executor=lambda *_args, **_kwargs: packets,
        manifest_consolidator=lambda *_args, **_kwargs: manifest,
        research_exhibit_builder=lambda *_args, **_kwargs: decisions,
        evidence_gate_evaluator=lambda *_args, **_kwargs: gate,
        document_planner=lambda *_args, **_kwargs: document_plan,
        document_generator=lambda *_args, **_kwargs: draft,
    )

    processor(_lease(query=_V2_QUERY), _Control())

    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_TRACK_ANALYSIS_ENABLED ")
    )
    payload = json.loads(record.split(" ", 1)[1])
    assert payload["failed_tracks"] == []
    assert payload["gapped_tracks"] == [
        {
            "track_id": gapped_track_id,
            "gaps": ["MISSING_DERIVED_METRIC_MOM_PERCENT_CHANGE"],
            "status": "partial",
        }
    ]


def test_a_model_level_checkpoint_rejection_names_the_rule(caplog):
    """An empty invalid_fields is what a model validator always produces.

    Every identity rule on ReportGenerationCheckpoint lives in a
    model_validator, and pydantic gives those an empty ``loc`` — so
    REPORT_CHECKPOINT_INVALID has never been able to name anything. Job
    e3f43e84 died at document_plan_ready with invalid_fields=[], the same
    blind spot the named-offender work was supposed to have closed.
    """

    (
        research_plan,
        _packets,
        manifest,
        _decisions,
        _gate,
        document_plan,
    ) = _document_components()

    with caplog.at_level(logging.WARNING, logger="Enai.ReportProcessor"):
        with pytest.raises(ReportJobFailure):
            ReportJobProcessor._safe_v3_checkpoint_payload(
                checkpoint_stage="document_plan_ready",
                research_plan=research_plan,
                manifest=manifest,
                # A document plan is required at this stage; omitting it trips
                # a model-level rule and nothing else.
                document_plan=None,
            )

    logged = [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("REPORT_CHECKPOINT_INVALID ")
    ]
    assert logged, "the rejection was not reported"
    assert logged[0]["checkpoint_stage"] == "document_plan_ready"
    assert logged[0]["invalid_rules"], "no rejected rule was named"
    assert any(
        "document plan" in rule.lower()
        for rule in logged[0]["invalid_rules"]
    ), logged[0]["invalid_rules"]


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
    lease = _lease(query=_V2_QUERY)
    # Thrown from the document generator, not the nested pipeline: narrative
    # enrichment is deliberately defensive and swallows its own failures, so a
    # deadline raised there would never reach the attempt-level mapping under
    # test.
    processor = _v2_processor(
        document_generator=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RequestDeadlineExceeded("report_pipeline")
        ),
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(lease, _Control())

    assert exc_info.value.error_code == "REPORT_DEADLINE_EXCEEDED"
    assert exc_info.value.retryable is True


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
        _v2_processor()(lease, _Control())

    assert exc_info.value.error_code == "REPORT_CHECKPOINT_INVALID"
    assert exc_info.value.retryable is False


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

    lease = _lease(query=_V2_QUERY)
    pipeline_calls: list = []
    scopes: list = []

    result = _v2_processor(
        pipeline_calls=pipeline_calls,
        execution_scopes=scopes,
        job_timeout_seconds=120,
    )(lease, _Control())

    ReportResultV2.model_validate(result)
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


def test_every_attempt_runs_the_document_pipeline():
    """One path: a lease reaches the document pipeline whatever the config says.

    The processor used to choose between two pipelines on a mode flag. With the
    retired path deleted there is nothing to choose, so a processor built
    without a mode still runs the document pipeline rather than the planner.
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

    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: pytest.fail(
            "no legacy enrichment path remains"
        ),
        track_analysis_mode="enabled",
        track_analyzer=lambda _query, track, **_kwargs: {
            packet.track_id: packet for packet in packets
        }[track.track_id],
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


def test_the_surviving_checkpoint_builder_still_separates_too_large_from_invalid():
    """The size cap follows the surviving builder.

    Three legacy builders mapped ReportCheckpointTooLargeError to
    REPORT_CHECKPOINT_TOO_LARGE and were deleted with the retired path. The
    distinction matters operationally -- too-large is a sizing bug, invalid is
    a contract bug, and they were separated deliberately -- so
    _safe_v3_checkpoint_payload has to keep making it.

    Driven through the builder's own failure modes rather than a manufactured
    megabyte: the payload is capped at 1 MB while a manifest is capped at
    768 KB, so no valid manifest can overflow a checkpoint on its own and any
    fixture large enough would be rejected as invalid first -- testing the
    fixture, not the mapping.
    """
    from contracts.report_generation import ReportCheckpointTooLargeError

    processor = ReportJobProcessor(
        query_pipeline=lambda *_args, **_kwargs: pytest.fail("not reached"),
    )

    def too_large(**_kwargs):
        raise ReportCheckpointTooLargeError("checkpoint exceeds its limit")

    def invalid(**_kwargs):
        raise ValueError("checkpoint identity is wrong")

    for build, expected in (
        (too_large, "REPORT_CHECKPOINT_TOO_LARGE"),
        (invalid, "REPORT_CHECKPOINT_INVALID"),
    ):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                ReportJobProcessor,
                "_v3_checkpoint_payload",
                staticmethod(build),
            )
            with pytest.raises(ReportJobFailure) as failure:
                processor._safe_v3_checkpoint_payload(
                    checkpoint_stage="evidence_ready",
                )
        assert failure.value.error_code == expected
        assert failure.value.retryable is False


def test_a_planner_knowledge_track_is_not_sent_to_the_analysis_pipeline(caplog):
    """A documented-knowledge track has no tabular evidence to fetch.

    Across jobs 40e55527, 5cb4d210, 106b043c, 70692961 and 26f3bbf6 the planner
    declared a track `knowledge` and the nested analyzer chose `tool` every
    time. On 26f3bbf6 that pulled 61 rows of prices and a 62 KB stats_hint into
    a question about documented mechanisms. Token guardrails cannot separate
    "explain the mechanism" from "explain this month's movement"; the planner
    already made the call.
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
    packet_by_track = {packet.track_id: packet for packet in packets}

    knowledge_track_id = research_plan.tracks[-1].track_id
    payload = research_plan.model_dump(mode="json")
    for track in payload["tracks"]:
        if track["track_id"] == knowledge_track_id:
            track["analysis_preferred_path"] = "knowledge"
    from contracts.report_research import ReportResearchPlan

    research_plan = ReportResearchPlan.model_validate(payload)

    analysis_calls = []

    def track_analyzer(_query, track, **_kwargs):
        analysis_calls.append(track.track_id)
        return packet_by_track[track.track_id]

    caplog.set_level("INFO", logger="Enai.ReportProcessor")
    processor = ReportJobProcessor(
        query_pipeline=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("enabled track analysis must not run globally")
        ),
        track_analysis_mode="enabled",
        track_analyzer=track_analyzer,
        research_planner=lambda *_args, **_kwargs: research_plan,
        research_executor=lambda *_args, **_kwargs: packets,
        manifest_consolidator=lambda *_a, **_k: manifest,
        research_exhibit_builder=lambda *_args, **_kwargs: decisions,
        evidence_gate_evaluator=lambda *_args, **_kwargs: gate,
        document_planner=lambda *_args, **_kwargs: document_plan,
        document_generator=lambda *_args, **_kwargs: draft,
    )

    ReportResultV2.model_validate(processor(_lease(query=_V2_QUERY), _Control()))

    assert knowledge_track_id not in analysis_calls
    assert sorted(analysis_calls) == sorted(
        track.track_id
        for track in research_plan.tracks
        if track.track_id != knowledge_track_id
    )
    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_TRACK_ANALYSIS_ENABLED ")
    )
    telemetry = json.loads(record.split(" ", 1)[1])
    # Deliberate, not degraded.
    assert telemetry["planner_knowledge_track_ids"] == [knowledge_track_id]
    assert telemetry["fallback_count"] == 0


def test_a_knowledge_track_with_no_baseline_evidence_is_still_analysed():
    """The skip must not turn a thin track into an empty one."""

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

    knowledge_track_id = research_plan.tracks[-1].track_id
    payload = research_plan.model_dump(mode="json")
    for track in payload["tracks"]:
        if track["track_id"] == knowledge_track_id:
            track["analysis_preferred_path"] = "knowledge"
    from contracts.report_research import ReportResearchPlan

    research_plan = ReportResearchPlan.model_validate(payload)

    # That track's deterministic packet carries nothing, so skipping it would
    # leave the report with no evidence for it at all.
    emptied = [
        packet
        for packet in packets
        if packet.track_id != knowledge_track_id
    ]

    analysis_calls = []

    def track_analyzer(_query, track, **_kwargs):
        analysis_calls.append(track.track_id)
        return packet_by_track[track.track_id]

    processor = ReportJobProcessor(
        query_pipeline=lambda *_a, **_k: None,
        track_analysis_mode="enabled",
        track_analyzer=track_analyzer,
        research_planner=lambda *_args, **_kwargs: research_plan,
        research_executor=lambda *_args, **_kwargs: emptied,
        manifest_consolidator=lambda *_a, **_k: manifest,
        research_exhibit_builder=lambda *_args, **_kwargs: decisions,
        evidence_gate_evaluator=lambda *_args, **_kwargs: gate,
        document_planner=lambda *_args, **_kwargs: document_plan,
        document_generator=lambda *_args, **_kwargs: draft,
    )

    ReportResultV2.model_validate(processor(_lease(query=_V2_QUERY), _Control()))

    assert knowledge_track_id in analysis_calls
