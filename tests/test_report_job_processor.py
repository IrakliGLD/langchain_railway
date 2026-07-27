"""End-to-end orchestration tests for one durable report job attempt."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from agent.report_planner import ReportPlanEvidenceError
from agent.report_sections import (
    ReportSectionGenerationError,
    generate_report_sections,
)
from contracts.report import ReportPlan
from contracts.report_generation import ReportGenerationCheckpoint
from contracts.report_jobs import ReportJobLease, ReportJobPhase
from contracts.report_result import ReportResult
from contracts.report_sections import ReportSectionDraft
from core.report_job_processor import ReportJobProcessor
from core.report_job_worker import ReportJobFailure
from models import QueryContext
from tests.test_report_planner import _manifest, _plan_payload
from tests.test_report_sections import _draft
from utils.provider_attempts import (
    ProviderDeliveryDisposition,
    ProviderExecutionError,
)


def _lease(
    *,
    checkpoint: dict | None = None,
    phase: str = "planning",
    progress_percent: int = 5,
) -> ReportJobLease:
    return ReportJobLease.model_validate(
        {
            "contract_version": "report-job-v1",
            "job_id": str(uuid4()),
            "request_id": "report:req-processor",
            "actor_user_id": str(uuid4()),
            "query": "Explain the price trend.",
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


def _processor(*, pipeline_calls: list | None = None, generated: list | None = None):
    calls = pipeline_calls if pipeline_calls is not None else []
    generated_ids = generated if generated is not None else []

    def pipeline(query, **kwargs):
        calls.append((query, kwargs))
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
        )

    return ReportJobProcessor(
        query_pipeline=pipeline,
        evidence_builder=lambda ctx: _manifest_for_query(ctx.query),
        planner=lambda _query, _manifest_value: ReportPlan.model_validate(
            _plan_payload()
        ),
        section_generator=sections,
        max_section_workers=5,
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


def test_fresh_job_runs_pipeline_parallel_sections_and_deterministic_assembly():
    pipeline_calls: list = []
    generated: list = []
    lease = _lease()
    control = _Control()

    raw_result = _processor(
        pipeline_calls=pipeline_calls,
        generated=generated,
    )(lease, control)
    result = ReportResult.model_validate(raw_result)

    assert result.contract_version == "report-result-v1"
    assert [section.section_id for section in result.sections] == [
        section["section_id"] for section in _plan_payload()["sections"]
    ]
    assert set(generated) == {
        section["section_id"] for section in _plan_payload()["sections"]
    }
    assert pipeline_calls == [
        (
            lease.query,
            {
                "trace_id": str(lease.job_id),
                "actor_id": str(lease.actor_user_id),
                "request_id": lease.request_id,
                "answer_mode": "report",
            },
        )
    ]
    assert control.heartbeats[0][:2] == (ReportJobPhase.PLANNING, 10)
    assert control.heartbeats[-1][:2] == (ReportJobPhase.ASSEMBLING, 90)
    assert all(
        earlier[1] <= later[1]
        for earlier, later in zip(control.heartbeats, control.heartbeats[1:])
    )


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

    def invalid_planner(*_args):
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
        planner=lambda *_args: (_ for _ in ()).throw(
            ValueError("The model returned an invalid plan.")
        ),
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(lease, _Control())

    assert exc_info.value.error_code == "REPORT_PLAN_INVALID"
    assert exc_info.value.retryable is False


def test_report_plan_provider_failure_remains_retryable_and_is_diagnosable(
    caplog,
):
    lease = _lease()

    def unavailable_planner(*_args):
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
    assert exc_info.value.retryable is True
    assert f"job_id={lease.job_id}" in caplog.text
    assert f"job_attempt={lease.attempt_count}" in caplog.text
    assert "provider=nvidia" in caplog.text
    assert "provider_stage=report_planner" in caplog.text
    assert "provider_disposition=timed_out" in caplog.text
    assert "provider secret" not in caplog.text


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
        planner=lambda *_args: ReportPlan.model_validate(_plan_payload()),
        section_generator=fail_sections,
    )

    with pytest.raises(ReportJobFailure) as exc_info:
        processor(lease, _Control())

    assert exc_info.value.error_code == expected_code
    assert exc_info.value.retryable is expected_retryable
