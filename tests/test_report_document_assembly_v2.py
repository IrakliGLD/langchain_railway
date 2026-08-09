"""Deterministic assembly tests for adaptive report documents."""

from __future__ import annotations

from agent.report_document_assembly import assemble_report_document
from agent.report_document_planner import build_report_document_plan
from agent.report_evidence_gate import evaluate_report_evidence
from contracts.report_charts import ReportChartBuildDecision
from contracts.report_result import ReportResultV2
from tests.test_report_document_pipeline_v2 import (
    _QUERY,
    _document_components,
    _ready_components,
    _valid_document_draft,
)


def test_document_assembly_preserves_adaptive_order_charts_and_citations():
    (
        research_plan,
        _,
        manifest,
        decisions,
        _,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)

    result = assemble_report_document(
        document_plan,
        research_plan,
        manifest,
        draft,
        decisions,
    )

    assert isinstance(result, ReportResultV2)
    assert result.contract_version == "report-result-v2"
    assert result.coverage_status == document_plan.coverage_status
    assert [section.section_id for section in result.sections] == [
        section.section_id for section in document_plan.sections
    ]
    assert [section.kind.value for section in result.sections] == [
        section.role.value for section in document_plan.sections
    ]
    assert {chart.chart_id for chart in result.charts} == {
        "prices_trend",
        "security_composition",
    }
    assert all(
        chart.section_id
        == next(
            request.section_id
            for request in document_plan.charts
            if request.chart_id == chart.chart_id
        )
        for chart in result.charts
    )
    used_refs = {
        ref
        for section in result.sections
        for ref in section.evidence_refs
    } | {
        ref
        for chart in result.charts
        for ref in chart.metadata.evidence_refs
    }
    assert {citation.evidence_ref for citation in result.citations} == used_refs
    assert result.word_count == sum(
        section.word_count for section in result.sections
    )


def test_document_assembly_discloses_an_expected_exhibit_omission():
    research_plan, packets, manifest, decisions, _ = _ready_components()
    unavailable = decisions[0]
    decisions[0] = ReportChartBuildDecision(
        chart_id=unavailable.chart_id,
        required=unavailable.required,
        status="omitted",
        reason_code="REPORT_CHART_TIME_AXIS_REQUIRED",
        artifact=None,
    )
    gate = evaluate_report_evidence(
        research_plan,
        packets,
        chart_decisions=decisions,
    )
    document_plan = build_report_document_plan(
        _QUERY,
        research_plan,
        packets,
        manifest,
        gate,
        decisions,
    )
    draft = _valid_document_draft(document_plan, manifest)

    result = assemble_report_document(
        document_plan,
        research_plan,
        manifest,
        draft,
        decisions,
    )

    assert result.coverage_status == "ready_with_gaps"
    assert [item.chart_id for item in result.omitted_charts] == [
        unavailable.chart_id
    ]
    assert result.omitted_charts[0].reason_code == (
        "REPORT_CHART_TIME_AXIS_REQUIRED"
    )
    assert unavailable.chart_id not in {
        chart_ref
        for section in result.sections
        for chart_ref in section.chart_refs
    }


def test_a_conceded_length_shortfall_still_assembles():
    """The gate ships a short document deliberately; assembly must agree.

    _concede_length_shortfall downgrades a pure length shortfall to a warning
    so the reader gets a usable report instead of a non-retryable failure. The
    concession lived only in the gate's validation object, and the assembler
    re-validated the draft from scratch and reached the opposite verdict — so
    every conceded document died at assembly. Jobs 6c01bd62 (one conceded
    section) and 70692961 (four, a Georgian report whose prose is far shorter
    than the English-calibrated bounds) both failed exactly there.
    """

    import pytest

    from agent.report_document_assembly import (
        ReportDocumentAssemblyError,
        assemble_report_document,
    )
    from agent.report_document_generation import validate_report_document
    from contracts.report import REPORT_SECTION_MAX_WORDS

    (
        research_plan,
        _packets,
        manifest,
        decisions,
        _gate,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)

    # Raise the bounds instead of damaging the prose, so the only complaint
    # the gate can raise is length and grounding stays intact.
    document_plan = document_plan.model_copy(
        update={
            "sections": [
                section.model_copy(
                    update={"target_words": REPORT_SECTION_MAX_WORDS}
                )
                for section in document_plan.sections
            ]
        }
    )
    validation = validate_report_document(
        draft, document_plan, manifest, research_plan
    )
    assert not validation.valid, "the bounds must be high enough to complain"
    assert not [
        code
        for codes in validation.section_errors.values()
        for code in codes
        if code != "WORD_COUNT_TOO_SHORT"
    ], "this fixture must isolate length, not grounding"

    result = assemble_report_document(
        document_plan,
        research_plan,
        manifest,
        draft,
        decisions,
    )

    assert isinstance(result, ReportResultV2)
