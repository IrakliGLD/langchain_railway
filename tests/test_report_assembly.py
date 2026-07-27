"""Deterministic final report assembly tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent.report_assembly import ReportAssemblyError, assemble_report
from agent.report_charts import build_report_charts
from contracts.report import ReportPlan
from contracts.report_sections import ReportSectionDraft
from tests.test_report_planner import _manifest, _plan_payload
from tests.test_report_sections import _draft


def _drafts(plan):
    return [
        ReportSectionDraft.model_validate(_draft(section))
        for section in plan.sections
    ]


def test_assembly_preserves_validated_section_text_order_charts_and_citations():
    plan = ReportPlan.model_validate(_plan_payload())
    drafts = _drafts(plan)
    result = assemble_report(
        plan,
        _manifest(),
        drafts,
        build_report_charts(plan, _manifest()),
    )

    assert result.contract_version == "report-result-v1"
    assert result.title == plan.title
    assert result.evidence_manifest_id == _manifest().manifest_id
    assert [section.section_id for section in result.sections] == [
        section.section_id for section in plan.sections
    ]
    assert [chart.chart_id for chart in result.charts] == ["price_trend"]
    assert {citation.evidence_ref for citation in result.citations} == {
        ref
        for section in plan.sections
        for ref in section.required_evidence_refs
    }

    positions = [
        result.content_markdown.index(f"## {section.title}")
        for section in plan.sections
    ]
    assert positions == sorted(positions)
    for draft in drafts:
        assert draft.paragraphs[0].text in result.content_markdown


def test_assembly_rejects_missing_sections_and_required_chart_omissions():
    plan = ReportPlan.model_validate(_plan_payload())

    with pytest.raises(ReportAssemblyError, match="section set"):
        assemble_report(
            plan,
            _manifest(),
            _drafts(plan)[:-1],
            build_report_charts(plan, _manifest()),
        )

    chart_payload = deepcopy(_plan_payload())
    chart_payload["charts"][0]["purpose"] = "relationship"
    invalid_chart_plan = ReportPlan.model_validate(chart_payload)
    with pytest.raises(ReportAssemblyError, match="required chart"):
        assemble_report(
            invalid_chart_plan,
            _manifest(),
            _drafts(invalid_chart_plan),
            build_report_charts(invalid_chart_plan, _manifest()),
        )


def test_assembly_accepts_sum_of_individually_valid_section_minimums():
    payload = deepcopy(_plan_payload())
    target_words = [185, 185, 185, 185, 160]
    for section_payload, target in zip(
        payload["sections"],
        target_words,
        strict=True,
    ):
        section_payload["target_words"] = target
    plan = ReportPlan.model_validate(payload)
    drafts = []
    for section in plan.sections:
        minimum_words = int(section.target_words * 0.9)
        drafts.append(
            ReportSectionDraft.model_validate(
                _draft(
                    section,
                    text=" ".join(["Evidence"] * minimum_words),
                )
            )
        )

    result = assemble_report(
        plan,
        _manifest(),
        drafts,
        build_report_charts(plan, _manifest()),
    )

    assert result.word_count == 808
