"""Deterministic final report assembly tests."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

from agent import report_grounding
from agent.report_assembly import ReportAssemblyError, assemble_report
from agent.report_charts import build_report_charts
from contracts.report import ReportIntent, ReportPlan, ReportSectionKind
from contracts.report_sections import ReportSectionDraft
from tests.fixtures_report_manifest import _manifest, _plan_payload
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


def test_assembly_reuses_grounding_facts_across_sections(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    manifest = _manifest()
    original = report_grounding._evidence_grounding_facts
    calls = Counter()

    def counted(item):
        calls[item.evidence_ref] += 1
        return original(item)

    monkeypatch.setattr(report_grounding, "_evidence_grounding_facts", counted)

    assemble_report(
        plan,
        manifest,
        _drafts(plan),
        build_report_charts(plan, manifest),
    )

    assert calls == Counter(
        {item.evidence_ref: 1 for item in manifest.items}
    )


def test_assembly_preserves_non_general_intent_structure_in_result():
    payload = deepcopy(_plan_payload())
    payload["intent"] = "trend"
    payload["sections"][2]["kind"] = "trend_analysis"
    plan = ReportPlan.model_validate(payload)

    result = assemble_report(
        plan,
        _manifest(),
        _drafts(plan),
        build_report_charts(plan, _manifest()),
    )

    assert result.intent is ReportIntent.TREND
    assert result.sections[2].kind is ReportSectionKind.TREND_ANALYSIS


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


def test_assembly_discloses_omitted_charts_with_their_reason_code():
    from contracts.report_charts import ReportChartBuildDecision

    payload = _plan_payload()
    payload["charts"][0]["required"] = False
    plan = ReportPlan.model_validate(payload)
    drafts = _drafts(plan)
    decisions = [
        ReportChartBuildDecision(
            chart_id=chart.chart_id,
            required=False,
            status="omitted",
            reason_code="REPORT_CHART_TIME_AXIS_REQUIRED",
            artifact=None,
        )
        for chart in plan.charts
    ]

    result = assemble_report(plan, _manifest(), drafts, decisions)

    assert [omission.chart_id for omission in result.omitted_charts] == [
        chart.chart_id for chart in plan.charts
    ]
    assert result.omitted_charts[0].reason_code == "REPORT_CHART_TIME_AXIS_REQUIRED"
    assert result.omitted_charts[0].title == "Observed electricity price"
    assert result.charts == []


def test_assembly_reports_no_omissions_when_every_chart_builds():
    plan = ReportPlan.model_validate(_plan_payload())
    result = assemble_report(
        plan,
        _manifest(),
        _drafts(plan),
        build_report_charts(plan, _manifest()),
    )

    assert result.omitted_charts == []
