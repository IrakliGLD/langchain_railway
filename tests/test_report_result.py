"""Independent validation tests for the published report result."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from agent.report_assembly import assemble_report
from agent.report_charts import build_report_charts
from contracts.report import ReportPlan
from contracts.report_result import ReportResult
from contracts.report_sections import ReportSectionDraft
from tests.test_report_planner import _manifest, _plan_payload
from tests.test_report_sections import _draft


def _valid_result_payload() -> dict:
    plan = ReportPlan.model_validate(_plan_payload())
    drafts = [
        ReportSectionDraft.model_validate(_draft(section))
        for section in plan.sections
    ]
    return assemble_report(
        plan,
        _manifest(),
        drafts,
        build_report_charts(plan, _manifest()),
    ).model_dump(mode="json")


def test_report_result_revalidates_standard_structure_and_chart_binding():
    wrong_order = _valid_result_payload()
    wrong_order["sections"][1], wrong_order["sections"][2] = (
        wrong_order["sections"][2],
        wrong_order["sections"][1],
    )
    with pytest.raises(ValidationError, match="standard section order"):
        ReportResult.model_validate(wrong_order)

    unbound_chart = _valid_result_payload()
    chart_id = unbound_chart["charts"][0]["chart_id"]
    for section in unbound_chart["sections"]:
        section["chart_refs"] = [
            ref for ref in section["chart_refs"] if ref != chart_id
        ]
    with pytest.raises(ValidationError, match="assigned section"):
        ReportResult.model_validate(unbound_chart)


def test_report_result_requires_citations_for_section_and_chart_evidence():
    payload = _valid_result_payload()
    missing_ref = payload["sections"][0]["evidence_refs"][0]
    payload["citations"] = [
        citation
        for citation in payload["citations"]
        if citation["evidence_ref"] != missing_ref
    ]

    with pytest.raises(ValidationError, match="citation"):
        ReportResult.model_validate(payload)


def test_report_result_is_bounded_to_the_standard_report_tolerance():
    payload = _valid_result_payload()
    payload["sections"][0]["word_count"] -= 100
    payload["word_count"] -= 100

    with pytest.raises(ValidationError, match="greater than or equal to 810"):
        ReportResult.model_validate(payload)


def test_report_assembly_cites_evidence_used_only_by_a_chart():
    manifest_payload = _manifest().model_dump(mode="json")
    chart_only_ref = "evidence:table:" + "8" * 16
    chart_only_item = deepcopy(manifest_payload["items"][0])
    chart_only_item.update(
        {
            "evidence_ref": chart_only_ref,
            "title": "Chart-only price evidence",
            "provenance_refs": ["query:tool:chart-only-prices"],
        }
    )
    manifest_payload["items"].append(chart_only_item)
    manifest = type(_manifest()).model_validate(manifest_payload)

    plan_payload = _plan_payload()
    plan_payload["charts"][0]["evidence_refs"] = [chart_only_ref]
    plan = ReportPlan.model_validate(plan_payload)
    drafts = [
        ReportSectionDraft.model_validate(_draft(section))
        for section in plan.sections
    ]

    result = assemble_report(
        plan,
        manifest,
        drafts,
        build_report_charts(plan, manifest),
    )

    assert chart_only_ref in {
        citation.evidence_ref for citation in result.citations
    }


def test_report_assembly_removes_references_to_an_optional_omitted_chart():
    plan_payload = _plan_payload()
    plan_payload["charts"][0].update(
        {
            "purpose": "relationship",
            "required": False,
        }
    )
    plan = ReportPlan.model_validate(plan_payload)
    drafts = [
        ReportSectionDraft.model_validate(_draft(section))
        for section in plan.sections
    ]

    result = assemble_report(
        plan,
        _manifest(),
        drafts,
        build_report_charts(plan, _manifest()),
    )

    assert result.charts == []
    assert all(not section.chart_refs for section in result.sections)
