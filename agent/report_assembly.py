"""Deterministically assemble validated sections, charts, and citations."""

from __future__ import annotations

import math

from agent.report_grounding import build_evidence_grounding_index
from agent.report_sections import validate_report_section
from contracts.report import ReportPlan
from contracts.report_charts import ReportChartBuildDecision
from contracts.report_evidence import ReportEvidenceManifest
from contracts.report_result import (
    ReportChartOmission,
    ReportCitation,
    ReportResult,
    ReportResultSection,
)
from contracts.report_sections import ReportSectionDraft


class ReportAssemblyError(ValueError):
    """Validated components cannot form the planned report."""


def assemble_report(
    plan: ReportPlan,
    manifest: ReportEvidenceManifest,
    drafts: list[ReportSectionDraft],
    chart_decisions: list[ReportChartBuildDecision],
) -> ReportResult:
    draft_by_id = {draft.section_id: draft for draft in drafts}
    expected_ids = [section.section_id for section in plan.sections]
    if len(draft_by_id) != len(drafts) or set(draft_by_id) != set(expected_ids):
        raise ReportAssemblyError("Report draft section set does not match the plan.")

    result_sections: list[ReportResultSection] = []
    used_refs: list[str] = []
    markdown_parts = [f"# {plan.title}"]
    total_words = 0
    item_by_ref = manifest.item_by_ref()
    grounding_index = build_evidence_grounding_index(
        item_by_ref,
        set(item_by_ref),
    )
    for section in plan.sections:
        draft = draft_by_id[section.section_id]
        validation = validate_report_section(
            draft,
            section,
            manifest,
            evidence_facts_by_ref=grounding_index,
        )
        if not validation.valid:
            raise ReportAssemblyError(
                f"Report section {section.section_id} is no longer valid."
            )
        section_refs = list(
            dict.fromkeys(
                ref
                for paragraph in draft.paragraphs
                for ref in paragraph.evidence_refs
            )
        )
        used_refs.extend(section_refs)
        total_words += validation.word_count
        result_sections.append(
            ReportResultSection(
                section_id=section.section_id,
                kind=section.kind,
                title=section.title,
                content_markdown=draft.content_markdown,
                evidence_refs=section_refs,
                chart_refs=section.chart_refs,
                word_count=validation.word_count,
            )
        )
        markdown_parts.extend(
            [
                f"## {section.title}",
                draft.content_markdown,
            ]
        )

    minimum_report_words = sum(
        math.floor(section.target_words * 0.9)
        for section in plan.sections
    )
    maximum_report_words = sum(
        math.ceil(section.target_words * 1.2)
        for section in plan.sections
    )
    if not minimum_report_words <= total_words <= maximum_report_words:
        raise ReportAssemblyError("Final report word count is outside the plan tolerance.")

    decision_by_id = {
        decision.chart_id: decision
        for decision in chart_decisions
    }
    if set(decision_by_id) != {chart.chart_id for chart in plan.charts}:
        raise ReportAssemblyError("Report chart decision set does not match the plan.")
    required_omissions = [
        chart.chart_id
        for chart in plan.charts
        if chart.required
        and decision_by_id[chart.chart_id].status != "built"
    ]
    if required_omissions:
        raise ReportAssemblyError(
            "A required chart could not be assembled: "
            + ", ".join(required_omissions)
        )
    charts = [
        decision.artifact
        for decision in chart_decisions
        if decision.status == "built" and decision.artifact is not None
    ]
    chart_title_by_id = {chart.chart_id: chart.title for chart in plan.charts}
    omitted_charts = [
        ReportChartOmission(
            chart_id=decision.chart_id,
            title=chart_title_by_id[decision.chart_id],
            reason_code=decision.reason_code or "REPORT_CHART_OMITTED",
        )
        for decision in chart_decisions
        if decision.status != "built"
    ]
    built_chart_ids = {chart.chart_id for chart in charts}
    result_sections = [
        section.model_copy(
            update={
                "chart_refs": [
                    chart_id
                    for chart_id in section.chart_refs
                    if chart_id in built_chart_ids
                ]
            }
        )
        for section in result_sections
    ]

    chart_refs = [
        ref
        for chart in charts
        for ref in chart.metadata.evidence_refs
    ]
    unique_used_refs = list(dict.fromkeys([*used_refs, *chart_refs]))
    citations = [
        ReportCitation(
            evidence_ref=ref,
            title=item_by_ref[ref].title,
            source=item_by_ref[ref].source,
            provenance_refs=item_by_ref[ref].provenance_refs,
        )
        for ref in unique_used_refs
    ]
    return ReportResult(
        contract_version="report-result-v1",
        intent=plan.intent,
        title=plan.title,
        objective=plan.objective,
        language_code=plan.language_code,
        evidence_manifest_id=manifest.manifest_id,
        content_markdown="\n\n".join(markdown_parts),
        sections=result_sections,
        charts=charts,
        omitted_charts=omitted_charts,
        citations=citations,
        word_count=total_words,
    )
