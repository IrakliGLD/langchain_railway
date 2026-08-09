"""Deterministic assembly for the adaptive whole-document report."""

from __future__ import annotations

from collections.abc import Sequence

from agent.report_document_generation import (
    _concede_length_shortfall,
    validate_report_document,
)
from agent.report_sections import count_section_words
from contracts.report_charts import (
    ReportChartArtifact,
    ReportChartBuildDecision,
)
from contracts.report_document import (
    ReportDocumentDraft,
    ReportDocumentPlan,
)
from contracts.report_evidence import ReportEvidenceManifest
from contracts.report_research import ReportResearchPlan
from contracts.report_result import (
    ReportChartOmission,
    ReportCitation,
    ReportResultV2,
    ReportResultV2Section,
)


class ReportDocumentAssemblyError(ValueError):
    """Validated document components cannot form one durable result."""


def assemble_report_document(
    plan: ReportDocumentPlan,
    research_plan: ReportResearchPlan,
    manifest: ReportEvidenceManifest,
    draft: ReportDocumentDraft,
    chart_decisions: Sequence[ReportChartBuildDecision],
) -> ReportResultV2:
    """Assemble prose, remapped exhibits, omissions, and exact citations."""

    validation = validate_report_document(
        draft,
        plan,
        manifest,
        research_plan,
    )
    if not validation.valid:
        # The gate reached this document only by deciding a pure length
        # shortfall is shippable, and said so with REPORT_LENGTH_CONCEDED.
        # That decision lived in its validation object, not in the draft, so
        # re-validating here from scratch reached the opposite verdict and
        # failed the report anyway — every conceded document died at assembly
        # (jobs 6c01bd62 and 70692961). One authority decides length now.
        validation = (
            _concede_length_shortfall(validation, stage="document_assembly")
            or validation
        )
    if not validation.valid:
        raise ReportDocumentAssemblyError(
            "Report document is no longer valid at assembly."
        )

    decision_by_id = {
        decision.chart_id: decision for decision in chart_decisions
    }
    if len(decision_by_id) != len(chart_decisions):
        raise ReportDocumentAssemblyError(
            "Report chart decisions must use unique IDs."
        )
    planned_chart_ids = {chart.chart_id for chart in plan.charts}
    if not planned_chart_ids.issubset(decision_by_id):
        raise ReportDocumentAssemblyError(
            "Report chart decision set does not cover the document plan."
        )
    required_omissions = [
        chart.chart_id
        for chart in plan.charts
        if (
            chart.required
            and decision_by_id[chart.chart_id].status != "built"
        )
    ]
    if required_omissions:
        raise ReportDocumentAssemblyError(
            "A required chart could not be assembled: "
            + ", ".join(required_omissions)
        )

    charts: list[ReportChartArtifact] = []
    omitted_charts: list[ReportChartOmission] = []
    for request in plan.charts:
        decision = decision_by_id[request.chart_id]
        if decision.status == "built" and decision.artifact is not None:
            charts.append(
                decision.artifact.model_copy(
                    update={"section_id": request.section_id}
                )
            )
        else:
            omitted_charts.append(
                ReportChartOmission(
                    chart_id=request.chart_id,
                    title=request.title,
                    reason_code=(
                        decision.reason_code
                        or "REPORT_CHART_OMITTED"
                    ),
                )
            )

    built_chart_ids = {chart.chart_id for chart in charts}
    spec_by_id = {
        section.section_id: section for section in plan.sections
    }
    result_sections: list[ReportResultV2Section] = []
    used_refs: list[str] = []
    markdown_parts = [f"# {plan.title}"]
    for section in draft.display_order_sections():
        spec = spec_by_id[section.section_id]
        section_refs = list(
            dict.fromkeys(
                ref
                for paragraph in section.paragraphs
                for ref in paragraph.evidence_refs
            )
        )
        used_refs.extend(section_refs)
        result_sections.append(
            ReportResultV2Section(
                section_id=section.section_id,
                kind=spec.role,
                title=section.title,
                content_markdown=section.content_markdown,
                evidence_refs=section_refs,
                chart_refs=[
                    chart_id
                    for chart_id in spec.chart_refs
                    if chart_id in built_chart_ids
                ],
                word_count=count_section_words(
                    section.content_markdown
                ),
            )
        )
        markdown_parts.extend(
            [f"## {section.title}", section.content_markdown]
        )

    chart_refs = [
        ref
        for chart in charts
        for ref in chart.metadata.evidence_refs
    ]
    unique_refs = list(dict.fromkeys([*used_refs, *chart_refs]))
    item_by_ref = manifest.item_by_ref()
    if any(ref not in item_by_ref for ref in unique_refs):
        raise ReportDocumentAssemblyError(
            "Report result cites evidence outside the manifest."
        )
    citations = [
        ReportCitation(
            evidence_ref=ref,
            title=item_by_ref[ref].title,
            source=item_by_ref[ref].source,
            provenance_refs=item_by_ref[ref].provenance_refs,
        )
        for ref in unique_refs
    ]
    return ReportResultV2(
        contract_version="report-result-v2",
        title=plan.title,
        objective=plan.objective,
        language_code=plan.language_code,
        evidence_manifest_id=manifest.manifest_id,
        coverage_status=plan.coverage_status,
        content_markdown="\n\n".join(markdown_parts),
        sections=result_sections,
        charts=charts,
        omitted_charts=omitted_charts,
        citations=citations,
        word_count=sum(
            section.word_count for section in result_sections
        ),
    )
