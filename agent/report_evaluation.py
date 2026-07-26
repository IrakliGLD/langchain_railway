"""Content-free readiness evaluation for report plans in shadow or enforce mode."""

from __future__ import annotations

from agent.report_charts import build_report_charts
from agent.report_planner import (
    ReportPlanEvidenceError,
    validate_report_plan_evidence,
)
from contracts.report import STANDARD_REPORT_SECTION_SEQUENCE, ReportPlan
from contracts.report_evaluation import ReportPlanEvaluation
from contracts.report_evidence import ReportEvidenceManifest


def evaluate_report_plan(
    plan: ReportPlan,
    manifest: ReportEvidenceManifest,
) -> ReportPlanEvaluation:
    findings: list[str] = []
    known_refs = set(manifest.item_by_ref())
    all_refs = [
        ref
        for section in plan.sections
        for ref in section.required_evidence_refs
    ] + [
        ref
        for chart in plan.charts
        for ref in chart.evidence_refs
    ]
    evidence_coverage = (
        sum(ref in known_refs for ref in all_refs) / len(all_refs)
        if all_refs
        else 0.0
    )

    try:
        validate_report_plan_evidence(plan, manifest)
    except ReportPlanEvidenceError:
        findings.append("PLAN_EVIDENCE_INVALID")

    present_kinds = {section.kind for section in plan.sections}
    required_section_coverage = (
        sum(kind in present_kinds for kind in STANDARD_REPORT_SECTION_SEQUENCE)
        / len(STANDARD_REPORT_SECTION_SEQUENCE)
    )
    if required_section_coverage < 1.0:
        findings.append("REQUIRED_SECTION_MISSING")

    chart_decisions = build_report_charts(plan, manifest)
    required_charts = [decision for decision in chart_decisions if decision.required]
    required_chart_build_rate = (
        sum(decision.status == "built" for decision in required_charts)
        / len(required_charts)
        if required_charts
        else 1.0
    )
    if required_chart_build_rate < 1.0:
        findings.append("REQUIRED_CHART_OMITTED")

    findings = list(dict.fromkeys(findings))
    return ReportPlanEvaluation(
        contract_version="report-evaluation-v1",
        manifest_id=manifest.manifest_id,
        ready_for_generation=not findings,
        evidence_reference_coverage=evidence_coverage,
        required_section_coverage=required_section_coverage,
        required_chart_build_rate=required_chart_build_rate,
        findings=findings,
    )
