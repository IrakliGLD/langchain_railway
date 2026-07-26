"""Evidence-bound standard report planning boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from contracts.report import ReportPlan, ReportSectionKind
from contracts.report_evidence import (
    ReportEvidenceKind,
    ReportEvidenceManifest,
)


class ReportPlanEvidenceError(ValueError):
    """The plan references evidence outside its immutable manifest."""


def validate_report_plan_evidence(
    plan: ReportPlan,
    manifest: ReportEvidenceManifest,
) -> None:
    if plan.evidence_manifest_id != manifest.manifest_id:
        raise ReportPlanEvidenceError("Report plan manifest identity does not match.")

    item_by_ref = manifest.item_by_ref()
    known_refs = set(item_by_ref)
    substantive_refs = {
        ref
        for ref, item in item_by_ref.items()
        if item.kind is not ReportEvidenceKind.LIMITATION
    }
    if not substantive_refs:
        raise ReportPlanEvidenceError(
            "A report plan requires at least one substantive evidence item."
        )
    referenced = {
        ref
        for section in plan.sections
        for ref in section.required_evidence_refs
    }
    referenced.update(
        ref
        for chart in plan.charts
        for ref in chart.evidence_refs
    )
    unknown = sorted(referenced - known_refs)
    if unknown:
        raise ReportPlanEvidenceError(
            "Report plan contains unknown evidence references: "
            + ", ".join(unknown)
        )

    limitations = next(
        section
        for section in plan.sections
        if section.kind is ReportSectionKind.LIMITATIONS
    )
    if not any(
        item_by_ref[ref].kind is ReportEvidenceKind.LIMITATION
        for ref in limitations.required_evidence_refs
    ):
        raise ReportPlanEvidenceError(
            "The limitations section must cite at least one typed limitation."
        )

    for section in plan.sections:
        if (
            section.kind is not ReportSectionKind.LIMITATIONS
            and not (set(section.required_evidence_refs) & substantive_refs)
        ):
            raise ReportPlanEvidenceError(
                f"Section {section.section_id} must cite substantive evidence."
            )

    for chart in plan.charts:
        chart_items = [item_by_ref[ref] for ref in chart.evidence_refs]
        if any(
            item.kind is not ReportEvidenceKind.TABLE
            for item in chart_items
        ):
            raise ReportPlanEvidenceError(
                f"Chart {chart.chart_id} may use only tabular evidence."
            )
        available_columns = {
            column
            for item in chart_items
            for column in item.columns
        }
        requested_fields = set(chart.series_fields)
        if chart.x_field is not None:
            requested_fields.add(chart.x_field)
        unknown_fields = sorted(requested_fields - available_columns)
        if unknown_fields:
            raise ReportPlanEvidenceError(
                f"Chart {chart.chart_id} contains unknown chart fields: "
                + ", ".join(unknown_fields)
            )
        if chart.x_field is not None and chart.x_field in chart.series_fields:
            raise ReportPlanEvidenceError(
                f"Chart {chart.chart_id} cannot use x_field as a series field."
            )


ReportPlanInvoker = Callable[[str, ReportEvidenceManifest], Any]


def plan_report(
    query: str,
    manifest: ReportEvidenceManifest,
    *,
    invoke_model: ReportPlanInvoker | None = None,
) -> ReportPlan:
    if invoke_model is None:
        from core.llm import llm_plan_report

        invoke_model = llm_plan_report
    raw_plan = invoke_model(query, manifest)
    plan = raw_plan if isinstance(raw_plan, ReportPlan) else ReportPlan.model_validate(raw_plan)
    validate_report_plan_evidence(plan, manifest)
    return plan
