"""Evidence-bound standard report planning boundary."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from agent.report_charts import build_report_charts
from contracts.report import (
    ReportPlan,
    ReportSectionKind,
    normalize_report_plan_word_budget,
)
from contracts.report_evidence import (
    ReportEvidenceKind,
    ReportEvidenceManifest,
)

_LOGGER = logging.getLogger("Enai.ReportPlanner")


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


def _repair_report_plan_evidence(
    plan: ReportPlan,
    manifest: ReportEvidenceManifest,
) -> ReportPlan:
    """Repair only evidence bindings that code can verify deterministically."""

    item_by_ref = manifest.item_by_ref()
    known_refs = set(item_by_ref)
    substantive_refs = [
        item.evidence_ref
        for item in manifest.items
        if item.kind is not ReportEvidenceKind.LIMITATION
    ]
    limitation_refs = [
        item.evidence_ref
        for item in manifest.items
        if item.kind is ReportEvidenceKind.LIMITATION
    ]
    if not substantive_refs:
        raise ReportPlanEvidenceError(
            "A report plan requires at least one substantive evidence item."
        )
    if not limitation_refs:
        raise ReportPlanEvidenceError(
            "A report plan requires at least one typed limitation."
        )

    payload = plan.model_dump(mode="json")
    payload["evidence_manifest_id"] = manifest.manifest_id

    substantive_set = set(substantive_refs)
    limitation_set = set(limitation_refs)
    for section in payload["sections"]:
        refs = list(
            dict.fromkeys(
                ref
                for ref in section["required_evidence_refs"]
                if ref in known_refs
            )
        )
        if section["kind"] == ReportSectionKind.LIMITATIONS.value:
            if not limitation_set.intersection(refs):
                refs.append(limitation_refs[0])
        elif not substantive_set.intersection(refs):
            refs.append(substantive_refs[0])
        section["required_evidence_refs"] = refs[:32]

    repaired_charts: list[dict[str, Any]] = []
    for chart in payload["charts"]:
        table_refs = [
            ref
            for ref in dict.fromkeys(chart["evidence_refs"])
            if (
                ref in item_by_ref
                and item_by_ref[ref].kind is ReportEvidenceKind.TABLE
            )
        ]
        if not table_refs:
            continue

        first_columns = item_by_ref[table_refs[0]].columns
        table_refs = [
            ref
            for ref in table_refs
            if item_by_ref[ref].columns == first_columns
        ]
        available_columns = set(first_columns)
        x_field = chart.get("x_field")
        if x_field not in available_columns:
            x_field = None
        chart["x_field"] = x_field
        chart["series_fields"] = [
            field
            for field in dict.fromkeys(chart.get("series_fields", []))
            if field in available_columns and field != x_field
        ]
        chart["evidence_refs"] = table_refs
        repaired_charts.append(chart)

    payload["charts"] = repaired_charts
    retained_chart_ids = {chart["chart_id"] for chart in repaired_charts}
    for section in payload["sections"]:
        section["chart_refs"] = [
            chart_ref
            for chart_ref in section["chart_refs"]
            if chart_ref in retained_chart_ids
        ]

    repaired = ReportPlan.model_validate(payload)
    omitted_chart_ids = {
        decision.chart_id
        for decision in build_report_charts(repaired, manifest)
        if decision.status == "omitted"
    }
    if omitted_chart_ids:
        payload["charts"] = [
            chart
            for chart in payload["charts"]
            if chart["chart_id"] not in omitted_chart_ids
        ]
        for section in payload["sections"]:
            section["chart_refs"] = [
                chart_ref for chart_ref in section["chart_refs"] if chart_ref not in omitted_chart_ids
            ]
        repaired = ReportPlan.model_validate(payload)

    validate_report_plan_evidence(repaired, manifest)
    return repaired


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
    plan = (
        raw_plan
        if isinstance(raw_plan, ReportPlan)
        else ReportPlan.model_validate(
            normalize_report_plan_word_budget(raw_plan)
        )
    )
    requires_repair = False
    try:
        validate_report_plan_evidence(plan, manifest)
    except ReportPlanEvidenceError:
        requires_repair = True
    else:
        requires_repair = any(
            decision.status == "omitted"
            for decision in build_report_charts(plan, manifest)
        )
    if requires_repair:
        plan = _repair_report_plan_evidence(plan, manifest)
        _LOGGER.warning(
            "Stabilized report plan evidence/chart bindings: manifest_id=%s",
            manifest.manifest_id,
        )
    return plan
