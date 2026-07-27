"""Evidence-bound standard report planning boundary."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from contracts.report import (
    ReportIntent,
    ReportPlan,
    ReportPlanningContext,
    ReportSectionKind,
    normalize_report_plan_semantics,
    normalize_report_plan_word_budget,
)
from contracts.report_evidence import (
    ReportEvidenceKind,
    ReportEvidenceManifest,
)

_LOGGER = logging.getLogger("Enai.ReportPlanner")


class ReportPlanEvidenceError(ValueError):
    """The plan references evidence outside its immutable manifest."""


class ReportPlanSemanticError(ValueError):
    """The plan conflicts with the authoritative report-planning context."""


def _fallback_planning_context(query: str) -> ReportPlanningContext:
    return ReportPlanningContext(
        contract_version="report-planning-context-v1",
        intent=ReportIntent.GENERAL,
        language_code="en",
        request_objective=str(query or "").strip(),
        requires_table=True,
        source="pipeline_fallback",
    )


def validate_report_plan_semantics(
    plan: ReportPlan,
    planning_context: ReportPlanningContext,
) -> None:
    if plan.intent is not planning_context.intent:
        raise ReportPlanSemanticError(
            "Report plan intent does not match its planning context."
        )
    if plan.language_code != planning_context.language_code:
        raise ReportPlanSemanticError(
            "Report plan language does not match its planning context."
        )


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
    validate_report_plan_evidence(repaired, manifest)
    return repaired


_AGGREGATE_CITING_SECTION_KINDS = frozenset(
    {
        ReportSectionKind.EXECUTIVE_SUMMARY,
        ReportSectionKind.CONCLUSION,
    }
)


def _ensure_summary_sections_cite_statistics(
    plan: ReportPlan,
    manifest: ReportEvidenceManifest,
) -> ReportPlan:
    """Give summary sections the evidence their aggregates actually live in.

    Column aggregates and seasonal statistics are projected into the statistics
    item, never into table rows. A section holding only table evidence
    therefore cannot state an average at all: expressing one as a derived claim
    needs an operand per row, and the contract caps operands at 32. On the
    2026-07-27 trace c7823cc9 the executive summary was assigned one table ref
    over a 138-row price series, so every attempt failed DERIVED_CLAIM_INVALID
    until the job exhausted its retries.
    """

    item_by_ref = manifest.item_by_ref()
    statistics_refs = [
        item.evidence_ref
        for item in manifest.items
        if item.kind is ReportEvidenceKind.STATISTICS
    ]
    if not statistics_refs:
        return plan

    payload = plan.model_dump(mode="json")
    granted = False
    for section in payload["sections"]:
        if ReportSectionKind(section["kind"]) not in _AGGREGATE_CITING_SECTION_KINDS:
            continue
        refs = section["required_evidence_refs"]
        if any(
            ref in item_by_ref
            and item_by_ref[ref].kind is ReportEvidenceKind.STATISTICS
            for ref in refs
        ):
            continue
        if len(refs) >= 32:
            continue
        refs.append(statistics_refs[0])
        granted = True

    if not granted:
        return plan
    repaired = ReportPlan.model_validate(payload)
    validate_report_plan_evidence(repaired, manifest)
    _LOGGER.info(
        "Granted statistics evidence to report summary sections: manifest_id=%s",
        manifest.manifest_id,
    )
    return repaired


ReportPlanInvoker = Callable[
    [str, ReportEvidenceManifest, ReportPlanningContext],
    Any,
]


def plan_report(
    query: str,
    manifest: ReportEvidenceManifest,
    *,
    planning_context: ReportPlanningContext | None = None,
    invoke_model: ReportPlanInvoker | None = None,
    repair_model: Callable[..., Any] | None = None,
) -> ReportPlan:
    planning_context = planning_context or _fallback_planning_context(query)
    if invoke_model is None:
        from core.llm import llm_plan_report

        invoke_model = llm_plan_report

    def _payload_of(raw_plan: Any) -> Any:
        return (
            raw_plan.model_dump(mode="json")
            if isinstance(raw_plan, ReportPlan)
            else raw_plan
        )

    def _materialize(raw_plan: Any) -> ReportPlan:
        plan = ReportPlan.model_validate(
            normalize_report_plan_semantics(
                normalize_report_plan_word_budget(_payload_of(raw_plan)),
                planning_context,
            )
        )
        validate_report_plan_semantics(plan, planning_context)
        return plan

    raw_plan = invoke_model(query, manifest, planning_context)
    try:
        plan = _materialize(raw_plan)
    except (ValidationError, ReportPlanSemanticError) as exc:
        # One bounded in-place repair. A schema or semantic slip otherwise
        # discards the entire evidence pipeline run as REPORT_PLAN_INVALID.
        error_code = (
            "PLAN_SEMANTIC_MISMATCH"
            if isinstance(exc, ReportPlanSemanticError)
            else "PLAN_SCHEMA_INVALID"
        )
        _LOGGER.warning(
            "Report plan rejected before repair: manifest_id=%s error_code=%s",
            manifest.manifest_id,
            error_code,
        )
        effective_repair = repair_model
        if effective_repair is None:
            from core.llm import llm_repair_report_plan

            effective_repair = llm_repair_report_plan
        plan = _materialize(
            effective_repair(
                query,
                manifest,
                planning_context,
                _payload_of(raw_plan),
                [error_code],
            )
        )
    try:
        validate_report_plan_evidence(plan, manifest)
    except ReportPlanEvidenceError:
        plan = _repair_report_plan_evidence(plan, manifest)
        _LOGGER.warning(
            "Stabilized report plan evidence bindings: manifest_id=%s",
            manifest.manifest_id,
        )
    return _ensure_summary_sections_cite_statistics(plan, manifest)
