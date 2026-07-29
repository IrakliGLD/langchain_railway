"""Whole-document generation with global validation and one targeted repair."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from agent.report_grounding import build_evidence_grounding_index
from agent.report_sections import count_section_words, validate_report_section
from contracts.report import (
    report_aggregate_word_bounds,
    report_section_validation_word_bounds,
)
from contracts.report_document import (
    ReportDocumentDraft,
    ReportDocumentPlan,
    ReportDocumentRepair,
    ReportDocumentSectionRole,
    ReportDocumentValidation,
)
from contracts.report_evidence import ReportEvidenceManifest
from contracts.report_research import (
    ReportEvidencePacket,
    ReportResearchPlan,
)
from contracts.report_sections import ReportSectionDraft

DocumentWriter = Callable[..., Any]
DocumentRepairer = Callable[..., Any]
_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_LOGGER = logging.getLogger("Enai.ReportDocument")


class ReportDocumentGenerationError(RuntimeError):
    """The whole-document draft remained invalid after its repair budget."""

    def __init__(self, validation: ReportDocumentValidation) -> None:
        self.validation = validation
        codes = list(validation.document_errors)
        codes.extend(
            code
            for errors in validation.section_errors.values()
            for code in errors
        )
        super().__init__(
            "Report document failed validation: "
            + ",".join(dict.fromkeys(codes))
        )


def _log_document_diagnostic(
    *,
    event: str,
    plan: ReportDocumentPlan,
    validation: ReportDocumentValidation,
    draft: ReportDocumentDraft | None,
    repair_section_ids: Sequence[str],
    pre_normalization_role_section_ids: dict[
        str,
        list[str],
    ] | None = None,
    role_normalization_applied: bool = False,
    baseline_draft: ReportDocumentDraft | None = None,
) -> None:
    minimum_words, maximum_words = report_aggregate_word_bounds(
        [section.target_words for section in plan.sections]
    )
    section_by_id = (
        {
            section.section_id: section
            for section in draft.generation_order_sections()
        }
        if draft is not None
        else {}
    )
    section_word_counts = {
        section_id: count_section_words(section.content_markdown)
        for section_id, section in section_by_id.items()
    }
    baseline_word_counts = (
        {
            section.section_id: count_section_words(
                section.content_markdown
            )
            for section in baseline_draft.generation_order_sections()
        }
        if baseline_draft is not None
        else {}
    )
    section_word_deltas = {
        section_id: word_count - baseline_word_counts[section_id]
        for section_id, word_count in section_word_counts.items()
        if section_id in baseline_word_counts
    }
    section_bounds = {
        section.section_id: {
            "minimum_words": report_section_validation_word_bounds(
                section.target_words
            )[0],
            "maximum_words": report_section_validation_word_bounds(
                section.target_words
            )[1],
        }
        for section in plan.sections
    }
    _LOGGER.info(
        "REPORT_DOCUMENT_DIAGNOSTIC %s",
        json.dumps(
            {
                "document_error_codes": validation.document_errors,
                "event": event,
                "expected_role_section_ids": {
                    role.value: _expected_role_ids(plan, role)
                    for role in ReportDocumentSectionRole
                },
                "failing_section_ids": sorted(
                    validation.section_errors
                ),
                "maximum_words": maximum_words,
                "minimum_words": minimum_words,
                "repair_section_ids": list(repair_section_ids),
                "role_normalization_applied": (
                    role_normalization_applied
                ),
                "section_bounds": section_bounds,
                "section_error_codes": validation.section_errors,
                "section_word_deltas": section_word_deltas,
                "section_word_counts": section_word_counts,
                "pre_normalization_role_section_ids": (
                    pre_normalization_role_section_ids or {}
                ),
                "word_count": validation.word_count,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _expected_role_ids(
    plan: ReportDocumentPlan,
    role: ReportDocumentSectionRole,
) -> list[str]:
    return [
        section.section_id
        for section in plan.sections
        if section.role is role
    ]


def _actual_role_ids(
    draft: ReportDocumentDraft,
    role: ReportDocumentSectionRole,
) -> list[str]:
    if role is ReportDocumentSectionRole.ANALYSIS:
        return [
            section.section_id for section in draft.analytical_sections
        ]
    if role is ReportDocumentSectionRole.EXECUTIVE_SUMMARY:
        return [draft.executive_summary.section_id]
    if role is ReportDocumentSectionRole.LIMITATIONS:
        return [draft.limitations_section.section_id]
    section = (
        draft.implications_section
        if role is ReportDocumentSectionRole.IMPLICATIONS
        else draft.conclusion_section
    )
    return [] if section is None else [section.section_id]


def _actual_role_section_ids(
    draft: ReportDocumentDraft,
) -> dict[str, list[str]]:
    return {
        role.value: _actual_role_ids(draft, role)
        for role in ReportDocumentSectionRole
    }


def _paragraph_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) > 2
    }


def _repeated_section_ids(
    draft: ReportDocumentDraft,
) -> set[str]:
    repeated: set[str] = set()
    seen: list[tuple[str, set[str]]] = []
    for section in draft.generation_order_sections():
        for paragraph in section.paragraphs:
            tokens = _paragraph_tokens(paragraph.text)
            if len(tokens) < 20:
                continue
            for _earlier_id, earlier_tokens in seen:
                union = tokens | earlier_tokens
                similarity = (
                    len(tokens & earlier_tokens) / len(union)
                    if union
                    else 0.0
                )
                if similarity >= 0.9:
                    repeated.add(section.section_id)
                    break
            seen.append((section.section_id, tokens))
    return repeated


def validate_report_document(
    draft: ReportDocumentDraft,
    plan: ReportDocumentPlan,
    manifest: ReportEvidenceManifest,
    research_plan: ReportResearchPlan,
) -> ReportDocumentValidation:
    """Validate sections, role binding, numbers, repetition, and total length."""

    section_errors: dict[str, list[str]] = {}
    document_errors: list[str] = []

    def add_section_error(section_id: str, code: str) -> None:
        section_errors.setdefault(section_id, [])
        if code not in section_errors[section_id]:
            section_errors[section_id].append(code)

    if (
        draft.query_digest != plan.query_digest
        or draft.evidence_manifest_id != plan.evidence_manifest_id
        or draft.coverage_status != plan.coverage_status
        or plan.evidence_manifest_id != manifest.manifest_id
        or plan.query_digest != research_plan.query_digest
    ):
        document_errors.append("DOCUMENT_IDENTITY_MISMATCH")

    for role in ReportDocumentSectionRole:
        if _actual_role_ids(draft, role) != _expected_role_ids(plan, role):
            document_errors.append("SECTION_ROLE_MISMATCH")
            break

    draft_by_id = {
        section.section_id: section
        for section in draft.generation_order_sections()
    }
    plan_by_id = {
        section.section_id: section for section in plan.sections
    }
    if set(draft_by_id) != set(plan_by_id):
        document_errors.append("SECTION_SET_MISMATCH")

    total_words = 0
    grounding_index = build_evidence_grounding_index(
        manifest.item_by_ref(),
        {
            evidence_ref
            for section in plan.sections
            for evidence_ref in section.required_evidence_refs
        },
    )
    for section_id in sorted(set(draft_by_id) & set(plan_by_id)):
        draft_section = draft_by_id[section_id]
        section_spec = plan_by_id[section_id]
        validation = validate_report_section(
            draft_section,
            section_spec,
            manifest,
            evidence_facts_by_ref=grounding_index,
        )
        total_words += validation.word_count
        for code in validation.error_codes:
            add_section_error(section_id, code)

        if section_spec.role is ReportDocumentSectionRole.ANALYSIS:
            needs_numbers = any(
                track.requested_metrics
                for track in research_plan.tracks
                if track.track_id in section_spec.track_ids
            )
            item_by_ref = manifest.item_by_ref()
            available_numeric_coordinates = sum(
                1
                for evidence_ref in section_spec.required_evidence_refs
                if (
                    evidence_ref in item_by_ref
                    and item_by_ref[evidence_ref].kind.value == "table"
                )
                for row in item_by_ref[evidence_ref].rows
                for column, value in row.items()
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and bool(
                        item_by_ref[evidence_ref].unit_by_column.get(
                            column
                        )
                    )
                )
            )
            required_numeric_claims = (
                min(2, available_numeric_coordinates)
                if available_numeric_coordinates
                else 1
            )
            numeric_claim_count = sum(
                len(paragraph.direct_claims)
                + len(paragraph.derived_claims)
                for paragraph in draft_section.paragraphs
            )
            if (
                needs_numbers
                and numeric_claim_count < required_numeric_claims
            ):
                add_section_error(
                    section_id,
                    "NUMERIC_FINDING_MISSING",
                )

    for section_id in _repeated_section_ids(draft):
        add_section_error(section_id, "CROSS_SECTION_REPETITION")

    minimum_words, maximum_words = report_aggregate_word_bounds(
        [section.target_words for section in plan.sections]
    )
    if total_words < minimum_words:
        document_errors.append("DOCUMENT_WORD_COUNT_TOO_SHORT")
    elif total_words > maximum_words:
        document_errors.append("DOCUMENT_WORD_COUNT_TOO_LONG")

    document_errors = list(dict.fromkeys(document_errors))
    return ReportDocumentValidation(
        contract_version="report-document-validation-v1",
        valid=not section_errors and not document_errors,
        section_errors=section_errors,
        document_errors=document_errors,
        word_count=total_words,
    )


def _merge_repairs(
    draft: ReportDocumentDraft,
    repair: ReportDocumentRepair,
) -> ReportDocumentDraft:
    replacements = {
        section.section_id: section for section in repair.sections
    }
    payload = draft.model_dump(mode="json")

    def replace(section_payload):
        if section_payload is None:
            return None
        return (
            replacements[section_payload["section_id"]].model_dump(
                mode="json"
            )
            if section_payload["section_id"] in replacements
            else section_payload
        )

    payload["analytical_sections"] = [
        replace(section)
        for section in payload["analytical_sections"]
    ]
    for field in (
        "implications_section",
        "limitations_section",
        "conclusion_section",
        "executive_summary",
    ):
        payload[field] = replace(payload[field])
    return ReportDocumentDraft.model_validate(payload)


def _document_from_sections(
    plan: ReportDocumentPlan,
    sections: Sequence[ReportSectionDraft],
) -> ReportDocumentDraft:
    section_by_id = {
        section.section_id: section for section in sections
    }

    def sections_for(
        role: ReportDocumentSectionRole,
    ) -> list[ReportSectionDraft]:
        return [
            section_by_id[section_id]
            for section_id in _expected_role_ids(plan, role)
        ]

    analytical_sections = sections_for(
        ReportDocumentSectionRole.ANALYSIS
    )
    implications_sections = sections_for(
        ReportDocumentSectionRole.IMPLICATIONS
    )
    limitations_sections = sections_for(
        ReportDocumentSectionRole.LIMITATIONS
    )
    conclusion_sections = sections_for(
        ReportDocumentSectionRole.CONCLUSION
    )
    summary_sections = sections_for(
        ReportDocumentSectionRole.EXECUTIVE_SUMMARY
    )
    return ReportDocumentDraft(
        contract_version="report-document-draft-v1",
        query_digest=plan.query_digest,
        evidence_manifest_id=plan.evidence_manifest_id,
        coverage_status=plan.coverage_status,
        analytical_sections=analytical_sections,
        implications_section=(
            implications_sections[0] if implications_sections else None
        ),
        limitations_section=limitations_sections[0],
        conclusion_section=(
            conclusion_sections[0] if conclusion_sections else None
        ),
        executive_summary=summary_sections[0],
    )


def _normalize_document_roles(
    draft: ReportDocumentDraft,
    plan: ReportDocumentPlan,
) -> tuple[ReportDocumentDraft, bool]:
    sections = draft.generation_order_sections()
    if {section.section_id for section in sections} != {
        section.section_id for section in plan.sections
    }:
        return draft, False
    roles_normalized = any(
        _actual_role_ids(draft, role) != _expected_role_ids(plan, role)
        for role in ReportDocumentSectionRole
    )
    return _document_from_sections(plan, sections), roles_normalized


def _repair_moves_word_counts_in_required_direction(
    draft: ReportDocumentDraft,
    repair: ReportDocumentRepair,
    validation: ReportDocumentValidation,
) -> bool:
    original_by_id = {
        section.section_id: section
        for section in draft.generation_order_sections()
    }
    for replacement in repair.sections:
        original = original_by_id[replacement.section_id]
        original_words = count_section_words(original.content_markdown)
        replacement_words = count_section_words(
            replacement.content_markdown
        )
        error_codes = set(
            validation.section_errors.get(replacement.section_id, [])
        )
        if (
            "WORD_COUNT_TOO_LONG" in error_codes
            and replacement_words >= original_words
        ):
            return False
        if (
            "WORD_COUNT_TOO_SHORT" in error_codes
            and replacement_words <= original_words
        ):
            return False
    return True


def generate_report_document(
    query: str,
    plan: ReportDocumentPlan,
    research_plan: ReportResearchPlan,
    manifest: ReportEvidenceManifest,
    packets: Sequence[ReportEvidencePacket],
    *,
    write_document: DocumentWriter | None = None,
    repair_sections: DocumentRepairer | None = None,
    allow_repair: bool = True,
) -> ReportDocumentDraft:
    """Use one whole-document call and at most one targeted repair call."""

    if write_document is None:
        from core.llm import llm_write_report_document

        write_document = llm_write_report_document
    raw_draft = write_document(
        query,
        plan,
        research_plan,
        manifest,
        list(packets),
    )
    pre_normalization_role_section_ids: dict[str, list[str]] | None = None
    role_normalization_applied = False
    try:
        draft: ReportDocumentDraft | None = (
            raw_draft
            if isinstance(raw_draft, ReportDocumentDraft)
            else ReportDocumentDraft.model_validate(raw_draft)
        )
    except ValidationError:
        draft = None
        validation = ReportDocumentValidation(
            contract_version="report-document-validation-v1",
            valid=False,
            section_errors={},
            document_errors=["DOCUMENT_SCHEMA_INVALID"],
            word_count=0,
        )
    else:
        pre_normalization_role_section_ids = (
            _actual_role_section_ids(draft)
        )
        draft, role_normalization_applied = (
            _normalize_document_roles(draft, plan)
        )
        validation = validate_report_document(
            draft,
            plan,
            manifest,
            research_plan,
        )
        if role_normalization_applied:
            _log_document_diagnostic(
                event="roles_normalized",
                plan=plan,
                validation=validation,
                draft=draft,
                repair_section_ids=[],
                pre_normalization_role_section_ids=(
                    pre_normalization_role_section_ids
                ),
                role_normalization_applied=True,
            )
        if validation.valid:
            return draft
    if not allow_repair:
        raise ReportDocumentGenerationError(validation)

    structural_repair = draft is None or bool(
        set(validation.document_errors)
        - {
            "DOCUMENT_WORD_COUNT_TOO_SHORT",
            "DOCUMENT_WORD_COUNT_TOO_LONG",
        }
    )
    if structural_repair:
        invalid_section_ids = [
            section.section_id for section in plan.sections
        ]
    else:
        invalid_section_ids = list(validation.section_errors)
    if repair_sections is None:
        from core.llm import llm_repair_report_document_sections

        repair_sections = llm_repair_report_document_sections
    _log_document_diagnostic(
        event="repair_requested",
        plan=plan,
        validation=validation,
        draft=draft,
        repair_section_ids=invalid_section_ids,
        pre_normalization_role_section_ids=(
            pre_normalization_role_section_ids
        ),
        role_normalization_applied=role_normalization_applied,
    )
    raw_repair = repair_sections(
        query,
        plan,
        research_plan,
        manifest,
        list(packets),
        raw_draft if draft is None else draft,
        validation,
        section_ids=invalid_section_ids,
    )
    try:
        repair = (
            raw_repair
            if isinstance(raw_repair, ReportDocumentRepair)
            else ReportDocumentRepair.model_validate(raw_repair)
        )
    except ValidationError as exc:
        raise ReportDocumentGenerationError(validation) from exc
    if {section.section_id for section in repair.sections} != set(
        invalid_section_ids
    ):
        raise ReportDocumentGenerationError(validation)
    repair_moves_in_required_direction = (
        structural_repair
        or _repair_moves_word_counts_in_required_direction(
            draft,
            repair,
            validation,
        )
    )

    repaired_draft = (
        _document_from_sections(plan, repair.sections)
        if structural_repair
        else _merge_repairs(draft, repair)
    )
    repaired_validation = validate_report_document(
        repaired_draft,
        plan,
        manifest,
        research_plan,
    )
    _log_document_diagnostic(
        event=(
            "repair_direction_rejected"
            if not repair_moves_in_required_direction
            else (
                "repair_validated"
                if repaired_validation.valid
                else "repair_rejected"
            )
        ),
        plan=plan,
        validation=repaired_validation,
        draft=repaired_draft,
        repair_section_ids=invalid_section_ids,
        pre_normalization_role_section_ids=(
            pre_normalization_role_section_ids
        ),
        role_normalization_applied=role_normalization_applied,
        baseline_draft=draft,
    )
    if not repair_moves_in_required_direction:
        raise ReportDocumentGenerationError(validation)
    if not repaired_validation.valid:
        raise ReportDocumentGenerationError(repaired_validation)
    return repaired_draft
