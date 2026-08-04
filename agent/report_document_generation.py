"""Adaptive report generation with staged writing and bounded repair."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from agent.report_grounding import (
    build_evidence_grounding_index,
    normalize_repairable_derived_claims,
    select_grounded_paragraphs,
)
from agent.report_sections import count_section_words, validate_report_section
from contracts.report import report_section_validation_word_bounds
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
SectionBatchWriter = Callable[..., Any]
_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_LOGGER = logging.getLogger("Enai.ReportDocument")
_WORD_COUNT_CODES = {
    "WORD_COUNT_TOO_SHORT",
    "WORD_COUNT_TOO_LONG",
}


def report_document_recommended_word_bounds(
    plan: ReportDocumentPlan,
) -> tuple[int, int]:
    """Return evidence-aware advisory bounds for the complete document."""

    exhibit_count = plan.evidence_capacity.usable_exhibit_count
    if exhibit_count <= 1:
        minimum_ratio = 0.5
    elif exhibit_count == 2:
        minimum_ratio = 0.65
    elif exhibit_count == 3:
        minimum_ratio = 0.75
    else:
        minimum_ratio = 0.8
    return (
        math.floor(plan.target_words * minimum_ratio),
        math.ceil(plan.target_words * 1.2),
    )


class ReportDocumentGenerationError(RuntimeError):
    """The adaptive draft remained invalid after its repair budget."""

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
    sections: Sequence[ReportSectionDraft] | None = None,
    manifest: ReportEvidenceManifest | None = None,
) -> None:
    minimum_words, maximum_words = report_document_recommended_word_bounds(
        plan
    )
    # A generation batch carries only part of the plan, so it reports its
    # sections directly rather than as a draft.
    reported_sections = (
        draft.generation_order_sections()
        if draft is not None
        else (sections or [])
    )
    section_by_id = {
        section.section_id: section for section in reported_sections
    }
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
    # Report the bounds validation actually applied. The floor scales with the
    # rows a section can cite, so logging the flat one would describe a gate
    # that never ran.
    section_bounds = {
        section.section_id: {
            "minimum_words": report_section_validation_word_bounds(
                section.target_words,
                evidence_row_count=(
                    None
                    if manifest is None
                    else manifest.assigned_row_count(
                        section.required_evidence_refs
                    )
                ),
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
                "document_warning_codes": validation.document_warnings,
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
                "section_warning_codes": validation.section_warnings,
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
    plan: ReportDocumentPlan,
    role: ReportDocumentSectionRole,
) -> list[str]:
    return [
        draft.sections[index].section_id
        for index, section in enumerate(plan.sections)
        if section.role is role and index < len(draft.sections)
    ]


def _actual_role_section_ids(
    draft: ReportDocumentDraft,
    plan: ReportDocumentPlan,
) -> dict[str, list[str]]:
    return {
        role.value: _actual_role_ids(draft, plan, role)
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
    section_warnings: dict[str, list[str]] = {}
    document_errors: list[str] = []
    document_warnings: list[str] = []

    def add_section_error(section_id: str, code: str) -> None:
        section_errors.setdefault(section_id, [])
        if code not in section_errors[section_id]:
            section_errors[section_id].append(code)

    def add_section_warning(section_id: str, code: str) -> None:
        section_warnings.setdefault(section_id, [])
        if code not in section_warnings[section_id]:
            section_warnings[section_id].append(code)

    if (
        draft.query_digest != plan.query_digest
        or draft.evidence_manifest_id != plan.evidence_manifest_id
        or draft.coverage_status != plan.coverage_status
        or plan.evidence_manifest_id != manifest.manifest_id
        or plan.query_digest != research_plan.query_digest
    ):
        document_errors.append("DOCUMENT_IDENTITY_MISMATCH")

    for role in ReportDocumentSectionRole:
        if _actual_role_ids(
            draft,
            plan,
            role,
        ) != _expected_role_ids(plan, role):
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
            if code in _WORD_COUNT_CODES:
                add_section_warning(section_id, code)
            else:
                add_section_error(section_id, code)

        if _analysis_numeric_finding_missing(
            draft_section,
            section_spec,
            manifest,
            research_plan,
        ):
            add_section_error(section_id, "NUMERIC_FINDING_MISSING")

    for section_id in _repeated_section_ids(draft):
        add_section_error(section_id, "CROSS_SECTION_REPETITION")

    minimum_words, maximum_words = report_document_recommended_word_bounds(
        plan
    )
    if total_words < minimum_words:
        document_warnings.append("DOCUMENT_WORD_COUNT_TOO_SHORT")
    elif total_words > maximum_words:
        document_warnings.append("DOCUMENT_WORD_COUNT_TOO_LONG")

    document_errors = list(dict.fromkeys(document_errors))
    document_warnings = list(dict.fromkeys(document_warnings))
    return ReportDocumentValidation(
        contract_version="report-document-validation-v1",
        valid=not section_errors and not document_errors,
        section_errors=section_errors,
        document_errors=document_errors,
        section_warnings=section_warnings,
        document_warnings=document_warnings,
        word_count=total_words,
    )


def _merge_repairs(
    draft: ReportDocumentDraft,
    repair: ReportDocumentRepair,
) -> ReportDocumentDraft:
    replacements = {
        section.section_id: section for section in repair.sections
    }
    return draft.model_copy(
        update={
            "sections": [
                replacements.get(section.section_id, section)
                for section in draft.sections
            ]
        }
    )


def _document_from_sections(
    plan: ReportDocumentPlan,
    sections: Sequence[ReportSectionDraft],
) -> ReportDocumentDraft:
    section_by_id = {
        section.section_id: section for section in sections
    }

    return ReportDocumentDraft(
        contract_version="report-document-draft-v1",
        query_digest=plan.query_digest,
        evidence_manifest_id=plan.evidence_manifest_id,
        coverage_status=plan.coverage_status,
        sections=[
            section_by_id[section.section_id]
            for section in plan.sections
        ],
    )


def _materialize_section_batch(
    raw_batch: Any,
    plan: ReportDocumentPlan,
    manifest: ReportEvidenceManifest,
    *,
    section_ids: Sequence[str],
    research_plan: ReportResearchPlan | None = None,
) -> tuple[list[ReportSectionDraft] | None, ReportDocumentValidation]:
    expected_ids = list(section_ids)
    try:
        batch = (
            raw_batch
            if isinstance(raw_batch, ReportDocumentRepair)
            else ReportDocumentRepair.model_validate(raw_batch)
        )
    except ValidationError:
        return None, ReportDocumentValidation(
            contract_version="report-document-validation-v1",
            valid=False,
            document_errors=["DOCUMENT_SCHEMA_INVALID"],
            word_count=0,
        )

    actual_ids = [section.section_id for section in batch.sections]
    if actual_ids != expected_ids:
        return None, ReportDocumentValidation(
            contract_version="report-document-validation-v1",
            valid=False,
            document_errors=["SECTION_SET_MISMATCH"],
            word_count=sum(
                count_section_words(section.content_markdown)
                for section in batch.sections
            ),
        )

    plan_by_id = {section.section_id: section for section in plan.sections}
    section_errors: dict[str, list[str]] = {}
    section_warnings: dict[str, list[str]] = {}
    word_count = 0
    normalized_sections: list[ReportSectionDraft] = []
    item_by_ref = manifest.item_by_ref()
    for draft_section in batch.sections:
        draft_section, normalized_count = normalize_repairable_derived_claims(
            draft_section,
            item_by_ref,
        )
        normalized_sections.append(draft_section)
        if normalized_count:
            _LOGGER.info(
                "REPORT_DERIVED_CLAIM_NORMALIZED section_id=%s count=%d",
                draft_section.section_id,
                normalized_count,
            )
        section_spec = plan_by_id[draft_section.section_id]
        validation = validate_report_section(
            draft_section,
            section_spec,
            manifest,
        )
        word_count += validation.word_count
        blocking_codes = [
            code
            for code in validation.error_codes
            if code not in _WORD_COUNT_CODES
        ]
        warning_codes = [
            code
            for code in validation.error_codes
            if code in _WORD_COUNT_CODES
        ]
        # The document gate rejects an analysis section with too few numbers,
        # so the gate that accepts one has to see the same thing. Without this
        # the single repair call is spent on grounding, and the shortfall
        # surfaces afterwards with no budget left (job 522b9b73).
        if research_plan is not None and _analysis_numeric_finding_missing(
            draft_section,
            section_spec,
            manifest,
            research_plan,
        ):
            blocking_codes = [*blocking_codes, "NUMERIC_FINDING_MISSING"]
        if blocking_codes:
            section_errors[draft_section.section_id] = blocking_codes
        if warning_codes:
            section_warnings[draft_section.section_id] = warning_codes
    return normalized_sections, ReportDocumentValidation(
        contract_version="report-document-validation-v1",
        valid=not section_errors,
        section_errors=section_errors,
        section_warnings=section_warnings,
        word_count=word_count,
    )


def _analysis_numeric_finding_missing(
    draft_section: ReportSectionDraft,
    section_spec: Any,
    manifest: ReportEvidenceManifest,
    research_plan: ReportResearchPlan,
) -> bool:
    """Return whether an analysis section carries too few verified numbers.

    Single authority for the rule, so the document gate and the grounded-subset
    guard cannot disagree about what counts as an analysis section worth
    shipping.
    """

    if section_spec.role is not ReportDocumentSectionRole.ANALYSIS:
        return False
    needs_numbers = any(
        track.requested_metrics
        for track in research_plan.tracks
        if track.track_id in section_spec.track_ids
    )
    if not needs_numbers:
        return False
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
            and bool(item_by_ref[evidence_ref].unit_by_column.get(column))
        )
    )
    required_numeric_claims = (
        min(2, available_numeric_coordinates)
        if available_numeric_coordinates
        else 1
    )
    numeric_claim_count = sum(
        len(paragraph.direct_claims) + len(paragraph.derived_claims)
        for paragraph in draft_section.paragraphs
    )
    return numeric_claim_count < required_numeric_claims


def _sections_or_grounded_subset(
    sections: list[ReportSectionDraft] | None,
    validation: ReportDocumentValidation,
    plan: ReportDocumentPlan,
    manifest: ReportEvidenceManifest,
    *,
    section_ids: Sequence[str],
    stage: str,
    research_plan: ReportResearchPlan,
) -> tuple[list[ReportSectionDraft] | None, ReportDocumentValidation]:
    """Ship the grounded subset when the repair budget did not converge.

    Called only where the pipeline is otherwise about to give up. Dropping the
    sentences the evidence cannot support costs the reader a shorter section;
    raising costs them the whole report, and REPORT_DOCUMENT_INVALID is not
    retryable. Fails closed: a subset that still does not validate is returned
    as-is so the caller raises exactly as before.
    """

    if sections is None or validation.valid:
        return sections, validation
    item_by_ref = manifest.item_by_ref()
    salvaged_sections: list[ReportSectionDraft] = []
    dropped_total = 0
    for section in sections:
        salvaged, dropped = select_grounded_paragraphs(section, item_by_ref)
        salvaged_sections.append(salvaged)
        dropped_total += dropped
    if not dropped_total:
        return sections, validation
    plan_by_id = {section.section_id: section for section in plan.sections}
    # A salvage that strips an analysis section of every number leaves prose
    # that reads like a report and reports nothing. Refuse it here rather than
    # letting it travel to the document gate as NUMERIC_FINDING_MISSING.
    if any(
        _analysis_numeric_finding_missing(
            section,
            plan_by_id[section.section_id],
            manifest,
            research_plan,
        )
        for section in salvaged_sections
        if section.section_id in plan_by_id
    ):
        return sections, validation
    subset_sections, subset_validation = _materialize_section_batch(
        ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=salvaged_sections,
        ),
        plan,
        manifest,
        section_ids=section_ids,
        research_plan=research_plan,
    )
    _LOGGER.info(
        "REPORT_GROUNDED_SUBSET %s",
        json.dumps(
            {
                "applied": subset_validation.valid,
                "dropped_sentence_count": dropped_total,
                "recovered_section_ids": sorted(
                    set(section_ids) - set(subset_validation.section_errors)
                ),
                "residual_error_codes": sorted(
                    {
                        code
                        for errors in subset_validation.section_errors.values()
                        for code in errors
                    }
                ),
                "stage": stage,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    if not subset_validation.valid:
        return sections, validation
    return subset_sections, subset_validation


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
        _actual_role_ids(draft, plan, role)
        != _expected_role_ids(plan, role)
        for role in ReportDocumentSectionRole
    )
    return _document_from_sections(plan, sections), roles_normalized


def _repair_section_batch(
    query: str,
    plan: ReportDocumentPlan,
    research_plan: ReportResearchPlan,
    manifest: ReportEvidenceManifest,
    packets: Sequence[ReportEvidencePacket],
    sections: list[ReportSectionDraft] | None,
    raw_batch: Any,
    validation: ReportDocumentValidation,
    repair_sections: DocumentRepairer,
    *,
    section_ids: Sequence[str],
) -> tuple[list[ReportSectionDraft] | None, ReportDocumentValidation]:
    """Spend one repair call on a rejected batch and re-validate the result.

    A batch that fails structurally (schema or section-set) has no usable
    sections, so the whole requested batch is rewritten from the raw payload
    the writer returned; otherwise only the sections the validator rejected
    are replaced.
    """

    structural = sections is None or bool(validation.document_errors)
    invalid_ids = (
        list(section_ids) if structural else list(validation.section_errors)
    )
    if not invalid_ids:
        return sections, validation
    _log_document_diagnostic(
        manifest=manifest,
        event="batch_repair_requested",
        plan=plan,
        validation=validation,
        draft=None,
        # A batch holds only part of the plan, so it cannot form a whole draft.
        # Report its sections directly: otherwise the diagnostic shows empty
        # word counts exactly when they would explain the rejection.
        sections=sections,
        repair_section_ids=invalid_ids,
    )
    raw_repair = repair_sections(
        query,
        plan,
        research_plan,
        manifest,
        list(packets),
        raw_batch if sections is None else list(sections),
        validation,
        section_ids=invalid_ids,
    )
    try:
        repair = (
            raw_repair
            if isinstance(raw_repair, ReportDocumentRepair)
            else ReportDocumentRepair.model_validate(raw_repair)
        )
    except ValidationError:
        return None, validation
    if {section.section_id for section in repair.sections} != set(invalid_ids):
        return None, validation
    replacements = {
        section.section_id: section for section in repair.sections
    }
    merged = (
        list(repair.sections)
        if structural
        else [
            replacements.get(section.section_id, section)
            for section in sections or []
        ]
    )
    return _materialize_section_batch(
        ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=merged,
        ),
        plan,
        manifest,
        section_ids=section_ids,
        research_plan=research_plan,
    )


def generate_report_document(
    query: str,
    plan: ReportDocumentPlan,
    research_plan: ReportResearchPlan,
    manifest: ReportEvidenceManifest,
    packets: Sequence[ReportEvidencePacket],
    *,
    write_document: DocumentWriter | None = None,
    write_analysis_sections: SectionBatchWriter | None = None,
    write_synthesis_sections: SectionBatchWriter | None = None,
    repair_sections: DocumentRepairer | None = None,
    allow_repair: bool = True,
) -> ReportDocumentDraft:
    """Generate an adaptive document and use at most one targeted repair."""

    if (
        write_document is not None
        or plan.profile.value == "compact"
    ):
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
    else:
        if write_analysis_sections is None:
            from core.llm import llm_write_report_analysis_sections

            write_analysis_sections = llm_write_report_analysis_sections
        if write_synthesis_sections is None:
            from core.llm import llm_write_report_synthesis_sections

            write_synthesis_sections = llm_write_report_synthesis_sections
        analysis_ids = [
            section.section_id
            for section in plan.sections
            if section.role is ReportDocumentSectionRole.ANALYSIS
        ]
        synthesis_ids = [
            section.section_id
            for section in plan.sections
            if section.role is not ReportDocumentSectionRole.ANALYSIS
        ]
        if repair_sections is None:
            from core.llm import llm_repair_report_document_sections

            repair_sections = llm_repair_report_document_sections
        raw_analysis = write_analysis_sections(
            query,
            plan,
            research_plan,
            manifest,
            list(packets),
            section_ids=analysis_ids,
        )
        analysis_sections, analysis_validation = _materialize_section_batch(
            raw_analysis,
            plan,
            manifest,
            section_ids=analysis_ids,
            research_plan=research_plan,
        )
        if analysis_sections is None or not analysis_validation.valid:
            if not allow_repair:
                analysis_sections, analysis_validation = (
                    _sections_or_grounded_subset(
                        analysis_sections,
                        analysis_validation,
                        plan,
                        manifest,
                        section_ids=analysis_ids,
                        stage="analysis_unrepaired",
                        research_plan=research_plan,
                    )
                )
                if analysis_sections is None or not analysis_validation.valid:
                    raise ReportDocumentGenerationError(analysis_validation)
            # Whichever stage spends the repair, the later gates short-circuit:
            # the processor's budget reserves exactly one repair call.
            allow_repair = False
            analysis_sections, analysis_validation = _repair_section_batch(
                query,
                plan,
                research_plan,
                manifest,
                packets,
                analysis_sections,
                raw_analysis,
                analysis_validation,
                repair_sections,
                section_ids=analysis_ids,
            )
            analysis_sections, analysis_validation = (
                _sections_or_grounded_subset(
                    analysis_sections,
                    analysis_validation,
                    plan,
                    manifest,
                    section_ids=analysis_ids,
                    stage="analysis_repaired",
                    research_plan=research_plan,
                )
            )
            if analysis_sections is None or not analysis_validation.valid:
                raise ReportDocumentGenerationError(analysis_validation)
        raw_synthesis = write_synthesis_sections(
            query,
            plan,
            research_plan,
            manifest,
            list(packets),
            analysis_sections=analysis_sections,
            section_ids=synthesis_ids,
        )
        synthesis_sections, synthesis_validation = _materialize_section_batch(
            raw_synthesis,
            plan,
            manifest,
            section_ids=synthesis_ids,
            research_plan=research_plan,
        )
        if synthesis_sections is None or not synthesis_validation.valid:
            if not allow_repair:
                synthesis_sections, synthesis_validation = (
                    _sections_or_grounded_subset(
                        synthesis_sections,
                        synthesis_validation,
                        plan,
                        manifest,
                        section_ids=synthesis_ids,
                        stage="synthesis_unrepaired",
                        research_plan=research_plan,
                    )
                )
                if (
                    synthesis_sections is None
                    or not synthesis_validation.valid
                ):
                    raise ReportDocumentGenerationError(synthesis_validation)
            allow_repair = False
            synthesis_sections, synthesis_validation = _repair_section_batch(
                query,
                plan,
                research_plan,
                manifest,
                packets,
                synthesis_sections,
                raw_synthesis,
                synthesis_validation,
                repair_sections,
                section_ids=synthesis_ids,
            )
            synthesis_sections, synthesis_validation = (
                _sections_or_grounded_subset(
                    synthesis_sections,
                    synthesis_validation,
                    plan,
                    manifest,
                    section_ids=synthesis_ids,
                    stage="synthesis_repaired",
                    research_plan=research_plan,
                )
            )
            if synthesis_sections is None or not synthesis_validation.valid:
                raise ReportDocumentGenerationError(synthesis_validation)
        raw_draft = _document_from_sections(
            plan,
            [*analysis_sections, *synthesis_sections],
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
            _actual_role_section_ids(draft, plan)
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
                manifest=manifest,
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

    structural_repair = draft is None or bool(validation.document_errors)
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
        manifest=manifest,
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
        manifest=manifest,
        event=(
            "repair_validated"
            if repaired_validation.valid
            else "repair_rejected"
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
    if not repaired_validation.valid:
        raise ReportDocumentGenerationError(repaired_validation)
    return repaired_draft
