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
    build_ungrounded_claim_repair_hints,
    drop_unrendered_claims,
    normalize_repairable_derived_claims,
    select_grounded_paragraphs,
)
from agent.report_sections import (
    count_section_words,
    uncited_required_evidence_refs,
    validate_report_section,
)
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
from utils.validation_diagnostics import (
    validation_error_locations,
    validation_error_rules,
)

DocumentWriter = Callable[..., Any]
DocumentRepairer = Callable[..., Any]
SectionBatchWriter = Callable[..., Any]
_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_LOGGER = logging.getLogger("Enai.ReportDocument")
_LONG_WORD_COUNT_CODE = "WORD_COUNT_TOO_LONG"
_SHORT_SECTION_CODE = "WORD_COUNT_TOO_SHORT"
_SHORT_DOCUMENT_CODE = "DOCUMENT_WORD_COUNT_TOO_SHORT"
_STRUCTURAL_DOCUMENT_ERROR_CODES = {
    "DOCUMENT_IDENTITY_MISMATCH",
    "DOCUMENT_SCHEMA_INVALID",
    "SECTION_ROLE_MISMATCH",
    "SECTION_SET_MISMATCH",
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


def _named_section_offenders(
    plan: ReportDocumentPlan,
    validation: ReportDocumentValidation,
    section_by_id: dict[str, ReportSectionDraft],
    manifest: ReportEvidenceManifest | None,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Name the refs left uncited and the values left unbacked.

    REQUIRED_EVIDENCE_NOT_USED and UNGROUNDED_NUMERIC_CLAIM each name a section
    and nothing else, so a run that spends its whole repair budget on them
    cannot be diagnosed from its log. Both offenders are derived from the same
    functions the validator and the repair prompt use, and only for the
    sections actually carrying the code, so a valid document pays nothing.
    """

    spec_by_id = {section.section_id: section for section in plan.sections}
    item_by_ref = {} if manifest is None else manifest.item_by_ref()
    uncited: dict[str, list[str]] = {}
    ungrounded_sections: list[ReportSectionDraft] = []
    for section_id, codes in validation.section_errors.items():
        section = section_by_id.get(section_id)
        if section is None:
            continue
        spec = spec_by_id.get(section_id)
        if "REQUIRED_EVIDENCE_NOT_USED" in codes and spec is not None:
            missing = uncited_required_evidence_refs(section, spec)
            if missing:
                uncited[section_id] = missing
        if "UNGROUNDED_NUMERIC_CLAIM" in codes and item_by_ref:
            ungrounded_sections.append(section)
    return uncited, build_ungrounded_claim_repair_hints(
        ungrounded_sections,
        item_by_ref,
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
    uncited_required_refs, ungrounded_value_hints = _named_section_offenders(
        plan,
        validation,
        section_by_id,
        manifest,
    )
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
                "uncited_required_refs": uncited_required_refs,
                "ungrounded_value_hints": ungrounded_value_hints,
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
            if code == _LONG_WORD_COUNT_CODE:
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
        document_errors.append("DOCUMENT_WORD_COUNT_TOO_SHORT")
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


def _without_free_unrendered_claims(
    draft_section: ReportSectionDraft,
    section_spec: Any,
    manifest: ReportEvidenceManifest,
    research_plan: ReportResearchPlan | None,
) -> ReportSectionDraft:
    """Sweep unused claim entries a repair call should never have to spend on.

    A verified claim the prose never rendered is surplus metadata; code can
    delete it for free, and doing so here keeps DIRECT_CLAIM_NOT_USED off the
    repair prompt entirely — the same bargain
    ``normalize_repairable_derived_claims`` already makes for derived displays.

    Held back when the drop would take an analysis section under its numeric
    floor. There the number is not surplus: the section owes it, so the writer
    must render it rather than have the evidence of the omission swept away.
    """

    swept, dropped = drop_unrendered_claims(
        draft_section,
        manifest.item_by_ref(),
    )
    if not dropped:
        return draft_section
    if research_plan is not None and _analysis_numeric_finding_missing(
        swept,
        section_spec,
        manifest,
        research_plan,
    ):
        return draft_section
    _LOGGER.info(
        "REPORT_UNRENDERED_CLAIM_DROPPED section_id=%s count=%d",
        draft_section.section_id,
        dropped,
    )
    return swept


def _log_schema_rejection(
    exc: Exception,
    *,
    stage: str,
    section_ids: Sequence[str],
) -> None:
    """Record which fields invalidated a model payload.

    ``DOCUMENT_SCHEMA_INVALID`` was the only blocking code with no named
    offender. Jobs 40e55527 and 5cb4d210 both discarded the analysis writer's
    entire output one millisecond after it returned and spent two of five
    generative calls rebuilding it, and neither log could say whether the
    cause was a missing contract_version, a duplicate paragraph, or a
    duplicate section id.
    """

    _LOGGER.warning(
        "REPORT_DOCUMENT_SCHEMA_INVALID %s",
        json.dumps(
            {
                "expected_section_ids": list(section_ids),
                "invalid_fields": validation_error_locations(exc),
                # A model_validator rejection has no field path at all, so
                # without this the payload names nothing.
                "invalid_rules": validation_error_rules(exc),
                "stage": stage,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
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
    except ValidationError as exc:
        _log_schema_rejection(exc, stage="section_batch", section_ids=expected_ids)
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
        if normalized_count:
            _LOGGER.info(
                "REPORT_DERIVED_CLAIM_NORMALIZED section_id=%s count=%d",
                draft_section.section_id,
                normalized_count,
            )
        section_spec = plan_by_id[draft_section.section_id]
        draft_section = _without_free_unrendered_claims(
            draft_section,
            section_spec,
            manifest,
            research_plan,
        )
        normalized_sections.append(draft_section)
        validation = validate_report_section(
            draft_section,
            section_spec,
            manifest,
        )
        word_count += validation.word_count
        blocking_codes = [
            code
            for code in validation.error_codes
            if code != _LONG_WORD_COUNT_CODE
        ]
        warning_codes = [
            code
            for code in validation.error_codes
            if code == _LONG_WORD_COUNT_CODE
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


def report_analysis_numeric_claim_requirement(
    draft_section: ReportSectionDraft,
    section_spec: Any,
    manifest: ReportEvidenceManifest,
    research_plan: ReportResearchPlan,
) -> tuple[int, int]:
    """Return ``(claims_made, claims_required)`` for one analysis section.

    Single authority for the rule, so the document gate, the grounded-subset
    guard, and the repair prompt cannot disagree about what counts as an
    analysis section worth shipping — or about how many numbers it still owes.

    The requirement never exceeds what the assigned evidence can support. A
    section whose tables declare no unit on any numeric column has nothing a
    claim could cite, so demanding one number of it is a gate no draft can
    pass: the writer either omits the claim and fails here, or invents one and
    fails grounding. That deadlock burned the repair budget on job cf47a2f6.
    """

    if section_spec.role is not ReportDocumentSectionRole.ANALYSIS:
        return (0, 0)
    needs_numbers = any(
        track.requested_metrics
        for track in research_plan.tracks
        if track.track_id in section_spec.track_ids
    )
    if not needs_numbers:
        return (0, 0)
    item_by_ref = manifest.item_by_ref()
    available_numeric_coordinates = sum(
        len(item.citable_numeric_coordinates())
        for evidence_ref in dict.fromkeys(section_spec.required_evidence_refs)
        if (item := item_by_ref.get(evidence_ref)) is not None
    )
    numeric_claim_count = sum(
        len(paragraph.direct_claims) + len(paragraph.derived_claims)
        for paragraph in draft_section.paragraphs
    )
    return (numeric_claim_count, min(2, available_numeric_coordinates))


def _analysis_numeric_finding_missing(
    draft_section: ReportSectionDraft,
    section_spec: Any,
    manifest: ReportEvidenceManifest,
    research_plan: ReportResearchPlan,
) -> bool:
    """Return whether an analysis section carries too few verified numbers."""

    made, required = report_analysis_numeric_claim_requirement(
        draft_section,
        section_spec,
        manifest,
        research_plan,
    )
    return made < required


def _concede_length_shortfall(
    validation: ReportDocumentValidation,
    *,
    stage: str,
) -> ReportDocumentValidation | None:
    """Downgrade a pure length shortfall to a warning, or return ``None``.

    Called only where the pipeline is otherwise about to give up. Prose length
    is a quality signal, not a correctness one: a section that is grounded,
    carries its numeric findings, cites its assigned evidence, and is merely
    shorter than its target is a report the reader can use. Failing it emits
    REPORT_DOCUMENT_INVALID, which is not retryable, so the reader gets nothing
    instead. Concedes nothing else — one non-length code and the document still
    fails exactly as before.

    A rising REPORT_LENGTH_CONCEDED rate means the planner is sizing sections
    above what the evidence can carry, not that generation is healthy.
    """

    if validation.valid:
        return None
    conceded_sections = {
        section_id: codes
        for section_id, codes in validation.section_errors.items()
        if codes == [_SHORT_SECTION_CODE]
    }
    if set(conceded_sections) != set(validation.section_errors):
        return None
    if set(validation.document_errors) - {_SHORT_DOCUMENT_CODE}:
        return None
    if not conceded_sections and not validation.document_errors:
        return None
    section_warnings = {
        section_id: list(codes)
        for section_id, codes in validation.section_warnings.items()
    }
    for section_id, codes in conceded_sections.items():
        section_warnings[section_id] = list(
            dict.fromkeys([*section_warnings.get(section_id, []), *codes])
        )
    conceded = validation.model_copy(
        update={
            "valid": True,
            "section_errors": {},
            "document_errors": [],
            "section_warnings": section_warnings,
            "document_warnings": list(
                dict.fromkeys(
                    [
                        *validation.document_warnings,
                        *validation.document_errors,
                    ]
                )
            ),
        }
    )
    _LOGGER.warning(
        "REPORT_LENGTH_CONCEDED %s",
        json.dumps(
            {
                "conceded_document_codes": validation.document_errors,
                "conceded_section_ids": sorted(conceded_sections),
                "stage": stage,
                "word_count": validation.word_count,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return conceded


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

    Two kinds of surplus are swept: prose the evidence cannot support, and
    verified claims the prose never rendered. The second was invisible here
    until job 827556eb, whose synthesis batch failed on DIRECT_CLAIM_NOT_USED
    alone — every sentence was grounded, so the sentence salvage found nothing
    to drop and returned before it could delete the one unused claim entry.
    """

    if sections is None or validation.valid:
        return sections, validation
    item_by_ref = manifest.item_by_ref()
    salvaged_sections: list[ReportSectionDraft] = []
    dropped_total = 0
    dropped_claim_total = 0
    for section in sections:
        salvaged, dropped = select_grounded_paragraphs(section, item_by_ref)
        salvaged, dropped_claims = drop_unrendered_claims(salvaged, item_by_ref)
        salvaged_sections.append(salvaged)
        dropped_total += dropped
        dropped_claim_total += dropped_claims
    if not dropped_total and not dropped_claim_total:
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
                "dropped_claim_count": dropped_claim_total,
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
    conceded = _concede_length_shortfall(
        subset_validation,
        stage=f"{stage}_grounded_subset",
    )
    if conceded is not None:
        subset_validation = conceded
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
    attempt_number: int,
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
        attempt_number=attempt_number,
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


def _repair_section_batch_until_valid(
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
    repair_attempts: int,
    first_attempt_number: int,
    stage: str,
) -> tuple[
    list[ReportSectionDraft] | None,
    ReportDocumentValidation,
    int,
]:
    """Repair one batch within budget, then try the grounded subset."""

    used = 0
    while (
        (sections is None or not validation.valid)
        and used < repair_attempts
    ):
        sections, validation = _repair_section_batch(
            query,
            plan,
            research_plan,
            manifest,
            packets,
            sections,
            raw_batch,
            validation,
            repair_sections,
            section_ids=section_ids,
            attempt_number=first_attempt_number + used,
        )
        used += 1
        # Without this the last repair's own result is never reported: the
        # batch diagnostic is emitted before each call, so an exhausted budget
        # ends with the failing word counts and codes unlogged — exactly the
        # numbers needed to tell a stalled writer from a wrong gate.
        _log_document_diagnostic(
            manifest=manifest,
            event=(
                "batch_repair_validated"
                if sections is not None and validation.valid
                else "batch_repair_rejected"
            ),
            plan=plan,
            validation=validation,
            draft=None,
            sections=sections,
            repair_section_ids=list(section_ids),
        )
    if sections is None or not validation.valid:
        # A batch that reached the salvage without spending a call logged
        # nothing at all: the in-loop diagnostic never ran. Job 827556eb spent
        # both repairs on analysis and its synthesis batch then failed with no
        # word counts, no section ids, and no codes recorded anywhere.
        if not used:
            _log_document_diagnostic(
                manifest=manifest,
                event="batch_repair_unbudgeted",
                plan=plan,
                validation=validation,
                draft=None,
                sections=sections,
                repair_section_ids=[],
            )
        sections, validation = _sections_or_grounded_subset(
            sections,
            validation,
            plan,
            manifest,
            section_ids=section_ids,
            stage=stage,
            research_plan=research_plan,
        )
    if sections is not None and not validation.valid:
        conceded = _concede_length_shortfall(validation, stage=stage)
        if conceded is not None:
            validation = conceded
    return sections, validation, used


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
    max_repair_attempts: int | None = None,
) -> ReportDocumentDraft:
    """Generate an adaptive document with a bounded targeted repair budget."""

    if max_repair_attempts is not None and not 0 <= max_repair_attempts <= 2:
        raise ValueError("max_repair_attempts must be between 0 and 2.")
    repair_attempt_budget = (
        (1 if allow_repair else 0)
        if max_repair_attempts is None
        else (max_repair_attempts if allow_repair else 0)
    )
    remaining_repairs = repair_attempt_budget

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
            analysis_sections, analysis_validation, repairs_used = (
                _repair_section_batch_until_valid(
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
                    repair_attempts=remaining_repairs,
                    first_attempt_number=(
                        2 + repair_attempt_budget - remaining_repairs
                    ),
                    stage="analysis_repair_exhausted",
                )
            )
            remaining_repairs -= repairs_used
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
            synthesis_sections, synthesis_validation, repairs_used = (
                _repair_section_batch_until_valid(
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
                    repair_attempts=remaining_repairs,
                    first_attempt_number=(
                        2 + repair_attempt_budget - remaining_repairs
                    ),
                    stage="synthesis_repair_exhausted",
                )
            )
            remaining_repairs -= repairs_used
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
    except ValidationError as exc:
        _log_schema_rejection(
            exc,
            stage="whole_document",
            section_ids=[section.section_id for section in plan.sections],
        )
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
    if repair_sections is None:
        from core.llm import llm_repair_report_document_sections

        repair_sections = llm_repair_report_document_sections
    while remaining_repairs > 0:
        structural_repair = draft is None or any(
            code in _STRUCTURAL_DOCUMENT_ERROR_CODES
            for code in validation.document_errors
        )
        invalid_section_ids = (
            [section.section_id for section in plan.sections]
            if structural_repair or not validation.section_errors
            else list(validation.section_errors)
        )
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
        baseline_draft = draft
        raw_repair = repair_sections(
            query,
            plan,
            research_plan,
            manifest,
            list(packets),
            raw_draft if draft is None else draft,
            validation,
            section_ids=invalid_section_ids,
            attempt_number=(
                2 + repair_attempt_budget - remaining_repairs
            ),
        )
        remaining_repairs -= 1
        try:
            repair = (
                raw_repair
                if isinstance(raw_repair, ReportDocumentRepair)
                else ReportDocumentRepair.model_validate(raw_repair)
            )
        except ValidationError:
            continue
        if {section.section_id for section in repair.sections} != set(
            invalid_section_ids
        ):
            continue
        draft = (
            _document_from_sections(plan, repair.sections)
            if structural_repair
            else _merge_repairs(draft, repair)
        )
        validation = validate_report_document(
            draft,
            plan,
            manifest,
            research_plan,
        )
        _log_document_diagnostic(
            manifest=manifest,
            event=(
                "repair_validated"
                if validation.valid
                else "repair_rejected"
            ),
            plan=plan,
            validation=validation,
            draft=draft,
            repair_section_ids=invalid_section_ids,
            pre_normalization_role_section_ids=(
                pre_normalization_role_section_ids
            ),
            role_normalization_applied=role_normalization_applied,
            baseline_draft=baseline_draft,
        )
        if validation.valid:
            return draft
    if draft is not None:
        conceded = _concede_length_shortfall(
            validation,
            stage="document_repair_exhausted",
        )
        if conceded is not None:
            return draft
    raise ReportDocumentGenerationError(validation)
