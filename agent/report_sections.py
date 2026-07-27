"""Parallel evidence-bound report section generation with bounded repair passes."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from threading import Event
from typing import Any

from pydantic import ValidationError

from agent.report_grounding import (
    build_evidence_grounding_index,
    validate_paragraph_grounding,
)
from contracts.report import ReportPlan, ReportSectionSpec
from contracts.report_evidence import ReportEvidenceManifest
from contracts.report_sections import (
    ReportSectionDraft,
    ReportSectionValidation,
)
from utils.provider_attempts import ProviderExecutionError

_WORD_PATTERN = re.compile(r"\b[\w]+(?:[.,'-][\w]+)*\b", re.UNICODE)
_LOGGER = logging.getLogger("Enai.ReportSections")


class ReportSectionGenerationError(RuntimeError):
    def __init__(
        self,
        section_id: str,
        error_codes: list[str],
        *,
        provider: str | None = None,
        provider_stage: str | None = None,
        provider_disposition: str | None = None,
    ) -> None:
        super().__init__(f"Report section {section_id} failed validation: " + ", ".join(error_codes))
        self.section_id = section_id
        self.error_codes = list(error_codes)
        self.provider = provider
        self.provider_stage = provider_stage
        self.provider_disposition = provider_disposition


class _SectionPhaseCancelled(RuntimeError):
    """Internal sentinel for queued work suppressed after a peer failure."""


def count_section_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(str(text or "")))


# The upper tolerance carries the model's systematic overshoot: gpt-oss-20b
# returned 136 and 141 words against a 109-word target and 159 against 118,
# repeating the same length across every repair (jobs c7823cc9 / acf48571).
# A +20% ceiling is simply unreachable for it, so the repair loop burned
# provider calls it could never satisfy. The lower bound stays tight — a short
# section is a content failure, an overlong one is a formatting one.
_SECTION_WORD_FLOOR_RATIO = 0.9
_SECTION_WORD_CEILING_RATIO = 1.35


def _section_word_bounds(section: ReportSectionSpec) -> tuple[int, int]:
    return (
        math.floor(section.target_words * _SECTION_WORD_FLOOR_RATIO),
        math.ceil(section.target_words * _SECTION_WORD_CEILING_RATIO),
    )


def _diagnostic_identifier(value: str | None) -> str:
    candidate = str(value or "")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", candidate):
        return candidate
    return "unknown"


def _diagnostic_error_codes(error_codes: list[str]) -> list[str]:
    return [code for code in error_codes[:16] if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code)]


def _log_section_diagnostic(
    *,
    event: str,
    section: ReportSectionSpec,
    attempt: int,
    started_at: float,
    error_codes: list[str],
    word_count: int | None = None,
    provider_error: ProviderExecutionError | None = None,
    level: int = logging.INFO,
) -> None:
    minimum_words, maximum_words = _section_word_bounds(section)
    payload: dict[str, Any] = {
        "attempt": attempt,
        "duration_ms": round(max(0.0, (time.monotonic() - started_at) * 1000), 2),
        "error_codes": _diagnostic_error_codes(error_codes),
        "event": event,
        "maximum_words": maximum_words,
        "minimum_words": minimum_words,
        "required_evidence_ref_count": len(section.required_evidence_refs),
        "section_id": section.section_id,
        "target_words": section.target_words,
        "word_count": word_count,
    }
    if provider_error is not None:
        payload.update(
            {
                "provider": _diagnostic_identifier(provider_error.provider),
                "provider_disposition": provider_error.disposition.value,
                "provider_stage": _diagnostic_identifier(provider_error.stage),
            }
        )
    _LOGGER.log(
        level,
        "REPORT_SECTION_DIAGNOSTIC %s",
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def validate_report_section(
    draft: ReportSectionDraft,
    section: ReportSectionSpec,
    manifest: ReportEvidenceManifest,
    *,
    evidence_facts_by_ref=None,
) -> ReportSectionValidation:
    errors: list[str] = []
    if draft.section_id != section.section_id:
        errors.append("SECTION_ID_MISMATCH")
    if draft.title != section.title:
        errors.append("SECTION_TITLE_MISMATCH")

    word_count = count_section_words(draft.content_markdown)
    minimum_words, maximum_words = _section_word_bounds(section)
    if not minimum_words <= word_count <= maximum_words:
        errors.append("WORD_COUNT_OUT_OF_RANGE")

    item_by_ref = manifest.item_by_ref()
    allowed_refs = set(section.required_evidence_refs)
    if evidence_facts_by_ref is None:
        paragraph_refs = {
            ref
            for paragraph in draft.paragraphs
            for ref in paragraph.evidence_refs
        }
        evidence_facts_by_ref = build_evidence_grounding_index(
            item_by_ref,
            allowed_refs & paragraph_refs,
        )
    used_refs: set[str] = set()
    for paragraph in draft.paragraphs:
        paragraph_refs = set(paragraph.evidence_refs)
        used_refs.update(paragraph_refs)
        if not paragraph_refs.issubset(allowed_refs):
            errors.append("EVIDENCE_REF_NOT_ALLOWED")
            continue
        if any(ref not in item_by_ref for ref in paragraph_refs):
            errors.append("EVIDENCE_REF_NOT_FOUND")
            continue
        errors.extend(
            validate_paragraph_grounding(
                paragraph,
                item_by_ref,
                evidence_facts_by_ref=evidence_facts_by_ref,
            )
        )

    if not allowed_refs.issubset(used_refs):
        errors.append("REQUIRED_EVIDENCE_NOT_USED")

    errors = list(dict.fromkeys(errors))
    return ReportSectionValidation(
        valid=not errors,
        error_codes=errors,
        word_count=word_count,
    )


SectionGenerator = Callable[
    [str, ReportPlan, ReportSectionSpec, ReportEvidenceManifest],
    ReportSectionDraft | dict[str, Any],
]
SectionRepairer = Callable[
    [
        str,
        ReportPlan,
        ReportSectionSpec,
        ReportEvidenceManifest,
        ReportSectionDraft | dict[str, Any],
        list[str],
    ],
    ReportSectionDraft | dict[str, Any],
]
ProgressCallback = Callable[[int, int, ReportSectionDraft], Any]


def generate_report_sections(
    query: str,
    plan: ReportPlan,
    manifest: ReportEvidenceManifest,
    *,
    existing_drafts: dict[str, ReportSectionDraft] | None = None,
    generate_section: SectionGenerator | None = None,
    repair_section: SectionRepairer | None = None,
    progress_callback: ProgressCallback | None = None,
    max_workers: int = 4,
    max_repair_attempts: int = 2,
) -> list[ReportSectionDraft]:
    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between 1 and 8.")
    if not 1 <= max_repair_attempts <= 3:
        raise ValueError("max_repair_attempts must be between 1 and 3.")
    if generate_section is None:
        from core.llm import llm_write_report_section

        generate_section = llm_write_report_section
    item_by_ref = manifest.item_by_ref()
    grounding_index = build_evidence_grounding_index(
        item_by_ref,
        set(item_by_ref),
    )
    completed: dict[str, ReportSectionDraft] = {}
    for section in plan.sections:
        existing = (existing_drafts or {}).get(section.section_id)
        if existing is None:
            continue
        validation = validate_report_section(
            existing,
            section,
            manifest,
            evidence_facts_by_ref=grounding_index,
        )
        if validation.valid:
            completed[section.section_id] = existing

    def generate_one(section: ReportSectionSpec) -> ReportSectionDraft:
        candidate_started_at = time.monotonic()
        try:
            raw_draft: ReportSectionDraft | dict[str, Any] = generate_section(
                query,
                plan,
                section,
                manifest,
            )
        except ProviderExecutionError as exc:
            _log_section_diagnostic(
                event="provider_failed",
                section=section,
                attempt=1,
                started_at=candidate_started_at,
                error_codes=["SECTION_WRITE_PROVIDER_FAILED"],
                provider_error=exc,
                level=logging.WARNING,
            )
            raise ReportSectionGenerationError(
                section.section_id,
                ["SECTION_WRITE_PROVIDER_FAILED"],
                provider=exc.provider,
                provider_stage=exc.stage,
                provider_disposition=exc.disposition.value,
            ) from exc
        try:
            draft = (
                raw_draft if isinstance(raw_draft, ReportSectionDraft) else ReportSectionDraft.model_validate(raw_draft)
            )
            validation = validate_report_section(
                draft,
                section,
                manifest,
                evidence_facts_by_ref=grounding_index,
            )
            error_codes = validation.error_codes
        except ValidationError:
            draft = raw_draft
            error_codes = ["SECTION_SCHEMA_INVALID"]
            validation = None

        if error_codes:
            _log_section_diagnostic(
                event="candidate_rejected",
                section=section,
                attempt=1,
                started_at=candidate_started_at,
                error_codes=error_codes,
                word_count=(validation.word_count if validation is not None else None),
            )
            effective_repair = repair_section
            uses_default_repair = effective_repair is None
            if uses_default_repair:
                from core.llm import llm_repair_report_section

                effective_repair = llm_repair_report_section
            current_draft = draft
            current_error_codes = error_codes
            for repair_index in range(max_repair_attempts):
                attempt = repair_index + 2
                repair_started_at = time.monotonic()
                try:
                    repair_args = (
                        query,
                        plan,
                        section,
                        manifest,
                        current_draft,
                        current_error_codes,
                    )
                    if uses_default_repair:
                        repaired_raw = effective_repair(
                            *repair_args,
                            attempt_number=attempt,
                        )
                    else:
                        repaired_raw = effective_repair(*repair_args)
                except ProviderExecutionError as exc:
                    _log_section_diagnostic(
                        event="provider_failed",
                        section=section,
                        attempt=attempt,
                        started_at=repair_started_at,
                        error_codes=["SECTION_REPAIR_PROVIDER_FAILED"],
                        provider_error=exc,
                        level=logging.WARNING,
                    )
                    raise ReportSectionGenerationError(
                        section.section_id,
                        ["SECTION_REPAIR_PROVIDER_FAILED"],
                        provider=exc.provider,
                        provider_stage=exc.stage,
                        provider_disposition=exc.disposition.value,
                    ) from exc
                except ValidationError as exc:
                    current_error_codes = ["SECTION_SCHEMA_INVALID"]
                    _log_section_diagnostic(
                        event="repair_rejected",
                        section=section,
                        attempt=attempt,
                        started_at=repair_started_at,
                        error_codes=current_error_codes,
                        level=logging.WARNING,
                    )
                    if repair_index + 1 == max_repair_attempts:
                        raise ReportSectionGenerationError(
                            section.section_id,
                            current_error_codes,
                        ) from exc
                    continue
                try:
                    repaired = (
                        repaired_raw
                        if isinstance(repaired_raw, ReportSectionDraft)
                        else ReportSectionDraft.model_validate(repaired_raw)
                    )
                except ValidationError as exc:
                    current_draft = repaired_raw
                    current_error_codes = ["SECTION_SCHEMA_INVALID"]
                    _log_section_diagnostic(
                        event="repair_rejected",
                        section=section,
                        attempt=attempt,
                        started_at=repair_started_at,
                        error_codes=current_error_codes,
                        level=logging.WARNING,
                    )
                    if repair_index + 1 == max_repair_attempts:
                        raise ReportSectionGenerationError(
                            section.section_id,
                            current_error_codes,
                        ) from exc
                    continue
                repaired_validation = validate_report_section(
                    repaired,
                    section,
                    manifest,
                    evidence_facts_by_ref=grounding_index,
                )
                if repaired_validation.valid:
                    _log_section_diagnostic(
                        event="repair_validated",
                        section=section,
                        attempt=attempt,
                        started_at=repair_started_at,
                        error_codes=[],
                        word_count=repaired_validation.word_count,
                    )
                    return repaired
                current_draft = repaired
                current_error_codes = repaired_validation.error_codes
                _log_section_diagnostic(
                    event="repair_rejected",
                    section=section,
                    attempt=attempt,
                    started_at=repair_started_at,
                    error_codes=current_error_codes,
                    word_count=repaired_validation.word_count,
                    level=logging.WARNING,
                )
            raise ReportSectionGenerationError(
                section.section_id,
                current_error_codes,
            )
        _log_section_diagnostic(
            event="candidate_validated",
            section=section,
            attempt=1,
            started_at=candidate_started_at,
            error_codes=[],
            word_count=validation.word_count,
        )
        return draft

    pending_sections = [section for section in plan.sections if section.section_id not in completed]
    if pending_sections:
        worker_count = min(max_workers, len(pending_sections))
        phase_failed = Event()

        def run_one(section: ReportSectionSpec) -> ReportSectionDraft:
            if phase_failed.is_set():
                raise _SectionPhaseCancelled()
            try:
                return generate_one(section)
            except BaseException:
                phase_failed.set()
                raise

        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="report-section",
        )
        future_to_section = {}
        try:
            for section in pending_sections:
                captured_context = copy_context()
                future = executor.submit(
                    captured_context.run,
                    run_one,
                    section,
                )
                future_to_section[future] = section
            for future in as_completed(future_to_section):
                if future.cancelled():
                    continue
                try:
                    draft = future.result()
                except _SectionPhaseCancelled:
                    continue
                completed[draft.section_id] = draft
                if progress_callback is not None:
                    progress_callback(
                        len(completed),
                        len(plan.sections),
                        draft,
                    )
        except Exception as phase_error:
            phase_failed.set()
            failed_in_future = any(
                future.done()
                and not future.cancelled()
                and future.exception() is phase_error
                for future in future_to_section
            )
            for future in future_to_section:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            if failed_in_future:
                for future in future_to_section:
                    if future.cancelled() or future.exception() is not None:
                        continue
                    draft = future.result()
                    if draft.section_id in completed:
                        continue
                    completed[draft.section_id] = draft
                    if progress_callback is not None:
                        try:
                            progress_callback(
                                len(completed),
                                len(plan.sections),
                                draft,
                            )
                        except Exception:
                            _LOGGER.warning(
                                "Could not checkpoint a peer section during "
                                "failure settlement: section_id=%s",
                                draft.section_id,
                            )
            raise
        else:
            executor.shutdown(wait=True)

    return [completed[section.section_id] for section in plan.sections]
