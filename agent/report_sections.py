"""Parallel evidence-bound report section generation with one repair pass."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import ValidationError

from contracts.report import ReportPlan, ReportSectionSpec
from contracts.report_evidence import ReportEvidenceManifest
from contracts.report_sections import (
    ReportSectionDraft,
    ReportSectionValidation,
)
from utils.provider_attempts import ProviderExecutionError

_WORD_PATTERN = re.compile(r"\b[\w]+(?:[.,'-][\w]+)*\b", re.UNICODE)
_NUMERIC_PATTERN = re.compile(
    r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?%?(?![\w])"
)
_LOGGER = logging.getLogger("Enai.ReportSections")
_RATIO_COLUMN_NAMES = {
    "balancing_share",
    "generation_share",
    "import_dependency_ratio",
}


class ReportSectionGenerationError(RuntimeError):
    def __init__(self, section_id: str, error_codes: list[str]) -> None:
        super().__init__(
            f"Report section {section_id} failed validation: "
            + ", ".join(error_codes)
        )
        self.section_id = section_id
        self.error_codes = list(error_codes)


def count_section_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(str(text or "")))


def _normalize_numeric_token(token: str) -> str:
    return token.replace(",", "").lstrip("+")


def _evidence_numeric_tokens(item) -> set[str]:
    serialized = json.dumps(
        item.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    tokens = {
        _normalize_numeric_token(token)
        for token in _NUMERIC_PATTERN.findall(serialized)
    }
    if not item.rows:
        return tokens

    for column in item.columns:
        normalized_column = str(column).strip().lower()
        normalized_unit = str(item.unit_by_column.get(column, "")).strip().lower()
        is_storage_ratio = (
            normalized_unit in {"ratio", "fraction"}
            or normalized_column.startswith("share_")
            or normalized_column in _RATIO_COLUMN_NAMES
        )
        if not is_storage_ratio:
            continue
        for row in item.rows:
            value = row.get(column)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or abs(float(value)) > 1
            ):
                continue
            percent_value = float(value) * 100
            for precision in range(5):
                rendered = f"{percent_value:.{precision}f}"
                if "." in rendered:
                    rendered = rendered.rstrip("0").rstrip(".")
                tokens.add(_normalize_numeric_token(rendered))
                tokens.add(_normalize_numeric_token(f"{rendered}%"))
    return tokens


def validate_report_section(
    draft: ReportSectionDraft,
    section: ReportSectionSpec,
    manifest: ReportEvidenceManifest,
) -> ReportSectionValidation:
    errors: list[str] = []
    if draft.section_id != section.section_id:
        errors.append("SECTION_ID_MISMATCH")
    if draft.title != section.title:
        errors.append("SECTION_TITLE_MISMATCH")

    word_count = count_section_words(draft.content_markdown)
    minimum_words = math.floor(section.target_words * 0.9)
    maximum_words = math.ceil(section.target_words * 1.2)
    if not minimum_words <= word_count <= maximum_words:
        errors.append("WORD_COUNT_OUT_OF_RANGE")

    item_by_ref = manifest.item_by_ref()
    allowed_refs = set(section.required_evidence_refs)
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
        evidence_tokens = set().union(
            *(_evidence_numeric_tokens(item_by_ref[ref]) for ref in paragraph_refs)
        )
        paragraph_tokens = {
            _normalize_numeric_token(token)
            for token in _NUMERIC_PATTERN.findall(paragraph.text)
        }
        if not paragraph_tokens.issubset(evidence_tokens):
            errors.append("UNGROUNDED_NUMERIC_CLAIM")

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
) -> list[ReportSectionDraft]:
    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between 1 and 8.")
    if generate_section is None:
        from core.llm import llm_write_report_section

        generate_section = llm_write_report_section
    completed: dict[str, ReportSectionDraft] = {}
    for section in plan.sections:
        existing = (existing_drafts or {}).get(section.section_id)
        if existing is None:
            continue
        validation = validate_report_section(existing, section, manifest)
        if validation.valid:
            completed[section.section_id] = existing

    def generate_one(section: ReportSectionSpec) -> ReportSectionDraft:
        try:
            raw_draft: ReportSectionDraft | dict[str, Any] = generate_section(
                query,
                plan,
                section,
                manifest,
            )
        except ProviderExecutionError as exc:
            raise ReportSectionGenerationError(
                section.section_id,
                ["SECTION_WRITE_PROVIDER_FAILED"],
            ) from exc
        try:
            draft = (
                raw_draft
                if isinstance(raw_draft, ReportSectionDraft)
                else ReportSectionDraft.model_validate(raw_draft)
            )
            validation = validate_report_section(draft, section, manifest)
            error_codes = validation.error_codes
        except ValidationError:
            draft = raw_draft
            error_codes = ["SECTION_SCHEMA_INVALID"]
            validation = None

        if error_codes:
            _LOGGER.info(
                "Report section candidate rejected: section_id=%s "
                "error_codes=%s word_count=%s",
                section.section_id,
                ",".join(error_codes),
                validation.word_count if validation is not None else "unknown",
            )
            effective_repair = repair_section
            if effective_repair is None:
                from core.llm import llm_repair_report_section

                effective_repair = llm_repair_report_section
            try:
                repaired_raw = effective_repair(
                    query,
                    plan,
                    section,
                    manifest,
                    draft,
                    error_codes,
                )
            except ProviderExecutionError as exc:
                raise ReportSectionGenerationError(
                    section.section_id,
                    ["SECTION_REPAIR_PROVIDER_FAILED"],
                ) from exc
            try:
                repaired = (
                    repaired_raw
                    if isinstance(repaired_raw, ReportSectionDraft)
                    else ReportSectionDraft.model_validate(repaired_raw)
                )
            except ValidationError as exc:
                raise ReportSectionGenerationError(
                    section.section_id,
                    ["SECTION_SCHEMA_INVALID"],
                ) from exc
            repaired_validation = validate_report_section(
                repaired,
                section,
                manifest,
            )
            if not repaired_validation.valid:
                _LOGGER.warning(
                    "Report section repair validation failed: section_id=%s "
                    "error_codes=%s word_count=%s",
                    section.section_id,
                    ",".join(repaired_validation.error_codes),
                    repaired_validation.word_count,
                )
                raise ReportSectionGenerationError(
                    section.section_id,
                    repaired_validation.error_codes,
                )
            return repaired
        return draft

    pending_sections = [
        section
        for section in plan.sections
        if section.section_id not in completed
    ]
    if pending_sections:
        worker_count = min(max_workers, len(pending_sections))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="report-section",
        ) as executor:
            future_to_section = {
                executor.submit(generate_one, section): section
                for section in pending_sections
            }
            try:
                for future in as_completed(future_to_section):
                    draft = future.result()
                    completed[draft.section_id] = draft
                    if progress_callback is not None:
                        progress_callback(
                            len(completed),
                            len(plan.sections),
                            draft,
                        )
            except Exception:
                for future in future_to_section:
                    future.cancel()
                raise

    return [completed[section.section_id] for section in plan.sections]
