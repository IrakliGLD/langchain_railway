"""Parallel evidence-bound report section generation with bounded repair passes."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
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
_NUMERIC_PATTERN = re.compile(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?%?(?![\w])")
_PERIOD_PATTERN = re.compile(
    r"(?<![\w])(?P<year>\d{4})[-/]"
    r"(?P<segment>\d{1,2}|[Qq][1-4])"
    r"(?:[-/](?P<day>\d{1,2}))?(?![\w])"
)
_LOGGER = logging.getLogger("Enai.ReportSections")
_RATIO_COLUMN_NAMES = {
    "balancing_share",
    "generation_share",
    "import_dependency_ratio",
}


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


def count_section_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(str(text or "")))


def _section_word_bounds(section: ReportSectionSpec) -> tuple[int, int]:
    return (
        math.floor(section.target_words * 0.9),
        math.ceil(section.target_words * 1.2),
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


@dataclass(frozen=True, slots=True)
class _NumericFact:
    value: Decimal
    is_percent: bool
    precision: int


@dataclass(frozen=True, slots=True)
class _PeriodFact:
    value: str


def _parse_numeric_token(token: str) -> _NumericFact | None:
    raw = str(token).strip()
    is_percent = raw.endswith("%")
    numeric = raw.removesuffix("%").replace(",", "").lstrip("+")
    try:
        value = Decimal(numeric)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    precision = len(numeric.rsplit(".", 1)[1]) if "." in numeric else 0
    return _NumericFact(
        value=value,
        is_percent=is_percent,
        precision=precision,
    )


def _normalized_period_fact(match: re.Match[str]) -> _PeriodFact | None:
    year = match.group("year")
    segment = match.group("segment")
    day = match.group("day")
    if segment.lower().startswith("q"):
        if day is not None:
            return None
        return _PeriodFact(f"{year}-{segment.upper()}")
    month_number = int(segment)
    if not 1 <= month_number <= 12:
        return None
    normalized = f"{year}-{month_number:02d}"
    if day is None:
        return _PeriodFact(normalized)
    day_number = int(day)
    if not 1 <= day_number <= 31:
        return None
    return _PeriodFact(f"{normalized}-{day_number:02d}")


def _grounding_facts_from_text(
    text: str,
) -> set[_NumericFact | _PeriodFact]:
    facts: set[_NumericFact | _PeriodFact] = set()

    def replace_period(match: re.Match[str]) -> str:
        period_fact = _normalized_period_fact(match)
        if period_fact is None:
            return match.group(0)
        facts.add(period_fact)
        year_fact = _parse_numeric_token(match.group("year"))
        if year_fact is not None:
            facts.add(year_fact)
        return " "

    remaining_text = _PERIOD_PATTERN.sub(replace_period, str(text or ""))
    facts.update(
        {
            fact
            for token in _NUMERIC_PATTERN.findall(remaining_text)
            if (fact := _parse_numeric_token(token)) is not None
        }
    )
    return facts


def _grounding_facts_from_value(
    value: Any,
) -> set[_NumericFact | _PeriodFact]:
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, float) and not math.isfinite(value):
            return set()
        fact = _parse_numeric_token(str(value))
        return {fact} if fact is not None else set()
    if isinstance(value, str):
        return _grounding_facts_from_text(value)
    return set()


def _is_ratio_column(item, column: str) -> bool:
    normalized_column = str(column).strip().lower()
    normalized_unit = str(item.unit_by_column.get(column, "")).strip().lower()
    return (
        normalized_unit in {"ratio", "fraction"}
        or normalized_column.startswith("share_")
        or normalized_column in _RATIO_COLUMN_NAMES
    )


def _evidence_grounding_facts(
    item,
) -> set[_NumericFact | _PeriodFact]:
    facts = _grounding_facts_from_text(item.content)
    for row in item.rows:
        for column, value in row.items():
            value_facts = _grounding_facts_from_value(value)
            facts.update(value_facts)
            if not _is_ratio_column(item, column):
                continue
            for fact in value_facts:
                if not isinstance(fact, _NumericFact) or fact.is_percent or abs(fact.value) > 1:
                    continue
                facts.add(
                    _NumericFact(
                        value=fact.value * Decimal(100),
                        is_percent=True,
                        precision=max(0, fact.precision - 2),
                    )
                )
    return facts


def _grounding_claim_is_supported(
    claim: _NumericFact | _PeriodFact,
    evidence_facts: set[_NumericFact | _PeriodFact],
) -> bool:
    if isinstance(claim, _PeriodFact):
        return claim in evidence_facts
    quantum = Decimal(1).scaleb(-claim.precision)
    for fact in evidence_facts:
        if not isinstance(fact, _NumericFact):
            continue
        if fact.is_percent != claim.is_percent:
            continue
        if fact.value == claim.value:
            return True
        if fact.precision <= claim.precision:
            continue
        if fact.value.quantize(quantum, rounding=ROUND_HALF_UP) == claim.value:
            return True
    return False


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
    minimum_words, maximum_words = _section_word_bounds(section)
    if not minimum_words <= word_count <= maximum_words:
        errors.append("WORD_COUNT_OUT_OF_RANGE")

    item_by_ref = manifest.item_by_ref()
    grounding_facts_by_ref = {ref: _evidence_grounding_facts(item) for ref, item in item_by_ref.items()}
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
        evidence_facts = set().union(*(grounding_facts_by_ref[ref] for ref in paragraph_refs))
        paragraph_claims = _grounding_facts_from_text(paragraph.text)
        if any(not _grounding_claim_is_supported(claim, evidence_facts) for claim in paragraph_claims):
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
    max_repair_attempts: int = 2,
) -> list[ReportSectionDraft]:
    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between 1 and 8.")
    if not 1 <= max_repair_attempts <= 3:
        raise ValueError("max_repair_attempts must be between 1 and 3.")
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
            validation = validate_report_section(draft, section, manifest)
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
            if effective_repair is None:
                from core.llm import llm_repair_report_section

                effective_repair = llm_repair_report_section
            current_draft = draft
            current_error_codes = error_codes
            for repair_index in range(max_repair_attempts):
                attempt = repair_index + 2
                repair_started_at = time.monotonic()
                try:
                    repaired_raw = effective_repair(
                        query,
                        plan,
                        section,
                        manifest,
                        current_draft,
                        current_error_codes,
                    )
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
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="report-section",
        ) as executor:
            future_to_section = {executor.submit(generate_one, section): section for section in pending_sections}
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
