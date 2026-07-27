"""Strict contracts for evidence-grounded standard report planning.

The report planner may choose titles, analytical subsections, evidence
assignments, and supported chart intents. Code owns the required structure,
word-budget arithmetic, identifier uniqueness, and cross-reference integrity.
"""

from __future__ import annotations

import math
from copy import deepcopy
from enum import Enum
from typing import Annotated, Any, List, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPORT_PLAN_CONTRACT_VERSION = "report-plan-v1"
STANDARD_REPORT_MIN_WORDS = 900
STANDARD_REPORT_MAX_WORDS = 1400
STANDARD_REPORT_MAX_SECTIONS = 8
STANDARD_REPORT_RESULT_MIN_WORDS = (
    math.floor(STANDARD_REPORT_MIN_WORDS * 0.9)
    - (STANDARD_REPORT_MAX_SECTIONS - 1)
)
STANDARD_REPORT_RESULT_MAX_WORDS = (
    math.ceil(STANDARD_REPORT_MAX_WORDS * 1.2)
    + (STANDARD_REPORT_MAX_SECTIONS - 1)
)
REPORT_SECTION_MIN_WORDS = 40
REPORT_SECTION_MAX_WORDS = 800


class ReportSectionKind(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    SCOPE_AND_EVIDENCE = "scope_and_evidence"
    KEY_FINDINGS = "key_findings"
    ANALYSIS = "analysis"
    IMPLICATIONS = "implications"
    LIMITATIONS = "limitations"
    CONCLUSION = "conclusion"


STANDARD_REPORT_SECTION_SEQUENCE = (
    ReportSectionKind.EXECUTIVE_SUMMARY,
    ReportSectionKind.SCOPE_AND_EVIDENCE,
    ReportSectionKind.KEY_FINDINGS,
    ReportSectionKind.LIMITATIONS,
    ReportSectionKind.CONCLUSION,
)


def normalize_report_plan_word_budget(payload: Any) -> Any:
    """Return a copy with valid integer section weights scaled to the report total."""

    if not isinstance(payload, dict):
        return payload
    target_words = payload.get("target_words")
    sections = payload.get("sections")
    if (
        isinstance(target_words, bool)
        or not isinstance(target_words, int)
        or not STANDARD_REPORT_MIN_WORDS
        <= target_words
        <= STANDARD_REPORT_MAX_WORDS
        or not isinstance(sections, list)
        or not 5 <= len(sections) <= 8
    ):
        return payload

    section_targets: list[int] = []
    for section in sections:
        if not isinstance(section, dict):
            return payload
        section_target = section.get("target_words")
        if (
            isinstance(section_target, bool)
            or not isinstance(section_target, int)
            or not REPORT_SECTION_MIN_WORDS
            <= section_target
            <= REPORT_SECTION_MAX_WORDS
        ):
            return payload
        section_targets.append(section_target)

    allocated_total = sum(section_targets)
    if allocated_total == target_words:
        return payload
    if not (
        len(sections) * REPORT_SECTION_MIN_WORDS
        <= target_words
        <= len(sections) * REPORT_SECTION_MAX_WORDS
    ):
        return payload

    scaled_numerators = [
        target_words * section_target
        for section_target in section_targets
    ]
    allocations = [
        min(
            REPORT_SECTION_MAX_WORDS,
            max(
                REPORT_SECTION_MIN_WORDS,
                numerator // allocated_total,
            ),
        )
        for numerator in scaled_numerators
    ]
    remainders = [
        numerator % allocated_total
        for numerator in scaled_numerators
    ]

    remaining = target_words - sum(allocations)
    while remaining > 0:
        progressed = False
        for index in sorted(
            range(len(allocations)),
            key=lambda item: (-remainders[item], item),
        ):
            if allocations[index] >= REPORT_SECTION_MAX_WORDS:
                continue
            allocations[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            return payload
    while remaining < 0:
        progressed = False
        for index in sorted(
            range(len(allocations)),
            key=lambda item: (remainders[item], item),
        ):
            if allocations[index] <= REPORT_SECTION_MIN_WORDS:
                continue
            allocations[index] -= 1
            remaining += 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            return payload

    normalized = deepcopy(payload)
    for section, allocation in zip(
        normalized["sections"],
        allocations,
        strict=True,
    ):
        section["target_words"] = allocation
    return normalized


def validate_standard_report_section_order(
    kinds: Sequence[ReportSectionKind],
) -> None:
    """Validate the canonical standard-report section structure."""

    missing = [
        kind.value
        for kind in STANDARD_REPORT_SECTION_SEQUENCE
        if kinds.count(kind) != 1
    ]
    if missing:
        raise ValueError(
            "Report must contain each of the required standard sections "
            f"exactly once: {', '.join(missing)}."
        )

    if not (
        kinds[0] == ReportSectionKind.EXECUTIVE_SUMMARY
        and kinds[1] == ReportSectionKind.SCOPE_AND_EVIDENCE
        and kinds[2] == ReportSectionKind.KEY_FINDINGS
        and kinds[-2] == ReportSectionKind.LIMITATIONS
        and kinds[-1] == ReportSectionKind.CONCLUSION
    ):
        raise ValueError(
            "Report violates the standard section order: executive summary, "
            "scope and evidence, key findings, optional analysis/implications, "
            "limitations, conclusion."
        )

    middle_kinds = kinds[3:-2]
    allowed_middle = {
        ReportSectionKind.ANALYSIS,
        ReportSectionKind.IMPLICATIONS,
    }
    if any(kind not in allowed_middle for kind in middle_kinds):
        raise ValueError("Report contains a section outside the standard section order.")
    if middle_kinds.count(ReportSectionKind.IMPLICATIONS) > 1:
        raise ValueError("Report may contain at most one implications section.")
    if (
        ReportSectionKind.IMPLICATIONS in middle_kinds
        and middle_kinds[-1] != ReportSectionKind.IMPLICATIONS
    ):
        raise ValueError("The implications section must follow all analytical sections.")


class ReportChartPurpose(str, Enum):
    TREND = "trend"
    COMPARISON = "comparison"
    COMPOSITION = "composition"
    RELATIONSHIP = "relationship"
    FORECAST = "forecast"
    TABLE = "table"


Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
EvidenceRef = Annotated[str, Field(min_length=1, max_length=256)]


class _StrictReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ReportSectionSpec(_StrictReportModel):
    section_id: Identifier
    kind: ReportSectionKind
    title: str = Field(min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=600)
    target_words: int = Field(
        ge=REPORT_SECTION_MIN_WORDS,
        le=REPORT_SECTION_MAX_WORDS,
    )
    required_evidence_refs: List[EvidenceRef] = Field(min_length=1, max_length=32)
    chart_refs: List[Identifier] = Field(default_factory=list, max_length=3)

    @field_validator("required_evidence_refs")
    @classmethod
    def _require_unique_evidence_refs(cls, refs: List[str]) -> List[str]:
        if len(refs) != len(set(refs)):
            raise ValueError("Report sections must use unique evidence references.")
        return refs


class ReportChartRequest(_StrictReportModel):
    chart_id: Identifier
    section_id: Identifier
    purpose: ReportChartPurpose
    title: str = Field(min_length=1, max_length=160)
    evidence_refs: List[EvidenceRef] = Field(min_length=1, max_length=16)
    x_field: str | None = Field(default=None, min_length=1, max_length=128)
    series_fields: List[str] = Field(default_factory=list, max_length=8)
    required: bool = False

    @field_validator("evidence_refs")
    @classmethod
    def _require_unique_evidence_refs(cls, refs: List[str]) -> List[str]:
        if len(refs) != len(set(refs)):
            raise ValueError("Report charts must use unique evidence references.")
        return refs

    @field_validator("series_fields")
    @classmethod
    def _require_unique_series_fields(cls, fields: List[str]) -> List[str]:
        if len(fields) != len(set(fields)):
            raise ValueError("Report chart series_fields must be unique.")
        if any(not field or len(field) > 128 for field in fields):
            raise ValueError("Report chart series_fields must be bounded.")
        return fields


class ReportPlan(_StrictReportModel):
    contract_version: Literal["report-plan-v1"]
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=1000)
    language_code: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
    target_words: int = Field(
        ge=STANDARD_REPORT_MIN_WORDS,
        le=STANDARD_REPORT_MAX_WORDS,
    )
    evidence_manifest_id: str = Field(min_length=1, max_length=128)
    sections: List[ReportSectionSpec] = Field(
        min_length=5,
        max_length=STANDARD_REPORT_MAX_SECTIONS,
    )
    charts: List[ReportChartRequest] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _validate_standard_structure(self) -> "ReportPlan":
        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Report sections must use a unique section_id.")

        chart_ids = [chart.chart_id for chart in self.charts]
        if len(chart_ids) != len(set(chart_ids)):
            raise ValueError("Report charts must use a unique chart_id.")

        kinds = [section.kind for section in self.sections]
        validate_standard_report_section_order(kinds)

        allocated_words = sum(section.target_words for section in self.sections)
        if allocated_words != self.target_words:
            raise ValueError(
                "The sum of section target_words must equal the report target_words."
            )

        known_section_ids = set(section_ids)
        known_chart_ids = set(chart_ids)
        section_by_id = {section.section_id: section for section in self.sections}

        for section in self.sections:
            unknown_refs = sorted(set(section.chart_refs) - known_chart_ids)
            if unknown_refs:
                raise ValueError(
                    f"Section {section.section_id} contains unknown chart_refs: "
                    f"{', '.join(unknown_refs)}."
                )
            if len(section.chart_refs) != len(set(section.chart_refs)):
                raise ValueError(
                    f"Section {section.section_id} contains duplicate chart_refs."
                )

        for chart in self.charts:
            if chart.section_id not in known_section_ids:
                raise ValueError(
                    f"Chart {chart.chart_id} targets unknown section_id {chart.section_id}."
                )
            assigned_section = section_by_id[chart.section_id]
            if chart.chart_id not in assigned_section.chart_refs:
                raise ValueError(
                    f"Chart {chart.chart_id} must be referenced by its assigned section "
                    f"{chart.section_id}."
                )
            other_sections = [
                section.section_id
                for section in self.sections
                if section.section_id != chart.section_id
                and chart.chart_id in section.chart_refs
            ]
            if other_sections:
                raise ValueError(
                    f"Chart {chart.chart_id} may only be referenced by its assigned section."
                )

        return self
