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
REPORT_MAX_EXHIBITS = 4
REPORT_SECTION_WORD_FLOOR_RATIO = 0.9
REPORT_SECTION_PROMPT_WORD_CEILING_RATIO = 1.2
REPORT_SECTION_VALIDATION_WORD_CEILING_RATIO = 1.35
REPORT_AGGREGATE_WORD_CEILING_RATIO = (
    REPORT_SECTION_VALIDATION_WORD_CEILING_RATIO
)
STANDARD_REPORT_RESULT_MIN_WORDS = (
    math.floor(
        STANDARD_REPORT_MIN_WORDS * REPORT_SECTION_WORD_FLOOR_RATIO
    )
    - (STANDARD_REPORT_MAX_SECTIONS - 1)
)
STANDARD_REPORT_RESULT_MAX_WORDS = (
    math.ceil(
        STANDARD_REPORT_MAX_WORDS * REPORT_AGGREGATE_WORD_CEILING_RATIO
    )
    + (STANDARD_REPORT_MAX_SECTIONS - 1)
)
REPORT_SECTION_MIN_WORDS = 40
REPORT_SECTION_MAX_WORDS = 800


def report_section_prompt_word_bounds(target_words: int) -> tuple[int, int]:
    return (
        math.floor(target_words * REPORT_SECTION_WORD_FLOOR_RATIO),
        math.ceil(
            target_words * REPORT_SECTION_PROMPT_WORD_CEILING_RATIO
        ),
    )


def report_section_validation_word_bounds(
    target_words: int,
) -> tuple[int, int]:
    return (
        math.floor(target_words * REPORT_SECTION_WORD_FLOOR_RATIO),
        math.ceil(
            target_words * REPORT_SECTION_VALIDATION_WORD_CEILING_RATIO
        ),
    )


def report_aggregate_word_bounds(
    section_targets: Sequence[int],
) -> tuple[int, int]:
    return (
        sum(
            math.floor(target * REPORT_SECTION_WORD_FLOOR_RATIO)
            for target in section_targets
        ),
        sum(
            math.ceil(target * REPORT_AGGREGATE_WORD_CEILING_RATIO)
            for target in section_targets
        ),
    )


class ReportSectionKind(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    SCOPE_AND_EVIDENCE = "scope_and_evidence"
    KEY_FINDINGS = "key_findings"
    TREND_ANALYSIS = "trend_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    COMPOSITION_ANALYSIS = "composition_analysis"
    DRIVER_ANALYSIS = "driver_analysis"
    FORECAST_OUTLOOK = "forecast_outlook"
    SCENARIO_ANALYSIS = "scenario_analysis"
    CONTEXT_AND_FRAMEWORK = "context_and_framework"
    ANALYSIS = "analysis"
    IMPLICATIONS = "implications"
    LIMITATIONS = "limitations"
    CONCLUSION = "conclusion"


class ReportIntent(str, Enum):
    GENERAL = "general"
    TREND = "trend"
    COMPARISON = "comparison"
    COMPOSITION = "composition"
    DRIVER_ANALYSIS = "driver_analysis"
    FORECAST = "forecast"
    SCENARIO = "scenario"
    KNOWLEDGE = "knowledge"


REPORT_INTENT_CORE_SECTION = {
    ReportIntent.GENERAL: ReportSectionKind.KEY_FINDINGS,
    ReportIntent.TREND: ReportSectionKind.TREND_ANALYSIS,
    ReportIntent.COMPARISON: ReportSectionKind.COMPARATIVE_ANALYSIS,
    ReportIntent.COMPOSITION: ReportSectionKind.COMPOSITION_ANALYSIS,
    ReportIntent.DRIVER_ANALYSIS: ReportSectionKind.DRIVER_ANALYSIS,
    ReportIntent.FORECAST: ReportSectionKind.FORECAST_OUTLOOK,
    ReportIntent.SCENARIO: ReportSectionKind.SCENARIO_ANALYSIS,
    ReportIntent.KNOWLEDGE: ReportSectionKind.CONTEXT_AND_FRAMEWORK,
}


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
    intent: ReportIntent = ReportIntent.GENERAL,
) -> None:
    """Validate the deterministic section profile for one report intent."""

    core_kind = REPORT_INTENT_CORE_SECTION[intent]
    required_sequence = (
        ReportSectionKind.EXECUTIVE_SUMMARY,
        ReportSectionKind.SCOPE_AND_EVIDENCE,
        core_kind,
        ReportSectionKind.LIMITATIONS,
        ReportSectionKind.CONCLUSION,
    )
    missing = [
        kind.value
        for kind in required_sequence
        if kinds.count(kind) != 1
    ]
    if missing:
        raise ValueError(
            f"Report intent {intent.value} must contain each of its required "
            "standard sections "
            f"exactly once: {', '.join(missing)}."
        )

    if not (
        kinds[0] == ReportSectionKind.EXECUTIVE_SUMMARY
        and kinds[1] == ReportSectionKind.SCOPE_AND_EVIDENCE
        and kinds[2] == core_kind
        and kinds[-2] == ReportSectionKind.LIMITATIONS
        and kinds[-1] == ReportSectionKind.CONCLUSION
    ):
        raise ValueError(
            f"Report intent {intent.value} violates the standard section order: executive "
            f"summary, scope and evidence, {core_kind.value}, optional "
            "analysis/implications, limitations, conclusion."
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


REPORT_INTENT_ALLOWED_CHART_PURPOSES = {
    ReportIntent.GENERAL: frozenset(ReportChartPurpose),
    ReportIntent.TREND: frozenset(
        {
            ReportChartPurpose.TREND,
            ReportChartPurpose.COMPARISON,
            ReportChartPurpose.TABLE,
        }
    ),
    ReportIntent.COMPARISON: frozenset(
        {
            ReportChartPurpose.COMPARISON,
            ReportChartPurpose.TREND,
            ReportChartPurpose.TABLE,
        }
    ),
    ReportIntent.COMPOSITION: frozenset(
        {
            ReportChartPurpose.COMPOSITION,
            ReportChartPurpose.COMPARISON,
            ReportChartPurpose.TREND,
            ReportChartPurpose.TABLE,
        }
    ),
    ReportIntent.DRIVER_ANALYSIS: frozenset(
        {
            ReportChartPurpose.RELATIONSHIP,
            ReportChartPurpose.TREND,
            ReportChartPurpose.COMPARISON,
            ReportChartPurpose.TABLE,
        }
    ),
    ReportIntent.FORECAST: frozenset(
        {
            ReportChartPurpose.FORECAST,
            ReportChartPurpose.TREND,
            ReportChartPurpose.TABLE,
        }
    ),
    ReportIntent.SCENARIO: frozenset(
        {
            ReportChartPurpose.COMPARISON,
            ReportChartPurpose.FORECAST,
            ReportChartPurpose.TREND,
            ReportChartPurpose.TABLE,
        }
    ),
    ReportIntent.KNOWLEDGE: frozenset({ReportChartPurpose.TABLE}),
}


Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
EvidenceRef = Annotated[str, Field(min_length=1, max_length=256)]


class _StrictReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ReportPlanningContext(_StrictReportModel):
    contract_version: Literal["report-planning-context-v1"]
    intent: ReportIntent
    language_code: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
    request_objective: str = Field(min_length=1, max_length=2000)
    requires_table: bool
    source: Literal["question_analysis", "pipeline_fallback"]


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
    chart_refs: List[Identifier] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )

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
    intent: ReportIntent = ReportIntent.GENERAL
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
    charts: List[ReportChartRequest] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )

    @model_validator(mode="after")
    def _validate_standard_structure(self) -> "ReportPlan":
        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Report sections must use a unique section_id.")

        chart_ids = [chart.chart_id for chart in self.charts]
        if len(chart_ids) != len(set(chart_ids)):
            raise ValueError("Report charts must use a unique chart_id.")

        kinds = [section.kind for section in self.sections]
        validate_standard_report_section_order(kinds, self.intent)

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
            if (
                chart.purpose
                not in REPORT_INTENT_ALLOWED_CHART_PURPOSES[self.intent]
            ):
                raise ValueError(
                    f"Report intent {self.intent.value} does not allow chart "
                    f"purpose {chart.purpose.value}."
                )
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


def required_report_section_sequence(
    intent: ReportIntent,
) -> tuple[ReportSectionKind, ...]:
    """Return the section kinds required by an intent profile."""

    return (
        ReportSectionKind.EXECUTIVE_SUMMARY,
        ReportSectionKind.SCOPE_AND_EVIDENCE,
        REPORT_INTENT_CORE_SECTION[intent],
        ReportSectionKind.LIMITATIONS,
        ReportSectionKind.CONCLUSION,
    )


def normalize_report_plan_semantics(
    payload: Any,
    planning_context: ReportPlanningContext,
) -> Any:
    """Bind model output to code-owned intent, language, and profile rules."""

    if not isinstance(payload, dict):
        return payload
    normalized = deepcopy(payload)
    normalized["intent"] = planning_context.intent.value
    normalized["language_code"] = planning_context.language_code

    sections = normalized.get("sections")

    charts = normalized.get("charts")
    if isinstance(charts, list):
        allowed_purposes = {
            purpose.value
            for purpose in REPORT_INTENT_ALLOWED_CHART_PURPOSES[
                planning_context.intent
            ]
        }
        known_purposes = {
            purpose.value
            for purpose in ReportChartPurpose
        }
        retained_charts = [
            chart
            for chart in charts
            if (
                not isinstance(chart, dict)
                or chart.get("purpose") not in known_purposes
                or chart.get("purpose") in allowed_purposes
            )
        ]
        retained_chart_ids = {
            chart.get("chart_id")
            for chart in retained_charts
            if isinstance(chart, dict)
        }
        normalized["charts"] = retained_charts
        if isinstance(sections, list):
            for section in sections:
                if not isinstance(section, dict):
                    continue
                chart_refs = section.get("chart_refs")
                if isinstance(chart_refs, list):
                    section["chart_refs"] = [
                        chart_ref
                        for chart_ref in chart_refs
                        if chart_ref in retained_chart_ids
                    ]

    return normalize_report_plan_word_budget(normalized)
