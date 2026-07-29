"""Contracts for adaptive whole-document report planning and drafting."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.report import (
    REPORT_MAX_EXHIBITS,
    REPORT_SECTION_MAX_WORDS,
    REPORT_SECTION_MIN_WORDS,
    STANDARD_REPORT_MAX_SECTIONS,
    STANDARD_REPORT_MAX_WORDS,
    ReportChartRequest,
)
from contracts.report_research import FindingCode, ReportCoverageStatus
from contracts.report_sections import ReportSectionDraft

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
EvidenceRef = Annotated[str, Field(min_length=1, max_length=256)]


class ReportDocumentSectionRole(str, Enum):
    ANALYSIS = "analysis"
    IMPLICATIONS = "implications"
    LIMITATIONS = "limitations"


class ReportDocumentProfile(str, Enum):
    COMPACT = "compact"
    FOCUSED = "focused"
    FULL = "full"


class _StrictDocumentModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class ReportEvidenceCapacity(_StrictDocumentModel):
    profile: ReportDocumentProfile
    usable_track_count: int = Field(ge=0, le=8)
    complete_track_count: int = Field(ge=0, le=8)
    partial_track_count: int = Field(ge=0, le=8)
    unavailable_track_count: int = Field(ge=0, le=8)
    usable_exhibit_count: int = Field(ge=0, le=REPORT_MAX_EXHIBITS)
    validated_finding_count: int = Field(ge=0, le=256)

    @model_validator(mode="after")
    def _validate_track_counts(self) -> "ReportEvidenceCapacity":
        if self.usable_track_count != (
            self.complete_track_count + self.partial_track_count
        ):
            raise ValueError(
                "Usable evidence tracks must equal complete plus partial "
                "tracks."
            )
        if (
            self.usable_track_count + self.unavailable_track_count
            > 8
        ):
            raise ValueError("Evidence capacity track counts exceed eight.")
        return self


class ReportDocumentSectionSpec(_StrictDocumentModel):
    section_id: Identifier
    role: ReportDocumentSectionRole
    title: str = Field(min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=600)
    target_words: int = Field(
        ge=REPORT_SECTION_MIN_WORDS,
        le=REPORT_SECTION_MAX_WORDS,
    )
    track_ids: List[Identifier] = Field(min_length=1, max_length=8)
    required_evidence_refs: List[EvidenceRef] = Field(
        min_length=1,
        max_length=32,
    )
    chart_refs: List[Identifier] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )

    @field_validator(
        "track_ids",
        "required_evidence_refs",
        "chart_refs",
    )
    @classmethod
    def _require_unique_values(cls, values: list) -> list:
        if len(values) != len(set(values)):
            raise ValueError("Document section lists must be unique.")
        return values


class ReportDocumentPlan(_StrictDocumentModel):
    contract_version: Literal["report-document-plan-v1"]
    query_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=1000)
    language_code: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
    profile: ReportDocumentProfile
    evidence_capacity: ReportEvidenceCapacity
    target_words: int = Field(
        ge=REPORT_SECTION_MIN_WORDS * 2,
        le=STANDARD_REPORT_MAX_WORDS,
    )
    evidence_manifest_id: str = Field(pattern=r"^manifest:[0-9a-f]{32}$")
    coverage_status: Literal["ready", "ready_with_gaps"]
    required_track_ids: List[Identifier] = Field(min_length=1, max_length=8)
    completed_track_ids: List[Identifier] = Field(min_length=1, max_length=8)
    gap_track_ids: List[Identifier] = Field(default_factory=list, max_length=8)
    sections: List[ReportDocumentSectionSpec] = Field(
        min_length=2,
        max_length=STANDARD_REPORT_MAX_SECTIONS,
    )
    charts: List[ReportChartRequest] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )

    @field_validator(
        "required_track_ids",
        "completed_track_ids",
        "gap_track_ids",
    )
    @classmethod
    def _require_unique_track_ids(cls, values: List[str]) -> List[str]:
        if len(values) != len(set(values)):
            raise ValueError("Document plan track ID lists must be unique.")
        return values

    @model_validator(mode="after")
    def _validate_document_plan(self) -> "ReportDocumentPlan":
        if self.evidence_capacity.profile is not self.profile:
            raise ValueError(
                "Document profile must match its evidence capacity."
            )
        if self.evidence_capacity.usable_track_count != len(
            self.completed_track_ids
        ):
            raise ValueError(
                "Document evidence capacity must match completed tracks."
            )
        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Document plan section IDs must be unique.")
        roles = [section.role for section in self.sections]
        if roles.count(ReportDocumentSectionRole.ANALYSIS) < 1:
            raise ValueError(
                "A document plan requires at least one analysis section."
            )
        if roles.count(ReportDocumentSectionRole.IMPLICATIONS) > 1:
            raise ValueError(
                "A document plan may contain at most one implications section."
            )
        if roles.count(ReportDocumentSectionRole.LIMITATIONS) != 1:
            raise ValueError(
                "A document plan requires exactly one limitations section."
            )
        limitations_index = roles.index(
            ReportDocumentSectionRole.LIMITATIONS
        )
        if limitations_index != len(roles) - 1:
            raise ValueError(
                "The limitations section must be last."
            )
        if ReportDocumentSectionRole.IMPLICATIONS in roles:
            implications_index = roles.index(
                ReportDocumentSectionRole.IMPLICATIONS
            )
            last_analysis_index = max(
                index
                for index, role in enumerate(roles)
                if role is ReportDocumentSectionRole.ANALYSIS
            )
            if not last_analysis_index < implications_index < limitations_index:
                raise ValueError(
                    "Implications must follow all analysis sections and "
                    "precede limitations."
                )

        if sum(section.target_words for section in self.sections) != (
            self.target_words
        ):
            raise ValueError(
                "Document section target_words must sum to target_words."
            )

        required = set(self.required_track_ids)
        completed = set(self.completed_track_ids)
        gaps = set(self.gap_track_ids)
        if not required.issubset(completed | gaps):
            raise ValueError(
                "Every required track must be completed or recorded as a gap."
            )
        if self.coverage_status == ReportCoverageStatus.READY.value:
            if gaps or not required.issubset(completed):
                raise ValueError(
                    "A ready document plan requires all required tracks "
                    "completed and no gaps."
                )
        elif not gaps:
            raise ValueError(
                "A ready_with_gaps document plan requires gap_track_ids."
            )

        analysis_track_ids = {
            track_id
            for section in self.sections
            if section.role is ReportDocumentSectionRole.ANALYSIS
            for track_id in section.track_ids
        }
        missing_analysis = sorted(completed - analysis_track_ids)
        if missing_analysis:
            raise ValueError(
                "Every completed track requires an analysis section: "
                + ", ".join(missing_analysis)
            )
        known_tracks = completed | gaps
        for section in self.sections:
            unknown_tracks = sorted(set(section.track_ids) - known_tracks)
            if unknown_tracks:
                raise ValueError(
                    "Document sections reference unknown tracks: "
                    + ", ".join(unknown_tracks)
                )

        chart_ids = [chart.chart_id for chart in self.charts]
        if len(chart_ids) != len(set(chart_ids)):
            raise ValueError("Document plan chart IDs must be unique.")
        known_chart_ids = set(chart_ids)
        section_by_id = {
            section.section_id: section for section in self.sections
        }
        for section in self.sections:
            unknown_charts = sorted(
                set(section.chart_refs) - known_chart_ids
            )
            if unknown_charts:
                raise ValueError(
                    "Document section references unknown charts: "
                    + ", ".join(unknown_charts)
                )
        for chart in self.charts:
            if chart.section_id not in section_by_id:
                raise ValueError(
                    "Document chart references an unknown section."
                )
            if chart.chart_id not in section_by_id[
                chart.section_id
            ].chart_refs:
                raise ValueError(
                    "Document chart must be referenced by its assigned "
                    "section."
                )
        return self


class ReportDocumentDraft(_StrictDocumentModel):
    contract_version: Literal["report-document-draft-v1"]
    query_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_manifest_id: str = Field(pattern=r"^manifest:[0-9a-f]{32}$")
    coverage_status: Literal["ready", "ready_with_gaps"]
    sections: List[ReportSectionDraft] = Field(
        min_length=2,
        max_length=STANDARD_REPORT_MAX_SECTIONS,
    )

    @model_validator(mode="after")
    def _validate_section_identity(self) -> "ReportDocumentDraft":
        section_ids = [
            section.section_id
            for section in self.generation_order_sections()
        ]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError(
                "Report document draft section IDs must be unique."
            )
        return self

    def generation_order_sections(self) -> List[ReportSectionDraft]:
        """Return the deterministic plan-owned section order."""

        return list(self.sections)

    def display_order_sections(self) -> List[ReportSectionDraft]:
        """Return the deterministic plan-owned section order."""

        return list(self.sections)


class ReportDocumentRepair(_StrictDocumentModel):
    contract_version: Literal["report-document-repair-v1"]
    sections: List[ReportSectionDraft] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _validate_section_ids(self) -> "ReportDocumentRepair":
        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Document repair section IDs must be unique.")
        return self


class ReportDocumentValidation(_StrictDocumentModel):
    contract_version: Literal["report-document-validation-v1"]
    valid: bool
    section_errors: Dict[Identifier, List[FindingCode]] = Field(
        default_factory=dict,
        max_length=8,
    )
    document_errors: List[FindingCode] = Field(
        default_factory=list,
        max_length=16,
    )
    section_warnings: Dict[Identifier, List[FindingCode]] = Field(
        default_factory=dict,
        max_length=8,
    )
    document_warnings: List[FindingCode] = Field(
        default_factory=list,
        max_length=16,
    )
    word_count: int = Field(ge=0, le=5000)

    @field_validator("section_errors", "section_warnings")
    @classmethod
    def _validate_section_findings(
        cls,
        section_findings: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        for findings in section_findings.values():
            if not findings or len(findings) > 16:
                raise ValueError(
                    "Document validation section findings must be bounded."
                )
            if len(findings) != len(set(findings)):
                raise ValueError(
                    "Document validation section findings must be unique."
                )
        return section_findings

    @field_validator("document_errors", "document_warnings")
    @classmethod
    def _validate_document_findings(
        cls,
        findings: List[str],
    ) -> List[str]:
        if len(findings) != len(set(findings)):
            raise ValueError(
                "Document validation findings must be unique."
            )
        return findings

    @model_validator(mode="after")
    def _validate_result(self) -> "ReportDocumentValidation":
        has_errors = bool(self.section_errors or self.document_errors)
        if self.valid == has_errors:
            raise ValueError(
                "Document validation valid must match its error fields."
            )
        return self
