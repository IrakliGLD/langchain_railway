"""Contracts for adaptive whole-document report planning and drafting."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.report import (
    REPORT_SECTION_MAX_WORDS,
    REPORT_SECTION_MIN_WORDS,
    STANDARD_REPORT_MAX_SECTIONS,
    STANDARD_REPORT_MAX_WORDS,
    STANDARD_REPORT_MIN_WORDS,
    ReportChartRequest,
)
from contracts.report_research import FindingCode, ReportCoverageStatus
from contracts.report_sections import ReportSectionDraft

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
EvidenceRef = Annotated[str, Field(min_length=1, max_length=256)]


class ReportDocumentSectionRole(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    ANALYSIS = "analysis"
    IMPLICATIONS = "implications"
    LIMITATIONS = "limitations"
    CONCLUSION = "conclusion"


class _StrictDocumentModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


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
    chart_refs: List[Identifier] = Field(default_factory=list, max_length=3)

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
    target_words: int = Field(
        ge=STANDARD_REPORT_MIN_WORDS,
        le=STANDARD_REPORT_MAX_WORDS,
    )
    evidence_manifest_id: str = Field(pattern=r"^manifest:[0-9a-f]{32}$")
    coverage_status: Literal["ready", "ready_with_gaps"]
    required_track_ids: List[Identifier] = Field(min_length=1, max_length=8)
    completed_track_ids: List[Identifier] = Field(min_length=1, max_length=8)
    gap_track_ids: List[Identifier] = Field(default_factory=list, max_length=8)
    sections: List[ReportDocumentSectionSpec] = Field(
        min_length=3,
        max_length=STANDARD_REPORT_MAX_SECTIONS,
    )
    charts: List[ReportChartRequest] = Field(default_factory=list, max_length=3)

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
        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Document plan section IDs must be unique.")
        roles = [section.role for section in self.sections]
        if (
            roles[0] is not ReportDocumentSectionRole.EXECUTIVE_SUMMARY
            or roles.count(ReportDocumentSectionRole.EXECUTIVE_SUMMARY) != 1
        ):
            raise ValueError(
                "A document plan requires one executive summary first."
            )
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
        if roles.count(ReportDocumentSectionRole.CONCLUSION) > 1:
            raise ValueError(
                "A document plan may contain at most one conclusion."
            )
        limitations_index = roles.index(
            ReportDocumentSectionRole.LIMITATIONS
        )
        conclusion_present = (
            ReportDocumentSectionRole.CONCLUSION in roles
        )
        expected_limitations_index = (
            len(roles) - 2 if conclusion_present else len(roles) - 1
        )
        if limitations_index != expected_limitations_index:
            raise ValueError(
                "The limitations section must follow analysis and "
                "implications."
            )
        if conclusion_present and (
            roles[-1] is not ReportDocumentSectionRole.CONCLUSION
        ):
            raise ValueError("The optional conclusion must be last.")
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
    analytical_sections: List[ReportSectionDraft] = Field(
        min_length=1,
        max_length=5,
    )
    implications_section: ReportSectionDraft | None = None
    limitations_section: ReportSectionDraft
    conclusion_section: ReportSectionDraft | None = None
    executive_summary: ReportSectionDraft

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
        """Return body-first generation order with the summary last."""

        sections = list(self.analytical_sections)
        if self.implications_section is not None:
            sections.append(self.implications_section)
        sections.append(self.limitations_section)
        if self.conclusion_section is not None:
            sections.append(self.conclusion_section)
        sections.append(self.executive_summary)
        return sections

    def display_order_sections(self) -> List[ReportSectionDraft]:
        """Return user-facing order with the summary first."""

        generated = self.generation_order_sections()
        return [self.executive_summary, *generated[:-1]]


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
    word_count: int = Field(ge=0, le=5000)

    @field_validator("section_errors")
    @classmethod
    def _validate_section_errors(
        cls,
        section_errors: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        for errors in section_errors.values():
            if not errors or len(errors) > 16:
                raise ValueError(
                    "Document validation section errors must be bounded."
                )
            if len(errors) != len(set(errors)):
                raise ValueError(
                    "Document validation section errors must be unique."
                )
        return section_errors

    @field_validator("document_errors")
    @classmethod
    def _validate_document_errors(cls, errors: List[str]) -> List[str]:
        if len(errors) != len(set(errors)):
            raise ValueError(
                "Document validation errors must be unique."
            )
        return errors

    @model_validator(mode="after")
    def _validate_result(self) -> "ReportDocumentValidation":
        has_errors = bool(self.section_errors or self.document_errors)
        if self.valid == has_errors:
            raise ValueError(
                "Document validation valid must match its error fields."
            )
        return self
