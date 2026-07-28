"""Public durable result contract for a completed analytical report."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.report import (
    REPORT_MAX_EXHIBITS,
    STANDARD_REPORT_RESULT_MAX_WORDS,
    STANDARD_REPORT_RESULT_MIN_WORDS,
    ReportIntent,
    ReportSectionKind,
    validate_standard_report_section_order,
)
from contracts.report_charts import ReportChartArtifact
from contracts.report_document import ReportDocumentSectionRole


class _StrictResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ReportResultSection(_StrictResultModel):
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    kind: ReportSectionKind
    title: str = Field(min_length=1, max_length=160)
    content_markdown: str = Field(min_length=20, max_length=50_000)
    evidence_refs: List[str] = Field(min_length=1, max_length=32)
    chart_refs: List[str] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )
    word_count: int = Field(ge=1, le=5000)

    @field_validator("evidence_refs", "chart_refs")
    @classmethod
    def _require_unique_references(cls, refs: List[str]) -> List[str]:
        if len(refs) != len(set(refs)):
            raise ValueError("Report result references must be unique.")
        return refs


class ReportCitation(_StrictResultModel):
    evidence_ref: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=200)
    source: str = Field(min_length=1, max_length=64)
    provenance_refs: List[str] = Field(default_factory=list, max_length=32)


class ReportChartOmission(_StrictResultModel):
    """A planned chart the deterministic builder could not produce."""

    chart_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    title: str = Field(min_length=1, max_length=160)
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")


class ReportResult(_StrictResultModel):
    contract_version: Literal["report-result-v1"]
    intent: ReportIntent = ReportIntent.GENERAL
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=1000)
    language_code: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
    evidence_manifest_id: str = Field(pattern=r"^manifest:[0-9a-f]{32}$")
    content_markdown: str = Field(min_length=100, max_length=500_000)
    sections: List[ReportResultSection] = Field(min_length=5, max_length=8)
    charts: List[ReportChartArtifact] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )
    omitted_charts: List[ReportChartOmission] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )
    citations: List[ReportCitation] = Field(min_length=1, max_length=32)
    word_count: int = Field(
        ge=STANDARD_REPORT_RESULT_MIN_WORDS,
        le=STANDARD_REPORT_RESULT_MAX_WORDS,
    )

    @model_validator(mode="after")
    def _validate_result_identity(self) -> "ReportResult":
        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Report result section IDs must be unique.")
        chart_ids = [chart.chart_id for chart in self.charts]
        if len(chart_ids) != len(set(chart_ids)):
            raise ValueError("Report result chart IDs must be unique.")
        citation_refs = [citation.evidence_ref for citation in self.citations]
        if len(citation_refs) != len(set(citation_refs)):
            raise ValueError("Report result citation refs must be unique.")
        validate_standard_report_section_order(
            [section.kind for section in self.sections],
            self.intent,
        )

        known_section_ids = set(section_ids)
        known_chart_ids = set(chart_ids)
        section_by_id = {
            section.section_id: section for section in self.sections
        }
        for section in self.sections:
            unknown_chart_refs = set(section.chart_refs) - known_chart_ids
            if unknown_chart_refs:
                raise ValueError(
                    f"Report section {section.section_id} references an unknown chart."
                )
        for chart in self.charts:
            if chart.section_id not in known_section_ids:
                raise ValueError("Report chart references an unknown section.")
            assigned_section = section_by_id[chart.section_id]
            if chart.chart_id not in assigned_section.chart_refs:
                raise ValueError(
                    f"Report chart {chart.chart_id} must be referenced by its "
                    "assigned section."
                )
            if any(
                chart.chart_id in section.chart_refs
                for section in self.sections
                if section.section_id != chart.section_id
            ):
                raise ValueError(
                    f"Report chart {chart.chart_id} may only be referenced by its "
                    "assigned section."
                )

        used_evidence_refs = {
            ref
            for section in self.sections
            for ref in section.evidence_refs
        }
        used_evidence_refs.update(
            ref
            for chart in self.charts
            for ref in chart.metadata.evidence_refs
        )
        if set(citation_refs) != used_evidence_refs:
            raise ValueError(
                "Report citations must exactly cover section and chart evidence."
            )
        if sum(section.word_count for section in self.sections) != self.word_count:
            raise ValueError("Report result word_count must equal its section totals.")
        return self


class ReportResultV2Section(_StrictResultModel):
    """One adaptive section in a track-driven report."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    kind: ReportDocumentSectionRole
    title: str = Field(min_length=1, max_length=160)
    content_markdown: str = Field(min_length=20, max_length=50_000)
    evidence_refs: List[str] = Field(min_length=1, max_length=32)
    chart_refs: List[str] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )
    word_count: int = Field(ge=1, le=5000)

    @field_validator("evidence_refs", "chart_refs")
    @classmethod
    def _require_unique_v2_references(
        cls,
        refs: List[str],
    ) -> List[str]:
        if len(refs) != len(set(refs)):
            raise ValueError("Report result references must be unique.")
        return refs


class ReportResultV2(_StrictResultModel):
    """Public durable result for the adaptive research/document pipeline."""

    contract_version: Literal["report-result-v2"]
    intent: Literal["general"] = "general"
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=1000)
    language_code: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
    evidence_manifest_id: str = Field(pattern=r"^manifest:[0-9a-f]{32}$")
    coverage_status: Literal["ready", "ready_with_gaps"]
    content_markdown: str = Field(min_length=100, max_length=500_000)
    sections: List[ReportResultV2Section] = Field(min_length=3, max_length=8)
    charts: List[ReportChartArtifact] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )
    omitted_charts: List[ReportChartOmission] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )
    citations: List[ReportCitation] = Field(min_length=1, max_length=32)
    word_count: int = Field(
        ge=STANDARD_REPORT_RESULT_MIN_WORDS,
        le=STANDARD_REPORT_RESULT_MAX_WORDS,
    )

    @model_validator(mode="after")
    def _validate_v2_result(self) -> "ReportResultV2":
        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Report result section IDs must be unique.")
        roles = [section.kind for section in self.sections]
        if (
            roles[0] is not ReportDocumentSectionRole.EXECUTIVE_SUMMARY
            or roles.count(ReportDocumentSectionRole.EXECUTIVE_SUMMARY) != 1
        ):
            raise ValueError(
                "An adaptive report requires one executive summary first."
            )
        if roles.count(ReportDocumentSectionRole.ANALYSIS) < 1:
            raise ValueError(
                "An adaptive report requires at least one analysis section."
            )
        if roles.count(ReportDocumentSectionRole.IMPLICATIONS) > 1:
            raise ValueError(
                "An adaptive report may contain at most one implications "
                "section."
            )
        if roles.count(ReportDocumentSectionRole.LIMITATIONS) != 1:
            raise ValueError(
                "An adaptive report requires exactly one limitations section."
            )
        if roles.count(ReportDocumentSectionRole.CONCLUSION) > 1:
            raise ValueError(
                "An adaptive report may contain at most one conclusion."
            )
        conclusion_present = ReportDocumentSectionRole.CONCLUSION in roles
        limitations_index = roles.index(
            ReportDocumentSectionRole.LIMITATIONS
        )
        if limitations_index != (
            len(roles) - 2 if conclusion_present else len(roles) - 1
        ):
            raise ValueError(
                "Adaptive report limitations must follow the analysis."
            )
        if conclusion_present and (
            roles[-1] is not ReportDocumentSectionRole.CONCLUSION
        ):
            raise ValueError(
                "An adaptive report conclusion must be last."
            )
        body_roles = roles[1:limitations_index]
        if ReportDocumentSectionRole.IMPLICATIONS in body_roles:
            implications_index = body_roles.index(
                ReportDocumentSectionRole.IMPLICATIONS
            )
            if (
                implications_index == 0
                or any(
                    role is not ReportDocumentSectionRole.ANALYSIS
                    for role in body_roles[:implications_index]
                )
                or implications_index != len(body_roles) - 1
            ):
                raise ValueError(
                    "Adaptive report implications must follow all analysis "
                    "sections."
                )
        elif any(
            role is not ReportDocumentSectionRole.ANALYSIS
            for role in body_roles
        ):
            raise ValueError(
                "Adaptive report body roles are out of order."
            )

        chart_ids = [chart.chart_id for chart in self.charts]
        if len(chart_ids) != len(set(chart_ids)):
            raise ValueError("Report result chart IDs must be unique.")
        citation_refs = [citation.evidence_ref for citation in self.citations]
        if len(citation_refs) != len(set(citation_refs)):
            raise ValueError("Report result citation refs must be unique.")
        known_chart_ids = set(chart_ids)
        section_by_id = {
            section.section_id: section for section in self.sections
        }
        for section in self.sections:
            if set(section.chart_refs) - known_chart_ids:
                raise ValueError(
                    "Report section references an unknown chart."
                )
        for chart in self.charts:
            section = section_by_id.get(chart.section_id)
            if section is None or chart.chart_id not in section.chart_refs:
                raise ValueError(
                    "Report chart must be referenced by its assigned section."
                )
            if any(
                chart.chart_id in candidate.chart_refs
                for candidate in self.sections
                if candidate.section_id != chart.section_id
            ):
                raise ValueError(
                    "Report chart may only be referenced by its assigned "
                    "section."
                )
        used_evidence_refs = {
            ref
            for section in self.sections
            for ref in section.evidence_refs
        }
        used_evidence_refs.update(
            ref
            for chart in self.charts
            for ref in chart.metadata.evidence_refs
        )
        if set(citation_refs) != used_evidence_refs:
            raise ValueError(
                "Report citations must exactly cover section and chart "
                "evidence."
            )
        if sum(section.word_count for section in self.sections) != self.word_count:
            raise ValueError(
                "Report result word_count must equal its section totals."
            )
        return self
