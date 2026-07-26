"""Report-specific deterministic chart artifacts and build decisions."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.report_evidence import EvidenceRow


class ReportChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    TABLE = "table"


class _StrictChartModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class ReportChartMetadata(_StrictChartModel):
    title: str = Field(min_length=1, max_length=160)
    deterministic: Literal[True]
    evidence_refs: List[str] = Field(min_length=1, max_length=16)
    x_axis: str = Field(min_length=1, max_length=128)
    series: List[str] = Field(min_length=1, max_length=8)
    unit_by_series: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_metadata_references(self) -> "ReportChartMetadata":
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("Report chart evidence_refs must be unique.")
        if len(self.series) != len(set(self.series)):
            raise ValueError("Report chart series must be unique.")
        if set(self.unit_by_series) - set(self.series):
            raise ValueError("Report chart units must reference a known series.")
        return self


class ReportChartArtifact(_StrictChartModel):
    chart_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    type: ReportChartType
    data: List[EvidenceRow] = Field(min_length=1, max_length=200)
    metadata: ReportChartMetadata


class ReportChartBuildDecision(_StrictChartModel):
    chart_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    required: bool
    status: Literal["built", "omitted"]
    reason_code: str = Field(default="", pattern=r"^(?:|[A-Z][A-Z0-9_]{0,63})$")
    artifact: ReportChartArtifact | None = None

    @model_validator(mode="after")
    def _validate_decision(self) -> "ReportChartBuildDecision":
        if self.status == "built" and (self.artifact is None or self.reason_code):
            raise ValueError("A built report chart requires only an artifact.")
        if self.status == "omitted" and (self.artifact is not None or not self.reason_code):
            raise ValueError("An omitted report chart requires only a reason_code.")
        return self
