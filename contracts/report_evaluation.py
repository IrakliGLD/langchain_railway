"""Content-free shadow evaluation for report plans and chart feasibility."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class ReportPlanEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["report-evaluation-v1"]
    manifest_id: str = Field(pattern=r"^manifest:[0-9a-f]{32}$")
    ready_for_generation: bool
    evidence_reference_coverage: float = Field(ge=0.0, le=1.0)
    required_section_coverage: float = Field(ge=0.0, le=1.0)
    required_chart_build_rate: float = Field(ge=0.0, le=1.0)
    findings: List[str] = Field(default_factory=list, max_length=32)
