"""Durable checkpoint contract for resumable multi-call report generation."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.report import ReportPlan
from contracts.report_evidence import ReportEvidenceManifest
from contracts.report_sections import ReportSectionDraft

REPORT_GENERATION_CHECKPOINT_MAX_BYTES = 1_048_576


class ReportGenerationCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    contract_version: Literal["report-generation-checkpoint-v1"]
    manifest: ReportEvidenceManifest
    plan: ReportPlan
    completed_sections: List[ReportSectionDraft] = Field(
        default_factory=list,
        max_length=8,
    )

    @model_validator(mode="after")
    def _validate_checkpoint_identity(self) -> "ReportGenerationCheckpoint":
        if self.plan.evidence_manifest_id != self.manifest.manifest_id:
            raise ValueError(
                "Report checkpoint plan and manifest identity must match."
            )
        plan_section_ids = {section.section_id for section in self.plan.sections}
        completed_ids = [
            section.section_id for section in self.completed_sections
        ]
        if len(completed_ids) != len(set(completed_ids)):
            raise ValueError(
                "Report checkpoint completed section IDs must be unique."
            )
        unknown_ids = sorted(set(completed_ids) - plan_section_ids)
        if unknown_ids:
            raise ValueError(
                "A completed report section is not present in the report plan: "
                + ", ".join(unknown_ids)
            )
        if len(self.model_dump_json().encode("utf-8")) > (
            REPORT_GENERATION_CHECKPOINT_MAX_BYTES
        ):
            raise ValueError("Report generation checkpoint exceeds 1 MiB.")
        return self
