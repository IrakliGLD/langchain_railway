"""Durable checkpoint contract for resumable multi-call report generation."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from contracts.report import ReportPlan, ReportPlanningContext
from contracts.report_evidence import ReportEvidenceManifest
from contracts.report_sections import ReportSectionDraft

REPORT_GENERATION_CHECKPOINT_MAX_BYTES = 1_048_576


class ReportCheckpointTooLargeError(ValueError):
    """The checkpoint is well formed but exceeds the durable payload ceiling.

    Distinct from the other checkpoint failures so callers can tell a payload
    that is too big from one that is structurally wrong.
    """


class ReportGenerationCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    _durable_json: str = PrivateAttr(default="")

    contract_version: Literal[
        "report-generation-checkpoint-v1",
        "report-generation-checkpoint-v2",
    ]
    checkpoint_stage: Literal["evidence_ready", "plan_ready"] | None = None
    manifest: ReportEvidenceManifest
    planning_context: ReportPlanningContext | None = None
    plan: ReportPlan | None = None
    completed_sections: List[ReportSectionDraft] = Field(
        default_factory=list,
        max_length=8,
    )

    @model_validator(mode="after")
    def _validate_checkpoint_identity(self) -> "ReportGenerationCheckpoint":
        if self.contract_version == "report-generation-checkpoint-v1":
            if (
                self.checkpoint_stage is not None
                or self.planning_context is not None
                or self.plan is None
            ):
                raise ValueError(
                    "Checkpoint v1 requires a plan and cannot carry v2 stage "
                    "or planning-context fields."
                )
        else:
            if self.checkpoint_stage is None or self.planning_context is None:
                raise ValueError(
                    "Checkpoint v2 requires checkpoint_stage and "
                    "planning_context."
                )
            if self.checkpoint_stage == "evidence_ready":
                if self.plan is not None or self.completed_sections:
                    raise ValueError(
                        "An evidence_ready checkpoint cannot carry a plan or "
                        "completed sections."
                    )
            elif self.plan is None:
                raise ValueError(
                    "A plan_ready checkpoint requires a report plan."
                )
            elif (
                self.plan.intent is not self.planning_context.intent
                or self.plan.language_code
                != self.planning_context.language_code
            ):
                raise ValueError(
                    "A plan_ready checkpoint plan must match its planning "
                    "context."
                )

        if self.plan is not None:
            if self.plan.evidence_manifest_id != self.manifest.manifest_id:
                raise ValueError(
                    "Report checkpoint plan and manifest identity must match."
                )
            plan_section_ids = {
                section.section_id for section in self.plan.sections
            }
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
                    "A completed report section is not present in the report "
                    "plan: "
                    + ", ".join(unknown_ids)
                )
        durable_json = self.model_dump_json(
            exclude_none=(
                self.contract_version
                == "report-generation-checkpoint-v1"
            )
        )
        if len(durable_json.encode("utf-8")) > (
            REPORT_GENERATION_CHECKPOINT_MAX_BYTES
        ):
            raise ReportCheckpointTooLargeError(
                "Report generation checkpoint exceeds 1 MiB."
            )
        self._durable_json = durable_json
        return self

    def durable_json(self) -> str:
        """Return the exact size-checked JSON prepared during validation."""

        return self._durable_json
