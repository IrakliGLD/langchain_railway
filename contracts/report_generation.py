"""Durable checkpoint contract for resumable multi-call report generation."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from contracts.report import ReportPlan, ReportPlanningContext
from contracts.report_document import (
    ReportDocumentDraft,
    ReportDocumentPlan,
)
from contracts.report_evidence import ReportEvidenceKind, ReportEvidenceManifest
from contracts.report_research import ReportEvidencePacket, ReportResearchPlan
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
        "report-generation-checkpoint-v3",
    ]
    checkpoint_stage: Literal[
        "evidence_ready",
        "plan_ready",
        "research_plan_ready",
        "evidence_collecting",
        "document_plan_ready",
        "draft_ready",
    ] | None = None
    manifest: ReportEvidenceManifest | None
    planning_context: ReportPlanningContext | None = None
    plan: ReportPlan | None = None
    completed_sections: List[ReportSectionDraft] = Field(
        default_factory=list,
        max_length=8,
    )
    research_plan: ReportResearchPlan | None = None
    completed_packets: List[ReportEvidencePacket] = Field(
        default_factory=list,
        max_length=8,
    )
    document_plan: ReportDocumentPlan | None = None
    document_draft: ReportDocumentDraft | None = None

    @model_validator(mode="after")
    def _validate_checkpoint_identity(self) -> "ReportGenerationCheckpoint":
        if self.contract_version == "report-generation-checkpoint-v1":
            if (
                self.checkpoint_stage is not None
                or self.planning_context is not None
                or self.manifest is None
                or self.plan is None
                or self.research_plan is not None
                or self.completed_packets
                or self.document_plan is not None
                or self.document_draft is not None
            ):
                raise ValueError(
                    "Checkpoint v1 requires a plan and cannot carry v2 stage "
                    "or planning-context fields."
                )
        elif self.contract_version == "report-generation-checkpoint-v2":
            if (
                self.checkpoint_stage not in {"evidence_ready", "plan_ready"}
                or self.planning_context is None
                or self.manifest is None
                or self.research_plan is not None
                or self.completed_packets
                or self.document_plan is not None
                or self.document_draft is not None
            ):
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
        else:
            self._validate_v3()

        if self.plan is not None:
            if self.manifest is None:
                raise ValueError(
                    "Report checkpoint plan requires a manifest."
                )
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

    def _validate_v3(self) -> None:
        if (
            self.checkpoint_stage
            not in {
                "research_plan_ready",
                "evidence_collecting",
                "evidence_ready",
                "document_plan_ready",
                "draft_ready",
            }
            or self.research_plan is None
            or self.planning_context is not None
            or self.plan is not None
            or self.completed_sections
        ):
            raise ValueError(
                "Checkpoint v3 requires a research stage and research plan "
                "without legacy planning fields."
            )

        packet_track_ids = [
            packet.track_id for packet in self.completed_packets
        ]
        if len(packet_track_ids) != len(set(packet_track_ids)):
            raise ValueError(
                "Checkpoint v3 packet track IDs must be unique."
            )
        research_track_ids = {
            track.track_id for track in self.research_plan.tracks
        }
        unknown_packets = sorted(
            set(packet_track_ids) - research_track_ids
        )
        if unknown_packets:
            raise ValueError(
                "Checkpoint v3 packets reference unknown research tracks: "
                + ", ".join(unknown_packets)
            )

        stage = self.checkpoint_stage
        if stage == "research_plan_ready":
            if (
                self.completed_packets
                or self.manifest is not None
                or self.document_plan is not None
                or self.document_draft is not None
            ):
                raise ValueError(
                    "A research_plan_ready checkpoint contains only the "
                    "research plan."
                )
            return
        if stage == "evidence_collecting":
            if (
                not self.completed_packets
                or self.manifest is not None
                or self.document_plan is not None
                or self.document_draft is not None
            ):
                raise ValueError(
                    "An evidence_collecting checkpoint requires packets "
                    "without a consolidated manifest."
                )
            return

        if self.manifest is None:
            raise ValueError(
                "Evidence-ready v3 checkpoints require a manifest."
            )
        if (
            self.manifest.query_digest != self.research_plan.query_digest
        ):
            raise ValueError(
                "Checkpoint v3 research plan and manifest query identity "
                "must match."
            )
        if self.completed_packets:
            raise ValueError(
                "Consolidated v3 checkpoints cannot duplicate evidence "
                "packets."
            )

        if stage == "evidence_ready":
            if (
                self.document_plan is not None
                or self.document_draft is not None
            ):
                raise ValueError(
                    "An evidence_ready v3 checkpoint cannot contain a "
                    "document plan or draft."
                )
            return
        if self.document_plan is None:
            raise ValueError(
                "Document-ready v3 checkpoints require a document plan."
            )
        if (
            self.document_plan.query_digest
            != self.research_plan.query_digest
            or self.document_plan.evidence_manifest_id
            != self.manifest.manifest_id
        ):
            raise ValueError(
                "Checkpoint v3 document plan must match its query and "
                "manifest."
            )
        planned_track_ids = (
            set(self.document_plan.required_track_ids)
            | set(self.document_plan.completed_track_ids)
            | set(self.document_plan.gap_track_ids)
        )
        if not planned_track_ids.issubset(research_track_ids):
            raise ValueError(
                "Checkpoint v3 document plan references unknown research "
                "tracks."
            )
        required_research_track_ids = {
            track.track_id
            for track in self.research_plan.tracks
            if track.required
        }
        if (
            set(self.document_plan.required_track_ids)
            != required_research_track_ids
        ):
            raise ValueError(
                "Checkpoint v3 document plan must preserve every required "
                "research track."
            )

        manifest_items = self.manifest.item_by_ref()
        manifest_refs = set(manifest_items)
        for section in self.document_plan.sections:
            if not set(section.required_evidence_refs).issubset(
                manifest_refs
            ):
                raise ValueError(
                    "Checkpoint v3 document plan references unknown manifest "
                    "evidence."
                )
        for chart in self.document_plan.charts:
            if not set(chart.evidence_refs).issubset(manifest_refs):
                raise ValueError(
                    "Checkpoint v3 document plan references unknown manifest "
                    "evidence."
                )
            if any(
                manifest_items[ref].kind is not ReportEvidenceKind.TABLE
                for ref in chart.evidence_refs
            ):
                raise ValueError(
                    "Checkpoint v3 document charts require table evidence."
                )

        if stage == "document_plan_ready":
            if self.document_draft is not None:
                raise ValueError(
                    "A document_plan_ready checkpoint cannot contain a "
                    "document draft."
                )
            return
        if self.document_draft is None:
            raise ValueError(
                "A draft_ready checkpoint requires a document draft."
            )
        if (
            self.document_draft.query_digest
            != self.research_plan.query_digest
            or self.document_draft.evidence_manifest_id
            != self.manifest.manifest_id
            or self.document_draft.coverage_status
            != self.document_plan.coverage_status
        ):
            raise ValueError(
                "Checkpoint v3 document draft must match its plan, query, "
                "and manifest."
            )
        plan_section_ids = [
            section.section_id for section in self.document_plan.sections
        ]
        draft_section_ids = [
            section.section_id
            for section in self.document_draft.generation_order_sections()
        ]
        if draft_section_ids != plan_section_ids:
            raise ValueError(
                "Checkpoint v3 document draft sections must match "
                "document-plan order."
            )

        draft_by_id = {
            section.section_id: section
            for section in self.document_draft.generation_order_sections()
        }
        for section_spec in self.document_plan.sections:
            draft_section = draft_by_id[section_spec.section_id]
            cited_refs = {
                ref
                for paragraph in draft_section.paragraphs
                for ref in paragraph.evidence_refs
            }
            cited_refs.update(
                claim.evidence_ref
                for paragraph in draft_section.paragraphs
                for claim in paragraph.direct_claims
            )
            cited_refs.update(
                operand.evidence_ref
                for paragraph in draft_section.paragraphs
                for claim in paragraph.derived_claims
                for operand in claim.operands
            )
            if not cited_refs.issubset(manifest_refs):
                raise ValueError(
                    "Checkpoint v3 document draft references unknown manifest "
                    "evidence."
                )
            if not set(section_spec.required_evidence_refs).issubset(
                cited_refs
            ):
                raise ValueError(
                    "Checkpoint v3 document draft omits required section "
                    "evidence."
                )

    def durable_json(self) -> str:
        """Return the exact size-checked JSON prepared during validation."""

        return self._durable_json
