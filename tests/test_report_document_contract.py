"""Adaptive whole-document planning, drafting, and checkpoint contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from contracts.report_document import (
    ReportDocumentDraft,
    ReportDocumentPlan,
    ReportDocumentSectionSpec,
)
from contracts.report_generation import ReportGenerationCheckpoint
from contracts.report_research import (
    ReportEvidencePacket,
    ReportResearchPlan,
    ReportResearchTrack,
)
from contracts.report_result import ReportResult, ReportResultV2
from tests.test_report_planner import _manifest
from tests.test_report_research_contract import (
    _complete_packet_payload,
    _research_plan_payload,
)

_TABLE_REF = "evidence:table:" + "1" * 16
_KNOWLEDGE_REF = "evidence:knowledge:" + "2" * 16
_LIMITATION_REF = "evidence:limitation:" + "3" * 16


def _document_plan_payload() -> dict:
    return {
        "contract_version": "report-document-plan-v1",
        "query_digest": "a" * 64,
        "title": "Electricity prices, security, and market design",
        "objective": "Assess observed market evidence without unsupported causal claims.",
        "language_code": "en",
        "target_words": 1100,
        "evidence_manifest_id": "manifest:" + "4" * 32,
        "coverage_status": "ready",
        "required_track_ids": ["prices", "security", "market_model"],
        "completed_track_ids": ["prices", "security", "market_model"],
        "gap_track_ids": [],
        "sections": [
            {
                "section_id": "executive_summary",
                "role": "executive_summary",
                "title": "Executive summary",
                "objective": "Summarize the strongest cross-track findings.",
                "target_words": 150,
                "track_ids": ["prices", "security", "market_model"],
                "required_evidence_refs": [_TABLE_REF, _KNOWLEDGE_REF],
                "chart_refs": [],
            },
            {
                "section_id": "price_dynamics",
                "role": "analysis",
                "title": "Price dynamics",
                "objective": "Present observed price levels and changes.",
                "target_words": 250,
                "track_ids": ["prices"],
                "required_evidence_refs": [_TABLE_REF],
                "chart_refs": ["price_trend"],
            },
            {
                "section_id": "energy_security",
                "role": "analysis",
                "title": "Energy security",
                "objective": "Assess the available security indicators.",
                "target_words": 220,
                "track_ids": ["security"],
                "required_evidence_refs": [_TABLE_REF],
                "chart_refs": ["security_mix"],
            },
            {
                "section_id": "market_model",
                "role": "analysis",
                "title": "Market model and legislation",
                "objective": "Describe the documented status and target model.",
                "target_words": 200,
                "track_ids": ["market_model"],
                "required_evidence_refs": [_KNOWLEDGE_REF],
                "chart_refs": [],
            },
            {
                "section_id": "implications",
                "role": "implications",
                "title": "Implications",
                "objective": "Synthesize relationships without claiming causality.",
                "target_words": 160,
                "track_ids": ["prices", "security", "market_model"],
                "required_evidence_refs": [_TABLE_REF, _KNOWLEDGE_REF],
                "chart_refs": [],
            },
            {
                "section_id": "limitations",
                "role": "limitations",
                "title": "Method and limitations",
                "objective": "State evidence gaps and analytical boundaries.",
                "target_words": 120,
                "track_ids": ["prices", "security", "market_model"],
                "required_evidence_refs": [_LIMITATION_REF],
                "chart_refs": [],
            },
        ],
        "charts": [
            {
                "chart_id": "price_trend",
                "section_id": "price_dynamics",
                "purpose": "trend",
                "title": "Monthly price",
                "evidence_refs": [_TABLE_REF],
                "x_field": "period",
                "series_fields": ["price_gel"],
                "required": True,
            },
            {
                "chart_id": "security_mix",
                "section_id": "energy_security",
                "purpose": "composition",
                "title": "Supply composition",
                "evidence_refs": [_TABLE_REF],
                "x_field": "period",
                "series_fields": ["import_dependency_ratio"],
                "required": True,
            },
        ],
    }


def _section_draft(section_id: str, title: str, evidence_ref: str) -> dict:
    return {
        "contract_version": "report-section-v1",
        "section_id": section_id,
        "title": title,
        "paragraphs": [
            {
                "text": (
                    f"{title} is presented only from the assigned evidence "
                    "and does not infer unavailable causes."
                ),
                "evidence_refs": [evidence_ref],
                "direct_claims": [],
                "derived_claims": [],
            }
        ],
    }


def _document_draft_payload() -> dict:
    return {
        "contract_version": "report-document-draft-v1",
        "query_digest": "a" * 64,
        "evidence_manifest_id": "manifest:" + "4" * 32,
        "coverage_status": "ready",
        "analytical_sections": [
            _section_draft("price_dynamics", "Price dynamics", _TABLE_REF),
            _section_draft("energy_security", "Energy security", _TABLE_REF),
            _section_draft(
                "market_model",
                "Market model and legislation",
                _KNOWLEDGE_REF,
            ),
        ],
        "implications_section": _section_draft(
            "implications",
            "Implications",
            _TABLE_REF,
        ),
        "limitations_section": _section_draft(
            "limitations",
            "Method and limitations",
            _LIMITATION_REF,
        ),
        "conclusion_section": None,
        "executive_summary": _section_draft(
            "executive_summary",
            "Executive summary",
            _TABLE_REF,
        ),
    }


def test_report_contracts_publish_a_four_exhibit_limit():
    assert (
        ReportResearchTrack.model_json_schema()["properties"][
            "expected_exhibits"
        ]["maxItems"]
        == 4
    )
    assert (
        ReportEvidencePacket.model_json_schema()["properties"][
            "chart_candidates"
        ]["maxItems"]
        == 4
    )
    assert (
        ReportDocumentSectionSpec.model_json_schema()["properties"][
            "chart_refs"
        ]["maxItems"]
        == 4
    )
    assert (
        ReportDocumentPlan.model_json_schema()["properties"]["charts"][
            "maxItems"
        ]
        == 4
    )
    for result_model in (ReportResult, ReportResultV2):
        schema = result_model.model_json_schema()["properties"]
        assert schema["charts"]["maxItems"] == 4
        assert schema["omitted_charts"]["maxItems"] == 4


def test_document_plan_supports_track_driven_structure_without_single_intent():
    plan = ReportDocumentPlan.model_validate(_document_plan_payload())

    assert [section.role.value for section in plan.sections] == [
        "executive_summary",
        "analysis",
        "analysis",
        "analysis",
        "implications",
        "limitations",
    ]
    assert plan.target_words == sum(
        section.target_words for section in plan.sections
    )
    assert ReportDocumentPlan.model_json_schema()["additionalProperties"] is False


def test_document_plan_rejects_missing_track_assignment_and_invalid_gap_status():
    payload = _document_plan_payload()
    payload["sections"] = [
        section
        for section in payload["sections"]
        if section["section_id"] != "energy_security"
    ]
    for section in payload["sections"]:
        if section["section_id"] == "price_dynamics":
            section["target_words"] += 220
    payload["target_words"] = sum(
        section["target_words"] for section in payload["sections"]
    )
    with pytest.raises(ValidationError, match="completed track"):
        ReportDocumentPlan.model_validate(payload)

    payload = _document_plan_payload()
    payload["coverage_status"] = "ready_with_gaps"
    with pytest.raises(ValidationError, match="gap_track_ids"):
        ReportDocumentPlan.model_validate(payload)


def test_document_plan_rejects_fixed_structure_regressions():
    payload = _document_plan_payload()
    payload["sections"][0]["role"] = "analysis"
    with pytest.raises(ValidationError, match="executive summary"):
        ReportDocumentPlan.model_validate(payload)

    payload = _document_plan_payload()
    payload["sections"] = [
        section
        for section in payload["sections"]
        if section["role"] != "limitations"
    ]
    payload["target_words"] = sum(
        section["target_words"] for section in payload["sections"]
    )
    with pytest.raises(ValidationError, match="limitations"):
        ReportDocumentPlan.model_validate(payload)


def test_document_draft_encodes_body_first_generation_and_summary_first_display():
    draft = ReportDocumentDraft.model_validate(_document_draft_payload())

    assert [
        section.section_id for section in draft.generation_order_sections()
    ] == [
        "price_dynamics",
        "energy_security",
        "market_model",
        "implications",
        "limitations",
        "executive_summary",
    ]
    assert draft.display_order_sections()[0].section_id == "executive_summary"

    payload = _document_draft_payload()
    payload["executive_summary"]["section_id"] = "price_dynamics"
    with pytest.raises(ValidationError, match="section IDs"):
        ReportDocumentDraft.model_validate(payload)


def _checkpoint_bound_payloads():
    manifest_payload = _manifest().model_dump(mode="json")
    research_payload = _research_plan_payload(
        query_digest=manifest_payload["query_digest"]
    )
    document_plan_payload = _document_plan_payload()
    document_plan_payload["query_digest"] = manifest_payload["query_digest"]
    document_plan_payload["evidence_manifest_id"] = manifest_payload[
        "manifest_id"
    ]
    document_plan_payload["sections"][0]["required_evidence_refs"] = [
        manifest_payload["items"][0]["evidence_ref"]
    ]
    for section in document_plan_payload["sections"]:
        section["required_evidence_refs"] = [
            manifest_payload["items"][0]["evidence_ref"]
        ]
        section["chart_refs"] = []
    document_plan_payload["charts"] = []
    draft_payload = _document_draft_payload()
    draft_payload["query_digest"] = manifest_payload["query_digest"]
    draft_payload["evidence_manifest_id"] = manifest_payload["manifest_id"]
    for section in (
        draft_payload["analytical_sections"]
        + [
            draft_payload["implications_section"],
            draft_payload["limitations_section"],
            draft_payload["executive_summary"],
        ]
    ):
        section["paragraphs"][0]["evidence_refs"] = [
            manifest_payload["items"][0]["evidence_ref"]
        ]
    return (
        manifest_payload,
        research_payload,
        document_plan_payload,
        draft_payload,
    )


def test_checkpoint_v3_supports_resumable_research_and_document_stages():
    manifest, research, document_plan, draft = _checkpoint_bound_payloads()
    packet_payload = _complete_packet_payload()
    packet_payload["items"][0] = manifest["items"][0]
    packet_payload["observations"][0]["evidence_refs"] = [
        manifest["items"][0]["evidence_ref"]
    ]
    packet_payload["observations"][0]["metric_values"][0][
        "evidence_refs"
    ] = [manifest["items"][0]["evidence_ref"]]
    packet_payload["chart_candidates"][0]["evidence_refs"] = [
        manifest["items"][0]["evidence_ref"]
    ]
    packet = ReportEvidencePacket.model_validate(packet_payload)
    research_plan = ReportResearchPlan.model_validate(research)

    research_ready = ReportGenerationCheckpoint.model_validate(
        {
            "contract_version": "report-generation-checkpoint-v3",
            "checkpoint_stage": "research_plan_ready",
            "research_plan": research_plan,
            "completed_packets": [],
            "manifest": None,
            "document_plan": None,
            "document_draft": None,
        }
    )
    assert research_ready.manifest is None

    collecting = ReportGenerationCheckpoint.model_validate(
        {
            **research_ready.model_dump(mode="json"),
            "checkpoint_stage": "evidence_collecting",
            "completed_packets": [packet],
        }
    )
    assert collecting.completed_packets[0].track_id == "prices"

    evidence_ready = ReportGenerationCheckpoint.model_validate(
        {
            **research_ready.model_dump(mode="json"),
            "checkpoint_stage": "evidence_ready",
            "manifest": manifest,
        }
    )
    assert evidence_ready.completed_packets == []

    plan_ready = ReportGenerationCheckpoint.model_validate(
        {
            **evidence_ready.model_dump(mode="json"),
            "checkpoint_stage": "document_plan_ready",
            "document_plan": ReportDocumentPlan.model_validate(
                document_plan
            ),
        }
    )
    assert plan_ready.document_plan is not None

    draft_ready = ReportGenerationCheckpoint.model_validate(
        {
            **plan_ready.model_dump(mode="json"),
            "checkpoint_stage": "draft_ready",
            "document_draft": ReportDocumentDraft.model_validate(draft),
        }
    )
    assert draft_ready.document_draft is not None


def test_checkpoint_v3_rejects_duplicate_packet_tracks_and_unbound_drafts():
    manifest, research, document_plan, draft = _checkpoint_bound_payloads()
    packet_payload = _complete_packet_payload()
    packet_payload["items"][0] = manifest["items"][0]
    evidence_ref = manifest["items"][0]["evidence_ref"]
    packet_payload["observations"][0]["evidence_refs"] = [evidence_ref]
    packet_payload["observations"][0]["metric_values"][0][
        "evidence_refs"
    ] = [evidence_ref]
    packet_payload["chart_candidates"][0]["evidence_refs"] = [evidence_ref]
    packet = ReportEvidencePacket.model_validate(packet_payload)

    with pytest.raises(ValidationError, match="packet track IDs"):
        ReportGenerationCheckpoint.model_validate(
            {
                "contract_version": "report-generation-checkpoint-v3",
                "checkpoint_stage": "evidence_collecting",
                "research_plan": ReportResearchPlan.model_validate(research),
                "completed_packets": [packet, packet],
                "manifest": None,
                "document_plan": None,
                "document_draft": None,
            }
        )

    draft["analytical_sections"][0]["section_id"] = "unknown_section"
    with pytest.raises(ValidationError, match="document draft"):
        ReportGenerationCheckpoint.model_validate(
            {
                "contract_version": "report-generation-checkpoint-v3",
                "checkpoint_stage": "draft_ready",
                "research_plan": ReportResearchPlan.model_validate(research),
                "completed_packets": [],
                "manifest": manifest,
                "document_plan": ReportDocumentPlan.model_validate(
                    document_plan
                ),
                "document_draft": ReportDocumentDraft.model_validate(draft),
            }
        )


def test_checkpoint_v3_rejects_unknown_tracks_and_manifest_evidence():
    manifest, research, document_plan, draft = _checkpoint_bound_payloads()
    document_plan["completed_track_ids"].append("unknown_track")
    document_plan["sections"][1]["track_ids"].append("unknown_track")

    with pytest.raises(ValidationError, match="unknown research tracks"):
        ReportGenerationCheckpoint.model_validate(
            {
                "contract_version": "report-generation-checkpoint-v3",
                "checkpoint_stage": "document_plan_ready",
                "research_plan": ReportResearchPlan.model_validate(research),
                "completed_packets": [],
                "manifest": manifest,
                "document_plan": ReportDocumentPlan.model_validate(
                    document_plan
                ),
                "document_draft": None,
            }
        )

    manifest, research, document_plan, draft = _checkpoint_bound_payloads()
    document_plan["sections"][1]["required_evidence_refs"] = [
        "evidence:table:" + "f" * 16
    ]
    with pytest.raises(ValidationError, match="unknown manifest evidence"):
        ReportGenerationCheckpoint.model_validate(
            {
                "contract_version": "report-generation-checkpoint-v3",
                "checkpoint_stage": "document_plan_ready",
                "research_plan": ReportResearchPlan.model_validate(research),
                "completed_packets": [],
                "manifest": manifest,
                "document_plan": ReportDocumentPlan.model_validate(
                    document_plan
                ),
                "document_draft": None,
            }
        )


def test_checkpoint_v3_binds_draft_roles_and_evidence_to_document_plan():
    manifest, research, document_plan, draft = _checkpoint_bound_payloads()
    draft["executive_summary"], draft["analytical_sections"][0] = (
        draft["analytical_sections"][0],
        draft["executive_summary"],
    )

    with pytest.raises(ValidationError, match="document-plan roles"):
        ReportGenerationCheckpoint.model_validate(
            {
                "contract_version": "report-generation-checkpoint-v3",
                "checkpoint_stage": "draft_ready",
                "research_plan": ReportResearchPlan.model_validate(research),
                "completed_packets": [],
                "manifest": manifest,
                "document_plan": ReportDocumentPlan.model_validate(
                    document_plan
                ),
                "document_draft": ReportDocumentDraft.model_validate(draft),
            }
        )

    manifest, research, document_plan, draft = _checkpoint_bound_payloads()
    draft["analytical_sections"][0]["paragraphs"][0]["evidence_refs"] = [
        "evidence:table:" + "f" * 16
    ]
    with pytest.raises(ValidationError, match="unknown manifest evidence"):
        ReportGenerationCheckpoint.model_validate(
            {
                "contract_version": "report-generation-checkpoint-v3",
                "checkpoint_stage": "draft_ready",
                "research_plan": ReportResearchPlan.model_validate(research),
                "completed_packets": [],
                "manifest": manifest,
                "document_plan": ReportDocumentPlan.model_validate(
                    document_plan
                ),
                "document_draft": ReportDocumentDraft.model_validate(draft),
            }
        )
