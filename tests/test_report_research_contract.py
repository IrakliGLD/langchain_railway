"""Contracts for multi-track report research and evidence coverage."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from contracts.report_research import (
    ReportEvidenceGate,
    ReportEvidencePacket,
    ReportResearchPlan,
    ReportResearchPlanDraft,
)


def _research_plan_payload(*, query_digest: str = "a" * 64) -> dict:
    return {
        "contract_version": "report-research-plan-v1",
        "query_digest": query_digest,
        "language_code": "en",
        "objective": "Assess prices, energy security, and the market model.",
        "scope": {
            "geography": "Georgia",
            "period_start": "2024-01-01",
            "period_end": "2025-12-31",
            "timezone": "Asia/Tbilisi",
            "grain": "month",
        },
        "request_topics": [
            {
                "topic_id": "price_dynamics",
                "label": "Electricity price dynamics",
                "required": True,
                "evidence_mode": "table",
            },
            {
                "topic_id": "energy_security",
                "label": "Energy security",
                "required": True,
                "evidence_mode": "mixed",
            },
            {
                "topic_id": "market_model",
                "label": "Market model and legislation",
                "required": True,
                "evidence_mode": "knowledge",
            },
        ],
        "tracks": [
            {
                "track_id": "prices",
                "title": "Price dynamics",
                "topic_ids": ["price_dynamics"],
                "required": True,
                "evidence_mode": "table",
                "collector_ids": ["prices"],
                "research_questions": [
                    "How did electricity prices change over the period?"
                ],
                "requested_metrics": [
                    "average_price",
                    "minimum_price",
                    "maximum_price",
                    "percent_change",
                ],
                "expected_exhibits": ["trend"],
            },
            {
                "track_id": "security",
                "title": "Energy security",
                "topic_ids": ["energy_security"],
                "required": True,
                "evidence_mode": "mixed",
                "collector_ids": [
                    "generation_mix",
                    "vector_knowledge",
                ],
                "research_questions": [
                    "What measurable supply-security risks are visible?"
                ],
                "requested_metrics": [
                    "import_dependency_ratio",
                    "generation_mix",
                ],
                "expected_exhibits": ["composition"],
            },
            {
                "track_id": "market_model",
                "title": "Market model",
                "topic_ids": ["market_model"],
                "required": True,
                "evidence_mode": "knowledge",
                "collector_ids": ["vector_knowledge"],
                "research_questions": [
                    "What is the deregulation stage and target model?"
                ],
                "requested_metrics": [],
                "expected_exhibits": ["table"],
            },
        ],
    }


def _table_item() -> dict:
    return {
        "evidence_ref": "evidence:table:" + "1" * 16,
        "kind": "table",
        "title": "Monthly balancing prices",
        "source": "get_prices",
        "provenance_refs": ["query:prices"],
        "columns": ["period", "price_gel"],
        "rows": [
            {"period": "2025-01", "price_gel": 100.0},
            {"period": "2025-02", "price_gel": 120.0},
        ],
        "content": "",
        "unit_by_column": {"price_gel": "GEL/MWh"},
        "total_row_count": 2,
        "truncated": False,
    }


def _complete_packet_payload() -> dict:
    evidence_ref = _table_item()["evidence_ref"]
    return {
        "contract_version": "report-evidence-packet-v1",
        "track_id": "prices",
        "status": "complete",
        "available_period_start": "2025-01-01",
        "available_period_end": "2025-02-28",
        "items": [_table_item()],
        "observations": [
            {
                "observation_id": "price_change",
                "statement": "The observed monthly price increased over the available period.",
                "evidence_refs": [evidence_ref],
                "metric_values": [
                    {
                        "metric_id": "percent_change",
                        "label": "Observed price change",
                        "value": 20.0,
                        "display_value": "20.0%",
                        "unit": "%",
                        "operation": "percent_change",
                        "evidence_refs": [evidence_ref],
                        "period_start": "2025-01-01",
                        "period_end": "2025-02-28",
                    }
                ],
            }
        ],
        "gaps": [],
        "chart_candidates": [
            {
                "chart_id": "price_trend",
                "purpose": "trend",
                "title": "Monthly balancing price",
                "evidence_refs": [evidence_ref],
                "x_field": "period",
                "series_fields": ["price_gel"],
                "required": True,
            }
        ],
    }


def test_compound_research_plan_is_closed_and_covers_every_required_topic():
    plan = ReportResearchPlan.model_validate(_research_plan_payload())

    assert len(plan.tracks) == 3
    assert plan.scope.timezone == "Asia/Tbilisi"
    schema = ReportResearchPlan.model_json_schema()
    assert schema["additionalProperties"] is False


def test_research_plan_draft_schema_is_strict_and_model_owned():
    schema = ReportResearchPlanDraft.model_json_schema()
    server_owned = {"contract_version", "query_digest", "language_code"}

    assert server_owned.isdisjoint(schema["properties"])

    def assert_strict_objects(node):
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(properties)
            for child in node.values():
                assert_strict_objects(child)
        elif isinstance(node, list):
            for child in node:
                assert_strict_objects(child)

    assert_strict_objects(schema)


def test_research_plan_rejects_uncovered_or_unknown_topics():
    payload = _research_plan_payload()
    payload["tracks"][-1]["topic_ids"] = ["unknown_topic"]

    with pytest.raises(ValidationError, match="unknown request topic"):
        ReportResearchPlan.model_validate(payload)

    payload = _research_plan_payload()
    payload["tracks"] = payload["tracks"][:-1]
    with pytest.raises(ValidationError, match="request topic"):
        ReportResearchPlan.model_validate(payload)


def test_research_scope_rejects_reversed_or_half_open_periods():
    payload = _research_plan_payload()
    payload["scope"]["period_start"] = "2026-01-01"

    with pytest.raises(ValidationError, match="period_start"):
        ReportResearchPlan.model_validate(payload)

    payload = _research_plan_payload()
    payload["scope"]["period_end"] = None
    with pytest.raises(ValidationError, match="both be present"):
        ReportResearchPlan.model_validate(payload)


def test_table_track_requires_a_tabular_collector():
    payload = _research_plan_payload()
    payload["tracks"][0]["collector_ids"] = ["vector_knowledge"]

    with pytest.raises(ValidationError, match="tabular collector"):
        ReportResearchPlan.model_validate(payload)


def test_evidence_packet_enforces_status_gaps_and_chart_grounding():
    packet = ReportEvidencePacket.model_validate(_complete_packet_payload())
    assert packet.numeric_observation_count == 1

    payload = _complete_packet_payload()
    payload["gaps"] = ["Missing one requested month."]
    with pytest.raises(ValidationError, match="complete packet"):
        ReportEvidencePacket.model_validate(payload)

    payload = _complete_packet_payload()
    payload["status"] = "partial"
    payload["gaps"] = ["Missing one requested month."]
    assert ReportEvidencePacket.model_validate(payload).status.value == "partial"

    payload = _complete_packet_payload()
    payload["chart_candidates"][0]["evidence_refs"] = [
        "evidence:table:" + "9" * 16
    ]
    with pytest.raises(ValidationError, match="unknown evidence"):
        ReportEvidencePacket.model_validate(payload)

    payload = _complete_packet_payload()
    payload["status"] = "unavailable"
    payload["items"] = []
    payload["observations"] = []
    payload["chart_candidates"] = []
    payload["gaps"] = ["No price rows were available."]
    unavailable = ReportEvidencePacket.model_validate(payload)
    assert unavailable.numeric_observation_count == 0


def _coverage(track_id: str, *, required: bool, status: str) -> dict:
    has_evidence = status in {"complete", "partial"}
    return {
        "track_id": track_id,
        "required": required,
        "status": status,
        "evidence_item_count": 1 if has_evidence else 0,
        "numeric_observation_count": 1 if has_evidence else 0,
        "chart_candidate_count": 1 if has_evidence else 0,
        "finding_codes": (
            [] if status == "complete" else ["TRACK_EVIDENCE_GAP"]
        ),
    }


def test_evidence_gate_distinguishes_ready_gapped_and_failed_outcomes():
    ready = ReportEvidenceGate.model_validate(
        {
            "contract_version": "report-evidence-gate-v1",
            "query_digest": "a" * 64,
            "status": "ready",
            "tracks": [
                _coverage("prices", required=True, status="complete"),
                _coverage("market_model", required=True, status="complete"),
            ],
            "finding_codes": [],
        }
    )
    assert ready.ready_for_writing is True

    gapped_payload = ready.model_dump(mode="json")
    gapped_payload["status"] = "ready_with_gaps"
    gapped_payload["tracks"][1] = _coverage(
        "market_model",
        required=True,
        status="unavailable",
    )
    gapped_payload["finding_codes"] = ["REQUIRED_TRACK_UNAVAILABLE"]
    gapped = ReportEvidenceGate.model_validate(gapped_payload)
    assert gapped.ready_for_writing is True

    invalid = deepcopy(gapped_payload)
    invalid["status"] = "ready"
    with pytest.raises(ValidationError, match="complete required tracks"):
        ReportEvidenceGate.model_validate(invalid)

    failed = deepcopy(gapped_payload)
    failed["status"] = "failed"
    failed["tracks"][0] = _coverage(
        "prices",
        required=True,
        status="failed",
    )
    assert ReportEvidenceGate.model_validate(failed).ready_for_writing is False
