"""Evidence sufficiency gates and deterministic research exhibit tests."""

from __future__ import annotations

import hashlib
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.report_charts import build_report_research_exhibits
from agent.report_evidence_gate import evaluate_report_evidence
from agent.report_research_execution import (
    ReportCollectorOutput,
    consolidate_report_evidence_packets,
    execute_report_research,
)
from contracts.report_evidence import ReportEvidenceItem
from contracts.report_research import (
    ReportCollectorId,
    ReportEvidenceGate,
    ReportResearchPlan,
    ReportTrackStatus,
)
from tests.test_report_research_contract import (
    _research_plan_payload,
    _table_item,
)

_QUERY = (
    "Assess prices, energy security, and the current electricity market model."
)


def _plan() -> ReportResearchPlan:
    payload = _research_plan_payload(
        query_digest=hashlib.sha256(_QUERY.encode("utf-8")).hexdigest()
    )
    payload["tracks"][2]["expected_exhibits"] = []
    return ReportResearchPlan.model_validate(payload)


def _output(
    collector_id: ReportCollectorId,
    item: ReportEvidenceItem,
) -> ReportCollectorOutput:
    return ReportCollectorOutput(collector_id=collector_id, items=(item,))


def _price_item() -> ReportEvidenceItem:
    return ReportEvidenceItem.model_validate(_table_item())


def _generation_item() -> ReportEvidenceItem:
    return ReportEvidenceItem.model_validate(
        {
            "evidence_ref": "evidence:table:" + "3" * 16,
            "kind": "table",
            "title": "Generation mix",
            "source": "get_generation_mix",
            "provenance_refs": ["tool:get_generation_mix"],
            "columns": ["period", "type_tech", "share_tech"],
            "rows": [
                {
                    "period": "2024",
                    "type_tech": "hydro",
                    "share_tech": 0.8,
                },
                {
                    "period": "2024",
                    "type_tech": "thermal",
                    "share_tech": 0.2,
                },
                {
                    "period": "2025",
                    "type_tech": "hydro",
                    "share_tech": 0.7,
                },
                {
                    "period": "2025",
                    "type_tech": "thermal",
                    "share_tech": 0.3,
                },
            ],
            "content": "",
            "unit_by_column": {"share_tech": "share (0-1)"},
            "total_row_count": 4,
            "truncated": False,
        }
    )


def _knowledge_item() -> ReportEvidenceItem:
    return ReportEvidenceItem.model_validate(
        {
            "evidence_ref": "evidence:knowledge:" + "4" * 16,
            "kind": "knowledge",
            "title": "Market model",
            "source": "vector",
            "provenance_refs": ["vector:market:model"],
            "columns": [],
            "rows": [],
            "content": (
                "Approved knowledge describes the documented electricity "
                "market model and implementation status."
            ),
            "unit_by_column": {},
            "total_row_count": 0,
            "truncated": False,
        }
    )


def _ready_packets():
    return execute_report_research(
        _QUERY,
        _plan(),
        max_workers=3,
        collectors={
            ReportCollectorId.PRICES: (
                lambda _query, _scope: _output(
                    ReportCollectorId.PRICES,
                    _price_item(),
                )
            ),
            ReportCollectorId.GENERATION_MIX: (
                lambda _query, _scope: _output(
                    ReportCollectorId.GENERATION_MIX,
                    _generation_item(),
                )
            ),
            ReportCollectorId.VECTOR_KNOWLEDGE: (
                lambda _query, _scope: _output(
                    ReportCollectorId.VECTOR_KNOWLEDGE,
                    _knowledge_item(),
                )
            ),
        },
    )


def test_ready_gate_requires_built_numeric_and_knowledge_evidence():
    packets = _ready_packets()
    manifest = consolidate_report_evidence_packets(_QUERY, packets)
    decisions = build_report_research_exhibits(packets, manifest)

    gate = evaluate_report_evidence(
        _plan(),
        packets,
        chart_decisions=decisions,
    )

    assert gate.status.value == "ready"
    assert gate.ready_for_writing is True
    assert gate.finding_codes == []
    assert [decision.status for decision in decisions] == ["built", "built"]
    composition = decisions[1].artifact
    assert composition is not None
    assert {row["period"] for row in composition.data} == {"2025"}
    security_changes = {
        round(metric.value, 6)
        for observation in packets[1].observations
        for metric in observation.metric_values
        if metric.operation.value == "percent_change"
    }
    assert security_changes == {-12.5, 50.0}


def test_required_collector_failure_allows_only_explicit_gapped_writing():
    failed = ReportCollectorOutput(
        collector_id=ReportCollectorId.VECTOR_KNOWLEDGE,
        gaps=("COLLECTOR_VECTOR_KNOWLEDGE_FAILED",),
        failed=True,
    )
    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=3,
        collectors={
            ReportCollectorId.PRICES: (
                lambda _query, _scope: _output(
                    ReportCollectorId.PRICES,
                    _price_item(),
                )
            ),
            ReportCollectorId.GENERATION_MIX: (
                lambda _query, _scope: _output(
                    ReportCollectorId.GENERATION_MIX,
                    _generation_item(),
                )
            ),
            ReportCollectorId.VECTOR_KNOWLEDGE: (
                lambda _query, _scope: failed
            ),
        },
    )
    manifest = consolidate_report_evidence_packets(_QUERY, packets)
    decisions = build_report_research_exhibits(packets, manifest)

    gate = evaluate_report_evidence(
        _plan(),
        packets,
        chart_decisions=decisions,
    )

    assert gate.status.value == "ready_with_gaps"
    assert "REQUIRED_TRACK_GAP" in gate.finding_codes
    assert "COLLECTOR_FAILURE" in gate.finding_codes


def test_a_declared_metric_gap_degrades_coverage_but_still_reports():
    """The safety property behind ENABLE_REPORT_PARTIAL_TRACK_EVIDENCE.

    Keeping a track whose derived metric could not be computed must move the
    gate from READY to READY_WITH_GAPS and no further — a report that says what
    it could not compute still reaches the reader, which discarding the track
    did not.
    """

    packets = _ready_packets()
    gapped = [
        packet.model_copy(
            update={
                "gaps": [
                    *packet.gaps,
                    "MISSING_DERIVED_METRIC_MOM_PERCENT_CHANGE",
                ],
                "status": ReportTrackStatus.PARTIAL,
            }
        )
        if index == 0
        else packet
        for index, packet in enumerate(packets)
    ]
    manifest = consolidate_report_evidence_packets(_QUERY, gapped)
    decisions = build_report_research_exhibits(gapped, manifest)

    gate = evaluate_report_evidence(
        _plan(),
        gapped,
        chart_decisions=decisions,
    )

    assert gate.status.value == "ready_with_gaps"
    assert gate.status.value != "failed"


def test_gate_fails_when_no_required_track_has_substantive_evidence():
    failed_outputs = {
        collector_id: ReportCollectorOutput(
            collector_id=collector_id,
            gaps=(f"COLLECTOR_{collector_id.value.upper()}_FAILED",),
            failed=True,
        )
        for collector_id in {
            ReportCollectorId.PRICES,
            ReportCollectorId.GENERATION_MIX,
            ReportCollectorId.VECTOR_KNOWLEDGE,
        }
    }
    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=3,
        collectors={
            collector_id: (
                lambda _query, _scope, output=output: output
            )
            for collector_id, output in failed_outputs.items()
        },
    )

    gate = evaluate_report_evidence(
        _plan(),
        packets,
        chart_decisions=[],
    )

    assert gate.status.value == "failed"
    assert gate.ready_for_writing is False


def test_failed_gate_contract_can_preserve_nonrequired_context():
    gate = ReportEvidenceGate.model_validate(
        {
            "contract_version": "report-evidence-gate-v1",
            "query_digest": hashlib.sha256(
                _QUERY.encode("utf-8")
            ).hexdigest(),
            "status": "failed",
            "tracks": [
                {
                    "track_id": "prices",
                    "required": True,
                    "status": "failed",
                    "evidence_item_count": 0,
                    "numeric_observation_count": 0,
                    "chart_candidate_count": 0,
                    "finding_codes": ["COLLECTOR_FAILURE"],
                },
                {
                    "track_id": "optional_context",
                    "required": False,
                    "status": "complete",
                    "evidence_item_count": 1,
                    "numeric_observation_count": 0,
                    "chart_candidate_count": 0,
                    "finding_codes": [],
                },
            ],
            "finding_codes": ["NO_REQUIRED_EVIDENCE"],
        }
    )

    assert gate.ready_for_writing is False
