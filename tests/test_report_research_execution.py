"""Deterministic, bounded, parallel report research execution tests."""

from __future__ import annotations

import hashlib
import os
import threading

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.report_research_execution import (
    DEFAULT_REPORT_COLLECTORS,
    ReportCollectorOutput,
    consolidate_report_evidence_packets,
    execute_report_research,
)
from contracts.report_evidence import ReportEvidenceItem, ReportEvidenceKind
from contracts.report_research import (
    ReportCollectorId,
    ReportEvidencePacket,
    ReportResearchPlan,
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


def _table_output(collector_id: ReportCollectorId) -> ReportCollectorOutput:
    return ReportCollectorOutput(
        collector_id=collector_id,
        items=(ReportEvidenceItem.model_validate(_table_item()),),
    )


def _knowledge_output() -> ReportCollectorOutput:
    return ReportCollectorOutput(
        collector_id=ReportCollectorId.VECTOR_KNOWLEDGE,
        items=(
            ReportEvidenceItem.model_validate(
                {
                    "evidence_ref": "evidence:knowledge:" + "2" * 16,
                    "kind": "knowledge",
                    "title": "Electricity market model",
                    "source": "vector",
                    "provenance_refs": ["vector:market:model"],
                    "columns": [],
                    "rows": [],
                    "content": (
                        "Approved knowledge states the documented market "
                        "model and implementation stage."
                    ),
                    "unit_by_column": {},
                    "total_row_count": 0,
                    "truncated": False,
                }
            ),
        ),
    )


def test_research_executes_unique_collectors_in_parallel_and_keeps_track_order():
    barrier = threading.Barrier(2)
    calls = []

    def prices(_query, _scope):
        calls.append("prices")
        barrier.wait(timeout=2)
        return _table_output(ReportCollectorId.PRICES)

    def generation(_query, _scope):
        calls.append("generation_mix")
        barrier.wait(timeout=2)
        return _table_output(ReportCollectorId.GENERATION_MIX)

    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=2,
        collectors={
            ReportCollectorId.PRICES: prices,
            ReportCollectorId.GENERATION_MIX: generation,
            ReportCollectorId.VECTOR_KNOWLEDGE: (
                lambda _query, _scope: _knowledge_output()
            ),
        },
    )

    assert set(calls) == {"prices", "generation_mix"}
    assert [packet.track_id for packet in packets] == [
        "prices",
        "security",
        "market_model",
    ]
    assert all(packet.status.value == "complete" for packet in packets)
    assert packets[0].numeric_observation_count >= 4


def test_collector_failure_becomes_a_typed_partial_packet_not_global_failure():
    def failed_knowledge(_query, _scope):
        raise RuntimeError("provider detail must not enter evidence")

    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=3,
        collectors={
            ReportCollectorId.PRICES: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.PRICES
                )
            ),
            ReportCollectorId.GENERATION_MIX: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.GENERATION_MIX
                )
            ),
            ReportCollectorId.VECTOR_KNOWLEDGE: failed_knowledge,
        },
    )

    security = packets[1]
    market_model = packets[2]
    assert security.status.value == "partial"
    assert market_model.status.value == "failed"
    assert "provider detail" not in " ".join(
        security.gaps + market_model.gaps
    )


def test_packet_metrics_and_manifest_are_deterministic_and_deduplicated():
    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=1,
        collectors={
            ReportCollectorId.PRICES: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.PRICES
                )
            ),
            ReportCollectorId.GENERATION_MIX: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.GENERATION_MIX
                )
            ),
            ReportCollectorId.VECTOR_KNOWLEDGE: (
                lambda _query, _scope: _knowledge_output()
            ),
        },
    )
    price_metrics = {
        metric.operation.value: metric.value
        for observation in packets[0].observations
        for metric in observation.metric_values
    }
    assert price_metrics == {
        "mean": 110.0,
        "minimum": 100.0,
        "maximum": 120.0,
        "percent_change": 20.0,
    }

    manifest = consolidate_report_evidence_packets(_QUERY, packets)
    assert manifest.query_digest == hashlib.sha256(
        _QUERY.encode("utf-8")
    ).hexdigest()
    assert len(
        {
            item.evidence_ref
            for packet in packets
            for item in packet.items
        }
    ) == 2
    assert len(manifest.items) == 3
    assert manifest.items[-1].kind.value == "limitation"


def test_table_metrics_are_chronological_when_tools_return_latest_first():
    table = _table_item()
    table["rows"] = list(reversed(table["rows"]))
    output = ReportCollectorOutput(
        collector_id=ReportCollectorId.PRICES,
        items=(ReportEvidenceItem.model_validate(table),),
    )
    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=1,
        collectors={
            ReportCollectorId.PRICES: lambda _query, _scope: output,
            ReportCollectorId.GENERATION_MIX: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.GENERATION_MIX
                )
            ),
            ReportCollectorId.VECTOR_KNOWLEDGE: (
                lambda _query, _scope: _knowledge_output()
            ),
        },
    )
    percent_change = next(
        metric.value
        for observation in packets[0].observations
        for metric in observation.metric_values
        if metric.operation.value == "percent_change"
    )

    assert percent_change == 20.0


def test_consolidation_reserves_space_for_the_limitation_item():
    items = [
        ReportEvidenceItem.model_validate(
            {
                **_knowledge_output().items[0].model_dump(mode="json"),
                "evidence_ref": f"evidence:knowledge:{index:016x}",
                "content": f"Approved bounded knowledge passage number {index}.",
            }
        )
        for index in range(32)
    ]
    packets = [
        ReportEvidencePacket.model_validate(
            {
                "contract_version": "report-evidence-packet-v1",
                "track_id": "market_model",
                "status": "complete",
                "items": items[:12],
                "observations": [
                    {
                        "observation_id": "documented_context",
                        "statement": (
                            "Approved knowledge evidence was retrieved for "
                            "the requested market-model topic."
                        ),
                        "evidence_refs": [items[0].evidence_ref],
                        "metric_values": [],
                    }
                ],
                "gaps": [],
                "chart_candidates": [],
            }
        ),
        ReportEvidencePacket.model_validate(
            {
                "contract_version": "report-evidence-packet-v1",
                "track_id": "market_rules",
                "status": "complete",
                "items": items[12:24],
                "observations": [
                    {
                        "observation_id": "documented_rules",
                        "statement": (
                            "Approved knowledge evidence was retrieved for "
                            "the requested market-rules topic."
                        ),
                        "evidence_refs": [items[12].evidence_ref],
                        "metric_values": [],
                    }
                ],
                "gaps": [],
                "chart_candidates": [],
            }
        ),
        ReportEvidencePacket.model_validate(
            {
                "contract_version": "report-evidence-packet-v1",
                "track_id": "market_status",
                "status": "complete",
                "items": items[24:],
                "observations": [
                    {
                        "observation_id": "documented_status",
                        "statement": (
                            "Approved knowledge evidence was retrieved for "
                            "the requested market-status topic."
                        ),
                        "evidence_refs": [items[24].evidence_ref],
                        "metric_values": [],
                    }
                ],
                "gaps": [],
                "chart_candidates": [],
            }
        ),
    ]

    manifest = consolidate_report_evidence_packets(_QUERY, packets)

    assert len(manifest.items) == 32
    assert manifest.items[-1].kind is ReportEvidenceKind.LIMITATION


def test_default_price_collector_uses_query_metric_and_currency(
    monkeypatch,
):
    captured = {}

    def fake_prices(**kwargs):
        captured.update(kwargs)
        table = _table_item()
        return None, table["columns"], [
            tuple(row[column] for column in table["columns"])
            for row in table["rows"]
        ]

    monkeypatch.setattr(
        "agent.report_research_execution.get_prices",
        fake_prices,
    )
    output = DEFAULT_REPORT_COLLECTORS[ReportCollectorId.PRICES](
        "Compare deregulated electricity prices in USD.",
        _plan().scope,
    )

    assert output.items
    assert captured["metric"] == "deregulated"
    assert captured["currency"] == "usd"


def test_research_execution_rejects_unbounded_worker_counts():
    for invalid in (0, 9):
        try:
            execute_report_research(
                _QUERY,
                _plan(),
                max_workers=invalid,
                collectors={},
            )
        except ValueError as exc:
            assert "max_workers" in str(exc)
        else:
            raise AssertionError("invalid worker count was accepted")


def test_consolidation_accepts_closed_packet_contracts():
    packet = ReportEvidencePacket.model_validate(
        {
            "contract_version": "report-evidence-packet-v1",
            "track_id": "market_model",
            "status": "complete",
            "items": [_knowledge_output().items[0]],
            "observations": [
                {
                    "observation_id": "documented_context",
                    "statement": (
                        "Approved knowledge evidence was retrieved for the "
                        "requested market-model topic."
                    ),
                    "evidence_refs": [
                        _knowledge_output().items[0].evidence_ref
                    ],
                    "metric_values": [],
                }
            ],
            "gaps": [],
            "chart_candidates": [],
        }
    )

    manifest = consolidate_report_evidence_packets(_QUERY, [packet])
    assert manifest.items[0].evidence_ref == packet.items[0].evidence_ref
