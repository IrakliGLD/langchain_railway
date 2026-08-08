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
                "analysis_query_type": "data_explanation",
                "analysis_preferred_path": "tool",
                "analysis_answer_kind": "timeseries",
                "analysis_derived_metrics": ["mom_percent_change"],
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
                "analysis_query_type": "comparison",
                "analysis_preferred_path": "sql",
                "analysis_answer_kind": "comparison",
                "analysis_derived_metrics": [],
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
                "analysis_query_type": "conceptual_definition",
                "analysis_preferred_path": "knowledge",
                "analysis_answer_kind": "knowledge",
                "analysis_derived_metrics": [],
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


def _plan_with_forecast_engine(query: str):
    """A plan whose model added a forecast collector, keyed to `query`."""

    import hashlib

    from contracts.report_research import ReportResearchPlan

    payload = _research_plan_payload(
        query_digest=hashlib.sha256(query.encode("utf-8")).hexdigest()
    )
    payload["tracks"][0]["collector_ids"] = ["prices", "forecast_engine"]
    return ReportResearchPlan.model_validate(payload)


def test_unrequested_forecast_collector_is_pruned_not_fatal():
    """A "future" query has no forecast keyword, but the model still plans one.

    `_FORECAST_SIGNALS` matches "forecast"/"projection"/"predict" and not
    "future" or "outlook", so the model's reading and the keyword list
    disagree. That disagreement used to fail the plan outright, and since the
    planner prompt is identical on every retry, all three attempts died the
    same way -- observed in production on 2026-07-31.
    """

    from agent.report_research_planner import (
        _prune_unrequested_engines,
        _recognized_requirements,
        validate_report_research_plan,
    )
    from contracts.report_research import ReportCollectorId

    query = "Report on the current and future market model and prices."
    plan = _plan_with_forecast_engine(query)

    # Precondition: unpruned, this plan is rejected.
    rejected = validate_report_research_plan(query, plan, max_tracks=4)
    assert not rejected.valid
    assert "UNREQUESTED_FORECAST_COLLECTOR" in rejected.finding_codes

    pruned = _prune_unrequested_engines(plan, _recognized_requirements(query))
    collectors = {
        collector for track in pruned.tracks for collector in track.collector_ids
    }
    assert ReportCollectorId.FORECAST_ENGINE not in collectors
    assert validate_report_research_plan(query, pruned, max_tracks=4).valid


def test_requested_forecast_collector_survives_pruning():
    """When the query does ask for a forecast, the collector must stay."""

    from agent.report_research_planner import (
        _prune_unrequested_engines,
        _recognized_requirements,
    )
    from contracts.report_research import ReportCollectorId

    query = "Report on the price forecast for the market model."
    plan = _plan_with_forecast_engine(query)

    pruned = _prune_unrequested_engines(plan, _recognized_requirements(query))
    collectors = {
        collector for track in pruned.tracks for collector in track.collector_ids
    }
    assert ReportCollectorId.FORECAST_ENGINE in collectors


def test_a_track_states_how_its_own_evidence_should_be_analysed():
    """The planner wrote the questions, so it owns the analysis decisions.

    requested_metrics names what to measure; analysis_derived_metrics names
    how to compare it over time. "percent_change" cannot distinguish
    month-on-month from year-on-year, and only the research question says
    which -- which is why re-deriving this per track from prose was lossy.
    """

    plan = ReportResearchPlan.model_validate(_research_plan_payload())
    prices = next(
        track for track in plan.tracks if track.track_id == "prices"
    )
    market_model = next(
        track for track in plan.tracks if track.track_id == "market_model"
    )

    assert prices.analysis_query_type.value == "data_explanation"
    assert prices.analysis_preferred_path.value == "tool"
    assert prices.analysis_answer_kind.value == "timeseries"
    assert [
        metric.value for metric in prices.analysis_derived_metrics
    ] == ["mom_percent_change"]
    # The two vocabularies stay separate: a track asks for percent_change and
    # says which comparison that means.
    assert "percent_change" in prices.requested_metrics
    assert market_model.analysis_preferred_path.value == "knowledge"
    assert market_model.analysis_derived_metrics == []


def test_a_persisted_plan_without_analysis_fields_still_validates():
    """Checkpoints written before the fields exist must still resume.

    The base contract carries defaults so an in-flight job is not failed by a
    deploy; only the model-facing draft demands the decisions.
    """

    payload = _research_plan_payload()
    for track in payload["tracks"]:
        for field in (
            "analysis_query_type",
            "analysis_preferred_path",
            "analysis_answer_kind",
            "analysis_derived_metrics",
        ):
            track.pop(field, None)

    plan = ReportResearchPlan.model_validate(payload)

    assert plan.tracks[0].analysis_query_type.value == "data_retrieval"
    assert plan.tracks[0].analysis_derived_metrics == []
    with pytest.raises(ValidationError):
        ReportResearchPlanDraft.model_validate(
            {
                "objective": payload["objective"],
                "scope": payload["scope"],
                "request_topics": payload["request_topics"],
                "tracks": payload["tracks"],
            }
        )
