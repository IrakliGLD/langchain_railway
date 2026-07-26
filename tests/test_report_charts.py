"""Deterministic report chart and shadow-evaluation tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent.report_charts import build_report_charts
from agent.report_evaluation import evaluate_report_plan
from agent.report_planner import ReportPlanEvidenceError, validate_report_plan_evidence
from contracts.report import ReportPlan
from contracts.report_evidence import ReportEvidenceManifest
from tests.test_report_planner import _manifest, _plan_payload


def test_trend_chart_is_built_deterministically_from_manifest_rows():
    plan = ReportPlan.model_validate(_plan_payload())

    first = build_report_charts(plan, _manifest())
    second = build_report_charts(plan, _manifest())

    assert first == second
    assert len(first) == 1
    decision = first[0]
    assert decision.status == "built"
    assert decision.artifact is not None
    assert decision.artifact.type == "line"
    assert decision.artifact.data == [
        {"period": "2026-01", "price": 120.0},
        {"period": "2026-02", "price": 130.0},
    ]
    assert decision.artifact.metadata.x_axis == "period"
    assert decision.artifact.metadata.series == ["price"]
    assert decision.artifact.metadata.deterministic is True


def test_unrenderable_required_chart_is_omitted_with_a_typed_reason():
    payload = _plan_payload()
    payload["charts"][0]["purpose"] = "relationship"
    plan = ReportPlan.model_validate(payload)

    decisions = build_report_charts(plan, _manifest())

    assert decisions[0].status == "omitted"
    assert decisions[0].reason_code == "REPORT_CHART_EXPLICIT_AXES_REQUIRED"
    assert decisions[0].artifact is None


def test_relationship_chart_requires_verified_explicit_numeric_axes():
    manifest_payload = _manifest().model_dump(mode="json")
    table = manifest_payload["items"][0]
    table["columns"].append("hydro_generation")
    table["rows"][0]["hydro_generation"] = 410.0
    table["rows"][1]["hydro_generation"] = 390.0
    table["unit_by_column"]["hydro_generation"] = "GWh"
    manifest = ReportEvidenceManifest.model_validate(manifest_payload)

    payload = _plan_payload()
    payload["charts"][0].update(
        {
            "purpose": "relationship",
            "x_field": "price",
            "series_fields": ["hydro_generation"],
        }
    )
    plan = ReportPlan.model_validate(payload)
    validate_report_plan_evidence(plan, manifest)

    decision = build_report_charts(plan, manifest)[0]

    assert decision.status == "built"
    assert decision.artifact is not None
    assert decision.artifact.type == "scatter"
    assert decision.artifact.metadata.x_axis == "price"
    assert decision.artifact.metadata.series == ["hydro_generation"]

    invalid_payload = deepcopy(payload)
    invalid_payload["charts"][0]["series_fields"] = ["invented_metric"]
    with pytest.raises(ReportPlanEvidenceError, match="unknown chart fields"):
        validate_report_plan_evidence(
            ReportPlan.model_validate(invalid_payload),
            manifest,
        )


def test_shadow_evaluation_is_content_free_and_blocks_required_chart_omission():
    valid = evaluate_report_plan(
        ReportPlan.model_validate(_plan_payload()),
        _manifest(),
    )

    assert valid.contract_version == "report-evaluation-v1"
    assert valid.ready_for_generation is True
    assert valid.evidence_reference_coverage == 1.0
    assert valid.required_chart_build_rate == 1.0
    assert valid.findings == []
    assert "120.0" not in valid.model_dump_json()

    payload = deepcopy(_plan_payload())
    payload["charts"][0]["purpose"] = "relationship"
    invalid = evaluate_report_plan(ReportPlan.model_validate(payload), _manifest())

    assert invalid.ready_for_generation is False
    assert invalid.required_chart_build_rate == 0.0
    assert "REQUIRED_CHART_OMITTED" in invalid.findings


def test_shadow_evaluation_fails_closed_without_throwing_on_unknown_chart_evidence():
    payload = deepcopy(_plan_payload())
    payload["charts"][0]["evidence_refs"] = [
        "evidence:table:" + "9" * 16
    ]
    invalid = evaluate_report_plan(ReportPlan.model_validate(payload), _manifest())

    assert invalid.ready_for_generation is False
    assert invalid.evidence_reference_coverage < 1.0
    assert "PLAN_EVIDENCE_INVALID" in invalid.findings
    assert "REQUIRED_CHART_OMITTED" in invalid.findings
