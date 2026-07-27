"""Evidence-bound report planning tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent.report_planner import (
    ReportPlanEvidenceError,
    plan_report,
    validate_report_plan_evidence,
)
from contracts.report import REPORT_PLAN_CONTRACT_VERSION, ReportPlan
from contracts.report_evidence import ReportEvidenceManifest

TABLE_REF = "evidence:table:" + "1" * 16
STATS_REF = "evidence:statistics:" + "2" * 16
LIMIT_REF = "evidence:limitation:" + "3" * 16


def _manifest() -> ReportEvidenceManifest:
    return ReportEvidenceManifest.model_validate(
        {
            "contract_version": "report-evidence-manifest-v1",
            "manifest_id": "manifest:" + "4" * 32,
            "query_digest": "5" * 64,
            "items": [
                {
                    "evidence_ref": TABLE_REF,
                    "kind": "table",
                    "title": "Prices",
                    "source": "tool",
                    "provenance_refs": ["query:tool:prices"],
                    "columns": ["period", "price"],
                    "rows": [
                        {"period": "2026-01", "price": 120.0},
                        {"period": "2026-02", "price": 130.0},
                    ],
                    "content": "",
                    "unit_by_column": {"price": "GEL/MWh"},
                    "total_row_count": 2,
                    "truncated": False,
                },
                {
                    "evidence_ref": STATS_REF,
                    "kind": "statistics",
                    "title": "Statistics",
                    "source": "derived",
                    "provenance_refs": [TABLE_REF],
                    "columns": [],
                    "rows": [],
                    "content": "Average price was 125 GEL/MWh.",
                    "unit_by_column": {},
                    "total_row_count": 0,
                    "truncated": False,
                },
                {
                    "evidence_ref": LIMIT_REF,
                    "kind": "limitation",
                    "title": "Evidence boundary",
                    "source": "system",
                    "provenance_refs": [],
                    "columns": [],
                    "rows": [],
                    "content": "Only the supplied periods and sources may be used.",
                    "unit_by_column": {},
                    "total_row_count": 0,
                    "truncated": False,
                },
            ],
        }
    )


def _plan_payload() -> dict:
    return {
        "contract_version": REPORT_PLAN_CONTRACT_VERSION,
        "title": "Electricity price trend report",
        "objective": "Explain the observed price trend from supplied evidence.",
        "language_code": "en",
        "target_words": 900,
        "evidence_manifest_id": _manifest().manifest_id,
        "sections": [
            {
                "section_id": "executive_summary",
                "kind": "executive_summary",
                "title": "Executive summary",
                "objective": "Summarize the principal evidence-backed finding.",
                "target_words": 120,
                "required_evidence_refs": [STATS_REF],
                "chart_refs": [],
            },
            {
                "section_id": "scope_and_evidence",
                "kind": "scope_and_evidence",
                "title": "Scope and evidence",
                "objective": "Describe the period, source, and measurement.",
                "target_words": 120,
                "required_evidence_refs": [TABLE_REF],
                "chart_refs": [],
            },
            {
                "section_id": "key_findings",
                "kind": "key_findings",
                "title": "Key findings",
                "objective": "Explain the observed values and direction.",
                "target_words": 420,
                "required_evidence_refs": [TABLE_REF, STATS_REF],
                "chart_refs": ["price_trend"],
            },
            {
                "section_id": "limitations",
                "kind": "limitations",
                "title": "Limitations",
                "objective": "State the evidence boundary.",
                "target_words": 100,
                "required_evidence_refs": [LIMIT_REF],
                "chart_refs": [],
            },
            {
                "section_id": "conclusion",
                "kind": "conclusion",
                "title": "Conclusion",
                "objective": "Answer the question within the evidence boundary.",
                "target_words": 140,
                "required_evidence_refs": [STATS_REF],
                "chart_refs": [],
            },
        ],
        "charts": [
            {
                "chart_id": "price_trend",
                "section_id": "key_findings",
                "purpose": "trend",
                "title": "Observed electricity price",
                "evidence_refs": [TABLE_REF],
                "required": True,
            }
        ],
    }


def test_plan_validation_binds_every_section_and_chart_to_the_manifest():
    plan = ReportPlan.model_validate(_plan_payload())

    validate_report_plan_evidence(plan, _manifest())


def test_plan_validation_rejects_manifest_mismatch_unknown_refs_and_non_table_charts():
    payload = _plan_payload()
    payload["evidence_manifest_id"] = "manifest:" + "9" * 32
    with pytest.raises(ReportPlanEvidenceError, match="manifest identity"):
        validate_report_plan_evidence(ReportPlan.model_validate(payload), _manifest())

    payload = _plan_payload()
    payload["sections"][0]["required_evidence_refs"] = [
        "evidence:statistics:" + "9" * 16
    ]
    with pytest.raises(ReportPlanEvidenceError, match="unknown evidence"):
        validate_report_plan_evidence(ReportPlan.model_validate(payload), _manifest())

    payload = _plan_payload()
    payload["charts"][0]["evidence_refs"] = [STATS_REF]
    with pytest.raises(ReportPlanEvidenceError, match="tabular evidence"):
        validate_report_plan_evidence(ReportPlan.model_validate(payload), _manifest())


def test_limitations_section_must_cite_a_typed_limitation():
    payload = _plan_payload()
    payload["sections"][3]["required_evidence_refs"] = [STATS_REF]

    with pytest.raises(ReportPlanEvidenceError, match="typed limitation"):
        validate_report_plan_evidence(ReportPlan.model_validate(payload), _manifest())


def test_plan_requires_substantive_evidence_outside_the_limitations_section():
    payload = _plan_payload()
    payload["charts"] = []
    for section in payload["sections"]:
        section["required_evidence_refs"] = [LIMIT_REF]
        section["chart_refs"] = []

    with pytest.raises(ReportPlanEvidenceError, match="substantive evidence"):
        validate_report_plan_evidence(ReportPlan.model_validate(payload), _manifest())


def test_planner_validates_model_output_before_returning_it():
    calls = []

    def model(query, manifest, planning_context):
        calls.append(
            (
                query,
                manifest.manifest_id,
                planning_context.intent.value,
            )
        )
        return deepcopy(_plan_payload())

    plan = plan_report(
        "Explain the price trend.",
        _manifest(),
        invoke_model=model,
    )

    assert plan.title == "Electricity price trend report"
    assert [chart.chart_id for chart in plan.charts] == ["price_trend"]
    assert calls == [
        (
            "Explain the price trend.",
            _manifest().manifest_id,
            "general",
        )
    ]


def test_planner_repairs_schema_valid_evidence_bindings_before_returning():
    def invalid_model(*_):
        payload = deepcopy(_plan_payload())
        payload["evidence_manifest_id"] = "manifest:" + "0" * 32
        for section in payload["sections"]:
            section["required_evidence_refs"] = [
                "evidence:statistics:" + "9" * 16
            ]
        payload["charts"][0]["evidence_refs"] = [STATS_REF]
        payload["charts"][0]["x_field"] = "invented_period"
        payload["charts"][0]["series_fields"] = ["invented_value"]
        return payload

    plan = plan_report(
        "Explain the price trend.",
        _manifest(),
        invoke_model=invalid_model,
    )

    validate_report_plan_evidence(plan, _manifest())
    assert plan.evidence_manifest_id == _manifest().manifest_id
    assert all(
        set(section.required_evidence_refs)
        <= {TABLE_REF, STATS_REF, LIMIT_REF}
        for section in plan.sections
    )
    assert LIMIT_REF in next(
        section
        for section in plan.sections
        if section.kind.value == "limitations"
    ).required_evidence_refs
    assert all(
        len(section.required_evidence_refs) == 1
        for section in plan.sections
    )
    assert plan.charts == []
    assert all(section.chart_refs == [] for section in plan.sections)


def test_planner_removes_required_chart_that_cannot_be_built():
    def invalid_chart_model(*_):
        payload = deepcopy(_plan_payload())
        payload["charts"][0]["purpose"] = "relationship"
        return payload

    plan = plan_report(
        "Explain the price trend.",
        _manifest(),
        invoke_model=invalid_chart_model,
    )

    validate_report_plan_evidence(plan, _manifest())
    assert plan.charts == []
    assert all(section.chart_refs == [] for section in plan.sections)
