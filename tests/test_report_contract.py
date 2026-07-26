"""Standard report-plan contract and runtime-skill tests."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from contracts.report import (
    REPORT_PLAN_CONTRACT_VERSION,
    STANDARD_REPORT_SECTION_SEQUENCE,
    ReportPlan,
    ReportSectionKind,
)
from skills.loader import get_report_guidance, validate_skills


def _valid_plan_payload() -> dict:
    return {
        "contract_version": REPORT_PLAN_CONTRACT_VERSION,
        "title": "Georgia electricity market outlook",
        "objective": "Explain the observed market trend and its implications.",
        "language_code": "en",
        "target_words": 920,
        "evidence_manifest_id": "evidence-manifest-1",
        "sections": [
            {
                "section_id": "executive_summary",
                "kind": "executive_summary",
                "title": "Executive summary",
                "objective": "State the report's principal evidence-backed findings.",
                "target_words": 120,
                "required_evidence_refs": ["statistics:market_summary"],
                "chart_refs": [],
            },
            {
                "section_id": "scope_and_evidence",
                "kind": "scope_and_evidence",
                "title": "Scope and evidence",
                "objective": "Define the period, metrics, sources, and analytical limits.",
                "target_words": 100,
                "required_evidence_refs": ["dataset:market_prices"],
                "chart_refs": [],
            },
            {
                "section_id": "key_findings",
                "kind": "key_findings",
                "title": "Key findings and analysis",
                "objective": "Explain the main observed trends using verified values.",
                "target_words": 420,
                "required_evidence_refs": [
                    "dataset:market_prices",
                    "statistics:market_summary",
                ],
                "chart_refs": ["chart_price_trend"],
            },
            {
                "section_id": "implications",
                "kind": "implications",
                "title": "Implications",
                "objective": "Describe evidence-consistent implications without overstating causality.",
                "target_words": 140,
                "required_evidence_refs": ["statistics:market_summary"],
                "chart_refs": [],
            },
            {
                "section_id": "limitations",
                "kind": "limitations",
                "title": "Limitations and data gaps",
                "objective": "State material evidence and methodology limitations.",
                "target_words": 80,
                "required_evidence_refs": ["evidence_gap:external_drivers"],
                "chart_refs": [],
            },
            {
                "section_id": "conclusion",
                "kind": "conclusion",
                "title": "Conclusion",
                "objective": "Close with the supported answer to the user's question.",
                "target_words": 60,
                "required_evidence_refs": ["statistics:market_summary"],
                "chart_refs": [],
            },
        ],
        "charts": [
            {
                "chart_id": "chart_price_trend",
                "section_id": "key_findings",
                "purpose": "trend",
                "title": "Monthly balancing price trend",
                "evidence_refs": ["dataset:market_prices"],
                "required": True,
            }
        ],
    }


def test_standard_report_plan_accepts_the_canonical_structure():
    plan = ReportPlan.model_validate(_valid_plan_payload())

    assert plan.contract_version == REPORT_PLAN_CONTRACT_VERSION
    assert [section.kind for section in plan.sections] == [
        ReportSectionKind.EXECUTIVE_SUMMARY,
        ReportSectionKind.SCOPE_AND_EVIDENCE,
        ReportSectionKind.KEY_FINDINGS,
        ReportSectionKind.IMPLICATIONS,
        ReportSectionKind.LIMITATIONS,
        ReportSectionKind.CONCLUSION,
    ]
    assert sum(section.target_words for section in plan.sections) == plan.target_words
    assert tuple(STANDARD_REPORT_SECTION_SEQUENCE) == (
        ReportSectionKind.EXECUTIVE_SUMMARY,
        ReportSectionKind.SCOPE_AND_EVIDENCE,
        ReportSectionKind.KEY_FINDINGS,
        ReportSectionKind.LIMITATIONS,
        ReportSectionKind.CONCLUSION,
    )


def test_report_plan_schema_is_closed_at_every_level():
    schema = ReportPlan.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["ReportSectionSpec"]["additionalProperties"] is False
    assert schema["$defs"]["ReportChartRequest"]["additionalProperties"] is False
    assert "optional" not in schema["$defs"]["ReportSectionSpec"]["properties"]

    payload = _valid_plan_payload()
    payload["unexpected"] = "not allowed"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ReportPlan.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["sections"].pop(4),
            "required standard sections",
        ),
        (
            lambda payload: payload["sections"].__setitem__(
                slice(1, 3),
                [payload["sections"][2], payload["sections"][1]],
            ),
            "standard section order",
        ),
        (
            lambda payload: payload["sections"][2].__setitem__(
                "section_id",
                "scope_and_evidence",
            ),
            "unique section_id",
        ),
        (
            lambda payload: payload.__setitem__("target_words", 1000),
            "sum of section target_words",
        ),
    ],
)
def test_report_plan_rejects_structural_drift(mutation, message):
    payload = _valid_plan_payload()
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        ReportPlan.model_validate(payload)


def test_report_plan_rejects_unbound_or_inconsistent_chart_references():
    unknown_section = _valid_plan_payload()
    unknown_section["charts"][0]["section_id"] = "missing_section"
    with pytest.raises(ValidationError, match="unknown section_id"):
        ReportPlan.model_validate(unknown_section)

    inconsistent_section = _valid_plan_payload()
    inconsistent_section["charts"][0]["section_id"] = "implications"
    with pytest.raises(ValidationError, match="must be referenced by its assigned section"):
        ReportPlan.model_validate(inconsistent_section)

    unknown_chart = _valid_plan_payload()
    unknown_chart["sections"][3]["chart_refs"] = ["chart_not_planned"]
    with pytest.raises(ValidationError, match="unknown chart_refs"):
        ReportPlan.model_validate(unknown_chart)


def test_report_plan_rejects_evidence_free_analytical_sections_and_charts():
    section_payload = _valid_plan_payload()
    section_payload["sections"][2]["required_evidence_refs"] = []
    with pytest.raises(ValidationError):
        ReportPlan.model_validate(section_payload)

    chart_payload = _valid_plan_payload()
    chart_payload["charts"][0]["evidence_refs"] = []
    with pytest.raises(ValidationError):
        ReportPlan.model_validate(chart_payload)


def test_report_composer_runtime_skill_is_complete_and_stage_addressable():
    assert validate_skills(raise_on_missing=True) == []

    expected_markers = {
        "structure": "# Standard Report Structure",
        "planning": "# Report Planning Contract",
        "section_writing": "# Section Writing Rules",
        "chart_integration": "# Chart Integration Rules",
        "final_assembly": "# Final Assembly Rules",
    }
    for stage, marker in expected_markers.items():
        guidance = get_report_guidance(stage)
        assert marker in guidance
        assert len(guidance) > len(marker)

    with pytest.raises(ValueError, match="Unknown report guidance stage"):
        get_report_guidance("invented_stage")


def test_report_contract_rejects_duplicate_chart_ids():
    payload = _valid_plan_payload()
    duplicate = deepcopy(payload["charts"][0])
    duplicate["section_id"] = "implications"
    payload["sections"][3]["chart_refs"] = [duplicate["chart_id"]]
    payload["charts"].append(duplicate)

    with pytest.raises(ValidationError, match="unique chart_id"):
        ReportPlan.model_validate(payload)


@pytest.mark.parametrize("target", ["section", "chart"])
def test_report_contract_rejects_duplicate_evidence_references(target):
    payload = _valid_plan_payload()
    if target == "section":
        payload["sections"][0]["required_evidence_refs"].append(
            payload["sections"][0]["required_evidence_refs"][0]
        )
    else:
        payload["charts"][0]["evidence_refs"].append(
            payload["charts"][0]["evidence_refs"][0]
        )

    with pytest.raises(ValidationError, match="unique evidence references"):
        ReportPlan.model_validate(payload)
