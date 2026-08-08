"""Evidence-bound report planning tests."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from agent import report_planner
from agent.report_planner import (
    ReportPlanEvidenceError,
    plan_report,
    validate_report_plan_evidence,
)
from contracts.report import ReportPlan
from contracts.report_evidence import ReportEvidenceManifest
from tests.fixtures_report_manifest import (
    LIMIT_REF,
    STATS_REF,
    TABLE_REF,
    _manifest,
    _plan_payload,
)


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
    # Repair grants exactly one substantive ref per section — never the bulk
    # add that made sections unwritable. Summary sections carry a second ref
    # only because their aggregates live in the statistics item.
    summary_kinds = {"executive_summary", "conclusion"}
    assert all(
        len(section.required_evidence_refs)
        == (2 if section.kind.value in summary_kinds else 1)
        for section in plan.sections
    )
    assert all(
        STATS_REF in section.required_evidence_refs
        for section in plan.sections
        if section.kind.value in summary_kinds
    )
    assert plan.charts == []
    assert all(section.chart_refs == [] for section in plan.sections)


def test_planner_defers_chart_buildability_to_single_pass_evaluation(
):
    def invalid_chart_model(*_):
        payload = deepcopy(_plan_payload())
        payload["charts"][0]["purpose"] = "relationship"
        return payload

    assert not hasattr(report_planner, "build_report_charts")
    plan = plan_report(
        "Explain the price trend.",
        _manifest(),
        invoke_model=invalid_chart_model,
    )

    validate_report_plan_evidence(plan, _manifest())
    assert [chart.chart_id for chart in plan.charts] == ["price_trend"]
    assert plan.sections[2].chart_refs == ["price_trend"]


def test_plan_report_repairs_one_invalid_plan_before_failing():
    calls = {"plan": 0, "repair": 0}

    def invalid_plan(*_args, **_kwargs):
        calls["plan"] += 1
        payload = deepcopy(_plan_payload())
        payload["sections"][0]["kind"] = "conclusion"
        return payload

    def repair(*args, **_kwargs):
        calls["repair"] += 1
        calls["error_codes"] = args[-1]
        return deepcopy(_plan_payload())

    plan = plan_report(
        "Explain the price trend.",
        _manifest(),
        invoke_model=invalid_plan,
        repair_model=repair,
    )

    assert calls["plan"] == 1
    assert calls["repair"] == 1
    assert calls["error_codes"] == ["PLAN_SCHEMA_INVALID"]
    assert plan.contract_version == "report-plan-v1"


def test_plan_report_normalizes_language_instead_of_spending_a_repair():
    """Code owns intent and language, so a mismatch is corrected, not repaired."""

    def wrong_language(*_args, **_kwargs):
        payload = deepcopy(_plan_payload())
        payload["language_code"] = "ka"
        payload["intent"] = "forecast"
        return payload

    def repair(*_args, **_kwargs):
        raise AssertionError("normalized semantics must not reach the repair call")

    plan = plan_report(
        "Explain the price trend.",
        _manifest(),
        invoke_model=wrong_language,
        repair_model=repair,
    )

    assert plan.language_code == "en"
    assert plan.intent.value == "general"


def test_plan_report_raises_when_the_repair_is_also_invalid():
    def invalid_plan(*_args, **_kwargs):
        payload = deepcopy(_plan_payload())
        payload["sections"][0]["kind"] = "conclusion"
        return payload

    with pytest.raises(ValidationError):
        plan_report(
            "Explain the price trend.",
            _manifest(),
            invoke_model=invalid_plan,
            repair_model=invalid_plan,
        )


def test_summary_sections_receive_statistics_evidence_for_their_aggregates():
    """Aggregates live in the statistics item, and a mean over a long table
    cannot be expressed within the 32-operand derived-claim limit."""

    def table_only_model(*_args, **_kwargs):
        payload = deepcopy(_plan_payload())
        for section in payload["sections"]:
            if section["kind"] in {"executive_summary", "conclusion"}:
                section["required_evidence_refs"] = [TABLE_REF]
        return payload

    plan = plan_report(
        "Explain the price trend.",
        _manifest(),
        invoke_model=table_only_model,
    )

    by_kind = {section.kind.value: section for section in plan.sections}
    assert STATS_REF in by_kind["executive_summary"].required_evidence_refs
    assert STATS_REF in by_kind["conclusion"].required_evidence_refs
    assert TABLE_REF in by_kind["executive_summary"].required_evidence_refs


def test_summary_sections_are_not_given_a_second_statistics_reference():
    plan = plan_report(
        "Explain the price trend.",
        _manifest(),
        invoke_model=lambda *_a, **_k: deepcopy(_plan_payload()),
    )

    executive_summary = plan.sections[0]
    assert executive_summary.required_evidence_refs.count(STATS_REF) == 1


def test_summary_sections_are_untouched_when_no_statistics_evidence_exists():
    manifest_payload = _manifest().model_dump(mode="json")
    manifest_payload["items"] = [
        item
        for item in manifest_payload["items"]
        if item["kind"] != "statistics"
    ]
    manifest = ReportEvidenceManifest.model_validate(manifest_payload)

    def table_only_model(*_args, **_kwargs):
        payload = deepcopy(_plan_payload())
        payload["evidence_manifest_id"] = manifest.manifest_id
        for section in payload["sections"]:
            if section["kind"] != "limitations":
                section["required_evidence_refs"] = [TABLE_REF]
        return payload

    plan = plan_report(
        "Explain the price trend.",
        manifest,
        invoke_model=table_only_model,
    )

    assert plan.sections[0].required_evidence_refs == [TABLE_REF]
