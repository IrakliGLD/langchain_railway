"""TDD coverage for the bounded report research planner and validator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.report_research_planner import (
    ReportResearchPlanError,
    build_report_planning_constraints,
    plan_report_research,
    validate_report_research_plan,
)
from contracts.report_research import ReportResearchPlan
from tests.test_report_research_contract import _research_plan_payload

_COMPOUND_QUERY = (
    "Assess current market model and prices. What is the deregulation stage "
    "and how will the new market model impact the market and energy security?"
)


def _bound_plan_payload(query: str = _COMPOUND_QUERY) -> dict:
    return _research_plan_payload(
        query_digest=hashlib.sha256(query.encode("utf-8")).hexdigest()
    )


def test_research_plan_validator_accepts_compound_required_coverage():
    assessment = validate_report_research_plan(
        _COMPOUND_QUERY,
        ReportResearchPlan.model_validate(_bound_plan_payload()),
        max_tracks=4,
    )

    assert assessment.valid is True
    assert assessment.finding_codes == []
    assert set(assessment.recognized_requirements) == {
        "energy_security",
        "market_knowledge",
        "prices",
    }


def test_planning_constraints_publish_required_exhibits_before_llm_call():
    constraints = build_report_planning_constraints(_COMPOUND_QUERY)

    assert constraints.maximum_total_exhibits == 4
    assert {
        (
            exhibit.requirement.value,
            exhibit.collector_id.value,
            exhibit.purpose.value,
        )
        for exhibit in constraints.required_exhibits
    } == {
        ("prices", "prices", "trend"),
        ("energy_security", "generation_mix", "composition"),
    }


def test_research_plan_validator_accepts_four_total_exhibits():
    payload = _bound_plan_payload()
    payload["tracks"][0]["expected_exhibits"].append("comparison")

    assessment = validate_report_research_plan(
        _COMPOUND_QUERY,
        ReportResearchPlan.model_validate(payload),
        max_tracks=4,
    )

    assert assessment.valid is True
    assert "EXHIBIT_LIMIT_EXCEEDED" not in assessment.finding_codes


def test_research_plan_validator_rejects_five_total_exhibits():
    payload = _bound_plan_payload()
    payload["tracks"][0]["expected_exhibits"].extend(
        ["comparison", "relationship"]
    )

    assessment = validate_report_research_plan(
        _COMPOUND_QUERY,
        ReportResearchPlan.model_validate(payload),
        max_tracks=4,
    )

    assert assessment.valid is False
    assert "EXHIBIT_LIMIT_EXCEEDED" in assessment.finding_codes


def test_research_plan_validator_rejects_missing_collectors_and_exhibits():
    payload = _bound_plan_payload()
    payload["tracks"][0]["collector_ids"] = ["tariffs"]
    payload["tracks"][1]["expected_exhibits"] = []
    assessment = validate_report_research_plan(
        _COMPOUND_QUERY,
        ReportResearchPlan.model_validate(payload),
        max_tracks=4,
    )

    assert assessment.valid is False
    assert set(assessment.finding_codes) >= {
        "PRICE_COLLECTOR_MISSING",
        "SECURITY_COMPOSITION_EXHIBIT_MISSING",
    }


def test_research_planner_binds_identity_language_and_uses_one_model_call():
    calls = []
    raw_payload = _bound_plan_payload("placeholder")
    raw_payload["language_code"] = "ru"

    def invoke_model(
        query: str,
        *,
        language_code: str,
        max_tracks: int,
        planning_constraints,
        topic_knowledge: str = "",
    ):
        calls.append(
            (
                query,
                language_code,
                max_tracks,
                planning_constraints,
            )
        )
        return raw_payload

    plan = plan_report_research(
        _COMPOUND_QUERY,
        max_tracks=4,
        invoke_model=invoke_model,
    )

    assert len(calls) == 1
    assert calls[0][:3] == (_COMPOUND_QUERY, "en", 4)
    assert calls[0][3] == build_report_planning_constraints(
        _COMPOUND_QUERY
    )
    assert plan.query_digest == hashlib.sha256(
        _COMPOUND_QUERY.encode("utf-8")
    ).hexdigest()
    assert plan.language_code == "en"


def test_research_planner_bounds_excess_exhibits_without_another_model_call():
    calls = []
    raw_payload = _bound_plan_payload()
    raw_payload["tracks"][0]["expected_exhibits"].extend(
        ["comparison", "relationship"]
    )
    raw_payload["tracks"] = [
        raw_payload["tracks"][0],
        raw_payload["tracks"][2],
        raw_payload["tracks"][1],
    ]

    def invoke_model(*_args, **_kwargs):
        calls.append("called")
        return raw_payload

    plan = plan_report_research(
        _COMPOUND_QUERY,
        max_tracks=4,
        invoke_model=invoke_model,
    )

    assert calls == ["called"]
    assert sum(
        len(track.expected_exhibits) for track in plan.tracks
    ) == 4
    assert plan.tracks[0].expected_exhibits == [
        "trend",
        "comparison",
        "relationship",
    ]
    assert plan.tracks[1].expected_exhibits == []
    assert plan.tracks[2].expected_exhibits == ["composition"]


def test_research_planner_converts_invoker_schema_errors_to_safe_findings():
    payload = _bound_plan_payload()
    payload["tracks"][0]["collector_ids"] = ["private-invalid-collector"]

    def invoke_model(*_args, **_kwargs):
        return ReportResearchPlan.model_validate(payload)

    with pytest.raises(ReportResearchPlanError) as exc_info:
        plan_report_research(
            _COMPOUND_QUERY,
            max_tracks=4,
            invoke_model=invoke_model,
        )

    assert exc_info.value.assessment.finding_codes == [
        "PLAN_SCHEMA_INVALID"
    ]
    assert (
        "SCHEMA_TRACKS_ITEM_COLLECTOR_IDS_ITEM_ENUM"
        in exc_info.value.schema_error_codes
    )
    assert "private-invalid-collector" not in str(exc_info.value)


def test_research_planner_does_not_spend_a_second_call_on_invalid_plan():
    calls = []
    raw_payload = _bound_plan_payload()
    raw_payload["tracks"][0]["collector_ids"] = ["tariffs"]

    def invoke_model(*_args, **_kwargs):
        calls.append("called")
        return raw_payload

    with pytest.raises(
        ReportResearchPlanError,
        match="PRICE_COLLECTOR_MISSING",
    ):
        plan_report_research(
            _COMPOUND_QUERY,
            max_tracks=4,
            invoke_model=invoke_model,
        )

    assert calls == ["called"]


def test_golden_query_profiles_are_recognized_without_an_analyzer_call():
    cases = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "report_research_golden_cases.json"
        ).read_text(encoding="utf-8")
    )["cases"]
    expected = {
        "compound": {"prices", "energy_security", "market_knowledge"},
        "quantitative": {"prices"},
        "knowledge": {"market_knowledge"},
        "mixed": {"prices", "market_knowledge"},
        "noisy": {"prices", "energy_security"},
        "multilingual": {"prices", "energy_security"},
        "ambiguous": {"market_knowledge"},
    }

    for case in cases:
        payload = _bound_plan_payload(case["query"])
        payload["language_code"] = case["language_code"]
        assessment = validate_report_research_plan(
            case["query"],
            ReportResearchPlan.model_validate(payload),
            max_tracks=4,
        )
        assert {
            requirement.value
            for requirement in assessment.recognized_requirements
        } == expected[case["profile"]]
        assert assessment.valid is True


def test_validator_rejects_unrequested_expensive_engines():
    payload = _bound_plan_payload()
    payload["tracks"][0]["collector_ids"].append("forecast_engine")
    assessment = validate_report_research_plan(
        _COMPOUND_QUERY,
        ReportResearchPlan.model_validate(payload),
        max_tracks=4,
    )

    assert "UNREQUESTED_FORECAST_COLLECTOR" in assessment.finding_codes


def test_planner_injects_topic_knowledge_for_the_request(monkeypatch):
    """The planner must see what the system knows about the topic first.

    Knowledge retrieval used to run per-track, after the tracks were already
    chosen, so the planner decided coverage without ever consulting
    knowledge/balancing_price.md -- whose own title is "Formation and Drivers".
    """
    from agent import report_research_planner
    from knowledge import load_knowledge

    # Same precondition the worker establishes at startup (report_worker.py).
    load_knowledge()
    query = "create a report about balancing prices in georgia"
    captured = {}

    def fake_invoke(user_query, **kwargs):
        captured.update(kwargs)
        captured["user_query"] = user_query
        return ReportResearchPlan.model_validate(
            _research_plan_payload(
                query_digest=hashlib.sha256(query.encode("utf-8")).hexdigest()
            )
        )

    report_research_planner.plan_report_research(
        query,
        max_tracks=4,
        invoke_model=fake_invoke,
    )

    knowledge = captured.get("topic_knowledge") or ""
    assert knowledge, "planner received no topic knowledge"
    assert "balancing" in knowledge.lower()


def test_injected_topic_knowledge_is_bounded(monkeypatch):
    """A whole knowledge file would crowd out the request and the catalog."""
    from agent import report_research_planner
    from knowledge import load_knowledge

    load_knowledge()
    knowledge = report_research_planner._planning_topic_knowledge(
        "create a report about balancing prices in georgia"
    )

    assert 0 < len(knowledge) <= report_research_planner._PLANNING_KNOWLEDGE_BUDGET_CHARS


def test_planning_topic_knowledge_survives_a_knowledge_failure(monkeypatch):
    """Planning must not die because a knowledge lookup did.

    The report already fails closed in enough places; losing the whole job to
    an optional prompt enrichment would be a strictly worse trade.
    """
    from agent import report_research_planner

    monkeypatch.setattr(
        report_research_planner,
        "infer_topic_matches",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("knowledge down")),
    )

    assert report_research_planner._planning_topic_knowledge("anything") == ""
