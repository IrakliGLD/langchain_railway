"""Tests for the question-analysis runtime contract."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts.question_analysis import (
    DerivedMetricName,
    DerivedMetricRequest,
    QuestionAnalysis,
    QueryType,
    ScenarioAggregation,
)


def _valid_payload() -> dict:
    return {
        "version": "question_analysis_v1",
        "raw_query": "why does balancing electricity price changed in november 2021?",
        "canonical_query_en": "Why did balancing electricity price change in November 2021?",
        "language": {
            "input_language": "en",
            "answer_language": "en",
        },
        "classification": {
            "query_type": "data_explanation",
            "analysis_mode": "analyst",
            "intent": "balancing_price_why",
            "needs_clarification": False,
            "confidence": 0.93,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "sql",
            "needs_sql": True,
            "needs_knowledge": True,
            "prefer_tool": False,
        },
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.97},
                {"name": "currency_influence", "score": 0.76},
            ]
        },
        "tooling": {
            "candidate_tools": [
                {
                    "name": "get_prices",
                    "score": 0.88,
                    "reason": "Primary metric is balancing price.",
                    "params_hint": {
                        "metric": "p_bal_gel",
                        "currency": "gel",
                        "granularity": "monthly",
                        "start_date": "2021-11-01",
                        "end_date": "2021-11-30",
                        "entities": [],
                        "types": [],
                        "mode": None,
                    },
                }
            ]
        },
        "sql_hints": {
            "metric": "p_bal_gel",
            "entities": [],
            "aggregation": "monthly",
            "dimensions": ["price", "xrate", "share"],
            "period": {
                "kind": "month",
                "start_date": "2021-11-01",
                "end_date": "2021-11-30",
                "granularity": "month",
                "raw_text": "November 2021",
            },
        },
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.82,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": True,
            "needs_trend_context": False,
            "needs_correlation_context": True,
            "derived_metrics": [
                {"metric_name": "mom_absolute_change", "metric": "p_bal_gel", "target_metric": None, "rank_limit": None},
                {"metric_name": "mom_percent_change", "metric": "p_bal_gel", "target_metric": None, "rank_limit": None},
                {"metric_name": "share_delta_mom", "metric": "share_import", "target_metric": None, "rank_limit": None},
            ],
        },
    }


def test_valid_question_analysis_payload():
    payload = _valid_payload()

    model = QuestionAnalysis.model_validate(payload)

    assert model.version == "question_analysis_v1"
    assert model.classification.intent == "balancing_price_why"
    assert model.sql_hints.period is not None
    assert model.sql_hints.period.start_date == "2021-11-01"


def test_required_version_field():
    payload = _valid_payload()
    payload.pop("version")

    with pytest.raises(ValidationError):
        QuestionAnalysis.model_validate(payload)


def test_whitespace_only_text_is_rejected():
    payload = _valid_payload()
    payload["raw_query"] = "   "

    with pytest.raises(ValidationError):
        QuestionAnalysis.model_validate(payload)


def test_unknown_field_is_rejected():
    payload = _valid_payload()
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        QuestionAnalysis.model_validate(payload)


def test_invalid_enum_is_rejected():
    payload = _valid_payload()
    payload["routing"]["preferred_path"] = "maybe"

    with pytest.raises(ValidationError):
        QuestionAnalysis.model_validate(payload)


def test_period_end_must_not_precede_start():
    payload = _valid_payload()
    payload["sql_hints"]["period"]["end_date"] = "2021-10-31"

    with pytest.raises(ValidationError):
        QuestionAnalysis.model_validate(payload)


def test_schema_snapshot_matches_runtime_model():
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "question_analysis.schema.json"
    snapshot = json.loads(schema_path.read_text(encoding="utf-8"))

    assert snapshot == QuestionAnalysis.model_json_schema()


def test_regulatory_procedure_query_type_is_supported():
    payload = _valid_payload()
    payload["classification"]["query_type"] = "regulatory_procedure"
    model = QuestionAnalysis.model_validate(payload)
    assert model.classification.query_type == QueryType.REGULATORY_PROCEDURE


# ---------------------------------------------------------------------------
# Scenario contract tests
# ---------------------------------------------------------------------------


def test_scenario_scale_valid():
    req = DerivedMetricRequest(
        metric_name="scenario_scale",
        metric="p_bal_gel",
        scenario_factor=1.34,
    )
    assert req.scenario_factor == 1.34
    assert req.scenario_volume is None
    assert req.scenario_aggregation is None


def test_scenario_payoff_volume_defaults_to_1():
    req = DerivedMetricRequest(
        metric_name="scenario_payoff",
        metric="p_bal_usd",
        scenario_factor=60.0,
    )
    assert req.scenario_volume == 1.0


def test_scenario_factor_required_for_scenario_metrics():
    with pytest.raises(ValidationError, match="scenario_factor"):
        DerivedMetricRequest(
            metric_name="scenario_scale",
            metric="p_bal_gel",
        )


def test_scenario_fields_forbidden_on_non_scenario():
    with pytest.raises(ValidationError, match="scenario fields must be None"):
        DerivedMetricRequest(
            metric_name="mom_absolute_change",
            metric="p_bal_gel",
            scenario_factor=1.5,
        )


def test_scenario_aggregation_enum_values():
    for agg in ("sum", "mean", "min", "max"):
        req = DerivedMetricRequest(
            metric_name="scenario_offset",
            metric="p_bal_usd",
            scenario_factor=10.0,
            scenario_aggregation=agg,
        )
        assert req.scenario_aggregation == ScenarioAggregation(agg)

    with pytest.raises(ValidationError):
        DerivedMetricRequest(
            metric_name="scenario_offset",
            metric="p_bal_usd",
            scenario_factor=10.0,
            scenario_aggregation="median",
        )
