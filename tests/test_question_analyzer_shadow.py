"""Tests for question-analyzer shadow-mode integration."""

import json
import os

import pytest
import sqlalchemy

# Ensure config validation passes before importing modules that depend on config.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


class _DummyResult:
    def fetchall(self):
        return []

    def keys(self):
        return []


class _DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        return _DummyResult()


class _DummyEngine:
    def connect(self):
        return _DummyConnection()


sqlalchemy.create_engine = lambda *args, **kwargs: _DummyEngine()  # type: ignore[assignment]

from agent import pipeline, planner  # noqa: E402
from contracts.question_analysis import QuestionAnalysis  # noqa: E402
from core import llm as llm_core  # noqa: E402
from models import QueryContext  # noqa: E402


def _valid_payload() -> dict:
    return {
        "version": "question_analysis_v1",
        "raw_query": "what is genex?",
        "canonical_query_en": "What is GENEX?",
        "language": {
            "input_language": "en",
            "answer_language": "en",
        },
        "classification": {
            "query_type": "conceptual_definition",
            "analysis_mode": "light",
            "intent": "market_participant_definition",
            "needs_clarification": False,
            "confidence": 0.98,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
        },
        "knowledge": {
            "candidate_topics": [{"name": "market_structure", "score": 0.98}],
        },
        "tooling": {"candidate_tools": []},
        "sql_hints": {
            "metric": None,
            "entities": ["genex"],
            "aggregation": None,
            "dimensions": [],
            "period": None,
        },
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.95,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_trend_context": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }


class _DummyCache:
    def __init__(self):
        self._store = {}

    def get(self, prompt: str):
        return self._store.get(prompt)

    def set(self, prompt: str, response: str):
        self._store[prompt] = response


class _DummyMessage:
    def __init__(self, payload: dict):
        self.content = json.dumps(payload)
        self.usage_metadata = {}
        self.response_metadata = {}


def test_llm_analyze_question_returns_validated_contract(monkeypatch):
    payload = _valid_payload()

    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "make_openai", lambda: object())
    monkeypatch.setattr(
        llm_core,
        "_invoke_with_resilience",
        lambda *_args, **_kwargs: _DummyMessage(payload),
    )
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    model = llm_core.llm_analyze_question("what is genex?")

    assert isinstance(model, QuestionAnalysis)
    assert model.classification.intent == "market_participant_definition"
    assert model.routing.preferred_path.value == "knowledge"


def test_planner_shadow_records_analysis(monkeypatch):
    expected = QuestionAnalysis.model_validate(_valid_payload())
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(query="what is genex?")
    out = planner.analyze_question_shadow(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.classification.query_type.value == "conceptual_definition"
    assert out.question_analysis_error == ""
    assert out.question_analysis_source == "llm_shadow"


def test_planner_shadow_logs_prompt_validation_artifact(monkeypatch):
    expected = QuestionAnalysis.model_validate(_valid_payload())
    captured = {}

    monkeypatch.setattr(planner, "ENABLE_TRACE_DEBUG_ARTIFACTS", True)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)
    monkeypatch.setattr(
        planner,
        "build_question_analyzer_prompt_validation_artifacts",
        lambda *_args, **_kwargs: {
            "current_prompt_chars": 1200,
            "legacy_prompt_chars": 2400,
            "chars_saved_vs_legacy": 1200,
        },
    )

    def _capture_trace(_log, _ctx, _stage, event, *, debug=False, **extra):
        if event == "artifact":
            captured["debug"] = debug
            captured["extra"] = extra

    monkeypatch.setattr(planner, "trace_detail", _capture_trace)

    ctx = QueryContext(query="what is genex?")
    out = planner.analyze_question_shadow(ctx)

    assert out.question_analysis_source == "llm_shadow"
    assert captured["debug"] is True
    assert captured["extra"]["prompt_validation"]["chars_saved_vs_legacy"] == 1200


def test_planner_shadow_records_error_without_raising(monkeypatch):
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad-json")))

    ctx = QueryContext(query="what is genex?")
    out = planner.analyze_question_shadow(ctx)

    assert out.question_analysis is None
    assert "bad-json" in out.question_analysis_error
    assert out.question_analysis_source == "llm_shadow_error"


def test_active_planner_coerces_month_specific_balancing_why_query(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "Why balancing electricity prices changed in November 2024?",
        "canonical_query_en": "Explain the reasons for changes in balancing electricity prices in November 2024.",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "unsupported",
            "analysis_mode": "light",
            "intent": "balancing_price_change_reason",
            "needs_clarification": False,
            "confidence": 0.99,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [{"name": "balancing_price", "score": 0.9}],
        },
        "tooling": {"candidate_tools": []},
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(query="Why balancing electricity prices changed in November 2024?")
    out = planner.analyze_question_active(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.classification.query_type.value == "data_explanation"
    assert out.question_analysis.classification.analysis_mode.value == "analyst"
    assert out.question_analysis.routing.preferred_path.value == "tool"
    assert out.question_analysis.routing.needs_multi_tool is True
    tool_names = [tool.name.value for tool in out.question_analysis.tooling.candidate_tools]
    assert tool_names[:2] == ["get_prices", "get_balancing_composition"]
    metric_names = {metric.metric_name.value for metric in out.question_analysis.analysis_requirements.derived_metrics}
    assert "mom_absolute_change" in metric_names
    assert out.question_analysis.sql_hints.period is not None
    assert out.question_analysis.sql_hints.period.start_date == "2024-11-01"
    assert out.question_analysis.sql_hints.period.end_date == "2024-11-30"


def test_active_planner_coerces_between_month_balancing_why_query(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "Why did balancing electricity price change between October and November 2024?",
        "canonical_query_en": "Why did balancing electricity price change between October and November 2024?",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "unsupported",
            "analysis_mode": "light",
            "intent": "balancing_price_change_reason",
            "needs_clarification": False,
            "confidence": 0.99,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [{"name": "balancing_price", "score": 0.9}],
        },
        "tooling": {"candidate_tools": []},
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(query="Why did balancing electricity price change between October and November 2024?")
    out = planner.analyze_question_active(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.classification.query_type.value == "data_explanation"
    assert out.question_analysis.classification.analysis_mode.value == "analyst"
    assert out.question_analysis.routing.preferred_path.value == "tool"
    assert out.question_analysis.routing.needs_multi_tool is True
    tool_names = [tool.name.value for tool in out.question_analysis.tooling.candidate_tools]
    assert tool_names[:2] == ["get_prices", "get_balancing_composition"]
    metric_names = {metric.metric_name.value for metric in out.question_analysis.analysis_requirements.derived_metrics}
    assert {"mom_absolute_change", "mom_percent_change", "yoy_absolute_change", "yoy_percent_change"} <= metric_names
    assert out.question_analysis.sql_hints.period is not None
    assert out.question_analysis.sql_hints.period.start_date == "2024-11-01"
    assert out.question_analysis.sql_hints.period.granularity.value == "month"


def test_active_planner_coerces_russian_month_specific_balancing_why_query(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "Почему изменилась цена балансирующей электроэнергии в ноябре 2024 года?",
        "canonical_query_en": "Why did balancing electricity price change in November 2024?",
        "language": {"input_language": "ru", "answer_language": "ru"},
        "classification": {
            "query_type": "unsupported",
            "analysis_mode": "light",
            "intent": "balancing_price_change_reason",
            "needs_clarification": False,
            "confidence": 0.99,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [{"name": "balancing_price", "score": 0.9}],
        },
        "tooling": {"candidate_tools": []},
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(query="Почему изменилась цена балансирующей электроэнергии в ноябре 2024 года?")
    out = planner.analyze_question_active(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.classification.query_type.value == "data_explanation"
    assert out.question_analysis.routing.preferred_path.value == "tool"
    assert out.question_analysis.sql_hints.period is not None
    assert out.question_analysis.sql_hints.period.start_date == "2024-11-01"
    assert out.question_analysis.sql_hints.period.end_date == "2024-11-30"


def test_active_planner_coerces_simple_balancing_price_forecast_query(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "Forecast balancing electricity price for 2030.",
        "canonical_query_en": "Forecast the balancing electricity price for the year 2030.",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "ambiguous",
            "analysis_mode": "light",
            "intent": "balancing_price_forecast",
            "needs_clarification": False,
            "confidence": 0.8,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [{"name": "balancing_price", "score": 0.9}],
        },
        "tooling": {
            "candidate_tools": [{"name": "get_prices", "score": 0.92}],
        },
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_trend_context": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(query="Forecast balancing electricity price for 2030.")
    out = planner.analyze_question_active(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.classification.query_type.value == "forecast"
    assert out.question_analysis.classification.analysis_mode.value == "analyst"
    assert out.question_analysis.routing.preferred_path.value == "tool"
    assert out.question_analysis.routing.needs_multi_tool is False
    assert [tool.name.value for tool in out.question_analysis.tooling.candidate_tools] == [
        "get_prices",
    ]
    assert out.question_analysis.sql_hints.metric == "balancing"
    metric_names = {metric.metric_name.value for metric in out.question_analysis.analysis_requirements.derived_metrics}
    assert "trend_slope" in metric_names


def test_active_planner_does_not_coerce_forecast_methodology_question(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "Why is forecasting balancing electricity price difficult?",
        "canonical_query_en": "Explain why forecasting balancing electricity price is difficult.",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "ambiguous",
            "analysis_mode": "light",
            "intent": "forecasting_methodology",
            "needs_clarification": False,
            "confidence": 0.85,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [{"name": "balancing_price", "score": 0.9}],
        },
        "tooling": {
            "candidate_tools": [{"name": "get_prices", "score": 0.92}],
        },
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(query="Why is forecasting balancing electricity price difficult?")
    out = planner.analyze_question_active(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.classification.query_type.value == "ambiguous"
    assert out.question_analysis.routing.preferred_path.value == "knowledge"


def test_active_planner_coerces_underdefined_numeric_computation_to_clarify(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "For the following periods, knowing the balancing electricity price and tariffs for regulated hydro and thermal, what is the weighted average price of the remaining energy?",
        "canonical_query_en": "Calculate the weighted average price of the remaining energy for the specified periods.",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "ambiguous",
            "analysis_mode": "light",
            "intent": "residual_weighted_price_calculation",
            "needs_clarification": False,
            "confidence": 0.8,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.95},
                {"name": "tariffs", "score": 0.9},
                {"name": "generation_mix", "score": 0.8},
            ],
        },
        "tooling": {
            "candidate_tools": [
                {"name": "get_prices", "score": 0.98},
                {"name": "get_tariffs", "score": 0.95},
                {"name": "get_balancing_composition", "score": 0.9},
            ],
        },
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(
        query=(
            "For the following periods, knowing the balancing electricity price and tariffs for regulated hydro and thermal, "
            "what is the weighted average price of the remaining energy?"
        )
    )
    out = planner.analyze_question_active(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.classification.query_type.value == "ambiguous"
    assert out.question_analysis.classification.needs_clarification is True
    assert out.question_analysis.routing.preferred_path.value == "clarify"
    assert out.clarify_reason == "underdefined_computed_target"


def test_active_planner_resolves_explicit_residual_bucket_query_from_current_turn(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "For the following periods, knowing the balancing electricity price, tariffs for regulated hydro and thermal, and deregulated power plant prices, what is the weighted average price of the remaining energy sold on the balancing market?",
        "canonical_query_en": "Calculate the weighted average price of electricity sold on the balancing market, excluding regulated hydro, regulated thermal, and deregulated power plants, for the specified months.",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "unsupported",
            "analysis_mode": "analyst",
            "intent": "residual_weighted_price_calculation",
            "needs_clarification": False,
            "confidence": 1.0,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.95},
                {"name": "tariffs", "score": 0.9},
                {"name": "market_structure", "score": 0.8},
            ],
        },
        "tooling": {
            "candidate_tools": [
                {"name": "get_prices", "score": 0.98},
                {"name": "get_tariffs", "score": 0.95},
                {"name": "get_balancing_composition", "score": 0.9},
            ],
        },
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(
        query=(
            "For the following periods, knowing the balancing electricity price, tariffs for regulated hydro and thermal, "
            "and deregulated power plant prices, what is the weighted average price of the remaining energy sold on the balancing market?"
        )
    )
    out = planner.analyze_question_active(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.classification.query_type.value == "data_retrieval"
    assert out.question_analysis.classification.needs_clarification is False
    assert out.question_analysis.routing.preferred_path.value == "tool"
    assert out.question_analysis.routing.needs_multi_tool is True
    assert [tool.name.value for tool in out.question_analysis.tooling.candidate_tools] == [
        "get_prices",
        "get_balancing_composition",
        "get_tariffs",
    ]
    assert out.clarify_reason == ""


def test_active_planner_resolves_deregulated_plants_alias_for_residual_bucket(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": (
            "Different entities sold on balancing segment. For some, like regulated hydro, "
            "deregulated plants, and all regulated thermals, the prices are known. "
            "Calculate the weighted average price of the remaining electricity."
        ),
        "canonical_query_en": (
            "Calculate the weighted average price of electricity sold on the balancing "
            "segment excluding regulated hydro, deregulated plants, and all regulated thermals."
        ),
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "ambiguous",
            "analysis_mode": "light",
            "intent": "residual_weighted_price_calculation",
            "needs_clarification": True,
            "confidence": 0.6,
            "ambiguities": ["remaining bucket could be interpreted multiple ways"],
        },
        "routing": {
            "preferred_path": "clarify",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.95},
                {"name": "tariffs", "score": 0.9},
            ],
        },
        "tooling": {
            "candidate_tools": [
                {"name": "get_prices", "score": 0.98},
                {"name": "get_tariffs", "score": 0.95},
                {"name": "get_balancing_composition", "score": 0.9},
            ],
        },
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(query=payload["raw_query"])
    out = planner.analyze_question_active(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.classification.query_type.value == "data_retrieval"
    assert out.question_analysis.classification.needs_clarification is False
    assert out.question_analysis.routing.preferred_path.value == "tool"
    assert out.question_analysis.routing.needs_multi_tool is True
    assert [tool.name.value for tool in out.question_analysis.tooling.candidate_tools] == [
        "get_prices",
        "get_balancing_composition",
        "get_tariffs",
    ]
    assert out.clarify_reason == ""


def test_active_planner_residual_month_list_guardrail_uses_sparse_month_bounds(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": (
            "Different entities sold on balancing segment. For some, like regulated hydro, deregulated plants, "
            "all regulated thermals, the prices are known. I want to calculate the weighted average price "
            "for the remaining electricity for the following dates: June 2020, July 2020, July 2021, "
            "September 2021, October 2021."
        ),
        "canonical_query_en": (
            "Calculate the weighted average price for electricity on the balancing segment, excluding "
            "regulated hydro, deregulated plants, and all regulated thermals, for June 2020, July 2020, "
            "July 2021, September 2021, and October 2021."
        ),
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "ambiguous",
            "analysis_mode": "light",
            "intent": "residual_weighted_price_calculation",
            "needs_clarification": True,
            "confidence": 0.6,
            "ambiguities": ["remaining bucket could be interpreted multiple ways"],
        },
        "routing": {
            "preferred_path": "clarify",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.95},
                {"name": "tariffs", "score": 0.9},
            ],
        },
        "tooling": {
            "candidate_tools": [
                {"name": "get_prices", "score": 0.98},
                {"name": "get_tariffs", "score": 0.95},
                {"name": "get_balancing_composition", "score": 0.9},
            ],
        },
        "sql_hints": {
            "metric": "balancing",
            "entities": [],
            "aggregation": None,
            "dimensions": [],
            "period": {
                "kind": "range",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "granularity": "range",
                "raw_text": "2020",
            },
        },
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(query=payload["raw_query"])
    out = planner.analyze_question_active(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.routing.preferred_path.value == "tool"
    assert out.question_analysis.tooling.candidate_tools[0].params_hint is not None
    assert out.question_analysis.tooling.candidate_tools[0].params_hint.start_date == "2020-06-01"
    assert out.question_analysis.tooling.candidate_tools[0].params_hint.end_date == "2021-10-01"
    assert out.question_analysis.tooling.candidate_tools[1].params_hint.start_date == "2020-06-01"
    assert out.question_analysis.tooling.candidate_tools[1].params_hint.end_date == "2021-10-01"
    assert out.question_analysis.tooling.candidate_tools[2].params_hint.start_date == "2020-06-01"
    assert out.question_analysis.tooling.candidate_tools[2].params_hint.end_date == "2021-10-01"


def test_active_planner_uses_history_to_resolve_residual_bucket_followup(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": (
            "Using the provided percentages, what is the weighted average price of the remaining energy "
            "for these periods given balancing electricity price and regulated hydro and thermal tariffs?"
        ),
        "canonical_query_en": (
            "Calculate the weighted average price of remaining energy for specified periods, "
            "given balancing electricity price and regulated hydro and thermal tariffs, using provided percentages."
        ),
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "ambiguous",
            "analysis_mode": "light",
            "intent": "residual_weighted_price_followup",
            "needs_clarification": True,
            "confidence": 0.7,
            "ambiguities": ["residual bucket not explicit in current turn"],
        },
        "routing": {
            "preferred_path": "clarify",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.95},
                {"name": "tariffs", "score": 0.9},
            ],
        },
        "tooling": {
            "candidate_tools": [
                {"name": "get_prices", "score": 0.95},
                {"name": "get_tariffs", "score": 0.93},
            ],
        },
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: expected)

    ctx = QueryContext(
        query=(
            "Using the provided percentages, what is the weighted average price of the remaining energy "
            "for these periods given balancing electricity price and regulated hydro and thermal tariffs?"
        ),
        conversation_history=[
            {
                "question": (
                    "For these periods, knowing the balancing electricity price, tariffs for regulated hydro "
                    "and thermal, and deregulated power plant prices, what is the weighted average price "
                    "of the other electricity sold on the balancing market?"
                ),
                "answer": "Please clarify what you mean by remaining energy.",
            }
        ],
    )
    out = planner.analyze_question_active(ctx)

    assert out.question_analysis is not None
    assert out.question_analysis.classification.query_type.value == "data_retrieval"
    assert out.question_analysis.classification.needs_clarification is False
    assert out.question_analysis.routing.preferred_path.value == "tool"
    assert out.question_analysis.routing.needs_multi_tool is True
    assert [tool.name.value for tool in out.question_analysis.tooling.candidate_tools] == [
        "get_prices",
        "get_balancing_composition",
        "get_tariffs",
    ]
    assert out.clarify_reason == ""


def test_pipeline_runs_shadow_stage_before_conceptual_return(monkeypatch):
    expected = QuestionAnalysis.model_validate(_valid_payload())

    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_SHADOW", True)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_HINTS", False)
    monkeypatch.setattr(pipeline.planner, "prepare_context", lambda ctx: setattr(ctx, "is_conceptual", True) or ctx)
    monkeypatch.setattr(pipeline.planner, "analyze_question_shadow", lambda ctx: setattr(ctx, "question_analysis", expected) or ctx)
    monkeypatch.setattr(
        pipeline.summarizer,
        "answer_conceptual",
        lambda ctx: setattr(ctx, "summary", "Conceptual answer") or ctx,
    )

    out = pipeline.process_query("what is genex?", trace_id="trace-qa", session_id="session-qa")

    assert out.summary == "Conceptual answer"
    assert out.question_analysis is not None
    assert out.stage_timings_ms["stage_0_2_question_analyzer"] >= 0.0


def test_pipeline_runs_active_stage_when_hints_enabled(monkeypatch):
    expected = QuestionAnalysis.model_validate(_valid_payload())

    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_SHADOW", False)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_HINTS", True)
    monkeypatch.setattr(pipeline.planner, "prepare_context", lambda ctx: setattr(ctx, "is_conceptual", True) or ctx)
    monkeypatch.setattr(
        pipeline.planner,
        "analyze_question_active",
        lambda ctx: setattr(ctx, "question_analysis", expected) or setattr(ctx, "question_analysis_source", "llm_active") or ctx,
    )
    monkeypatch.setattr(
        pipeline.summarizer,
        "answer_conceptual",
        lambda ctx: setattr(ctx, "summary", "Active conceptual answer") or ctx,
    )

    out = pipeline.process_query("what is genex?", trace_id="trace-qa-active", session_id="session-qa-active")

    assert out.summary == "Active conceptual answer"
    assert out.question_analysis_source == "llm_active"
    assert out.stage_timings_ms["stage_0_2_question_analyzer"] >= 0.0


def test_pipeline_reject_path_short_circuits_to_clarify_without_plan_generation(monkeypatch):
    payload = {
        "version": "question_analysis_v1",
        "raw_query": (
            "Calculate a custom weighted average balancing price for specific entity groups and periods, "
            "conditional on balancing composition, given that prices for PPA and import are not known."
        ),
        "canonical_query_en": (
            "Calculate a custom weighted average balancing price for specific entity groups and periods, "
            "conditional on balancing composition, given that prices for PPA and import are not known."
        ),
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "unsupported",
            "analysis_mode": "light",
            "intent": "custom_weighted_average_price",
            "needs_clarification": False,
            "confidence": 1.0,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "reject",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": False,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.9},
                {"name": "cfd_ppa", "score": 0.8},
            ],
        },
        "tooling": {
            "candidate_tools": [
                {"name": "get_balancing_composition", "score": 0.9},
                {"name": "get_prices", "score": 0.8},
            ],
        },
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }
    expected = QuestionAnalysis.model_validate(payload)

    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_SHADOW", False)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_HINTS", True)
    monkeypatch.setattr(pipeline.planner, "prepare_context", lambda ctx: setattr(ctx, "is_conceptual", False) or ctx)
    monkeypatch.setattr(
        pipeline.planner,
        "analyze_question_active",
        lambda ctx: setattr(ctx, "question_analysis", expected) or setattr(ctx, "question_analysis_source", "llm_active") or ctx,
    )
    monkeypatch.setattr(
        pipeline.planner,
        "generate_plan",
        lambda _ctx: (_ for _ in ()).throw(AssertionError("generate_plan should not run for preferred_path=reject")),
    )
    monkeypatch.setattr(
        pipeline.summarizer,
        "answer_clarify",
        lambda ctx: setattr(ctx, "summary", "Clarify instead of plan") or ctx,
    )

    out = pipeline.process_query(
        "unsupported custom weighted average calculation",
        trace_id="trace-reject",
        session_id="session-reject",
    )

    assert out.summary == "Clarify instead of plan"
    assert out.resolution_policy == "clarify"
    assert out.clarify_reason == "request_not_supported_as_phrased"


# -----------------------------------------------------------------------------
# Regression: _extract_json_payload tolerates trailing content after the JSON
# object. Production trace 4e9b17da (2026-05-16, Q7 multi-tool guaranteed-source)
# failed with "Extra data: line 1 column 1407 (char 1406)" — Gemini emitted a
# valid JSON object followed by extra commentary, which the previous
# json.loads-then-find/rfind fallback couldn't recover. The new raw_decode
# strategy extracts the first complete JSON object and discards trailing data.
# -----------------------------------------------------------------------------


def test_extract_json_payload_tolerates_trailing_text():
    raw = '{"version": "question_analysis_v1", "raw_query": "x"}\n\nSome commentary the LLM should not have emitted.'
    payload = llm_core._extract_json_payload(raw)
    assert payload == {"version": "question_analysis_v1", "raw_query": "x"}


def test_extract_json_payload_tolerates_trailing_second_object():
    """If the LLM emits two JSON objects, take the first — raw_decode stops
    cleanly at the first object's closing brace."""
    raw = '{"a": 1}\n{"b": 2}'
    payload = llm_core._extract_json_payload(raw)
    assert payload == {"a": 1}


def test_extract_json_payload_strips_markdown_fences():
    raw = "```json\n{\"version\": \"question_analysis_v1\"}\n```"
    payload = llm_core._extract_json_payload(raw)
    assert payload == {"version": "question_analysis_v1"}


def test_extract_json_payload_handles_leading_prose():
    raw = 'Here is the analysis you requested:\n{"answer": "ok"}'
    payload = llm_core._extract_json_payload(raw)
    assert payload == {"answer": "ok"}


def test_extract_json_payload_rejects_non_object_root():
    with pytest.raises(ValueError):
        llm_core._extract_json_payload('["just", "an", "array"]')


def test_extract_json_payload_rejects_empty():
    with pytest.raises(ValueError):
        llm_core._extract_json_payload("")


# -----------------------------------------------------------------------------
# Regression: _sanitize_question_analysis_payload normalizes relative-date
# tokens before Pydantic validation. Production trace 7f6fc4b0 (2026-05-16,
# Q3 trend+structure query) failed because Gemini emitted
# ``sql_hints.period.start_date="12m"`` and ``end_date="now"`` rather than
# the contracted ISO ``^\d{4}-\d{2}-\d{2}$`` format. Phase 2 coerces
# recognized relative tokens to ISO; unrecognized tokens cause the period
# block to be dropped.
# -----------------------------------------------------------------------------
import datetime as _datetime_for_tests  # noqa: E402
import re as _re_for_tests  # noqa: E402

_ISO_DATE_PATTERN = _re_for_tests.compile(r"^\d{4}-\d{2}-\d{2}$")


def _date_today_iso() -> str:
    return _datetime_for_tests.date.today().isoformat()


def test_coerce_relative_date_passes_iso_through():
    out = llm_core._coerce_relative_date(
        "2024-06-15", anchor=_datetime_for_tests.date(2026, 5, 16), role="start"
    )
    assert out == "2024-06-15"


def test_coerce_relative_date_now_returns_anchor():
    anchor = _datetime_for_tests.date(2026, 5, 16)
    assert llm_core._coerce_relative_date("now", anchor=anchor, role="end") == "2026-05-16"
    assert llm_core._coerce_relative_date("TODAY", anchor=anchor, role="end") == "2026-05-16"


def test_coerce_relative_date_months_calendar_aware():
    anchor = _datetime_for_tests.date(2026, 5, 16)
    # 12 months back from 2026-05-16 is 2025-05-16 (calendar-aware, not 365 days).
    out = llm_core._coerce_relative_date("12m", anchor=anchor, role="start")
    assert out == "2025-05-16"


def test_coerce_relative_date_units():
    anchor = _datetime_for_tests.date(2026, 5, 16)
    assert llm_core._coerce_relative_date("7d", anchor=anchor, role="start") == "2026-05-09"
    assert llm_core._coerce_relative_date("2w", anchor=anchor, role="start") == "2026-05-02"
    assert llm_core._coerce_relative_date("1q", anchor=anchor, role="start") == "2026-02-16"
    assert llm_core._coerce_relative_date("3y", anchor=anchor, role="start") == "2023-05-16"


def test_coerce_relative_date_end_role_pins_to_anchor():
    """For role='end', relative tokens collapse to anchor (rolling-window semantics)."""
    anchor = _datetime_for_tests.date(2026, 5, 16)
    assert llm_core._coerce_relative_date("12m", anchor=anchor, role="end") == "2026-05-16"


def test_coerce_relative_date_unrecognized_token_returns_none():
    anchor = _datetime_for_tests.date(2026, 5, 16)
    assert llm_core._coerce_relative_date("yesterday", anchor=anchor, role="start") is None
    assert llm_core._coerce_relative_date("last quarter", anchor=anchor, role="start") is None
    assert llm_core._coerce_relative_date("", anchor=anchor, role="start") is None
    assert llm_core._coerce_relative_date(None, anchor=anchor, role="start") is None
    assert llm_core._coerce_relative_date(42, anchor=anchor, role="start") is None


def test_sanitize_coerces_period_with_12m_now_drift():
    """Q3 trace 7f6fc4b0: LLM emitted start='12m', end='now' — must coerce."""
    payload = {
        "sql_hints": {
            "period": {
                "kind": "range",
                "start_date": "12m",
                "end_date": "now",
                "granularity": "month",
            }
        }
    }
    sanitized = llm_core._sanitize_question_analysis_payload(payload)
    period = sanitized["sql_hints"].get("period")
    assert period is not None, "period should be preserved after coercion"
    assert _ISO_DATE_PATTERN.match(period["start_date"]), period["start_date"]
    assert _ISO_DATE_PATTERN.match(period["end_date"]), period["end_date"]
    assert period["end_date"] == _date_today_iso()


def test_sanitize_drops_period_when_token_unrecognized():
    payload = {
        "sql_hints": {
            "period": {
                "kind": "range",
                "start_date": "yesterday",
                "end_date": "now",
                "granularity": "month",
            }
        }
    }
    sanitized = llm_core._sanitize_question_analysis_payload(payload)
    assert sanitized["sql_hints"].get("period") is None, "unrecognized start should drop period"


def test_sanitize_coerces_params_hint_dates():
    """params_hint.start_date/end_date also have ISODate contract — coerce them too."""
    payload = {
        "tooling": {
            "candidate_tools": [
                {
                    "name": "get_prices",
                    "score": 0.8,
                    "params_hint": {
                        "metric": "p_bal_gel",
                        "start_date": "6m",
                        "end_date": "now",
                    },
                }
            ]
        }
    }
    sanitized = llm_core._sanitize_question_analysis_payload(payload)
    hint = sanitized["tooling"]["candidate_tools"][0]["params_hint"]
    assert _ISO_DATE_PATTERN.match(hint["start_date"]), hint["start_date"]
    assert _ISO_DATE_PATTERN.match(hint["end_date"]), hint["end_date"]
    assert hint["end_date"] == _date_today_iso()


def test_sanitize_drops_unrecognized_params_hint_date():
    """params_hint dates are Optional — null them rather than drop the whole hint."""
    payload = {
        "tooling": {
            "candidate_tools": [
                {
                    "name": "get_prices",
                    "score": 0.8,
                    "params_hint": {
                        "metric": "p_bal_gel",
                        "start_date": "long ago",
                        "end_date": "now",
                    },
                }
            ]
        }
    }
    sanitized = llm_core._sanitize_question_analysis_payload(payload)
    hint = sanitized["tooling"]["candidate_tools"][0]["params_hint"]
    assert hint.get("start_date") is None
    assert hint.get("end_date") == _date_today_iso()
    # Other params_hint fields preserved.
    assert hint["metric"] == "p_bal_gel"


def test_q3_payload_end_to_end_validates():
    """Build a minimal Q3-shaped payload with relative-date drift and verify
    the full sanitize → QuestionAnalysis.model_validate path succeeds."""
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "Show the trend of balancing electricity prices over the last 12 months",
        "canonical_query_en": "Trend of balancing electricity prices over the last 12 months.",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "data_explanation",
            "analysis_mode": "analyst",
            "intent": "balancing_price_trend",
            "needs_clarification": False,
            "confidence": 0.92,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "sql",
            "needs_sql": True,
            "needs_knowledge": True,
            "prefer_tool": True,
        },
        "knowledge": {"candidate_topics": [{"name": "balancing_price", "score": 0.9}]},
        "tooling": {
            "candidate_tools": [
                {
                    "name": "get_prices",
                    "score": 0.9,
                    "params_hint": {
                        "metric": "p_bal_gel",
                        "start_date": "12m",
                        "end_date": "now",
                    },
                }
            ]
        },
        "sql_hints": {
            "metric": "p_bal_gel",
            "entities": [],
            "aggregation": "monthly",
            "dimensions": ["price"],
            "period": {
                "kind": "range",
                "start_date": "12m",
                "end_date": "now",
                "granularity": "month",
            },
        },
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": True,
            "chart_confidence": 0.9,
        },
    }
    sanitized = llm_core._sanitize_question_analysis_payload(payload)
    model = QuestionAnalysis.model_validate(sanitized)
    assert model.sql_hints.period is not None
    assert _ISO_DATE_PATTERN.match(model.sql_hints.period.start_date)
    assert model.sql_hints.period.end_date == _date_today_iso()
    assert model.tooling.candidate_tools[0].params_hint.end_date == _date_today_iso()


# -----------------------------------------------------------------------------
# Regression: _sanitize_question_analysis_payload drops unknown topic names
# from knowledge.candidate_topics. Production trace b19e2464 (2026-05-16, Q6
# minimum-MW query) failed because Gemini emitted
# candidate_topics[1].name = "regulatory_procedure" which is not in the
# ``KnowledgeTopicName`` enum. Without filtering, the entire QuestionAnalysis
# validation crashed and the pipeline fell back to heuristic routing.
# -----------------------------------------------------------------------------


def test_sanitize_drops_unknown_topic_name_from_direct_path():
    """The common case: LLM emits candidate_topics directly on knowledge."""
    payload = {
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.9},
                {"name": "regulatory_procedure", "score": 0.7},  # unknown
                {"name": "market_structure", "score": 0.6},
            ]
        }
    }
    sanitized = llm_core._sanitize_question_analysis_payload(payload)
    names = [t["name"] for t in sanitized["knowledge"]["candidate_topics"]]
    assert names == ["balancing_price", "market_structure"]


def test_sanitize_drops_unknown_topic_name_from_top_level_merge_path():
    """Edge case: LLM emits candidate_topics at the top level instead of nested."""
    payload = {
        "knowledge": {"candidate_topics": []},
        "candidate_topics": [
            {"name": "balancing_price", "score": 0.9},
            {"name": "unknown_topic_xyz", "score": 0.5},
        ],
    }
    sanitized = llm_core._sanitize_question_analysis_payload(payload)
    names = [t["name"] for t in sanitized["knowledge"]["candidate_topics"]]
    assert names == ["balancing_price"]


def test_sanitize_keeps_all_known_topics():
    """Sanity check: when all topic names are valid, nothing is dropped."""
    payload = {
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.9},
                {"name": "tariffs", "score": 0.8},
                {"name": "market_structure", "score": 0.7},
            ]
        }
    }
    sanitized = llm_core._sanitize_question_analysis_payload(payload)
    names = [t["name"] for t in sanitized["knowledge"]["candidate_topics"]]
    assert names == ["balancing_price", "tariffs", "market_structure"]


def test_sanitize_all_unknown_topics_yields_empty_list():
    """When every emitted topic is unknown, candidate_topics becomes empty
    (still a valid QuestionAnalysis — the field defaults to empty list)."""
    payload = {
        "knowledge": {
            "candidate_topics": [
                {"name": "regulatory_procedure", "score": 0.7},
                {"name": "made_up_category", "score": 0.5},
            ]
        }
    }
    sanitized = llm_core._sanitize_question_analysis_payload(payload)
    assert sanitized["knowledge"]["candidate_topics"] == []


def test_q6_payload_end_to_end_validates():
    """Build a Q6-shaped payload with the exact 'regulatory_procedure' drift
    and verify the full sanitize → QuestionAnalysis.model_validate path
    succeeds. Previously this exact shape crashed validation."""
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "What is the minimum installed capacity for balancing market?",
        "canonical_query_en": "What is the minimum installed capacity required for participation in the balancing market?",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "factual_lookup",
            "analysis_mode": "light",
            "intent": "minimum_capacity_threshold",
            "needs_clarification": False,
            "confidence": 0.85,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
        },
        "knowledge": {
            "candidate_topics": [
                {"name": "market_structure", "score": 0.9},
                {"name": "regulatory_procedure", "score": 0.8},  # drift token
            ]
        },
        "tooling": {"candidate_tools": []},
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
        },
    }
    sanitized = llm_core._sanitize_question_analysis_payload(payload)
    model = QuestionAnalysis.model_validate(sanitized)
    # The valid topic is preserved; the drift token is dropped.
    topic_names = [t.name.value for t in model.knowledge.candidate_topics]
    assert topic_names == ["market_structure"]
    # Routing and classification fields survived unchanged.
    assert model.routing.preferred_path.value == "knowledge"
    assert model.classification.query_type.value == "factual_lookup"
