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

from contracts.question_analysis import QuestionAnalysis  # noqa: E402
from core import llm as llm_core  # noqa: E402
from agent import planner, pipeline  # noqa: E402
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


def test_active_planner_does_not_coerce_multi_month_comparison_query(monkeypatch):
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
    assert out.question_analysis.classification.query_type.value == "unsupported"
    assert out.question_analysis.routing.preferred_path.value == "knowledge"


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
