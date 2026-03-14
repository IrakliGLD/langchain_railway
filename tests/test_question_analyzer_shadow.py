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
