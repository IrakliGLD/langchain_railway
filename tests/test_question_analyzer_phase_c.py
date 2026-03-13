"""Tests for active question-analyzer hint integration in Phase C."""

import json
import os

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

from agent import planner, summarizer  # noqa: E402
from contracts.question_analysis import QuestionAnalysis  # noqa: E402
from core import llm as llm_core  # noqa: E402
from core.llm import SummaryEnvelope  # noqa: E402
from models import QueryContext  # noqa: E402


class _DummyCache:
    def __init__(self):
        self._store = {}

    def get(self, prompt: str):
        return self._store.get(prompt)

    def set(self, prompt: str, response: str):
        self._store[prompt] = response


class _DummyMessage:
    def __init__(self, content: str):
        self.content = content
        self.usage_metadata = {}
        self.response_metadata = {}


def _analytical_payload() -> QuestionAnalysis:
    return QuestionAnalysis.model_validate(
        {
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
        }
    )


def _conceptual_payload() -> QuestionAnalysis:
    return QuestionAnalysis.model_validate(
        {
            "version": "question_analysis_v1",
            "raw_query": "wat is genx?",
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
        }
    )


def test_generate_plan_passes_only_active_question_analysis(monkeypatch):
    captured = {}
    qa = _analytical_payload()

    def _fake_generate(*, user_query, analysis_mode, lang_instruction, question_analysis=None):
        captured["user_query"] = user_query
        captured["analysis_mode"] = analysis_mode
        captured["lang_instruction"] = lang_instruction
        captured["question_analysis"] = question_analysis
        return {"intent": "general", "target": "", "period": ""}, "SELECT 1"

    monkeypatch.setattr(planner, "_generate_plan_and_sql_with_retry", _fake_generate)
    monkeypatch.setattr(planner, "should_skip_sql_execution", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(planner, "detect_aggregation_intent", lambda *_args, **_kwargs: {"aggregation_type": "none"})

    active_ctx = QueryContext(
        query="why does balancing electricity price changed in november 2021?",
        mode="analyst",
        lang_instruction="Respond in English.",
        question_analysis=qa,
        question_analysis_source="llm_active",
    )
    out = planner.generate_plan(active_ctx)
    assert captured["question_analysis"] is qa
    assert out.raw_sql == "SELECT 1"

    shadow_ctx = QueryContext(
        query="why does balancing electricity price changed in november 2021?",
        mode="analyst",
        lang_instruction="Respond in English.",
        question_analysis=qa,
        question_analysis_source="llm_shadow",
    )
    planner.generate_plan(shadow_ctx)
    assert captured["question_analysis"] is None


def test_answer_conceptual_uses_active_analyzer_topics_and_canonical_query(monkeypatch):
    captured = {}
    qa = _conceptual_payload()

    def _fake_get_relevant_domain_knowledge(user_query, use_cache=True, preferred_topics=None):
        captured["user_query"] = user_query
        captured["preferred_topics"] = preferred_topics
        return '{"market_structure": "### GENEX\\nGeorgian Energy Exchange."}'

    monkeypatch.setattr(summarizer, "get_relevant_domain_knowledge", _fake_get_relevant_domain_knowledge)
    monkeypatch.setattr(
        summarizer,
        "llm_summarize_structured",
        lambda *_args, **kwargs: SummaryEnvelope(
            answer="GENEX is the Georgian Energy Exchange.",
            claims=["GENEX is the Georgian Energy Exchange."],
            citations=["domain_knowledge"],
            confidence=0.9,
        ),
    )

    active_ctx = QueryContext(
        query="wat is genx?",
        lang_instruction="Respond in English.",
        question_analysis=qa,
        question_analysis_source="llm_active",
    )
    summarizer.answer_conceptual(active_ctx)
    assert captured["user_query"] == "What is GENEX?"
    assert captured["preferred_topics"] == ["market_structure"]

    shadow_ctx = QueryContext(
        query="wat is genx?",
        lang_instruction="Respond in English.",
        question_analysis=qa,
        question_analysis_source="llm_shadow",
    )
    summarizer.answer_conceptual(shadow_ctx)
    assert captured["user_query"] == "wat is genx?"
    assert captured["preferred_topics"] is None


def test_llm_generate_plan_and_sql_uses_canonical_query_for_retrieval(monkeypatch):
    qa = _analytical_payload()
    captured = {}

    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    captured["knowledge_calls"] = []
    captured["example_calls"] = []

    def _capture_knowledge(user_query, use_cache=True, preferred_topics=None):
        captured["knowledge_calls"].append((user_query, use_cache, preferred_topics))
        return "{}"

    def _capture_examples(user_query, max_categories=2):
        captured["example_calls"].append((user_query, max_categories))
        return ""

    monkeypatch.setattr(llm_core, "get_relevant_domain_knowledge", _capture_knowledge)
    monkeypatch.setattr(llm_core, "get_relevant_examples", _capture_examples)
    monkeypatch.setattr(llm_core, "make_openai", lambda: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    prompts = []

    def _capture_invoke(_llm, messages, _model_name):
        prompts.append(messages[1][1])
        return _DummyMessage('{"intent":"general","target":"","period":""}---SQL---SELECT 1')

    monkeypatch.setattr(llm_core, "_invoke_with_resilience", _capture_invoke)

    out = llm_core.llm_generate_plan_and_sql(
        user_query="why does balancing electricity price changed in november 2021?",
        analysis_mode="analyst",
        lang_instruction="Respond in English.",
        question_analysis=qa,
    )

    assert out.endswith("SELECT 1")
    assert captured["knowledge_calls"][0][0] == "Why did balancing electricity price change in November 2021?"
    assert captured["knowledge_calls"][0][2] == ["balancing_price", "currency_influence"]
    assert captured["example_calls"][0][0] == "Why did balancing electricity price change in November 2021?"
    assert "QUESTION_ANALYZER_HINTS" in prompts[0]
    assert "balancing_price_why" in prompts[0]
