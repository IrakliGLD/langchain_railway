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
            "analysis_requirements": {
                "needs_driver_analysis": True,
                "needs_trend_context": False,
                "needs_correlation_context": True,
                "derived_metrics": [
                    {"metric_name": "mom_absolute_change", "metric": "p_bal_gel", "target_metric": None, "rank_limit": None},
                    {"metric_name": "mom_percent_change", "metric": "p_bal_gel", "target_metric": None, "rank_limit": None},
                    {"metric_name": "mom_absolute_change", "metric": "xrate", "target_metric": None, "rank_limit": None},
                    {"metric_name": "share_delta_mom", "metric": "share_import", "target_metric": None, "rank_limit": None},
                ],
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
            "analysis_requirements": {
                "needs_driver_analysis": False,
                "needs_trend_context": False,
                "needs_correlation_context": False,
                "derived_metrics": [],
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


def test_generate_plan_preserves_authoritative_stage_0_2_semantics(monkeypatch):
    qa = _analytical_payload()

    monkeypatch.setattr(
        planner,
        "_generate_plan_and_sql_with_retry",
        lambda **_kwargs: (
            {
                "intent": "general",
                "target": "wrong_target",
                "period": "wrong_period",
                "chart_strategy": "grouped",
                "chart_groups": [{"metrics": ["p_bal_gel"]}],
            },
            "SELECT 1",
        ),
    )
    monkeypatch.setattr(planner, "should_skip_sql_execution", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(planner, "detect_aggregation_intent", lambda *_args, **_kwargs: {"aggregation_type": "none"})

    ctx = QueryContext(
        query="why does balancing electricity price changed in november 2021?",
        mode="analyst",
        lang_instruction="Respond in English.",
        question_analysis=qa,
        question_analysis_source="llm_active",
    )

    out = planner.generate_plan(ctx)

    assert out.raw_sql == "SELECT 1"
    assert out.plan["intent"] == qa.classification.intent
    assert out.plan["target"] == "p_bal_gel"
    assert out.plan["period"] == "November 2021"
    assert out.plan["chart_strategy"] == "grouped"
    assert out.plan["chart_groups"] == [{"metrics": ["p_bal_gel"]}]


def test_generate_plan_target_falls_back_to_semantic_intent_not_tool_id(monkeypatch):
    qa = QuestionAnalysis.model_validate(
        {
            "version": "question_analysis_v1",
            "raw_query": "what was import share in june 2024?",
            "canonical_query_en": "What was the import share in June 2024?",
            "language": {"input_language": "en", "answer_language": "en"},
            "classification": {
                "query_type": "data_retrieval",
                "analysis_mode": "analyst",
                "intent": "import share in balancing electricity",
                "needs_clarification": False,
                "confidence": 0.94,
                "ambiguities": [],
            },
            "routing": {
                "preferred_path": "tool",
                "needs_sql": False,
                "needs_knowledge": False,
                "prefer_tool": True,
            },
            "knowledge": {"candidate_topics": []},
            "tooling": {
                "candidate_tools": [
                    {
                        "name": "get_balancing_composition",
                        "score": 0.96,
                        "reason": "Composition share lookup",
                        "params_hint": {
                            "metric": None,
                            "currency": None,
                            "granularity": None,
                            "start_date": "2024-06-01",
                            "end_date": "2024-06-30",
                            "entities": [],
                            "types": [],
                            "mode": None,
                        },
                    }
                ]
            },
            "sql_hints": {
                "metric": None,
                "entities": [],
                "aggregation": None,
                "dimensions": ["share"],
                "period": {
                    "kind": "month",
                    "start_date": "2024-06-01",
                    "end_date": "2024-06-30",
                    "granularity": "month",
                    "raw_text": "June 2024",
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
                "needs_trend_context": False,
                "needs_correlation_context": False,
                "derived_metrics": [],
            },
        }
    )

    monkeypatch.setattr(
        planner,
        "_generate_plan_and_sql_with_retry",
        lambda **_kwargs: (
            {"intent": "general", "target": "get_balancing_composition", "period": "wrong_period"},
            "SELECT 1",
        ),
    )
    monkeypatch.setattr(planner, "should_skip_sql_execution", lambda *_args, **_kwargs: (False, ""))
    monkeypatch.setattr(planner, "detect_aggregation_intent", lambda *_args, **_kwargs: {"aggregation_type": "none"})

    ctx = QueryContext(
        query="what was import share in june 2024?",
        mode="analyst",
        lang_instruction="Respond in English.",
        question_analysis=qa,
        question_analysis_source="llm_active",
    )

    out = planner.generate_plan(ctx)

    assert out.plan["target"] == "import share in balancing electricity"
    assert out.plan["target"] != "get_balancing_composition"


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


def test_llm_analyze_question_prompt_mentions_regulatory_procedure(monkeypatch):
    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    captured = {}

    def _capture_invoke(_llm, messages, _model_name):
        captured["prompt"] = messages[1][1]
        return _DummyMessage(json.dumps(_conceptual_payload().model_dump(mode="json")))

    monkeypatch.setattr(llm_core, "_invoke_with_resilience", _capture_invoke)

    llm_core.llm_analyze_question("Who is eligible to participate in the electricity exchange?")

    prompt = captured["prompt"]
    assert "regulatory_procedure" in prompt
    assert "use `knowledge` for `conceptual_definition`, `regulatory_procedure`" in prompt
    assert "chart_intent" in prompt
    assert "target_series" in prompt
    assert "observed`, `reference`, `derived`, `component_primary`, `component_secondary`" in prompt
    assert "For `get_prices`, valid `params_hint.metric` values are only:" in prompt
    assert "`balancing`" in prompt
    assert "`exchange_rate`" in prompt
    assert "never emit raw DB column names" in prompt


def test_llm_analyze_question_sanitizes_invalid_chart_hints(monkeypatch):
    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    payload = _analytical_payload().model_dump(mode="json")
    payload["visualization"]["chart_recommended"] = True
    payload["visualization"]["chart_intent"] = "not_real"
    payload["visualization"]["target_series"] = ["observed", "wrong_role"]

    monkeypatch.setattr(
        llm_core,
        "_invoke_with_resilience",
        lambda *_args, **_kwargs: _DummyMessage(json.dumps(payload)),
    )

    result = llm_core.llm_analyze_question("compare price against a benchmark")

    assert result.visualization.chart_intent is None
    assert result.visualization.target_series == []


def test_llm_analyze_question_preserves_valid_chart_hints(monkeypatch):
    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    payload = _analytical_payload().model_dump(mode="json")
    payload["visualization"]["chart_recommended"] = True
    payload["visualization"]["chart_intent"] = "trend_compare"
    payload["visualization"]["target_series"] = ["observed", "derived"]

    monkeypatch.setattr(
        llm_core,
        "_invoke_with_resilience",
        lambda *_args, **_kwargs: _DummyMessage(json.dumps(payload)),
    )

    result = llm_core.llm_analyze_question("what if balancing price were 20 percent higher")

    assert result.visualization.chart_intent.value == "trend_compare"
    assert [role.value for role in result.visualization.target_series] == ["observed", "derived"]


def test_sanitize_chart_hints_clears_semantic_hints_when_chart_not_requested():
    payload = _analytical_payload().model_dump(mode="json")
    payload["visualization"]["chart_requested_by_user"] = False
    payload["visualization"]["chart_recommended"] = False
    payload["visualization"]["chart_intent"] = "trend_compare"
    payload["visualization"]["target_series"] = ["observed", "reference"]

    sanitized = llm_core._sanitize_chart_hints(payload)

    assert "chart_intent" not in sanitized["visualization"]
    assert "target_series" not in sanitized["visualization"]


def test_sanitize_question_analysis_payload_drops_incomplete_period():
    payload = _analytical_payload().model_dump(mode="json")
    payload["sql_hints"]["period"] = {
        "kind": "range",
        "start_date": None,
        "end_date": None,
        "granularity": "range",
        "raw_text": "all available months",
    }

    sanitized = llm_core._sanitize_question_analysis_payload(payload)

    assert "period" not in sanitized["sql_hints"]


def test_llm_analyze_question_tolerates_null_period_stub(monkeypatch):
    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    payload = _conceptual_payload().model_dump(mode="json")
    payload["raw_query"] = (
        "what are the months where the total share of renewable ppa, regulated hydro, "
        "and regulated thermals in balancing electricity is more than 99%?"
    )
    payload["canonical_query_en"] = (
        "Retrieve the months when the combined share of renewable PPA, regulated hydro, "
        "and regulated thermals in balancing electricity exceeded 99%."
    )
    payload["classification"]["query_type"] = "data_retrieval"
    payload["routing"]["preferred_path"] = "tool"
    payload["routing"]["needs_knowledge"] = False
    payload["routing"]["prefer_tool"] = True
    payload["tooling"]["candidate_tools"] = [
        {
            "name": "get_balancing_composition",
            "score": 0.98,
            "reason": "Need monthly balancing composition shares.",
            "params_hint": {
                "metric": None,
                "currency": None,
                "granularity": "monthly",
                "start_date": None,
                "end_date": None,
                "entities": [],
                "types": [],
                "mode": None,
            },
        }
    ]
    payload["sql_hints"]["period"] = {
        "kind": "range",
        "start_date": None,
        "end_date": None,
        "granularity": "range",
        "raw_text": "all available months",
    }

    monkeypatch.setattr(
        llm_core,
        "_invoke_with_resilience",
        lambda *_args, **_kwargs: _DummyMessage(json.dumps(payload)),
    )

    result = llm_core.llm_analyze_question(payload["raw_query"])

    assert result.classification.query_type.value == "data_retrieval"
    assert result.routing.preferred_path.value == "tool"
    assert result.sql_hints.period is None
    assert result.tooling.candidate_tools[0].name.value == "get_balancing_composition"


def test_llm_analyze_question_tolerates_null_dimensions_stub(monkeypatch):
    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    payload = _analytical_payload().model_dump(mode="json")
    payload["raw_query"] = "forecast balancing electricity price to 2030"
    payload["canonical_query_en"] = "Forecast balancing electricity price to 2030."
    payload["classification"]["query_type"] = "forecast"
    payload["routing"]["preferred_path"] = "sql"
    payload["routing"]["needs_knowledge"] = False
    payload["routing"]["prefer_tool"] = False
    payload["sql_hints"]["dimensions"] = None

    monkeypatch.setattr(
        llm_core,
        "_invoke_with_resilience",
        lambda *_args, **_kwargs: _DummyMessage(json.dumps(payload)),
    )

    result = llm_core.llm_analyze_question(payload["raw_query"])

    assert result.classification.query_type.value == "forecast"
    assert result.routing.preferred_path.value == "sql"
    assert result.sql_hints.dimensions == []


def test_llm_analyze_question_tolerates_null_sql_hints(monkeypatch):
    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    payload = _analytical_payload().model_dump(mode="json")
    payload["raw_query"] = "forecast balancing electricity price for 2030"
    payload["canonical_query_en"] = "Forecast balancing electricity price for 2030."
    payload["classification"]["query_type"] = "forecast"
    payload["routing"]["preferred_path"] = "tool"
    payload["routing"]["needs_knowledge"] = False
    payload["routing"]["prefer_tool"] = True
    payload["sql_hints"] = None

    monkeypatch.setattr(
        llm_core,
        "_invoke_with_resilience",
        lambda *_args, **_kwargs: _DummyMessage(json.dumps(payload)),
    )

    result = llm_core.llm_analyze_question(payload["raw_query"])

    assert result.classification.query_type.value == "forecast"
    assert result.routing.preferred_path.value == "tool"
    assert result.sql_hints.metric is None
    assert result.sql_hints.dimensions == []


# ---------------------------------------------------------------------------
# Scenario keyword tests
# ---------------------------------------------------------------------------


def test_scenario_keywords_trigger_analyst_mode():
    """Scenario/what-if keywords in detect_analysis_mode must produce 'analyst'."""
    scenario_queries = [
        "What if prices were 34% higher?",
        "Calculate CfD payoff with strike 60",
        "hypothetical scenario for balancing price",
        "if prices were 10 USD higher",
        "strike price sensitivity analysis",
        # Real production query
        "if balancing price is considered as a strike price and i have a cfd contract of 1 mw for 60 usd/mwh, what would be my income?",
    ]
    for query in scenario_queries:
        result = planner.detect_analysis_mode(query)
        assert result == "analyst", f"Expected analyst for: {query!r}, got: {result!r}"


def test_non_scenario_queries_remain_light():
    """Ensure basic factual queries are not promoted to analyst by scenario keywords."""
    light_queries = [
        "What is the current balancing price?",
        "List all entities",
        "Show me tariffs for Enguri",
        "What is CfD?",
        "What is strike price?",
    ]
    for query in light_queries:
        result = planner.detect_analysis_mode(query)
        assert result == "light", f"Expected light for: {query!r}, got: {result!r}"
