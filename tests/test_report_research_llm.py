"""Provider and prompt boundary tests for report research planning."""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import core.llm as llm
from agent.report_research_planner import build_report_planning_constraints
from contracts.report_research import ReportResearchPlan, ReportResearchPlanDraft
from tests.test_report_research_contract import _research_plan_payload

_QUERY = (
    "Assess current market model and prices and explain the implications for "
    "energy security."
)


def _response_payload() -> dict:
    return _research_plan_payload(
        query_digest=hashlib.sha256(_QUERY.encode("utf-8")).hexdigest()
    )


def test_report_research_planner_uses_report_model_without_fallback(monkeypatch):
    captured = {}
    report_client = object()
    monkeypatch.setattr(llm, "REPORT_MODEL_TYPE", "openai", raising=False)
    monkeypatch.setattr(llm, "REPORT_MODEL", "gpt-5.6-luna", raising=False)
    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "cache-token"),
    )
    monkeypatch.setattr(llm, "_cache_set", lambda *_args: None)

    def get_stage(*_args, **kwargs):
        captured["stage_kwargs"] = kwargs
        return report_client

    monkeypatch.setattr(llm, "get_llm_for_stage", get_stage)

    def invoke(factory, model_name, messages, **kwargs):
        captured["client"] = factory()
        captured["model_name"] = model_name
        captured["messages"] = messages
        captured["invoke_kwargs"] = kwargs
        return SimpleNamespace(content=json.dumps(_response_payload()))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    plan = llm.llm_plan_report_research(
        _QUERY,
        language_code="en",
        max_tracks=4,
    )

    assert isinstance(plan, ReportResearchPlan)
    assert captured["client"] is report_client
    assert captured["model_name"] == "gpt-5.6-luna"
    assert captured["stage_kwargs"]["report_profile"] is True
    assert captured["invoke_kwargs"]["allow_openai_fallback"] is False
    assert (
        captured["invoke_kwargs"]["attempt_stage"]
        == "report_research_planner"
    )


@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
def test_openai_report_research_planner_uses_configured_structured_output(
    monkeypatch,
    method,
):
    captured = {}
    structured_client = object()

    class _ReportClient:
        def with_structured_output(self, schema, **kwargs):
            captured["structured_schema"] = schema
            captured["structured_kwargs"] = kwargs
            return structured_client

    monkeypatch.setattr(llm, "REPORT_MODEL_TYPE", "openai", raising=False)
    monkeypatch.setattr(llm, "REPORT_MODEL", "gpt-5.6-luna", raising=False)
    monkeypatch.setattr(
        llm,
        "REPORT_STRUCTURED_OUTPUT_METHOD",
        method,
        raising=False,
    )
    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "cache-token"),
    )
    monkeypatch.setattr(llm, "_cache_set", lambda *_args: None)
    monkeypatch.setattr(
        llm,
        "get_llm_for_stage",
        lambda *_args, **_kwargs: _ReportClient(),
    )

    def invoke(factory, _model, _messages, **_kwargs):
        captured["client"] = factory()
        parsed = _response_payload()
        for field in ("contract_version", "query_digest", "language_code"):
            parsed.pop(field)
        return {
            "raw": SimpleNamespace(
                content=json.dumps(parsed)
            ),
            "parsed": parsed,
            "parsing_error": None,
        }

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    plan = llm.llm_plan_report_research(
        _QUERY,
        language_code="en",
        max_tracks=4,
    )

    assert isinstance(plan, ReportResearchPlan)
    assert captured["client"] is structured_client
    assert isinstance(captured["structured_schema"], dict)
    assert (
        captured["structured_schema"]
        == ReportResearchPlanDraft.model_json_schema()
    )
    assert captured["structured_kwargs"] == {
        "method": method,
        "include_raw": True,
        "strict": True,
    }


def test_report_research_planner_can_use_portable_prompt_output(monkeypatch):
    report_client = object()
    monkeypatch.setattr(llm, "REPORT_MODEL_TYPE", "openai", raising=False)
    monkeypatch.setattr(llm, "REPORT_MODEL", "future-model", raising=False)
    monkeypatch.setattr(
        llm,
        "REPORT_STRUCTURED_OUTPUT_METHOD",
        "prompt",
        raising=False,
    )
    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "cache-token"),
    )
    monkeypatch.setattr(llm, "_cache_set", lambda *_args: None)
    monkeypatch.setattr(
        llm,
        "get_llm_for_stage",
        lambda *_args, **_kwargs: report_client,
    )

    def invoke(factory, _model, _messages, **_kwargs):
        assert factory() is report_client
        return SimpleNamespace(content=json.dumps(_response_payload()))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    assert isinstance(
        llm.llm_plan_report_research(
            _QUERY,
            language_code="en",
            max_tracks=4,
        ),
        ReportResearchPlan,
    )


def test_report_research_planner_does_not_touch_cache_after_provider_failure(
    monkeypatch,
):
    acquired = []
    cancelled = []
    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda key: acquired.append(key),
    )
    monkeypatch.setattr(
        llm,
        "_cache_cancel_in_flight",
        lambda key, token: cancelled.append((key, token)),
    )
    monkeypatch.setattr(
        llm,
        "_invoke_with_openai_fallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("provider failed")
        ),
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        llm.llm_plan_report_research(
            _QUERY,
            language_code="en",
            max_tracks=4,
        )

    assert acquired == []
    assert cancelled == []


def test_report_research_prompt_is_bounded_and_does_not_reclassify_mode(
    monkeypatch,
):
    captured = {}
    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "cache-token"),
    )
    monkeypatch.setattr(llm, "_cache_set", lambda *_args: None)
    monkeypatch.setattr(
        llm,
        "get_llm_for_stage",
        lambda *_args, **_kwargs: object(),
    )

    def invoke(_factory, _model, messages, **_kwargs):
        captured["messages"] = messages
        return SimpleNamespace(content=json.dumps(_response_payload()))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    llm.llm_plan_report_research(
        _QUERY,
        language_code="en",
        max_tracks=4,
        planning_constraints=build_report_planning_constraints(_QUERY),
    )

    system, user = captured["messages"]
    assert "report mode is already selected" in system[1].lower()
    assert "do not classify" in system[1].lower()
    assert "prices" in user[1]
    assert "generation_mix" in user[1]
    assert "vector_knowledge" in user[1]
    assert "MAX_RESEARCH_TRACKS:\n4" in user[1]
    assert "MAX_TOTAL_EXHIBITS:\n4" in user[1]
    assert "REQUIRED_EXHIBITS:" in user[1]
    assert '"purpose":"trend"' in user[1]
    assert '"purpose":"composition"' in user[1]
    assert "must include every required exhibit" in system[1].lower()
    assert "REQUIRED_QUERY_DIGEST" not in user[1]
    assert "REQUIRED_LANGUAGE_CODE" not in user[1]
    assert "same language as USER_REPORT_REQUEST" in system[1]
    assert "period_start and period_end must both be null" in system[1]
    assert "every request topic must be covered" in system[1]
    assert "knowledge mode requires vector_knowledge" in system[1]
    assert "mixed mode requires both" in system[1]
    assert "all lists within a track must contain unique values" in system[1]
    assert len(user[1]) < 30_000


def test_report_research_planner_does_not_use_the_generic_response_cache(
    monkeypatch,
):
    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (_ for _ in ()).throw(
            AssertionError("research plans must not use the generic cache")
        ),
    )
    monkeypatch.setattr(
        llm,
        "get_llm_for_stage",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        llm,
        "_invoke_with_openai_fallback",
        lambda *_args, **_kwargs: SimpleNamespace(
            content=json.dumps(_response_payload())
        ),
    )

    plan = llm.llm_plan_report_research(
        _QUERY,
        language_code="en",
        max_tracks=4,
        planning_constraints=build_report_planning_constraints(_QUERY),
    )

    assert isinstance(plan, ReportResearchPlan)


def _capture_planner_messages(monkeypatch, captured, **kwargs):
    """Drive llm_plan_report_research far enough to capture its messages."""
    monkeypatch.setattr(llm, "REPORT_MODEL_TYPE", "openai", raising=False)
    monkeypatch.setattr(llm, "REPORT_MODEL", "gpt-5.6-luna", raising=False)
    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "cache-token"),
    )
    monkeypatch.setattr(llm, "_cache_set", lambda *_args: None)
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *_a, **_k: object())

    def invoke(_factory, _model_name, messages, **_kwargs):
        captured["messages"] = messages
        return SimpleNamespace(content=json.dumps(_response_payload()))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)
    return llm.llm_plan_report_research(
        _QUERY,
        language_code="en",
        max_tracks=4,
        **kwargs,
    )


def test_research_planner_system_prompt_carries_a_coverage_rubric(monkeypatch):
    """The planner must be told what a report has to cover, not just its budget.

    Job ff5d1142 planned "balancing prices in Georgia" as price levels plus a
    market-design explainer: on topic, but 9 numeric observations and no
    composition, currency, or tariff context. The prompt constrained track
    count and legal collectors and said nothing about coverage.
    """
    captured = {}
    _capture_planner_messages(monkeypatch, captured)

    system = dict(captured["messages"])["system"]
    lowered = system.lower()
    # Domain-agnostic obligations, so a new topic needs no code change.
    assert "drives" in lowered or "driver" in lowered
    assert "composition" in lowered or "composes" in lowered
    assert "limitation" in lowered or "gap" in lowered


def test_research_planner_rubric_precedes_the_variable_request(monkeypatch):
    """Stable guidance belongs in the cached prefix, ahead of the request.

    OpenAI caches by prefix, so a rubric in the system message rides the cache
    while anything query-shaped must follow it.
    """
    captured = {}
    _capture_planner_messages(monkeypatch, captured)

    roles = [role for role, _ in captured["messages"]]
    assert roles.index("system") < roles.index("user")
    user_prompt = dict(captured["messages"])["user"]
    assert user_prompt.index("COLLECTOR_CATALOG") < user_prompt.index(
        "USER_REPORT_REQUEST"
    )


def test_research_planner_prompt_carries_injected_topic_knowledge(monkeypatch):
    captured = {}
    _capture_planner_messages(
        monkeypatch,
        captured,
        topic_knowledge="Balancing price is driven by composition and FX.",
    )

    user_prompt = dict(captured["messages"])["user"]
    assert "TOPIC_KNOWLEDGE" in user_prompt
    assert "driven by composition and FX" in user_prompt
    # Variable content must not land ahead of the stable catalog.
    assert user_prompt.index("COLLECTOR_CATALOG") < user_prompt.index(
        "TOPIC_KNOWLEDGE"
    )


def test_research_planner_omits_the_knowledge_block_when_absent(monkeypatch):
    captured = {}
    _capture_planner_messages(monkeypatch, captured)

    assert "TOPIC_KNOWLEDGE" not in dict(captured["messages"])["user"]


_REPORT_PROMPT_BUILDERS = (
    "llm_plan_report_research",
    "llm_plan_report",
    "llm_repair_report_plan",
    "llm_write_report_document",
    "_llm_write_report_section_batch",
    "llm_repair_report_document_sections",
    "llm_write_report_section",
)


@pytest.mark.parametrize("builder_name", _REPORT_PROMPT_BUILDERS)
def test_report_prompts_put_the_schema_before_the_variable_request(builder_name):
    """Constants must precede per-request content, or nothing caches.

    Providers cache on longest common prefix. Every report prompt used to put
    OUTPUT_JSON_SCHEMA — its largest constant — last, behind the user request
    and the evidence, so the cacheable prefix ended a few lines in and every
    report ran at cached_prompt_tokens=0.
    """
    import inspect

    source = inspect.getsource(getattr(llm, builder_name))
    schema_at = source.find('"OUTPUT_JSON_SCHEMA:')
    request_at = source.find('"USER_REPORT_REQUEST:')

    assert schema_at != -1, f"{builder_name} has no schema block"
    assert request_at != -1, f"{builder_name} has no request block"
    assert schema_at < request_at, (
        f"{builder_name} emits the constant schema after the variable request, "
        "which poisons the cacheable prefix"
    )
