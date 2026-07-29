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


def test_report_research_planner_releases_cache_after_provider_failure(
    monkeypatch,
):
    cancelled = []
    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "cache-token"),
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

    assert len(cancelled) == 1
    assert cancelled[0][1] == "cache-token"


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
    )

    system, user = captured["messages"]
    assert "report mode is already selected" in system[1].lower()
    assert "do not classify" in system[1].lower()
    assert "prices" in user[1]
    assert "generation_mix" in user[1]
    assert "vector_knowledge" in user[1]
    assert "MAX_RESEARCH_TRACKS:\n4" in user[1]
    assert "MAX_TOTAL_EXHIBITS:\n4" in user[1]
    assert "REQUIRED_QUERY_DIGEST" not in user[1]
    assert "REQUIRED_LANGUAGE_CODE" not in user[1]
    assert "same language as USER_REPORT_REQUEST" in system[1]
    assert "period_start and period_end must both be null" in system[1]
    assert "every request topic must be covered" in system[1]
    assert "knowledge mode requires vector_knowledge" in system[1]
    assert "mixed mode requires both" in system[1]
    assert "all lists within a track must contain unique values" in system[1]
    assert len(user[1]) < 30_000
