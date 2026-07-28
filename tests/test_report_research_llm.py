"""Provider and prompt boundary tests for report research planning."""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import core.llm as llm
from contracts.report_research import ReportResearchPlan
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
    assert len(user[1]) < 30_000
