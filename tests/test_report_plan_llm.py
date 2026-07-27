"""Report planning provider prompt, cache, and schema-boundary tests."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from types import SimpleNamespace

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import core.llm as llm
from contracts.report import (
    ReportIntent,
    ReportPlan,
    ReportPlanningContext,
    ReportSectionKind,
)
from tests.test_report_planner import _manifest, _plan_payload


def test_report_planner_prompt_uses_structure_skill_and_evidence_catalog(monkeypatch):
    captured = {}

    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda key: (None, "token"))
    monkeypatch.setattr(
        llm,
        "_cache_set",
        lambda key, value, token: captured.update(
            cache_key=key,
            cached_value=value,
            cache_token=token,
        ),
    )
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, messages, **kwargs):
        captured["messages"] = messages
        captured["label"] = kwargs["label"]
        return SimpleNamespace(content=json.dumps(_plan_payload()))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    plan = llm.llm_plan_report("Explain the price trend.", _manifest())

    assert isinstance(plan, ReportPlan)
    assert captured["label"] == "Report planner"
    system, user = captured["messages"]
    assert "untrusted evidence data" in system[1].lower()
    assert "# Standard Report Structure" in user[1]
    assert "# Report Planning Contract" in user[1]
    assert _manifest().manifest_id in user[1]
    assert "evidence:table:" in user[1]
    assert '"price":120.0' not in user[1]
    assert _manifest().manifest_id in captured["cache_key"]
    assert captured["cache_token"] == "token"


def test_report_planner_cache_hit_still_validates_the_strict_plan(monkeypatch):
    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (json.dumps(_plan_payload()), None),
    )
    monkeypatch.setattr(
        llm,
        "_invoke_with_openai_fallback",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("provider must not run on a cache hit")
        ),
    )

    plan = llm.llm_plan_report("Explain the price trend.", _manifest())

    assert plan.evidence_manifest_id == _manifest().manifest_id


def test_report_planner_normalizes_section_word_allocation_before_validation(
    monkeypatch,
):
    payload = deepcopy(_plan_payload())
    payload["sections"][-1]["target_words"] -= 7
    captured = {}

    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda _key: (None, "token"))
    monkeypatch.setattr(
        llm,
        "_cache_set",
        lambda _key, value, _token: captured.update(value=value),
    )
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())
    monkeypatch.setattr(
        llm,
        "_invoke_with_openai_fallback",
        lambda *_args, **_kwargs: SimpleNamespace(content=json.dumps(payload)),
    )

    plan = llm.llm_plan_report("Explain the price trend.", _manifest())

    assert sum(section.target_words for section in plan.sections) == 900
    assert plan.target_words == 900
    assert ReportPlan.model_validate_json(captured["value"]) == plan


def test_report_planner_binds_intent_language_and_core_structure_from_context(
    monkeypatch,
):
    captured = {}
    planning_context = ReportPlanningContext(
        contract_version="report-planning-context-v1",
        intent="trend",
        language_code="ka",
        request_objective="Analyze the observed electricity-price trend.",
        requires_table=True,
        source="question_analysis",
    )

    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda key: (None, "token"))
    monkeypatch.setattr(llm, "_cache_set", lambda *_args: None)
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, messages, **_kwargs):
        captured["messages"] = messages
        payload = _plan_payload()
        payload["sections"][2]["kind"] = "trend_analysis"
        return SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    plan = llm.llm_plan_report(
        "მომიმზადე ფასების ტენდენციის ანგარიში.",
        _manifest(),
        planning_context=planning_context,
    )

    assert plan.intent is ReportIntent.TREND
    assert plan.language_code == "ka"
    assert plan.sections[2].kind is ReportSectionKind.TREND_ANALYSIS
    assert '"intent":"trend"' in captured["messages"][1][1]
    assert '"language_code":"ka"' in captured["messages"][1][1]
    assert "Do not reclassify the report intent" in captured["messages"][0][1]


def test_report_planner_prompt_exposes_chart_column_roles(monkeypatch):
    captured = {}

    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda key: (None, "token"))
    monkeypatch.setattr(llm, "_cache_set", lambda *_a, **_k: None)
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, messages, **_kwargs):
        captured["messages"] = messages
        return SimpleNamespace(content=json.dumps(_plan_payload()))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    llm.llm_plan_report("Explain the price trend.", _manifest())

    _system, user = captured["messages"]
    assert '"column_roles"' in user[1]
    assert '"temporal":["period"]' in user[1]
    assert '"numeric":["price"]' in user[1]
    assert '"price":120.0' not in user[1]


def test_report_plan_repair_is_uncached_and_bounds_its_error_codes(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (_ for _ in ()).throw(
            AssertionError("plan repair must not read the response cache")
        ),
    )
    monkeypatch.setattr(
        llm,
        "_cache_set",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("plan repair must not write the response cache")
        ),
    )
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, messages, **kwargs):
        captured["messages"] = messages
        captured["attempt_stage"] = kwargs["attempt_stage"]
        return SimpleNamespace(content=json.dumps(_plan_payload()))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    plan = llm.llm_repair_report_plan(
        "Explain the price trend.",
        _manifest(),
        ReportPlanningContext(
            contract_version="report-planning-context-v1",
            intent=ReportIntent.GENERAL,
            language_code="en",
            request_objective="Explain the price trend.",
            requires_table=True,
            source="question_analysis",
        ),
        _plan_payload(),
        ["PLAN_SCHEMA_INVALID", "drop me", "ALSO_FINE"],
    )

    assert isinstance(plan, ReportPlan)
    assert captured["attempt_stage"] == "report_plan_repair"
    _system, user = captured["messages"]
    assert '["PLAN_SCHEMA_INVALID","ALSO_FINE"]' in user[1]
    assert "drop me" not in user[1]
    assert "untrusted" in _system[1].lower()
