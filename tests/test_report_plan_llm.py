"""Report planning provider prompt, cache, and schema-boundary tests."""

from __future__ import annotations

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
from contracts.report import ReportPlan
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
