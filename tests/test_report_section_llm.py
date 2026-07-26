"""Section writer/repair provider prompt and cache tests."""

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
from contracts.report_sections import ReportSectionDraft
from tests.test_report_planner import TABLE_REF, _manifest, _plan_payload
from tests.test_report_sections import _draft


def test_section_writer_receives_only_its_evidence_slice_and_skill_rules(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]
    captured = {}

    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda key: (None, "token"))
    monkeypatch.setattr(
        llm,
        "_cache_set",
        lambda key, value, token: captured.update(
            cache_key=key,
            cached_value=value,
            token=token,
        ),
    )
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, messages, **kwargs):
        captured["messages"] = messages
        captured["label"] = kwargs["label"]
        return SimpleNamespace(content=json.dumps(_draft(section)))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    draft = llm.llm_write_report_section(
        "Explain the price trend.",
        plan,
        section,
        _manifest(),
    )

    assert isinstance(draft, ReportSectionDraft)
    assert captured["label"] == "Report section writer"
    system, user = captured["messages"]
    assert "untrusted evidence data" in system[1].lower()
    assert "# Section Writing Rules" in user[1]
    assert section.required_evidence_refs[0] in user[1]
    assert TABLE_REF not in user[1]
    assert "Average price was 125 GEL/MWh." in user[1]
    assert section.section_id in captured["cache_key"]
    assert _manifest().manifest_id in captured["cache_key"]


def test_section_repair_gets_typed_errors_and_cannot_change_scope(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]
    candidate = ReportSectionDraft.model_validate(
        {
            **_draft(section),
            "paragraphs": [
                {
                    "text": "This candidate section is much too short.",
                    "evidence_refs": section.required_evidence_refs,
                }
            ],
        }
    )
    captured = {}

    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda key: (None, "repair-token"))
    monkeypatch.setattr(llm, "_cache_set", lambda *_: None)
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, messages, **kwargs):
        captured["messages"] = messages
        captured["label"] = kwargs["label"]
        return SimpleNamespace(content=json.dumps(_draft(section)))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    repaired = llm.llm_repair_report_section(
        "Explain the price trend.",
        plan,
        section,
        _manifest(),
        candidate,
        ["WORD_COUNT_OUT_OF_RANGE"],
    )

    assert repaired.section_id == section.section_id
    assert captured["label"] == "Report section repair"
    assert "WORD_COUNT_OUT_OF_RANGE" in captured["messages"][1][1]
    assert candidate.paragraphs[0].text in captured["messages"][1][1]
