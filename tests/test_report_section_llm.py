"""Section writer/repair provider prompt and cache tests."""

from __future__ import annotations

import json
import math
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
from agent.report_sections import generate_report_sections
from contracts.report import ReportPlan
from contracts.report_sections import ReportSectionDraft
from tests.test_report_planner import TABLE_REF, _manifest, _plan_payload
from tests.test_report_sections import _draft


def test_section_writer_receives_only_its_evidence_slice_and_skill_rules(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
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
    assert TABLE_REF in user[1]
    assert "Average price was 125 GEL/MWh." not in user[1]
    assert (
        f'"minimum_words":{math.floor(section.target_words * 0.9)}'
        in user[1]
    )
    assert (
        f'"maximum_words":{math.ceil(section.target_words * 1.2)}'
        in user[1]
    )
    assert "Use every required_evidence_refs value at least once" in user[1]
    assert '"row_index_base":0' in user[1]
    assert '"row_index":0' in user[1]
    assert "derived_claims" in user[1]
    assert "code-verifiable derived claims" in system[1]
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

    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (_ for _ in ()).throw(
            AssertionError("section repairs must not use the response cache")
        ),
    )
    monkeypatch.setattr(
        llm,
        "_cache_set",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("section repairs must not populate the response cache")
        ),
    )
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
    assert (
        f'"minimum_words":{math.floor(section.target_words * 0.9)}'
        in captured["messages"][1][1]
    )
    assert (
        "Use every required_evidence_refs value at least once"
        in captured["messages"][1][1]
    )


def test_section_repair_can_recover_from_schema_invalid_candidate(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]
    captured = {}
    invalid_candidate = {
        "section_id": section.section_id,
        "unexpected_shape": "Schema-invalid candidate.",
    }

    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, messages, **_kwargs):
        captured["messages"] = messages
        return SimpleNamespace(content=json.dumps(_draft(section)))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    repaired = llm.llm_repair_report_section(
        "Explain the price trend.",
        plan,
        section,
        _manifest(),
        invalid_candidate,
        ["SECTION_SCHEMA_INVALID"],
    )

    assert repaired.section_id == section.section_id
    assert "Schema-invalid candidate." in captured["messages"][1][1]
    assert "SECTION_SCHEMA_INVALID" in captured["messages"][1][1]


def test_section_writer_cancels_cache_reservation_on_provider_failure(
    monkeypatch,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]
    cancelled = {}

    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "writer-token"),
    )
    monkeypatch.setattr(
        llm,
        "_cache_cancel_in_flight",
        lambda key, token: cancelled.update(key=key, token=token),
    )
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())
    monkeypatch.setattr(
        llm,
        "_invoke_with_openai_fallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TimeoutError("provider timed out")
        ),
    )

    with pytest.raises(TimeoutError, match="provider timed out"):
        llm.llm_write_report_section(
            "Explain the price trend.",
            plan,
            section,
            _manifest(),
        )

    assert cancelled["token"] == "writer-token"
    assert section.section_id in cancelled["key"]


def test_section_writer_does_not_cache_semantically_invalid_candidate(
    monkeypatch,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]
    cancelled = {}

    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "writer-token"),
    )
    monkeypatch.setattr(
        llm,
        "_cache_set",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("invalid report sections must not be cached")
        ),
    )
    monkeypatch.setattr(
        llm,
        "_cache_cancel_in_flight",
        lambda key, token: cancelled.update(key=key, token=token),
    )
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())
    monkeypatch.setattr(
        llm,
        "_invoke_with_openai_fallback",
        lambda *_args, **_kwargs: SimpleNamespace(
            content=json.dumps(
                {
                    **_draft(section),
                    "paragraphs": [
                        {
                            "text": "This candidate is too short.",
                            "evidence_refs": section.required_evidence_refs,
                        }
                    ],
                }
            )
        ),
    )

    draft = llm.llm_write_report_section(
        "Explain the price trend.",
        plan,
        section,
        _manifest(),
    )

    assert draft.paragraphs[0].text == "This candidate is too short."
    assert cancelled["token"] == "writer-token"


def test_section_writer_returns_schema_invalid_payload_for_local_repair(
    monkeypatch,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]
    cancelled = {}
    invalid_payload = {
        "section_id": section.section_id,
        "unexpected_shape": "Needs schema repair.",
    }

    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "writer-token"),
    )
    monkeypatch.setattr(
        llm,
        "_cache_set",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("schema-invalid report sections must not be cached")
        ),
    )
    monkeypatch.setattr(
        llm,
        "_cache_cancel_in_flight",
        lambda key, token: cancelled.update(key=key, token=token),
    )
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())
    monkeypatch.setattr(
        llm,
        "_invoke_with_openai_fallback",
        lambda *_args, **_kwargs: SimpleNamespace(
            content=json.dumps(invalid_payload)
        ),
    )

    draft = llm.llm_write_report_section(
        "Explain the price trend.",
        plan,
        section,
        _manifest(),
    )

    assert draft == invalid_payload
    assert cancelled["token"] == "writer-token"


def test_default_section_writer_and_repair_converge_after_schema_drift(
    monkeypatch,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]
    existing_drafts = {
        other.section_id: ReportSectionDraft.model_validate(_draft(other))
        for other in plan.sections[1:]
    }
    labels = []

    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "writer-token"),
    )
    monkeypatch.setattr(llm, "_cache_cancel_in_flight", lambda *_args: True)
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, _messages, **kwargs):
        labels.append(kwargs["label"])
        payload = (
            {
                "section_id": section.section_id,
                "unexpected_shape": "Needs schema repair.",
            }
            if kwargs["label"] == "Report section writer"
            else _draft(section)
        )
        return SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    drafts = generate_report_sections(
        "Explain the price trend.",
        plan,
        _manifest(),
        existing_drafts=existing_drafts,
        max_workers=1,
    )

    assert labels == ["Report section writer", "Report section repair"]
    assert drafts[0].section_id == section.section_id


def test_section_writer_bypasses_semantically_invalid_cache_entry(
    monkeypatch,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]
    invalid_cached = {
        **_draft(section),
        "paragraphs": [
            {
                "text": "Cached but too short.",
                "evidence_refs": section.required_evidence_refs,
            }
        ],
    }
    provider_calls = []
    cached = {}

    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (json.dumps(invalid_cached), None),
    )
    monkeypatch.setattr(
        llm,
        "_cache_set",
        lambda key, value, token: cached.update(
            key=key,
            value=value,
            token=token,
        ),
    )
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(*_args, **_kwargs):
        provider_calls.append(True)
        return SimpleNamespace(content=json.dumps(_draft(section)))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    draft = llm.llm_write_report_section(
        "Explain the price trend.",
        plan,
        section,
        _manifest(),
    )

    assert provider_calls == [True]
    assert draft.paragraphs[0].text != "Cached but too short."
    assert cached["token"] is None
