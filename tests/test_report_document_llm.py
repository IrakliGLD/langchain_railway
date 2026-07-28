"""Whole-document writer and targeted-repair LLM boundary tests."""

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
from agent.report_document_generation import validate_report_document
from contracts.report_document import (
    ReportDocumentDraft,
    ReportDocumentRepair,
)
from tests.test_report_document_pipeline_v2 import (
    _QUERY,
    _document_components,
    _valid_document_draft,
)


def test_document_writer_uses_one_dedicated_report_call_without_fallback(
    monkeypatch,
):
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    expected = _valid_document_draft(document_plan, manifest)
    captured = {}
    report_client = object()
    monkeypatch.setattr(llm, "REPORT_MODEL_TYPE", "openai", raising=False)
    monkeypatch.setattr(llm, "REPORT_MODEL", "gpt-5.6-luna", raising=False)
    monkeypatch.setattr(
        llm,
        "_cache_get_or_reserve",
        lambda _key: (None, "document-cache"),
    )
    monkeypatch.setattr(
        llm,
        "_cache_set",
        lambda key, value, token: captured.update(
            cache_key=key,
            cache_value=value,
            cache_token=token,
        ),
    )

    def get_stage(*_args, **kwargs):
        captured["stage_kwargs"] = kwargs
        return report_client

    monkeypatch.setattr(llm, "get_llm_for_stage", get_stage)

    def invoke(factory, model_name, messages, **kwargs):
        captured["client"] = factory()
        captured["model_name"] = model_name
        captured["messages"] = messages
        captured["invoke_kwargs"] = kwargs
        return SimpleNamespace(content=expected.model_dump_json())

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)
    monkeypatch.setattr(
        ReportDocumentDraft,
        "model_json_schema",
        lambda: (_ for _ in ()).throw(
            AssertionError("document schema must be precomputed")
        ),
    )

    result = llm.llm_write_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
    )

    assert result == expected
    assert captured["client"] is report_client
    assert captured["model_name"] == "gpt-5.6-luna"
    assert captured["stage_kwargs"]["report_profile"] is True
    assert captured["invoke_kwargs"]["allow_openai_fallback"] is False
    assert (
        captured["invoke_kwargs"]["attempt_stage"]
        == "report_document_writer"
    )
    system, user = captured["messages"]
    assert "report mode is already selected" in system[1].lower()
    assert "do not classify" in system[1].lower()
    assert "body first" in system[1].lower()
    assert "executive summary last" in system[1].lower()
    assert "NUMERIC_OBSERVATIONS" in user[1]
    assert '"row_index":0' in user[1]
    assert "prompt_projection_truncated" in user[1]
    assert len(user[1]) <= llm._REPORT_DOCUMENT_PROMPT_BUDGET_CHARS
    assert captured["cache_token"] == "document-cache"


def test_document_repair_receives_only_rejected_sections_and_no_fallback(
    monkeypatch,
):
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    payload = draft.model_dump(mode="json")
    payload["analytical_sections"][0]["paragraphs"][0][
        "direct_claims"
    ] = []
    payload["analytical_sections"][0]["paragraphs"][0]["text"] = " ".join(
        "unsupported" for _ in range(260)
    )
    rejected = ReportDocumentDraft.model_validate(payload)
    validation = validate_report_document(
        rejected,
        document_plan,
        manifest,
        research_plan,
    )
    replacement = draft.analytical_sections[0]
    expected = ReportDocumentRepair(
        contract_version="report-document-repair-v1",
        sections=[replacement],
    )
    captured = {}
    monkeypatch.setattr(
        llm,
        "get_llm_for_stage",
        lambda *_args, **_kwargs: object(),
    )

    def invoke(_factory, _model_name, messages, **kwargs):
        captured["messages"] = messages
        captured["invoke_kwargs"] = kwargs
        return SimpleNamespace(content=expected.model_dump_json())

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    result = llm.llm_repair_report_document_sections(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        rejected,
        validation,
        section_ids=["prices"],
    )

    assert result == expected
    assert captured["invoke_kwargs"]["allow_openai_fallback"] is False
    assert (
        captured["invoke_kwargs"]["attempt_stage"]
        == "report_document_repair"
    )
    system, user = captured["messages"]
    assert "only the rejected sections" in system[1].lower()
    assert '"section_id":"prices"' in user[1]
    assert '"section_id":"security"' not in user[1]
    assert '"track_id":"prices"' in user[1]
    assert '"track_id":"security"' not in user[1]
    assert "VALIDATION_ERRORS" in user[1]
