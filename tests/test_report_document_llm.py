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
from agent.report_document_generation import (
    validate_report_document,
)
from agent.report_sections import count_section_words
from contracts.report import (
    report_aggregate_word_bounds,
    report_section_validation_word_bounds,
)
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
        "json_schema",
        raising=False,
    )
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
        return _ReportClient()

    monkeypatch.setattr(llm, "get_llm_for_stage", get_stage)

    def invoke(factory, model_name, messages, **kwargs):
        captured["client"] = factory()
        captured["model_name"] = model_name
        captured["messages"] = messages
        captured["invoke_kwargs"] = kwargs
        parsed = expected.model_dump(mode="json")
        for field in (
            "contract_version",
            "query_digest",
            "evidence_manifest_id",
            "coverage_status",
        ):
            parsed.pop(field)
        return {
            "raw": SimpleNamespace(content=json.dumps(parsed)),
            "parsed": parsed,
            "parsing_error": None,
        }

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
    assert captured["client"] is structured_client
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
    assert (
        "all evidence and claim lists must contain unique values"
        in system[1].lower()
    )
    assert (
        "sum and mean require at least two unique operands"
        in system[1].lower()
    )
    assert "require exactly two unique operands" in system[1].lower()
    assert "NUMERIC_OBSERVATIONS" in user[1]
    assert '"row_index":0' in user[1]
    assert "prompt_projection_truncated" in user[1]
    assert len(user[1]) <= llm._REPORT_DOCUMENT_PROMPT_BUDGET_CHARS
    assert captured["cache_token"] == "document-cache"
    assert captured["structured_kwargs"] == {
        "method": "json_schema",
        "include_raw": True,
        "strict": True,
    }
    assert {
        "contract_version",
        "query_digest",
        "evidence_manifest_id",
        "coverage_status",
    }.isdisjoint(captured["structured_schema"]["properties"])

    def assert_strict_objects(node):
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(properties)
            assert "default" not in node
            for child in node.values():
                assert_strict_objects(child)
        elif isinstance(node, list):
            for child in node:
                assert_strict_objects(child)

    assert_strict_objects(captured["structured_schema"])


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
    spec_by_id = {
        section.section_id: section for section in document_plan.sections
    }
    section_payloads = [
        *payload["analytical_sections"],
        payload["implications_section"],
        payload["limitations_section"],
        payload["executive_summary"],
    ]
    price_spec = spec_by_id["prices"]
    _, price_maximum = report_section_validation_word_bounds(
        price_spec.target_words
    )
    for section_payload in section_payloads:
        section_id = section_payload["section_id"]
        _, maximum_words = report_section_validation_word_bounds(
            spec_by_id[section_id].target_words
        )
        paragraph = section_payload["paragraphs"][0]
        current_words = count_section_words(paragraph["text"])
        paragraph["text"] += " " + " ".join(
            "f"
            for _ in range(
                maximum_words
                + (section_id == "prices")
                - current_words
            )
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
        "json_schema",
        raising=False,
    )
    monkeypatch.setattr(
        llm,
        "get_llm_for_stage",
        lambda *_args, **_kwargs: _ReportClient(),
    )

    def invoke(_factory, _model_name, messages, **kwargs):
        captured["client"] = _factory()
        captured["messages"] = messages
        captured["invoke_kwargs"] = kwargs
        return {
            "raw": SimpleNamespace(content=expected.model_dump_json()),
            "parsed": expected.model_dump(mode="json"),
            "parsing_error": None,
        }

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
    assert captured["structured_kwargs"] == {
        "method": "json_schema",
        "include_raw": True,
        "strict": True,
    }
    assert isinstance(captured["structured_schema"], dict)
    system, user = captured["messages"]
    assert "only the rejected sections" in system[1].lower()
    assert '"section_id":"prices"' in user[1]
    assert '"section_id":"security"' not in user[1]
    assert '"track_id":"prices"' in user[1]
    assert '"track_id":"security"' not in user[1]
    assert "VALIDATION_ERRORS" in user[1]
    errors_json = user[1].split(
        "VALIDATION_ERRORS:\n",
        1,
    )[1].split("\n\nREJECTED_SECTIONS:", 1)[0]
    repair_context = json.loads(errors_json)
    document_minimum, document_maximum = report_aggregate_word_bounds(
        [section.target_words for section in document_plan.sections]
    )
    assert repair_context["document"] == {
        "error_codes": validation.document_errors,
        "word_count": validation.word_count,
        "minimum_words": document_minimum,
        "maximum_words": document_maximum,
        "required_reduction_words": 1,
        "required_additional_words": 0,
    }
    assert repair_context["document"]["error_codes"] == [
        "DOCUMENT_WORD_COUNT_TOO_LONG"
    ]
    assert repair_context["sections"]["prices"] == {
        "error_codes": validation.section_errors["prices"],
        "word_count": price_maximum + 1,
        "minimum_words": report_section_validation_word_bounds(
            price_spec.target_words
        )[0],
        "maximum_words": price_maximum,
        "required_reduction_words": 1,
        "required_additional_words": 0,
    }
    plan_json = user[1].split(
        "REJECTED_SECTION_PLAN_AND_WORD_BOUNDS:\n",
        1,
    )[1].split("\n\nRESEARCH_SCOPE:", 1)[0]
    repair_plan = json.loads(plan_json)
    assert repair_plan["sections"][0]["maximum_words"] == price_maximum
