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
from contracts.report import report_section_validation_word_bounds
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
    assert "plan order" in system[1].lower()
    assert "executive summary" not in system[1].lower()
    assert (
        "all evidence and claim lists must contain unique values"
        in system[1].lower()
    )
    assert (
        "sum and mean require at least two unique operands"
        in system[1].lower()
    )
    assert "require exactly two unique operands" in system[1].lower()
    # The gate rejects a section under its floor and REPORT_DOCUMENT_INVALID is
    # not retryable, so the writer must not be told the floor is advisory.
    assert "minimum_words is enforced" in system[1].lower()
    assert "recommended_maximum_words is guidance" in system[1].lower()
    assert "never pad" in system[1].lower()
    assert "word targets are recommendations" not in system[1].lower()
    assert "do not append a sources" in system[1].lower()
    assert "verify every section" not in system[1].lower()
    assert "prefer direct observations" in system[1].lower()
    assert (
        "do not introduce new arithmetic unless"
        in system[1].lower()
    )
    assert "do not emit unused claim entries" in system[1].lower()
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


def test_analysis_writer_projects_only_analysis_sections(monkeypatch):
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    analysis_ids = [
        section.section_id
        for section in document_plan.sections
        if section.role.value == "analysis"
    ]
    expected = ReportDocumentRepair(
        contract_version="report-document-repair-v1",
        sections=[
            section
            for section in draft.sections
            if section.section_id in analysis_ids
        ],
    )
    captured = {}

    def invoke_contract(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(llm, "_invoke_report_document_contract", invoke_contract)

    result = llm.llm_write_report_analysis_sections(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        section_ids=analysis_ids,
    )

    assert result == expected
    assert captured["attempt_stage"] == "report_analysis_writer"
    assert "do not append a sources" in captured["system"].lower()
    assert captured["payload_bindings"] == {
        "contract_version": "report-document-repair-v1"
    }
    assert '"section_id":"prices"' in captured["prompt"]
    assert '"section_id":"implications"' not in captured["prompt"]
    assert '"section_id":"limitations"' not in captured["prompt"]


def test_synthesis_writer_consumes_validated_analysis(monkeypatch):
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    analysis_sections = [
        section
        for section, spec in zip(
            draft.sections,
            document_plan.sections,
            strict=True,
        )
        if spec.role.value == "analysis"
    ]
    synthesis_ids = [
        section.section_id
        for section in document_plan.sections
        if section.role.value != "analysis"
    ]
    expected = ReportDocumentRepair(
        contract_version="report-document-repair-v1",
        sections=[
            section
            for section in draft.sections
            if section.section_id in synthesis_ids
        ],
    )
    captured = {}

    def invoke_contract(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(llm, "_invoke_report_document_contract", invoke_contract)

    result = llm.llm_write_report_synthesis_sections(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        analysis_sections=analysis_sections,
        section_ids=synthesis_ids,
    )

    assert result == expected
    assert captured["attempt_stage"] == "report_synthesis_writer"
    assert "VALIDATED_ANALYSIS_SECTIONS" in captured["prompt"]
    assert '"section_id":"prices"' in captured["prompt"]
    assert "must not introduce a numeric claim" in captured["system"].lower()
    assert "do not append a sources" in captured["system"].lower()
    assert '"section_id":"implications"' in captured["prompt"]


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
    payload["sections"][0]["paragraphs"][0]["direct_claims"] = []
    rejected = ReportDocumentDraft.model_validate(payload)
    validation = validate_report_document(
        rejected,
        document_plan,
        manifest,
        research_plan,
    )
    validation = validation.model_copy(
        update={
            "valid": False,
            "document_errors": ["DOCUMENT_IDENTITY_MISMATCH"],
        }
    )
    replacement = draft.sections[0]
    expected = ReportDocumentRepair(
        contract_version="report-document-repair-v1",
        sections=[replacement],
    )
    captured = {}
    structured_client = object()
    project_inputs = llm._report_document_prompt_inputs

    def capture_projection(*args, **kwargs):
        captured["projection_kwargs"] = kwargs
        return project_inputs(*args, **kwargs)

    monkeypatch.setattr(
        llm,
        "_report_document_prompt_inputs",
        capture_projection,
    )

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
        attempt_number=3,
    )

    assert result == expected
    assert captured["invoke_kwargs"]["allow_openai_fallback"] is False
    assert captured["projection_kwargs"]["evidence_budget_chars"] == 20_000
    assert captured["projection_kwargs"]["observation_budget_chars"] == 6_000
    assert (
        captured["invoke_kwargs"]["attempt_stage"]
        == "report_document_repair_attempt_3"
    )
    assert captured["structured_kwargs"] == {
        "method": "json_schema",
        "include_raw": True,
        "strict": True,
    }
    assert isinstance(captured["structured_schema"], dict)
    system, user = captured["messages"]
    assert "only the rejected sections" in system[1].lower()
    assert "do not append a sources" in system[1].lower()
    assert "blocking validation errors" in system[1].lower()
    assert "word_count_too_short" in system[1].lower()
    assert "minimum_words" in system[1].lower()
    assert "word_count_too_long" in system[1].lower()
    assert "maximum_words" in system[1].lower()
    # A repairer cannot count its own words, so the codes above are actionable
    # only alongside the measured count and the deficit.
    assert "words_to_add" in system[1].lower()
    assert "do not estimate the length you produced" in system[1].lower()
    assert "claimable_coordinates" in system[1].lower()
    assert "prefer direct observations" in system[1].lower()
    assert (
        "do not introduce new arithmetic unless"
        in system[1].lower()
    )
    assert "do not emit unused claim entries" in system[1].lower()
    assert "do not delete a repairable analytical finding" in system[1].lower()
    assert '"section_id":"prices"' in user[1]
    assert '"section_id":"security"' not in user[1]
    assert '"track_id":"prices"' in user[1]
    assert '"track_id":"security"' not in user[1]
    assert "VALIDATION_ERRORS" in user[1]
    assert "VERIFIED_DERIVED_REPAIR_HINTS" in user[1]
    # A bare UNGROUNDED_NUMERIC_CLAIM makes the repairer guess which of its
    # numbers offended, and the code is not retryable, so a wrong guess costs
    # the whole report. The prompt must name the values.
    assert "UNGROUNDED_VALUE_REPAIR_HINTS" in user[1]
    assert "ungrounded_value_repair_hints" in system[1].lower()
    ungrounded_json = user[1].split(
        "UNGROUNDED_VALUE_REPAIR_HINTS:\n",
        1,
    )[1].split("\n\nCLAIMABLE_COORDINATES:", 1)[0]
    ungrounded_hints = json.loads(ungrounded_json)
    assert [hint["section_id"] for hint in ungrounded_hints] == ["prices"]
    stripped_paragraph = rejected.sections[0].paragraphs[0]
    assert ungrounded_hints[0]["ungrounded_values"]
    assert all(
        value in stripped_paragraph.text
        for value in ungrounded_hints[0]["ungrounded_values"]
    )
    errors_json = user[1].split(
        "VALIDATION_ERRORS:\n",
        1,
    )[1].split("\n\nREJECTED_SECTIONS:", 1)[0]
    repair_context = json.loads(errors_json)
    assert repair_context["document"]["error_codes"] == [
        "DOCUMENT_IDENTITY_MISMATCH"
    ]
    assert repair_context["sections"]["prices"]["error_codes"] == (
        [
            "DOCUMENT_IDENTITY_MISMATCH",
            *validation.section_errors["prices"],
        ]
    )
    assert validation.section_errors["prices"]
    plan_json = user[1].split(
        "REJECTED_SECTION_PLAN_AND_RECOMMENDED_WORD_TARGETS:\n",
        1,
    )[1].split("\n\nRESEARCH_SCOPE:", 1)[0]
    repair_plan = json.loads(plan_json)
    assert (
        repair_plan["sections"][0]["recommended_maximum_words"]
        > repair_plan["sections"][0]["minimum_words"]
    )
    # The floor is enforced, so the payload must not label it a recommendation.
    assert "recommended_minimum_words" not in repair_plan["sections"][0]


def test_every_report_writer_prompt_carries_the_claim_contract(monkeypatch):
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    analysis_ids = [
        section.section_id
        for section in document_plan.sections
        if section.role.value == "analysis"
    ]
    synthesis_ids = [
        section.section_id
        for section in document_plan.sections
        if section.role.value != "analysis"
    ]
    analysis_sections = [
        section
        for section in draft.sections
        if section.section_id in analysis_ids
    ]
    captured = []

    def invoke_contract(**kwargs):
        captured.append(kwargs["system"])
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=analysis_sections,
        )

    monkeypatch.setattr(
        llm, "_invoke_report_document_contract", invoke_contract
    )

    llm.llm_write_report_analysis_sections(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        section_ids=analysis_ids,
    )
    llm.llm_write_report_synthesis_sections(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        analysis_sections=analysis_sections,
        section_ids=synthesis_ids,
    )

    assert len(captured) == 2
    for system in captured:
        assert llm._REPORT_CLAIM_CONTRACT_RULES in system


def _capture_repair_invocation(monkeypatch, *, attempt_number: int = 2):
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    validation = validate_report_document(
        draft, document_plan, manifest, research_plan
    )
    captured = {}

    def invoke_contract(**kwargs):
        captured.update(kwargs)
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[draft.sections[0]],
        )

    monkeypatch.setattr(
        llm, "_invoke_report_document_contract", invoke_contract
    )
    llm.llm_repair_report_document_sections(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        draft,
        validation,
        section_ids=[draft.sections[0].section_id],
        attempt_number=attempt_number,
    )
    return captured


def _repair_prompt_block(prompt: str, header: str) -> str:
    body = prompt.split(f"{header}:\n", 1)[1]
    return body.split("\n\n", 1)[0]


def test_repair_prompt_names_the_word_count_shortfall(monkeypatch):
    """A repairer cannot count its own words, so the deficit must be given.

    Job cf47a2f6 was told "at least 241" twice, produced 202 both times, and
    lost the report to a non-retryable REPORT_DOCUMENT_INVALID. Naming the
    measured count and the gap is the contract that made ungrounded values
    repairable.
    """

    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    section_id = draft.sections[0].section_id
    payload = draft.model_dump(mode="json")
    paragraph = payload["sections"][0]["paragraphs"][0]
    paragraph["text"] = " ".join(paragraph["text"].split()[:40])
    short_draft = ReportDocumentDraft.model_validate(payload)
    validation = validate_report_document(
        short_draft, document_plan, manifest, research_plan
    )
    captured = {}

    def invoke_contract(**kwargs):
        captured.update(kwargs)
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[draft.sections[0]],
        )

    monkeypatch.setattr(
        llm, "_invoke_report_document_contract", invoke_contract
    )
    llm.llm_repair_report_document_sections(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        short_draft,
        validation,
        section_ids=[section_id],
    )

    assert "WORD_COUNT_TOO_SHORT" in validation.section_errors[section_id]
    errors = json.loads(
        _repair_prompt_block(captured["prompt"], "VALIDATION_ERRORS")
    )
    context = errors["sections"][section_id]
    minimum_words, maximum_words = report_section_validation_word_bounds(
        next(
            section
            for section in document_plan.sections
            if section.section_id == section_id
        ).target_words,
        evidence_row_count=manifest.assigned_row_count(
            next(
                section
                for section in document_plan.sections
                if section.section_id == section_id
            ).required_evidence_refs
        ),
    )
    assert context["word_count"] == count_section_words(
        short_draft.sections[0].content_markdown
    )
    assert context["minimum_words"] == minimum_words
    assert context["maximum_words"] == maximum_words
    assert context["words_to_add"] == minimum_words - context["word_count"]
    assert context["words_to_add"] > 0


def test_repair_prompt_names_the_coordinates_a_numberless_section_may_cite(
    monkeypatch,
):
    """NUMERIC_FINDING_MISSING alone leaves the repairer inventing a number.

    Every listed coordinate is verified against the same function that judges
    the writer's copy of it, so citing one cannot trade this code for
    UNGROUNDED_NUMERIC_CLAIM.
    """

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
    for section_payload in payload["sections"]:
        for paragraph in section_payload["paragraphs"]:
            paragraph["direct_claims"] = []
            paragraph["derived_claims"] = []
            paragraph["text"] = " ".join(
                word
                for word in paragraph["text"].split()
                if not any(character.isdigit() for character in word)
            )
    numberless = ReportDocumentDraft.model_validate(payload)
    validation = validate_report_document(
        numberless, document_plan, manifest, research_plan
    )
    flagged_ids = [
        section_id
        for section_id, codes in validation.section_errors.items()
        if "NUMERIC_FINDING_MISSING" in codes
    ]
    assert flagged_ids, validation.section_errors
    captured = {}

    def invoke_contract(**kwargs):
        captured.update(kwargs)
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                section
                for section in draft.sections
                if section.section_id in flagged_ids
            ],
        )

    monkeypatch.setattr(
        llm, "_invoke_report_document_contract", invoke_contract
    )
    llm.llm_repair_report_document_sections(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        numberless,
        validation,
        section_ids=flagged_ids,
    )

    hints = json.loads(
        _repair_prompt_block(captured["prompt"], "CLAIMABLE_COORDINATES")
    )
    assert [hint["section_id"] for hint in hints] == flagged_ids
    item_by_ref = manifest.item_by_ref()
    for hint in hints:
        assert hint["claimable_coordinates"]
        for coordinate in hint["claimable_coordinates"]:
            item = item_by_ref[coordinate["evidence_ref"]]
            assert coordinate["column"] in item.citable_numeric_columns()
    errors = json.loads(
        _repair_prompt_block(captured["prompt"], "VALIDATION_ERRORS")
    )
    for section_id in flagged_ids:
        context = errors["sections"][section_id]
        assert context["numeric_claim_count"] == 0
        assert context["required_numeric_claims"] > 0
        assert context["numeric_claims_to_add"] == (
            context["required_numeric_claims"]
        )


def test_document_repair_attempts_use_distinct_provider_stages(monkeypatch):
    first = _capture_repair_invocation(monkeypatch, attempt_number=2)
    second = _capture_repair_invocation(monkeypatch, attempt_number=3)

    assert first["attempt_stage"] == "report_document_repair_attempt_2"
    assert second["attempt_stage"] == "report_document_repair_attempt_3"


def test_document_repair_resamples_when_the_model_accepts_temperature(
    monkeypatch,
):
    monkeypatch.setattr(llm, "REPORT_REASONING_EFFORT", None, raising=False)

    captured = _capture_repair_invocation(monkeypatch)

    assert captured["sampling_temperature"] == (
        llm._repair_sampling_temperature(2)
    )
    assert captured["use_cache"] is False


def test_document_repair_sends_no_temperature_to_a_reasoning_model(
    monkeypatch,
):
    monkeypatch.setattr(
        llm, "REPORT_REASONING_EFFORT", "medium", raising=False
    )

    captured = _capture_repair_invocation(monkeypatch)

    assert captured["sampling_temperature"] is None
    assert captured["use_cache"] is False


def test_compact_writer_and_repair_share_the_claim_contract(monkeypatch):
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    draft = _valid_document_draft(document_plan, manifest)
    validation = validate_report_document(
        draft, document_plan, manifest, research_plan
    )
    captured = []

    def invoke_contract(**kwargs):
        captured.append(kwargs["system"])
        return draft

    monkeypatch.setattr(
        llm, "_invoke_report_document_contract", invoke_contract
    )

    llm.llm_write_report_document(
        _QUERY, document_plan, research_plan, manifest, packets
    )
    llm.llm_repair_report_document_sections(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        draft,
        validation,
        section_ids=[document_plan.sections[0].section_id],
    )

    assert len(captured) == 2
    for system in captured:
        assert llm._REPORT_CLAIM_CONTRACT_RULES in system
