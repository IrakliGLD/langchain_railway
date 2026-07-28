"""Section writer/repair provider prompt and cache tests."""

from __future__ import annotations

import json
import logging
import math
import os
from copy import deepcopy
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
from core.llm import _report_section_evidence_slice
from tests.test_report_planner import TABLE_REF, _manifest, _plan_payload
from tests.test_report_sections import _draft


def test_section_writer_receives_only_its_evidence_slice_and_skill_rules(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    captured = {}

    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda key: (None, "token"))
    monkeypatch.setattr(
        ReportSectionDraft,
        "model_json_schema",
        lambda: (_ for _ in ()).throw(
            AssertionError("report section schema must be precomputed")
        ),
    )
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
    assert "direct_claims" in user[1]
    assert "derived_claims" in user[1]
    assert "exact evidence_ref" in system[1]
    assert "code-verifiable derived claims" in system[1]
    assert section.section_id in captured["cache_key"]
    assert _manifest().manifest_id in captured["cache_key"]


def test_section_provider_attempt_stages_are_unique_per_section(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    attempt_stages = []

    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda _key: (None, "token"))
    monkeypatch.setattr(llm, "_cache_set", lambda *_args: None)
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, _messages, **kwargs):
        attempt_stages.append(kwargs["attempt_stage"])
        section = plan.sections[len(attempt_stages) - 1]
        return SimpleNamespace(content=json.dumps(_draft(section)))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    for section in plan.sections[:2]:
        llm.llm_write_report_section(
            "Explain the price trend.",
            plan,
            section,
            _manifest(),
        )

    assert attempt_stages == [
        f"report_section_writer_{section.section_id}"
        for section in plan.sections[:2]
    ]


def test_section_writer_uses_dedicated_model_and_disables_fallback(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]
    captured = {}
    report_client = object()

    monkeypatch.setattr(llm, "REPORT_MODEL_TYPE", "openai", raising=False)
    monkeypatch.setattr(llm, "REPORT_MODEL", "gpt-5.6-terra", raising=False)
    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda _key: (None, "token"))
    monkeypatch.setattr(llm, "_cache_set", lambda *_args: None)

    def get_stage(*_args, **kwargs):
        captured["stage_kwargs"] = kwargs
        return report_client

    monkeypatch.setattr(llm, "get_llm_for_stage", get_stage)

    def invoke(factory, model_name, _messages, **kwargs):
        captured["client"] = factory()
        captured["model_name"] = model_name
        captured["invoke_kwargs"] = kwargs
        return SimpleNamespace(content=json.dumps(_draft(section)))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    llm.llm_write_report_section(
        "Explain the price trend.",
        plan,
        section,
        _manifest(),
    )

    assert captured["client"] is report_client
    assert captured["model_name"] == "gpt-5.6-terra"
    assert captured["stage_kwargs"]["report_profile"] is True
    assert captured["invoke_kwargs"]["allow_openai_fallback"] is False


def test_section_writer_accepts_openai_responses_content_blocks(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]

    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda _key: (None, "token"))
    monkeypatch.setattr(llm, "_cache_set", lambda *_args: None)
    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        llm,
        "_invoke_with_openai_fallback",
        lambda *_args, **_kwargs: SimpleNamespace(
            content=[
                {"type": "reasoning", "summary": []},
                {
                    "type": "text",
                    "text": json.dumps(_draft(section)),
                    "annotations": [],
                },
            ]
        ),
    )

    draft = llm.llm_write_report_section(
        "Explain the price trend.",
        plan,
        section,
        _manifest(),
    )

    assert isinstance(draft, ReportSectionDraft)


def test_section_repair_attempt_stage_includes_local_attempt_number(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[0]
    captured = {}

    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, _messages, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=json.dumps(_draft(section)))

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    llm.llm_repair_report_section(
        "Explain the price trend.",
        plan,
        section,
        _manifest(),
        _draft(section),
        ["WORD_COUNT_OUT_OF_RANGE"],
        attempt_number=2,
    )

    assert captured["attempt_stage"] == (
        f"report_section_repair_{section.section_id}_attempt_2"
    )


def test_evidence_slice_does_not_reserialize_a_growing_table(monkeypatch):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    manifest_payload = deepcopy(_manifest().model_dump(mode="json"))
    table = manifest_payload["items"][0]
    table["rows"] = [
        {"period": f"2026-{index:03d}", "price": 100 + index}
        for index in range(150)
    ]
    table["total_row_count"] = len(table["rows"])
    manifest = _manifest().model_validate(manifest_payload)
    original = llm._compact_json
    growing_row_counts = []

    def observed(value):
        if (
            isinstance(value, dict)
            and isinstance(value.get("rows"), list)
            and value["rows"]
        ):
            growing_row_counts.append(len(value["rows"]))
        return original(value)

    monkeypatch.setattr(llm, "_compact_json", observed)

    packet = llm._report_section_evidence_slice(section, manifest)

    assert len(packet) <= llm._REPORT_SECTION_EVIDENCE_BUDGET_CHARS
    assert growing_row_counts == []
    decoded = json.loads(packet)
    assert decoded[0]["rows"][0]["row_index"] == 0
    assert decoded[0]["included_row_count"] <= table["total_row_count"]


def test_evidence_slice_preserves_table_boundaries_when_prompt_budget_truncates(
    monkeypatch,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    manifest_payload = deepcopy(_manifest().model_dump(mode="json"))
    table = manifest_payload["items"][0]
    table["rows"] = [
        {
            "period": f"period-{index:03d}-" + ("x" * 120),
            "price": 100 + index,
        }
        for index in range(100)
    ]
    table["total_row_count"] = len(table["rows"])
    table["truncated"] = False
    manifest = _manifest().model_validate(manifest_payload)
    monkeypatch.setattr(llm, "_REPORT_SECTION_EVIDENCE_BUDGET_CHARS", 1600)

    decoded = json.loads(
        llm._report_section_evidence_slice(section, manifest)
    )
    projected = decoded[0]
    row_indexes = [row["row_index"] for row in projected["rows"]]

    assert projected["prompt_projection_truncated"] is True
    assert row_indexes[0] == 0
    assert row_indexes[-1] == 99
    assert row_indexes != list(range(len(row_indexes)))


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


def test_section_evidence_slice_discloses_observed_period_span():
    from core.llm import _report_section_evidence_slice

    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]

    packet = _report_section_evidence_slice(section, _manifest())

    assert '"observed_period_span"' in packet
    assert '"first":"2026-01"' in packet
    assert '"last":"2026-02"' in packet


def test_section_evidence_slice_shadows_the_grounding_scope_gap(caplog):
    from contracts.report_evidence import ReportEvidenceManifest
    from core.llm import _report_section_evidence_slice

    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["columns"] = ["period", "price", "note"]
    table["rows"] = [
        {
            "period": f"2026-{(index % 12) + 1:02d}",
            "price": 100.0 + index,
            "note": "observed settlement commentary " * 10,
        }
        for index in range(200)
    ]
    table["total_row_count"] = 200
    manifest = ReportEvidenceManifest.model_validate(payload)

    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]

    with caplog.at_level(logging.INFO, logger="Enai"):
        packet = json.loads(_report_section_evidence_slice(section, manifest))

    shadow = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("REPORT_GROUNDING_SCOPE_SHADOW ")
    ]
    projected = next(
        entry for entry in packet if entry["kind"] == "table"
    )["included_row_count"]

    assert len(shadow) == 1
    assert shadow[0]["manifest_rows"] == 200
    assert shadow[0]["projected_rows"] == projected
    assert shadow[0]["section_id"] == section.section_id
    assert shadow[0]["section_kind"] == section.kind.value
    assert shadow[0]["target_words"] == section.target_words
    assert shadow[0]["evidence_budget_chars"] == 30_000
    assert shadow[0]["assigned_item_count"] == len(
        section.required_evidence_refs
    )
    assert shadow[0]["per_item_budget_chars"] > 0
    assert shadow[0]["projected_item_chars"] > 0
    assert 0 <= shadow[0]["projection_ratio"] < 1
    assert projected < 200


def test_section_evidence_slice_is_silent_when_every_row_is_projected(caplog):
    from core.llm import _report_section_evidence_slice

    plan = ReportPlan.model_validate(_plan_payload())

    with caplog.at_level(logging.INFO, logger="Enai"):
        _report_section_evidence_slice(plan.sections[1], _manifest())

    assert not [
        record
        for record in caplog.records
        if record.message.startswith("REPORT_GROUNDING_SCOPE_SHADOW ")
    ]


def test_section_evidence_slice_warns_when_a_table_projects_to_no_rows(
    monkeypatch,
    caplog,
):
    from core import llm as llm_module

    monkeypatch.setattr(
        llm_module,
        "projected_row_indices",
        lambda _item, **_kwargs: [],
    )

    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]

    with caplog.at_level(logging.INFO, logger="Enai"):
        _report_section_evidence_slice(section, _manifest())

    shadow = [
        record
        for record in caplog.records
        if record.message.startswith("REPORT_GROUNDING_SCOPE_SHADOW ")
    ]

    assert len(shadow) == 1
    assert shadow[0].levelno == logging.WARNING
    assert json.loads(shadow[0].message.split(" ", 1)[1])["projected_rows"] == 0


def test_section_repair_resamples_instead_of_regenerating_the_same_draft(monkeypatch):
    """Job acf48571: scope_and_evidence returned exactly 136 words on the
    candidate and on both repairs. At temperature 0 a repair re-emits the same
    text, so the extra provider calls cannot converge."""

    from tests.test_report_planner import _manifest, _plan_payload

    captured = []

    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())

    def invoke(_factory, _model, _messages, **kwargs):
        captured.append(kwargs.get("sampling_temperature"))
        return SimpleNamespace(
            content=json.dumps(_draft(ReportPlan.model_validate(_plan_payload()).sections[1]))
        )

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    draft = ReportSectionDraft.model_validate(_draft(section))

    for attempt_number in (2, 3):
        llm.llm_repair_report_section(
            "Explain the price trend.",
            plan,
            section,
            _manifest(),
            draft,
            ["WORD_COUNT_OUT_OF_RANGE"],
            attempt_number=attempt_number,
        )

    assert all(temperature is not None for temperature in captured)
    assert captured[0] > 0.0
    assert captured[1] > captured[0]


def test_section_writer_keeps_deterministic_sampling(monkeypatch):
    from tests.test_report_planner import _manifest, _plan_payload

    captured = []

    monkeypatch.setattr(llm, "get_llm_for_stage", lambda *a, **k: object())
    monkeypatch.setattr(llm, "_cache_get_or_reserve", lambda _k: (None, "token"))
    monkeypatch.setattr(llm, "_cache_set", lambda *_a, **_k: None)

    def invoke(_factory, _model, _messages, **kwargs):
        captured.append(kwargs.get("sampling_temperature"))
        return SimpleNamespace(
            content=json.dumps(_draft(ReportPlan.model_validate(_plan_payload()).sections[1]))
        )

    monkeypatch.setattr(llm, "_invoke_with_openai_fallback", invoke)

    plan = ReportPlan.model_validate(_plan_payload())
    llm.llm_write_report_section(
        "Explain the price trend.",
        plan,
        plan.sections[1],
        _manifest(),
    )

    assert captured == [None]
