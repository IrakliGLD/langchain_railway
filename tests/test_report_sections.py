"""Parallel report-section generation, validation, and repair tests."""

from __future__ import annotations

import threading
from collections import Counter

import pytest

from agent.report_sections import (
    ReportSectionGenerationError,
    generate_report_sections,
    validate_report_section,
)
from contracts.report import ReportPlan
from contracts.report_sections import ReportSectionDraft
from tests.test_report_planner import _manifest, _plan_payload


def _words(count: int, *, prefix: str = "Evidence") -> str:
    return " ".join([prefix, *(["supports"] * (count - 1))])


def _draft(section, *, text: str | None = None) -> dict:
    return {
        "contract_version": "report-section-v1",
        "section_id": section.section_id,
        "title": section.title,
        "paragraphs": [
            {
                "text": text or _words(section.target_words),
                "evidence_refs": section.required_evidence_refs,
            }
        ],
    }


def test_section_validation_enforces_budget_refs_and_numeric_grounding():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[2]
    grounded_text = "Price was 120.0 GEL per MWh. " + _words(
        section.target_words - 7,
        prefix="Evidence",
    )
    grounded = ReportSectionDraft.model_validate(
        _draft(section, text=grounded_text)
    )

    valid = validate_report_section(grounded, section, _manifest())

    assert valid.valid is True
    assert valid.error_codes == []
    assert valid.word_count >= int(section.target_words * 0.9)

    unsupported = ReportSectionDraft.model_validate(
        _draft(
            section,
            text="Price was 999.0 GEL per MWh. "
            + _words(section.target_words - 7),
        )
    )
    invalid = validate_report_section(unsupported, section, _manifest())
    assert invalid.valid is False
    assert "UNGROUNDED_NUMERIC_CLAIM" in invalid.error_codes

    wrong_ref_payload = _draft(section)
    wrong_ref_payload["paragraphs"][0]["evidence_refs"] = [
        "evidence:statistics:" + "9" * 16
    ]
    wrong_ref = validate_report_section(
        ReportSectionDraft.model_validate(wrong_ref_payload),
        section,
        _manifest(),
    )
    assert "EVIDENCE_REF_NOT_ALLOWED" in wrong_ref.error_codes
    assert "REQUIRED_EVIDENCE_NOT_USED" in wrong_ref.error_codes


def test_sections_generate_in_parallel_and_return_in_plan_order():
    plan = ReportPlan.model_validate(_plan_payload())
    barrier = threading.Barrier(len(plan.sections))
    active = 0
    max_active = 0
    lock = threading.Lock()

    def generate(_query, _plan, section, _manifest):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2)
        try:
            return _draft(section)
        finally:
            with lock:
                active -= 1

    drafts = generate_report_sections(
        "Explain the price trend.",
        plan,
        _manifest(),
        generate_section=generate,
        max_workers=len(plan.sections),
    )

    assert max_active == len(plan.sections)
    assert [draft.section_id for draft in drafts] == [
        section.section_id for section in plan.sections
    ]


def test_only_invalid_sections_receive_one_repair_call():
    plan = ReportPlan.model_validate(_plan_payload())
    generated = Counter()
    repaired = Counter()

    def generate(_query, _plan, section, _manifest):
        generated[section.section_id] += 1
        payload = _draft(section)
        if section.section_id == "key_findings":
            payload["paragraphs"][0]["text"] = "This section remains much too short."
        return payload

    def repair(_query, _plan, section, _manifest, _draft_value, error_codes):
        repaired[section.section_id] += 1
        assert "WORD_COUNT_OUT_OF_RANGE" in error_codes
        return _draft(section)

    drafts = generate_report_sections(
        "Explain the price trend.",
        plan,
        _manifest(),
        generate_section=generate,
        repair_section=repair,
        max_workers=len(plan.sections),
    )

    assert len(drafts) == len(plan.sections)
    assert repaired == Counter({"key_findings": 1})
    assert all(count == 1 for count in generated.values())


def test_failed_repair_aborts_with_typed_section_error():
    plan = ReportPlan.model_validate(_plan_payload())

    with pytest.raises(ReportSectionGenerationError) as exc_info:
        generate_report_sections(
            "Explain the price trend.",
            plan,
            _manifest(),
            generate_section=lambda _q, _p, section, _m: {
                **_draft(section),
                "paragraphs": [
                    {
                        "text": "This section remains much too short.",
                        "evidence_refs": section.required_evidence_refs,
                    }
                ],
            },
            repair_section=lambda _q, _p, section, _m, _d, _e: {
                **_draft(section),
                "paragraphs": [
                    {
                        "text": "This repaired section is still much too short.",
                        "evidence_refs": section.required_evidence_refs,
                    }
                ],
            },
            max_workers=len(plan.sections),
        )

    assert exc_info.value.section_id in {
        section.section_id for section in plan.sections
    }
    assert "WORD_COUNT_OUT_OF_RANGE" in exc_info.value.error_codes


def test_valid_resume_drafts_are_not_regenerated_and_progress_is_checkpointable():
    plan = ReportPlan.model_validate(_plan_payload())
    resumed = ReportSectionDraft.model_validate(_draft(plan.sections[0]))
    generated = []
    progress = []

    drafts = generate_report_sections(
        "Explain the price trend.",
        plan,
        _manifest(),
        existing_drafts={resumed.section_id: resumed},
        generate_section=lambda _q, _p, section, _m: (
            generated.append(section.section_id) or _draft(section)
        ),
        progress_callback=lambda completed, total, draft: progress.append(
            (completed, total, draft.section_id)
        ),
        max_workers=len(plan.sections),
    )

    assert resumed.section_id not in generated
    assert [draft.section_id for draft in drafts] == [
        section.section_id for section in plan.sections
    ]
    assert progress[-1][:2] == (len(plan.sections), len(plan.sections))
