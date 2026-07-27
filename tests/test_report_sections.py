"""Parallel report-section generation, validation, and repair tests."""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from copy import deepcopy

import pytest
from pydantic import ValidationError

from agent.report_sections import (
    ReportSectionGenerationError,
    generate_report_sections,
    validate_report_section,
)
from contracts.report import ReportPlan
from contracts.report_sections import ReportSectionDraft
from tests.test_report_planner import _manifest, _plan_payload
from utils.provider_attempts import (
    ProviderDeliveryDisposition,
    ProviderExecutionError,
)


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


@pytest.mark.parametrize(
    ("column", "unit", "expected_valid"),
    [
        ("share_import", "ratio", True),
        ("generation_quantity", "MWh", False),
    ],
)
def test_section_validation_accepts_percent_conversion_only_for_ratio_evidence(
    column: str,
    unit: str,
    expected_valid: bool,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[2]
    manifest_payload = deepcopy(_manifest().model_dump(mode="json"))
    table = manifest_payload["items"][0]
    table["columns"] = ["period", column]
    table["rows"] = [
        {"period": "2026-01", column: 0.1438},
        {"period": "2026-02", column: 0.1512},
    ]
    table["unit_by_column"] = {column: unit}
    manifest = _manifest().model_validate(manifest_payload)
    draft_payload = _draft(
        section,
        text=(
            "The observed value was 14.38% in the supplied evidence. "
            + _words(section.target_words)
        ),
    )
    draft_payload["paragraphs"][0]["evidence_refs"] = (
        section.required_evidence_refs
    )

    validation = validate_report_section(
        ReportSectionDraft.model_validate(draft_payload),
        section,
        manifest,
    )

    assert validation.valid is expected_valid
    assert (
        "UNGROUNDED_NUMERIC_CLAIM" in validation.error_codes
    ) is (not expected_valid)


@pytest.mark.parametrize(
    ("evidence_value", "claim", "expected_valid"),
    [
        (120.0, "120", True),
        (120, "120.00", True),
        (1234.56, "1,234.6", True),
        (1234.56, "1,235", True),
        (1234.56, "1,236", False),
    ],
)
def test_section_numeric_grounding_accepts_equivalent_display_values(
    evidence_value,
    claim,
    expected_valid,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    manifest_payload = deepcopy(_manifest().model_dump(mode="json"))
    manifest_payload["items"][0]["rows"] = [
        {"period": "2026-01", "price": evidence_value},
    ]
    manifest_payload["items"][0]["total_row_count"] = 1
    manifest = _manifest().model_validate(manifest_payload)
    draft = ReportSectionDraft.model_validate(
        _draft(
            section,
            text=(
                f"Observed price was {claim} GEL per MWh. "
                + _words(section.target_words - 8)
            ),
        )
    )

    validation = validate_report_section(draft, section, manifest)

    assert validation.valid is expected_valid
    assert (
        "UNGROUNDED_NUMERIC_CLAIM" in validation.error_codes
    ) is (not expected_valid)


def test_section_numeric_grounding_does_not_treat_row_count_as_a_fact():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    draft = ReportSectionDraft.model_validate(
        _draft(
            section,
            text=(
                "Observed price was 2 GEL per MWh. "
                + _words(section.target_words - 8)
            ),
        )
    )

    validation = validate_report_section(draft, section, _manifest())

    assert "UNGROUNDED_NUMERIC_CLAIM" in validation.error_codes


@pytest.mark.parametrize("claim", ["1e3", ".5"])
def test_section_numeric_grounding_does_not_allow_numeric_notation_bypasses(
    claim,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    draft = ReportSectionDraft.model_validate(
        _draft(
            section,
            text=(
                f"Unsupported price was {claim} GEL per MWh. "
                + _words(section.target_words - 8)
            ),
        )
    )

    validation = validate_report_section(draft, section, _manifest())

    assert "UNGROUNDED_NUMERIC_CLAIM" in validation.error_codes


def test_section_numeric_grounding_accepts_supported_scientific_notation():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    manifest_payload = deepcopy(_manifest().model_dump(mode="json"))
    manifest_payload["items"][0]["rows"] = [
        {"period": "2026-01", "price": 1000},
    ]
    manifest_payload["items"][0]["total_row_count"] = 1
    manifest = _manifest().model_validate(manifest_payload)
    draft = ReportSectionDraft.model_validate(
        _draft(
            section,
            text=(
                "Observed price was 1e3 GEL per MWh. "
                + _words(section.target_words - 8)
            ),
        )
    )

    validation = validate_report_section(draft, section, manifest)

    assert validation.valid is True


@pytest.mark.parametrize(
    ("period", "expected_valid"),
    [
        ("2026-02", True),
        ("2026-2", True),
        ("2026-03", False),
    ],
)
def test_section_grounding_matches_periods_as_typed_facts(
    period,
    expected_valid,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    draft = ReportSectionDraft.model_validate(
        _draft(
            section,
            text=(
                f"Observed price in {period} was 130 GEL per MWh. "
                + _words(section.target_words - 10)
            ),
        )
    )

    validation = validate_report_section(draft, section, _manifest())

    assert validation.valid is expected_valid
    assert (
        "UNGROUNDED_NUMERIC_CLAIM" in validation.error_codes
    ) is (not expected_valid)


def test_section_grounding_rejects_untyped_derived_arithmetic():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    draft = ReportSectionDraft.model_validate(
        _draft(
            section,
            text=(
                "The derived average price was 125 GEL per MWh. "
                + _words(section.target_words - 9)
            ),
        )
    )

    validation = validate_report_section(draft, section, _manifest())

    assert "UNGROUNDED_NUMERIC_CLAIM" in validation.error_codes


def _derived_claim(
    *,
    operation: str = "mean",
    display_value: str = "125",
    unit: str = "GEL/MWh",
    row_indexes: tuple[int, ...] = (0, 1),
    column: str = "price",
) -> dict:
    return {
        "operation": operation,
        "operands": [
            {
                "evidence_ref": "evidence:table:" + "1" * 16,
                "row_index": row_index,
                "column": column,
            }
            for row_index in row_indexes
        ],
        "display_value": display_value,
        "unit": unit,
    }


def test_section_contract_accepts_bounded_typed_derived_claims():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    payload = _draft(
        section,
        text=(
            "The code-verified mean price was 125 GEL/MWh. "
            + _words(section.target_words - 8)
        ),
    )
    payload["paragraphs"][0]["derived_claims"] = [_derived_claim()]

    draft = ReportSectionDraft.model_validate(payload)

    claim = draft.paragraphs[0].derived_claims[0]
    assert claim.operation == "mean"
    assert claim.operands[1].row_index == 1
    assert claim.display_value == "125"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda claim: claim["operands"].pop(),
        lambda claim: claim["operands"].append(dict(claim["operands"][0])),
        lambda claim: claim.__setitem__("display_value", "1e3"),
        lambda claim: claim.__setitem__("display_value", "1,2,5"),
        lambda claim: claim.__setitem__("unit", "%"),
    ],
)
def test_section_contract_rejects_malformed_derived_claims(mutation):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    payload = _draft(section)
    claim = _derived_claim()
    mutation(claim)
    payload["paragraphs"][0]["derived_claims"] = [claim]

    with pytest.raises(ValidationError):
        ReportSectionDraft.model_validate(payload)


def test_section_grounding_accepts_code_verified_mean_claim():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    payload = _draft(
        section,
        text=(
            "The code-verified mean price was 125 GEL/MWh. "
            + _words(section.target_words - 8)
        ),
    )
    payload["paragraphs"][0]["derived_claims"] = [_derived_claim()]

    validation = validate_report_section(
        ReportSectionDraft.model_validate(payload),
        section,
        _manifest(),
    )

    assert validation.valid is True
    assert validation.error_codes == []


def test_section_grounding_rejects_incorrect_or_unresolvable_derived_claims():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]

    for claim in (
        _derived_claim(display_value="126"),
        _derived_claim(row_indexes=(0, 99)),
        _derived_claim(column="missing_column"),
    ):
        payload = _draft(
            section,
            text=(
                f"The claimed derived price was {claim['display_value']} GEL/MWh. "
                + _words(section.target_words - 8)
            ),
        )
        payload["paragraphs"][0]["derived_claims"] = [claim]

        validation = validate_report_section(
            ReportSectionDraft.model_validate(payload),
            section,
            _manifest(),
        )

        assert "DERIVED_CLAIM_INVALID" in validation.error_codes
        assert "UNGROUNDED_NUMERIC_CLAIM" in validation.error_codes


def test_section_grounding_rejects_unused_derived_claim():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    payload = _draft(
        section,
        text=(
            "The evidence supports a code-verified average price observation. "
            + _words(section.target_words - 8)
        ),
    )
    payload["paragraphs"][0]["derived_claims"] = [_derived_claim()]

    validation = validate_report_section(
        ReportSectionDraft.model_validate(payload),
        section,
        _manifest(),
    )

    assert "DERIVED_CLAIM_NOT_USED" in validation.error_codes


def test_section_grounding_accepts_rounded_percent_change_claim():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    payload = _draft(
        section,
        text=(
            "The code-verified price increase was 8.3%. "
            + _words(section.target_words - 7)
        ),
    )
    payload["paragraphs"][0]["derived_claims"] = [
        _derived_claim(
            operation="percent_change",
            display_value="8.3%",
            unit="%",
        )
    ]

    validation = validate_report_section(
        ReportSectionDraft.model_validate(payload),
        section,
        _manifest(),
    )

    assert validation.valid is True
    assert validation.error_codes == []


@pytest.mark.parametrize(
    ("operation", "display_value", "unit"),
    [
        ("difference", "10", "GEL/MWh"),
        ("ratio", "92.3%", "%"),
    ],
)
def test_section_grounding_verifies_supported_derived_operations(
    operation,
    display_value,
    unit,
):
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    payload = _draft(
        section,
        text=(
            f"The code-verified result was {display_value} {unit}. "
            + _words(section.target_words - 7)
        ),
    )
    payload["paragraphs"][0]["derived_claims"] = [
        _derived_claim(
            operation=operation,
            display_value=display_value,
            unit=unit,
        )
    ]

    validation = validate_report_section(
        ReportSectionDraft.model_validate(payload),
        section,
        _manifest(),
    )

    assert validation.valid is True


def test_section_grounding_allows_sum_only_for_additive_units():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    manifest_payload = deepcopy(_manifest().model_dump(mode="json"))
    table = manifest_payload["items"][0]
    table["columns"] = ["period", "generation_mwh"]
    table["rows"] = [
        {"period": "2026-01", "generation_mwh": 120},
        {"period": "2026-02", "generation_mwh": 130},
    ]
    table["unit_by_column"] = {"generation_mwh": "MWh"}
    manifest = _manifest().model_validate(manifest_payload)
    payload = _draft(
        section,
        text=(
            "The code-verified generation total was 250 MWh. "
            + _words(section.target_words - 7)
        ),
    )
    payload["paragraphs"][0]["derived_claims"] = [
        _derived_claim(
            operation="sum",
            display_value="250",
            unit="MWh",
            column="generation_mwh",
        )
    ]

    valid = validate_report_section(
        ReportSectionDraft.model_validate(payload),
        section,
        manifest,
    )

    assert valid.valid is True

    invalid_payload = _draft(
        section,
        text=(
            "The claimed sum of prices was 250 GEL/MWh. "
            + _words(section.target_words - 8)
        ),
    )
    invalid_payload["paragraphs"][0]["derived_claims"] = [
        _derived_claim(
            operation="sum",
            display_value="250",
            unit="GEL/MWh",
        )
    ]
    invalid = validate_report_section(
        ReportSectionDraft.model_validate(invalid_payload),
        section,
        _manifest(),
    )

    assert "DERIVED_CLAIM_INVALID" in invalid.error_codes


def test_section_grounding_verifies_percentage_point_change_for_ratio_evidence():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    manifest_payload = deepcopy(_manifest().model_dump(mode="json"))
    table = manifest_payload["items"][0]
    table["columns"] = ["period", "share_import"]
    table["rows"] = [
        {"period": "2026-01", "share_import": 0.14},
        {"period": "2026-02", "share_import": 0.17},
    ]
    table["unit_by_column"] = {"share_import": "ratio"}
    manifest = _manifest().model_validate(manifest_payload)
    payload = _draft(
        section,
        text=(
            "The code-verified increase was 3 percentage points. "
            + _words(section.target_words - 7)
        ),
    )
    payload["paragraphs"][0]["derived_claims"] = [
        _derived_claim(
            operation="percentage_point_change",
            display_value="3",
            unit="percentage points",
            column="share_import",
        )
    ]

    validation = validate_report_section(
        ReportSectionDraft.model_validate(payload),
        section,
        manifest,
    )

    assert validation.valid is True


def test_section_grounding_rejects_wrong_derived_unit_in_paragraph_text():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    payload = _draft(
        section,
        text=(
            "The code-verified mean was 125 MW. "
            + _words(section.target_words - 7)
        ),
    )
    payload["paragraphs"][0]["derived_claims"] = [_derived_claim()]

    validation = validate_report_section(
        ReportSectionDraft.model_validate(payload),
        section,
        _manifest(),
    )

    assert "DERIVED_CLAIM_NOT_USED" in validation.error_codes


def test_section_grounding_rejects_division_by_zero_without_crashing():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[1]
    manifest_payload = deepcopy(_manifest().model_dump(mode="json"))
    manifest_payload["items"][0]["rows"][1]["price"] = 0
    manifest = _manifest().model_validate(manifest_payload)
    payload = _draft(
        section,
        text=(
            "The claimed code-verified ratio was 100%. "
            + _words(section.target_words - 7)
        ),
    )
    payload["paragraphs"][0]["derived_claims"] = [
        _derived_claim(
            operation="ratio",
            display_value="100%",
            unit="%",
        )
    ]

    validation = validate_report_section(
        ReportSectionDraft.model_validate(payload),
        section,
        manifest,
    )

    assert "DERIVED_CLAIM_INVALID" in validation.error_codes


def test_section_grounding_rejects_derived_operand_outside_paragraph_scope():
    plan = ReportPlan.model_validate(_plan_payload())
    section = plan.sections[2]
    payload = _draft(
        section,
        text=(
            "The code-verified mean price was 125 GEL/MWh. "
            + _words(section.target_words - 8)
        ),
    )
    payload["paragraphs"][0]["evidence_refs"] = [
        section.required_evidence_refs[1]
    ]
    payload["paragraphs"][0]["derived_claims"] = [_derived_claim()]

    validation = validate_report_section(
        ReportSectionDraft.model_validate(payload),
        section,
        _manifest(),
    )

    assert "DERIVED_CLAIM_INVALID" in validation.error_codes


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


def test_invalid_section_can_converge_on_second_local_repair():
    plan = ReportPlan.model_validate(_plan_payload())
    failed_section = plan.sections[0]
    existing_drafts = {
        section.section_id: ReportSectionDraft.model_validate(_draft(section))
        for section in plan.sections[1:]
    }
    repair_calls = []

    def repair(_query, _plan, section, _manifest, _draft_value, error_codes):
        repair_calls.append((section.section_id, list(error_codes)))
        if len(repair_calls) == 1:
            return {
                **_draft(section),
                "paragraphs": [
                    {
                        "text": (
                            "Unsupported value was 999. "
                            + _words(section.target_words - 4)
                        ),
                        "evidence_refs": section.required_evidence_refs,
                    }
                ],
            }
        return _draft(section)

    drafts = generate_report_sections(
        "Explain the price trend.",
        plan,
        _manifest(),
        existing_drafts=existing_drafts,
        generate_section=lambda _q, _p, section, _m: {
            **_draft(section),
            "paragraphs": [
                {
                    "text": "Candidate remains too short.",
                    "evidence_refs": section.required_evidence_refs,
                }
            ],
        },
        repair_section=repair,
        max_repair_attempts=2,
        max_workers=1,
    )

    assert [call[0] for call in repair_calls] == [
        failed_section.section_id,
        failed_section.section_id,
    ]
    assert "WORD_COUNT_OUT_OF_RANGE" in repair_calls[0][1]
    assert repair_calls[1][1] == ["UNGROUNDED_NUMERIC_CLAIM"]
    assert drafts[0].section_id == failed_section.section_id


def test_invalid_section_stops_at_local_repair_bound():
    plan = ReportPlan.model_validate(_plan_payload())
    failed_section = plan.sections[0]
    existing_drafts = {
        section.section_id: ReportSectionDraft.model_validate(_draft(section))
        for section in plan.sections[1:]
    }
    repair_calls = []

    def invalid_repair(_query, _plan, section, _manifest, _draft_value, _errors):
        repair_calls.append(section.section_id)
        return {
            **_draft(section),
            "paragraphs": [
                {
                    "text": "Repair remains too short.",
                    "evidence_refs": section.required_evidence_refs,
                }
            ],
        }

    with pytest.raises(ReportSectionGenerationError):
        generate_report_sections(
            "Explain the price trend.",
            plan,
            _manifest(),
            existing_drafts=existing_drafts,
            generate_section=lambda _q, _p, section, _m: {
                **_draft(section),
                "paragraphs": [
                    {
                        "text": "Candidate remains too short.",
                        "evidence_refs": section.required_evidence_refs,
                    }
                ],
            },
            repair_section=invalid_repair,
            max_repair_attempts=2,
            max_workers=1,
        )

    assert repair_calls == [
        failed_section.section_id,
        failed_section.section_id,
    ]


@pytest.mark.parametrize("max_repair_attempts", [0, 4])
def test_local_section_repair_bound_is_validated(max_repair_attempts):
    with pytest.raises(ValueError, match="max_repair_attempts"):
        generate_report_sections(
            "Explain the price trend.",
            ReportPlan.model_validate(_plan_payload()),
            _manifest(),
            max_repair_attempts=max_repair_attempts,
        )


def test_failed_repair_aborts_with_typed_section_error(caplog):
    plan = ReportPlan.model_validate(_plan_payload())

    with caplog.at_level(logging.WARNING, logger="Enai.ReportSections"):
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
    assert '"section_id":' in caplog.text
    assert "WORD_COUNT_OUT_OF_RANGE" in caplog.text
    assert "This repaired section" not in caplog.text


def test_section_validation_diagnostics_are_structured_and_content_free(caplog):
    plan = ReportPlan.model_validate(_plan_payload())
    failed_section = plan.sections[0]
    existing_drafts = {
        section.section_id: ReportSectionDraft.model_validate(_draft(section))
        for section in plan.sections[1:]
    }

    with caplog.at_level(logging.INFO, logger="Enai.ReportSections"):
        with pytest.raises(ReportSectionGenerationError):
            generate_report_sections(
                "Sensitive request content.",
                plan,
                _manifest(),
                existing_drafts=existing_drafts,
                generate_section=lambda _q, _p, section, _m: {
                    **_draft(section),
                    "paragraphs": [
                        {
                            "text": "Sensitive candidate content.",
                            "evidence_refs": section.required_evidence_refs,
                        }
                    ],
                },
                repair_section=lambda _q, _p, section, _m, _d, _e: {
                    **_draft(section),
                    "paragraphs": [
                        {
                            "text": "Sensitive repaired content.",
                            "evidence_refs": section.required_evidence_refs,
                        }
                    ],
                },
                max_workers=1,
            )

    prefix = "REPORT_SECTION_DIAGNOSTIC "
    diagnostics = [
        json.loads(record.getMessage()[len(prefix):])
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]

    assert [item["event"] for item in diagnostics] == [
        "candidate_rejected",
        "repair_rejected",
        "repair_rejected",
    ]
    assert [item["attempt"] for item in diagnostics] == [1, 2, 3]
    assert all(item["section_id"] == failed_section.section_id for item in diagnostics)
    assert all(item["duration_ms"] >= 0 for item in diagnostics)
    assert all(item["target_words"] == failed_section.target_words for item in diagnostics)
    assert all(
        item["minimum_words"] <= item["maximum_words"]
        for item in diagnostics
    )
    assert "Sensitive request content" not in caplog.text
    assert "Sensitive candidate content" not in caplog.text
    assert "Sensitive repaired content" not in caplog.text


def test_provider_failure_during_repair_becomes_typed_section_error(caplog):
    plan = ReportPlan.model_validate(_plan_payload())

    def repair(*_args):
        raise ProviderExecutionError(
            "provider secret must not be logged",
            provider="nvidia",
            stage="report_section_repair",
            disposition=ProviderDeliveryDisposition.TIMED_OUT,
        )

    with caplog.at_level(logging.INFO, logger="Enai.ReportSections"):
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
                repair_section=repair,
                max_workers=len(plan.sections),
            )

    assert exc_info.value.error_codes == ["SECTION_REPAIR_PROVIDER_FAILED"]
    assert exc_info.value.provider == "nvidia"
    assert exc_info.value.provider_stage == "report_section_repair"
    assert exc_info.value.provider_disposition == "timed_out"
    assert '"event":"provider_failed"' in caplog.text
    assert '"attempt":2' in caplog.text
    assert '"provider":"nvidia"' in caplog.text
    assert '"provider_disposition":"timed_out"' in caplog.text
    assert "provider secret" not in caplog.text


def test_provider_failure_is_not_replayed_by_local_section_retries():
    plan = ReportPlan.model_validate(_plan_payload())
    existing_drafts = {
        section.section_id: ReportSectionDraft.model_validate(_draft(section))
        for section in plan.sections[1:]
    }
    repair_calls = []

    def repair(*_args):
        repair_calls.append(True)
        raise ProviderExecutionError(
            "ambiguous provider failure",
            provider="nvidia",
            stage="report_section_repair",
            disposition=ProviderDeliveryDisposition.AMBIGUOUS,
        )

    with pytest.raises(ReportSectionGenerationError) as exc_info:
        generate_report_sections(
            "Explain the price trend.",
            plan,
            _manifest(),
            existing_drafts=existing_drafts,
            generate_section=lambda _q, _p, section, _m: {
                **_draft(section),
                "paragraphs": [
                    {
                        "text": "Candidate remains too short.",
                        "evidence_refs": section.required_evidence_refs,
                    }
                ],
            },
            repair_section=repair,
            max_repair_attempts=3,
            max_workers=1,
        )

    assert repair_calls == [True]
    assert exc_info.value.error_codes == ["SECTION_REPAIR_PROVIDER_FAILED"]


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
