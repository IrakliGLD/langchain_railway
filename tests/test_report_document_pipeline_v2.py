"""Adaptive document planning, whole-draft validation, and repair tests."""

from __future__ import annotations

import json
import logging
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.report_charts import build_report_research_exhibits
from agent.report_document_generation import (
    ReportDocumentGenerationError,
    generate_report_document,
    validate_report_document,
)
from agent.report_document_planner import build_report_document_plan
from agent.report_evidence_gate import evaluate_report_evidence
from agent.report_research_execution import (
    consolidate_report_evidence_packets,
)
from agent.report_sections import count_section_words
from contracts.report import report_section_validation_word_bounds
from contracts.report_charts import ReportChartBuildDecision
from contracts.report_document import (
    ReportDocumentDraft,
    ReportDocumentRepair,
)
from contracts.report_evidence import ReportEvidenceKind
from contracts.report_sections import ReportSectionDraft
from tests.test_report_evidence_gate_v2 import _QUERY, _plan, _ready_packets


def _ready_components():
    research_plan = _plan()
    packets = _ready_packets()
    manifest = consolidate_report_evidence_packets(_QUERY, packets)
    decisions = build_report_research_exhibits(packets, manifest)
    gate = evaluate_report_evidence(
        research_plan,
        packets,
        chart_decisions=decisions,
    )
    return research_plan, packets, manifest, decisions, gate


def _document_components():
    research_plan, packets, manifest, decisions, gate = _ready_components()
    document_plan = build_report_document_plan(
        _QUERY,
        research_plan,
        packets,
        manifest,
        gate,
        decisions,
    )
    return (
        research_plan,
        packets,
        manifest,
        decisions,
        gate,
        document_plan,
    )


def _draft_section(
    section,
    manifest,
    *,
    repeated_text: str | None = None,
    max_numeric_claims: int = 2,
):
    item_by_ref = manifest.item_by_ref()
    word_target = section.target_words
    text_words = [
        f"{section.section_id}evidence"
        for _ in range(word_target)
    ]
    direct_claims = []
    claim_words = []
    for table_ref in (
        ref
        for ref in section.required_evidence_refs
        if item_by_ref[ref].kind is ReportEvidenceKind.TABLE
    ):
        table = item_by_ref[table_ref]
        numeric_cells = (
            (row_index, column, value)
            for row_index, row in enumerate(table.rows)
            for column, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
        for row_index, column, value in numeric_cells:
            unit = table.unit_by_column.get(column, "value")
            display = f"{value:g}"
            claim_words.extend(
                (
                    f"The observed value was {display} {unit} in the "
                    "cited data."
                ).split()
            )
            direct_claims.append(
                {
                    "evidence_ref": table_ref,
                    "row_index": row_index,
                    "column": column,
                    "display_value": display,
                    "unit": unit,
                }
            )
            if len(direct_claims) >= max_numeric_claims:
                break
        if len(direct_claims) >= max_numeric_claims:
            break
    text_words[: len(claim_words)] = claim_words
    text = repeated_text or " ".join(text_words)
    return {
        "contract_version": "report-section-v1",
        "section_id": section.section_id,
        "title": section.title,
        "paragraphs": [
            {
                "text": text,
                "evidence_refs": section.required_evidence_refs,
                "direct_claims": direct_claims,
                "derived_claims": [],
            }
        ],
    }


def _valid_document_draft(document_plan, manifest) -> ReportDocumentDraft:
    by_role = {
        role: [
            section
            for section in document_plan.sections
            if section.role.value == role
        ]
        for role in {
            "executive_summary",
            "analysis",
            "implications",
            "limitations",
            "conclusion",
        }
    }
    return ReportDocumentDraft.model_validate(
        {
            "contract_version": "report-document-draft-v1",
            "query_digest": document_plan.query_digest,
            "evidence_manifest_id": document_plan.evidence_manifest_id,
            "coverage_status": document_plan.coverage_status,
            "analytical_sections": [
                _draft_section(section, manifest)
                for section in by_role["analysis"]
            ],
            "implications_section": (
                _draft_section(by_role["implications"][0], manifest)
                if by_role["implications"]
                else None
            ),
            "limitations_section": _draft_section(
                by_role["limitations"][0],
                manifest,
            ),
            "conclusion_section": (
                _draft_section(by_role["conclusion"][0], manifest)
                if by_role["conclusion"]
                else None
            ),
            "executive_summary": _draft_section(
                by_role["executive_summary"][0],
                manifest,
            ),
        }
    )


def test_document_plan_is_deterministic_track_driven_and_nonrepetitive():
    research_plan, packets, manifest, decisions, gate = _ready_components()

    first = build_report_document_plan(
        _QUERY,
        research_plan,
        packets,
        manifest,
        gate,
        decisions,
    )
    second = build_report_document_plan(
        _QUERY,
        research_plan,
        packets,
        manifest,
        gate,
        decisions,
    )

    assert first == second
    assert [section.role.value for section in first.sections] == [
        "executive_summary",
        "analysis",
        "analysis",
        "analysis",
        "implications",
        "limitations",
    ]
    assert [
        section.section_id
        for section in first.sections
        if section.role.value == "analysis"
    ] == ["prices", "security", "market_model"]
    assert first.target_words == sum(
        section.target_words for section in first.sections
    )
    assert 900 <= first.target_words <= 1400
    assert {chart.chart_id for chart in first.charts} == {
        "prices_trend",
        "security_composition",
    }
    assert all(
        chart.section_id in {"prices", "security"}
        for chart in first.charts
    )


def test_document_plan_rejects_failed_evidence_gate():
    research_plan, packets, manifest, decisions, gate = _ready_components()
    failed_gate = gate.model_copy(
        update={
            "status": "failed",
            "finding_codes": ["NO_REQUIRED_EVIDENCE"],
        }
    )

    try:
        build_report_document_plan(
            _QUERY,
            research_plan,
            packets,
            manifest,
            failed_gate,
            decisions,
        )
    except ValueError as exc:
        assert "not ready" in str(exc)
    else:
        raise AssertionError("failed evidence gate was accepted")


def test_document_plan_preserves_an_unavailable_expected_exhibit():
    research_plan, packets, manifest, decisions, _ = _ready_components()
    omitted = decisions[0]
    decisions[0] = ReportChartBuildDecision(
        chart_id=omitted.chart_id,
        required=omitted.required,
        status="omitted",
        reason_code="REPORT_CHART_TIME_AXIS_REQUIRED",
        artifact=None,
    )
    gate = evaluate_report_evidence(
        research_plan,
        packets,
        chart_decisions=decisions,
    )

    plan = build_report_document_plan(
        _QUERY,
        research_plan,
        packets,
        manifest,
        gate,
        decisions,
    )

    request = next(
        chart
        for chart in plan.charts
        if chart.chart_id == omitted.chart_id
    )
    assert request.required is False
    assert omitted.chart_id in next(
        section.chart_refs
        for section in plan.sections
        if section.section_id == request.section_id
    )
    assert plan.coverage_status == "ready_with_gaps"


def test_whole_document_validation_catches_repetition_and_missing_numbers():
    research_plan, _, manifest, _, _, document_plan = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)

    valid = validate_report_document(
        valid_draft,
        document_plan,
        manifest,
        research_plan,
    )
    assert valid.valid is True

    payload = valid_draft.model_dump(mode="json")
    repeated = " ".join(
        f"repeatedword{chr(97 + (index % 26))}"
        f"{chr(97 + ((index // 26) % 26))}"
        for index in range(220)
    )
    payload["analytical_sections"][0]["paragraphs"][0]["text"] = repeated
    payload["analytical_sections"][0]["paragraphs"][0]["direct_claims"] = []
    payload["analytical_sections"][1]["paragraphs"][0]["text"] = repeated
    payload["analytical_sections"][1]["paragraphs"][0]["direct_claims"] = []
    invalid_draft = ReportDocumentDraft.model_validate(payload)

    invalid = validate_report_document(
        invalid_draft,
        document_plan,
        manifest,
        research_plan,
    )

    assert invalid.valid is False
    assert "CROSS_SECTION_REPETITION" in invalid.section_errors["security"]
    assert "NUMERIC_FINDING_MISSING" in invalid.section_errors["prices"]


def test_document_word_count_errors_are_directional():
    research_plan, _, manifest, _, _, document_plan = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    spec_by_id = {
        section.section_id: section for section in document_plan.sections
    }

    def section_payloads(payload):
        return [
            *payload["analytical_sections"],
            *(
                [payload["implications_section"]]
                if payload["implications_section"] is not None
                else []
            ),
            payload["limitations_section"],
            *(
                [payload["conclusion_section"]]
                if payload["conclusion_section"] is not None
                else []
            ),
            payload["executive_summary"],
        ]

    payload = valid_draft.model_dump(mode="json")
    for section_payload in section_payloads(payload):
        minimum_words, _ = report_section_validation_word_bounds(
            spec_by_id[section_payload["section_id"]].target_words
        )
        section_payload["paragraphs"][0]["text"] = " ".join(
            "shortword" for _ in range(minimum_words - 1)
        )
    short_validation = validate_report_document(
        ReportDocumentDraft.model_validate(payload),
        document_plan,
        manifest,
        research_plan,
    )

    payload = valid_draft.model_dump(mode="json")
    for section_payload in section_payloads(payload):
        _, maximum_words = report_section_validation_word_bounds(
            spec_by_id[section_payload["section_id"]].target_words
        )
        section_payload["paragraphs"][0]["text"] = " ".join(
            "longword" for _ in range(maximum_words + 1)
        )
    long_validation = validate_report_document(
        ReportDocumentDraft.model_validate(payload),
        document_plan,
        manifest,
        research_plan,
    )

    assert "WORD_COUNT_TOO_SHORT" in short_validation.section_errors[
        "prices"
    ]
    assert (
        "DOCUMENT_WORD_COUNT_TOO_SHORT"
        in short_validation.document_errors
    )
    assert "WORD_COUNT_TOO_LONG" in long_validation.section_errors["prices"]
    assert (
        "DOCUMENT_WORD_COUNT_TOO_LONG"
        in long_validation.document_errors
    )


def test_numeric_analysis_requires_two_findings_when_two_cells_are_available():
    research_plan, _, manifest, _, _, document_plan = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    price_spec = next(
        section
        for section in document_plan.sections
        if section.section_id == "prices"
    )
    payload = valid_draft.model_dump(mode="json")
    payload["analytical_sections"][0] = _draft_section(
        price_spec,
        manifest,
        max_numeric_claims=1,
    )

    validation = validate_report_document(
        ReportDocumentDraft.model_validate(payload),
        document_plan,
        manifest,
        research_plan,
    )

    assert "NUMERIC_FINDING_MISSING" in validation.section_errors["prices"]


def test_document_generation_normalizes_plan_owned_section_roles(caplog):
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    misplaced_payloads = []

    summary_swap = valid_draft.model_dump(mode="json")
    summary_swap["executive_summary"], summary_swap[
        "limitations_section"
    ] = (
        summary_swap["limitations_section"],
        summary_swap["executive_summary"],
    )
    misplaced_payloads.append(summary_swap)

    analysis_swap = valid_draft.model_dump(mode="json")
    analysis_swap["analytical_sections"][0], analysis_swap[
        "implications_section"
    ] = (
        analysis_swap["implications_section"],
        analysis_swap["analytical_sections"][0],
    )
    misplaced_payloads.append(analysis_swap)

    with caplog.at_level(logging.INFO, logger="Enai.ReportDocument"):
        for payload in misplaced_payloads:
            misplaced_draft = ReportDocumentDraft.model_validate(payload)
            assert "SECTION_ROLE_MISMATCH" in validate_report_document(
                misplaced_draft,
                document_plan,
                manifest,
                research_plan,
            ).document_errors

            normalized = generate_report_document(
                _QUERY,
                document_plan,
                research_plan,
                manifest,
                packets,
                write_document=lambda *_args, value=misplaced_draft: value,
                repair_sections=lambda *_args, **_kwargs: (
                    (_ for _ in ()).throw(
                        AssertionError(
                            "plan-owned role placement must not use repair"
                        )
                    )
                ),
            )

            assert normalized == valid_draft

    diagnostics = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("REPORT_DOCUMENT_DIAGNOSTIC ")
    ]
    normalized_events = [
        item for item in diagnostics if item["event"] == "roles_normalized"
    ]
    assert len(normalized_events) == 2
    assert all(
        item["role_normalization_applied"]
        and item["expected_role_section_ids"]
        != item["pre_normalization_role_section_ids"]
        for item in normalized_events
    )


def test_document_set_mismatch_repairs_the_complete_planned_set():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    payload = valid_draft.model_dump(mode="json")
    payload["analytical_sections"][0]["section_id"] = (
        "unexpected_section"
    )
    mismatched_draft = ReportDocumentDraft.model_validate(payload)
    expected_ids = [
        section.section_id for section in document_plan.sections
    ]
    repair_calls = []

    def repair_sections(
        _query,
        _plan,
        _research_plan,
        _manifest,
        _packets,
        _draft,
        _validation,
        *,
        section_ids,
    ):
        repair_calls.append(list(section_ids))
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=valid_draft.generation_order_sections(),
        )

    repaired = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_document=lambda *_args: mismatched_draft,
        repair_sections=repair_sections,
    )

    assert repair_calls == [expected_ids]
    assert repaired == valid_draft


def test_document_generation_repairs_only_invalid_sections_once():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    payload = valid_draft.model_dump(mode="json")
    repeated = " ".join(
        f"duplicateword{chr(97 + (index % 26))}"
        f"{chr(97 + ((index // 26) % 26))}"
        for index in range(220)
    )
    for index in (0, 1):
        payload["analytical_sections"][index]["paragraphs"][0]["text"] = repeated
        payload["analytical_sections"][index]["paragraphs"][0][
            "direct_claims"
        ] = []
    invalid_draft = ReportDocumentDraft.model_validate(payload)
    repair_calls = []

    def repair_sections(
        _query,
        _plan,
        _research_plan,
        _manifest,
        _packets,
        rejected,
        validation,
        *,
        section_ids,
    ):
        repair_calls.append(
            (
                rejected,
                validation,
                list(section_ids),
            )
        )
        replacements = []
        section_by_id = {
            section.section_id: section
            for section in document_plan.sections
        }
        for section_id in section_ids:
            replacements.append(
                _draft_section(section_by_id[section_id], manifest)
            )
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                ReportSectionDraft.model_validate(section)
                for section in replacements
            ],
        )

    repaired = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_document=lambda *_args: invalid_draft,
        repair_sections=repair_sections,
    )

    assert len(repair_calls) == 1
    assert set(repair_calls[0][2]) == {"prices", "security"}
    assert validate_report_document(
        repaired,
        document_plan,
        manifest,
        research_plan,
    ).valid


def test_document_overflow_repairs_only_the_section_above_its_hard_limit():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    payload = valid_draft.model_dump(mode="json")
    spec_by_id = {
        section.section_id: section for section in document_plan.sections
    }
    section_payloads = [
        *payload["analytical_sections"],
        payload["implications_section"],
        payload["limitations_section"],
        payload["executive_summary"],
    ]
    for section_index, section_payload in enumerate(section_payloads):
        section_id = section_payload["section_id"]
        _, maximum_words = report_section_validation_word_bounds(
            spec_by_id[section_id].target_words
        )
        current_words = count_section_words(
            section_payload["paragraphs"][0]["text"]
        )
        desired_words = maximum_words + (section_id == "prices")
        section_payload["paragraphs"][0]["text"] += " " + " ".join(
            f"f{section_index}"
            for _ in range(desired_words - current_words)
        )
    invalid_draft = ReportDocumentDraft.model_validate(payload)
    validation = validate_report_document(
        invalid_draft,
        document_plan,
        manifest,
        research_plan,
    )
    repair_calls = []

    def repair_sections(
        _query,
        _plan,
        _research_plan,
        _manifest,
        _packets,
        _rejected,
        _validation,
        *,
        section_ids,
    ):
        repair_calls.append(list(section_ids))
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[valid_draft.analytical_sections[0]],
        )

    repaired = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_document=lambda *_args: invalid_draft,
        repair_sections=repair_sections,
    )

    assert validation.document_errors == [
        "DOCUMENT_WORD_COUNT_TOO_LONG"
    ]
    assert set(validation.section_errors) == {"prices"}
    assert repair_calls == [["prices"]]
    assert validate_report_document(
        repaired,
        document_plan,
        manifest,
        research_plan,
    ).valid


def test_document_rejects_a_word_repair_that_moves_the_wrong_direction(
    caplog,
):
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    price_spec = next(
        section
        for section in document_plan.sections
        if section.section_id == "prices"
    )
    _, maximum_words = report_section_validation_word_bounds(
        price_spec.target_words
    )

    invalid_payload = valid_draft.model_dump(mode="json")
    invalid_paragraph = invalid_payload["analytical_sections"][0][
        "paragraphs"
    ][0]
    invalid_word_count = count_section_words(invalid_paragraph["text"])
    invalid_paragraph["text"] += " " + " ".join(
        "overflow"
        for _ in range(maximum_words + 1 - invalid_word_count)
    )
    invalid_draft = ReportDocumentDraft.model_validate(invalid_payload)
    initial_validation = validate_report_document(
        invalid_draft,
        document_plan,
        manifest,
        research_plan,
    )

    worse_payload = valid_draft.analytical_sections[0].model_dump(mode="json")
    worse_paragraph = worse_payload["paragraphs"][0]
    worse_word_count = count_section_words(worse_paragraph["text"])
    worse_paragraph["text"] += " " + " ".join(
        "overflow"
        for _ in range(maximum_words + 2 - worse_word_count)
    )
    worse_section = ReportSectionDraft.model_validate(worse_payload)

    with caplog.at_level(logging.INFO, logger="Enai.ReportDocument"):
        try:
            generate_report_document(
                _QUERY,
                document_plan,
                research_plan,
                manifest,
                packets,
                write_document=lambda *_args: invalid_draft,
                repair_sections=lambda *_args, **_kwargs: (
                    ReportDocumentRepair(
                        contract_version="report-document-repair-v1",
                        sections=[worse_section],
                    )
                ),
            )
        except ReportDocumentGenerationError as exc:
            assert exc.validation.word_count == initial_validation.word_count
        else:
            raise AssertionError("a length-increasing repair was accepted")

    assert '"event":"repair_direction_rejected"' in caplog.text


def test_document_repair_diagnostics_include_safe_counts_and_bounds(caplog):
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    price_spec = next(
        section
        for section in document_plan.sections
        if section.section_id == "prices"
    )
    _, maximum_words = report_section_validation_word_bounds(
        price_spec.target_words
    )
    payload = valid_draft.model_dump(mode="json")
    price_paragraph = payload["analytical_sections"][0]["paragraphs"][0]
    price_words = count_section_words(price_paragraph["text"])
    price_paragraph["text"] += " " + " ".join(
        "sensitiveoverflow"
        for _ in range(maximum_words + 1 - price_words)
    )
    invalid_draft = ReportDocumentDraft.model_validate(payload)

    with caplog.at_level(
        logging.INFO,
        logger="Enai.ReportDocument",
    ):
        generate_report_document(
            _QUERY,
            document_plan,
            research_plan,
            manifest,
            packets,
            write_document=lambda *_args: invalid_draft,
            repair_sections=lambda *_args, **_kwargs: ReportDocumentRepair(
                contract_version="report-document-repair-v1",
                sections=[valid_draft.analytical_sections[0]],
            ),
        )

    diagnostics = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("REPORT_DOCUMENT_DIAGNOSTIC ")
    ]
    requested = next(
        item
        for item in diagnostics
        if item["event"] == "repair_requested"
    )
    validated = next(
        item
        for item in diagnostics
        if item["event"] == "repair_validated"
    )
    assert requested["word_count"] > 0
    assert requested["minimum_words"] < requested["maximum_words"]
    assert requested["failing_section_ids"] == ["prices"]
    assert requested["repair_section_ids"] == ["prices"]
    assert requested["section_word_counts"]["prices"] == maximum_words + 1
    assert requested["section_bounds"]["prices"]["maximum_words"] == (
        maximum_words
    )
    assert validated["word_count"] < requested["word_count"]
    assert validated["section_word_deltas"]["prices"] < 0
    assert "sensitiveoverflow" not in caplog.text


def test_document_generation_repairs_a_schema_invalid_draft_once():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    invalid_payload = valid_draft.model_dump(mode="json")
    invalid_payload.pop("analytical_sections")
    repair_calls = []

    def repair_sections(
        _query,
        _plan,
        _research_plan,
        _manifest,
        _packets,
        rejected,
        validation,
        *,
        section_ids,
    ):
        repair_calls.append((rejected, validation, list(section_ids)))
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=valid_draft.generation_order_sections(),
        )

    repaired = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_document=lambda *_args: invalid_payload,
        repair_sections=repair_sections,
    )

    assert repaired == valid_draft
    assert len(repair_calls) == 1
    assert repair_calls[0][0] is invalid_payload
    assert repair_calls[0][1].document_errors == [
        "DOCUMENT_SCHEMA_INVALID"
    ]
    assert repair_calls[0][2] == [
        section.section_id for section in document_plan.sections
    ]


def test_document_generation_fails_after_one_invalid_repair():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    payload = valid_draft.model_dump(mode="json")
    payload["analytical_sections"][0]["paragraphs"][0]["direct_claims"] = []
    payload["analytical_sections"][0]["paragraphs"][0]["text"] = " ".join(
        "nonnumericword" for _ in range(300)
    )
    invalid_draft = ReportDocumentDraft.model_validate(payload)
    calls = []

    def invalid_repair(*_args, section_ids, **_kwargs):
        calls.append(list(section_ids))
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[invalid_draft.analytical_sections[0]],
        )

    try:
        generate_report_document(
            _QUERY,
            document_plan,
            research_plan,
            manifest,
            packets,
            write_document=lambda *_args: invalid_draft,
            repair_sections=invalid_repair,
        )
    except ReportDocumentGenerationError:
        pass
    else:
        raise AssertionError("invalid targeted repair was accepted")

    assert len(calls) == 1


def test_document_generation_respects_a_two_call_budget_without_repair():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    invalid_draft = _valid_document_draft(
        document_plan,
        manifest,
    ).model_copy(
        update={
            "analytical_sections": [
                section.model_copy(
                    update={
                        "paragraphs": [
                            paragraph.model_copy(
                                update={"direct_claims": []}
                            )
                            for paragraph in section.paragraphs
                        ]
                    }
                )
                if section.section_id == "prices"
                else section
                for section in _valid_document_draft(
                    document_plan,
                    manifest,
                ).analytical_sections
            ]
        }
    )

    try:
        generate_report_document(
            _QUERY,
            document_plan,
            research_plan,
            manifest,
            packets,
            write_document=lambda *_args: invalid_draft,
            repair_sections=lambda *_args, **_kwargs: (
                (_ for _ in ()).throw(
                    AssertionError("repair exceeds the call budget")
                )
            ),
            allow_repair=False,
        )
    except ReportDocumentGenerationError:
        pass
    else:
        raise AssertionError("invalid unrepairable draft was accepted")
