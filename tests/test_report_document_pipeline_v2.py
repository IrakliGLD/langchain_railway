"""Adaptive document planning, whole-draft validation, and repair tests."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

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
from agent.report_document_planner import (
    allocate_report_word_targets,
    assess_report_evidence_capacity,
    build_report_document_plan,
    classify_report_document_profile,
)
from agent.report_evidence_gate import evaluate_report_evidence
from agent.report_research_execution import (
    consolidate_report_evidence_packets,
)
from agent.report_sections import count_section_words
from contracts.report import report_section_validation_word_bounds
from contracts.report_charts import ReportChartBuildDecision
from contracts.report_document import (
    ReportDocumentDraft,
    ReportDocumentProfile,
    ReportDocumentRepair,
    ReportDocumentSectionRole,
    ReportEvidenceCapacity,
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


def test_evidence_capacity_profiles_follow_golden_architecture_cases():
    cases = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "report_architecture_cases.json"
        ).read_text(encoding="utf-8")
    )["cases"]

    for case in cases:
        assert classify_report_document_profile(
            usable_track_count=case["usable_tracks"],
            usable_exhibit_count=case["usable_exhibits"],
            validated_finding_count=case["validated_findings"],
        ).value == case["expected_profile"]


def test_report_wide_narrative_evidence_is_assigned_to_analysis_sections():
    """Manifest-level statistics and knowledge must reach the writer.

    A section's evidence comes from its research packets, but the standard
    pipeline's computed statistics and curated knowledge are report-wide and
    belong to no packet. Without an explicit assignment they sit in the
    manifest uncited, so their facts never ground a sentence and the writer
    cannot use them at all.
    """

    from agent.report_evidence import make_report_narrative_evidence_item
    from agent.report_research_execution import (
        consolidate_report_evidence_packets,
    )

    research_plan, packets, _, decisions, gate = _ready_components()
    statistics = make_report_narrative_evidence_item(
        kind=ReportEvidenceKind.STATISTICS,
        title="Verified statistics",
        source="derived",
        content="Observed mean balancing price was 141.0 GEL/MWh.",
    )
    knowledge = make_report_narrative_evidence_item(
        kind=ReportEvidenceKind.KNOWLEDGE,
        title="Curated domain knowledge",
        source="curated_knowledge",
        content="The balancing market settles hourly.",
    )
    manifest = consolidate_report_evidence_packets(
        _QUERY,
        packets,
        extra_items=[statistics, knowledge],
    )

    plan = build_report_document_plan(
        _QUERY,
        research_plan,
        packets,
        manifest,
        gate,
        decisions,
    )

    analysis_sections = [
        section
        for section in plan.sections
        if section.role is ReportDocumentSectionRole.ANALYSIS
    ]
    assert analysis_sections
    for section in analysis_sections:
        assert statistics.evidence_ref in section.required_evidence_refs
        assert knowledge.evidence_ref in section.required_evidence_refs


def test_track_owned_narrative_evidence_is_not_broadcast_to_other_tracks():
    from agent.report_evidence import make_report_narrative_evidence_item
    from contracts.report_research import ReportEvidencePacket

    research_plan, packets, _, decisions, gate = _ready_components()
    statistics = make_report_narrative_evidence_item(
        kind=ReportEvidenceKind.STATISTICS,
        title="Track verified statistics",
        source="derived",
        content="This finding belongs only to the first research track.",
    )
    first_payload = packets[0].model_dump(mode="json")
    first_payload["items"].append(statistics.model_dump(mode="json"))
    first_payload["observations"].append(
        {
            "observation_id": "documented_track_statistic",
            "statement": (
                "Approved statistics evidence was retrieved for this track."
            ),
            "evidence_refs": [statistics.evidence_ref],
            "metric_values": [],
        }
    )
    packets = [
        ReportEvidencePacket.model_validate(first_payload),
        *packets[1:],
    ]
    manifest = consolidate_report_evidence_packets(_QUERY, packets)

    plan = build_report_document_plan(
        _QUERY,
        research_plan,
        packets,
        manifest,
        gate,
        decisions,
    )

    owner_track_id = packets[0].track_id
    owner_sections = [
        section
        for section in plan.sections
        if owner_track_id in section.track_ids
    ]
    other_sections = [
        section
        for section in plan.sections
        if (
            section.role is ReportDocumentSectionRole.ANALYSIS
            and owner_track_id not in section.track_ids
        )
    ]
    assert owner_sections
    assert other_sections
    assert all(
        statistics.evidence_ref in section.required_evidence_refs
        for section in owner_sections
        if section.role is ReportDocumentSectionRole.ANALYSIS
    )
    assert all(
        statistics.evidence_ref not in section.required_evidence_refs
        for section in other_sections
    )


def test_document_plan_profile_uses_collected_evidence_not_planned_exhibits():
    research_plan, packets, manifest, decisions, gate = _ready_components()
    capacity = assess_report_evidence_capacity(
        packets,
        gate,
        decisions,
    )
    plan = build_report_document_plan(
        _QUERY,
        research_plan,
        packets,
        manifest,
        gate,
        decisions,
    )

    assert capacity.usable_track_count == 3
    assert capacity.usable_exhibit_count == 2
    assert capacity.profile is ReportDocumentProfile.FULL
    assert plan.profile is capacity.profile
    assert plan.evidence_capacity == capacity


def test_word_targets_scale_with_evidence_capacity_without_padding():
    compact = allocate_report_word_targets(
        ReportEvidenceCapacity(
            profile="compact",
            usable_track_count=1,
            complete_track_count=1,
            partial_track_count=0,
            unavailable_track_count=0,
            usable_exhibit_count=1,
            validated_finding_count=2,
        ),
        analysis_count=1,
        include_implications=False,
    )
    focused = allocate_report_word_targets(
        ReportEvidenceCapacity(
            profile="focused",
            usable_track_count=2,
            complete_track_count=2,
            partial_track_count=0,
            unavailable_track_count=0,
            usable_exhibit_count=2,
            validated_finding_count=5,
        ),
        analysis_count=2,
        include_implications=True,
    )
    full = allocate_report_word_targets(
        ReportEvidenceCapacity(
            profile="full",
            usable_track_count=4,
            complete_track_count=3,
            partial_track_count=1,
            unavailable_track_count=0,
            usable_exhibit_count=4,
            validated_finding_count=10,
        ),
        analysis_count=4,
        include_implications=True,
    )

    assert 300 <= compact[0] < focused[0] < full[0] <= 1300
    for (
        target_words,
        analysis_words,
        implication_words,
        limitation_words,
    ) in (
        compact,
        focused,
        full,
    ):
        assert target_words == (
            sum(analysis_words) + implication_words + limitation_words
        )
        assert all(words >= 40 for words in analysis_words)


def _draft_section(
    section,
    manifest,
    *,
    repeated_text: str | None = None,
    max_numeric_claims: int = 2,
):
    item_by_ref = manifest.item_by_ref()
    word_target = section.target_words
    section_token = f"{section.section_id[:4]}evidence"
    text_words = [
        section_token
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
    return ReportDocumentDraft.model_validate(
        {
            "contract_version": "report-document-draft-v1",
            "query_digest": document_plan.query_digest,
            "evidence_manifest_id": document_plan.evidence_manifest_id,
            "coverage_status": document_plan.coverage_status,
            "sections": [
                _draft_section(section, manifest)
                for section in document_plan.sections
            ],
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
    payload["sections"][0]["paragraphs"][0]["text"] = repeated
    payload["sections"][0]["paragraphs"][0]["direct_claims"] = []
    payload["sections"][1]["paragraphs"][0]["text"] = repeated
    payload["sections"][1]["paragraphs"][0]["direct_claims"] = []
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


def test_document_word_count_policy_blocks_short_and_warns_on_long():
    research_plan, _, manifest, _, _, document_plan = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    spec_by_id = {
        section.section_id: section for section in document_plan.sections
    }

    def section_payloads(payload):
        return payload["sections"]

    payload = valid_draft.model_dump(mode="json")
    for section_payload in section_payloads(payload):
        section_spec = spec_by_id[section_payload["section_id"]]
        desired_words = max(40, int(section_spec.target_words * 0.4))
        current_words = section_payload["paragraphs"][0]["text"].split()
        section_payload["paragraphs"][0]["text"] = " ".join(
            current_words[:desired_words]
        )
    short_validation = validate_report_document(
        ReportDocumentDraft.model_validate(payload),
        document_plan,
        manifest,
        research_plan,
    )

    payload = valid_draft.model_dump(mode="json")
    for section_index, section_payload in enumerate(section_payloads(payload)):
        _, maximum_words = report_section_validation_word_bounds(
            spec_by_id[section_payload["section_id"]].target_words
        )
        paragraph = section_payload["paragraphs"][0]
        current_words = count_section_words(paragraph["text"])
        paragraph["text"] += " " + " ".join(
            f"longword{section_index}"
            for _ in range(maximum_words + 1 - current_words)
        )
    long_validation = validate_report_document(
        ReportDocumentDraft.model_validate(payload),
        document_plan,
        manifest,
        research_plan,
    )

    assert short_validation.valid is False
    assert "WORD_COUNT_TOO_SHORT" in short_validation.section_errors[
        "prices"
    ]
    assert (
        "DOCUMENT_WORD_COUNT_TOO_SHORT"
        in short_validation.document_errors
    )
    assert long_validation.valid is True
    assert "WORD_COUNT_TOO_LONG" in long_validation.section_warnings["prices"]
    assert (
        "DOCUMENT_WORD_COUNT_TOO_LONG"
        in long_validation.document_warnings
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
    payload["sections"][0] = _draft_section(
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


def test_full_document_generation_writes_analysis_before_synthesis():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section_by_id = {
        section.section_id: section for section in valid_draft.sections
    }
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
    calls = []

    def write_analysis(*_args, section_ids):
        calls.append(("analysis", list(section_ids)))
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[section_by_id[section_id] for section_id in section_ids],
        )

    def write_synthesis(*_args, analysis_sections, section_ids):
        calls.append(
            (
                "synthesis",
                [section.section_id for section in analysis_sections],
                list(section_ids),
            )
        )
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[section_by_id[section_id] for section_id in section_ids],
        )

    generated = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_analysis_sections=write_analysis,
        write_synthesis_sections=write_synthesis,
    )

    assert generated == valid_draft
    assert calls == [
        ("analysis", analysis_ids),
        ("synthesis", analysis_ids, synthesis_ids),
    ]


def test_compact_document_generation_uses_one_document_writer():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    compact_plan = document_plan.model_copy(
        update={"profile": ReportDocumentProfile.COMPACT}
    )
    valid_draft = _valid_document_draft(compact_plan, manifest)
    calls = []

    generated = generate_report_document(
        _QUERY,
        compact_plan,
        research_plan,
        manifest,
        packets,
        write_document=lambda *_args: calls.append("document") or valid_draft,
        write_analysis_sections=lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(
                AssertionError("compact report must not split analysis")
            )
        ),
        write_synthesis_sections=lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(
                AssertionError("compact report must not split synthesis")
            )
        ),
    )

    assert generated == valid_draft
    assert calls == ["document"]


def test_invalid_analysis_batch_stops_before_synthesis():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section_by_id = {
        section.section_id: section for section in valid_draft.sections
    }
    analysis_ids = [
        section.section_id
        for section in document_plan.sections
        if section.role.value == "analysis"
    ]
    invalid_analysis = section_by_id[analysis_ids[0]].model_copy(
        update={"title": "Wrong title"}
    )

    def write_analysis(*_args, section_ids):
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                invalid_analysis
                if section_id == invalid_analysis.section_id
                else section_by_id[section_id]
                for section_id in section_ids
            ],
        )

    try:
        generate_report_document(
            _QUERY,
            document_plan,
            research_plan,
            manifest,
            packets,
            write_analysis_sections=write_analysis,
            write_synthesis_sections=lambda *_args, **_kwargs: (
                (_ for _ in ()).throw(
                    AssertionError(
                        "synthesis must not consume invalid analysis"
                    )
                )
            ),
            allow_repair=False,
        )
    except ReportDocumentGenerationError as exc:
        assert exc.validation.section_errors[
            invalid_analysis.section_id
        ] == ["SECTION_TITLE_MISMATCH"]
    else:
        raise AssertionError("invalid analysis batch was accepted")


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

    limitations_swap = valid_draft.model_dump(mode="json")
    limitations_swap["sections"][0], limitations_swap["sections"][-1] = (
        limitations_swap["sections"][-1],
        limitations_swap["sections"][0],
    )
    misplaced_payloads.append(limitations_swap)

    analysis_swap = valid_draft.model_dump(mode="json")
    analysis_swap["sections"][0], analysis_swap["sections"][3] = (
        analysis_swap["sections"][3],
        analysis_swap["sections"][0],
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
    payload["sections"][0]["section_id"] = (
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
        attempt_number,
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
        payload["sections"][index]["paragraphs"][0]["text"] = repeated
        payload["sections"][index]["paragraphs"][0][
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
        attempt_number,
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


def test_document_upper_length_warning_does_not_trigger_repair():
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
    price_paragraph = payload["sections"][0]["paragraphs"][0]
    price_words = count_section_words(price_paragraph["text"])
    price_paragraph["text"] += " " + " ".join(
        "overflow"
        for _ in range(maximum_words + 1 - price_words)
    )
    long_draft = ReportDocumentDraft.model_validate(payload)
    generated = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_document=lambda *_args: long_draft,
        repair_sections=lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(
                AssertionError("word-count warnings must not trigger repair")
            )
        ),
    )

    validation = validate_report_document(
        generated,
        document_plan,
        manifest,
        research_plan,
    )
    assert validation.valid is True
    assert validation.section_warnings["prices"] == [
        "WORD_COUNT_TOO_LONG"
    ]


def test_document_short_section_is_repaired_to_its_evidence_aware_floor():
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
    price_section = next(
        section
        for section in payload["sections"]
        if section["section_id"] == "prices"
    )
    price_paragraph = price_section["paragraphs"][0]
    price_paragraph["text"] = " ".join(
        price_paragraph["text"].split()[:50]
    )
    short_draft = ReportDocumentDraft.model_validate(payload)
    repair_calls = []

    def repair_sections(
        _query,
        _plan,
        _research_plan,
        _manifest,
        _packets,
        _rejected,
        validation,
        *,
        section_ids,
        attempt_number,
    ):
        repair_calls.append((validation, list(section_ids)))
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                section
                for section in valid_draft.sections
                if section.section_id in section_ids
            ],
        )

    repaired = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_document=lambda *_args: short_draft,
        repair_sections=repair_sections,
    )

    assert repair_calls[0][1] == ["prices"]
    assert repair_calls[0][0].section_errors["prices"] == [
        "WORD_COUNT_TOO_SHORT"
    ]
    assert validate_report_document(
        repaired,
        document_plan,
        manifest,
        research_plan,
    ).valid


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
    invalid_payload.pop("sections")
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
        attempt_number,
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
    payload["sections"][0]["paragraphs"][0]["direct_claims"] = []
    invalid_draft = ReportDocumentDraft.model_validate(payload)
    invalid_replacement = valid_draft.sections[0].model_copy(
        update={"title": "Wrong repaired title"}
    )
    calls = []

    def invalid_repair(*_args, section_ids, **_kwargs):
        calls.append(list(section_ids))
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[invalid_replacement],
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
    except ReportDocumentGenerationError as exc:
        assert exc.validation.section_errors["prices"] == [
            "SECTION_TITLE_MISMATCH"
        ]
        assert "NUMERIC_FINDING_MISSING" not in exc.validation.section_errors[
            "prices"
        ]
    else:
        raise AssertionError("invalid targeted repair was accepted")

    assert len(calls) == 1


def test_document_generation_uses_a_second_repair_when_budgeted():
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
    payload["sections"][0]["paragraphs"][0]["direct_claims"] = []
    invalid_draft = ReportDocumentDraft.model_validate(payload)
    calls = []
    attempt_numbers = []

    def repair(*_args, section_ids, **kwargs):
        calls.append(list(section_ids))
        attempt_numbers.append(kwargs["attempt_number"])
        if len(calls) == 1:
            replacement = valid_draft.sections[0].model_copy(
                update={"title": "Still invalid"}
            )
        else:
            replacement = valid_draft.sections[0]
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[replacement],
        )

    repaired = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_document=lambda *_args: invalid_draft,
        repair_sections=repair,
        max_repair_attempts=2,
    )

    assert calls == [["prices"], ["prices"]]
    assert attempt_numbers == [2, 3]
    assert repaired == valid_draft


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
            "sections": [
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
                ).sections
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


def _ungrounded_section(section):
    return section.model_copy(
        update={
            "paragraphs": [
                paragraph.model_copy(update={"direct_claims": []})
                for paragraph in section.paragraphs
            ]
        }
    )


def test_invalid_analysis_batch_is_repaired_before_synthesis():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section_by_id = {
        section.section_id: section for section in valid_draft.sections
    }
    analysis_ids = [
        section.section_id
        for section in document_plan.sections
        if section.role.value == "analysis"
    ]
    calls = []

    def write_analysis(*_args, section_ids):
        calls.append("analysis")
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                _ungrounded_section(section_by_id[section_id])
                if index == 0
                else section_by_id[section_id]
                for index, section_id in enumerate(section_ids)
            ],
        )

    def write_synthesis(*_args, analysis_sections, section_ids):
        calls.append("synthesis")
        assert list(analysis_sections) == [
            section_by_id[section_id] for section_id in analysis_ids
        ], "synthesis must receive the repaired analysis sections"
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[section_by_id[sid] for sid in section_ids],
        )

    repaired_ids = []

    def repair_sections(*_args, section_ids, **_kwargs):
        calls.append("repair")
        repaired_ids.append(list(section_ids))
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[section_by_id[sid] for sid in section_ids],
        )

    generated = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_analysis_sections=write_analysis,
        write_synthesis_sections=write_synthesis,
        repair_sections=repair_sections,
        allow_repair=True,
    )

    assert calls == ["analysis", "repair", "synthesis"]
    assert repaired_ids == [["prices"]]
    assert generated == valid_draft


def test_invalid_analysis_batch_still_fails_when_repair_is_not_allowed():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section_by_id = {
        section.section_id: section for section in valid_draft.sections
    }

    def write_analysis(*_args, section_ids):
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                _ungrounded_section(section_by_id[section_id])
                for section_id in section_ids
            ],
        )

    try:
        generate_report_document(
            _QUERY,
            document_plan,
            research_plan,
            manifest,
            packets,
            write_analysis_sections=write_analysis,
            write_synthesis_sections=lambda *_a, **_k: (
                (_ for _ in ()).throw(
                    AssertionError("synthesis must not run")
                )
            ),
            repair_sections=lambda *_a, **_k: (
                (_ for _ in ()).throw(
                    AssertionError("repair exceeds the call budget")
                )
            ),
            allow_repair=False,
        )
    except ReportDocumentGenerationError as exc:
        assert "UNGROUNDED_NUMERIC_CLAIM" in exc.validation.section_errors[
            "prices"
        ]
    else:
        raise AssertionError("invalid unrepairable batch was accepted")


def test_invalid_synthesis_batch_is_repaired():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section_by_id = {
        section.section_id: section for section in valid_draft.sections
    }
    calls = []

    def write_analysis(*_args, section_ids):
        calls.append("analysis")
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[section_by_id[sid] for sid in section_ids],
        )

    def write_synthesis(*_args, analysis_sections, section_ids):
        calls.append("synthesis")
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                _ungrounded_section(section_by_id[section_id])
                if index == 0
                else section_by_id[section_id]
                for index, section_id in enumerate(section_ids)
            ],
        )

    def repair_sections(*_args, section_ids, **_kwargs):
        calls.append("repair")
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[section_by_id[sid] for sid in section_ids],
        )

    generated = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_analysis_sections=write_analysis,
        write_synthesis_sections=write_synthesis,
        repair_sections=repair_sections,
        allow_repair=True,
    )

    assert calls == ["analysis", "synthesis", "repair"]
    assert generated == valid_draft


def test_analysis_section_fails_when_two_repairs_leave_a_stray_number():
    """The production shape: repair does not converge on one stray figure.

    Job 33403df2 died with DERIVED_CLAIM_NOT_USED,UNGROUNDED_NUMERIC_CLAIM
    after its single repair call, and the reader got nothing. The verified
    claims around the stray value are still publishable.
    """

    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section_by_id = {
        section.section_id: section for section in valid_draft.sections
    }

    def with_stray_number(section):
        first, *rest = section.paragraphs
        return section.model_copy(
            update={
                "paragraphs": [
                    first.model_copy(
                        update={
                            "text": (
                                first.text
                                + " A stray reading reached 987.6 GEL/MWh."
                            )
                        }
                    ),
                    *rest,
                ]
            }
        )

    # Production job 33403df2 had a single analysis section, and only it
    # carried the stray value.
    poisoned_id = "prices"

    def batch(section_ids):
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                with_stray_number(section_by_id[section_id])
                if section_id == poisoned_id
                else section_by_id[section_id]
                for section_id in section_ids
            ],
        )

    def write_analysis(*_args, section_ids):
        return batch(section_ids)

    def write_synthesis(*_args, analysis_sections, section_ids):
        return batch(section_ids)

    repair_calls = []

    def unconverged_repair(*_args, section_ids, **_kwargs):
        # The repairer returns the same stray value, exactly as in production.
        repair_calls.append(list(section_ids))
        return batch(section_ids)

    with pytest.raises(ReportDocumentGenerationError):
        generate_report_document(
            _QUERY,
            document_plan,
            research_plan,
            manifest,
            packets,
            write_analysis_sections=write_analysis,
            write_synthesis_sections=write_synthesis,
            repair_sections=unconverged_repair,
            allow_repair=True,
            max_repair_attempts=2,
        )

    assert repair_calls == [[poisoned_id], [poisoned_id]]


def test_batch_path_uses_two_budgeted_repairs_across_writer_batches():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section_by_id = {
        section.section_id: section for section in valid_draft.sections
    }

    def write_analysis(*_args, section_ids):
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                _ungrounded_section(section_by_id[section_id])
                for section_id in section_ids
            ],
        )

    def write_synthesis(*_args, analysis_sections, section_ids):
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                _ungrounded_section(section_by_id[section_id])
                for section_id in section_ids
            ],
        )

    repair_calls = []
    attempt_numbers = []

    def repair_sections(*_args, section_ids, **kwargs):
        repair_calls.append(list(section_ids))
        attempt_numbers.append(kwargs["attempt_number"])
        # The two-call repair budget can repair analysis and synthesis once.
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[section_by_id[sid] for sid in section_ids],
        )

    generated = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_analysis_sections=write_analysis,
        write_synthesis_sections=write_synthesis,
        repair_sections=repair_sections,
        allow_repair=True,
        max_repair_attempts=2,
    )

    assert len(repair_calls) == 2
    assert attempt_numbers == [2, 3]
    assert generated == valid_draft


def test_batch_repair_returning_the_wrong_section_set_is_rejected():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section_by_id = {
        section.section_id: section for section in valid_draft.sections
    }
    synthesis_ids = [
        section.section_id
        for section in document_plan.sections
        if section.role.value != "analysis"
    ]

    def write_analysis(*_args, section_ids):
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                _ungrounded_section(section_by_id[section_id])
                for section_id in section_ids
            ],
        )

    def repair_sections(*_args, section_ids, **_kwargs):
        # Answers with synthesis sections instead of the rejected analysis set.
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[section_by_id[sid] for sid in synthesis_ids],
        )

    try:
        generate_report_document(
            _QUERY,
            document_plan,
            research_plan,
            manifest,
            packets,
            write_analysis_sections=write_analysis,
            write_synthesis_sections=lambda *_a, **_k: (
                (_ for _ in ()).throw(
                    AssertionError("synthesis must not run")
                )
            ),
            repair_sections=repair_sections,
            allow_repair=True,
        )
    except ReportDocumentGenerationError as exc:
        assert "UNGROUNDED_NUMERIC_CLAIM" in exc.validation.section_errors[
            "prices"
        ]
    else:
        raise AssertionError("mismatched batch repair was accepted")


def test_schema_invalid_analysis_batch_repairs_from_the_raw_payload():
    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section_by_id = {
        section.section_id: section for section in valid_draft.sections
    }
    raw_payload = {"contract_version": "report-document-repair-v1"}
    rejected_inputs = []

    def write_analysis(*_args, section_ids):
        return raw_payload

    def write_synthesis(*_args, analysis_sections, section_ids):
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[section_by_id[sid] for sid in section_ids],
        )

    def repair_sections(
        _query,
        _plan,
        _research_plan,
        _manifest,
        _packets,
        rejected,
        _validation,
        *,
        section_ids,
        attempt_number,
    ):
        rejected_inputs.append(rejected)
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[section_by_id[sid] for sid in section_ids],
        )

    generated = generate_report_document(
        _QUERY,
        document_plan,
        research_plan,
        manifest,
        packets,
        write_analysis_sections=write_analysis,
        write_synthesis_sections=write_synthesis,
        repair_sections=repair_sections,
        allow_repair=True,
    )

    assert rejected_inputs == [raw_payload]
    assert generated == valid_draft


def _numberless_section(section):
    """A section with no claims and no numbers — clean to the section gate."""
    return section.model_copy(
        update={
            "paragraphs": [
                paragraph.model_copy(
                    update={
                        # Distinct per paragraph: ReportSectionDraft requires
                        # unique texts, and a duplicate would fail on schema
                        # rather than on the numeric floor under test.
                        # Number-free: a digit anywhere in the prose — even a
                        # paragraph index — reads as an ungrounded claim.
                        "text": (
                            "abcdefghij"[index % 10]
                            + " "
                            + " ".join(["evidence"] * 120)
                        ),
                        "direct_claims": [],
                        "derived_claims": [],
                    }
                )
                for index, paragraph in enumerate(section.paragraphs)
            ]
        }
    )


def test_materialization_reports_a_numberless_analysis_section():
    """The accepting gate must see what the publishing gate will reject.

    Job 522b9b73: the writer's sections failed grounding, the one repair call
    was spent clearing UNGROUNDED_NUMERIC_CLAIM by deleting the numbers, and
    the document gate then rejected NUMERIC_FINDING_MISSING with no budget
    left. Reporting the shortfall here lets a single repair address both.
    """
    from agent import report_document_generation as generation

    (
        research_plan,
        _packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section_by_id = {
        section.section_id: section for section in valid_draft.sections
    }
    analysis_ids = [
        section.section_id
        for section in document_plan.sections
        if section.role is ReportDocumentSectionRole.ANALYSIS
    ]
    batch = ReportDocumentRepair(
        contract_version="report-document-repair-v1",
        sections=[
            _numberless_section(section_by_id[section_id])
            for section_id in analysis_ids
        ],
    )

    _sections, validation = generation._materialize_section_batch(
        batch,
        document_plan,
        manifest,
        section_ids=analysis_ids,
        research_plan=research_plan,
    )

    assert validation.valid is False
    flagged = {
        section_id
        for section_id, codes in validation.section_errors.items()
        if "NUMERIC_FINDING_MISSING" in codes
    }
    # Only tracks that actually requested metrics owe numbers; a knowledge-only
    # analysis track is exempt, which is why this is not every analysis id.
    assert flagged, validation.section_errors
    assert flagged <= set(analysis_ids)
