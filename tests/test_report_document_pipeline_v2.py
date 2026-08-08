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
from agent.report_sections import count_section_words, validate_report_section
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
from contracts.report_research import ReportEvidenceMode
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
        # Reachable by every analysis section...
        citable = set(section.required_evidence_refs) | set(
            section.optional_evidence_refs
        )
        assert statistics.evidence_ref in citable
        assert knowledge.evidence_ref in citable
        # ...but owed by none of them, or all four end up discussing the same
        # passage to satisfy REQUIRED_EVIDENCE_NOT_USED.
        assert statistics.evidence_ref not in section.required_evidence_refs
        assert knowledge.evidence_ref not in section.required_evidence_refs

    # Owed by exactly one section, so a manifest item cannot go uncited.
    limitations = next(
        section
        for section in plan.sections
        if section.role is ReportDocumentSectionRole.LIMITATIONS
    )
    assert statistics.evidence_ref in limitations.required_evidence_refs
    assert knowledge.evidence_ref in limitations.required_evidence_refs


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


def _full_capacity() -> ReportEvidenceCapacity:
    return ReportEvidenceCapacity(
        profile="full",
        usable_track_count=4,
        complete_track_count=3,
        partial_track_count=1,
        unavailable_track_count=0,
        usable_exhibit_count=4,
        validated_finding_count=10,
    )


def test_a_documented_context_section_is_not_sized_like_a_data_section():
    """An even split is what fills a knowledge track with generic caveats.

    Job 5e6b0cf3 gave market_design_context the same target as a track holding
    sixty-one rows of prices; its floor then obliged the writer to reach that
    length from market-design prose alone.
    """

    even = allocate_report_word_targets(
        _full_capacity(),
        analysis_count=4,
        include_implications=True,
    )
    weighted = allocate_report_word_targets(
        _full_capacity(),
        analysis_count=4,
        include_implications=True,
        analysis_weights=[2, 2, 2, 1],
    )

    assert even[0] == weighted[0]
    # The document budget is unchanged; only its distribution moves.
    assert sum(weighted[1]) == sum(even[1])
    assert weighted[1][3] < even[1][3]
    assert weighted[1][0] > even[1][0]
    # Equal weights land within the one word an indivisible total leaves over.
    assert max(weighted[1][:3]) - min(weighted[1][:3]) <= 1
    assert all(words >= 40 for words in weighted[1])


def test_a_knowledge_track_is_weighted_as_context_even_when_it_holds_numbers():
    """A knowledge track still collects numeric tools for context.

    Counting observations alone would size market_design_context like a price
    section, which is the sizing that filled it with accounting caveats.
    """

    research_plan, packets, manifest, decisions, gate = _ready_components()
    knowledge_ids = {
        track.track_id
        for track in research_plan.tracks
        if track.evidence_mode is ReportEvidenceMode.KNOWLEDGE
    }
    assert knowledge_ids, [
        track.evidence_mode for track in research_plan.tracks
    ]

    document_plan = build_report_document_plan(
        _QUERY,
        research_plan,
        packets,
        manifest,
        gate,
        decisions,
    )

    analysis = [
        section
        for section in document_plan.sections
        if section.role is ReportDocumentSectionRole.ANALYSIS
    ]
    context_sections = [
        section
        for section in analysis
        if knowledge_ids.issuperset(section.track_ids)
    ]
    data_sections = [
        section for section in analysis if section not in context_sections
    ]
    assert context_sections and data_sections
    assert max(
        section.target_words for section in context_sections
    ) < min(section.target_words for section in data_sections)


def test_word_weights_that_would_starve_a_section_fall_back_to_an_even_split():
    """A plan whose section drops under the schema minimum cannot validate."""

    capacity = ReportEvidenceCapacity(
        profile="compact",
        usable_track_count=1,
        complete_track_count=1,
        partial_track_count=0,
        unavailable_track_count=0,
        usable_exhibit_count=1,
        validated_finding_count=2,
    )

    even = allocate_report_word_targets(
        capacity,
        analysis_count=5,
        include_implications=False,
    )
    weighted = allocate_report_word_targets(
        capacity,
        analysis_count=5,
        include_implications=False,
        analysis_weights=[8, 1, 1, 1, 1],
    )

    assert weighted[1] == even[1]
    assert all(words >= 40 for words in weighted[1])


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


def test_analysis_section_ships_grounded_subset_when_a_stray_number_survives():
    """The production shape: repair does not converge on one stray figure.

    Job 33403df2 died with DERIVED_CLAIM_NOT_USED,UNGROUNDED_NUMERIC_CLAIM
    after its single repair call, and the reader got nothing. The verified
    claims around the stray value are still publishable, so the salvage
    publishes them — the stray value itself must not reach the reader, and the
    shorter section it leaves behind is conceded, not failed.
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

    published = generate_report_document(
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
    assert all(
        "987.6" not in paragraph.text
        for section in published.generation_order_sections()
        for paragraph in section.paragraphs
    )
    residual = validate_report_document(
        published,
        document_plan,
        manifest,
        research_plan,
    )
    assert set(residual.section_errors) <= {poisoned_id}
    assert residual.section_errors.get(poisoned_id, []) in (
        [],
        ["WORD_COUNT_TOO_SHORT"],
    )


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


def _unrendered_direct_claim(section, section_spec, manifest):
    """A coordinate that verifies but whose value the prose never renders."""
    from agent.report_grounding import _claimable_direct_claim

    item_by_ref = manifest.item_by_ref()
    cited = {
        (claim.evidence_ref, claim.row_index, claim.column)
        for paragraph in section.paragraphs
        for claim in paragraph.direct_claims
    }
    text = " ".join(paragraph.text for paragraph in section.paragraphs)
    for evidence_ref in section_spec.required_evidence_refs:
        item = item_by_ref.get(evidence_ref)
        if item is None:
            continue
        for row_index, column, _value in item.citable_numeric_coordinates():
            if (evidence_ref, row_index, column) in cited:
                continue
            claim = _claimable_direct_claim(
                item,
                evidence_ref,
                row_index,
                column,
            )
            if claim is None or claim.display_value in text:
                continue
            return claim
    return None


def _with_unrendered_claim(section, section_spec, manifest):
    claim = _unrendered_direct_claim(section, section_spec, manifest)
    assert claim is not None, section.section_id
    first, *rest = section.paragraphs
    return section.model_copy(
        update={
            "paragraphs": [
                first.model_copy(
                    update={"direct_claims": [*first.direct_claims, claim]}
                ),
                *rest,
            ]
        }
    )


def test_a_claim_the_prose_never_rendered_is_swept_not_repaired():
    """Deleting surplus metadata costs nothing and needs no provider call.

    Job 827556eb spent both repairs converging its analysis batch, and the
    synthesis batch then failed on DIRECT_CLAIM_NOT_USED alone with no budget
    left. The claim was correct about its coordinate and simply absent from the
    prose, so code can drop it.
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
    spec_by_id = {
        section.section_id: section for section in document_plan.sections
    }
    # A synthesis section owes no numeric findings, so an unused claim there is
    # pure surplus — the exact shape that sank job 827556eb.
    synthesis_ids = [
        section.section_id
        for section in document_plan.sections
        if section.role is not ReportDocumentSectionRole.ANALYSIS
        and _unrendered_direct_claim(
            section_by_id[section.section_id],
            section,
            manifest,
        )
        is not None
    ]
    assert synthesis_ids
    polluted = [
        _with_unrendered_claim(
            section_by_id[section_id],
            spec_by_id[section_id],
            manifest,
        )
        for section_id in synthesis_ids
    ]
    for section in polluted:
        assert "DIRECT_CLAIM_NOT_USED" in validate_report_section(
            section,
            spec_by_id[section.section_id],
            manifest,
        ).error_codes

    sections, validation = generation._materialize_section_batch(
        ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=polluted,
        ),
        document_plan,
        manifest,
        section_ids=synthesis_ids,
        research_plan=research_plan,
    )

    assert validation.valid, validation.section_errors
    assert sections is not None
    # The prose is untouched; only the surplus metadata entry is gone.
    for swept, original in zip(sections, polluted, strict=True):
        assert [
            paragraph.text for paragraph in swept.paragraphs
        ] == [paragraph.text for paragraph in original.paragraphs]
        assert sum(
            len(paragraph.direct_claims) for paragraph in swept.paragraphs
        ) == sum(
            len(paragraph.direct_claims) for paragraph in original.paragraphs
        ) - 1


def test_an_unrendered_claim_an_analysis_section_owes_is_not_swept():
    """Below the numeric floor the number is owed, not surplus.

    Sweeping it would trade DIRECT_CLAIM_NOT_USED for NUMERIC_FINDING_MISSING
    and hide the fact that the writer has to render the value.
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
    spec_by_id = {
        section.section_id: section for section in document_plan.sections
    }
    owing_id = next(
        section_id
        for section_id, spec in spec_by_id.items()
        if generation.report_analysis_numeric_claim_requirement(
            section_by_id[section_id],
            spec,
            manifest,
            research_plan,
        )[1]
        >= 2
    )
    single_claim = ReportSectionDraft.model_validate(
        _draft_section(
            spec_by_id[owing_id],
            manifest,
            max_numeric_claims=1,
        )
    )
    polluted = _with_unrendered_claim(
        single_claim,
        spec_by_id[owing_id],
        manifest,
    )

    kept = generation._without_free_unrendered_claims(
        polluted,
        spec_by_id[owing_id],
        manifest,
        research_plan,
    )

    assert kept == polluted
    assert generation.report_analysis_numeric_claim_requirement(
        polluted,
        spec_by_id[owing_id],
        manifest,
        research_plan,
    ) == (2, 2)


def test_an_unverified_claim_is_never_swept():
    """A wrong coordinate is a writer error the repair pass has to see."""
    from agent.report_grounding import drop_unrendered_claims

    (
        _research_plan,
        _packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    section = valid_draft.sections[0]
    first, *rest = section.paragraphs
    fabricated = first.direct_claims[0].model_copy(
        update={"display_value": "999999", "unit": "GEL/MWh"}
    )
    polluted = section.model_copy(
        update={
            "paragraphs": [
                first.model_copy(
                    update={
                        "direct_claims": [*first.direct_claims, fabricated]
                    }
                ),
                *rest,
            ]
        }
    )

    swept, dropped = drop_unrendered_claims(polluted, manifest.item_by_ref())

    assert dropped == 0
    assert swept is polluted


def test_a_batch_that_reaches_salvage_unbudgeted_still_logs_its_state(caplog):
    """A give-up with no repair call must not be silent.

    Job 827556eb's synthesis batch failed with zero budget, so the in-loop
    diagnostic never ran and nothing recorded which section or code sank it.
    """
    from agent import report_document_generation as generation

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
        if section.role is ReportDocumentSectionRole.ANALYSIS
    ]
    broken = [
        _numberless_section(section_by_id[section_id])
        for section_id in analysis_ids
    ]
    _sections, validation = generation._materialize_section_batch(
        ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=broken,
        ),
        document_plan,
        manifest,
        section_ids=analysis_ids,
        research_plan=research_plan,
    )
    assert not validation.valid

    with caplog.at_level(logging.INFO, logger="Enai.ReportDocument"):
        _result, _validation, used = (
            generation._repair_section_batch_until_valid(
                _QUERY,
                document_plan,
                research_plan,
                manifest,
                packets,
                broken,
                None,
                validation,
                lambda *_args, **_kwargs: pytest.fail(
                    "no repair may run without budget"
                ),
                section_ids=analysis_ids,
                repair_attempts=0,
                first_attempt_number=2,
                stage="analysis_repair_exhausted",
            )
        )

    assert used == 0
    unbudgeted = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("REPORT_DOCUMENT_DIAGNOSTIC ")
        and '"batch_repair_unbudgeted"' in record.message
    ]
    assert unbudgeted, [record.message for record in caplog.records]
    assert set(unbudgeted[-1]["section_word_counts"]) == set(analysis_ids)
    assert unbudgeted[-1]["section_error_codes"]


def _short_but_grounded_section(section, section_spec, manifest):
    """Keep every claim and citation; trim the prose under the floor."""

    minimum_words, _ = report_section_validation_word_bounds(
        section_spec.target_words,
        evidence_row_count=manifest.assigned_row_count(
            section_spec.required_evidence_refs
        ),
    )
    keep = max(12, (minimum_words - 8) // len(section.paragraphs))
    trimmed = section.model_copy(
        update={
            "paragraphs": [
                paragraph.model_copy(
                    update={"text": " ".join(paragraph.text.split()[:keep])}
                )
                for paragraph in section.paragraphs
            ]
        }
    )
    assert count_section_words(trimmed.content_markdown) < minimum_words
    return trimmed


def test_document_ships_when_only_length_falls_short_after_repairs(caplog):
    """A grounded report that is merely short must reach the reader.

    REPORT_DOCUMENT_INVALID is not retryable, so failing a section that carries
    every claim, cites every assigned ref, and lands under its target trades a
    slightly thin report for no report at all — job cf47a2f6.
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
    spec_by_id = {
        section.section_id: section for section in document_plan.sections
    }
    short_draft = valid_draft.model_copy(
        update={
            "sections": [
                _short_but_grounded_section(
                    section,
                    spec_by_id[section.section_id],
                    manifest,
                )
                for section in valid_draft.sections
            ]
        }
    )
    repair_calls = []

    def stubborn_repair(*_args, section_ids, **_kwargs):
        repair_calls.append(list(section_ids))
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                section
                for section in short_draft.sections
                if section.section_id in section_ids
            ],
        )

    with caplog.at_level(logging.WARNING, logger="Enai.ReportDocument"):
        published = generate_report_document(
            _QUERY,
            document_plan,
            research_plan,
            manifest,
            packets,
            write_document=lambda *_args: short_draft,
            repair_sections=stubborn_repair,
            max_repair_attempts=2,
        )

    assert published == short_draft
    # The budget is still spent trying before anything is conceded.
    assert len(repair_calls) == 2
    conceded = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("REPORT_LENGTH_CONCEDED ")
    ]
    assert conceded, [record.message for record in caplog.records]
    assert conceded[-1]["stage"] == "document_repair_exhausted"
    assert conceded[-1]["conceded_section_ids"]
    residual = validate_report_document(
        published,
        document_plan,
        manifest,
        research_plan,
    )
    assert {
        code
        for codes in residual.section_errors.values()
        for code in codes
    } == {"WORD_COUNT_TOO_SHORT"}


def test_length_concession_never_rescues_an_ungrounded_document():
    """One non-length code and the document fails exactly as before."""

    (
        research_plan,
        packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    valid_draft = _valid_document_draft(document_plan, manifest)
    spec_by_id = {
        section.section_id: section for section in document_plan.sections
    }
    broken = valid_draft.model_copy(
        update={
            "sections": [
                _numberless_section(
                    _short_but_grounded_section(
                        section,
                        spec_by_id[section.section_id],
                        manifest,
                    )
                )
                for section in valid_draft.sections
            ]
        }
    )

    def stubborn_repair(*_args, section_ids, **_kwargs):
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                section
                for section in broken.sections
                if section.section_id in section_ids
            ],
        )

    with pytest.raises(ReportDocumentGenerationError):
        generate_report_document(
            _QUERY,
            document_plan,
            research_plan,
            manifest,
            packets,
            write_document=lambda *_args: broken,
            repair_sections=stubborn_repair,
            max_repair_attempts=2,
        )


def test_numeric_finding_is_not_demanded_without_a_citable_coordinate():
    """A gate no draft can pass is a deadlock, not a standard.

    When a section's tables declare no unit on any numeric column there is no
    coordinate a claim could cite: omitting the number fails this gate and
    inventing one fails grounding, so both repairs go to a contradiction.
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
    spec_by_id = {
        section.section_id: section for section in document_plan.sections
    }
    flagged_id = next(
        section_id
        for section_id in spec_by_id
        if generation._analysis_numeric_finding_missing(
            _numberless_section(section_by_id[section_id]),
            spec_by_id[section_id],
            manifest,
            research_plan,
        )
    )
    numberless = _numberless_section(section_by_id[flagged_id])
    stripped_manifest = manifest.model_copy(
        update={
            "items": [
                item.model_copy(update={"unit_by_column": {}})
                if item.kind is ReportEvidenceKind.TABLE
                else item
                for item in manifest.items
            ]
        }
    )

    assert generation.report_analysis_numeric_claim_requirement(
        numberless,
        spec_by_id[flagged_id],
        stripped_manifest,
        research_plan,
    ) == (0, 0)
    assert not generation._analysis_numeric_finding_missing(
        numberless,
        spec_by_id[flagged_id],
        stripped_manifest,
        research_plan,
    )


def test_exhausted_batch_repair_logs_the_result_it_could_not_fix(caplog):
    """The last repair's own word counts must not be invisible.

    The batch diagnostic is emitted before each call, so an exhausted budget
    used to end with the failing counts unlogged — the numbers needed to tell a
    stalled writer from an unreachable gate (job cf47a2f6).
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
    analysis_ids = [
        section.section_id
        for section in document_plan.sections
        if section.role is ReportDocumentSectionRole.ANALYSIS
    ]

    def batch(section_ids):
        return ReportDocumentRepair(
            contract_version="report-document-repair-v1",
            sections=[
                _numberless_section(section_by_id[section_id])
                for section_id in section_ids
            ],
        )

    with caplog.at_level(logging.INFO, logger="Enai.ReportDocument"):
        with pytest.raises(ReportDocumentGenerationError):
            generate_report_document(
                _QUERY,
                document_plan,
                research_plan,
                manifest,
                packets,
                write_analysis_sections=(
                    lambda *_args, section_ids: batch(section_ids)
                ),
                write_synthesis_sections=(
                    lambda *_args, analysis_sections, section_ids: batch(
                        section_ids
                    )
                ),
                repair_sections=(
                    lambda *_args, section_ids, **_kwargs: batch(section_ids)
                ),
                max_repair_attempts=2,
            )

    diagnostics = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("REPORT_DOCUMENT_DIAGNOSTIC ")
    ]
    exhausted = [
        payload
        for payload in diagnostics
        if payload["event"] == "batch_repair_rejected"
    ]
    assert exhausted, [payload["event"] for payload in diagnostics]
    assert set(exhausted[-1]["section_word_counts"]) == set(analysis_ids)
    assert exhausted[-1]["section_error_codes"]


def test_optional_evidence_may_be_cited_without_being_owed():
    """Shared context should reach every section, be owed by one.

    required_evidence_refs is both the whitelist and the obligation, so the
    only way to let sections share the report-wide statistics and knowledge is
    to force all of them to write about it -- which is what produces the same
    framing in four sections.
    """

    (
        _research_plan,
        _packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    spec = next(
        section
        for section in document_plan.sections
        if section.role is ReportDocumentSectionRole.ANALYSIS
        and len(section.required_evidence_refs) > 1
    )
    shared_ref = spec.required_evidence_refs[0]
    trimmed = spec.model_copy(
        update={
            "required_evidence_refs": spec.required_evidence_refs[1:],
            "optional_evidence_refs": [shared_ref],
        }
    )
    drafted = ReportSectionDraft.model_validate(
        _draft_section(trimmed, manifest)
    )
    # Cite the optional ref alongside the required ones. Allowed but not owed
    # is the whole point: today the citation is rejected outright, because one
    # list serves as both whitelist and obligation.
    citing = drafted.model_copy(
        update={
            "paragraphs": [
                paragraph.model_copy(
                    update={
                        "evidence_refs": [
                            *paragraph.evidence_refs,
                            shared_ref,
                        ]
                    }
                )
                for paragraph in drafted.paragraphs
            ]
        }
    )

    citing_validation = validate_report_section(citing, trimmed, manifest)
    silent_validation = validate_report_section(drafted, trimmed, manifest)

    assert "EVIDENCE_REF_NOT_ALLOWED" not in citing_validation.error_codes
    # And leaving it uncited is not an error either.
    assert "REQUIRED_EVIDENCE_NOT_USED" not in silent_validation.error_codes


def test_an_uncited_required_ref_is_still_reported():
    """The obligation survives for the refs that keep it."""

    (
        _research_plan,
        _packets,
        manifest,
        _,
        _,
        document_plan,
    ) = _document_components()
    spec = next(
        section
        for section in document_plan.sections
        if section.role is ReportDocumentSectionRole.ANALYSIS
        and len(section.required_evidence_refs) > 1
    )
    drafted = ReportSectionDraft.model_validate(_draft_section(spec, manifest))
    # Cite one fewer ref than the section is obliged to use.
    starved = drafted.model_copy(
        update={
            "paragraphs": [
                paragraph.model_copy(
                    update={
                        "evidence_refs": spec.required_evidence_refs[:1],
                        "direct_claims": [],
                        "derived_claims": [],
                    }
                )
                for paragraph in drafted.paragraphs
            ]
        }
    )

    validation = validate_report_section(starved, spec, manifest)

    assert "REQUIRED_EVIDENCE_NOT_USED" in validation.error_codes
