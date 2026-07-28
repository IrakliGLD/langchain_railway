"""Deterministic adaptive document planning from gated research tracks."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from contracts.report import ReportChartRequest
from contracts.report_charts import ReportChartBuildDecision
from contracts.report_document import (
    ReportDocumentPlan,
    ReportDocumentSectionRole,
    ReportDocumentSectionSpec,
)
from contracts.report_evidence import (
    ReportEvidenceKind,
    ReportEvidenceManifest,
)
from contracts.report_research import (
    ReportCoverageStatus,
    ReportEvidenceGate,
    ReportEvidencePacket,
    ReportResearchPlan,
    ReportTrackStatus,
)

_ROLE_LABELS = {
    "en": {
        "summary": "Executive summary",
        "implications": "Cross-cutting implications",
        "limitations": "Method and limitations",
    },
    "ka": {
        "summary": "აღმასრულებელი შეჯამება",
        "implications": "საკითხთაშორისი შედეგები",
        "limitations": "მეთოდი და შეზღუდვები",
    },
    "ru": {
        "summary": "Краткое резюме",
        "implications": "Межтематические выводы",
        "limitations": "Метод и ограничения",
    },
}


def _labels(language_code: str) -> dict[str, str]:
    return _ROLE_LABELS.get(language_code, _ROLE_LABELS["en"])


def _analysis_groups(track_ids: list[str]) -> list[list[str]]:
    if len(track_ids) <= 5:
        return [[track_id] for track_id in track_ids]
    return [
        *[[track_id] for track_id in track_ids[:4]],
        track_ids[4:],
    ]


def _word_allocations(
    analysis_count: int,
    *,
    include_implications: bool,
) -> tuple[int, int, list[int], int]:
    target_words = min(1400, max(900, 850 + 100 * analysis_count))
    summary_words = min(160, 120 + 10 * analysis_count)
    limitation_words = 100
    implication_words = 140 if include_implications else 0
    analysis_total = (
        target_words
        - summary_words
        - limitation_words
        - implication_words
    )
    base, remainder = divmod(analysis_total, analysis_count)
    analysis_words = [
        base + (1 if index < remainder else 0)
        for index in range(analysis_count)
    ]
    return (
        target_words,
        summary_words,
        analysis_words,
        limitation_words,
    )


def build_report_document_plan(
    query: str,
    research_plan: ReportResearchPlan,
    packets: Sequence[ReportEvidencePacket],
    manifest: ReportEvidenceManifest,
    gate: ReportEvidenceGate,
    chart_decisions: Sequence[ReportChartBuildDecision],
) -> ReportDocumentPlan:
    """Build a compact report structure from usable tracks and built charts."""

    if not gate.ready_for_writing:
        raise ValueError("Report evidence is not ready for document planning.")
    query_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    if not (
        research_plan.query_digest
        == manifest.query_digest
        == gate.query_digest
        == query_digest
    ):
        raise ValueError("Document planning inputs have mismatched queries.")

    packet_by_id = {packet.track_id: packet for packet in packets}
    if len(packet_by_id) != len(packets):
        raise ValueError("Document planning packets must be unique.")
    research_track_by_id = {
        track.track_id: track for track in research_plan.tracks
    }
    coverage_by_id = {track.track_id: track for track in gate.tracks}
    completed_track_ids = [
        track.track_id
        for track in research_plan.tracks
        if (
            track.track_id in coverage_by_id
            and coverage_by_id[track.track_id].status
            in {ReportTrackStatus.COMPLETE, ReportTrackStatus.PARTIAL}
        )
    ]
    if not completed_track_ids:
        raise ValueError("Document planning requires usable evidence.")
    gap_track_ids = [
        track.track_id
        for track in research_plan.tracks
        if (
            track.track_id not in coverage_by_id
            or coverage_by_id[track.track_id].status
            is not ReportTrackStatus.COMPLETE
        )
    ]
    groups = _analysis_groups(completed_track_ids)
    include_implications = len(completed_track_ids) >= 2
    (
        target_words,
        summary_words,
        analysis_words,
        limitation_words,
    ) = _word_allocations(
        len(groups),
        include_implications=include_implications,
    )
    implication_words = 140 if include_implications else 0
    labels = _labels(research_plan.language_code)
    manifest_refs = set(manifest.item_by_ref())

    group_section_ids: list[str] = []
    track_to_section: dict[str, str] = {}
    for index, group in enumerate(groups):
        section_id = (
            group[0]
            if len(group) == 1
            else (
                "combined_analysis"
                if "combined_analysis" not in research_track_by_id
                else f"combined_tracks_{index + 1}"
            )
        )
        group_section_ids.append(section_id)
        track_to_section.update(
            {track_id: section_id for track_id in group}
        )

    candidate_by_id = {
        candidate.chart_id: (packet.track_id, candidate)
        for packet in packets
        for candidate in packet.chart_candidates
    }
    charts: list[ReportChartRequest] = []
    chart_ids_by_section: dict[str, list[str]] = {
        section_id: [] for section_id in group_section_ids
    }
    for decision in chart_decisions:
        candidate_entry = candidate_by_id.get(decision.chart_id)
        if candidate_entry is None:
            raise ValueError(
                "Research exhibit decision has no packet candidate."
            )
        track_id, candidate = candidate_entry
        section_id = track_to_section.get(track_id)
        if section_id is None:
            continue
        charts.append(
            ReportChartRequest(
                chart_id=candidate.chart_id,
                section_id=section_id,
                purpose=candidate.purpose,
                title=candidate.title,
                evidence_refs=candidate.evidence_refs,
                x_field=candidate.x_field,
                series_fields=candidate.series_fields,
                # Evidence gating has already converted an unavailable
                # expected exhibit into an explicit coverage gap. Preserve
                # the request for final disclosure, but do not let the same
                # known omission abort assembly.
                required=(
                    candidate.required and decision.status == "built"
                ),
            )
        )
        chart_ids_by_section[section_id].append(candidate.chart_id)
    if len(charts) > 3:
        raise ValueError("Document plan cannot contain more than three charts.")

    analysis_sections: list[ReportDocumentSectionSpec] = []
    all_analysis_refs: list[str] = []
    for index, (group, section_id) in enumerate(
        zip(groups, group_section_ids, strict=True)
    ):
        evidence_refs = list(
            dict.fromkeys(
                item.evidence_ref
                for track_id in group
                for item in packet_by_id[track_id].items
                if (
                    item.kind is not ReportEvidenceKind.LIMITATION
                    and item.evidence_ref in manifest_refs
                )
            )
        )
        if not evidence_refs:
            raise ValueError(
                "A usable analysis track has no manifest evidence."
            )
        all_analysis_refs.extend(evidence_refs)
        track_titles = [
            research_track_by_id[track_id].title for track_id in group
        ]
        analysis_sections.append(
            ReportDocumentSectionSpec(
                section_id=section_id,
                role=ReportDocumentSectionRole.ANALYSIS,
                title=" / ".join(track_titles)[:160],
                objective=(
                    "Analyze the assigned research track evidence, state "
                    "observed values, and avoid unsupported causal claims."
                ),
                target_words=analysis_words[index],
                track_ids=group,
                required_evidence_refs=evidence_refs[:32],
                chart_refs=chart_ids_by_section[section_id],
            )
        )

    all_analysis_refs = list(dict.fromkeys(all_analysis_refs))[:32]
    limitation_refs = [
        item.evidence_ref
        for item in manifest.items
        if item.kind is ReportEvidenceKind.LIMITATION
    ]
    if not limitation_refs:
        raise ValueError("Document plan requires limitation evidence.")
    summary = ReportDocumentSectionSpec(
        section_id="executive_summary",
        role=ReportDocumentSectionRole.EXECUTIVE_SUMMARY,
        title=labels["summary"],
        objective=(
            "Summarize the strongest cross-track findings after drafting "
            "the analytical body."
        ),
        target_words=summary_words,
        track_ids=completed_track_ids,
        required_evidence_refs=all_analysis_refs,
        chart_refs=[],
    )
    sections: list[ReportDocumentSectionSpec] = [
        summary,
        *analysis_sections,
    ]
    if include_implications:
        sections.append(
            ReportDocumentSectionSpec(
                section_id="implications",
                role=ReportDocumentSectionRole.IMPLICATIONS,
                title=labels["implications"],
                objective=(
                    "Synthesize relationships across completed tracks while "
                    "distinguishing observation from interpretation."
                ),
                target_words=implication_words,
                track_ids=completed_track_ids,
                required_evidence_refs=all_analysis_refs,
                chart_refs=[],
            )
        )
    sections.append(
        ReportDocumentSectionSpec(
            section_id="limitations",
            role=ReportDocumentSectionRole.LIMITATIONS,
            title=labels["limitations"],
            objective=(
                "Disclose collection gaps, unavailable periods, source "
                "boundaries, and limits on causal interpretation."
            ),
            target_words=limitation_words,
            track_ids=[
                track.track_id for track in research_plan.tracks
            ],
            required_evidence_refs=limitation_refs[:32],
            chart_refs=[],
        )
    )
    return ReportDocumentPlan(
        contract_version="report-document-plan-v1",
        query_digest=query_digest,
        title=research_plan.objective[:200],
        objective=research_plan.objective[:1000],
        language_code=research_plan.language_code,
        target_words=target_words,
        evidence_manifest_id=manifest.manifest_id,
        coverage_status=gate.status.value,
        required_track_ids=[
            track.track_id
            for track in research_plan.tracks
            if track.required
        ],
        completed_track_ids=completed_track_ids,
        gap_track_ids=gap_track_ids,
        sections=sections,
        charts=charts,
    )
