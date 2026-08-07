"""Deterministic adaptive document planning from gated research tracks."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from contracts.report import (
    REPORT_MAX_EXHIBITS,
    REPORT_SECTION_MIN_WORDS,
    ReportChartRequest,
)
from contracts.report_charts import ReportChartBuildDecision
from contracts.report_document import (
    ReportDocumentPlan,
    ReportDocumentProfile,
    ReportDocumentSectionRole,
    ReportDocumentSectionSpec,
    ReportEvidenceCapacity,
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
        "implications": "Cross-cutting implications",
        "limitations": "Method and limitations",
    },
    "ka": {
        "implications": "საკითხთაშორისი შედეგები",
        "limitations": "მეთოდი და შეზღუდვები",
    },
    "ru": {
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


def classify_report_document_profile(
    *,
    usable_track_count: int,
    usable_exhibit_count: int,
    validated_finding_count: int,
) -> ReportDocumentProfile:
    """Choose document breadth from evidence that actually exists."""

    breadth = max(usable_track_count, usable_exhibit_count)
    if breadth >= 3:
        return ReportDocumentProfile.FULL
    if breadth >= 2 or validated_finding_count >= 5:
        return ReportDocumentProfile.FOCUSED
    return ReportDocumentProfile.COMPACT


def assess_report_evidence_capacity(
    packets: Sequence[ReportEvidencePacket],
    gate: ReportEvidenceGate,
    chart_decisions: Sequence[ReportChartBuildDecision],
) -> ReportEvidenceCapacity:
    """Summarize usable tracks, findings, and built exhibits."""

    packet_by_id = {packet.track_id: packet for packet in packets}
    if len(packet_by_id) != len(packets):
        raise ValueError("Evidence-capacity packets must be unique.")
    decision_ids = [decision.chart_id for decision in chart_decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("Evidence-capacity chart decisions must be unique.")

    complete_track_count = sum(
        track.status is ReportTrackStatus.COMPLETE for track in gate.tracks
    )
    partial_track_count = sum(
        track.status is ReportTrackStatus.PARTIAL for track in gate.tracks
    )
    unavailable_track_count = (
        len(gate.tracks) - complete_track_count - partial_track_count
    )
    usable_track_ids = {
        track.track_id
        for track in gate.tracks
        if track.status
        in {ReportTrackStatus.COMPLETE, ReportTrackStatus.PARTIAL}
    }
    validated_finding_count = sum(
        len(packet_by_id[track_id].observations)
        for track_id in usable_track_ids
        if track_id in packet_by_id
    )
    usable_exhibit_count = sum(
        decision.status == "built" for decision in chart_decisions
    )
    usable_track_count = complete_track_count + partial_track_count
    profile = classify_report_document_profile(
        usable_track_count=usable_track_count,
        usable_exhibit_count=usable_exhibit_count,
        validated_finding_count=validated_finding_count,
    )
    return ReportEvidenceCapacity(
        profile=profile,
        usable_track_count=usable_track_count,
        complete_track_count=complete_track_count,
        partial_track_count=partial_track_count,
        unavailable_track_count=unavailable_track_count,
        usable_exhibit_count=usable_exhibit_count,
        validated_finding_count=validated_finding_count,
    )


def _weighted_word_split(total: int, weights: Sequence[int]) -> list[int]:
    """Split ``total`` across ``weights`` exactly, largest remainder first."""

    weight_total = sum(weights)
    if weight_total <= 0:
        base, remainder = divmod(total, len(weights))
        return [
            base + (1 if index < remainder else 0)
            for index in range(len(weights))
        ]
    exact = [total * weight / weight_total for weight in weights]
    allocated = [int(value) for value in exact]
    for index in sorted(
        range(len(exact)),
        key=lambda position: (
            -(exact[position] - allocated[position]),
            position,
        ),
    )[: total - sum(allocated)]:
        allocated[index] += 1
    return allocated


def allocate_report_word_targets(
    evidence_capacity: ReportEvidenceCapacity,
    *,
    analysis_count: int,
    include_implications: bool,
    analysis_weights: Sequence[int] | None = None,
) -> tuple[int, list[int], int, int]:
    """Allocate prose targets from evidence breadth, not report-mode padding.

    ``analysis_weights`` scales each analysis section to what its track can
    actually say. An even split gave a documented-context track the same target
    as one holding sixty-one rows of prices, and the section floor then obliged
    the writer to reach it from market-design prose alone — which is how job
    5e6b0cf3 filled a section with generic accounting caveats nobody asked for.
    """

    profile = evidence_capacity.profile
    if profile is ReportDocumentProfile.COMPACT:
        raw_target = (
            300
            + 30 * evidence_capacity.usable_track_count
            + 25 * evidence_capacity.usable_exhibit_count
            + 10 * min(evidence_capacity.validated_finding_count, 6)
        )
        target_words = min(500, max(300, raw_target))
        implication_words = 0
        limitation_words = 60
    elif profile is ReportDocumentProfile.FOCUSED:
        raw_target = (
            420
            + 60 * evidence_capacity.usable_track_count
            + 40 * evidence_capacity.usable_exhibit_count
            + 10 * min(evidence_capacity.validated_finding_count, 8)
        )
        target_words = min(800, max(450, raw_target))
        implication_words = 100 if include_implications else 0
        limitation_words = 70
    else:
        raw_target = (
            620
            + 90 * evidence_capacity.usable_track_count
            + 55 * evidence_capacity.usable_exhibit_count
            + 12 * min(evidence_capacity.validated_finding_count, 12)
        )
        target_words = min(1300, max(750, raw_target))
        implication_words = 130 if include_implications else 0
        limitation_words = 90
    analysis_total = (
        target_words
        - limitation_words
        - implication_words
    )
    base, remainder = divmod(analysis_total, analysis_count)
    analysis_words = [
        base + (1 if index < remainder else 0)
        for index in range(analysis_count)
    ]
    if analysis_weights is not None and len(analysis_weights) == analysis_count:
        weighted = _weighted_word_split(analysis_total, analysis_weights)
        # A weight that starves a section below the schema minimum is worse
        # than an even split: the section still has to exist, and the plan
        # would not validate.
        if min(weighted) >= REPORT_SECTION_MIN_WORDS:
            analysis_words = weighted
    return (
        target_words,
        analysis_words,
        implication_words,
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
    # Two shares for a group that carries verified numbers, one for a group
    # that can only restate documented context. Deliberately coarse: the point
    # is that a knowledge-only section stops being asked for as much prose as a
    # data section, not that prose scales with row count.
    analysis_weights = [
        2
        if any(
            packet_by_id[track_id].numeric_observation_count
            for track_id in group
            if track_id in packet_by_id
        )
        else 1
        for group in groups
    ]
    evidence_capacity = assess_report_evidence_capacity(
        packets,
        gate,
        chart_decisions,
    )
    include_implications = len(completed_track_ids) >= 2
    (
        target_words,
        analysis_words,
        implication_words,
        limitation_words,
    ) = allocate_report_word_targets(
        evidence_capacity,
        analysis_count=len(groups),
        include_implications=include_implications,
        analysis_weights=analysis_weights,
    )
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
    if len(charts) > REPORT_MAX_EXHIBITS:
        raise ValueError(
            "Document plan exceeds the configured exhibit limit."
        )

    # The standard pipeline's computed statistics and curated knowledge are
    # report-wide: they belong to no research packet, so a track-scoped
    # assignment would leave them in the manifest uncited and therefore unable
    # to ground a single sentence. Every analysis section may cite them, and
    # they lead the list because a section's analysis rests on them.
    packet_refs = {
        item.evidence_ref
        for packet in packets
        for item in packet.items
    }
    shared_narrative_refs = [
        item.evidence_ref
        for item in manifest.items
        if (
            item.evidence_ref not in packet_refs
            and item.kind is not ReportEvidenceKind.LIMITATION
        )
    ]

    analysis_sections: list[ReportDocumentSectionSpec] = []
    all_analysis_refs: list[str] = []
    for index, (group, section_id) in enumerate(
        zip(groups, group_section_ids, strict=True)
    ):
        track_refs = list(
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
        if not track_refs:
            raise ValueError(
                "A usable analysis track has no manifest evidence."
            )
        evidence_refs = list(
            dict.fromkeys([*shared_narrative_refs, *track_refs])
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
    sections: list[ReportDocumentSectionSpec] = list(analysis_sections)
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
        profile=evidence_capacity.profile,
        evidence_capacity=evidence_capacity,
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
