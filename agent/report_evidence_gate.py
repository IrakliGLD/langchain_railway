"""Deterministic sufficiency policy for collected report evidence."""

from __future__ import annotations

from collections.abc import Sequence

from contracts.report_charts import ReportChartBuildDecision
from contracts.report_evidence import ReportEvidenceKind
from contracts.report_research import (
    ReportCoverageStatus,
    ReportEvidenceGate,
    ReportEvidenceMode,
    ReportEvidencePacket,
    ReportResearchPlan,
    ReportTrackCoverage,
    ReportTrackStatus,
)


def _gap_findings(gaps: Sequence[str]) -> set[str]:
    findings: set[str] = set()
    for gap in gaps:
        if "FAILED" in gap:
            findings.add("COLLECTOR_FAILURE")
        elif "EXHIBIT" in gap:
            findings.add("EXPECTED_EXHIBIT_UNAVAILABLE")
        elif "NO_EVIDENCE" in gap or "UNAVAILABLE" in gap:
            findings.add("EVIDENCE_UNAVAILABLE")
        else:
            findings.add("TRACK_EVIDENCE_GAP")
    return findings


def evaluate_report_evidence(
    plan: ReportResearchPlan,
    packets: Sequence[ReportEvidencePacket],
    *,
    chart_decisions: Sequence[ReportChartBuildDecision],
) -> ReportEvidenceGate:
    """Evaluate packet completeness without reading evidence prose or values."""

    packet_ids = [packet.track_id for packet in packets]
    if len(packet_ids) != len(set(packet_ids)):
        raise ValueError("Evidence packets must use unique track IDs.")
    known_track_ids = {track.track_id for track in plan.tracks}
    unknown_packets = sorted(set(packet_ids) - known_track_ids)
    if unknown_packets:
        raise ValueError(
            "Evidence packets reference unknown research tracks: "
            + ", ".join(unknown_packets)
        )

    decision_ids = [decision.chart_id for decision in chart_decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("Chart decisions must use unique chart IDs.")
    known_chart_ids = {
        candidate.chart_id
        for packet in packets
        for candidate in packet.chart_candidates
    }
    if not set(decision_ids).issubset(known_chart_ids):
        raise ValueError(
            "Chart decisions reference unknown research exhibits."
        )

    packets_by_id = {packet.track_id: packet for packet in packets}
    decisions_by_id = {
        decision.chart_id: decision for decision in chart_decisions
    }
    coverage: list[ReportTrackCoverage] = []
    global_findings: set[str] = set()

    for track in plan.tracks:
        packet = packets_by_id.get(track.track_id)
        if packet is None:
            coverage.append(
                ReportTrackCoverage(
                    track_id=track.track_id,
                    required=track.required,
                    status=ReportTrackStatus.UNAVAILABLE,
                    evidence_item_count=0,
                    numeric_observation_count=0,
                    chart_candidate_count=0,
                    finding_codes=["PACKET_MISSING"],
                )
            )
            global_findings.add("PACKET_MISSING")
            continue

        findings = _gap_findings(packet.gaps)
        kinds = {item.kind for item in packet.items}
        if (
            track.evidence_mode is ReportEvidenceMode.TABLE
            and ReportEvidenceKind.TABLE not in kinds
        ):
            findings.add("TABLE_EVIDENCE_MISSING")
        elif (
            track.evidence_mode is ReportEvidenceMode.KNOWLEDGE
            and ReportEvidenceKind.KNOWLEDGE not in kinds
        ):
            findings.add("KNOWLEDGE_EVIDENCE_MISSING")
        elif track.evidence_mode is ReportEvidenceMode.MIXED and not {
            ReportEvidenceKind.TABLE,
            ReportEvidenceKind.KNOWLEDGE,
        }.issubset(kinds):
            findings.add("MIXED_EVIDENCE_INCOMPLETE")

        required_numeric_count = min(len(track.requested_metrics), 4)
        if (
            required_numeric_count
            and packet.numeric_observation_count < required_numeric_count
        ):
            findings.add("NUMERIC_EVIDENCE_INSUFFICIENT")

        built_chart_count = 0
        candidates_by_purpose = {
            candidate.purpose: candidate
            for candidate in packet.chart_candidates
        }
        for purpose in track.expected_exhibits:
            candidate = candidates_by_purpose.get(purpose)
            if candidate is None:
                findings.add("EXPECTED_EXHIBIT_MISSING")
                continue
            decision = decisions_by_id.get(candidate.chart_id)
            if decision is None or decision.status != "built":
                findings.add("REQUIRED_EXHIBIT_OMITTED")
            else:
                built_chart_count += 1

        status = packet.status
        if (
            status is ReportTrackStatus.COMPLETE
            and findings
            and packet.items
        ):
            status = ReportTrackStatus.PARTIAL
        if status is ReportTrackStatus.PARTIAL and not findings:
            findings.add("TRACK_EVIDENCE_GAP")
        if status in {
            ReportTrackStatus.UNAVAILABLE,
            ReportTrackStatus.FAILED,
        } and not findings:
            findings.add("EVIDENCE_UNAVAILABLE")

        coverage.append(
            ReportTrackCoverage(
                track_id=track.track_id,
                required=track.required,
                status=status,
                evidence_item_count=len(packet.items),
                numeric_observation_count=(
                    packet.numeric_observation_count
                ),
                chart_candidate_count=built_chart_count,
                finding_codes=sorted(findings),
            )
        )
        global_findings.update(findings)

    incomplete_required = any(
        track.required and track.status is not ReportTrackStatus.COMPLETE
        for track in coverage
    )
    incomplete_optional = any(
        not track.required and track.status is not ReportTrackStatus.COMPLETE
        for track in coverage
    )
    successful_required = any(
        track.required
        and track.status
        in {ReportTrackStatus.COMPLETE, ReportTrackStatus.PARTIAL}
        for track in coverage
    )
    if not global_findings and not incomplete_required and not incomplete_optional:
        status = ReportCoverageStatus.READY
    elif successful_required:
        status = ReportCoverageStatus.READY_WITH_GAPS
        if incomplete_required:
            global_findings.add("REQUIRED_TRACK_GAP")
        if incomplete_optional:
            global_findings.add("OPTIONAL_TRACK_GAP")
    else:
        status = ReportCoverageStatus.FAILED
        global_findings.add("NO_REQUIRED_EVIDENCE")

    return ReportEvidenceGate(
        contract_version="report-evidence-gate-v1",
        query_digest=plan.query_digest,
        status=status,
        tracks=coverage,
        finding_codes=sorted(global_findings),
    )
