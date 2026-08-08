"""Compare the research planner's analysis decisions against the analyzer's.

A report track is analysed today by a nested pipeline whose own model re-derives
what the planner already decided, from the track serialized back into prose.
Before that call is removed, the two must be shown to agree: ``query_type`` and
``preferred_path`` steer routing, and ``derived_metrics`` decides which
comparisons the evidence stage must produce — an unmet one costs the track its
evidence.

This module only observes. Nothing here changes what a track analyses.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from contracts.report_research import ReportResearchTrack

_LOGGER = logging.getLogger("Enai.ReportTrackSpec")

# One line per track, listing every field that differs. A line per
# disagreement would make a four-track report shout and bury the shape of the
# difference, which is the thing worth reading.
_MAXIMUM_REPORTED_METRICS = 8


def _enum_value(candidate: Any) -> str:
    """Render an enum for telemetry, never free text."""

    value = getattr(candidate, "value", candidate)
    return str(value) if value is not None else ""


def report_track_spec_disagreements(
    track: ReportResearchTrack,
    analysis: Any,
) -> list[dict[str, Any]]:
    """Return the fields where the planner and the analyzer disagree.

    Empty when they agree, or when no analysis was produced — an absent
    analysis is a pipeline failure that its own telemetry already reports, and
    counting it here would read as a planner disagreement it is not.
    """

    if analysis is None:
        return []
    analyzer_metrics = sorted(
        {
            _enum_value(request.metric_name)
            for request in (
                analysis.analysis_requirements.derived_metrics or []
            )
        }
    )
    planner_metrics = sorted(
        {_enum_value(name) for name in track.analysis_derived_metrics}
    )
    comparisons = (
        (
            "query_type",
            _enum_value(track.analysis_query_type),
            _enum_value(analysis.classification.query_type),
        ),
        (
            "preferred_path",
            _enum_value(track.analysis_preferred_path),
            _enum_value(analysis.routing.preferred_path),
        ),
        (
            "answer_kind",
            _enum_value(track.analysis_answer_kind),
            _enum_value(getattr(analysis, "answer_kind", None)),
        ),
        (
            "derived_metrics",
            ",".join(planner_metrics[:_MAXIMUM_REPORTED_METRICS]),
            ",".join(analyzer_metrics[:_MAXIMUM_REPORTED_METRICS]),
        ),
    )
    return [
        {"field": field, "planner": planner, "analyzer": analyzer}
        for field, planner, analyzer in comparisons
        if planner != analyzer
    ]


def log_report_track_spec_disagreements(
    track: ReportResearchTrack,
    analysis: Any,
) -> list[dict[str, Any]]:
    """Record how far the planner's decisions sit from the analyzer's.

    Values are enum members from our own contracts, so the line carries no
    query text, no evidence, and nothing a caller supplied.
    """

    disagreements = report_track_spec_disagreements(track, analysis)
    _LOGGER.info(
        "REPORT_TRACK_SPEC_DISAGREEMENT %s",
        json.dumps(
            {
                "agreed": not disagreements,
                "disagreements": disagreements,
                "track_id": track.track_id,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return disagreements
