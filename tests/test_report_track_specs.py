"""Shadow comparison between planner and analyzer track decisions."""

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

from agent.report_track_specs import (
    log_report_track_spec_disagreements,
    report_track_spec_disagreements,
)
from contracts.question_analysis import DerivedMetricName, PreferredPath, QueryType
from contracts.report_research import ReportResearchPlan
from tests.test_report_research_contract import _research_plan_payload
from tests.test_semantic_lock import _make_qa


def _track(track_id: str = "prices"):
    plan = ReportResearchPlan.model_validate(_research_plan_payload())
    return next(track for track in plan.tracks if track.track_id == track_id)


def test_matching_decisions_report_no_disagreement():
    track = _track()
    analysis = _make_qa(
        query_type=QueryType.DATA_EXPLANATION,
        preferred_path=PreferredPath.TOOL,
        derived_metrics=[
            _derived(DerivedMetricName.MOM_PERCENT_CHANGE, "p_bal_gel")
        ],
    )
    analysis.answer_kind = track.analysis_answer_kind

    assert report_track_spec_disagreements(track, analysis) == []


def test_a_different_comparison_is_reported_as_the_field_that_differs():
    """The MoM/YoY distinction is the one the prose round-trip loses."""

    track = _track()
    analysis = _make_qa(
        query_type=QueryType.DATA_EXPLANATION,
        preferred_path=PreferredPath.TOOL,
        derived_metrics=[
            _derived(DerivedMetricName.YOY_PERCENT_CHANGE, "p_bal_gel")
        ],
    )
    analysis.answer_kind = track.analysis_answer_kind

    disagreements = report_track_spec_disagreements(track, analysis)

    assert disagreements == [
        {
            "field": "derived_metrics",
            "planner": "mom_percent_change",
            "analyzer": "yoy_percent_change",
        }
    ]


def test_a_missing_analysis_is_not_counted_as_a_disagreement():
    """A failed pipeline is its own telemetry, not a planner mismatch."""

    assert report_track_spec_disagreements(_track(), None) == []


def test_the_shadow_line_carries_enum_values_and_no_query_text(caplog):
    track = _track()
    analysis = _make_qa(
        query_type=QueryType.FORECAST,
        preferred_path=PreferredPath.SQL,
        canonical_query="secret balancing question about May 2026 prices",
    )

    with caplog.at_level(logging.INFO, logger="Enai.ReportTrackSpec"):
        log_report_track_spec_disagreements(track, analysis)

    record = next(
        item.message
        for item in caplog.records
        if item.message.startswith("REPORT_TRACK_SPEC_DISAGREEMENT ")
    )
    payload = json.loads(record.split(" ", 1)[1])
    assert payload["track_id"] == "prices"
    assert payload["agreed"] is False
    assert {entry["field"] for entry in payload["disagreements"]} >= {
        "query_type",
        "preferred_path",
    }
    assert "secret" not in record
    assert "May 2026" not in record


def _derived(metric_name, metric):
    from contracts.question_analysis import DerivedMetricRequest

    return DerivedMetricRequest(metric_name=metric_name, metric=metric)
