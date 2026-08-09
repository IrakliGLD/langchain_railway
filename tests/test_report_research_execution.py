"""Deterministic, bounded, parallel report research execution tests."""

from __future__ import annotations

import hashlib
import os
import threading
from copy import deepcopy

import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import report_evidence
from agent.report_research_execution import (
    DEFAULT_REPORT_COLLECTORS,
    ReportCollectorOutput,
    build_report_track_analysis_query,
    consolidate_report_evidence_packets,
    execute_report_research,
    execute_report_track_analysis,
    merge_report_track_analysis_packet,
)
from contracts.report_evidence import (
    ReportEvidenceItem,
    ReportEvidenceKind,
    ReportKnowledgeEvidenceRole,
)
from contracts.report_research import (
    ReportCollectorId,
    ReportEvidencePacket,
    ReportResearchPlan,
    ReportResearchTrack,
    ReportTrackStatus,
)
from contracts.vector_knowledge import (
    RetrievalStrategy,
    RetrievalStrategyVersion,
    VectorChunkRecord,
    VectorKnowledgeBundle,
    VectorKnowledgeMode,
    VectorRetrievalFailure,
    VectorRetrievalFailureStage,
    VectorRetrievalOutcome,
)
from models import QueryContext
from tests.test_report_research_contract import (
    _research_plan_payload,
    _table_item,
)

_QUERY = (
    "Assess prices, energy security, and the current electricity market model."
)


def _plan() -> ReportResearchPlan:
    payload = _research_plan_payload(
        query_digest=hashlib.sha256(_QUERY.encode("utf-8")).hexdigest()
    )
    payload["tracks"][2]["expected_exhibits"] = []
    return ReportResearchPlan.model_validate(payload)


def _table_output(collector_id: ReportCollectorId) -> ReportCollectorOutput:
    return ReportCollectorOutput(
        collector_id=collector_id,
        items=(ReportEvidenceItem.model_validate(_table_item()),),
    )


def _knowledge_output() -> ReportCollectorOutput:
    return ReportCollectorOutput(
        collector_id=ReportCollectorId.VECTOR_KNOWLEDGE,
        items=(
            ReportEvidenceItem.model_validate(
                {
                    "evidence_ref": "evidence:knowledge:" + "2" * 16,
                    "kind": "knowledge",
                    "title": "Electricity market model",
                    "source": "vector",
                    "provenance_refs": ["vector:market:model"],
                    "columns": [],
                    "rows": [],
                    "content": (
                        "Approved knowledge states the documented market "
                        "model and implementation stage."
                    ),
                    "unit_by_column": {},
                    "total_row_count": 0,
                    "truncated": False,
                }
            ),
        ),
    )


def test_track_analysis_query_uses_one_primary_question_and_bounded_coverage():
    track = _plan().tracks[0]

    query = build_report_track_analysis_query(_QUERY, track)

    assert query.startswith(track.research_questions[0])
    assert f"Research track: {track.title}" in query
    assert f"Report context: {_QUERY}" in query
    for coverage_question in track.research_questions[1:]:
        assert coverage_question in query
    assert len(query) <= 4000


def test_track_analysis_query_reserves_room_for_report_context():
    payload = _plan().tracks[0].model_dump(mode="json")
    payload["research_questions"] = [
        f"question-{index}-" + "q" * 580 for index in range(6)
    ]
    track = ReportResearchTrack.model_validate(payload)
    report_query = "report-context-" + "r" * 3000

    query = build_report_track_analysis_query(report_query, track)

    assert query.startswith(track.research_questions[0])
    assert f"Report context: {report_query[:1000]}" in query
    assert len(query) <= 4000


def test_track_analysis_runs_report_pipeline_once_and_builds_existing_packet():
    calls = []

    def query_pipeline(query, **kwargs):
        calls.append((query, kwargs))
        return QueryContext(
            query=query,
            cols=["period", "price_gel"],
            rows=[
                ["2025-01", 100.0],
                ["2025-02", 120.0],
            ],
            provenance_cols=["period", "price_gel"],
            provenance_rows=[
                ["2025-01", 100.0],
                ["2025-02", 120.0],
            ],
            provenance_refs=["query:track:prices"],
            provenance_source="pipeline",
            stats_hint="Mean observed price was 110 GEL/MWh.",
            summary_domain_knowledge=(
                "Prices are formed under the applicable market rules."
            ),
            answer_mode="report",
        )

    packet = execute_report_track_analysis(
        _QUERY,
        _plan().tracks[0],
        query_pipeline=query_pipeline,
        trace_id="report-job-1",
        actor_id="actor-1",
        request_id="request-1:track:prices",
        request_deadline="deadline-sentinel",
    )

    assert len(calls) == 1
    query, kwargs = calls[0]
    assert query.startswith(_plan().tracks[0].research_questions[0])
    assert kwargs == {
        "trace_id": "report-job-1",
        "actor_id": "actor-1",
        "request_id": "request-1:track:prices",
        "request_deadline": "deadline-sentinel",
        "answer_mode": "report",
    }
    assert packet.track_id == "prices"
    assert packet.status.value == "complete"
    assert {item.kind for item in packet.items} == {
        ReportEvidenceKind.TABLE,
        ReportEvidenceKind.STATISTICS,
        ReportEvidenceKind.KNOWLEDGE,
    }
    assert all(
        item.kind is not ReportEvidenceKind.LIMITATION
        for item in packet.items
    )


def test_track_analysis_preserves_deterministic_derived_chart_evidence():
    def query_pipeline(query, **_kwargs):
        return QueryContext(
            query=query,
            cols=["period", "p_bal_gel"],
            rows=[["2025-01", 100.0], ["2025-02", 120.0]],
            provenance_cols=["period", "p_bal_gel"],
            provenance_rows=[["2025-01", 100.0], ["2025-02", 120.0]],
            provenance_refs=["query:track:prices"],
            provenance_source="pipeline",
            stats_hint="Observed price change was 20 percent.",
            chart_override_specs=[
                {
                    "type": "line",
                    "data": [
                        {"period": "2025-01", "mom_pct": 0.0},
                        {"period": "2025-02", "mom_pct": 20.0},
                    ],
                    "metadata": {"title": "Month-on-month price change"},
                }
            ],
            answer_mode="report",
        )

    packet = execute_report_track_analysis(
        _QUERY,
        _plan().tracks[0],
        query_pipeline=query_pipeline,
    )

    candidate = packet.chart_candidates[0]
    evidence = {
        item.evidence_ref: item for item in packet.items
    }[candidate.evidence_refs[0]]
    assert evidence.title == "Month-on-month price change"
    assert candidate.x_field == "period"
    assert candidate.series_fields == ["mom_pct"]


def _composition_track() -> ReportResearchTrack:
    """The security track, asking for a month-on-month share comparison."""

    payload = _plan().tracks[1].model_dump(mode="json")
    payload["requested_metrics"] = ["share_delta_mom"]
    payload["expected_exhibits"] = ["composition"]
    return ReportResearchTrack.model_validate(payload)


def _paired_panel_context(query: str) -> QueryContext:
    """The two frames the analyzer emits for a month-on-month chart.

    The change panel is built by renaming the levels it was computed from, so
    both frames carry the same display labels and only the title and the
    declared transform say which is which.
    """

    return QueryContext(
        query=query,
        cols=["date", "share_hydro", "share_thermal"],
        rows=[["2026-04", 0.61, 0.39], ["2026-05", 0.72, 0.28]],
        provenance_cols=["date", "share_hydro", "share_thermal"],
        provenance_rows=[["2026-04", 0.61, 0.39], ["2026-05", 0.72, 0.28]],
        provenance_refs=["query:track:generation"],
        provenance_source="pipeline",
        stats_hint="Hydro share rose over the month.",
        chart_override_specs=[
            {
                "type": "line",
                "data": [
                    {"date": "2026-04", "Share Hydro": 0.61, "Share Thermal": 0.39},
                    {"date": "2026-05", "Share Hydro": 0.72, "Share Thermal": 0.28},
                ],
                "metadata": {"title": "Observed Data", "role": "observed"},
            },
            {
                "type": "bar",
                "data": [
                    {"date": "2026-05", "Share Hydro": 18.03, "Share Thermal": -28.21},
                ],
                "metadata": {
                    "title": "MoM Change (%)",
                    "role": "derived",
                    "measureTransform": "mom_pct",
                },
            },
        ],
        answer_mode="report",
    )


def test_a_change_panel_declares_percent_not_the_unit_of_its_levels():
    """Both frames carry the same labels; only one holds those labels' values.

    Inference reads the labels, so without the builder saying otherwise the
    month-on-month percentages would be declared as shares and a claim on
    -28.21 would verify as a ratio.
    """

    packet = execute_report_track_analysis(
        _QUERY,
        _composition_track(),
        query_pipeline=lambda query, **_kwargs: _paired_panel_context(query),
    )

    units_by_title = {
        item.title: item.unit_by_column
        for item in packet.items
        if item.kind is ReportEvidenceKind.TABLE
    }
    assert units_by_title["Observed Data"]["Share Hydro"] == "ratio"
    assert units_by_title["MoM Change (%)"]["Share Hydro"] == "%"
    assert "date" not in units_by_title["MoM Change (%)"]


def test_a_composition_exhibit_is_drawn_from_levels_not_from_changes():
    """"share_delta_mom" names a subject and a comparison.

    On job 40e55527 the comparison words scored the change panel above the
    levels it came from, and the balancing composition exhibit was built from a
    single row of deltas, then omitted for having one category.
    """

    packet = execute_report_track_analysis(
        _QUERY,
        _composition_track(),
        query_pipeline=lambda query, **_kwargs: _paired_panel_context(query),
    )

    title_by_ref = {item.evidence_ref: item.title for item in packet.items}
    composition = next(
        candidate
        for candidate in packet.chart_candidates
        if candidate.purpose.value == "composition"
    )
    assert title_by_ref[composition.evidence_refs[0]] == "Observed Data"


def test_a_trend_exhibit_keeps_only_what_two_axes_can_carry():
    """An enriched balancing frame carries three kinds of number at once.

    On job 5cb4d210 the trend candidate took the top numeric columns of such a
    frame — prices in GEL/MWh, prices in USD/MWh, and shares as ratios — and
    the builder omitted the whole exhibit as REPORT_CHART_INCOMPATIBLE_UNITS.
    The reader got no chart rather than the prices the track was about.
    """

    from agent.report_charts import _axis_metadata
    from contracts.report_charts import ReportChartType

    payload = _plan().tracks[0].model_dump(mode="json")
    payload["requested_metrics"] = ["average_price"]
    payload["expected_exhibits"] = ["trend"]
    track = ReportResearchTrack.model_validate(payload)

    rows = [
        {
            "date": "2026-04",
            "p_bal_gel": 155.61,
            "p_bal_usd": 57.80,
            "share_import": 0.0942,
            "share_thermal_ppa": 0.2419,
        },
        {
            "date": "2026-05",
            "p_bal_gel": 137.86,
            "p_bal_usd": 51.65,
            "share_import": 0.0680,
            "share_thermal_ppa": 0.0,
        },
    ]

    def query_pipeline(query, **_kwargs):
        return QueryContext(
            query=query,
            cols=list(rows[0]),
            rows=[list(row.values()) for row in rows],
            provenance_cols=list(rows[0]),
            provenance_rows=[list(row.values()) for row in rows],
            provenance_refs=["query:track:balancing"],
            provenance_source="pipeline",
            stats_hint="The balancing price fell over the month.",
            answer_mode="report",
        )

    packet = execute_report_track_analysis(
        _QUERY,
        track,
        query_pipeline=query_pipeline,
    )

    candidate = next(
        item
        for item in packet.chart_candidates
        if item.purpose.value == "trend"
    )
    item = {
        entry.evidence_ref: entry for entry in packet.items
    }[candidate.evidence_refs[0]]
    assert candidate.series_fields, "trend exhibit lost every series"
    # The rule the builder applies, asserted against what the candidate chose.
    assert (
        _axis_metadata(
            ReportChartType.LINE,
            list(candidate.series_fields),
            item.unit_by_column,
        )
        is not None
    )


def test_track_analysis_rejects_pipeline_context_with_missing_derived_evidence(
    monkeypatch,
):
    monkeypatch.setattr(
        report_evidence,
        "ENABLE_REPORT_PARTIAL_TRACK_EVIDENCE",
        False,
    )

    def query_pipeline(query, **_kwargs):
        return QueryContext(
            query=query,
            cols=["period", "price_gel"],
            rows=[["2026-04", 100.0], ["2026-05", 120.0]],
            stats_hint="Mean observed price was 110 GEL/MWh.",
            terminal_outcome="clarification_required",
            missing_evidence_for_metrics=["mom_percent_change"],
            answer_mode="report",
        )

    with pytest.raises(ValueError, match="missing_derived_evidence"):
        execute_report_track_analysis(
            _QUERY,
            _plan().tracks[0],
            query_pipeline=query_pipeline,
        )


def _context_with_rows_and_a_missing_metric(query, **_kwargs):
    """A track that fetched its rows but could not derive one metric."""

    return QueryContext(
        query=query,
        cols=["period", "price_gel"],
        rows=[["2026-04", 100.0], ["2026-05", 120.0]],
        provenance_cols=["period", "price_gel"],
        provenance_rows=[["2026-04", 100.0], ["2026-05", 120.0]],
        provenance_refs=["query:track:prices"],
        provenance_source="pipeline",
        stats_hint="Mean observed price was 110 GEL/MWh.",
        missing_evidence_for_metrics=["mom_percent_change"],
        answer_mode="report",
    )


def test_partial_track_evidence_keeps_the_rows_and_declares_the_gap(
    monkeypatch,
):
    """A metric that could not be derived must not discard fetched rows.

    Job 827556eb lost supply_mix_trade after it had already fetched its rows,
    computed aggregates over ten numeric columns, and built its chart, because
    two MoM metrics were underived and report mode has no user to clarify with.
    """

    monkeypatch.setattr(
        report_evidence,
        "ENABLE_REPORT_PARTIAL_TRACK_EVIDENCE",
        True,
    )

    packet = execute_report_track_analysis(
        _QUERY,
        _plan().tracks[0],
        query_pipeline=_context_with_rows_and_a_missing_metric,
    )

    assert packet.items
    assert packet.status is ReportTrackStatus.PARTIAL
    assert "MISSING_DERIVED_METRIC_MOM_PERCENT_CHANGE" in packet.gaps


def test_partial_track_evidence_still_rejects_an_unusable_context(monkeypatch):
    """The flag concedes incompleteness, never usability."""

    monkeypatch.setattr(
        report_evidence,
        "ENABLE_REPORT_PARTIAL_TRACK_EVIDENCE",
        True,
    )

    def query_pipeline(query, **_kwargs):
        context = _context_with_rows_and_a_missing_metric(query)
        context.terminal_outcome = "clarification_required"
        return context

    with pytest.raises(ValueError, match="terminal_clarification_required"):
        execute_report_track_analysis(
            _QUERY,
            _plan().tracks[0],
            query_pipeline=query_pipeline,
        )


def test_a_declared_metric_gap_survives_the_baseline_merge(monkeypatch):
    """The document can only declare a gap that reaches the merged packet.

    EXPECTED_EXHIBIT_ gaps are stripped by the merge; this one must not be.
    """

    monkeypatch.setattr(
        report_evidence,
        "ENABLE_REPORT_PARTIAL_TRACK_EVIDENCE",
        True,
    )
    track = _plan().tracks[0]
    analysis = execute_report_track_analysis(
        _QUERY,
        track,
        query_pipeline=_context_with_rows_and_a_missing_metric,
    )
    baseline = execute_report_track_analysis(
        _QUERY,
        track,
        query_pipeline=lambda query, **_kwargs: QueryContext(
            query=query,
            cols=["period", "price_gel"],
            rows=[["2026-03", 90.0]],
            provenance_cols=["period", "price_gel"],
            provenance_rows=[["2026-03", 90.0]],
            provenance_refs=["query:baseline:prices"],
            provenance_source="pipeline",
            stats_hint="Mean observed price was 90 GEL/MWh.",
            answer_mode="report",
        ),
    )

    merged = merge_report_track_analysis_packet(track, baseline, analysis)

    assert "MISSING_DERIVED_METRIC_MOM_PERCENT_CHANGE" in merged.gaps
    assert merged.items


def test_observations_stay_inside_the_packet_contract_cap():
    """Overflowing the cap costs the whole track, not a trimmed packet.

    The cap was enforced only inside the table branch; a narrative item took
    the ``continue`` above it and skipped the check, so a rich track could
    build a packet its own contract rejects (job 5e6b0cf3).
    """
    from agent.report_evidence import make_report_narrative_evidence_item
    from agent.report_research_execution import (
        _numeric_observations,
        _packet_from_items,
    )
    from contracts.report_research import REPORT_PACKET_MAX_OBSERVATIONS

    wide_table = ReportEvidenceItem(
        evidence_ref="evidence:table:" + "a" * 16,
        kind=ReportEvidenceKind.TABLE,
        title="Wide analytical table",
        source="tool",
        columns=["period", *[f"metric_{index}_gwh" for index in range(30)]],
        rows=[
            {
                "period": f"2026-{month:02d}",
                **{
                    f"metric_{index}_gwh": float(index + month)
                    for index in range(30)
                },
            }
            for month in range(1, 4)
        ],
        unit_by_column={
            f"metric_{index}_gwh": "GWh" for index in range(30)
        },
        total_row_count=3,
        truncated=False,
    )
    narrative = [
        make_report_narrative_evidence_item(
            kind=ReportEvidenceKind.STATISTICS,
            title=f"Computed statistics {index}",
            source="derived",
            content=f"Statistics payload {index}.",
        )
        for index in range(6)
    ]

    observations = _numeric_observations([wide_table, *narrative])

    assert len(observations) <= REPORT_PACKET_MAX_OBSERVATIONS
    # The packet is the authority; building one must not raise.
    packet = _packet_from_items(_plan().tracks[0], [wide_table, *narrative])
    assert packet.observations == observations[: len(packet.observations)]


def test_track_analysis_reserves_packet_capacity_for_analysis_and_knowledge():
    supporting = {
        f"support_{index:02d}": {
            "tool": f"support_tool_{index:02d}",
            "cols": ["period", "price_gel"],
            "rows": [["2025-01", float(index + 1)]],
            "provenance_refs": [f"query:support:{index:02d}"],
        }
        for index in range(12)
    }

    def query_pipeline(query, **_kwargs):
        return QueryContext(
            query=query,
            cols=["period", "price_gel"],
            rows=[["2025-01", 100.0]],
            evidence_collected=supporting,
            stats_hint="Mean observed price was 100 GEL/MWh.",
            summary_domain_knowledge="Applicable market rules govern prices.",
            answer_mode="report",
        )

    packet = execute_report_track_analysis(
        _QUERY,
        _plan().tracks[0],
        query_pipeline=query_pipeline,
    )

    kinds = {item.kind for item in packet.items}
    assert ReportEvidenceKind.TABLE in kinds
    assert ReportEvidenceKind.STATISTICS in kinds
    assert ReportEvidenceKind.KNOWLEDGE in kinds
    assert len(packet.items) == 12


def test_track_analysis_merge_keeps_deterministic_table_and_track_findings():
    plan = _plan()
    baseline = execute_report_research(
        _QUERY,
        plan,
        max_workers=3,
        collectors={
            ReportCollectorId.PRICES: lambda *_args: _table_output(
                ReportCollectorId.PRICES
            ),
            ReportCollectorId.GENERATION_MIX: lambda *_args: _table_output(
                ReportCollectorId.GENERATION_MIX
            ),
            ReportCollectorId.VECTOR_KNOWLEDGE: lambda *_args: (
                _knowledge_output()
            ),
        },
    )[0]

    def query_pipeline(query, **_kwargs):
        return QueryContext(
            query=query,
            stats_hint="Mean observed price was 110 GEL/MWh.",
            summary_domain_knowledge="Applicable market rules govern prices.",
            answer_mode="report",
        )

    analysis = execute_report_track_analysis(
        _QUERY,
        plan.tracks[0],
        query_pipeline=query_pipeline,
    )
    merged = merge_report_track_analysis_packet(
        plan.tracks[0],
        baseline,
        analysis,
    )

    assert {item.kind for item in merged.items} == {
        ReportEvidenceKind.TABLE,
        ReportEvidenceKind.STATISTICS,
        ReportEvidenceKind.KNOWLEDGE,
    }
    assert merged.status.value == "complete"
    assert merged.gaps == []


def test_research_executes_unique_collectors_in_parallel_and_keeps_track_order():
    barrier = threading.Barrier(2)
    calls = []

    def prices(_query, _scope):
        calls.append("prices")
        barrier.wait(timeout=2)
        return _table_output(ReportCollectorId.PRICES)

    def generation(_query, _scope):
        calls.append("generation_mix")
        barrier.wait(timeout=2)
        return _table_output(ReportCollectorId.GENERATION_MIX)

    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=2,
        collectors={
            ReportCollectorId.PRICES: prices,
            ReportCollectorId.GENERATION_MIX: generation,
            ReportCollectorId.VECTOR_KNOWLEDGE: (
                lambda _query, _scope: _knowledge_output()
            ),
        },
    )

    assert set(calls) == {"prices", "generation_mix"}
    assert [packet.track_id for packet in packets] == [
        "prices",
        "security",
        "market_model",
    ]
    assert all(packet.status.value == "complete" for packet in packets)
    assert packets[0].numeric_observation_count >= 4


def test_collector_failure_becomes_a_typed_partial_packet_not_global_failure():
    def failed_knowledge(_query, _scope):
        raise RuntimeError("provider detail must not enter evidence")

    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=3,
        collectors={
            ReportCollectorId.PRICES: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.PRICES
                )
            ),
            ReportCollectorId.GENERATION_MIX: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.GENERATION_MIX
                )
            ),
            ReportCollectorId.VECTOR_KNOWLEDGE: failed_knowledge,
        },
    )

    security = packets[1]
    market_model = packets[2]
    assert security.status.value == "partial"
    assert market_model.status.value == "failed"
    assert "provider detail" not in " ".join(
        security.gaps + market_model.gaps
    )


def test_knowledge_collector_runs_per_track_with_track_research_questions():
    knowledge_queries = []

    def knowledge(query, _scope):
        knowledge_queries.append(query)
        return _knowledge_output()

    execute_report_research(
        _QUERY,
        _plan(),
        max_workers=3,
        collectors={
            ReportCollectorId.PRICES: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.PRICES
                )
            ),
            ReportCollectorId.GENERATION_MIX: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.GENERATION_MIX
                )
            ),
            ReportCollectorId.VECTOR_KNOWLEDGE: knowledge,
        },
    )

    assert len(knowledge_queries) == 2
    assert any(
        "measurable supply-security risks" in query
        for query in knowledge_queries
    )
    assert any(
        "deregulation stage and target model" in query
        for query in knowledge_queries
    )
    assert all(_QUERY in query for query in knowledge_queries)


def test_identical_table_requests_are_deduplicated_across_tracks():
    payload = _research_plan_payload(
        query_digest=hashlib.sha256(_QUERY.encode("utf-8")).hexdigest()
    )
    second_topic = {
        **payload["request_topics"][0],
        "topic_id": "price_context",
        "label": "Price context",
    }
    second_track = deepcopy(payload["tracks"][0])
    second_track.update(
        {
            "track_id": "price_context",
            "title": "Price context",
            "topic_ids": ["price_context"],
            "research_questions": [
                "What is the minimum observed electricity price?"
            ],
            "requested_metrics": ["minimum_price"],
            "expected_exhibits": [],
        }
    )
    payload["request_topics"] = [payload["request_topics"][0], second_topic]
    payload["tracks"] = [payload["tracks"][0], second_track]
    plan = ReportResearchPlan.model_validate(payload)
    calls = []

    def prices(query, _scope):
        calls.append(query)
        return _table_output(ReportCollectorId.PRICES)

    packets = execute_report_research(
        _QUERY,
        plan,
        max_workers=2,
        collectors={ReportCollectorId.PRICES: prices},
    )

    assert calls == [_QUERY]
    assert [packet.track_id for packet in packets] == [
        "prices",
        "price_context",
    ]


def test_requested_metrics_limit_deterministic_metric_operations():
    plan_payload = _research_plan_payload(
        query_digest=hashlib.sha256(_QUERY.encode("utf-8")).hexdigest()
    )
    plan_payload["tracks"] = [plan_payload["tracks"][0]]
    plan_payload["request_topics"] = [plan_payload["request_topics"][0]]
    plan_payload["tracks"][0]["requested_metrics"] = ["average_price"]
    plan = ReportResearchPlan.model_validate(plan_payload)

    packets = execute_report_research(
        _QUERY,
        plan,
        max_workers=1,
        collectors={
            ReportCollectorId.PRICES: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.PRICES
                )
            ),
        },
    )

    operations = {
        metric.operation.value
        for observation in packets[0].observations
        for metric in observation.metric_values
    }
    assert operations == {"mean"}


def test_metrics_are_not_emitted_for_columns_without_a_declared_unit():
    """A metric on a unit-less column is a number the writer cannot ground.

    The grounding validator refuses any claim whose column has no unit, so
    advertising such a metric -- previously under a fabricated "value" unit --
    points the writer at numbers that can only fail validation.
    """

    item = _table_item()
    item["columns"] = ["period", "price_gel", "mystery_metric"]
    item["rows"] = [
        {"period": "2025-01", "price_gel": 100.0, "mystery_metric": 7.0},
        {"period": "2025-02", "price_gel": 120.0, "mystery_metric": 9.0},
    ]
    item["unit_by_column"] = {"price_gel": "GEL/MWh"}
    output = ReportCollectorOutput(
        collector_id=ReportCollectorId.PRICES,
        items=(ReportEvidenceItem.model_validate(item),),
    )

    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=1,
        collectors={
            ReportCollectorId.PRICES: (lambda _query, _scope: output),
        },
    )

    metrics = [
        metric
        for packet in packets
        for observation in packet.observations
        for metric in observation.metric_values
    ]
    assert metrics, "the declared column should still produce metrics"
    assert all(metric.unit != "value" for metric in metrics)
    assert not [
        metric for metric in metrics if "mystery" in metric.metric_id
    ], "a column with no declared unit must not be advertised as claimable"


def test_packet_metrics_and_manifest_are_deterministic_and_deduplicated():
    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=1,
        collectors={
            ReportCollectorId.PRICES: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.PRICES
                )
            ),
            ReportCollectorId.GENERATION_MIX: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.GENERATION_MIX
                )
            ),
            ReportCollectorId.VECTOR_KNOWLEDGE: (
                lambda _query, _scope: _knowledge_output()
            ),
        },
    )
    price_metrics = {
        metric.operation.value: metric.value
        for observation in packets[0].observations
        for metric in observation.metric_values
    }
    assert price_metrics == {
        "mean": 110.0,
        "minimum": 100.0,
        "maximum": 120.0,
        "percent_change": 20.0,
    }

    manifest = consolidate_report_evidence_packets(_QUERY, packets)
    assert manifest.query_digest == hashlib.sha256(
        _QUERY.encode("utf-8")
    ).hexdigest()
    assert len(
        {
            item.evidence_ref
            for packet in packets
            for item in packet.items
        }
    ) == 2
    assert len(manifest.items) == 3
    assert manifest.items[-1].kind.value == "limitation"


def test_table_metrics_are_chronological_when_tools_return_latest_first():
    table = _table_item()
    table["rows"] = list(reversed(table["rows"]))
    output = ReportCollectorOutput(
        collector_id=ReportCollectorId.PRICES,
        items=(ReportEvidenceItem.model_validate(table),),
    )
    packets = execute_report_research(
        _QUERY,
        _plan(),
        max_workers=1,
        collectors={
            ReportCollectorId.PRICES: lambda _query, _scope: output,
            ReportCollectorId.GENERATION_MIX: (
                lambda _query, _scope: _table_output(
                    ReportCollectorId.GENERATION_MIX
                )
            ),
            ReportCollectorId.VECTOR_KNOWLEDGE: (
                lambda _query, _scope: _knowledge_output()
            ),
        },
    )
    percent_change = next(
        metric.value
        for observation in packets[0].observations
        for metric in observation.metric_values
        if metric.operation.value == "percent_change"
    )

    assert percent_change == 20.0


def test_consolidation_reserves_space_for_the_limitation_item():
    items = [
        ReportEvidenceItem.model_validate(
            {
                **_knowledge_output().items[0].model_dump(mode="json"),
                "evidence_ref": f"evidence:knowledge:{index:016x}",
                "content": f"Approved bounded knowledge passage number {index}.",
            }
        )
        for index in range(32)
    ]
    packets = [
        ReportEvidencePacket.model_validate(
            {
                "contract_version": "report-evidence-packet-v1",
                "track_id": "market_model",
                "status": "complete",
                "items": items[:12],
                "observations": [
                    {
                        "observation_id": "documented_context",
                        "statement": (
                            "Approved knowledge evidence was retrieved for "
                            "the requested market-model topic."
                        ),
                        "evidence_refs": [items[0].evidence_ref],
                        "metric_values": [],
                    }
                ],
                "gaps": [],
                "chart_candidates": [],
            }
        ),
        ReportEvidencePacket.model_validate(
            {
                "contract_version": "report-evidence-packet-v1",
                "track_id": "market_rules",
                "status": "complete",
                "items": items[12:24],
                "observations": [
                    {
                        "observation_id": "documented_rules",
                        "statement": (
                            "Approved knowledge evidence was retrieved for "
                            "the requested market-rules topic."
                        ),
                        "evidence_refs": [items[12].evidence_ref],
                        "metric_values": [],
                    }
                ],
                "gaps": [],
                "chart_candidates": [],
            }
        ),
        ReportEvidencePacket.model_validate(
            {
                "contract_version": "report-evidence-packet-v1",
                "track_id": "market_status",
                "status": "complete",
                "items": items[24:],
                "observations": [
                    {
                        "observation_id": "documented_status",
                        "statement": (
                            "Approved knowledge evidence was retrieved for "
                            "the requested market-status topic."
                        ),
                        "evidence_refs": [items[24].evidence_ref],
                        "metric_values": [],
                    }
                ],
                "gaps": [],
                "chart_candidates": [],
            }
        ),
    ]

    manifest = consolidate_report_evidence_packets(_QUERY, packets)

    assert len(manifest.items) == 32
    assert manifest.items[-1].kind is ReportEvidenceKind.LIMITATION


def test_default_price_collector_uses_query_metric_and_currency(
    monkeypatch,
):
    captured = {}

    def fake_prices(**kwargs):
        captured.update(kwargs)
        table = _table_item()
        return None, table["columns"], [
            tuple(row[column] for column in table["columns"])
            for row in table["rows"]
        ]

    monkeypatch.setattr(
        "agent.report_research_execution.get_prices",
        fake_prices,
    )
    output = DEFAULT_REPORT_COLLECTORS[ReportCollectorId.PRICES](
        "Compare deregulated electricity prices in USD.",
        _plan().scope,
    )

    assert output.items
    assert captured["metric"] == "deregulated"
    assert captured["currency"] == "usd"


def test_default_vector_collector_distinguishes_unavailable_from_no_evidence(
    monkeypatch,
):
    def unavailable(*_args, **_kwargs):
        return VectorKnowledgeBundle(
            query="query",
            retrieval_mode=VectorKnowledgeMode.active,
            strategy=RetrievalStrategy.dense_with_deterministic_rerank,
            top_k=6,
            outcome=VectorRetrievalOutcome.unavailable,
            failure=VectorRetrievalFailure(
                stage=VectorRetrievalFailureStage.vector_search,
                reason="ConnectionError",
            ),
        )

    monkeypatch.setattr(
        "agent.report_research_execution.retrieve_vector_knowledge",
        unavailable,
    )
    failed = DEFAULT_REPORT_COLLECTORS[
        ReportCollectorId.VECTOR_KNOWLEDGE
    ]("query", _plan().scope)

    monkeypatch.setattr(
        "agent.report_research_execution.retrieve_vector_knowledge",
        lambda *_args, **_kwargs: VectorKnowledgeBundle(
            query="query",
            retrieval_mode=VectorKnowledgeMode.active,
            strategy=(
                RetrievalStrategy.dense_with_deterministic_rerank
            ),
            top_k=6,
            outcome=VectorRetrievalOutcome.no_matches,
        ),
    )
    empty = DEFAULT_REPORT_COLLECTORS[
        ReportCollectorId.VECTOR_KNOWLEDGE
    ]("query", _plan().scope)

    assert failed.failed is True
    assert failed.gaps == ("COLLECTOR_VECTOR_KNOWLEDGE_FAILED",)
    assert empty.failed is False
    assert empty.gaps == ("COLLECTOR_VECTOR_KNOWLEDGE_NO_EVIDENCE",)


def test_vector_collector_preserves_primary_and_reference_evidence_roles(
    monkeypatch,
):
    primary = VectorChunkRecord(
        id="primary-1",
        document_id="doc-1",
        document_title="Market Rules",
        source_key="market-rules",
        section_title="Direct match",
        text_content="Primary matched evidence.",
    )
    reference = VectorChunkRecord(
        id="reference-1",
        document_id="doc-1",
        document_title="Market Rules",
        source_key="market-rules",
        section_title="Referenced article",
        text_content="Explicitly referenced supporting evidence.",
    )
    adjacent = VectorChunkRecord(
        id="adjacent-1",
        document_id="doc-1",
        document_title="Market Rules",
        source_key="market-rules",
        section_title="Adjacent context",
        text_content="Context that must not become report evidence.",
    )
    bundle = VectorKnowledgeBundle(
        query="query",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.dense_with_deterministic_rerank,
        strategy_version=(
            RetrievalStrategyVersion.dense_cosine_rerank_v2
        ),
        top_k=6,
        chunk_count=1,
        chunks=[primary],
        reference_chunks=[reference, primary],
        adjacent_chunks=[adjacent],
        outcome=VectorRetrievalOutcome.matches,
    )
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "on")
    monkeypatch.setattr(
        "agent.report_research_execution.retrieve_vector_knowledge",
        lambda *_args, **_kwargs: bundle,
    )

    output = DEFAULT_REPORT_COLLECTORS[
        ReportCollectorId.VECTOR_KNOWLEDGE
    ]("query", _plan().scope)

    assert [item.knowledge_role for item in output.items] == [
        ReportKnowledgeEvidenceRole.primary,
        ReportKnowledgeEvidenceRole.supporting_reference,
    ]
    assert [item.content for item in output.items] == [
        "Primary matched evidence.",
        "Explicitly referenced supporting evidence.",
    ]
    assert all(
        item.source == "vector:dense_cosine_rerank_v2"
        for item in output.items
    )
    assert output.items[0].provenance_refs == [
        "vector:primary:market-rules:primary-1"
    ]
    assert output.items[1].provenance_refs == [
        "vector:supporting_reference:market-rules:reference-1"
    ]


def test_vector_collector_keeps_reference_shadow_out_of_report_evidence(
    monkeypatch,
):
    primary = VectorChunkRecord(
        id="primary-1",
        document_id="doc-1",
        text_content="Primary matched evidence.",
    )
    reference = VectorChunkRecord(
        id="reference-1",
        document_id="doc-1",
        text_content="Shadow reference evidence.",
    )
    bundle = VectorKnowledgeBundle(
        query="query",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.dense_with_deterministic_rerank,
        strategy_version=(
            RetrievalStrategyVersion.dense_cosine_rerank_v2
        ),
        top_k=6,
        chunk_count=1,
        chunks=[primary],
        reference_chunks=[reference],
        outcome=VectorRetrievalOutcome.matches,
    )
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "shadow")
    monkeypatch.setattr(
        "agent.report_research_execution.retrieve_vector_knowledge",
        lambda *_args, **_kwargs: bundle,
    )

    output = DEFAULT_REPORT_COLLECTORS[
        ReportCollectorId.VECTOR_KNOWLEDGE
    ]("query", _plan().scope)

    assert len(output.items) == 1
    assert (
        output.items[0].knowledge_role
        is ReportKnowledgeEvidenceRole.primary
    )


def test_research_execution_rejects_unbounded_worker_counts():
    for invalid in (0, 9):
        try:
            execute_report_research(
                _QUERY,
                _plan(),
                max_workers=invalid,
                collectors={},
            )
        except ValueError as exc:
            assert "max_workers" in str(exc)
        else:
            raise AssertionError("invalid worker count was accepted")


def test_consolidation_accepts_closed_packet_contracts():
    packet = ReportEvidencePacket.model_validate(
        {
            "contract_version": "report-evidence-packet-v1",
            "track_id": "market_model",
            "status": "complete",
            "items": [_knowledge_output().items[0]],
            "observations": [
                {
                    "observation_id": "documented_context",
                    "statement": (
                        "Approved knowledge evidence was retrieved for the "
                        "requested market-model topic."
                    ),
                    "evidence_refs": [
                        _knowledge_output().items[0].evidence_ref
                    ],
                    "metric_values": [],
                }
            ],
            "gaps": [],
            "chart_candidates": [],
        }
    )

    manifest = consolidate_report_evidence_packets(_QUERY, [packet])
    assert manifest.items[0].evidence_ref == packet.items[0].evidence_ref
