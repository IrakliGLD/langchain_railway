"""Bounded parallel execution of deterministic report research collectors."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real

from agent.report_evidence import (
    build_report_manifest_from_items,
    make_report_narrative_evidence_item,
    make_report_table_evidence_item,
)
from agent.router import extract_currency, extract_price_metric
from agent.tools.composition_tools import get_balancing_composition
from agent.tools.generation_tools import get_generation_mix
from agent.tools.price_tools import get_prices
from agent.tools.tariff_tools import get_tariffs
from contracts.report import REPORT_MAX_EXHIBITS, ReportChartPurpose
from contracts.report_evidence import (
    ReportEvidenceItem,
    ReportEvidenceKind,
    ReportEvidenceManifest,
    ReportKnowledgeEvidenceRole,
)
from contracts.report_research import (
    ReportChartCandidate,
    ReportCollectorId,
    ReportEvidenceObservation,
    ReportEvidencePacket,
    ReportMetricOperation,
    ReportMetricValue,
    ReportResearchPlan,
    ReportResearchScope,
    ReportResearchTrack,
    ReportTrackStatus,
)
from contracts.vector_knowledge import (
    VectorKnowledgeMode,
    VectorRetrievalOutcome,
    VectorRetrievalTier,
)
from knowledge.vector_retrieval import (
    get_reference_expansion_mode,
    retrieve_vector_knowledge,
)
from utils.request_deadline import (
    bind_request_execution_scope_snapshot,
    current_request_execution_scope,
)

Collector = Callable[
    [str, ReportResearchScope],
    "ReportCollectorOutput",
]
CollectorRequestKey = tuple[ReportCollectorId, str, str]


@dataclass(frozen=True, slots=True)
class ReportCollectorRequest:
    collector_id: ReportCollectorId
    query: str
    scope: ReportResearchScope


@dataclass(frozen=True, slots=True)
class ReportCollectorOutput:
    collector_id: ReportCollectorId
    items: tuple[ReportEvidenceItem, ...] = ()
    gaps: tuple[str, ...] = ()
    failed: bool = False


def _granularity(scope: ReportResearchScope) -> str:
    return "yearly" if scope.grain.value == "year" else "monthly"


def _date_params(scope: ReportResearchScope) -> dict[str, str | None]:
    return {
        "start_date": (
            scope.period_start.isoformat()
            if scope.period_start is not None
            else None
        ),
        "end_date": (
            scope.period_end.isoformat()
            if scope.period_end is not None
            else None
        ),
    }


def _table_collector_output(
    *,
    collector_id: ReportCollectorId,
    query: str,
    title: str,
    source: str,
    result,
    gaps: tuple[str, ...] = (),
) -> ReportCollectorOutput:
    _frame, columns, rows = result
    rows = _chronological_rows(columns, rows)
    item = make_report_table_evidence_item(
        query=query,
        title=title,
        source=source,
        columns=columns,
        rows=rows,
        provenance_refs=[f"tool:{source}"],
    )
    if item is None:
        return ReportCollectorOutput(
            collector_id=collector_id,
            gaps=(f"COLLECTOR_{collector_id.value.upper()}_NO_EVIDENCE",),
        )
    return ReportCollectorOutput(
        collector_id=collector_id,
        items=(item,),
        gaps=gaps,
    )


def _chronological_rows(columns, rows):
    materialized = list(rows)
    time_index = next(
        (
            index
            for index, column in enumerate(columns)
            if any(
                token in str(column).casefold()
                for token in ("date", "period", "year", "month")
            )
        ),
        None,
    )
    if time_index is None:
        return materialized

    def key(row):
        value = (
            row.get(columns[time_index])
            if isinstance(row, dict)
            else row[time_index]
        )
        return str(value)

    return sorted(materialized, key=key)


def _collect_prices(
    query: str,
    scope: ReportResearchScope,
) -> ReportCollectorOutput:
    query_lower = query.casefold()
    currency = extract_currency(query_lower)
    if not any(
        signal in query_lower
        for signal in (
            "usd",
            "dollar",
            "gel",
            "lari",
            "დოლარ",
            "ლარ",
            "доллар",
            "лари",
        )
    ):
        currency = "both"
    return _table_collector_output(
        collector_id=ReportCollectorId.PRICES,
        query=query,
        title="Observed electricity prices",
        source="get_prices",
        result=get_prices(
            **_date_params(scope),
            currency=currency,
            metric=extract_price_metric(query_lower),
            granularity=_granularity(scope),
        ),
        gaps=(
            (
                f"REQUESTED_GRAIN_{scope.grain.value.upper()}_"
                "COARSENED_TO_MONTH"
            ),
        )
        if scope.grain.value not in {"none", "month", "year"}
        else (),
    )


def _collect_balancing_composition(
    query: str,
    scope: ReportResearchScope,
) -> ReportCollectorOutput:
    return _table_collector_output(
        collector_id=ReportCollectorId.BALANCING_COMPOSITION,
        query=query,
        title="Observed balancing-market composition",
        source="get_balancing_composition",
        result=get_balancing_composition(**_date_params(scope)),
        gaps=(
            (
                f"REQUESTED_GRAIN_{scope.grain.value.upper()}_"
                "UNAVAILABLE"
            ),
        )
        if scope.grain.value not in {"none", "month"}
        else (),
    )


def _collect_generation_mix(
    query: str,
    scope: ReportResearchScope,
) -> ReportCollectorOutput:
    return _table_collector_output(
        collector_id=ReportCollectorId.GENERATION_MIX,
        query=query,
        title="Observed generation and supply mix",
        source="get_generation_mix",
        result=get_generation_mix(
            **_date_params(scope),
            mode="share",
            share_basis="generation",
            granularity=_granularity(scope),
        ),
        gaps=(
            (
                f"REQUESTED_GRAIN_{scope.grain.value.upper()}_"
                "COARSENED_TO_MONTH"
            ),
        )
        if scope.grain.value not in {"none", "month", "year"}
        else (),
    )


def _collect_tariffs(
    query: str,
    scope: ReportResearchScope,
) -> ReportCollectorOutput:
    return _table_collector_output(
        collector_id=ReportCollectorId.TARIFFS,
        query=query,
        title="Observed regulated tariffs",
        source="get_tariffs",
        result=get_tariffs(
            **_date_params(scope),
            currency="both",
        ),
        gaps=(
            (
                f"REQUESTED_GRAIN_{scope.grain.value.upper()}_"
                "UNAVAILABLE"
            ),
        )
        if scope.grain.value not in {"none", "month"}
        else (),
    )


def _collect_vector_knowledge(
    query: str,
    _scope: ReportResearchScope,
) -> ReportCollectorOutput:
    bundle = retrieve_vector_knowledge(
        query,
        retrieval_mode=VectorKnowledgeMode.active,
        tier=VectorRetrievalTier.FULL,
    )
    if bundle.outcome is VectorRetrievalOutcome.unavailable:
        return ReportCollectorOutput(
            collector_id=ReportCollectorId.VECTOR_KNOWLEDGE,
            gaps=("COLLECTOR_VECTOR_KNOWLEDGE_FAILED",),
            failed=True,
        )
    role_chunks = [
        (ReportKnowledgeEvidenceRole.primary, chunk)
        for chunk in bundle.chunks[:6]
    ]
    if get_reference_expansion_mode() == "on":
        role_chunks.extend(
            (
                ReportKnowledgeEvidenceRole.supporting_reference,
                chunk,
            )
            for chunk in bundle.reference_chunks[:4]
        )
    seen_chunk_ids: set[str] = set()
    items_list: list[ReportEvidenceItem] = []
    for role, chunk in role_chunks:
        if chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.id)
        items_list.append(
            make_report_narrative_evidence_item(
                kind=ReportEvidenceKind.KNOWLEDGE,
                title=(
                    chunk.document_title
                    or chunk.section_title
                    or "Approved knowledge passage"
                )[:200],
                source=f"vector:{bundle.strategy_version.value}",
                knowledge_role=role,
                provenance_refs=[
                    "vector:"
                    f"{role.value}:"
                    f"{chunk.source_key or chunk.document_id}:"
                    f"{chunk.id}"
                ],
                content=chunk.text_content,
            )
        )
    items = tuple(items_list)
    if not items:
        return ReportCollectorOutput(
            collector_id=ReportCollectorId.VECTOR_KNOWLEDGE,
            gaps=("COLLECTOR_VECTOR_KNOWLEDGE_NO_EVIDENCE",),
        )
    return ReportCollectorOutput(
        collector_id=ReportCollectorId.VECTOR_KNOWLEDGE,
        items=items,
    )


def _unsupported_collector(
    collector_id: ReportCollectorId,
) -> Collector:
    def collect(
        _query: str,
        _scope: ReportResearchScope,
    ) -> ReportCollectorOutput:
        return ReportCollectorOutput(
            collector_id=collector_id,
            gaps=(f"COLLECTOR_{collector_id.value.upper()}_UNAVAILABLE",),
        )

    return collect


DEFAULT_REPORT_COLLECTORS: Mapping[ReportCollectorId, Collector] = {
    ReportCollectorId.PRICES: _collect_prices,
    ReportCollectorId.BALANCING_COMPOSITION: (
        _collect_balancing_composition
    ),
    ReportCollectorId.GENERATION_MIX: _collect_generation_mix,
    ReportCollectorId.TARIFFS: _collect_tariffs,
    ReportCollectorId.VECTOR_KNOWLEDGE: _collect_vector_knowledge,
    ReportCollectorId.FORECAST_ENGINE: _unsupported_collector(
        ReportCollectorId.FORECAST_ENGINE
    ),
    ReportCollectorId.SCENARIO_ENGINE: _unsupported_collector(
        ReportCollectorId.SCENARIO_ENGINE
    ),
}


def _safe_metric_id(column: str, operation: str) -> str:
    base = re.sub(r"[^a-z0-9_]+", "_", column.casefold()).strip("_")
    if not base or not base[0].isalpha():
        base = f"metric_{base}"
    return f"{base[:40]}_{operation}"[:64]


def _display_number(value: float, *, percent: bool = False) -> str:
    display = f"{value:.6g}"
    return f"{display}%" if percent else display


def _requested_metric_operations(
    requested_metrics: Sequence[str],
) -> set[ReportMetricOperation]:
    tokens = {
        token
        for metric in requested_metrics
        for token in re.findall(r"[a-z]+", metric.casefold())
    }
    operations: set[ReportMetricOperation] = set()
    if tokens & {"average", "avg", "mean"}:
        operations.add(ReportMetricOperation.MEAN)
    if tokens & {"minimum", "min"}:
        operations.add(ReportMetricOperation.MINIMUM)
    if tokens & {"maximum", "max"}:
        operations.add(ReportMetricOperation.MAXIMUM)
    if "change" in tokens and tokens & {"percent", "percentage"}:
        operations.add(ReportMetricOperation.PERCENT_CHANGE)
    return operations


def _numeric_observations(
    items: Sequence[ReportEvidenceItem],
    requested_metrics: Sequence[str] = (),
) -> list[ReportEvidenceObservation]:
    observations: list[ReportEvidenceObservation] = []
    requested_operations = _requested_metric_operations(requested_metrics)
    for item in items:
        if item.kind is not ReportEvidenceKind.TABLE:
            observations.append(
                ReportEvidenceObservation(
                    observation_id=(
                        f"documented_{item.evidence_ref.rsplit(':', 1)[-1][:8]}"
                    ),
                    statement=(
                        "Approved knowledge evidence was retrieved for "
                        f"{item.title}."
                    ),
                    evidence_refs=[item.evidence_ref],
                    metric_values=[],
                )
            )
            continue
        rows = _chronological_rows(item.columns, item.rows)
        numeric_columns = [
            column
            for column in item.columns
            if any(
                isinstance(row.get(column), Real)
                and not isinstance(row.get(column), bool)
                for row in rows
            )
        ]
        time_columns = [
            column
            for column in item.columns
            if any(
                token in column.casefold()
                for token in ("date", "period", "year", "month")
            )
        ]
        category_columns = [
            column
            for column in item.columns
            if column not in numeric_columns and column not in time_columns
        ]
        grouped_rows: list[tuple[str, list[dict]]] = [("", rows)]
        if time_columns and category_columns:
            materialized_groups: dict[tuple[str, ...], list[dict]] = {}
            for row in rows:
                key = tuple(
                    str(row.get(column)) for column in category_columns
                )
                materialized_groups.setdefault(key, []).append(row)
            grouped_rows = [
                (
                    ", ".join(
                        f"{column}={value}"
                        for column, value in zip(
                            category_columns,
                            key,
                            strict=True,
                        )
                    )[:200],
                    group_rows,
                )
                for key, group_rows in sorted(
                    materialized_groups.items(),
                    key=lambda entry: entry[0],
                )
            ]
        for context, observation_rows in grouped_rows:
            for column in numeric_columns:
                values = [
                    float(row[column])
                    for row in observation_rows
                    if (
                        isinstance(row.get(column), Real)
                        and not isinstance(row.get(column), bool)
                        and math.isfinite(float(row[column]))
                    )
                ]
                if not values:
                    continue
                unit = item.unit_by_column.get(column, "value")
                mean_value = sum(values) / len(values)
                specs: list[
                    tuple[ReportMetricOperation, float, str, str]
                ] = [
                    (
                        ReportMetricOperation.MEAN,
                        mean_value,
                        _display_number(mean_value),
                        unit,
                    ),
                    (
                        ReportMetricOperation.MINIMUM,
                        min(values),
                        _display_number(min(values)),
                        unit,
                    ),
                    (
                        ReportMetricOperation.MAXIMUM,
                        max(values),
                        _display_number(max(values)),
                        unit,
                    ),
                ]
                if len(values) >= 2 and values[0] != 0:
                    percent_change = (
                        (values[-1] - values[0]) / abs(values[0])
                    ) * 100
                    specs.append(
                        (
                            ReportMetricOperation.PERCENT_CHANGE,
                            percent_change,
                            _display_number(percent_change, percent=True),
                            "%",
                        )
                    )
                if requested_operations:
                    specs = [
                        spec
                        for spec in specs
                        if spec[0] in requested_operations
                    ]
                metric_context = f" for {context}" if context else ""
                metrics = [
                    ReportMetricValue(
                        metric_id=_safe_metric_id(
                            column,
                            operation.value,
                        ),
                        label=(
                            f"{operation.value.replace('_', ' ')} {column}"
                            f"{metric_context}"
                        )[:160],
                        value=value,
                        display_value=display,
                        unit=metric_unit,
                        operation=operation,
                        evidence_refs=[item.evidence_ref],
                    )
                    for operation, value, display, metric_unit in specs
                ]
                context_digest = hashlib.sha256(
                    context.encode("utf-8")
                ).hexdigest()[:6]
                observations.append(
                    ReportEvidenceObservation(
                        observation_id=(
                            "observed_"
                            f"{_safe_metric_id(column, 'values')[:30]}_"
                            f"{context_digest}_"
                            f"{item.evidence_ref.rsplit(':', 1)[-1][:8]}"
                        ),
                        statement=(
                            f"Observed {item.title}"
                            f"{metric_context} column {column} contains "
                            f"{len(values)} numeric values."
                        )[:1200],
                        evidence_refs=[item.evidence_ref],
                        metric_values=metrics,
                    )
                )
                if len(observations) == 32:
                    return observations
    return observations


def _chart_candidates(
    track_id: str,
    purposes: Sequence[ReportChartPurpose],
    items: Sequence[ReportEvidenceItem],
    *,
    required: bool,
) -> tuple[list[ReportChartCandidate], list[str]]:
    tables = [
        item for item in items if item.kind is ReportEvidenceKind.TABLE
    ]
    candidates: list[ReportChartCandidate] = []
    gaps: list[str] = []
    for purpose in purposes[:REPORT_MAX_EXHIBITS]:
        built = None
        for item in tables:
            numeric_fields = [
                column
                for column in item.columns
                if any(
                    isinstance(row.get(column), Real)
                    and not isinstance(row.get(column), bool)
                    for row in item.rows
                )
            ]
            dimension_fields = [
                column
                for column in item.columns
                if column not in numeric_fields
            ]
            if not numeric_fields:
                continue
            preferred_x_tokens = (
                ("type", "entity", "segment", "category")
                if purpose is ReportChartPurpose.COMPOSITION
                else ("date", "period", "year", "month")
            )
            x_field = next(
                (
                    column
                    for column in dimension_fields
                    if any(
                        token in column.casefold()
                        for token in preferred_x_tokens
                    )
                ),
                dimension_fields[0] if dimension_fields else None,
            )
            series_fields = [
                column for column in numeric_fields if column != x_field
            ][:8]
            if purpose is ReportChartPurpose.COMPOSITION:
                series_fields = sorted(
                    series_fields,
                    key=lambda column: (
                        "share" not in column.casefold(),
                        column,
                    ),
                )[:1]
            if not series_fields:
                continue
            built = ReportChartCandidate(
                chart_id=f"{track_id}_{purpose.value}"[:64],
                purpose=purpose,
                title=(
                    f"{purpose.value.replace('_', ' ').title()}: "
                    f"{item.title}"
                )[:160],
                evidence_refs=[item.evidence_ref],
                x_field=x_field,
                series_fields=series_fields,
                required=required,
            )
            break
        if built is None:
            gaps.append(
                f"EXPECTED_EXHIBIT_{purpose.value.upper()}_UNAVAILABLE"
            )
        else:
            candidates.append(built)
    return candidates, gaps


def _packet_for_track(
    track,
    outputs: Mapping[ReportCollectorId, ReportCollectorOutput],
) -> ReportEvidencePacket:
    items: list[ReportEvidenceItem] = []
    gaps: list[str] = []
    failed = False
    for collector_id in track.collector_ids:
        output = outputs[collector_id]
        items.extend(output.items)
        gaps.extend(output.gaps)
        failed = failed or output.failed
    items = list(
        {
            item.evidence_ref: item
            for item in items
        }.values()
    )[:12]
    observations = _numeric_observations(
        items,
        track.requested_metrics,
    )
    candidates, exhibit_gaps = _chart_candidates(
        track.track_id,
        track.expected_exhibits,
        items,
        required=track.required,
    )
    gaps.extend(exhibit_gaps)
    gaps = list(dict.fromkeys(gaps))[:12]
    if items and observations:
        status = (
            ReportTrackStatus.PARTIAL
            if gaps
            else ReportTrackStatus.COMPLETE
        )
    else:
        status = (
            ReportTrackStatus.FAILED
            if failed
            else ReportTrackStatus.UNAVAILABLE
        )
        if not gaps:
            gaps = ["TRACK_EVIDENCE_UNAVAILABLE"]
    return ReportEvidencePacket(
        contract_version="report-evidence-packet-v1",
        track_id=track.track_id,
        status=status,
        items=items,
        observations=observations,
        gaps=gaps,
        chart_candidates=candidates,
    )


def _collector_query(
    user_query: str,
    track: ReportResearchTrack,
    collector_id: ReportCollectorId,
) -> str:
    if collector_id is not ReportCollectorId.VECTOR_KNOWLEDGE:
        return user_query
    return "\n".join(
        [
            f"Research track: {track.title}",
            *track.research_questions,
            f"Report context: {user_query}",
        ]
    )


def execute_report_research(
    query: str,
    plan: ReportResearchPlan,
    *,
    max_workers: int,
    collectors: Mapping[ReportCollectorId, Collector] | None = None,
) -> list[ReportEvidencePacket]:
    """Execute each unique deterministic collector once, bounded in parallel."""

    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between 1 and 8.")
    registry = collectors or DEFAULT_REPORT_COLLECTORS
    scope_key = plan.scope.model_dump_json()
    requests: dict[CollectorRequestKey, ReportCollectorRequest] = {}
    request_keys_by_track: dict[
        str,
        dict[ReportCollectorId, CollectorRequestKey],
    ] = {}
    for track in plan.tracks:
        track_keys: dict[ReportCollectorId, CollectorRequestKey] = {}
        for collector_id in track.collector_ids:
            collector_query = _collector_query(query, track, collector_id)
            request_key = (collector_id, collector_query, scope_key)
            requests.setdefault(
                request_key,
                ReportCollectorRequest(
                    collector_id=collector_id,
                    query=collector_query,
                    scope=plan.scope,
                ),
            )
            track_keys[collector_id] = request_key
        request_keys_by_track[track.track_id] = track_keys
    parent_scope = current_request_execution_scope()

    def run(request: ReportCollectorRequest) -> ReportCollectorOutput:
        collector = registry.get(request.collector_id)
        if collector is None:
            return ReportCollectorOutput(
                collector_id=request.collector_id,
                gaps=(
                    f"COLLECTOR_{request.collector_id.value.upper()}_UNAVAILABLE",
                ),
                failed=True,
            )
        with bind_request_execution_scope_snapshot(
            deadline=(
                parent_scope.deadline
                if parent_scope is not None
                else None
            ),
            scope=parent_scope,
        ):
            try:
                output = collector(request.query, request.scope)
                if output.collector_id is not request.collector_id:
                    raise ValueError("Collector output identity mismatch.")
                return output
            except Exception:
                return ReportCollectorOutput(
                    collector_id=request.collector_id,
                    gaps=(
                        "COLLECTOR_"
                        f"{request.collector_id.value.upper()}_FAILED",
                    ),
                    failed=True,
                )

    outputs: dict[CollectorRequestKey, ReportCollectorOutput] = {}
    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(requests)),
        thread_name_prefix="report-research",
    ) as pool:
        futures = {
            request_key: pool.submit(run, request)
            for request_key, request in requests.items()
        }
        for request_key, future in futures.items():
            outputs[request_key] = future.result()
    return [
        _packet_for_track(
            track,
            {
                collector_id: outputs[request_key]
                for collector_id, request_key in request_keys_by_track[
                    track.track_id
                ].items()
            },
        )
        for track in plan.tracks
    ]


def consolidate_report_evidence_packets(
    query: str,
    packets: Sequence[ReportEvidencePacket],
) -> ReportEvidenceManifest:
    """Freeze packet evidence and explicit gaps into one closed manifest."""

    items: list[ReportEvidenceItem] = []
    seen_refs: set[str] = set()
    for packet in packets:
        for item in packet.items:
            if item.evidence_ref in seen_refs:
                continue
            seen_refs.add(item.evidence_ref)
            items.append(item)
            if len(items) == 31:
                break
        if len(items) == 31:
            break
    gap_codes = list(
        dict.fromkeys(
            gap
            for packet in packets
            for gap in packet.gaps
        )
    )
    limitation_lines = [
        "The report may make claims only from the evidence items in this "
        "manifest; unavailable causes and missing periods remain limitations."
    ]
    limitation_lines.extend(
        f"Evidence gap: {code}." for code in gap_codes[:12]
    )
    items.append(
        make_report_narrative_evidence_item(
            kind=ReportEvidenceKind.LIMITATION,
            title="Research evidence boundary",
            source="system",
            content="\n".join(limitation_lines),
        )
    )
    return build_report_manifest_from_items(query, items)
