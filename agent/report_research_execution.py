"""Bounded parallel execution of deterministic report research collectors."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real
from typing import Any

from agent.report_chart_rules import evidence_dimension
from agent.report_charts import chart_column_roles
from agent.report_evidence import (
    build_report_evidence_manifest,
    build_report_manifest_from_items,
    build_report_narrative_items,
    declared_unit_spelling,
    make_report_narrative_evidence_item,
    make_report_table_evidence_item,
    report_pipeline_context_block_reason,
    report_pipeline_context_gaps,
)
from agent.report_track_specs import log_report_track_spec_disagreements
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
    REPORT_PACKET_MAX_OBSERVATIONS,
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

_LOGGER = logging.getLogger("Enai.ReportResearch")

# The prompt's evidence budget is divided evenly across manifest items, so each
# retrieved passage takes room away from the data tables. Ten passages left a
# price table roughly 2,600 characters of a 32,000-character budget and the
# writer produced market-design prose instead of price analysis.
_REPORT_VECTOR_PRIMARY_CHUNKS = 2
_REPORT_VECTOR_REFERENCE_CHUNKS = 1

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
        for chunk in bundle.chunks[:_REPORT_VECTOR_PRIMARY_CHUNKS]
    ]
    if get_reference_expansion_mode() == "on":
        role_chunks.extend(
            (
                ReportKnowledgeEvidenceRole.supporting_reference,
                chunk,
            )
            for chunk in bundle.reference_chunks[
                :_REPORT_VECTOR_REFERENCE_CHUNKS
            ]
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
        if len(observations) >= REPORT_PACKET_MAX_OBSERVATIONS:
            return observations
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
        undeclared_numeric_columns = [
            column
            for column in numeric_columns
            if not str(item.unit_by_column.get(column, "")).strip()
        ]
        if undeclared_numeric_columns:
            # Without this the drop is invisible, and a thin report looks like
            # a lazy writer rather than evidence the writer could not cite.
            _LOGGER.warning(
                "Report evidence numeric columns have no declared unit and "
                "cannot be claimed: evidence_ref=%s source=%s columns=%s",
                item.evidence_ref,
                item.source,
                ",".join(undeclared_numeric_columns),
            )
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
                # The grounding validator refuses every claim whose column has
                # no declared unit, so a metric here could only be advertised
                # under a fabricated unit and then fail verification. Leave the
                # column out rather than point the writer at an unusable number.
                unit = str(item.unit_by_column.get(column, "")).strip()
                if not unit:
                    continue
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
                if len(observations) >= REPORT_PACKET_MAX_OBSERVATIONS:
                    return observations
    return observations


# A line or bar exhibit has a left axis and at most a right one, so at most two
# (dimension, unit) groups can share it — the rule ``_axis_metadata`` enforces
# when it builds. The other purposes are unaffected: a table legitimately shows
# many units, a pie and a scatter have one axis apiece.
_DUAL_AXIS_CHART_PURPOSES = frozenset(
    {
        ReportChartPurpose.COMPARISON,
        ReportChartPurpose.FORECAST,
        ReportChartPurpose.TREND,
    }
)


def _plottable_series(
    ranked_columns: list[str],
    *,
    purpose: ReportChartPurpose,
    unit_by_column: Mapping[str, str],
) -> list[str]:
    """Drop series a two-axis chart cannot honestly carry.

    Taking the highest-ranked numeric columns of a twenty-eight-column enriched
    frame mixed GEL/MWh, USD/MWh and ratios on job 5cb4d210. The builder refused
    the whole exhibit as incompatible, so the reader got nothing rather than the
    prices the track was about. Selecting here keeps the ranking's own priority:
    whatever the most relevant columns measure in becomes the chart's axes.
    """

    if purpose not in _DUAL_AXIS_CHART_PURPOSES:
        return ranked_columns
    plottable: list[str] = []
    axes: list[tuple[str, str]] = []
    for column in ranked_columns:
        axis = (
            evidence_dimension(column),
            str(unit_by_column.get(column, "")).strip(),
        )
        if axis not in axes:
            if len(axes) >= 2:
                continue
            axes.append(axis)
        plottable.append(column)
    return plottable


def _chart_candidates(
    track_id: str,
    purposes: Sequence[ReportChartPurpose],
    items: Sequence[ReportEvidenceItem],
    *,
    required: bool,
    requested_metrics: Sequence[str] = (),
) -> tuple[list[ReportChartCandidate], list[str]]:
    tables = [
        item for item in items if item.kind is ReportEvidenceKind.TABLE
    ]
    # requested_metrics is free text the planner writes, not a closed catalog,
    # so this list cannot be derived from one. It exists only to drop words
    # that describe a comparison or an aggregation rather than a subject, and
    # would otherwise match any column spelling them. Words naming a
    # period-over-period transform are deliberately absent: they were matching
    # chart titles, which ``table_score`` no longer reads.
    requested_tokens = {
        token
        for metric in requested_metrics
        for token in re.findall(r"[a-z0-9]+", metric.casefold())
        if token not in {
            "average",
            "change",
            "maximum",
            "minimum",
            "percent",
            "ratio",
        }
    }

    def table_score(item: ReportEvidenceItem, purpose: ReportChartPurpose) -> int:
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
            column for column in item.columns if column not in numeric_fields
        ]
        # What the table holds, not the prose a chart builder labelled it
        # with. An observed panel and the month-on-month panel computed from
        # it carry the same source and the same columns, so on job 40e55527
        # the title was the only discriminator — and "MoM Change (%)" matched
        # the comparison words in a requested metric, scoring one row of
        # deltas above the levels it came from.
        material = " ".join([item.source, *item.columns]).casefold()
        score = sum(6 for token in requested_tokens if token in material)
        if purpose in {ReportChartPurpose.TREND, ReportChartPurpose.FORECAST}:
            score += 8 if any(
                any(token in column.casefold() for token in ("date", "period", "year", "month"))
                for column in dimension_fields
            ) else -20
        elif purpose is ReportChartPurpose.COMPOSITION:
            score += 10 if any("share" in column.casefold() for column in numeric_fields) else 0
            score += 5 if any(
                any(token in column.casefold() for token in ("type", "entity", "segment", "category"))
                for column in dimension_fields
            ) else 0
        return score + min(len(numeric_fields), 8)

    candidates: list[ReportChartCandidate] = []
    gaps: list[str] = []
    for purpose in purposes[:REPORT_MAX_EXHIBITS]:
        built = None
        ranked_tables = sorted(
            enumerate(tables),
            key=lambda pair: (-table_score(pair[1], purpose), pair[0]),
        )
        for _table_index, item in ranked_tables:
            # The builder's own axis typing, so the candidate cannot promise an
            # exhibit the builder will refuse. A composition chart slices one
            # whole by category, so a period column cannot be its axis; jobs
            # 5cb4d210 and 106b043c both lost their balancing composition
            # exhibit because the candidate offered the date it found first.
            roles = chart_column_roles(item)
            numeric_fields = roles["numeric"]
            dimension_fields = [*roles["temporal"], *roles["categorical"]]
            if not numeric_fields:
                continue
            if purpose is ReportChartPurpose.COMPOSITION:
                # The builder slices a composition by category when the table
                # has one, and otherwise pivots the latest period's numeric
                # columns into slices. Offer only a table one of those two
                # branches accepts, and an axis it will agree with: jobs
                # 5cb4d210 and 106b043c both lost their balancing composition
                # exhibit because the candidate offered the date column of a
                # table that also had a category column.
                if roles["categorical"]:
                    dimension_fields = roles["categorical"]
                elif not (roles["temporal"] and len(numeric_fields) >= 2):
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
            series_fields = _plottable_series(
                sorted(
                    [
                    column for column in numeric_fields if column != x_field
                    ],
                    key=lambda column: (
                        -sum(token in column.casefold() for token in requested_tokens),
                        "share" not in column.casefold()
                        if purpose is ReportChartPurpose.COMPOSITION
                        else False,
                        numeric_fields.index(column),
                    ),
                ),
                purpose=purpose,
                unit_by_column=item.unit_by_column,
            )[:8]
            if purpose is ReportChartPurpose.COMPOSITION:
                has_category_axis = x_field is not None and any(
                    token in x_field.casefold()
                    for token in ("type", "entity", "segment", "category")
                )
                if has_category_axis:
                    series_fields = series_fields[:1]
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


def _packet_from_items(
    track: ReportResearchTrack,
    items: Sequence[ReportEvidenceItem],
    *,
    gaps: Sequence[str] = (),
    failed: bool = False,
) -> ReportEvidencePacket:
    gaps = list(gaps)
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
        requested_metrics=track.requested_metrics,
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


def _packet_for_track(
    track: ReportResearchTrack,
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
    return _packet_from_items(
        track,
        items,
        gaps=gaps,
        failed=failed,
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


def build_report_track_analysis_query(
    report_query: str,
    track: ReportResearchTrack,
) -> str:
    """Build one bounded analytical query for a planned research track.

    The first planner question is authoritative. Remaining questions are
    coverage constraints, not independent pipeline runs; treating every list
    entry as a new request would multiply one track into as many as six model
    and database executions.
    """

    def bounded(value: str, limit: int) -> str:
        return str(value or "").strip()[:limit]

    primary, *coverage = track.research_questions
    parts = [
        bounded(primary, 600),
        f"Research track: {bounded(track.title, 160)}",
    ]
    if coverage:
        parts.extend(
            [
                "Required coverage:",
                *(f"- {bounded(question, 300)}" for question in coverage),
            ]
        )
    parts.append(f"Report context: {bounded(report_query, 1000)}")
    return "\n".join(parts)


def _derived_chart_evidence_items(context: Any) -> list[ReportEvidenceItem]:
    """Preserve analyzer-built chart frames as deterministic report tables."""

    items: list[ReportEvidenceItem] = []
    for index, spec in enumerate(
        list(getattr(context, "chart_override_specs", None) or [])
    ):
        if not isinstance(spec, dict):
            continue
        rows = spec.get("data")
        if not isinstance(rows, list) or not rows or not all(
            isinstance(row, dict) for row in rows
        ):
            continue
        columns = list(
            dict.fromkeys(
                column
                for row in rows
                for column in row
            )
        )
        metadata = spec.get("metadata") if isinstance(spec.get("metadata"), dict) else {}
        title = str(metadata.get("title") or "").strip()
        # A percent-change panel is built by renaming the levels it was computed
        # from, so its columns still read "Balancing electricity price
        # (GEL/MWh)" while holding month-on-month percentages. Only the builder
        # knows that; the column names never will.
        transform = str(metadata.get("measureTransform") or "").lower()
        numeric_columns = [
            column
            for column in columns
            if any(
                isinstance(row.get(column), Real)
                and not isinstance(row.get(column), bool)
                for row in rows
            )
        ]
        if any(token in transform for token in ("pct", "percent")):
            # A percent-change panel keeps the labels of the levels it was
            # computed from, so inference would declare those levels' units.
            declared_units = {column: "%" for column in numeric_columns}
        elif len(numeric_columns) == 1:
            # A melted frame names its measure column "value" and puts what it
            # measures on the axis instead — where analyzer.py already writes
            # the resolved unit for Standard's own charts.
            axis_unit = declared_unit_spelling(metadata.get("yAxisTitle", ""))
            declared_units = (
                {numeric_columns[0]: axis_unit} if axis_unit else {}
            )
        else:
            declared_units = {}
        item = make_report_table_evidence_item(
            query=str(getattr(context, "query", "") or ""),
            title=title or f"Derived chart evidence {index + 1}",
            source="derived_chart",
            columns=columns,
            rows=rows,
            provenance_refs=list(
                getattr(context, "provenance_refs", None) or []
            ),
            max_rows=200,
            unit_by_column=declared_units,
        )
        if item is not None:
            items.append(item)
    return items[:REPORT_MAX_EXHIBITS]


class ReportTrackAnalysisUnusable(ValueError):
    """One track's pipeline output cannot back report evidence.

    Carries the track and the stable block reason so the rollout telemetry can
    report which tracks fell back to deterministic evidence and why. The bare
    ValueError it replaces recorded only its own type name, which left a report
    degraded on two of four tracks with nothing to say which two (job cf47a2f6).
    """

    def __init__(self, track_id: str, reason: str) -> None:
        super().__init__(
            f"Track analysis pipeline context is not usable: {reason}."
        )
        self.track_id = track_id
        self.reason = reason


def execute_report_track_analysis(
    report_query: str,
    track: ReportResearchTrack,
    *,
    query_pipeline: Callable[..., Any],
    trace_id: str = "",
    actor_id: str = "",
    request_id: str = "",
    request_deadline: Any = None,
) -> ReportEvidencePacket:
    """Run one standard-quality, evidence-only pipeline for one track."""

    context = query_pipeline(
        build_report_track_analysis_query(report_query, track),
        trace_id=trace_id,
        actor_id=actor_id,
        request_id=request_id,
        request_deadline=request_deadline,
        answer_mode="report",
    )
    # Observe only: the analyzer's decision still stands. This records how far
    # the planner's would have been from it, so the nested call can be removed
    # on evidence rather than on hope.
    log_report_track_spec_disagreements(
        track,
        getattr(context, "question_analysis", None),
    )
    block_reason = report_pipeline_context_block_reason(context)
    if block_reason:
        raise ReportTrackAnalysisUnusable(track.track_id, block_reason)
    manifest = build_report_evidence_manifest(context)
    # Statistics and curated knowledge are the analyzed findings the report
    # writers need most. Put them ahead of supporting raw tables so the packet
    # cap cannot silently discard them on evidence-heavy tracks.
    items = build_report_narrative_items(context)
    items.extend(_derived_chart_evidence_items(context))
    items.extend(
        item
        for item in manifest.items
        if item.kind is not ReportEvidenceKind.LIMITATION
    )
    # A metric the pipeline could not derive is declared, not fatal. The packet
    # keeps every row the track did fetch and drops to PARTIAL, which the
    # evidence gate already treats as usable coverage.
    return _packet_from_items(
        track,
        items,
        gaps=report_pipeline_context_gaps(context),
    )


def merge_report_track_analysis_packet(
    track: ReportResearchTrack,
    baseline: ReportEvidencePacket,
    analysis: ReportEvidencePacket,
) -> ReportEvidencePacket:
    """Attach track findings while retaining deterministic evidence fallback."""

    if baseline.track_id != track.track_id or analysis.track_id != track.track_id:
        raise ValueError("Track packet identity mismatch.")
    gaps = [
        gap
        for packet in (analysis, baseline)
        for gap in packet.gaps
        if not gap.startswith("EXPECTED_EXHIBIT_")
    ]
    return _packet_from_items(
        track,
        [*analysis.items, *baseline.items],
        gaps=gaps,
        failed=(
            baseline.status is ReportTrackStatus.FAILED
            and analysis.status is ReportTrackStatus.FAILED
        ),
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


# ReportEvidenceManifest holds at most 32 items and consolidation always
# appends the limitation note, so packet evidence may fill only the rest.
_MAXIMUM_CONSOLIDATED_EVIDENCE_ITEMS = 31


def consolidate_report_evidence_packets(
    query: str,
    packets: Sequence[ReportEvidencePacket],
    *,
    extra_items: Sequence[ReportEvidenceItem] = (),
) -> ReportEvidenceManifest:
    """Freeze packet evidence and explicit gaps into one closed manifest.

    ``extra_items`` carries the standard pipeline's computed statistics and
    curated knowledge. They lead the manifest so the item cap can never drop
    them: they are the only evidence that states an analysis rather than a
    raw cell.
    """

    items: list[ReportEvidenceItem] = []
    seen_refs: set[str] = set()
    for item in extra_items:
        if item.evidence_ref in seen_refs:
            continue
        seen_refs.add(item.evidence_ref)
        items.append(item)
    # Take one round from every packet before anyone takes seconds. The cap is
    # smaller than what a full research plan carries -- a packet holds 12
    # items, four tracks are allowed, and only 31 slots exist -- so filling in
    # packet order spent them all on the earliest tracks and left a late one
    # with nothing to write from (job e3f43e84 discarded 17 items that way).
    # Each packet already orders its own evidence most-important-first, so a
    # round takes each track's next-best item and the shortfall is shared.
    dropped_by_track: dict[str, int] = {}
    depth = 0
    remaining = True
    while remaining:
        remaining = False
        for packet in packets:
            if depth >= len(packet.items):
                continue
            remaining = True
            item = packet.items[depth]
            if item.evidence_ref in seen_refs:
                continue
            if len(items) >= _MAXIMUM_CONSOLIDATED_EVIDENCE_ITEMS:
                # Keep counting rather than stop: which tracks lost how much is
                # the whole point of the line below, and the kept set is
                # already fixed.
                dropped_by_track[packet.track_id] = (
                    dropped_by_track.get(packet.track_id, 0) + 1
                )
                continue
            seen_refs.add(item.evidence_ref)
            items.append(item)
        depth += 1
    if dropped_by_track:
        # A track can reach the writer with no table at all this way, which
        # reads as a collector that returned nothing. Job e3f43e84 discarded
        # 17 of 48 items in silence and then died at the checkpoint because a
        # chart still pointed at one of them.
        _LOGGER.warning(
            "REPORT_MANIFEST_TRUNCATED %s",
            json.dumps(
                {
                    "dropped_by_track": dropped_by_track,
                    "dropped_item_count": sum(dropped_by_track.values()),
                    "kept_item_count": len(items) + 1,
                    "maximum_item_count": (
                        _MAXIMUM_CONSOLIDATED_EVIDENCE_ITEMS
                    ),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
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
