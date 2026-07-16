"""Strict additive chat-gateway-v2 response models and projection.

The pipeline context is an internal, mutable object. This module is the only
place that turns it into the public v2 response. Every field is allow-listed
and typed so model telemetry, SQL, prompts, raw session identifiers, and other
internal state cannot leak through an open-ended metadata dictionary.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from agent.public_metadata import project_public_chart_metadata
from models import CHAT_GATEWAY_V2_CONTRACT_VERSION, TerminalOutcome

JsonScalar = Union[str, int, float, bool, None]
ChartRow = Dict[str, JsonScalar]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    DUAL_AXIS = "dualaxis"
    STACKED_BAR = "stackedbar"


class SeriesType(str, Enum):
    LINE = "line"
    BAR = "bar"


class AxisSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class ChartSeriesConfig(_StrictModel):
    type: SeriesType
    y_axis: AxisSide = Field(alias="yAxis")
    stack: Optional[str] = None
    dash_style: Optional[str] = Field(default=None, alias="dashStyle")


class TrendlineData(_StrictModel):
    dates: List[str]
    values: List[float]
    equation: str
    r_squared: float
    slope: float
    intercept: float


class Trendline(_StrictModel):
    column: str
    label: str
    data: TrendlineData
    original_label: str


class CompanionTable(_StrictModel):
    columns: List[str]
    rows: List[ChartRow]


class ChartMetadata(_StrictModel):
    aggregation: Optional[str] = None
    axis_mode: Optional[Literal["single", "dual"]] = Field(default=None, alias="axisMode")
    companion_table: Optional[CompanionTable] = Field(default=None, alias="companionTable")
    evidence_filter_applied: Optional[bool] = Field(default=None, alias="evidenceFilterApplied")
    evidence_source: Optional[str] = Field(default=None, alias="evidenceSource")
    evidence_unit: Optional[str] = Field(default=None, alias="evidenceUnit")
    group_index: Optional[int] = Field(default=None, alias="groupIndex", ge=0)
    group_source: Optional[str] = Field(default=None, alias="groupSource")
    has_projection: Optional[bool] = None
    labels: Optional[List[str]] = None
    long_frame: Optional[List[ChartRow]] = Field(default=None, alias="longFrame")
    long_frame_columns: Optional[List[str]] = Field(default=None, alias="longFrameColumns")
    measure_transform: Optional[str] = Field(default=None, alias="measureTransform")
    projection_to: Optional[str] = None
    provenance_refs: Optional[List[str]] = Field(default=None, alias="provenanceRefs")
    role: Optional[str] = None
    season_labels: Optional[Dict[str, str]] = Field(default=None, alias="seasonLabels")
    series_config: Optional[Dict[str, ChartSeriesConfig]] = Field(default=None, alias="seriesConfig")
    source_metrics: Optional[List[str]] = Field(default=None, alias="sourceMetrics")
    time_grain: Optional[str] = Field(default=None, alias="timeGrain")
    title: Optional[str] = None
    trendlines: Optional[List[Trendline]] = None
    visual_goal: Optional[str] = Field(default=None, alias="visualGoal")
    x_axis_title: Optional[str] = Field(default=None, alias="xAxisTitle")
    y_axis_left: Optional[str] = Field(default=None, alias="yAxisLeft")
    y_axis_right: Optional[str] = Field(default=None, alias="yAxisRight")
    y_axis_title: Optional[str] = Field(default=None, alias="yAxisTitle")


class PeriodIdentity(_StrictModel):
    grain: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None


class EvidenceIdentity(_StrictModel):
    source: Optional[str] = None
    refs: List[str] = Field(default_factory=list)


class ChartIdentity(_StrictModel):
    filter_applied: bool
    unit: Optional[str] = None
    period: PeriodIdentity
    evidence: EvidenceIdentity


class ChartSpec(_StrictModel):
    type: ChartType
    data: List[ChartRow]
    metadata: ChartMetadata
    identity: ChartIdentity


class AnalyzerProvenance(_StrictModel):
    authoritative: bool
    query_type: str
    answer_kind: str
    confidence: float = Field(ge=0.0, le=1.0)


class GroundingGate(_StrictModel):
    passed: bool
    reason: str
    coverage: float = Field(ge=0.0, le=1.0)


class AnswerProvenance(_StrictModel):
    answer_path: str
    summary_source: str
    data_source: str
    provenance_refs: List[str]
    tool_name: str
    used_sql: bool
    retrieval_tier: str
    analyzer: AnalyzerProvenance
    grounding_gate: GroundingGate


class TrustState(_StrictModel):
    confidence: float = Field(ge=0.0, le=1.0)
    provenance_coverage: float = Field(ge=0.0, le=1.0)
    grounding_passed: bool
    grounding_reason: str
    citations: List[str]


class PublicSessionState(_StrictModel):
    continuity_available: bool


class ChatGatewayV2Response(_StrictModel):
    contract_version: Literal["chat-gateway-v2"] = CHAT_GATEWAY_V2_CONTRACT_VERSION
    answer: str
    terminal_outcome: TerminalOutcome
    charts: List[ChartSpec]
    provenance: AnswerProvenance
    trust: TrustState
    request_id: str
    session: PublicSessionState
    execution_time: float = Field(ge=0.0)


def _bounded_float(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _json_scalar(value: Any) -> JsonScalar:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_scalar(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _chart_rows(value: Any) -> List[ChartRow]:
    if not isinstance(value, list):
        return []
    rows: List[ChartRow] = []
    for row in value:
        if isinstance(row, dict):
            rows.append({str(key): _json_scalar(item) for key, item in row.items()})
    return rows


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return _json_scalar(value)


def _period_identity(metadata: dict[str, Any], rows: List[ChartRow]) -> PeriodIdentity:
    period_values: List[str] = []
    for row in rows:
        period_key = next(
            (key for key in row if any(token in key.lower() for token in ("date", "year", "month", "time", "period"))),
            None,
        )
        if period_key is not None and row.get(period_key) is not None:
            period_values.append(str(row[period_key]))
    ordered = sorted(set(period_values))
    return PeriodIdentity(
        grain=str(metadata.get("timeGrain") or "") or None,
        start=ordered[0] if ordered else None,
        end=ordered[-1] if ordered else None,
    )


def _chart_spec(raw_chart: Any) -> Optional[ChartSpec]:
    if not isinstance(raw_chart, dict):
        return None
    rows = _chart_rows(raw_chart.get("data"))
    raw_metadata = project_public_chart_metadata(raw_chart.get("metadata"))
    normalized_metadata = _normalize_json(raw_metadata)
    metadata = ChartMetadata.model_validate(normalized_metadata)
    return ChartSpec(
        type=raw_chart.get("type"),
        data=rows,
        metadata=metadata,
        identity=ChartIdentity(
            filter_applied=bool(raw_metadata.get("evidenceFilterApplied", False)),
            unit=str(raw_metadata.get("evidenceUnit") or "") or None,
            period=_period_identity(raw_metadata, rows),
            evidence=EvidenceIdentity(
                source=str(raw_metadata.get("evidenceSource") or "") or None,
                refs=[str(ref) for ref in (raw_metadata.get("provenanceRefs") or []) if ref is not None],
            ),
        ),
    )


def build_chat_gateway_v2_response(
    ctx: Any,
    *,
    answer: str,
    request_id: str,
    execution_time: float,
    answer_provenance: dict[str, Any],
    session_continuity_available: bool,
    terminal_outcome: TerminalOutcome | str | None = None,
) -> ChatGatewayV2Response:
    """Project one internal request result into the strict public v2 DTO."""

    raw_charts = list(getattr(ctx, "charts", None) or [])
    if not raw_charts and getattr(ctx, "chart_data", None) is not None:
        raw_charts = [
            {
                "data": getattr(ctx, "chart_data", None),
                "type": getattr(ctx, "chart_type", None),
                "metadata": getattr(ctx, "chart_meta", None),
            }
        ]
    charts = [chart for chart in (_chart_spec(raw_chart) for raw_chart in raw_charts) if chart is not None]

    resolved_outcome = terminal_outcome or getattr(ctx, "terminal_outcome", "")
    if not resolved_outcome:
        resolved_outcome = (
            TerminalOutcome.CONCEPTUAL_ANSWER
            if bool(getattr(ctx, "is_conceptual", False))
            else TerminalOutcome.DATA_ANSWER
        )

    normalized_provenance = _normalize_json(answer_provenance)
    grounding = normalized_provenance.get("grounding_gate", {})
    return ChatGatewayV2Response(
        answer=answer,
        terminal_outcome=resolved_outcome,
        charts=charts,
        provenance=AnswerProvenance.model_validate(normalized_provenance),
        trust=TrustState(
            confidence=_bounded_float(getattr(ctx, "summary_confidence", 0.0)),
            provenance_coverage=_bounded_float(getattr(ctx, "summary_provenance_coverage", 0.0)),
            grounding_passed=bool(grounding.get("passed", False)),
            grounding_reason=str(grounding.get("reason", "") or ""),
            citations=[str(citation) for citation in (getattr(ctx, "summary_citations", []) or [])],
        ),
        request_id=request_id,
        session=PublicSessionState(continuity_available=bool(session_continuity_available)),
        execution_time=max(0.0, float(execution_time)),
    )
