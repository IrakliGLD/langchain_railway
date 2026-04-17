"""Structured contract for the question-analyzer stage."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


ISODate = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]


# Core enums describe the analyzer's language, routing, period, and metric vocabulary.
class LanguageCode(str, Enum):
    EN = "en"
    KA = "ka"
    RU = "ru"
    ZH = "zh"
    OTHER = "other"


class QueryType(str, Enum):
    CONCEPTUAL_DEFINITION = "conceptual_definition"
    REGULATORY_PROCEDURE = "regulatory_procedure"
    FACTUAL_LOOKUP = "factual_lookup"
    DATA_RETRIEVAL = "data_retrieval"
    DATA_EXPLANATION = "data_explanation"
    COMPARISON = "comparison"
    FORECAST = "forecast"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


class AnalysisMode(str, Enum):
    LIGHT = "light"
    ANALYST = "analyst"


class PreferredPath(str, Enum):
    KNOWLEDGE = "knowledge"
    TOOL = "tool"
    SQL = "sql"
    CLARIFY = "clarify"
    REJECT = "reject"


class AnswerKind(str, Enum):
    """Shape of the final answer the user expects."""

    SCALAR = "scalar"
    LIST = "list"
    TIMESERIES = "timeseries"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    FORECAST = "forecast"
    SCENARIO = "scenario"
    KNOWLEDGE = "knowledge"
    CLARIFY = "clarify"


class RenderStyle(str, Enum):
    """Whether the answer should be formatted deterministically or via LLM narrative."""

    DETERMINISTIC = "deterministic"
    NARRATIVE = "narrative"


class Grouping(str, Enum):
    """How the answer rows should be grouped in the final output."""

    NONE = "none"
    BY_ENTITY = "by_entity"
    BY_PERIOD = "by_period"
    BY_METRIC = "by_metric"


class KnowledgeTopicName(str, Enum):
    GENERAL_DEFINITIONS = "general_definitions"
    BALANCING_PRICE = "balancing_price"
    MARKET_STRUCTURE = "market_structure"
    TARIFFS = "tariffs"
    CFD_PPA = "cfd_ppa"
    CURRENCY_INFLUENCE = "currency_influence"
    SEASONAL_PATTERNS = "seasonal_patterns"
    GENERATION_MIX = "generation_mix"
    PSO_TRADING = "pso_trading"
    SQL_EXAMPLES = "sql_examples"


class ToolName(str, Enum):
    GET_PRICES = "get_prices"
    GET_TARIFFS = "get_tariffs"
    GET_GENERATION_MIX = "get_generation_mix"
    GET_BALANCING_COMPOSITION = "get_balancing_composition"


class EvidenceRole(str, Enum):
    """Semantic role of a dataset in a multi-evidence execution plan."""

    PRIMARY_DATA = "primary_data"
    COMPOSITION_CONTEXT = "composition_context"
    TARIFF_CONTEXT = "tariff_context"
    CORRELATION_DRIVER = "correlation_driver"


class PeriodKind(str, Enum):
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    RANGE = "range"
    RELATIVE = "relative"


class PeriodGranularity(str, Enum):
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    RANGE = "range"
    RELATIVE = "relative"


class SqlAggregation(str, Enum):
    NONE = "none"
    DAILY = "daily"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class DimensionName(str, Enum):
    PRICE = "price"
    XRATE = "xrate"
    SHARE = "share"
    TARIFF = "tariff"
    ENERGY_QTY = "energy_qty"
    GENERATION = "generation"
    CPI = "cpi"
    SEASONALITY = "seasonality"
    MARKET_STRUCTURE = "market_structure"
    ENTITY = "entity"
    POWER_PLANT = "power_plant"
    REGULATION_STATUS = "regulation_status"
    PERIOD = "period"


class ChartFamily(str, Enum):
    LINE = "line"
    BAR = "bar"
    STACKED = "stacked"
    PIE = "pie"
    DUALAXIS = "dualaxis"


class PresentationMode(str, Enum):
    CHART = "chart"
    TABLE = "table"
    TEXT = "text"
    CHART_PLUS_TABLE = "chart_plus_table"


class VisualGoal(str, Enum):
    TREND = "trend"
    COMPARE = "compare"
    COMPOSITION = "composition"
    DECOMPOSITION = "decomposition"
    RANKING = "ranking"
    RELATIONSHIP = "relationship"
    THRESHOLD_SCAN = "threshold_scan"


class MeasureTransform(str, Enum):
    RAW = "raw"
    SUM = "sum"
    AVG = "avg"
    WEIGHTED_AVG = "weighted_avg"
    SHARE_OF_TOTAL = "share_of_total"
    MOM_DELTA = "mom_delta"
    MOM_PCT = "mom_pct"
    YOY_DELTA = "yoy_delta"
    YOY_PCT = "yoy_pct"
    CAGR = "cagr"
    INDEX_100 = "index_100"


class VisualizationTimeGrain(str, Enum):
    RAW = "raw"
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    SEASON = "season"
    YEAR = "year"


class SeriesSplitMode(str, Enum):
    SINGLE_CHART = "single_chart"
    MULTI_PANEL = "multi_panel"


class ChartIntent(str, Enum):
    TREND_COMPARE = "trend_compare"
    DECOMPOSITION = "decomposition"


class SemanticRole(str, Enum):
    OBSERVED = "observed"
    REFERENCE = "reference"
    DERIVED = "derived"
    COMPONENT_PRIMARY = "component_primary"
    COMPONENT_SECONDARY = "component_secondary"


class Season(str, Enum):
    SUMMER = "summer"
    WINTER = "winter"
    FULL = "full"


class ScenarioAggregation(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"


class DerivedMetricName(str, Enum):
    MOM_ABSOLUTE_CHANGE = "mom_absolute_change"
    MOM_PERCENT_CHANGE = "mom_percent_change"
    YOY_ABSOLUTE_CHANGE = "yoy_absolute_change"
    YOY_PERCENT_CHANGE = "yoy_percent_change"
    SHARE_DELTA_MOM = "share_delta_mom"
    CORRELATION_TO_TARGET = "correlation_to_target"
    TREND_SLOPE = "trend_slope"
    SCENARIO_SCALE = "scenario_scale"
    SCENARIO_OFFSET = "scenario_offset"
    SCENARIO_PAYOFF = "scenario_payoff"


# Nested models capture each structured sub-decision emitted by the analyzer.
class LanguageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_language: LanguageCode
    answer_language: LanguageCode


class ClassificationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_type: QueryType
    analysis_mode: AnalysisMode
    intent: str = Field(min_length=1, max_length=512)
    needs_clarification: bool
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguities: List[str] = Field(default_factory=list, max_length=25)


class RoutingInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_path: PreferredPath
    needs_sql: bool
    needs_knowledge: bool
    prefer_tool: bool
    needs_multi_tool: bool = False
    evidence_roles: List[EvidenceRole] = Field(default_factory=list, max_length=4)


class TopicCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: KnowledgeTopicName
    score: float = Field(ge=0.0, le=1.0)


class KnowledgeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_topics: List[TopicCandidate] = Field(default_factory=list, max_length=5)


class FilterOperator(str, Enum):
    """Comparison operators for value-based evidence filtering."""

    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"


class FilterCondition(BaseModel):
    """Value-based filter applied during evidence collection (after fetch, before framing)."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1, max_length=64)
    operator: FilterOperator
    value: float
    unit: Optional[str] = Field(default=None, max_length=16)


class ToolParamsHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Optional[str] = Field(default=None, max_length=64)
    currency: Optional[str] = Field(default=None, max_length=16)
    granularity: Optional[str] = Field(default=None, max_length=16)
    start_date: Optional[ISODate] = None
    end_date: Optional[ISODate] = None
    entities: List[str] = Field(default_factory=list, max_length=25)
    types: List[str] = Field(default_factory=list, max_length=25)
    mode: Optional[str] = Field(default=None, max_length=32)
    filter: Optional[FilterCondition] = None

    @field_validator("end_date")
    @classmethod
    def _validate_date_order(cls, end_date: Optional[str], info) -> Optional[str]:
        start_date = info.data.get("start_date")
        if start_date and end_date and date.fromisoformat(end_date) < date.fromisoformat(start_date):
            raise ValueError("end_date must be on or after start_date")
        return end_date


class ToolCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ToolName
    score: float = Field(ge=0.0, le=1.0)
    reason: Optional[str] = Field(default=None, max_length=240)
    params_hint: Optional[ToolParamsHint] = None


class ToolingInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_tools: List[ToolCandidate] = Field(default_factory=list, max_length=5)


class PeriodInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: PeriodKind
    start_date: ISODate
    end_date: ISODate
    granularity: PeriodGranularity
    raw_text: Optional[str] = Field(default=None, max_length=120)

    @field_validator("end_date")
    @classmethod
    def _validate_date_order(cls, end_date: str, info) -> str:
        start_date = info.data.get("start_date")
        if start_date and date.fromisoformat(end_date) < date.fromisoformat(start_date):
            raise ValueError("end_date must be on or after start_date")
        return end_date


class SqlHints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Optional[str] = Field(default=None, max_length=64)
    entities: List[str] = Field(default_factory=list, max_length=25)
    aggregation: Optional[SqlAggregation] = None
    dimensions: List[DimensionName] = Field(default_factory=list, max_length=8)
    period: Optional[PeriodInfo] = None
    filter: Optional[FilterCondition] = None


class VisualizationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chart_requested_by_user: bool
    chart_recommended: bool
    chart_confidence: float = Field(ge=0.0, le=1.0)
    preferred_chart_family: Optional[ChartFamily] = None
    primary_presentation: Optional[PresentationMode] = None
    visual_goal: Optional[VisualGoal] = None
    measure_transform: MeasureTransform = MeasureTransform.RAW
    time_grain: Optional[VisualizationTimeGrain] = None
    series_split_mode: SeriesSplitMode = SeriesSplitMode.SINGLE_CHART
    max_series: Optional[int] = Field(default=None, ge=1, le=8)
    include_reference_lines: bool = False
    chart_intent: Optional[ChartIntent] = None
    target_series: List[SemanticRole] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def _validate_semantic_chart_hints(self) -> "VisualizationInfo":
        if (
            self.primary_presentation in {PresentationMode.CHART, PresentationMode.CHART_PLUS_TABLE}
            and not self.chart_requested_by_user
            and not self.chart_recommended
        ):
            # A first-class chart presentation implies that visualization is
            # relevant even if the analyzer forgot to set chart_recommended.
            self.chart_recommended = True

        if not self.chart_requested_by_user and not self.chart_recommended:
            self.chart_intent = None
            self.target_series = []
            return self

        if self.chart_intent is None:
            self.target_series = []
            return self

        allowed_roles = _VALID_ROLES_BY_INTENT.get(self.chart_intent, frozenset())
        if not self.target_series or any(role not in allowed_roles for role in self.target_series):
            self.chart_intent = None
            self.target_series = []
        return self


_SCENARIO_METRIC_NAMES = frozenset({
    DerivedMetricName.SCENARIO_SCALE,
    DerivedMetricName.SCENARIO_OFFSET,
    DerivedMetricName.SCENARIO_PAYOFF,
})

_VALID_ROLES_BY_INTENT = {
    ChartIntent.TREND_COMPARE: frozenset({
        SemanticRole.OBSERVED,
        SemanticRole.REFERENCE,
        SemanticRole.DERIVED,
    }),
    ChartIntent.DECOMPOSITION: frozenset({
        SemanticRole.COMPONENT_PRIMARY,
        SemanticRole.COMPONENT_SECONDARY,
    }),
}


# Derived-metric requests carry the extra parameters needed for analytical calculations.
class DerivedMetricRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: DerivedMetricName
    metric: str = Field(min_length=1, max_length=64)
    target_metric: Optional[str] = Field(default=None, max_length=64)
    rank_limit: Optional[int] = Field(default=None, ge=1, le=10)
    scenario_factor: Optional[float] = Field(default=None, ge=-1e9, le=1e9)
    scenario_volume: Optional[float] = Field(default=None, gt=0, le=1e6)
    scenario_aggregation: Optional[ScenarioAggregation] = Field(default=None)
    season: Optional[Season] = Field(default=None)

    @field_validator("metric", "target_metric")
    @classmethod
    def _strip_metric_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("metric fields must not be empty or whitespace")
        return trimmed

    @model_validator(mode="after")
    def _validate_scenario_fields(self) -> "DerivedMetricRequest":
        is_scenario = self.metric_name in _SCENARIO_METRIC_NAMES
        if is_scenario:
            if self.scenario_factor is None:
                raise ValueError("scenario_factor is required for scenario metrics")
            if self.metric_name == DerivedMetricName.SCENARIO_PAYOFF and self.scenario_volume is None:
                self.scenario_volume = 1.0
        else:
            if any(v is not None for v in [self.scenario_factor, self.scenario_volume, self.scenario_aggregation]):
                raise ValueError("scenario fields must be None for non-scenario metrics")
        return self


class AnalysisRequirementsInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_driver_analysis: bool = False
    needs_trend_context: bool = False
    needs_correlation_context: bool = False
    derived_metrics: List[DerivedMetricRequest] = Field(default_factory=list, max_length=12)


# Top-level payload validated once, then reused by the rest of the pipeline.
class QuestionAnalysis(BaseModel):
    """Validated output for the question-analyzer LLM call."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["question_analysis_v1"]
    raw_query: str = Field(min_length=1, max_length=2000)
    canonical_query_en: str = Field(min_length=1, max_length=2000)
    language: LanguageInfo
    classification: ClassificationInfo
    routing: RoutingInfo
    knowledge: KnowledgeInfo
    tooling: ToolingInfo
    sql_hints: SqlHints
    visualization: VisualizationInfo
    analysis_requirements: AnalysisRequirementsInfo = Field(default_factory=AnalysisRequirementsInfo)

    # --- Phase 1 contract extensions (answer shape emitted by LLM) ---
    answer_kind: Optional[AnswerKind] = Field(
        default=None,
        description=(
            "Shape of the final answer. The LLM should emit this directly; "
            "downstream stages trust it without re-interpretation."
        ),
    )
    render_style: Optional[RenderStyle] = Field(
        default=None,
        description=(
            "Whether the answer should be rendered deterministically (data lookup) "
            "or via LLM narrative (explanation, causal reasoning)."
        ),
    )
    grouping: Optional[Grouping] = Field(
        default=None,
        description="How answer rows should be grouped: none, by_entity, by_period, by_metric.",
    )
    entity_scope: Optional[str] = Field(
        default=None,
        max_length=120,
        description=(
            "Scoping hint for entity selection, e.g. 'regulated_plants', 'all', "
            "or specific entity names. Tells downstream stages which entities to include."
        ),
    )

    @field_validator("raw_query", "canonical_query_en")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("value must not be empty or whitespace")
        return trimmed
