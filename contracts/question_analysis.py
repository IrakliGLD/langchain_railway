"""Structured contract for the question-analyzer stage."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


ISODate = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class LanguageCode(str, Enum):
    EN = "en"
    KA = "ka"
    RU = "ru"
    ZH = "zh"
    OTHER = "other"


class QueryType(str, Enum):
    CONCEPTUAL_DEFINITION = "conceptual_definition"
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


class KnowledgeTopicName(str, Enum):
    GENERAL_DEFINITIONS = "general_definitions"
    BALANCING_PRICE = "balancing_price"
    MARKET_STRUCTURE = "market_structure"
    TARIFFS = "tariffs"
    CFD_PPA = "cfd_ppa"
    CURRENCY_INFLUENCE = "currency_influence"
    SEASONAL_PATTERNS = "seasonal_patterns"
    GENERATION_MIX = "generation_mix"
    SQL_EXAMPLES = "sql_examples"


class ToolName(str, Enum):
    GET_PRICES = "get_prices"
    GET_TARIFFS = "get_tariffs"
    GET_GENERATION_MIX = "get_generation_mix"
    GET_BALANCING_COMPOSITION = "get_balancing_composition"


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


class ChartFamily(str, Enum):
    LINE = "line"
    BAR = "bar"
    STACKED = "stacked"
    PIE = "pie"
    DUALAXIS = "dualaxis"


class DerivedMetricName(str, Enum):
    MOM_ABSOLUTE_CHANGE = "mom_absolute_change"
    MOM_PERCENT_CHANGE = "mom_percent_change"
    YOY_ABSOLUTE_CHANGE = "yoy_absolute_change"
    YOY_PERCENT_CHANGE = "yoy_percent_change"
    SHARE_DELTA_MOM = "share_delta_mom"
    CORRELATION_TO_TARGET = "correlation_to_target"
    TREND_SLOPE = "trend_slope"


class LanguageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_language: LanguageCode
    answer_language: LanguageCode


class ClassificationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_type: QueryType
    analysis_mode: AnalysisMode
    intent: str = Field(min_length=1, max_length=128)
    needs_clarification: bool
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguities: List[str] = Field(default_factory=list, max_length=25)


class RoutingInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_path: PreferredPath
    needs_sql: bool
    needs_knowledge: bool
    prefer_tool: bool


class TopicCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: KnowledgeTopicName
    score: float = Field(ge=0.0, le=1.0)


class KnowledgeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_topics: List[TopicCandidate] = Field(default_factory=list, max_length=5)


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


class VisualizationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chart_requested_by_user: bool
    chart_recommended: bool
    chart_confidence: float = Field(ge=0.0, le=1.0)
    preferred_chart_family: Optional[ChartFamily] = None


class DerivedMetricRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: DerivedMetricName
    metric: str = Field(min_length=1, max_length=64)
    target_metric: Optional[str] = Field(default=None, max_length=64)
    rank_limit: Optional[int] = Field(default=None, ge=1, le=10)

    @field_validator("metric", "target_metric")
    @classmethod
    def _strip_metric_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("metric fields must not be empty or whitespace")
        return trimmed


class AnalysisRequirementsInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_driver_analysis: bool = False
    needs_trend_context: bool = False
    needs_correlation_context: bool = False
    derived_metrics: List[DerivedMetricRequest] = Field(default_factory=list, max_length=12)


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

    @field_validator("raw_query", "canonical_query_en")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("value must not be empty or whitespace")
        return trimmed
