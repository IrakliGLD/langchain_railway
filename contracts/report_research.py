"""Strict contracts for multi-track report research and evidence coverage."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated, List, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from contracts.question_analysis import (
    AnswerKind,
    DerivedMetricName,
    PreferredPath,
    QueryType,
)
from contracts.report import REPORT_MAX_EXHIBITS, ReportChartPurpose
from contracts.report_evidence import (
    ReportEvidenceItem,
    ReportEvidenceKind,
)

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
EvidenceRef = Annotated[
    str,
    Field(
        pattern=(
            r"^evidence:(?:table|statistics|knowledge|limitation):"
            r"[0-9a-f]{16}$"
        )
    ),
]
FindingCode = Annotated[
    str,
    Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$"),
]


class ReportEvidenceMode(str, Enum):
    ANY = "any"
    TABLE = "table"
    KNOWLEDGE = "knowledge"
    MIXED = "mixed"


class ReportCollectorId(str, Enum):
    PRICES = "prices"
    BALANCING_COMPOSITION = "balancing_composition"
    GENERATION_MIX = "generation_mix"
    TARIFFS = "tariffs"
    VECTOR_KNOWLEDGE = "vector_knowledge"
    FORECAST_ENGINE = "forecast_engine"
    SCENARIO_ENGINE = "scenario_engine"


_TABULAR_COLLECTORS = {
    ReportCollectorId.PRICES,
    ReportCollectorId.BALANCING_COMPOSITION,
    ReportCollectorId.GENERATION_MIX,
    ReportCollectorId.TARIFFS,
    ReportCollectorId.FORECAST_ENGINE,
    ReportCollectorId.SCENARIO_ENGINE,
}
_KNOWLEDGE_COLLECTORS = {ReportCollectorId.VECTOR_KNOWLEDGE}


class ReportTrackStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class ReportCoverageStatus(str, Enum):
    READY = "ready"
    READY_WITH_GAPS = "ready_with_gaps"
    FAILED = "failed"


class ReportResearchRequirement(str, Enum):
    PRICES = "prices"
    TARIFFS = "tariffs"
    ENERGY_SECURITY = "energy_security"
    MARKET_KNOWLEDGE = "market_knowledge"
    FORECAST = "forecast"
    SCENARIO = "scenario"


class ReportMetricOperation(str, Enum):
    OBSERVED = "observed"
    MEAN = "mean"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    DIFFERENCE = "difference"
    PERCENT_CHANGE = "percent_change"
    RATIO = "ratio"
    SHARE = "share"
    COUNT = "count"


class ReportTimeGrain(str, Enum):
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    NONE = "none"


class _StrictResearchModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class ReportRequiredExhibit(_StrictResearchModel):
    requirement: ReportResearchRequirement
    collector_id: ReportCollectorId
    purpose: ReportChartPurpose


class ReportPlanningConstraints(_StrictResearchModel):
    contract_version: Literal["report-planning-constraints-v1"]
    maximum_total_exhibits: int = Field(
        ge=1,
        le=REPORT_MAX_EXHIBITS,
    )
    required_exhibits: List[ReportRequiredExhibit] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )

    @model_validator(mode="after")
    def _require_unique_exhibits(self) -> "ReportPlanningConstraints":
        identities = [
            (
                exhibit.requirement,
                exhibit.collector_id,
                exhibit.purpose,
            )
            for exhibit in self.required_exhibits
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "Required planning exhibits must be unique."
            )
        return self


class ReportResearchScope(_StrictResearchModel):
    geography: str = Field(min_length=1, max_length=120)
    period_start: date | None = None
    period_end: date | None = None
    timezone: str = Field(
        default="Asia/Tbilisi",
        min_length=1,
        max_length=64,
        pattern=r"^(?:UTC|[A-Za-z_]+(?:/[A-Za-z0-9_+-]+)+)$",
    )
    grain: ReportTimeGrain = ReportTimeGrain.NONE

    @model_validator(mode="after")
    def _validate_period(self) -> "ReportResearchScope":
        if (self.period_start is None) != (self.period_end is None):
            raise ValueError(
                "Research period_start and period_end must both be present "
                "or both be absent."
            )
        if (
            self.period_start is not None
            and self.period_end is not None
            and self.period_start > self.period_end
        ):
            raise ValueError(
                "Research period_start cannot be later than period_end."
            )
        return self


class ReportResearchScopeDraft(ReportResearchScope):
    """Model-owned scope fields with explicit nullable values."""

    period_start: date | None
    period_end: date | None
    timezone: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^(?:UTC|[A-Za-z_]+(?:/[A-Za-z0-9_+-]+)+)$",
    )
    grain: ReportTimeGrain


class ReportRequestTopic(_StrictResearchModel):
    topic_id: Identifier
    label: str = Field(min_length=1, max_length=160)
    required: bool
    evidence_mode: ReportEvidenceMode


class ReportResearchTrack(_StrictResearchModel):
    track_id: Identifier
    title: str = Field(min_length=1, max_length=160)
    topic_ids: List[Identifier] = Field(min_length=1, max_length=8)
    required: bool
    evidence_mode: ReportEvidenceMode
    collector_ids: List[ReportCollectorId] = Field(
        min_length=1,
        max_length=6,
    )
    research_questions: List[str] = Field(min_length=1, max_length=6)
    requested_metrics: List[Identifier] = Field(
        default_factory=list,
        max_length=16,
    )
    expected_exhibits: List[ReportChartPurpose] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )
    # What the track's own analysis should conclude, decided here rather than
    # re-derived per track by a second model reading this track serialized back
    # into prose. Distinct from requested_metrics, which names what to measure:
    # these name how to analyse it. "percent_change" cannot say whether it
    # means month-on-month or year-on-year, and only whoever wrote the research
    # question knows -- which is the planner.
    analysis_query_type: QueryType = QueryType.DATA_RETRIEVAL
    analysis_preferred_path: PreferredPath = PreferredPath.TOOL
    analysis_answer_kind: AnswerKind | None = None
    analysis_derived_metrics: List[DerivedMetricName] = Field(
        default_factory=list,
        max_length=8,
    )

    @field_validator(
        "topic_ids",
        "collector_ids",
        "requested_metrics",
        "expected_exhibits",
        "analysis_derived_metrics",
    )
    @classmethod
    def _require_unique_values(cls, values: list) -> list:
        if len(values) != len(set(values)):
            raise ValueError("Report research track lists must be unique.")
        return values

    @field_validator("research_questions")
    @classmethod
    def _validate_questions(cls, values: List[str]) -> List[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 600 for value in normalized):
            raise ValueError(
                "Research questions must be bounded non-empty strings."
            )
        if len(normalized) != len(set(normalized)):
            raise ValueError("Research questions must be unique.")
        return normalized

    @model_validator(mode="after")
    def _validate_collector_compatibility(self) -> "ReportResearchTrack":
        collectors = set(self.collector_ids)
        has_table = bool(collectors & _TABULAR_COLLECTORS)
        has_knowledge = bool(collectors & _KNOWLEDGE_COLLECTORS)
        if (
            self.evidence_mode is ReportEvidenceMode.TABLE
            and not has_table
        ):
            raise ValueError(
                "A table research track requires a tabular collector."
            )
        if (
            self.evidence_mode is ReportEvidenceMode.KNOWLEDGE
            and not has_knowledge
        ):
            raise ValueError(
                "A knowledge research track requires a knowledge collector."
            )
        if self.evidence_mode is ReportEvidenceMode.MIXED and not (
            has_table and has_knowledge
        ):
            raise ValueError(
                "A mixed research track requires tabular and knowledge "
                "collectors."
            )
        if (
            self.evidence_mode is ReportEvidenceMode.KNOWLEDGE
            and self.requested_metrics
        ):
            raise ValueError(
                "A knowledge-only track cannot request numeric metrics."
            )
        return self


class ReportResearchTrackDraft(ReportResearchTrack):
    """Model-owned track fields without application defaults.

    Defaults are omitted so every field appears in the strict JSON schema's
    ``required`` list and the model must state each decision rather than
    inheriting one silently.
    """

    requested_metrics: List[Identifier] = Field(max_length=16)
    expected_exhibits: List[ReportChartPurpose] = Field(
        max_length=REPORT_MAX_EXHIBITS,
    )
    analysis_query_type: QueryType
    analysis_preferred_path: PreferredPath
    analysis_answer_kind: AnswerKind | None
    analysis_derived_metrics: List[DerivedMetricName] = Field(max_length=8)


class ReportResearchPlanDraft(_StrictResearchModel):
    """Provider response contract before application-owned identity binding."""

    objective: str = Field(min_length=1, max_length=2000)
    scope: ReportResearchScopeDraft
    request_topics: List[ReportRequestTopic] = Field(
        min_length=1,
        max_length=8,
    )
    tracks: List[ReportResearchTrackDraft] = Field(
        min_length=1,
        max_length=8,
    )


class ReportResearchPlan(_StrictResearchModel):
    contract_version: Literal["report-research-plan-v1"]
    query_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    language_code: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
    objective: str = Field(min_length=1, max_length=2000)
    scope: ReportResearchScope
    request_topics: List[ReportRequestTopic] = Field(
        min_length=1,
        max_length=8,
    )
    tracks: List[ReportResearchTrack] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _validate_topic_coverage(self) -> "ReportResearchPlan":
        topic_ids = [topic.topic_id for topic in self.request_topics]
        track_ids = [track.track_id for track in self.tracks]
        if len(topic_ids) != len(set(topic_ids)):
            raise ValueError("Report request topic IDs must be unique.")
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Report research track IDs must be unique.")

        known_topics = set(topic_ids)
        for track in self.tracks:
            unknown = sorted(set(track.topic_ids) - known_topics)
            if unknown:
                raise ValueError(
                    "A report research track references an unknown request "
                    f"topic: {', '.join(unknown)}."
                )

        for topic in self.request_topics:
            covering = [
                track
                for track in self.tracks
                if topic.topic_id in track.topic_ids
            ]
            if not covering:
                raise ValueError(
                    "Every request topic must be covered by a research track."
                )
            if topic.required and not any(
                track.required for track in covering
            ):
                raise ValueError(
                    "Every required request topic must be covered by at "
                    "least one required research track."
                )
        return self


class ReportResearchPlanAssessment(_StrictResearchModel):
    contract_version: Literal["report-research-plan-assessment-v1"]
    query_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    valid: bool
    recognized_requirements: List[ReportResearchRequirement] = Field(
        default_factory=list,
        max_length=8,
    )
    finding_codes: List[FindingCode] = Field(
        default_factory=list,
        max_length=16,
    )

    @field_validator("recognized_requirements", "finding_codes")
    @classmethod
    def _require_unique_values(cls, values: list) -> list:
        if len(values) != len(set(values)):
            raise ValueError(
                "Research-plan assessment lists must be unique."
            )
        return values

    @model_validator(mode="after")
    def _validate_result(self) -> "ReportResearchPlanAssessment":
        if self.valid == bool(self.finding_codes):
            raise ValueError(
                "Research-plan assessment valid must match finding_codes."
            )
        return self


class ReportMetricValue(_StrictResearchModel):
    metric_id: Identifier
    label: str = Field(min_length=1, max_length=160)
    value: float
    display_value: str = Field(
        min_length=1,
        max_length=32,
        pattern=(
            r"^[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
            r"(?:[eE][-+]?\d+)?%?$"
        ),
    )
    unit: str = Field(min_length=1, max_length=64)
    operation: ReportMetricOperation
    evidence_refs: List[EvidenceRef] = Field(min_length=1, max_length=16)
    period_start: date | None = None
    period_end: date | None = None

    @field_validator("evidence_refs")
    @classmethod
    def _require_unique_refs(cls, refs: List[str]) -> List[str]:
        if len(refs) != len(set(refs)):
            raise ValueError("Metric evidence_refs must be unique.")
        return refs

    @model_validator(mode="after")
    def _validate_metric_shape(self) -> "ReportMetricValue":
        if (self.period_start is None) != (self.period_end is None):
            raise ValueError(
                "Metric period_start and period_end must both be present "
                "or both be absent."
            )
        if (
            self.period_start is not None
            and self.period_end is not None
            and self.period_start > self.period_end
        ):
            raise ValueError(
                "Metric period_start cannot be later than period_end."
            )
        percent_operation = self.operation in {
            ReportMetricOperation.PERCENT_CHANGE,
            ReportMetricOperation.RATIO,
            ReportMetricOperation.SHARE,
        }
        if percent_operation and (
            self.unit != "%" or not self.display_value.endswith("%")
        ):
            raise ValueError(
                "Percent, ratio, and share metrics require percent display "
                "and unit."
            )
        return self


class ReportEvidenceObservation(_StrictResearchModel):
    observation_id: Identifier
    statement: str = Field(min_length=20, max_length=1200)
    evidence_refs: List[EvidenceRef] = Field(min_length=1, max_length=16)
    metric_values: List[ReportMetricValue] = Field(
        default_factory=list,
        max_length=8,
    )

    @field_validator("evidence_refs")
    @classmethod
    def _require_unique_refs(cls, refs: List[str]) -> List[str]:
        if len(refs) != len(set(refs)):
            raise ValueError("Observation evidence_refs must be unique.")
        return refs

    @model_validator(mode="after")
    def _validate_metric_grounding(self) -> "ReportEvidenceObservation":
        known_refs = set(self.evidence_refs)
        metric_ids = [metric.metric_id for metric in self.metric_values]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError(
                "Observation metric IDs must be unique."
            )
        for metric in self.metric_values:
            if not set(metric.evidence_refs).issubset(known_refs):
                raise ValueError(
                    "Observation metrics may reference only observation "
                    "evidence."
                )
        return self


class ReportChartCandidate(_StrictResearchModel):
    chart_id: Identifier
    purpose: ReportChartPurpose
    title: str = Field(min_length=1, max_length=160)
    evidence_refs: List[EvidenceRef] = Field(min_length=1, max_length=16)
    x_field: str | None = Field(default=None, min_length=1, max_length=128)
    series_fields: List[str] = Field(default_factory=list, max_length=8)
    required: bool

    @field_validator("evidence_refs", "series_fields")
    @classmethod
    def _require_unique_values(cls, values: list) -> list:
        if len(values) != len(set(values)):
            raise ValueError("Chart candidate lists must be unique.")
        return values


# Named so producers can bound themselves against the same number the packet
# enforces. A producer that counts to a literal of its own drifts, and the
# consequence is a ValidationError that discards a whole track's evidence
# (job 5e6b0cf3, supply_mix_and_flows).
REPORT_PACKET_MAX_OBSERVATIONS = 32


class ReportEvidencePacket(_StrictResearchModel):
    contract_version: Literal["report-evidence-packet-v1"]
    track_id: Identifier
    status: ReportTrackStatus
    available_period_start: date | None = None
    available_period_end: date | None = None
    items: List[ReportEvidenceItem] = Field(default_factory=list, max_length=12)
    observations: List[ReportEvidenceObservation] = Field(
        default_factory=list,
        max_length=REPORT_PACKET_MAX_OBSERVATIONS,
    )
    gaps: List[str] = Field(default_factory=list, max_length=12)
    chart_candidates: List[ReportChartCandidate] = Field(
        default_factory=list,
        max_length=REPORT_MAX_EXHIBITS,
    )

    @field_validator("gaps")
    @classmethod
    def _validate_gaps(cls, gaps: List[str]) -> List[str]:
        normalized = [gap.strip() for gap in gaps]
        if any(not gap or len(gap) > 600 for gap in normalized):
            raise ValueError("Evidence gaps must be bounded strings.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Evidence gaps must be unique.")
        return normalized

    @model_validator(mode="after")
    def _validate_packet(self) -> "ReportEvidencePacket":
        if (
            self.available_period_start is None
        ) != (self.available_period_end is None):
            raise ValueError(
                "Packet available periods must both be present or absent."
            )
        if (
            self.available_period_start is not None
            and self.available_period_end is not None
            and self.available_period_start > self.available_period_end
        ):
            raise ValueError(
                "Packet available_period_start cannot be later than "
                "available_period_end."
            )

        item_refs = [item.evidence_ref for item in self.items]
        if len(item_refs) != len(set(item_refs)):
            raise ValueError("Packet evidence references must be unique.")
        observation_ids = [
            observation.observation_id
            for observation in self.observations
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("Packet observation IDs must be unique.")
        known_refs = set(item_refs)
        for observation in self.observations:
            if not set(observation.evidence_refs).issubset(known_refs):
                raise ValueError(
                    "Packet observations reference unknown evidence."
                )

        item_by_ref = {item.evidence_ref: item for item in self.items}
        chart_ids = [
            candidate.chart_id for candidate in self.chart_candidates
        ]
        if len(chart_ids) != len(set(chart_ids)):
            raise ValueError("Packet chart candidate IDs must be unique.")
        for candidate in self.chart_candidates:
            unknown = set(candidate.evidence_refs) - known_refs
            if unknown:
                raise ValueError(
                    "Packet chart candidate references unknown evidence."
                )
            if any(
                item_by_ref[ref].kind is not ReportEvidenceKind.TABLE
                for ref in candidate.evidence_refs
            ):
                raise ValueError(
                    "Packet chart candidates require table evidence."
                )

        if self.status is ReportTrackStatus.COMPLETE:
            if not self.items or not self.observations:
                raise ValueError(
                    "A complete packet requires evidence and observations."
                )
            if self.gaps:
                raise ValueError("A complete packet cannot contain gaps.")
        elif self.status is ReportTrackStatus.PARTIAL:
            if not self.items or not self.observations or not self.gaps:
                raise ValueError(
                    "A partial packet requires evidence, observations, "
                    "and gaps."
                )
        elif self.items or self.observations or self.chart_candidates:
            raise ValueError(
                "Unavailable or failed packets cannot carry evidence."
            )
        elif not self.gaps:
            raise ValueError(
                "Unavailable or failed packets require an evidence gap."
            )
        return self

    @property
    def numeric_observation_count(self) -> int:
        return sum(
            len(observation.metric_values)
            for observation in self.observations
        )


class ReportTrackCoverage(_StrictResearchModel):
    track_id: Identifier
    required: bool
    status: ReportTrackStatus
    evidence_item_count: int = Field(ge=0, le=32)
    numeric_observation_count: int = Field(ge=0, le=256)
    chart_candidate_count: int = Field(ge=0, le=3)
    finding_codes: List[FindingCode] = Field(
        default_factory=list,
        max_length=16,
    )

    @field_validator("finding_codes")
    @classmethod
    def _require_unique_findings(cls, values: List[str]) -> List[str]:
        if len(values) != len(set(values)):
            raise ValueError("Track finding codes must be unique.")
        return values

    @model_validator(mode="after")
    def _validate_coverage_counts(self) -> "ReportTrackCoverage":
        if self.status is ReportTrackStatus.COMPLETE:
            if self.evidence_item_count < 1 or self.finding_codes:
                raise ValueError(
                    "Complete track coverage requires evidence and no "
                    "findings."
                )
        elif self.status is ReportTrackStatus.PARTIAL:
            if self.evidence_item_count < 1 or not self.finding_codes:
                raise ValueError(
                    "Partial track coverage requires evidence and findings."
                )
        elif (
            self.evidence_item_count
            or self.numeric_observation_count
            or self.chart_candidate_count
            or not self.finding_codes
        ):
            raise ValueError(
                "Unavailable or failed track coverage requires findings "
                "and zero evidence counts."
            )
        return self


class ReportEvidenceGate(_StrictResearchModel):
    contract_version: Literal["report-evidence-gate-v1"]
    query_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ReportCoverageStatus
    tracks: List[ReportTrackCoverage] = Field(min_length=1, max_length=8)
    finding_codes: List[FindingCode] = Field(
        default_factory=list,
        max_length=32,
    )

    @field_validator("finding_codes")
    @classmethod
    def _require_unique_findings(cls, values: List[str]) -> List[str]:
        if len(values) != len(set(values)):
            raise ValueError("Evidence-gate finding codes must be unique.")
        return values

    @model_validator(mode="after")
    def _validate_gate_status(self) -> "ReportEvidenceGate":
        track_ids = [track.track_id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Evidence-gate track IDs must be unique.")
        required = [track for track in self.tracks if track.required]
        successful = [
            track
            for track in self.tracks
            if track.status
            in {ReportTrackStatus.COMPLETE, ReportTrackStatus.PARTIAL}
        ]
        noncomplete = [
            track
            for track in self.tracks
            if track.status is not ReportTrackStatus.COMPLETE
        ]

        if self.status is ReportCoverageStatus.READY:
            if any(
                track.status is not ReportTrackStatus.COMPLETE
                for track in required
            ):
                raise ValueError(
                    "A ready evidence gate requires complete required tracks."
                )
            if noncomplete or self.finding_codes:
                raise ValueError(
                    "A ready evidence gate cannot contain gaps or findings."
                )
        elif self.status is ReportCoverageStatus.READY_WITH_GAPS:
            successful_required = [
                track
                for track in required
                if track.status
                in {
                    ReportTrackStatus.COMPLETE,
                    ReportTrackStatus.PARTIAL,
                }
            ]
            if (
                not successful
                or (required and not successful_required)
                or not noncomplete
                or not self.finding_codes
            ):
                raise ValueError(
                    "A ready_with_gaps evidence gate requires useful "
                    "required evidence, gaps, and findings."
                )
        else:
            successful_required = [
                track
                for track in required
                if track.status
                in {
                    ReportTrackStatus.COMPLETE,
                    ReportTrackStatus.PARTIAL,
                }
            ]
            if (
                successful_required
                or (not required and successful)
                or not self.finding_codes
            ):
                raise ValueError(
                    "A failed evidence gate requires no successful required "
                    "tracks and at least one finding."
                )
        return self

    @property
    def ready_for_writing(self) -> bool:
        return self.status in {
            ReportCoverageStatus.READY,
            ReportCoverageStatus.READY_WITH_GAPS,
        }
