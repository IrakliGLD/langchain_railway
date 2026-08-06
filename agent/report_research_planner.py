"""One-call research planning with deterministic semantic coverage checks."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from contracts.report import REPORT_MAX_EXHIBITS, ReportChartPurpose
from contracts.report_research import (
    ReportCollectorId,
    ReportPlanningConstraints,
    ReportRequiredExhibit,
    ReportResearchPlan,
    ReportResearchPlanAssessment,
    ReportResearchRequirement,
)
from knowledge import get_knowledge_for_topics, infer_topic_matches
from utils.language import resolve_answer_language

ResearchPlanInvoker = Callable[..., Any]

# Enough for a topic's drivers without crowding out the collector catalog or
# the request itself. The planner prompt is ~2k tokens today; this roughly
# doubles it once, and the provider's prefix cache absorbs the rest.
_PLANNING_KNOWLEDGE_BUDGET_CHARS = 8_000

_PRICE_SIGNALS = (
    "price",
    "prices",
    "cost",
    "prce",
    "ფას",
    "цен",
)
_TARIFF_SIGNALS = ("tariff", "ტარიფ", "тариф")
_SECURITY_SIGNALS = (
    "energy security",
    "security of supply",
    "enrgy secur",
    "ენერგეტიკული უსაფრთხო",
    "энергетическ",
)
_MARKET_KNOWLEDGE_SIGNALS = (
    "market model",
    "deregulat",
    "legislation",
    "regulation",
    "market rule",
    "legal stage",
    "target model",
    "ბაზრის მოდელ",
    "დერეგულ",
    "კანონ",
    "რეგულ",
    "модель рынка",
    "дерегул",
    "закон",
    "регулир",
)
_FORECAST_SIGNALS = (
    "forecast",
    "projection",
    "predict",
    "პროგნოზ",
    "прогноз",
)
_SCENARIO_SIGNALS = (
    "scenario",
    "what if",
    "hypothetical",
    "სცენარ",
    "сценар",
)
_REQUIRED_EXHIBIT_POLICIES = (
    (
        ReportResearchRequirement.PRICES,
        ReportCollectorId.PRICES,
        ReportChartPurpose.TREND,
    ),
    (
        ReportResearchRequirement.ENERGY_SECURITY,
        ReportCollectorId.GENERATION_MIX,
        ReportChartPurpose.COMPOSITION,
    ),
)


class ReportResearchPlanError(ValueError):
    """The one-call research plan failed schema or semantic validation."""

    def __init__(
        self,
        assessment: ReportResearchPlanAssessment,
        *,
        schema_error_codes: tuple[str, ...] = (),
    ) -> None:
        self.assessment = assessment
        self.schema_error_codes = schema_error_codes
        super().__init__(",".join(assessment.finding_codes))


def _schema_error_codes(error: Exception) -> tuple[str, ...]:
    if not isinstance(error, ValidationError):
        return (
            "PLAN_TYPE_INVALID"
            if isinstance(error, TypeError)
            else "PLAN_PARSE_INVALID",
        )

    codes: list[str] = []
    for finding in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[:16]:
        location = finding.get("loc") or ("root",)
        location_parts = [
            "ITEM"
            if isinstance(part, int)
            else re.sub(
                r"[^A-Z0-9]+",
                "_",
                str(part).upper(),
            ).strip("_")
            or "FIELD"
            for part in location
        ]
        error_type = re.sub(
            r"[^A-Z0-9]+",
            "_",
            str(finding.get("type") or "INVALID").upper(),
        ).strip("_")
        code = re.sub(
            r"_+",
            "_",
            "_".join(
                ["SCHEMA", *location_parts, error_type or "INVALID"]
            ),
        )[:64].rstrip("_")
        if code and code not in codes:
            codes.append(code)
    return tuple(codes) or ("PLAN_SCHEMA_INVALID",)


def _contains_any(query: str, signals: tuple[str, ...]) -> bool:
    lowered = query.casefold()
    return any(signal in lowered for signal in signals)


def _recognized_requirements(
    query: str,
) -> set[ReportResearchRequirement]:
    requirements: set[ReportResearchRequirement] = set()
    if _contains_any(query, _PRICE_SIGNALS):
        requirements.add(ReportResearchRequirement.PRICES)
    if _contains_any(query, _TARIFF_SIGNALS):
        requirements.add(ReportResearchRequirement.TARIFFS)
    if _contains_any(query, _SECURITY_SIGNALS):
        requirements.add(ReportResearchRequirement.ENERGY_SECURITY)
    if _contains_any(query, _MARKET_KNOWLEDGE_SIGNALS):
        requirements.add(ReportResearchRequirement.MARKET_KNOWLEDGE)
    if _contains_any(query, _FORECAST_SIGNALS):
        requirements.add(ReportResearchRequirement.FORECAST)
    if _contains_any(query, _SCENARIO_SIGNALS):
        requirements.add(ReportResearchRequirement.SCENARIO)
    if (
        "market" in query.casefold()
        and not requirements
    ):
        requirements.add(ReportResearchRequirement.MARKET_KNOWLEDGE)
    return requirements


def _assessment(
    query_digest: str,
    requirements: set[ReportResearchRequirement],
    findings: set[str],
) -> ReportResearchPlanAssessment:
    return ReportResearchPlanAssessment(
        contract_version="report-research-plan-assessment-v1",
        query_digest=query_digest,
        valid=not findings,
        recognized_requirements=sorted(
            requirements,
            key=lambda requirement: requirement.value,
        ),
        finding_codes=sorted(findings),
    )


def _planning_topic_knowledge(query: str) -> str:
    """Return the request's topic knowledge for the planner prompt.

    The planner used to choose its tracks before any knowledge was retrieved —
    per-track retrieval runs afterwards — so it decided what a report should
    cover without consulting what the system already knows about the subject.
    Handing it the topic's own Markdown keeps the coverage rubric general: a
    new topic needs a knowledge file, not a code change.

    Read-only over the files loaded at boot, and fail-open: this enriches a
    prompt, so losing it must never cost the job.
    """

    try:
        topics = infer_topic_matches(query)
        knowledge = get_knowledge_for_topics(
            sorted(topics),
            fallback_query=query,
        )
    except Exception:  # pragma: no cover - defensive
        return ""
    text = str(knowledge or "").strip()
    if len(text) <= _PLANNING_KNOWLEDGE_BUDGET_CHARS:
        return text
    # Cut on a line boundary so the tail is not a severed sentence the model
    # might read as a fact.
    clipped = text[:_PLANNING_KNOWLEDGE_BUDGET_CHARS]
    boundary = clipped.rfind("\n")
    return (clipped[:boundary] if boundary > 0 else clipped).rstrip()


def build_report_planning_constraints(
    query: str,
) -> ReportPlanningConstraints:
    """Publish deterministic exhibit obligations before model planning."""

    requirements = _recognized_requirements(query)
    return ReportPlanningConstraints(
        contract_version="report-planning-constraints-v1",
        maximum_total_exhibits=REPORT_MAX_EXHIBITS,
        required_exhibits=[
            ReportRequiredExhibit(
                requirement=requirement,
                collector_id=collector,
                purpose=purpose,
            )
            for requirement, collector, purpose in (
                _REQUIRED_EXHIBIT_POLICIES
            )
            if requirement in requirements
        ],
    )


def _enforce_exhibit_budget(
    plan: ReportResearchPlan,
    requirements: set[ReportResearchRequirement],
) -> ReportResearchPlan:
    exhibit_count = sum(
        len(track.expected_exhibits) for track in plan.tracks
    )
    if exhibit_count <= REPORT_MAX_EXHIBITS:
        return plan

    essential: set[tuple[int, int]] = set()
    for requirement, collector, purpose in _REQUIRED_EXHIBIT_POLICIES:
        if requirement not in requirements:
            continue
        for track_index, track in enumerate(plan.tracks):
            if (
                track.required
                and collector in track.collector_ids
                and purpose in track.expected_exhibits
            ):
                essential.add(
                    (
                        track_index,
                        track.expected_exhibits.index(purpose),
                    )
                )
                break

    candidates = [
        (track_index, exhibit_index)
        for track_index, track in enumerate(plan.tracks)
        for exhibit_index, _purpose in enumerate(track.expected_exhibits)
    ]
    candidates.sort(
        key=lambda candidate: (
            candidate in essential,
            any(
                collector is not ReportCollectorId.VECTOR_KNOWLEDGE
                for collector in plan.tracks[candidate[0]].collector_ids
            ),
            plan.tracks[candidate[0]].required,
            -candidate[0],
            -candidate[1],
        ),
        reverse=True,
    )
    retained = set(candidates[:REPORT_MAX_EXHIBITS])
    payload = plan.model_dump(mode="json")
    for track_index, track in enumerate(payload["tracks"]):
        track["expected_exhibits"] = [
            purpose
            for exhibit_index, purpose in enumerate(
                track["expected_exhibits"]
            )
            if (track_index, exhibit_index) in retained
        ]
    return ReportResearchPlan.model_validate(payload)


_UNREQUESTED_ENGINE_POLICIES: tuple[
    tuple[ReportResearchRequirement, ReportCollectorId], ...
] = (
    (ReportResearchRequirement.FORECAST, ReportCollectorId.FORECAST_ENGINE),
    (ReportResearchRequirement.SCENARIO, ReportCollectorId.SCENARIO_ENGINE),
)


def _prune_unrequested_engines(
    plan: ReportResearchPlan,
    requirements: set[ReportResearchRequirement],
) -> ReportResearchPlan:
    """Drop engine collectors the recognized requirements never asked for.

    The requirement side is a keyword list, so it reads "forecast" but not
    "future" or "outlook"; the model reads the query as a whole. That
    disagreement used to be fatal -- UNREQUESTED_FORECAST_COLLECTOR failed the
    plan, and because the prompt is identical on every retry, all three
    attempts failed the same way and the job died without producing anything.

    Scope control is the rule's real purpose, and dropping the collector
    achieves it. A track left with no collectors is dropped too, since it can
    no longer gather anything.
    """

    unrequested = {
        collector
        for requirement, collector in _UNREQUESTED_ENGINE_POLICIES
        if requirement not in requirements
    }
    if not unrequested:
        return plan
    if not any(
        collector in unrequested
        for track in plan.tracks
        for collector in track.collector_ids
    ):
        return plan

    payload = plan.model_dump(mode="json")
    dropped = {collector.value for collector in unrequested}
    retained_tracks = []
    for track in payload["tracks"]:
        track["collector_ids"] = [
            collector
            for collector in track["collector_ids"]
            if collector not in dropped
        ]
        if track["collector_ids"]:
            retained_tracks.append(track)
    if not retained_tracks:
        # Never prune the plan out of existence: an empty plan fails validation
        # anyway, and the original at least carries the model's intent.
        return plan
    payload["tracks"] = retained_tracks
    return ReportResearchPlan.model_validate(payload)


def validate_report_research_plan(
    query: str,
    plan: ReportResearchPlan,
    *,
    max_tracks: int,
) -> ReportResearchPlanAssessment:
    """Check query identity and high-value collector/exhibit coverage."""

    if not 1 <= max_tracks <= 8:
        raise ValueError("max_tracks must be between 1 and 8.")
    query_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    requirements = _recognized_requirements(query)
    findings: set[str] = set()
    if plan.query_digest != query_digest:
        findings.add("QUERY_DIGEST_MISMATCH")
    if plan.language_code != resolve_answer_language(query):
        findings.add("LANGUAGE_MISMATCH")
    if len(plan.tracks) > max_tracks:
        findings.add("TRACK_LIMIT_EXCEEDED")
    if (
        sum(len(track.expected_exhibits) for track in plan.tracks)
        > REPORT_MAX_EXHIBITS
    ):
        findings.add("EXHIBIT_LIMIT_EXCEEDED")

    required_tracks = [track for track in plan.tracks if track.required]
    if not required_tracks:
        findings.add("NO_REQUIRED_TRACK")
    collectors = {
        collector
        for track in required_tracks
        for collector in track.collector_ids
    }
    exhibits = {
        exhibit
        for track in required_tracks
        for exhibit in track.expected_exhibits
    }
    if (
        ReportResearchRequirement.PRICES in requirements
        and ReportCollectorId.PRICES not in collectors
    ):
        findings.add("PRICE_COLLECTOR_MISSING")
    if (
        ReportResearchRequirement.PRICES in requirements
        and ReportChartPurpose.TREND not in exhibits
    ):
        findings.add("PRICE_TREND_EXHIBIT_MISSING")
    if (
        ReportResearchRequirement.TARIFFS in requirements
        and ReportCollectorId.TARIFFS not in collectors
    ):
        findings.add("TARIFF_COLLECTOR_MISSING")
    if (
        ReportResearchRequirement.ENERGY_SECURITY in requirements
        and ReportCollectorId.GENERATION_MIX not in collectors
    ):
        findings.add("SECURITY_COLLECTOR_MISSING")
    if (
        ReportResearchRequirement.ENERGY_SECURITY in requirements
        and ReportChartPurpose.COMPOSITION not in exhibits
    ):
        findings.add("SECURITY_COMPOSITION_EXHIBIT_MISSING")
    if (
        ReportResearchRequirement.MARKET_KNOWLEDGE in requirements
        and ReportCollectorId.VECTOR_KNOWLEDGE not in collectors
    ):
        findings.add("KNOWLEDGE_COLLECTOR_MISSING")
    if (
        ReportResearchRequirement.FORECAST in requirements
        and ReportCollectorId.FORECAST_ENGINE not in collectors
    ):
        findings.add("FORECAST_COLLECTOR_MISSING")
    if (
        ReportResearchRequirement.FORECAST not in requirements
        and ReportCollectorId.FORECAST_ENGINE in collectors
    ):
        findings.add("UNREQUESTED_FORECAST_COLLECTOR")
    if (
        ReportResearchRequirement.SCENARIO in requirements
        and ReportCollectorId.SCENARIO_ENGINE not in collectors
    ):
        findings.add("SCENARIO_COLLECTOR_MISSING")
    if (
        ReportResearchRequirement.SCENARIO not in requirements
        and ReportCollectorId.SCENARIO_ENGINE in collectors
    ):
        findings.add("UNREQUESTED_SCENARIO_COLLECTOR")
    return _assessment(query_digest, requirements, findings)


def plan_report_research(
    query: str,
    *,
    max_tracks: int,
    invoke_model: ResearchPlanInvoker | None = None,
) -> ReportResearchPlan:
    """Create and validate one research plan without an LLM repair call."""

    if not 1 <= max_tracks <= 8:
        raise ValueError("max_tracks must be between 1 and 8.")
    query_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    language_code = resolve_answer_language(query)
    planning_constraints = build_report_planning_constraints(query)
    if invoke_model is None:
        from core.llm import llm_plan_report_research

        invoke_model = llm_plan_report_research

    try:
        raw_plan = invoke_model(
            query,
            language_code=language_code,
            max_tracks=max_tracks,
            planning_constraints=planning_constraints,
            topic_knowledge=_planning_topic_knowledge(query),
        )
        payload = (
            raw_plan.model_dump(mode="json")
            if isinstance(raw_plan, ReportResearchPlan)
            else dict(raw_plan)
            if isinstance(raw_plan, dict)
            else raw_plan
        )
        if isinstance(payload, dict):
            payload["contract_version"] = "report-research-plan-v1"
            payload["query_digest"] = query_digest
            payload["language_code"] = language_code
        plan = ReportResearchPlan.model_validate(payload)
    except (ValidationError, TypeError, ValueError) as exc:
        assessment = _assessment(
            query_digest,
            _recognized_requirements(query),
            {"PLAN_SCHEMA_INVALID"},
        )
        raise ReportResearchPlanError(
            assessment,
            schema_error_codes=_schema_error_codes(exc),
        ) from exc

    recognized = _recognized_requirements(query)
    plan = _prune_unrequested_engines(plan, recognized)
    plan = _enforce_exhibit_budget(plan, recognized)
    assessment = validate_report_research_plan(
        query,
        plan,
        max_tracks=max_tracks,
    )
    if not assessment.valid:
        raise ReportResearchPlanError(assessment)
    return plan
