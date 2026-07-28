"""One-call research planning with deterministic semantic coverage checks."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from contracts.report import ReportChartPurpose
from contracts.report_research import (
    ReportCollectorId,
    ReportResearchPlan,
    ReportResearchPlanAssessment,
    ReportResearchRequirement,
)
from utils.language import detect_language

ResearchPlanInvoker = Callable[..., Any]

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


class ReportResearchPlanError(ValueError):
    """The one-call research plan failed schema or semantic validation."""

    def __init__(self, assessment: ReportResearchPlanAssessment) -> None:
        self.assessment = assessment
        super().__init__(",".join(assessment.finding_codes))


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
    if plan.language_code != detect_language(query):
        findings.add("LANGUAGE_MISMATCH")
    if len(plan.tracks) > max_tracks:
        findings.add("TRACK_LIMIT_EXCEEDED")
    if sum(len(track.expected_exhibits) for track in plan.tracks) > 3:
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
    language_code = detect_language(query)
    if invoke_model is None:
        from core.llm import llm_plan_report_research

        invoke_model = llm_plan_report_research

    raw_plan = invoke_model(
        query,
        language_code=language_code,
        max_tracks=max_tracks,
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
    try:
        plan = ReportResearchPlan.model_validate(payload)
    except (ValidationError, TypeError, ValueError) as exc:
        assessment = _assessment(
            query_digest,
            _recognized_requirements(query),
            {"PLAN_SCHEMA_INVALID"},
        )
        raise ReportResearchPlanError(assessment) from exc

    assessment = validate_report_research_plan(
        query,
        plan,
        max_tracks=max_tracks,
    )
    if not assessment.valid:
        raise ReportResearchPlanError(assessment)
    return plan
