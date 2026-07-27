"""Derive report planning semantics from authoritative pipeline state."""

from __future__ import annotations

import re
from typing import Any

from contracts.question_analysis import (
    AnswerKind,
    PreferredPath,
    QueryType,
    VisualGoal,
)
from contracts.report import ReportIntent, ReportPlanningContext
from models import QueryContext

_LANGUAGE_CODE_PATTERN = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _report_intent_from_analysis(analysis: Any) -> ReportIntent:
    answer_kind = _enum_value(getattr(analysis, "answer_kind", None))
    query_type = _enum_value(analysis.classification.query_type)
    visualization = getattr(analysis, "visualization", None)
    visual_goal = _enum_value(
        getattr(visualization, "visual_goal", None)
    )
    requirements = getattr(analysis, "analysis_requirements", None)

    if answer_kind == AnswerKind.SCENARIO.value:
        return ReportIntent.SCENARIO
    if (
        answer_kind == AnswerKind.FORECAST.value
        or query_type == QueryType.FORECAST.value
    ):
        return ReportIntent.FORECAST
    if visual_goal in {
        VisualGoal.COMPOSITION.value,
        VisualGoal.DECOMPOSITION.value,
    }:
        return ReportIntent.COMPOSITION
    if (
        bool(getattr(requirements, "needs_driver_analysis", False))
        or bool(getattr(requirements, "needs_correlation_context", False))
        or visual_goal == VisualGoal.RELATIONSHIP.value
    ):
        return ReportIntent.DRIVER_ANALYSIS
    if (
        answer_kind == AnswerKind.COMPARISON.value
        or query_type == QueryType.COMPARISON.value
        or visual_goal
        in {
            VisualGoal.COMPARE.value,
            VisualGoal.RANKING.value,
        }
    ):
        return ReportIntent.COMPARISON
    if (
        answer_kind == AnswerKind.TIMESERIES.value
        or visual_goal == VisualGoal.TREND.value
    ):
        return ReportIntent.TREND
    if (
        answer_kind == AnswerKind.KNOWLEDGE.value
        or query_type
        in {
            QueryType.CONCEPTUAL_DEFINITION.value,
            QueryType.REGULATORY_PROCEDURE.value,
        }
    ):
        return ReportIntent.KNOWLEDGE
    return ReportIntent.GENERAL


def _analysis_requires_table(analysis: Any) -> bool:
    answer_kind = _enum_value(getattr(analysis, "answer_kind", None))
    query_type = _enum_value(analysis.classification.query_type)
    routing = analysis.routing
    preferred_path = _enum_value(routing.preferred_path)
    requirements = getattr(analysis, "analysis_requirements", None)

    if answer_kind in {
        AnswerKind.TIMESERIES.value,
        AnswerKind.FORECAST.value,
        AnswerKind.SCENARIO.value,
    }:
        return True
    if query_type in {
        QueryType.DATA_RETRIEVAL.value,
        QueryType.DATA_EXPLANATION.value,
        QueryType.FORECAST.value,
    }:
        return True
    if getattr(requirements, "derived_metrics", []):
        return True
    if preferred_path == PreferredPath.KNOWLEDGE.value:
        return False
    return bool(
        preferred_path
        in {PreferredPath.TOOL.value, PreferredPath.SQL.value}
        or getattr(routing, "needs_sql", False)
        or getattr(routing, "prefer_tool", False)
    )


def _language_code(ctx: QueryContext, analysis: Any | None) -> str:
    candidates = []
    if analysis is not None:
        language = getattr(analysis, "language", None)
        candidates.append(
            _enum_value(getattr(language, "answer_language", None))
        )
    candidates.extend([getattr(ctx, "lang_code", ""), "en"])
    for candidate in candidates:
        normalized = str(candidate or "").strip()
        if _LANGUAGE_CODE_PATTERN.fullmatch(normalized):
            return normalized
    return "en"


def build_report_planning_context(
    ctx: QueryContext,
) -> ReportPlanningContext:
    """Build the single semantic input consumed by report planning."""

    if ctx.has_authoritative_question_analysis:
        analysis = ctx.question_analysis
        return ReportPlanningContext(
            contract_version="report-planning-context-v1",
            intent=_report_intent_from_analysis(analysis),
            language_code=_language_code(ctx, analysis),
            request_objective=(
                str(getattr(analysis, "canonical_query_en", "")).strip()
                or str(getattr(ctx, "effective_query", ""))
                or str(ctx.query)
            ),
            requires_table=_analysis_requires_table(analysis),
            source="question_analysis",
        )

    conceptual = bool(ctx.is_conceptual)
    return ReportPlanningContext(
        contract_version="report-planning-context-v1",
        intent=(
            ReportIntent.KNOWLEDGE
            if conceptual
            else ReportIntent.GENERAL
        ),
        language_code=_language_code(ctx, None),
        request_objective=str(
            getattr(ctx, "effective_query", "") or ctx.query
        ).strip(),
        requires_table=not conceptual,
        source="pipeline_fallback",
    )


def report_context_requires_table(ctx: QueryContext) -> bool:
    """Return the language-neutral report table requirement."""

    return build_report_planning_context(ctx).requires_table
