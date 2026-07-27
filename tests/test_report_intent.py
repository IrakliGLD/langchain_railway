"""Structured report-intent derivation tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.report_intent import build_report_planning_context
from contracts.question_analysis import (
    AnswerKind,
    PreferredPath,
    QueryType,
    VisualGoal,
)
from contracts.report import ReportIntent
from models import QueryContext


def _context(
    *,
    answer_kind: AnswerKind,
    query_type: QueryType,
    preferred_path: PreferredPath = PreferredPath.TOOL,
    visual_goal: VisualGoal | None = None,
    needs_driver_analysis: bool = False,
    needs_correlation_context: bool = False,
    language: str = "en",
    freeform_intent: str = "irrelevant free-form intent",
) -> QueryContext:
    ctx = QueryContext(query="მომიმზადე ანგარიში")
    ctx.lang_code = language
    ctx.question_analysis_source = "llm_active"
    ctx.question_analysis = SimpleNamespace(
        canonical_query_en="Prepare the requested analytical report.",
        answer_kind=answer_kind,
        classification=SimpleNamespace(
            query_type=query_type,
            intent=freeform_intent,
        ),
        routing=SimpleNamespace(
            preferred_path=preferred_path,
            needs_sql=preferred_path is PreferredPath.SQL,
            prefer_tool=preferred_path is PreferredPath.TOOL,
        ),
        language=SimpleNamespace(
            answer_language=SimpleNamespace(value=language),
        ),
        visualization=SimpleNamespace(visual_goal=visual_goal),
        analysis_requirements=SimpleNamespace(
            needs_driver_analysis=needs_driver_analysis,
            needs_correlation_context=needs_correlation_context,
            derived_metrics=[],
        ),
    )
    return ctx


@pytest.mark.parametrize(
    ("expected", "kwargs"),
    [
        (
            ReportIntent.SCENARIO,
            {
                "answer_kind": AnswerKind.SCENARIO,
                "query_type": QueryType.DATA_EXPLANATION,
            },
        ),
        (
            ReportIntent.FORECAST,
            {
                "answer_kind": AnswerKind.FORECAST,
                "query_type": QueryType.FORECAST,
            },
        ),
        (
            ReportIntent.COMPOSITION,
            {
                "answer_kind": AnswerKind.COMPARISON,
                "query_type": QueryType.COMPARISON,
                "visual_goal": VisualGoal.COMPOSITION,
            },
        ),
        (
            ReportIntent.DRIVER_ANALYSIS,
            {
                "answer_kind": AnswerKind.EXPLANATION,
                "query_type": QueryType.DATA_EXPLANATION,
                "needs_driver_analysis": True,
            },
        ),
        (
            ReportIntent.COMPARISON,
            {
                "answer_kind": AnswerKind.COMPARISON,
                "query_type": QueryType.COMPARISON,
            },
        ),
        (
            ReportIntent.TREND,
            {
                "answer_kind": AnswerKind.TIMESERIES,
                "query_type": QueryType.DATA_RETRIEVAL,
            },
        ),
        (
            ReportIntent.KNOWLEDGE,
            {
                "answer_kind": AnswerKind.KNOWLEDGE,
                "query_type": QueryType.REGULATORY_PROCEDURE,
                "preferred_path": PreferredPath.KNOWLEDGE,
            },
        ),
    ],
)
def test_report_intent_is_derived_from_structured_question_analysis(
    expected,
    kwargs,
):
    planning_context = build_report_planning_context(_context(**kwargs))

    assert planning_context.intent is expected
    assert planning_context.source == "question_analysis"
    assert planning_context.language_code == "en"


def test_freeform_intent_text_cannot_override_structured_report_intent():
    planning_context = build_report_planning_context(
        _context(
            answer_kind=AnswerKind.TIMESERIES,
            query_type=QueryType.DATA_RETRIEVAL,
            freeform_intent="compare market composition and forecast prices",
        )
    )

    assert planning_context.intent is ReportIntent.TREND


@pytest.mark.parametrize(
    "visual_goal",
    [VisualGoal.COMPARE, VisualGoal.RANKING],
)
def test_visual_comparison_goals_select_the_comparison_profile(visual_goal):
    planning_context = build_report_planning_context(
        _context(
            answer_kind=AnswerKind.EXPLANATION,
            query_type=QueryType.DATA_EXPLANATION,
            visual_goal=visual_goal,
        )
    )

    assert planning_context.intent is ReportIntent.COMPARISON


def test_table_requirement_uses_structured_routing_not_report_keywords():
    knowledge_comparison = build_report_planning_context(
        _context(
            answer_kind=AnswerKind.COMPARISON,
            query_type=QueryType.COMPARISON,
            preferred_path=PreferredPath.KNOWLEDGE,
        )
    )
    data_comparison = build_report_planning_context(
        _context(
            answer_kind=AnswerKind.COMPARISON,
            query_type=QueryType.COMPARISON,
            preferred_path=PreferredPath.TOOL,
        )
    )

    assert knowledge_comparison.intent is ReportIntent.COMPARISON
    assert knowledge_comparison.requires_table is False
    assert data_comparison.requires_table is True


def test_analyzer_unavailable_fallback_is_language_neutral_and_conservative():
    ctx = QueryContext(query="ელექტროენერგიის ბაზრის მოწყობა")
    ctx.lang_code = "ka"
    ctx.is_conceptual = True

    planning_context = build_report_planning_context(ctx)

    assert planning_context.intent is ReportIntent.KNOWLEDGE
    assert planning_context.language_code == "ka"
    assert planning_context.requires_table is False
    assert planning_context.source == "pipeline_fallback"


@pytest.mark.parametrize(
    ("language", "query"),
    [
        ("ru", "Устройство рынка электроэнергии"),
        ("zh", "电力市场结构"),
    ],
)
def test_fallback_semantics_do_not_depend_on_query_language(language, query):
    ctx = QueryContext(query=query)
    ctx.lang_code = language
    ctx.is_conceptual = True

    planning_context = build_report_planning_context(ctx)

    assert planning_context.intent is ReportIntent.KNOWLEDGE
    assert planning_context.language_code == language
    assert planning_context.requires_table is False
    assert planning_context.source == "pipeline_fallback"


def test_shadow_analysis_is_not_used_as_report_semantic_authority():
    ctx = _context(
        answer_kind=AnswerKind.FORECAST,
        query_type=QueryType.FORECAST,
    )
    ctx.question_analysis_source = "llm_shadow"
    ctx.lang_code = "ka"
    ctx.is_conceptual = True

    planning_context = build_report_planning_context(ctx)

    assert planning_context.intent is ReportIntent.KNOWLEDGE
    assert planning_context.language_code == "ka"
    assert planning_context.source == "pipeline_fallback"
