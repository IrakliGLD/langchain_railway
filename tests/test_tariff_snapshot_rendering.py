"""Focused tests for single-period tariff snapshot rendering and answer-shape preservation."""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import pipeline, summarizer  # noqa: E402
from agent.evidence_validator import validate_evidence  # noqa: E402
from agent.frame_adapters import adapt_tool_result  # noqa: E402
from contracts.question_analysis import (  # noqa: E402
    AnalysisMode,
    AnalysisRequirementsInfo,
    AnswerKind,
    ClassificationInfo,
    Grouping,
    KnowledgeInfo,
    LanguageCode,
    LanguageInfo,
    PreferredPath,
    QueryType,
    QuestionAnalysis,
    RenderStyle,
    RoutingInfo,
    SqlHints,
    ToolCandidate,
    ToolName,
    ToolingInfo,
    VisualizationInfo,
)
from models import QueryContext  # noqa: E402


def _make_tariff_snapshot_qa() -> QuestionAnalysis:
    return QuestionAnalysis(
        version="question_analysis_v1",
        raw_query="for which entities tariff was set in july 2023 and what were those tariffs?",
        canonical_query_en="List the entities that had tariffs in July 2023 and report their tariff values.",
        language=LanguageInfo(input_language=LanguageCode.EN, answer_language=LanguageCode.EN),
        classification=ClassificationInfo(
            query_type=QueryType.DATA_RETRIEVAL,
            analysis_mode=AnalysisMode.LIGHT,
            intent="tariff entity snapshot",
            needs_clarification=False,
            confidence=0.95,
        ),
        routing=RoutingInfo(
            preferred_path=PreferredPath.TOOL,
            needs_sql=False,
            needs_knowledge=False,
            prefer_tool=True,
        ),
        knowledge=KnowledgeInfo(),
        tooling=ToolingInfo(
            candidate_tools=[ToolCandidate(name=ToolName.GET_TARIFFS, score=0.99, reason="tariff snapshot")]
        ),
        sql_hints=SqlHints(),
        visualization=VisualizationInfo(
            chart_requested_by_user=False,
            chart_recommended=False,
            chart_confidence=0.0,
        ),
        analysis_requirements=AnalysisRequirementsInfo(),
        answer_kind=AnswerKind.LIST,
        render_style=RenderStyle.DETERMINISTIC,
        grouping=Grouping.BY_ENTITY,
    )


def test_cross_check_keeps_authoritative_list_for_data_retrieval():
    qa = _make_tariff_snapshot_qa()
    ctx = QueryContext(
        query=qa.raw_query,
        question_analysis=qa,
        question_analysis_source="llm_active",
    )

    pipeline._cross_check_answer_kind(ctx)

    assert qa.answer_kind == AnswerKind.LIST


def test_tariff_snapshot_uses_generic_renderer_without_absence_claims(monkeypatch):
    def _unexpected_structured(*_args, **_kwargs):
        raise AssertionError("LLM summarization should not run for deterministic tariff snapshots")

    monkeypatch.setattr(summarizer, "llm_summarize_structured", _unexpected_structured)

    qa = _make_tariff_snapshot_qa()
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-07-01"]),
            "enguri_hpp_tariff_gel": [18.57],
            "khramhesi_i_tariff_gel": [27.31],
            "gardabani_tpp_tariff_gel": [134.65],
            "gpower_tpp_tariff_gel": [None],
        }
    )

    ctx = QueryContext(
        query=qa.raw_query,
        trace_id="tariff-snapshot-1",
        session_id="tariff-snapshot-1",
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        used_tool=True,
        tool_name="get_tariffs",
        tool_params={"currency": "gel"},
        question_analysis=qa,
        question_analysis_source="llm_active",
    )
    ctx.evidence_frame = adapt_tool_result("get_tariffs", df, answer_kind=AnswerKind.LIST)

    out = summarizer.summarize_data(ctx)

    assert out.summary_source == "generic_renderer"
    assert "Period:** July 1, 2023" in out.summary
    assert "Enguri" in out.summary and "18.57 GEL/MWh" in out.summary
    assert "Khrami I" in out.summary and "27.31 GEL/MWh" in out.summary
    assert "Gardabani" in out.summary and "134.65 GEL/MWh" in out.summary
    assert "G-POWER" not in out.summary
    assert "no active tariff" not in out.summary.lower()


def test_grouped_tariff_aliases_still_use_direct_regulated_list(monkeypatch):
    def _unexpected_structured(*_args, **_kwargs):
        raise AssertionError("LLM summarization should not run for direct regulated tariff lists")

    monkeypatch.setattr(summarizer, "llm_summarize_structured", _unexpected_structured)

    qa = _make_tariff_snapshot_qa()
    qa.raw_query = "Which power plants are under price regulation?"
    qa.canonical_query_en = "List power plants that are under price regulation."
    qa.classification.query_type = QueryType.FACTUAL_LOOKUP

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "regulated_hpp_tariff_gel": [39.1, 39.4],
            "regulated_new_tpp_tariff_gel": [185.2, 186.0],
            "regulated_old_tpp_tariff_gel": [178.6, 179.1],
        }
    )

    ctx = QueryContext(
        query=qa.raw_query,
        trace_id="regulated-tariff-authoritative-1",
        session_id="regulated-tariff-authoritative-1",
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        used_tool=True,
        tool_name="get_tariffs",
        tool_params={
            "entities": ["regulated_hpp", "regulated_new_tpp", "regulated_old_tpp"],
            "currency": "gel",
        },
        question_analysis=qa,
        question_analysis_source="llm_active",
    )
    ctx.evidence_frame = adapt_tool_result("get_tariffs", df, answer_kind=AnswerKind.LIST)

    out = summarizer.summarize_data(ctx)

    assert out.summary_source == "deterministic_regulated_tariff_list_direct"
    assert "Enguri Hydropower Plant" in out.summary
    assert "Gardabani Thermal Power Plant" in out.summary


def test_tariff_snapshot_same_month_multiple_dates_preserves_subperiod_values(monkeypatch):
    def _unexpected_structured(*_args, **_kwargs):
        raise AssertionError("LLM summarization should not run for deterministic tariff snapshots")

    monkeypatch.setattr(summarizer, "llm_summarize_structured", _unexpected_structured)

    qa = _make_tariff_snapshot_qa()
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-07-01", "2023-07-15"]),
            "enguri_hpp_tariff_gel": [18.57, 19.10],
        }
    )

    ctx = QueryContext(
        query=qa.raw_query,
        trace_id="tariff-snapshot-2",
        session_id="tariff-snapshot-2",
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        used_tool=True,
        tool_name="get_tariffs",
        tool_params={"currency": "gel"},
        question_analysis=qa,
        question_analysis_source="llm_active",
    )
    ctx.evidence_frame = adapt_tool_result("get_tariffs", df, answer_kind=AnswerKind.LIST)

    out = summarizer.summarize_data(ctx)

    assert out.summary_source == "generic_renderer"
    assert "Period:** July 2023" in out.summary
    assert "Enguri" in out.summary
    assert "July 1, 2023: 18.57 GEL/MWh" in out.summary
    assert "July 15, 2023: 19.10 GEL/MWh" in out.summary


def test_single_period_tariff_snapshot_timeseries_misclassification_still_renders_directly(monkeypatch):
    def _unexpected_structured(*_args, **_kwargs):
        raise AssertionError("LLM summarization should not run for single-period tariff snapshots")

    monkeypatch.setattr(summarizer, "llm_summarize_structured", _unexpected_structured)

    qa = _make_tariff_snapshot_qa()
    qa.answer_kind = AnswerKind.TIMESERIES

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-06-01"]),
            "enguri_hpp_tariff_gel": [18.57],
            "khramhesi_i_tariff_gel": [108.37],
            "khramhesi_ii_tariff_gel": [123.04],
            "gpower_tpp_tariff_gel": [141.06],
        }
    )

    ctx = QueryContext(
        query="for which entities tariff was set in june 2021 and what were those tariffs?",
        trace_id="tariff-snapshot-timeseries-1",
        session_id="tariff-snapshot-timeseries-1",
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        used_tool=True,
        tool_name="get_tariffs",
        tool_params={"currency": "gel"},
        question_analysis=qa,
        question_analysis_source="llm_active",
    )
    ctx.evidence_frame = adapt_tool_result("get_tariffs", df, answer_kind=AnswerKind.TIMESERIES)
    ctx.evidence_gap = validate_evidence(ctx.evidence_frame, qa.answer_kind)

    out = summarizer.summarize_data(ctx)

    assert out.summary_source == "generic_renderer"
    assert "Khrami I Hydropower Plant" in out.summary
    assert "Khrami II Hydropower Plant" in out.summary
    assert "108.37" in out.summary
    assert "123.04" in out.summary
    assert "does not establish" not in out.summary.lower()
