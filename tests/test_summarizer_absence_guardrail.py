"""Guardrail tests for unsupported absence claims in Stage 4 summaries."""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import summarizer  # noqa: E402
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
from core.llm import SummaryEnvelope  # noqa: E402
from models import QueryContext  # noqa: E402


def _make_narrative_tariff_qa() -> QuestionAnalysis:
    return QuestionAnalysis(
        version="question_analysis_v1",
        raw_query="Summarize tariff coverage in July 2023.",
        canonical_query_en="Summarize tariff coverage in July 2023.",
        language=LanguageInfo(input_language=LanguageCode.EN, answer_language=LanguageCode.EN),
        classification=ClassificationInfo(
            query_type=QueryType.DATA_RETRIEVAL,
            analysis_mode=AnalysisMode.LIGHT,
            intent="tariff summary",
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
            candidate_tools=[ToolCandidate(name=ToolName.GET_TARIFFS, score=0.99, reason="tariff summary")]
        ),
        sql_hints=SqlHints(),
        visualization=VisualizationInfo(
            chart_requested_by_user=False,
            chart_recommended=False,
            chart_confidence=0.0,
        ),
        analysis_requirements=AnalysisRequirementsInfo(),
        answer_kind=AnswerKind.EXPLANATION,
        render_style=RenderStyle.NARRATIVE,
        grouping=Grouping.NONE,
    )


def _make_tariff_ctx() -> QueryContext:
    qa = _make_narrative_tariff_qa()
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-07-01"]),
            "enguri_hpp_tariff_gel": [18.57],
            "khramhesi_i_tariff_gel": [27.31],
            "gpower_tpp_tariff_gel": [None],
        }
    )
    return QueryContext(
        query=qa.raw_query,
        trace_id="absence-guardrail",
        session_id="absence-guardrail",
        preview=df.to_string(index=False),
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        used_tool=True,
        tool_name="get_tariffs",
        tool_params={"currency": "gel"},
        question_analysis=qa,
        question_analysis_source="llm_active",
    )


def test_absence_claim_guardrail_replaces_unsupported_summary(monkeypatch):
    def _fake_structured(*_args, **_kwargs):
        return SummaryEnvelope(
            answer=(
                "Several entities, including Khramhesi I and G-POWER, did not have active tariff "
                "values recorded for this specific month."
            ),
            claims=[],
            citations=["data_preview"],
            confidence=0.95,
        )

    monkeypatch.setattr(summarizer, "llm_summarize_structured", _fake_structured)
    monkeypatch.setattr(summarizer, "get_relevant_domain_knowledge", lambda *_args, **_kwargs: "")

    out = summarizer.summarize_data(_make_tariff_ctx())

    assert out.summary_source == "absence_claim_guardrail"
    assert "omitted or blank values" in out.summary
    assert "did not have active tariff values recorded" not in out.summary
    assert out.summary_citations == ["absence_claim_guardrail"]


def test_absence_claim_guardrail_allows_conservative_limited_availability_language(monkeypatch):
    def _fake_structured(*_args, **_kwargs):
        return SummaryEnvelope(
            answer=(
                "The provided data does not establish a tariff value for G-POWER in the retrieved rows."
            ),
            claims=[],
            citations=["data_preview"],
            confidence=0.90,
        )

    monkeypatch.setattr(summarizer, "llm_summarize_structured", _fake_structured)
    monkeypatch.setattr(summarizer, "get_relevant_domain_knowledge", lambda *_args, **_kwargs: "")

    out = summarizer.summarize_data(_make_tariff_ctx())

    assert out.summary_source == "structured_summary"
    assert "does not establish a tariff value" in out.summary.lower()
    assert out.summary_citations == ["data_preview"]
