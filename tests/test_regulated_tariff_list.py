"""Focused tests for regulated tariff entity lookup and direct list answers."""

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
from agent.planner import resolve_tool_params  # noqa: E402
from contracts.question_analysis import (  # noqa: E402
    AnalysisMode,
    AnalysisRequirementsInfo,
    ClassificationInfo,
    KnowledgeInfo,
    LanguageCode,
    LanguageInfo,
    PreferredPath,
    QueryType,
    QuestionAnalysis,
    RoutingInfo,
    SqlHints,
    ToolCandidate,
    ToolName,
    ToolingInfo,
    VisualizationInfo,
)
from models import QueryContext  # noqa: E402


def _make_tariff_lookup_qa() -> QuestionAnalysis:
    return QuestionAnalysis(
        version="question_analysis_v1",
        raw_query="Which power plants are under price regulation?",
        canonical_query_en="List power plants that are under price regulation.",
        language=LanguageInfo(input_language=LanguageCode.EN, answer_language=LanguageCode.EN),
        classification=ClassificationInfo(
            query_type=QueryType.FACTUAL_LOOKUP,
            analysis_mode=AnalysisMode.LIGHT,
            intent="list regulated power plants",
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
            candidate_tools=[ToolCandidate(name=ToolName.GET_TARIFFS, score=0.98, reason="regulated tariffs")]
        ),
        sql_hints=SqlHints(),
        visualization=VisualizationInfo(
            chart_requested_by_user=False,
            chart_recommended=False,
            chart_confidence=0.0,
        ),
        analysis_requirements=AnalysisRequirementsInfo(),
    )


def test_resolve_tool_params_expands_generic_regulated_tariff_lookup():
    qa = _make_tariff_lookup_qa()

    params = resolve_tool_params(
        qa,
        ToolName.GET_TARIFFS.value,
        "List power plants that are under price regulation.",
    )

    assert params["entities"] == ["regulated_hpp", "regulated_new_tpp", "regulated_old_tpp"]
    assert params["currency"] == "gel"


def test_summarize_data_returns_direct_regulated_tariff_plant_list(monkeypatch):
    def _fail_if_llm_called(*_args, **_kwargs):
        raise AssertionError("LLM summarization should not run for direct regulated tariff lists")

    monkeypatch.setattr(summarizer, "llm_summarize_structured", _fail_if_llm_called)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "regulated_hpp_tariff_gel": [39.1, 39.4],
            "regulated_new_tpp_tariff_gel": [185.2, 186.0],
            "regulated_old_tpp_tariff_gel": [178.6, 179.1],
        }
    )
    ctx = QueryContext(
        query="Which power plants are under price regulation?",
        trace_id="regulated-tariff-list-1",
        session_id="regulated-tariff-list-1",
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        used_tool=True,
        tool_name="get_tariffs",
        tool_params={
            "entities": ["regulated_hpp", "regulated_new_tpp", "regulated_old_tpp"],
            "currency": "gel",
        },
    )

    out = summarizer.summarize_data(ctx)

    assert out.summary_source == "deterministic_regulated_tariff_list_direct"
    assert "Enguri Hydropower Plant" in out.summary
    assert "Gumati Hydropower Plant" in out.summary
    assert "Dzevruli Hydropower Plant" in out.summary
    assert "Lajanuri Hydropower Plant" in out.summary
    assert "Zhinvali Hydropower Plant" in out.summary
    assert "Vardnili Hydropower Plant" in out.summary
    assert "Vartsikhe Hydropower Plant" in out.summary
    assert "Khrami I Hydropower Plant" in out.summary
    assert "Khrami II Hydropower Plant" in out.summary
    assert "Shaori Hydropower Plant" not in out.summary
    assert "Rioni Hydropower Plant" not in out.summary
    assert "Gardabani Thermal Power Plant" in out.summary
    assert "Mtkvari Energy" in out.summary
    assert "Tbilisi Thermal Power Plant" in out.summary
    assert "G-POWER" in out.summary
    assert out.summary_citations == ["deterministic_regulated_tariff_list_direct"]
