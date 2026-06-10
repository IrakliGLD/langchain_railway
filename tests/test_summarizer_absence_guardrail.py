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
    ToolingInfo,
    ToolName,
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


def test_absence_guardrail_allows_data_shape_equivalence_mapping():
    """Fix F (2026-05-17) — Q2 production trace 5a00ee06.

    When the LLM follows the DATA-SHAPE RULE by explicitly mapping the
    user's categories to specific data columns and disclosing which
    categories lack a dedicated column (e.g. "no dedicated wind
    column — wind is grouped under price_renewable_ppa_*"), the
    absence-claim guardrail must NOT replace the substantive answer
    with a generic refusal.

    Detection signals:
      - phrases like "DATA-SHAPE RULE", "is grouped under",
        "does not contain a dedicated", "no dedicated"
      - column-name citations in backticks (``price_*``, ``share_*``,
        ``tariff_*``, etc.)
    """
    q2_text = (
        "Based on the provided 2024 data, a comparison can be made.\n\n"
        "**Data Mapping:**\n"
        "In accordance with the DATA-SHAPE RULE, the categories are mapped as:\n"
        "*   Small Hydro: `price_regulated_hpp_gel`\n"
        "*   Thermal: `price_regulated_old_tpp_gel` and `price_regulated_new_tpp_gel`\n"
        "*   Wind: no dedicated wind column. Wind is grouped under `price_renewable_ppa_gel`.\n\n"
        "Year-end values for some categories were not available."
    )
    assert summarizer._has_unsupported_absence_claims(q2_text) is False


def test_absence_guardrail_still_catches_bare_hallucination():
    """Negative control: a hallucination without any mapping context
    must still trip the guardrail."""
    hallucination = "Entity X had no values recorded during this period."
    assert summarizer._has_unsupported_absence_claims(hallucination) is True


def test_absence_guardrail_catches_blunt_not_available_without_mapping():
    """Negative control: a blunt 'not available' assertion without
    column-mapping context must still trip the guardrail."""
    blunt = "The price for entity Y is not available."
    assert summarizer._has_unsupported_absence_claims(blunt) is True


def test_absence_guardrail_allows_column_citation_only():
    """When the answer cites at least one data column by name in
    backticks, treat it as transparent mapping rather than absence
    claim. This is a weaker signal than the DATA-SHAPE RULE phrases
    but is sufficient on its own."""
    proxy_only = (
        "We used `price_regulated_hpp_gel` as the small-hydro proxy. "
        "Year-end values were not available for some periods."
    )
    assert summarizer._has_unsupported_absence_claims(proxy_only) is False
