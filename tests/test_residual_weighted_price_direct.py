"""Focused tests for deterministic residual weighted-price answers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import analyzer, summarizer
from contracts.question_analysis import AnswerKind, QuestionAnalysis, RenderStyle
from core.llm import SummaryEnvelope
from models import QueryContext


def _residual_qa(intent="residual_weighted_price_calculation"):
    return SimpleNamespace(
        answer_kind=AnswerKind.TIMESERIES,
        render_style=RenderStyle.DETERMINISTIC,
        classification=SimpleNamespace(intent=intent),
    )


def _explanation_qa() -> QuestionAnalysis:
    return QuestionAnalysis.model_validate(
        {
            "version": "question_analysis_v1",
            "raw_query": "Why did the balancing price change in November 2024?",
            "canonical_query_en": "Why did the balancing price change in November 2024?",
            "language": {"input_language": "en", "answer_language": "en"},
            "classification": {
                "query_type": "data_explanation",
                "analysis_mode": "analyst",
                "intent": "balancing_price_change_reason",
                "needs_clarification": False,
                "confidence": 0.95,
                "ambiguities": [],
            },
            "routing": {
                "preferred_path": "tool",
                "needs_sql": False,
                "needs_knowledge": False,
                "prefer_tool": True,
                "needs_multi_tool": True,
                "evidence_roles": ["primary_data", "composition_context"],
            },
            "knowledge": {"candidate_topics": []},
            "tooling": {"candidate_tools": []},
            "sql_hints": {
                "metric": "balancing",
                "entities": [],
                "aggregation": "monthly",
                "dimensions": [],
                "period": None,
            },
            "visualization": {
                "chart_requested_by_user": False,
                "chart_recommended": False,
                "chart_confidence": 0.0,
                "preferred_chart_family": "line",
            },
            "analysis_requirements": {
                "needs_driver_analysis": True,
                "needs_trend_context": False,
                "needs_correlation_context": False,
                "derived_metrics": [],
            },
            "answer_kind": "explanation",
            "render_style": "narrative",
        }
    )


def test_explicit_residual_component_query_filters_to_threshold_months():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01", "2021-07-01", "2021-08-01"]),
            "share_ppa_import_total": [0.999, 0.994, 0.880],
            "residual_contribution_ppa_import_gel": [72.7272, 66.6362, 44.0],
            "residual_contribution_ppa_import_usd": [25.225, 21.7686, 15.0],
        }
    )
    ctx = QueryContext(
        query=(
            "Calculate the weighted average balancing price for electricity from Renewable PPA, Import, "
            "Thermal Generation PPA, and CfD Scheme for June 2020, July 2021, and August 2021, only if "
            "these entities collectively contribute 99% or more to the total balancing composition in those months."
        ),
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        question_analysis=_residual_qa(),
        question_analysis_source="llm_active",
    )

    out = summarizer.summarize_data(ctx)

    assert out.summary_source == "deterministic_residual_weighted_price_direct"
    # "Thermal PPA" (thermal_ppa label), not "Thermal Generation PPA": the
    # latter was an over-scrub artifact of the removed bare "thermal" →
    # "Thermal Generation" VALUE_LABELS entry (2026-07-08). It now matches
    # the answer's own header, which always rendered "Thermal PPA".
    assert "Renewable PPA + Import + Thermal PPA + CfD Scheme" in out.summary
    assert "June 2020" in out.summary
    assert "July 2021" in out.summary
    assert "August 2021" not in out.summary
    assert "99.0%" in out.summary


def test_residual_direct_answer_does_not_hijack_explanation_queries(monkeypatch):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-10-01", "2024-11-01"]),
            "share_ppa_import_total": [0.61, 0.66],
            "residual_contribution_ppa_import_gel": [80.0, 95.0],
            "residual_contribution_ppa_import_usd": [28.0, 34.0],
        }
    )

    qa = _explanation_qa()

    monkeypatch.setattr(summarizer, "get_relevant_domain_knowledge", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(summarizer, "_is_summary_grounded", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        summarizer,
        "llm_summarize_structured",
        lambda *_args, **_kwargs: SummaryEnvelope(
            answer="Structured explanation path used.",
            claims=[],
            citations=["statistics"],
            confidence=0.9,
        ),
    )

    ctx = QueryContext(
        query="Why did the balancing price change in November 2024?",
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        preview="date,share_ppa_import_total\n2024-10-01,0.61\n2024-11-01,0.66",
        stats_hint="why-context available",
        question_analysis=qa,
        question_analysis_source="llm_active",
        response_mode="data_primary",
    )

    out = summarizer.summarize_data(ctx)

    assert out.summary_source == "structured_summary"
    assert out.summary == "Structured explanation path used."


def test_sub_one_percent_threshold_is_not_treated_as_fraction():
    query = "Find months where share of import is less than 0.2%."
    assert analyzer._extract_share_threshold(query) == ("lt", 0.002, "was below")
    assert summarizer._extract_residual_share_threshold(query) == ("lt", 0.002, "was below")


def test_implied_ppa_cfd_price_filters_import_and_excludes_it_from_denominator(monkeypatch):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "share_import": [0.001, 0.003],
            "share_renewable_ppa": [0.40, 0.40],
            "share_thermal_ppa": [0.10, 0.10],
            "share_cfd_scheme": [0.10, 0.10],
            "known_price_coverage_ok": [True, True],
            "share_ppa_import_total": [0.601, 0.603],
            "residual_contribution_ppa_import_gel": [60.0, 66.0],
            "residual_contribution_ppa_import_usd": [24.0, 26.4],
        }
    )
    ctx = QueryContext(
        query=(
            "Prices of regulated and deregulated plants are known. Find months where the share of import "
            "in the balancing basket is less than 0.2% and calculate the weighted average PPA/CfD price."
        ),
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        question_analysis=_residual_qa("implied_ppa_cfd_price_approximation"),
        question_analysis_source="llm_active",
    )

    monkeypatch.setattr(summarizer, "_try_generic_renderer", lambda _ctx: "wrong generic answer")

    out = summarizer.summarize_data(ctx)

    assert out.summary_source == "deterministic_residual_weighted_price_direct"
    assert "January 2024" in out.summary
    assert "February 2024" not in out.summary
    assert "100.0 GEL/MWh" in out.summary
    assert "99.8" not in out.summary


def test_implied_ppa_cfd_price_rejects_incomplete_known_price_coverage():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "share_import": [0.001],
            "share_renewable_ppa": [0.4],
            "share_thermal_ppa": [0.1],
            "share_cfd_scheme": [0.1],
            "known_price_coverage_ok": [False],
            "share_ppa_import_total": [0.601],
            "residual_contribution_ppa_import_gel": [60.0],
            "residual_contribution_ppa_import_usd": [24.0],
        }
    )
    ctx = QueryContext(
        query=(
            "Prices of regulated and deregulated plants are known. Find months where the share of import "
            "in the balancing basket is less than 0.2% and calculate the weighted average PPA/CfD price."
        ),
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
        question_analysis=_residual_qa("implied_ppa_cfd_price_approximation"),
        question_analysis_source="llm_active",
    )

    out = summarizer.summarize_data(ctx)

    assert out.summary_source == "deterministic_residual_weighted_price_direct"
    assert "did not contain complete prices" in out.summary
    assert "GEL/MWh" not in out.summary
