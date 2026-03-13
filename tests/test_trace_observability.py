"""Tests for structured trace observability around planning and summarization."""

import json
import logging
import os

import pandas as pd
import sqlalchemy

# Ensure config validation passes before importing modules that depend on config.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


class _DummyResult:
    def fetchall(self):
        return []

    def keys(self):
        return []


class _DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        return _DummyResult()


class _DummyEngine:
    def connect(self):
        return _DummyConnection()


sqlalchemy.create_engine = lambda *args, **kwargs: _DummyEngine()  # type: ignore[assignment]

from agent import analyzer, planner, summarizer  # noqa: E402
from contracts.question_analysis import QuestionAnalysis  # noqa: E402
from core.llm import SummaryEnvelope  # noqa: E402
from models import QueryContext  # noqa: E402


def _parse_trace_payloads(records):
    payloads = []
    for record in records:
        if not str(record.message).startswith("TRACE_DETAIL "):
            continue
        payloads.append(json.loads(str(record.message).split("TRACE_DETAIL ", 1)[1]))
    return payloads


def _conceptual_analysis() -> QuestionAnalysis:
    return QuestionAnalysis.model_validate(
        {
            "version": "question_analysis_v1",
            "raw_query": "wat is genx?",
            "canonical_query_en": "What is GENEX?",
            "language": {"input_language": "en", "answer_language": "en"},
            "classification": {
                "query_type": "conceptual_definition",
                "analysis_mode": "light",
                "intent": "market_participant_definition",
                "needs_clarification": False,
                "confidence": 0.98,
                "ambiguities": [],
            },
            "routing": {
                "preferred_path": "knowledge",
                "needs_sql": False,
                "needs_knowledge": True,
                "prefer_tool": False,
            },
            "knowledge": {"candidate_topics": [{"name": "market_structure", "score": 0.98}]},
            "tooling": {"candidate_tools": []},
            "sql_hints": {"metric": None, "entities": ["genex"], "aggregation": None, "dimensions": [], "period": None},
            "visualization": {
                "chart_requested_by_user": False,
                "chart_recommended": False,
                "chart_confidence": 0.95,
                "preferred_chart_family": None,
            },
        }
    )


def test_question_analyzer_emits_validated_trace(monkeypatch, caplog):
    monkeypatch.setattr(planner, "llm_analyze_question", lambda **_kwargs: _conceptual_analysis())
    ctx = QueryContext(query="wat is genx?", trace_id="trace-qa", session_id="session-qa")

    with caplog.at_level(logging.INFO, logger="Enai"):
        planner.analyze_question_active(ctx)

    payloads = _parse_trace_payloads(caplog.records)
    validated = next(p for p in payloads if p["event"] == "validated")
    assert validated["stage"] == "stage_0_2_question_analyzer"
    assert validated["extra"]["source"] == "llm_active"
    assert validated["extra"]["candidate_topics"] == ["market_structure"]
    assert validated["extra"]["canonical_query_en"] == "What is GENEX?"


def test_why_context_emits_signal_trace(monkeypatch, caplog):
    monkeypatch.setattr(analyzer, "stamp_provenance", lambda *_args, **_kwargs: None)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-10-01", "2021-11-01"]),
            "p_bal_gel": [50.0, 60.0],
            "xrate": [3.1, 3.2],
            "share_import": [0.10, 0.18],
            "share_regulated_hpp": [0.55, 0.47],
        }
    )
    ctx = QueryContext(
        query="why did balancing electricity price change in november 2021?",
        trace_id="trace-why",
        session_id="session-why",
        df=df,
        cols=list(df.columns),
        rows=[tuple(r) for r in df.itertuples(index=False, name=None)],
    )

    with caplog.at_level(logging.INFO, logger="Enai"):
        analyzer._build_why_context(ctx)

    payloads = _parse_trace_payloads(caplog.records)
    why_trace = next(p for p in payloads if p["event"] == "why_context")
    assert why_trace["stage"] == "stage_3_analyzer_enrich"
    assert why_trace["extra"]["why_override_generated"] is True
    assert why_trace["extra"]["why_claim_count"] >= 1
    assert "p_bal_gel" in why_trace["extra"]["signals"]


def test_summarizer_logs_pre_gate_and_provenance_failure(monkeypatch, caplog):
    monkeypatch.setattr(
        summarizer,
        "llm_summarize_structured",
        lambda *_args, **_kwargs: SummaryEnvelope(
            answer="Balancing price was 60.0 GEL/MWh and xrate was 12.0.",
            claims=["Balancing price was 60.0 GEL/MWh and xrate was 12.0."],
            citations=["data_preview", "statistics"],
            confidence=0.8,
        ),
    )

    ctx = QueryContext(
        query="why did balancing electricity price change in november 2021?",
        trace_id="trace-sum",
        session_id="session-sum",
        preview="date balancing_price_gel\n2021-11-01 60.0",
        stats_hint="Exchange rate was 12.0 in the comparison period.",
        provenance_cols=["date", "balancing_price_gel"],
        provenance_rows=[("2021-11-01", 60.0)],
    )

    with caplog.at_level(logging.INFO, logger="Enai"):
        summarizer.summarize_data(ctx)

    payloads = _parse_trace_payloads(caplog.records)
    pre_gate = next(p for p in payloads if p["event"] == "pre_gate")
    gate = next(p for p in payloads if p["event"] == "provenance_gate")

    assert pre_gate["extra"]["summary_source"] == "structured_summary"
    assert gate["extra"]["gate_passed"] is False
    assert "unmatched_tokens" in gate["extra"]
    assert ctx.summary_source == "citation_gate_fallback"
