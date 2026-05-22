import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import summarizer
from models import QueryContext, QuestionAnalysis


def _make_forecast_qa(query: str = "test query") -> QuestionAnalysis:
    """Build a minimal QuestionAnalysis with answer_kind=FORECAST for forecast tests."""
    return QuestionAnalysis.model_validate({
        "version": "question_analysis_v1",
        "raw_query": query,
        "canonical_query_en": query,
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "forecast",
            "analysis_mode": "analyst",
            "intent": "forecast_price",
            "needs_clarification": False,
            "confidence": 0.95,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "sql",
            "needs_sql": True,
            "needs_knowledge": False,
            "prefer_tool": False,
        },
        "knowledge": {"candidate_topics": []},
        "tooling": {"candidate_tools": []},
        "sql_hints": {
            "metric": "p_bal_gel",
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
            "needs_driver_analysis": False,
            "needs_trend_context": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
        "answer_kind": "forecast",
    })


# ---------------------------------------------------------------------------
# Phase C (2026-05-22): FORECAST answers route through the LLM, not the
# deterministic ``generic_renderer._render_forecast``.
#
# Pre-Phase-C tests in this file asserted ``summary_source == "generic_renderer"``
# and monkey-patched ``llm_summarize_structured`` to raise — locking in the
# LLM bypass that produced the "Winter (GEL): 12.49 GEL/MWh" nonsense in
# production trace c507e4d7. The deterministic renderer cannot apply R²
# caveats, structural-driver rules for long horizons, or the July 2027
# regime-break warning from ``skills/answer-composer/references/forecast-caveats.md``.
#
# Post-Phase-C, trendline values still get computed in Stage 3 enrichment
# and added to ``ctx.stats_hint`` (the ``--- TRENDLINE FORECASTS ---`` block);
# the LLM consumes that block and produces the narrative with the right
# caveats. The two tests below verify the new invariants:
#   1. The LLM IS called for FORECAST queries (not bypassed).
#   2. The trendline data reaches the LLM via ``stats_hint``.
#   3. ``summary_source`` is the LLM-path label, not ``generic_renderer``.
#   4. ``_try_generic_renderer`` returns None for FORECAST.
# ---------------------------------------------------------------------------


def _capture_llm_calls(monkeypatch, envelope_answer="LLM-rendered forecast narrative"):
    """Helper — replace llm_summarize_structured with a capture that records
    call args and returns a controlled SummaryEnvelope so summarize_data can
    proceed through the LLM path in tests."""
    from core.llm import SummaryEnvelope

    captured: dict = {"called": False, "args": None, "kwargs": None}

    def _capture(*args, **kwargs):
        captured["called"] = True
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SummaryEnvelope(
            answer=envelope_answer,
            claims=[],
            citations=["statistics", "data_preview"],
            confidence=0.85,
        )

    monkeypatch.setattr(summarizer, "llm_summarize_structured", _capture)
    # Bypass the grounding gate so the controlled envelope passes through
    # without re-routing into the conservative fallback.
    monkeypatch.setattr(summarizer, "_is_summary_grounded", lambda *_a, **_kw: True)
    return captured


def test_forecast_routes_through_llm_with_trendline_evidence_in_stats_hint(monkeypatch):
    """Phase C invariant — FORECAST queries must call llm_summarize_structured,
    and the stats_hint argument passed to the LLM must contain the
    pre-computed trendline values from Stage 3 enrichment so the LLM can
    cite them verbatim. Replaces the pre-Phase-C tests that locked in the
    deterministic-renderer bypass."""
    stats_hint = """Yearly CAGR=-0.61%. Forecast years: 2030.

--- TRENDLINE FORECASTS (Linear Regression) ---
Target date: 2030-12-01

p_bal_gel:
  - Forecast value: 145.25
  - Equation: y = -0.002x + 150
  - R² (goodness of fit): 0.412

p_bal_usd:
  - Forecast value: 52.40
  - Equation: y = -0.001x + 54
  - R² (goodness of fit): 0.398"""

    captured = _capture_llm_calls(monkeypatch)

    qa = _make_forecast_qa("forecast balancing electricity price for 2030")
    ctx = QueryContext(
        query="forecast balancing electricity price for 2030",
        preview="date,p_bal_gel,p_bal_usd\n2025-09-01,131.7,47.5",
        stats_hint=stats_hint,
        provenance_cols=["date", "p_bal_gel", "p_bal_usd"],
        provenance_rows=[("2025-09-01", 131.7, 47.5)],
        question_analysis=qa,
        question_analysis_source="llm_active",
    )

    summarizer.summarize_data(ctx)

    # 1. LLM was actually called.
    assert captured["called"], "Phase C: FORECAST must route through llm_summarize_structured"

    # 2. The stats_hint passed to the LLM (positional arg index 2) contains
    #    the pre-computed trendline values so the LLM can cite them.
    llm_stats_hint = captured["args"][2]
    assert "TRENDLINE FORECASTS" in llm_stats_hint
    assert "145.25" in llm_stats_hint
    assert "52.40" in llm_stats_hint
    assert "0.412" in llm_stats_hint  # R² value reaches the LLM

    # 3. summary_source is the LLM path, not generic_renderer.
    assert ctx.summary_source == "structured_summary"
    assert ctx.summary_source != "generic_renderer"


def test_forecast_does_not_invoke_generic_renderer(monkeypatch):
    """Phase C invariant — ``_try_generic_renderer`` must return None for
    FORECAST answer_kind, so the dispatch falls through to the LLM. Pin
    this so a future edit can't quietly re-route FORECAST back to
    ``generic_renderer._render_forecast``."""
    from contracts.question_analysis import AnswerKind

    qa = _make_forecast_qa("forecast prices to 2035")
    ctx = QueryContext(
        query="forecast prices to 2035",
        preview="date,p_bal_gel\n2025-01-01,100.0",
        stats_hint=(
            "--- TRENDLINE FORECASTS (Linear Regression) ---\n"
            "Target date: 2035-12-01\n"
            "p_bal_gel:\n  - Forecast value: 150.0"
        ),
        provenance_cols=["date", "p_bal_gel"],
        provenance_rows=[("2025-01-01", 100.0)],
        question_analysis=qa,
        question_analysis_source="llm_active",
    )
    # Sanity-check the precondition.
    assert ctx.question_analysis.answer_kind == AnswerKind.FORECAST

    # Direct test of the dispatch helper — must return None for FORECAST.
    result = summarizer._try_generic_renderer(ctx)
    assert result is None, (
        "Phase C: _try_generic_renderer must skip FORECAST so the LLM "
        "path runs (and forecast-caveats.md judgment gets applied)."
    )


def test_forecast_generic_renderer_ignores_scratch_and_reference_cols():
    """The CAGR-row extractor must pick the real metric column and ignore
    ``is_forecast``, ``xrate``, and any ``__forecast_*`` leftover."""
    import pandas as pd
    from agent.summarizer import _build_forecast_frame_from_cagr_rows

    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01", "2035-06-01"]),
        "p_bal_gel": [110.0, 190.0],
        "xrate": [2.7, 3.1],
        "is_forecast": [False, True],
        "__forecast_year": [2024, 2035],
    })

    qa = _make_forecast_qa("forecast balancing price until 2035")
    ctx = QueryContext(
        query="forecast balancing price until 2035",
        question_analysis=qa,
        question_analysis_source="llm_active",
    )
    ctx.df = df

    target_date, entries = _build_forecast_frame_from_cagr_rows(ctx)
    assert target_date is not None and "2035" in target_date
    assert len(entries) == 1
    assert entries[0]["metric"] == "p_bal_gel"
    assert entries[0]["forecast_value"] == 190.0
    assert entries[0]["r_squared"] is None
