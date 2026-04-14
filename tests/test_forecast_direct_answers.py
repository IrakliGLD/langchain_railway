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


def test_summarize_data_uses_generic_renderer_for_forecast(monkeypatch):
    def _unexpected_structured(*_args, **_kwargs):
        raise AssertionError("llm_summarize_structured should not be called for deterministic forecast answers")

    monkeypatch.setattr(summarizer, "llm_summarize_structured", _unexpected_structured)

    stats_hint = """
Yearly CAGR=-0.61%. Forecast years: 2030.

--- TRENDLINE FORECASTS (Linear Regression) ---
Target date: 2030-12-01

p_bal_gel:
  - Forecast value: 145.25
  - Equation: y = -0.002x + 150
  - R² (goodness of fit): 0.412

p_bal_usd:
  - Forecast value: 52.40
  - Equation: y = -0.001x + 54
  - R² (goodness of fit): 0.398

xrate:
  - Forecast value: 2.77
  - Equation: y = 0.0001x + 2.6
  - R² (goodness of fit): 0.221
""".strip()

    qa = _make_forecast_qa("forecast balancing electricity price for 2030")
    ctx = QueryContext(
        query="forecast balancing electricity price for 2030",
        preview="date,p_bal_gel,p_bal_usd,xrate\n2025-09-01,131.7,47.5,2.77",
        stats_hint=stats_hint,
        provenance_cols=["date", "p_bal_gel", "p_bal_usd", "xrate"],
        provenance_rows=[("2025-09-01", 131.7, 47.5, 2.77)],
        question_analysis=qa,
        question_analysis_source="llm_active",
    )

    summarizer.summarize_data(ctx)

    assert ctx.summary_source == "generic_renderer"
    assert "145.25 GEL/MWh" in ctx.summary
    assert "52.40 USD/MWh" in ctx.summary
    assert "December 2030" in ctx.summary
    assert "R\u00b2=0.412" in ctx.summary
    assert "R\u00b2=0.398" in ctx.summary
    assert "2.77" not in ctx.summary


def test_summarize_data_renders_overall_and_seasonal_forecast_bundle(monkeypatch):
    def _unexpected_structured(*_args, **_kwargs):
        raise AssertionError("llm_summarize_structured should not be called for deterministic forecast answers")

    monkeypatch.setattr(summarizer, "llm_summarize_structured", _unexpected_structured)

    stats_hint = """
--- TRENDLINE FORECASTS (Linear Regression) ---
Target date: 2035-12-01

p_bal_gel:
  - Forecast value: 182.10
  - Equation: y = 0.01x + 120
  - RÂ² (goodness of fit): 0.612

p_bal_usd:
  - Forecast value: 63.40
  - Equation: y = 0.004x + 45
  - RÂ² (goodness of fit): 0.422

p_bal_gel (summer):
  - Forecast value: 160.55
  - Equation: y = 0.008x + 110
  - RÂ² (goodness of fit): 0.588

p_bal_usd (summer):
  - Forecast value: 55.80
  - Equation: y = 0.003x + 40
  - RÂ² (goodness of fit): 0.401

p_bal_gel (winter):
  - Forecast value: 208.46
  - Equation: y = 0.012x + 135
  - RÂ² (goodness of fit): 0.662

p_bal_usd (winter):
  - Forecast value: 62.59
  - Equation: y = 0.002x + 47
  - RÂ² (goodness of fit): 0.365
""".strip()

    qa = _make_forecast_qa("forecast balancing electricity price for 2035")
    ctx = QueryContext(
        query="forecast balancing electricity price for 2035",
        preview="date,p_bal_gel,p_bal_usd\n2025-09-01,159.81,58.99",
        stats_hint=stats_hint,
        provenance_cols=["date", "p_bal_gel", "p_bal_usd"],
        provenance_rows=[("2025-09-01", 159.81, 58.99)],
        question_analysis=qa,
        question_analysis_source="llm_active",
    )

    summarizer.summarize_data(ctx)

    assert ctx.summary_source == "generic_renderer"
    assert "Overall (GEL): 182.10 GEL/MWh" in ctx.summary
    assert "Overall (USD): 63.40 USD/MWh" in ctx.summary
    assert "Summer (GEL): 160.55 GEL/MWh" in ctx.summary
    assert "Summer (USD): 55.80 USD/MWh" in ctx.summary
    assert "Winter (GEL): 208.46 GEL/MWh" in ctx.summary
    assert "Winter (USD): 62.59 USD/MWh" in ctx.summary
