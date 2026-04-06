import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import summarizer
from models import QueryContext


def test_summarize_data_uses_deterministic_trendline_forecast_answer(monkeypatch):
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

    ctx = QueryContext(
        query="forecast balancing electricity price for 2030",
        preview="date,p_bal_gel,p_bal_usd,xrate\n2025-09-01,131.7,47.5,2.77",
        stats_hint=stats_hint,
        provenance_cols=["date", "p_bal_gel", "p_bal_usd", "xrate"],
        provenance_rows=[("2025-09-01", 131.7, 47.5, 2.77)],
    )

    summarizer.summarize_data(ctx)

    assert ctx.summary_source == "deterministic_trendline_forecast_direct"
    assert "145.25 GEL/MWh" in ctx.summary
    assert "52.40 USD/MWh" in ctx.summary
    assert "December 2030" in ctx.summary
    assert "R²=0.412" in ctx.summary
    assert "R²=0.398" in ctx.summary
    assert "2.77" not in ctx.summary
