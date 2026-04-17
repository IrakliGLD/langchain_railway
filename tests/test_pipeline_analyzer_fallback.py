"""F1 regression: analyzer-absent Stage 3 routing.

When the question analyzer is shadow/failed, Stage 3 enrichments (share,
forecast, why/explanation) must still fire via the keyword-derived fallback
published at ``ctx.effective_answer_kind``.  See plan
``dreamy-imagining-ritchie.md`` Finding 1.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd  # noqa: E402

from agent import pipeline  # noqa: E402
from contracts.question_analysis import AnswerKind  # noqa: E402
from models import QueryContext  # noqa: E402


def _ctx(query: str) -> QueryContext:
    ctx = QueryContext(query=query)
    # Explicitly ensure no authoritative analyzer output so we exercise
    # the keyword-fallback branch only.
    ctx.question_analysis = None
    ctx.question_analysis_source = ""
    return ctx


def test_resolve_effective_answer_kind_returns_qa_value_when_authoritative():
    """Authoritative QA wins — no keyword inference runs."""
    from types import SimpleNamespace

    ctx = QueryContext(query="why did balancing price rise")
    # Simulate authoritative analyzer output without reconstructing the full
    # pydantic tree.  `has_authoritative_question_analysis` just checks that
    # `question_analysis` is non-None and the source tag is ``llm_active``.
    ctx.question_analysis = SimpleNamespace(answer_kind=AnswerKind.FORECAST)
    ctx.question_analysis_source = "llm_active"

    assert pipeline._resolve_effective_answer_kind(ctx) == AnswerKind.FORECAST


def test_resolve_effective_answer_kind_forecast_keyword_fallback():
    ctx = _ctx("forecast the balancing price for next year")
    assert pipeline._resolve_effective_answer_kind(ctx) == AnswerKind.FORECAST


def test_resolve_effective_answer_kind_explanation_keyword_fallback():
    ctx = _ctx("why did balancing price rise in Q1")
    assert pipeline._resolve_effective_answer_kind(ctx) == AnswerKind.EXPLANATION


def test_resolve_effective_answer_kind_comparison_keyword_fallback():
    ctx = _ctx("compare balancing and deregulated prices in 2024")
    assert pipeline._resolve_effective_answer_kind(ctx) == AnswerKind.COMPARISON


def test_resolve_effective_answer_kind_share_keyword_maps_to_comparison():
    ctx = _ctx("what is the share of hydro in the balancing composition")
    # Share/composition queries route via COMPARISON shape (entity rows +
    # contribution framing).  The exact tool still comes from the router.
    assert pipeline._resolve_effective_answer_kind(ctx) == AnswerKind.COMPARISON


def test_resolve_effective_answer_kind_list_via_classify_query_type():
    ctx = _ctx("list all regulated plants")
    assert pipeline._resolve_effective_answer_kind(ctx) == AnswerKind.LIST


def test_resolve_effective_answer_kind_historical_trend_maps_to_timeseries():
    ctx = _ctx("show the historical trend of balancing price")
    assert pipeline._resolve_effective_answer_kind(ctx) == AnswerKind.TIMESERIES


def test_resolve_effective_answer_kind_empty_query_returns_none():
    ctx = _ctx("")
    assert pipeline._resolve_effective_answer_kind(ctx) is None


def test_analyzer_enrich_forecast_fires_without_authoritative_qa(monkeypatch):
    """Stage 3 forecast branch must run when effective_answer_kind=FORECAST,
    even with no authoritative QA (the F1 regression surface).
    """
    from agent import analyzer

    ctx = _ctx("forecast balancing price for 2026")
    ctx.effective_answer_kind = AnswerKind.FORECAST
    ctx.df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=12, freq="MS"),
        "p_bal_gel": [10.0 + i for i in range(12)],
    })
    ctx.cols = list(ctx.df.columns)
    ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False, name=None)]

    called = {"forecast": False}

    def _fake_forecast(df, query):
        called["forecast"] = True
        return df, "FAKE FORECAST NOTE"

    monkeypatch.setattr(analyzer, "_generate_cagr_forecast", _fake_forecast)
    # Short-circuit heavy branches so the test stays focused.
    monkeypatch.setattr(analyzer, "_build_why_context", lambda ctx: None)
    monkeypatch.setattr(analyzer, "_build_standalone_analysis_evidence", lambda ctx: None)
    monkeypatch.setattr(analyzer, "_materialize_chart_override", lambda ctx: None)
    monkeypatch.setattr(analyzer, "_append_column_aggregates", lambda ctx: None)

    analyzer.enrich(ctx)

    assert called["forecast"] is True
    assert "FAKE FORECAST NOTE" in ctx.stats_hint


def test_analyzer_enrich_why_fires_without_authoritative_qa(monkeypatch):
    """Stage 3 why/EXPLANATION branch must run with no authoritative QA."""
    from agent import analyzer

    ctx = _ctx("why did balancing price rise in early 2024")
    ctx.effective_answer_kind = AnswerKind.EXPLANATION
    ctx.df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="MS"),
        "p_bal_gel": [10.0, 12.0, 15.0],
    })
    ctx.cols = list(ctx.df.columns)
    ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False, name=None)]

    called = {"why": False}

    def _fake_why(c):
        called["why"] = True

    monkeypatch.setattr(analyzer, "_build_why_context", _fake_why)
    monkeypatch.setattr(analyzer, "_generate_cagr_forecast", lambda df, q: (df, ""))
    monkeypatch.setattr(analyzer, "_build_standalone_analysis_evidence", lambda ctx: None)
    monkeypatch.setattr(analyzer, "_materialize_chart_override", lambda ctx: None)
    monkeypatch.setattr(analyzer, "_append_column_aggregates", lambda ctx: None)

    analyzer.enrich(ctx)

    assert called["why"] is True


def test_analyzer_enrich_timeseries_does_not_activate_forecast_trendlines(monkeypatch):
    """Historical trend summaries must not trigger future trendline extension."""
    from agent import analyzer

    ctx = _ctx("show the historical trend of balancing price")
    ctx.effective_answer_kind = AnswerKind.TIMESERIES
    ctx.df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=4, freq="MS"),
        "p_bal_gel": [100.0, 105.0, 102.0, 108.0],
    })
    ctx.cols = list(ctx.df.columns)
    ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False, name=None)]

    monkeypatch.setattr(analyzer, "_build_why_context", lambda ctx: None)
    monkeypatch.setattr(
        analyzer,
        "_generate_cagr_forecast",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("forecast branch should not run")),
    )
    monkeypatch.setattr(
        analyzer,
        "_precalculate_trendlines",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("trendline extension should not run")),
    )
    monkeypatch.setattr(analyzer, "_build_standalone_analysis_evidence", lambda ctx: None)
    monkeypatch.setattr(analyzer, "_materialize_chart_override", lambda ctx: None)
    monkeypatch.setattr(analyzer, "_append_column_aggregates", lambda ctx: None)

    analyzer.enrich(ctx)

    assert ctx.add_trendlines is False
    assert ctx.trendline_extend_to is None
    assert "TRENDLINE FORECASTS" not in ctx.stats_hint


def test_generate_cagr_forecast_supports_yearly_integer_price_history():
    from agent import analyzer

    df = pd.DataFrame({
        "year": [2021, 2022, 2023, 2024, 2025],
        "p_bal_gel": [90.0, 100.0, 110.0, 121.0, 133.1],
    })

    out, note = analyzer._generate_cagr_forecast(
        df,
        "Forecast balancing electricity price for the next 5 years, but do not use 2020 data.",
    )

    forecast_rows = out[out["is_forecast"]]

    assert "Forecast skipped" not in note
    assert "yearly averages" in note
    assert len(forecast_rows) == 5
    assert set(forecast_rows["year"].dt.year.tolist()) == {2026, 2027, 2028, 2029, 2030}
    assert forecast_rows["year"].dt.year.min() == 2026
    assert "season" not in forecast_rows.columns or forecast_rows["season"].isna().all()


def test_generate_cagr_forecast_reports_usable_yearly_point_count():
    from agent import analyzer

    df = pd.DataFrame({
        "year": [2025],
        "p_bal_gel": [133.1],
    })

    out, note = analyzer._generate_cagr_forecast(
        df,
        "Forecast balancing electricity price for the next 5 years, but do not use 2020 data.",
    )

    assert out.equals(df)
    assert note == "Forecast skipped: only 1 usable yearly point after normalization/filtering."
