"""Catalog-level regressions for analyzer contract wording."""

from contracts.question_analysis_catalogs import (
    QUESTION_ANALYSIS_ANSWER_KIND_GUIDE,
    QUESTION_ANALYSIS_QUERY_TYPE_GUIDE,
)


def _entry(entries, name: str) -> dict:
    for entry in entries:
        if entry["name"] == name:
            return entry
    raise AssertionError(f"missing entry: {name}")


def test_answer_kind_guide_distinguishes_historical_trend_from_forecast():
    timeseries = _entry(QUESTION_ANALYSIS_ANSWER_KIND_GUIDE, "timeseries")["use_for"].lower()
    forecast = _entry(QUESTION_ANALYSIS_ANSWER_KIND_GUIDE, "forecast")["use_for"].lower()

    assert "historical trend" in timeseries
    assert "historical trend" in forecast
    assert "not forecast" in forecast


def test_query_type_guide_distinguishes_historical_trend_from_forecast():
    data_retrieval = _entry(QUESTION_ANALYSIS_QUERY_TYPE_GUIDE, "data_retrieval")["use_for"].lower()
    forecast = _entry(QUESTION_ANALYSIS_QUERY_TYPE_GUIDE, "forecast")["use_for"].lower()

    assert "trend summaries" in data_retrieval
    assert "not forecast" in forecast
