"""Production-drift regressions for structured question-analysis payloads."""

from core.llm_payloads import _sanitize_question_analysis_payload


def test_invalid_sql_dimensions_are_mapped_or_dropped():
    payload = {
        "sql_hints": {
            "dimensions": [
                "share_import",
                "balancing",
                "deregulated",
                "regulated",
                "ppa_cfd_price",
                "invented_dimension",
            ]
        }
    }

    sanitized = _sanitize_question_analysis_payload(payload)

    assert sanitized["sql_hints"]["dimensions"] == ["share", "price", "regulation_status"]


def test_analysis_requirement_flags_are_relocated_out_of_classification():
    """2026-07-27 trace c7823cc9: the analyzer emitted the three requirement
    flags inside classification, where extra=forbid rejected the entire
    QuestionAnalysis and dropped the pipeline to heuristic routing."""

    payload = {
        "classification": {
            "query_type": "data_retrieval",
            "needs_driver_analysis": True,
            "needs_trend_context": True,
            "needs_correlation_context": False,
        }
    }

    sanitized = _sanitize_question_analysis_payload(payload)

    assert "needs_driver_analysis" not in sanitized["classification"]
    assert "needs_trend_context" not in sanitized["classification"]
    assert "needs_correlation_context" not in sanitized["classification"]
    assert sanitized["analysis_requirements"] == {
        "needs_driver_analysis": True,
        "needs_trend_context": True,
        "needs_correlation_context": False,
    }


def test_relocated_flags_never_override_the_analyzer_s_own_requirements():
    payload = {
        "classification": {"needs_driver_analysis": True},
        "analysis_requirements": {"needs_driver_analysis": False},
    }

    sanitized = _sanitize_question_analysis_payload(payload)

    assert "needs_driver_analysis" not in sanitized["classification"]
    assert sanitized["analysis_requirements"]["needs_driver_analysis"] is False
