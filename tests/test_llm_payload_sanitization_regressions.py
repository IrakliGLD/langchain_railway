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
