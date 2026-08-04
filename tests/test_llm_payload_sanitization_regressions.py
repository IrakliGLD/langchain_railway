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


def test_overlong_candidate_topics_are_ranked_and_clamped():
    """2026-08-04 job 83010f04: six valid topics failed the whole analysis.

    KnowledgeInfo caps candidate_topics at five. The sanitizer already drops
    unknown names rather than crash, but never enforced the length, so one
    extra valid topic discarded the entire QuestionAnalysis and the
    market_design_context track ran with no analyzer at all.
    """
    from contracts.question_analysis import MAX_CANDIDATE_TOPICS

    payload = {
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.95},
                {"name": "market_structure", "score": 0.9},
                {"name": "generation_mix", "score": 0.85},
                {"name": "cross_border_trade", "score": 0.8},
                {"name": "currency_influence", "score": 0.75},
                {"name": "pso_trading", "score": 0.7},
            ]
        }
    }

    sanitized = _sanitize_question_analysis_payload(payload)
    topics = sanitized["knowledge"]["candidate_topics"]

    assert len(topics) == MAX_CANDIDATE_TOPICS
    # The least confident topic is the one dropped, not an arbitrary one.
    assert [topic["name"] for topic in topics] == [
        "balancing_price",
        "market_structure",
        "generation_mix",
        "cross_border_trade",
        "currency_influence",
    ]


def test_clamped_candidate_topics_still_validate_against_the_contract():
    """The clamp exists to keep the payload constructible, so prove it does."""
    from contracts.question_analysis import KnowledgeInfo

    payload = {
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.5},
                {"name": "market_structure", "score": 0.5},
                {"name": "generation_mix", "score": 0.5},
                {"name": "cross_border_trade", "score": 0.5},
                {"name": "currency_influence", "score": 0.5},
                {"name": "pso_trading", "score": 0.5},
                {"name": "seasonal_patterns", "score": 0.5},
            ]
        }
    }

    sanitized = _sanitize_question_analysis_payload(payload)

    KnowledgeInfo.model_validate(sanitized["knowledge"])


def test_equal_scoring_candidate_topics_clamp_deterministically():
    """Ties must not reorder between runs and change which topic survives."""
    payload = {
        "knowledge": {
            "candidate_topics": [
                {"name": "seasonal_patterns", "score": 0.6},
                {"name": "balancing_price", "score": 0.6},
                {"name": "market_structure", "score": 0.6},
                {"name": "generation_mix", "score": 0.6},
                {"name": "cross_border_trade", "score": 0.6},
                {"name": "currency_influence", "score": 0.6},
            ]
        }
    }

    first = _sanitize_question_analysis_payload(payload)["knowledge"][
        "candidate_topics"
    ]
    second = _sanitize_question_analysis_payload(payload)["knowledge"][
        "candidate_topics"
    ]

    assert first == second
    assert len(first) == 5


def test_candidate_topics_within_the_cap_keep_their_given_order():
    """Clamping must not reshuffle a list that was already valid."""
    payload = {
        "knowledge": {
            "candidate_topics": [
                {"name": "balancing_price", "score": 0.4},
                {"name": "market_structure", "score": 0.9},
            ]
        }
    }

    topics = _sanitize_question_analysis_payload(payload)["knowledge"][
        "candidate_topics"
    ]

    assert [topic["name"] for topic in topics] == [
        "balancing_price",
        "market_structure",
    ]
