"""Stable acceptance fixtures for the report research-track redesign."""

from __future__ import annotations

import json
from pathlib import Path

_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "report_research_golden_cases.json"
)


def test_report_research_golden_cases_cover_required_routing_profiles():
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))

    assert payload["contract_version"] == "report-golden-cases-v1"
    cases = payload["cases"]
    assert len(cases) >= 7
    profiles = {case["profile"] for case in cases}
    assert {
        "compound",
        "quantitative",
        "knowledge",
        "mixed",
        "noisy",
        "multilingual",
        "ambiguous",
    }.issubset(profiles)

    compound = next(case for case in cases if case["id"] == "prices_security_market_model")
    assert compound["expected_required_topics"] == [
        "price_dynamics",
        "energy_security",
        "market_model",
    ]
    assert set(compound["expected_collectors"]) >= {
        "prices",
        "generation_mix",
        "vector_knowledge",
    }
    assert compound["minimum_numeric_observations"] >= 4
    assert compound["minimum_chart_count"] >= 2

    multilingual = next(
        case for case in cases if case["profile"] == "multilingual"
    )
    assert any(
        "\u10a0" <= character <= "\u10ff"
        for character in multilingual["query"]
    )


def test_report_research_golden_cases_have_bounded_consistent_expectations():
    cases = json.loads(_FIXTURE.read_text(encoding="utf-8"))["cases"]

    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["query"].strip()
        assert 1 <= len(case["expected_required_topics"]) <= 4
        assert 1 <= len(case["expected_collectors"]) <= 6
        assert 0 <= case["minimum_numeric_observations"] <= 20
        assert 0 <= case["minimum_chart_count"] <= 3
        if case["minimum_chart_count"]:
            assert case["minimum_numeric_observations"] > 0
