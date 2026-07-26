"""Scenario evidence rendering must preserve dimensional and currency semantics."""

from __future__ import annotations

from agent.generic_renderer import render
from contracts.evidence_frames import ScenarioFrame
from contracts.question_analysis import AnswerKind


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "metric_name": "scenario_payoff",
        "metric": "p_bal_gel",
        "scenario_factor": 70.0,
        "scenario_energy_mwh": None,
        "scenario_capacity_mw": None,
        "scenario_scope": "latest",
        "scenario_aggregation": "mean",
        "aggregate_result": 10.0,
        "row_count": 1,
        "period_range": "2024-03 to 2024-03",
        "min_period_value": 10.0,
        "max_period_value": 10.0,
        "mean_period_value": 10.0,
        "formula": "70 - p_bal_gel",
        "source_unit": "GEL/MWh",
        "result_unit": "GEL/MWh",
        "positive_sum": None,
        "negative_sum": None,
        "positive_count": 1,
        "negative_count": 0,
        "market_component_aggregate": None,
        "combined_total_aggregate": None,
        "baseline_aggregate": None,
        "delta_aggregate": None,
        "delta_percent": None,
    }
    row.update(overrides)
    return row


def test_payoff_without_energy_is_rendered_as_rate_not_currency_total():
    frame = ScenarioFrame(rows=[_row(scenario_capacity_mw=2.0)])

    answer = render(frame, AnswerKind.SCENARIO)

    assert answer is not None
    assert "payoff rate" in answer.lower()
    assert "10.00 GEL/MWh" in answer
    assert "2.0 MW" in answer
    assert "capacity alone" in answer.lower()
    assert "Net total payoff" not in answer
    assert "USD" not in answer


def test_payoff_with_energy_uses_metric_currency_for_totals():
    frame = ScenarioFrame(rows=[_row(
        scenario_energy_mwh=100.0,
        scenario_capacity_mw=2.0,
        scenario_aggregation="sum",
        aggregate_result=1000.0,
        result_unit="GEL",
        positive_sum=1000.0,
        market_component_aggregate=6000.0,
        combined_total_aggregate=7000.0,
    )])

    answer = render(frame, AnswerKind.SCENARIO)

    assert answer is not None
    assert "100.0 MWh per period" in answer
    assert "1,000" in answer
    assert "GEL" in answer
    assert "USD" not in answer


def test_scale_result_includes_the_source_metric_unit_and_scope():
    frame = ScenarioFrame(rows=[_row(
        metric_name="scenario_scale",
        scenario_factor=1.2,
        scenario_energy_mwh=None,
        scenario_capacity_mw=None,
        aggregate_result=72.0,
        baseline_aggregate=60.0,
        delta_aggregate=12.0,
        delta_percent=20.0,
        min_period_value=72.0,
        max_period_value=72.0,
        mean_period_value=72.0,
        result_unit="GEL/MWh",
    )])

    answer = render(frame, AnswerKind.SCENARIO)

    assert answer is not None
    assert "72.00 GEL/MWh" in answer
    assert "latest observation" in answer.lower()


def test_all_sensitivity_rows_are_rendered():
    first = _row(
        metric_name="scenario_scale",
        scenario_factor=1.1,
        aggregate_result=66.0,
        baseline_aggregate=60.0,
        delta_aggregate=6.0,
        delta_percent=10.0,
        result_unit="GEL/MWh",
    )
    second = _row(
        metric_name="scenario_scale",
        scenario_factor=1.2,
        aggregate_result=72.0,
        baseline_aggregate=60.0,
        delta_aggregate=12.0,
        delta_percent=20.0,
        result_unit="GEL/MWh",
    )

    answer = render(ScenarioFrame(rows=[first, second]), AnswerKind.SCENARIO)

    assert answer is not None
    assert "Scenario 1" in answer
    assert "Scenario 2" in answer
    assert "66.00 GEL/MWh" in answer
    assert "72.00 GEL/MWh" in answer
