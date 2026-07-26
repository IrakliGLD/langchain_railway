"""Deterministic scenario-input grounding and dimensional-contract tests."""

from __future__ import annotations

from agent.scenario_contract import ground_scenario_request


def _request(
    metric_name: str,
    metric: str,
    factor: float,
    **extra: object,
) -> dict[str, object]:
    return {
        "metric_name": metric_name,
        "metric": metric,
        "scenario_factor": factor,
        **extra,
    }


def test_date_is_not_a_scenario_parameter_anchor():
    request = _request("scenario_scale", "p_bal_usd", 1.34)

    assert ground_scenario_request(
        "what if balancing prices in 2024 change?",
        request,
    ) is None


def test_analyzer_factor_must_match_the_user_parameter():
    request = _request("scenario_scale", "p_bal_usd", 1.34)

    assert ground_scenario_request(
        "what if balancing prices rise by 20%?",
        request,
    ) is None


def test_parameter_subject_must_match_the_metric_being_transformed():
    request = _request("scenario_scale", "p_bal_usd", 1.2)

    assert ground_scenario_request(
        "if PPA share rises by 20%, how will balancing price change?",
        request,
    ) is None


def test_cross_metric_driver_question_is_not_a_mechanical_scenario():
    request = _request("scenario_scale", "share_all_ppa", 1.2)

    assert ground_scenario_request(
        "if PPA share rises by 20%, how will balancing price change?",
        request,
    ) is None


def test_cross_metric_guard_uses_canonical_translation_for_multilingual_query():
    request = _request("scenario_scale", "share_all_ppa", 1.2)

    assert ground_scenario_request(
        "თუ PPA წილი 20%-ით გაიზრდება, ფასი როგორ შეიცვლება?",
        request,
        canonical_query=(
            "If PPA share rises by 20%, how will balancing price change?"
        ),
    ) is None


def test_scale_parameter_and_subject_are_normalized_from_the_query():
    request = _request("scenario_scale", "p_bal_usd", 1.2)

    grounded = ground_scenario_request(
        "what if the balancing price rises by 20%?",
        request,
    )

    assert grounded is not None
    assert grounded["metric"] == "p_bal_usd"
    assert grounded["scenario_factor"] == 1.2
    assert grounded["scenario_scope"] == "latest"
    assert grounded["scenario_aggregation"] == "mean"


def test_offset_lower_is_normalized_as_a_negative_amount():
    request = _request("scenario_offset", "p_bal_gel", -30.0)

    grounded = ground_scenario_request(
        "what if balancing price were 30 GEL/MWh lower?",
        request,
    )

    assert grounded is not None
    assert grounded["scenario_factor"] == -30.0


def test_offset_currency_mismatch_fails_closed():
    request = _request("scenario_offset", "p_bal_gel", 30.0)

    assert ground_scenario_request(
        "what if balancing price were 30 USD/MWh higher?",
        request,
    ) is None


def test_semantic_offset_metric_resolves_explicit_currency():
    request = _request("scenario_offset", "balancing", 30.0)

    grounded = ground_scenario_request(
        "what if balancing price were 30 USD/MWh higher?",
        request,
    )

    assert grounded is not None
    assert grounded["metric"] == "p_bal_usd"


def test_unsupported_offset_currency_fails_closed():
    request = _request("scenario_offset", "balancing", 30.0)

    assert ground_scenario_request(
        "what if balancing price were 30 EUR/MWh higher?",
        request,
    ) is None


def test_payoff_capacity_is_not_reinterpreted_as_energy():
    request = _request("scenario_payoff", "p_bal_usd", 60.0)

    grounded = ground_scenario_request(
        "CfD payoff against balancing price at a 60 USD/MWh strike for 2 MW",
        request,
    )

    assert grounded is not None
    assert grounded["scenario_capacity_mw"] == 2.0
    assert grounded["scenario_energy_mwh"] is None


def test_payoff_energy_is_grounded_and_normalized_to_mwh():
    request = _request("scenario_payoff", "p_bal_usd", 60.0)

    grounded = ground_scenario_request(
        "CfD payoff against balancing price at 60 USD/MWh for 2 GWh per month",
        request,
    )

    assert grounded is not None
    assert grounded["scenario_energy_mwh"] == 2000.0
    assert grounded["scenario_capacity_mw"] is None
    assert grounded["scenario_aggregation"] == "sum"


def test_total_payoff_without_energy_stays_a_per_mwh_mean():
    request = _request("scenario_payoff", "p_bal_usd", 60.0)

    grounded = ground_scenario_request(
        "Calculate total CfD payoff against balancing price at a "
        "60 USD/MWh strike for 2 MW",
        request,
    )

    assert grounded is not None
    assert grounded["scenario_energy_mwh"] is None
    assert grounded["scenario_aggregation"] == "mean"


def test_semantic_price_metric_resolves_explicit_currency():
    request = _request("scenario_payoff", "balancing", 60.0)

    grounded = ground_scenario_request(
        "CfD payoff against balancing price at a 60 USD/MWh strike",
        request,
    )

    assert grounded is not None
    assert grounded["metric"] == "p_bal_usd"


def test_unsupported_payoff_currency_fails_closed():
    request = _request("scenario_payoff", "balancing", 60.0)

    assert ground_scenario_request(
        "CfD payoff against balancing price at a 60 EUR/MWh strike",
        request,
    ) is None


def test_over_baseline_phrase_does_not_create_a_period_scope():
    request = _request("scenario_scale", "p_bal_gel", 1.2)

    grounded = ground_scenario_request(
        "What if balancing price rises 20% over baseline?",
        request,
    )

    assert grounded is not None
    assert grounded["scenario_scope"] == "latest"


def test_payoff_without_an_explicit_reference_metric_fails_closed():
    request = _request("scenario_payoff", "p_bal_usd", 60.0)

    assert ground_scenario_request(
        "what is the CfD payoff at a strike of 60 USD/MWh?",
        request,
    ) is None


def test_historical_price_near_cfd_text_is_not_reinterpreted_as_a_strike():
    request = _request("scenario_payoff", "p_bal_usd", 60.0)

    assert ground_scenario_request(
        "The balancing price was 60 USD/MWh in a 2024 CfD report.",
        request,
    ) is None


def test_currency_mismatch_fails_closed():
    request = _request("scenario_payoff", "p_bal_gel", 60.0)

    assert ground_scenario_request(
        "CfD payoff against balancing price at a 60 USD/MWh strike",
        request,
    ) is None


def test_legacy_volume_field_fails_closed_in_grounding():
    request = _request(
        "scenario_payoff",
        "p_bal_usd",
        60.0,
        scenario_volume=2.0,
    )

    assert ground_scenario_request(
        "CfD payoff against balancing price at a 60 USD/MWh strike",
        request,
    ) is None
