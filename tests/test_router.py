"""
Tests for fast pre-LLM typed tool routing.
"""
from agent.router import match_tool


def test_match_tariff_tool():
    inv = match_tool("Compare Enguri and Gardabani tariff in 2024 in USD")
    assert inv is not None
    assert inv.name == "get_tariffs"
    assert inv.params["currency"] == "usd"
    assert inv.params["start_date"] == "2024-01-01"
    assert inv.params["end_date"] == "2024-12-31"
    assert "enguri" in (inv.params.get("entities") or [])
    assert "gardabani_tpp" in (inv.params.get("entities") or [])


def test_match_balancing_composition_tool():
    inv = match_tool("What was share of import in balancing electricity between 2022 and 2024?")
    assert inv is not None
    assert inv.name == "get_balancing_composition"
    assert inv.params["start_date"] == "2022-01-01"
    assert inv.params["end_date"] == "2024-12-31"
    assert "import" in (inv.params.get("entities") or [])


def test_match_generation_mix_tool():
    inv = match_tool("Show generation mix by technology from 2020 to 2023")
    assert inv is not None
    assert inv.name == "get_generation_mix"
    assert inv.params["mode"] == "quantity"
    assert inv.params["granularity"] == "monthly"
    assert inv.params["start_date"] == "2020-01-01"
    assert inv.params["end_date"] == "2023-12-31"


def test_match_price_tool():
    inv = match_tool("Balancing price trend in USD from 2021 to 2022")
    assert inv is not None
    assert inv.name == "get_prices"
    assert inv.params["metric"] == "balancing"
    assert inv.params["currency"] == "usd"
    assert inv.params["start_date"] == "2021-01-01"
    assert inv.params["end_date"] == "2022-12-31"


def test_semantic_fallback_matches_long_tail_price_intent():
    inv = match_tool("How did balancing electricity cost evolve annually in USD since 2020?")
    assert inv is not None
    assert inv.name == "get_prices"
    assert inv.params["currency"] == "usd"
    assert inv.params["metric"] == "balancing"
    assert inv.params["granularity"] == "yearly"
    assert "semantic fallback" in (inv.reason or "").lower()


def test_no_tool_match_for_generic_text():
    inv = match_tool("hello, can you help me")
    assert inv is None


def test_explanation_mode_expands_month_range():
    # Regular matching
    inv_normal = match_tool("balancing price in november 2022")
    assert inv_normal.params["start_date"] == "2022-11-01"
    assert inv_normal.params["end_date"] == "2022-11-01"
    
    # Explanation matching expands to start of previous month
    inv_exp = match_tool("why did balancing price change in november 2022?", is_explanation=True)
    assert inv_exp.params["start_date"] == "2022-10-01"
    assert inv_exp.params["end_date"] == "2022-11-01"


def test_explanation_mode_expands_year_range():
    # Regular matching
    inv_normal = match_tool("generation mix in 2024")
    assert inv_normal.params["start_date"] == "2024-01-01"
    assert inv_normal.params["end_date"] == "2024-12-31"
    
    # Explanation matching expands to start of previous year
    inv_exp = match_tool("explain the generation mix change in 2024", is_explanation=True)
    assert inv_exp.params["start_date"] == "2023-01-01"
    assert inv_exp.params["end_date"] == "2024-12-31"
