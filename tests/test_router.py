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
    
    # Explanation matching expands back 1 year + 1 month for YoY/MoM enrichment
    inv_exp = match_tool("why did balancing price change in november 2022?", is_explanation=True)
    assert inv_exp.params["start_date"] == "2021-10-01"
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


def test_semantic_threshold_routes_at_055():
    """Query scoring ~0.57 in semantic fallback routes at the new 0.55 threshold.

    'regulated gnerc' hits 2 of 10 get_tariffs terms → 2/3.5 ≈ 0.571.
    No other tool scores above 0. Margin > 0.08.  Old threshold 0.62 would reject.
    """
    inv = match_tool("regulated gnerc decisions in 2023")
    assert inv is not None
    assert inv.name == "get_tariffs"
    assert "semantic fallback" in (inv.reason or "").lower()


def test_semantic_margin_gate_rejects_ambiguous():
    """When top two tool scores are within 0.08, query is rejected even above 0.55.

    'costs dynamics usd gel volatility of output hydro solar' scores:
      get_prices:         5 hits / 7.35 denom ≈ 0.680
      get_generation_mix: 3 hits / 4.90 denom ≈ 0.612
    Margin ≈ 0.068 < 0.08 → rejected despite both being above 0.55.
    None of these words trigger deterministic rules.
    """
    inv = match_tool("costs dynamics usd gel volatility of output hydro solar")
    assert inv is None


def test_semantic_scores_populated_on_miss():
    """_last_semantic_scores is populated even when semantic match is rejected."""
    from agent.router import _last_semantic_scores
    # Generic query that won't match anything deterministically
    match_tool("hello what is happening")
    # Scores should have been computed for all tools
    assert isinstance(_last_semantic_scores, dict)
    assert len(_last_semantic_scores) == len(
        __import__("agent.router", fromlist=["_SEMANTIC_TOOL_TERMS"])._SEMANTIC_TOOL_TERMS
    )
