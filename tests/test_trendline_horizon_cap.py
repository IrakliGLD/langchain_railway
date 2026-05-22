"""Phase D (2026-05-22) — trendline horizon cap at half the history depth.

Production trace c507e4d7 (pre-Phase-B) had 12 monthly rows of source data
but ``trendline_extend_to`` was ``2036-12-01`` (~11 years out). Extrapolating
linear regression across 11 future years from 1 year of data is mathematically
meaningless. Phase D caps the trendline projection at ``history_years // 2``
so the deterministic NUMBERS in ``stats_hint`` stay within statistical
plausibility, regardless of what horizon the user asked for in the query
text. The LLM can still discuss longer horizons via structural-drivers
caveats in the narrative (per ``skills/answer-composer/references/forecast-caveats.md``).
"""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


def test_cap_trims_long_horizon_to_history_half():
    """The Q2 production scenario — 1 year of data, 11-year horizon → cap to 1."""
    from agent.analyzer import _cap_trendline_horizon_to_history_depth

    df = pd.DataFrame({
        "date": pd.to_datetime([f"2025-{m:02d}-01" for m in range(1, 13)]),
        "p_bal_gel": list(range(100, 112)),
    })
    capped = _cap_trendline_horizon_to_history_depth(
        "2036-12-01", df, ["date", "p_bal_gel"]
    )
    # history_years = max(1, (2025-12 - 2025-01)/365 ≈ 0) → floor at 1.
    # max_horizon = max(1, 1//2) = 1. Last date is 2025-12-01, +1 year → 2026-12-01.
    assert capped == "2026-12-01"


def test_cap_trims_to_half_with_multi_year_history():
    """8 years of data → max 4-year projection. Query asking for 10-year
    horizon (2036) gets capped to ~2030."""
    from agent.analyzer import _cap_trendline_horizon_to_history_depth

    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", "2026-04-01", freq="MS"),
        "p_bal_gel": list(range(100, 100 + 100))[:100],  # arbitrary values
    })
    df = df.iloc[:100]
    capped = _cap_trendline_horizon_to_history_depth(
        "2036-12-01", df, ["date", "p_bal_gel"]
    )
    # history = 2018-01-01 → ~2026-04-01 = ~8 years.
    # max_horizon = 8//2 = 4. Last date + 4y ≈ 2030-04-01.
    parsed = pd.to_datetime(capped)
    assert 2030 <= parsed.year <= 2031, f"expected ~2030, got {capped}"


def test_cap_preserves_horizon_within_bounds():
    """If the requested horizon is already within history//2, leave it alone."""
    from agent.analyzer import _cap_trendline_horizon_to_history_depth

    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", "2026-04-01", freq="MS"),
        "p_bal_gel": [1.0] * 100,
    })
    df = df.iloc[:100]
    # 8-year history → max 4-year horizon. Asking for 2027 is well within.
    capped = _cap_trendline_horizon_to_history_depth(
        "2027-06-01", df, ["date", "p_bal_gel"]
    )
    assert capped == "2027-06-01"


def test_cap_floor_at_one_year_for_minimal_history():
    """Even with < 1 year of data, the floor lets a 1-year horizon survive."""
    from agent.analyzer import _cap_trendline_horizon_to_history_depth

    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
        "p_bal_gel": [100.0, 105.0, 110.0],
    })
    # 3 months of history → floor at 1-year horizon.
    capped = _cap_trendline_horizon_to_history_depth(
        "2030-12-01", df, ["date", "p_bal_gel"]
    )
    # hist_max = 2026-03-01 + 1 year = 2027-03-01.
    parsed = pd.to_datetime(capped)
    assert parsed.year == 2027


def test_cap_returns_input_when_no_data():
    """Empty/missing dataframe → cap is a no-op."""
    from agent.analyzer import _cap_trendline_horizon_to_history_depth

    assert _cap_trendline_horizon_to_history_depth("2036-12-01", None, []) == "2036-12-01"
    assert _cap_trendline_horizon_to_history_depth("2036-12-01", pd.DataFrame(), []) == "2036-12-01"


def test_cap_returns_input_when_no_time_column():
    """No date/year/month/period column in cols → cap is a no-op."""
    from agent.analyzer import _cap_trendline_horizon_to_history_depth

    df = pd.DataFrame({"name": ["a", "b"], "value": [1.0, 2.0]})
    assert _cap_trendline_horizon_to_history_depth(
        "2036-12-01", df, ["name", "value"]
    ) == "2036-12-01"


def test_cap_returns_input_when_extend_to_unparseable():
    """If the extend_to string is malformed, leave it alone — caller will
    handle the downstream error."""
    from agent.analyzer import _cap_trendline_horizon_to_history_depth

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", "2026-04-01", freq="MS"),
        "p_bal_gel": [1.0] * 76,
    })
    df = df.iloc[:76]
    assert _cap_trendline_horizon_to_history_depth("not-a-date", df, ["date"]) == "not-a-date"
    assert _cap_trendline_horizon_to_history_depth(None, df, ["date"]) is None


def test_cap_handles_unsorted_dates():
    """The cap uses min/max of the date series, so unsorted input still works."""
    from agent.analyzer import _cap_trendline_horizon_to_history_depth

    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-06-01", "2018-01-01", "2026-04-01", "2020-09-01",
        ]),
        "p_bal_gel": [1.0, 2.0, 3.0, 4.0],
    })
    capped = _cap_trendline_horizon_to_history_depth(
        "2050-01-01", df, ["date", "p_bal_gel"]
    )
    # min=2018-01-01, max=2026-04-01, history=~8 years, max_horizon=4.
    # hist_max + 4y ≈ 2030-04-01.
    parsed = pd.to_datetime(capped)
    assert 2030 <= parsed.year <= 2031
