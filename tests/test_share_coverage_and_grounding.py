"""2026-07-08 round: generation-share coverage guard (B), grounding rounding
tolerance (Issue 1), and the existing correlation-excludes-constants guard (C).
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import math

import pandas as pd

from agent import summarizer
from analysis.system_quantities import canonicalize_generation_mix_df

# ---------------------------------------------------------------------------
# B — generation-share coverage guard
# ---------------------------------------------------------------------------

def _canon(rows):
    df = pd.DataFrame(rows, columns=["date", "type_tech", "quantity_tech"])
    out = canonicalize_generation_mix_df(df).set_index("period")
    out.index = out.index.astype(str)
    return out


def test_multi_tech_period_keeps_real_share():
    out = _canon([
        ("2023-09-01", "hydro", 807.58),
        ("2023-09-01", "thermal", 301.94),
        ("2023-09-01", "wind", 8.37),
    ])
    row = out.loc[out.index.str.startswith("2023-09")].iloc[0]
    assert round(float(row["share_hydro"]), 2) == 0.72


def test_single_tech_period_share_is_na_not_fabricated_100pct():
    out = _canon([
        ("2023-09-01", "hydro", 807.58),
        ("2023-09-01", "thermal", 301.94),
        ("2023-09-01", "wind", 8.37),
        ("2020-01-01", "hydro", 500.0),  # only hydro recorded → coverage incomplete
    ])
    hydro_only = out.loc[out.index.str.startswith("2020-01")].iloc[0]
    assert math.isnan(float(hydro_only["share_hydro"]))
    # Aggregates would no longer read as "constant 100%": the two periods
    # now have {0.72, NaN} rather than {0.72, 1.0}.
    assert out["share_hydro"].dropna().nunique() == 1
    assert float(out["share_hydro"].max()) < 1.0


def test_real_zero_techs_preserve_genuine_100pct():
    out = _canon([
        ("2020-02-01", "hydro", 400.0),
        ("2020-02-01", "thermal", 0.0),  # explicit 0 row = observed, not absent
        ("2020-02-01", "wind", 0.0),
    ])
    row = out.loc[out.index.str.startswith("2020-02")].iloc[0]
    assert float(row["share_hydro"]) == 1.0


# ---------------------------------------------------------------------------
# Issue 1 — grounding rounding tolerance
# ---------------------------------------------------------------------------

def test_rounded_variants_let_llm_rounding_match_full_precision():
    tokens = {"1514.2836", "859.4"}
    summarizer._add_rounded_source_variants(tokens)
    # LLM wrote "1,514 thousand MWh" → token "1514"; "859" from 859.4.
    assert "1514" in tokens
    assert "1514.28" in tokens
    assert "859" in tokens


def test_rounding_preserves_anti_hallucination_property():
    tokens = {"1514.2836"}
    summarizer._add_rounded_source_variants(tokens)
    # A fabricated number that is NOT a rounding of any real source value
    # still must not appear.
    assert "9999" not in tokens
    assert "1600" not in tokens


def test_integers_are_untouched():
    tokens = {"42", "1000"}
    summarizer._add_rounded_source_variants(tokens)
    assert tokens == {"42", "1000"}


def test_aggregate_tokens_added_in_light_mode():
    """2026-07-09: light-mode descriptive queries quote column means/sums; the
    aggregate tokens must be present (no longer gated on analyst mode) so those
    quotes can ground. Combined with rounding, the LLM's '820' matches."""
    from models import QueryContext

    ctx = QueryContext(query="what can you say about generation mix")
    ctx.mode = "light"
    ctx.df = pd.DataFrame({"quantity_hydro": [810.0, 820.94, 830.0, 819.0]})

    tokens: set = set()
    summarizer._add_aggregate_tokens(tokens, ctx)
    # Mean ~819.985 present at full precision...
    assert any(t.startswith("819") for t in tokens)
    # ...and rounds to the LLM-quoted "820".
    summarizer._add_rounded_source_variants(tokens)
    assert "820" in tokens


def test_aggregate_tokens_skipped_when_no_frame():
    from models import QueryContext

    ctx = QueryContext(query="x")
    ctx.mode = "light"
    tokens: set = set()
    summarizer._add_aggregate_tokens(tokens, ctx)  # empty df → no-op, no raise
    assert tokens == set()


# ---------------------------------------------------------------------------
# C — correlation already excludes zero-variance (constant) columns
# ---------------------------------------------------------------------------

def test_correlation_excludes_constant_columns():
    from agent.analyzer import _build_correlation_matrix_from_frame

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
        "p_bal_gel": [100.0, 110.0, 95.0, 120.0, 130.0, 90.0],
        "share_hydro": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # constant → no correlation
        "total_domestic_generation": [500.0, 480.0, 520.0, 510.0, 530.0, 470.0],
    })
    result = _build_correlation_matrix_from_frame(df)
    # A constant column can have no Pearson correlation, so it is never a driver.
    for target, drivers in result.items():
        assert "share_hydro" not in drivers


def test_supply_and_demand_shares_use_separate_denominators():
    out = _canon([
        ("2024-01-01", "hydro", 60.0),
        ("2024-01-01", "thermal", 40.0),
        ("2024-01-01", "abkhazeti", 20.0),
        ("2024-01-01", "supply-distribution", 80.0),
    ])
    row = out.loc[out.index.str.startswith("2024-01")].iloc[0]

    assert float(row["total_supply"]) == 100.0
    assert float(row["total_demand"]) == 100.0
    assert float(row["share_hydro"]) == 0.6
    assert float(row["share_thermal"]) == 0.4
    assert float(row["share_abkhazeti"]) == 0.2
    assert float(row["share_supply-distribution"]) == 0.8


def test_mixed_side_raw_shares_are_recomputed_by_side():
    df = pd.DataFrame(
        [
            ("2024-01-01", "hydro", 60.0, 0.30),
            ("2024-01-01", "thermal", 40.0, 0.20),
            ("2024-01-01", "abkhazeti", 20.0, 0.10),
            ("2024-01-01", "supply-distribution", 80.0, 0.40),
        ],
        columns=["date", "type_tech", "quantity_tech", "share_tech"],
    )
    out = canonicalize_generation_mix_df(df).set_index("period")
    out.index = out.index.astype(str)
    row = out.loc[out.index.str.startswith("2024-01")].iloc[0]

    assert float(row["share_hydro"]) == 0.6
    assert float(row["share_thermal"]) == 0.4
    assert float(row["share_abkhazeti"]) == 0.2
    assert float(row["share_supply-distribution"]) == 0.8


def test_filtered_raw_share_is_preserved_when_denominator_is_not_visible():
    df = pd.DataFrame(
        [("2024-01-01", "hydro", 60.0, 0.55)],
        columns=["date", "type_tech", "quantity_tech", "share_tech"],
    )
    out = canonicalize_generation_mix_df(df).set_index("period")
    out.index = out.index.astype(str)
    row = out.loc[out.index.str.startswith("2024-01")].iloc[0]

    assert float(row["share_hydro"]) == 0.55


def test_import_dependency_ratio_uses_supply_denominator():
    out = _canon([
        ("2024-01-01", "hydro", 50.0),
        ("2024-01-01", "thermal", 25.0),
        ("2024-01-01", "import", 25.0),
        ("2024-01-01", "abkhazeti", 200.0),
    ])
    row = out.loc[out.index.str.startswith("2024-01")].iloc[0]

    assert float(row["import_dependent_supply"]) == 50.0
    assert float(row["total_supply"]) == 100.0
    assert float(row["import_dependency_ratio"]) == 0.5
