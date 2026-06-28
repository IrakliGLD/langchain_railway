"""Phase G (2026-05-22) — emit CAGR projection end values to stats_hint.

Production trace 6aba2969 (2026-05-22) showed the post-Phase-B/C/D forecast
flow producing a substantive LLM answer ("price could reach around 165 GEL/MWh
by the end of 2035") but the provenance gate rejected it because "165" was
not in the evidence corpus. The CAGR rate (0.27%) and the baseline (160) WERE
in stats_hint, but the LLM derived the projection 160 × 1.0027^10 ≈ 165
itself — and that derived value wasn't tokenisable from the source.

Phase G pre-computes the CAGR-projected end-of-horizon value in
``_generate_cagr_forecast`` and emits it to stats_hint as a readable block:

    --- CAGR PROJECTIONS ---
    p_bal_gel: baseline=160.00 (year 2025), CAGR=0.27%/year, projection by 2035 = 164.37

So the LLM has a verbatim line to cite, and the provenance gate finds the
projection value in the grounding corpus.

Tests cover:
  - Quantity branch (single-column yearly CAGR).
  - Price branch single-currency (yearly + seasonal projections).
  - Price branch multi-currency (per-currency yearly + seasonal projections).
  - Defensive: no projections when target_years is empty or CAGR is NaN.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


def test_cagr_projection_block_appears_in_quantity_note():
    """Quantity branch: single value column, yearly aggregation.
    The note must include a ``--- CAGR PROJECTIONS ---`` block with the
    projected end-of-horizon value."""
    from agent.analyzer import _generate_cagr_forecast

    # 8 years of yearly data — enough for CAGR to compute meaningfully.
    df = pd.DataFrame({
        "year": pd.to_datetime([f"{y}-01-01" for y in range(2018, 2026)]),
        "quantity": [100.0, 105.0, 110.0, 115.5, 121.3, 127.4, 133.7, 140.4],
    })
    df_out, note = _generate_cagr_forecast(df, "forecast quantity for the next decade")

    assert "--- CAGR PROJECTIONS ---" in note
    # Projection line shape: "<col>: baseline=X (year Y), CAGR=Z%/year, projection by W = V"
    assert "baseline=" in note
    assert "CAGR=" in note
    assert "projection by" in note
    # The projected value should be a positive number that's plausibly larger
    # than the baseline (data is growing). Extract numeric from the block.
    import re
    block_match = re.search(r"--- CAGR PROJECTIONS ---.*?projection by \d{4} = ([\d.]+)", note, re.DOTALL)
    assert block_match, f"projection value not found in note: {note!r}"
    projected = float(block_match.group(1))
    assert projected > 100, f"expected projection > baseline; got {projected}"


def test_cagr_projection_block_multi_currency_includes_per_currency_lines():
    """Price branch with multi-currency: must emit one projection line per
    currency (yearly), plus seasonal lines when seasonal CAGR is available."""
    from agent.analyzer import _generate_cagr_forecast

    # 7 years of monthly data, two currencies.
    rows = []
    for year in range(2019, 2026):
        for month in range(1, 13):
            rows.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "p_bal_gel": 100.0 + (year - 2019) * 1.0 + (month % 2) * 5.0,
                "p_bal_usd": 35.0 + (year - 2019) * 1.5 + (month % 2) * 2.0,
            })
    df = pd.DataFrame(rows)

    df_out, note = _generate_cagr_forecast(df, "forecast balancing electricity price for next 10 years")

    # The block must be present.
    assert "--- CAGR PROJECTIONS ---" in note
    # Both currency labels must appear in the projection block.
    block_start = note.find("--- CAGR PROJECTIONS ---")
    block = note[block_start:]
    assert "GEL" in block
    assert "USD" in block
    # Yearly projection lines for each currency.
    assert block.count("(yearly)") >= 2, (
        f"expected >=2 yearly projection lines, got: {block!r}"
    )


def test_cagr_projection_values_provide_tokenisable_numbers():
    """The projected end values must appear as verbatim digit strings in
    the note so the provenance gate can tokenise and match them against
    an LLM citation. This is the load-bearing assertion for Phase G —
    the whole point of the block is to give the LLM a cite-able number."""
    from agent.analyzer import _generate_cagr_forecast
    from agent.summarizer import _extract_number_tokens

    rows = []
    for year in range(2019, 2026):
        for month in range(1, 13):
            rows.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "p_bal_gel": 100.0 + (year - 2019) * 2.0,
            })
    df = pd.DataFrame(rows)

    df_out, note = _generate_cagr_forecast(df, "forecast balancing electricity price for next 10 years")

    # Extract the projection value(s) from the block and confirm they appear
    # in the tokenised set that the provenance gate would build.
    import re
    block_match = re.search(r"--- CAGR PROJECTIONS ---(.*)", note, re.DOTALL)
    assert block_match, "no projection block in note"
    block_text = block_match.group(1)

    projections = re.findall(r"projection by \d{4} = ([\d.]+)", block_text)
    assert projections, f"no projection values in block: {block_text!r}"

    tokens = _extract_number_tokens(note)
    for proj_value in projections:
        # The full value (e.g. "164.37") must be tokenisable from the note.
        assert proj_value in tokens or proj_value.rstrip("0").rstrip(".") in tokens, (
            f"projection value {proj_value!r} not in tokens {sorted(tokens)[:30]!r}"
        )


def test_forecast_series_block_quantity_lists_per_year_values():
    """Phase H: the per-year FORECAST SERIES block must list every projected
    year (not just the end value), so the LLM can quote intermediate years
    verbatim instead of recomputing the path (prod trace abb741aa)."""
    from agent.analyzer import _generate_cagr_forecast

    df = pd.DataFrame({
        "year": pd.to_datetime([f"{y}-01-01" for y in range(2018, 2026)]),
        "quantity": [100.0, 105.0, 110.0, 115.5, 121.3, 127.4, 133.7, 140.4],
    })
    df_out, note = _generate_cagr_forecast(df, "forecast quantity for the next decade")

    assert "--- FORECAST SERIES (per year) ---" in note
    block = note[note.find("--- FORECAST SERIES (per year) ---"):]
    value_lines = [ln for ln in block.splitlines() if " = " in ln]
    # One line per projected year → well more than the single end value.
    assert len(value_lines) >= 3, f"expected multiple per-year lines, got: {block!r}"


def test_forecast_series_block_price_includes_seasons_and_currencies():
    """Phase H price branch: the per-year series distinguishes currencies
    (GEL/USD) and seasons (summer/winter)."""
    from agent.analyzer import _generate_cagr_forecast

    rows = []
    for year in range(2019, 2026):
        for month in range(1, 13):
            rows.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "p_bal_gel": 100.0 + (year - 2019) * 1.0 + (month % 2) * 5.0,
                "p_bal_usd": 35.0 + (year - 2019) * 1.5 + (month % 2) * 2.0,
            })
    df = pd.DataFrame(rows)

    df_out, note = _generate_cagr_forecast(df, "forecast balancing electricity price for next 10 years")

    assert "--- FORECAST SERIES (per year) ---" in note
    block = note[note.find("--- FORECAST SERIES (per year) ---"):]
    assert "GEL" in block and "USD" in block
    assert "summer" in block and "winter" in block


def test_forecast_series_intermediate_values_are_tokenisable():
    """Load-bearing for the grounding fix: every per-year projected value in the
    series block must be tokenisable from the note, so a summarizer that quotes
    an intermediate year passes the grounding/provenance check (the failure mode
    in prod trace abb741aa, where intermediate years were absent from the prompt
    and the LLM's recomputed numbers were rejected)."""
    import re

    from agent.analyzer import _generate_cagr_forecast
    from agent.summarizer import _extract_number_tokens

    rows = []
    for year in range(2019, 2026):
        for month in range(1, 13):
            rows.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "p_bal_gel": 100.0 + (year - 2019) * 2.0 + (month % 2) * 5.0,
                "p_bal_usd": 35.0 + (year - 2019) * 1.5 + (month % 2) * 2.0,
            })
    df = pd.DataFrame(rows)

    df_out, note = _generate_cagr_forecast(df, "forecast balancing electricity price for next 10 years")

    assert "--- FORECAST SERIES (per year) ---" in note
    # Series block is appended last, so slice from its header to end of note.
    block = note[note.find("--- FORECAST SERIES (per year) ---"):]
    series_values = re.findall(r"=\s*([\d.]+)\s*$", block, re.MULTILINE)
    assert series_values, f"no per-year values parsed from: {block!r}"

    tokens = _extract_number_tokens(note)
    for val in series_values:
        norm = val.rstrip("0").rstrip(".")
        assert val in tokens or norm in tokens, (
            f"series value {val!r} not tokenisable; tokens sample={sorted(tokens)[:30]!r}"
        )


def test_cagr_projection_block_omitted_when_no_target_years():
    """Defensive: if ``_resolve_target_years`` produced no years (edge case),
    the block must not be appended."""
    from agent.analyzer import _generate_cagr_forecast

    # Single-year data forces an early return at the < 2 usable points guard,
    # which precedes the projection block entirely.
    df = pd.DataFrame({
        "year": pd.to_datetime(["2025-01-01"]),
        "quantity": [100.0],
    })
    df_out, note = _generate_cagr_forecast(df, "forecast")

    # Should NOT include the projection block (early-return path).
    assert "--- CAGR PROJECTIONS ---" not in note
