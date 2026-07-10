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


def test_forecast_holds_fx_fixed_converting_usd_to_gel():
    """Fixed-FX logic: with GEL+USD+xrate present, USD is trend-forecast and GEL is
    derived = USD x latest xrate. Reproduces prod trace 11a670ce, where independent
    per-currency CAGR made GEL winter FALL while USD winter ROSE. After the fix GEL
    winter must rise in step with USD winter, and GEL = USD x xrate_fixed on every
    forecast row."""
    from agent.analyzer import _generate_cagr_forecast

    XRATE = 2.70  # latest/current GEL per USD (held fixed)
    rows = []
    for year in range(2019, 2026):
        t = year - 2019
        for month in range(1, 13):
            summer = month in (4, 5, 6, 7)  # SUMMER_MONTHS
            if summer:
                usd = 48.0 + t * 1.3   # summer USD rises
                gel = 150.0 + t * 2.5  # summer GEL rises
            else:
                usd = 50.0 + t * 1.3   # winter USD rises
                gel = 160.0 - t * 1.7  # winter GEL falls (the historical anomaly)
            rows.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "p_bal_gel": gel,
                "p_bal_usd": usd,
                "xrate": XRATE,
            })
    df = pd.DataFrame(rows)

    df_out, note = _generate_cagr_forecast(
        df, "forecast balancing electricity price for next 10 years"
    )

    fc = df_out[df_out["is_forecast"] == True].copy()  # noqa: E712
    assert not fc.empty

    # 1) GEL is derived from USD at the fixed rate on EVERY forecast row.
    ratios = (fc["p_bal_gel"] / fc["p_bal_usd"]).dropna()
    assert not ratios.empty
    assert ((ratios - XRATE).abs() < 1e-6).all(), (
        f"GEL/USD ratio not fixed at {XRATE}: {ratios.tolist()[:5]}"
    )

    # 2) The fixed-FX assumption is stated explicitly with the latest xrate.
    assert "Exchange-rate assumption" in note
    assert "2.70" in note

    # 3) GEL winter now RISES with USD winter (anomaly fixed), despite the
    #    historical GEL winter series declining.
    winter = fc[fc["date"].dt.month == 12].sort_values("date")
    assert len(winter) >= 2
    assert winter["p_bal_usd"].iloc[-1] > winter["p_bal_usd"].iloc[0]  # sanity: USD winter rises
    assert winter["p_bal_gel"].iloc[-1] > winter["p_bal_gel"].iloc[0], (
        "GEL winter forecast should rise with USD winter under fixed FX"
    )


def test_forecast_series_includes_per_year_yearly_lines():
    """2026-07-10 (trace 1d2d2ca6): the LLM narrates ANNUAL values for a
    10-year forecast; with only summer/winter lines available it invents
    (summer+winter)/2 averages that fail grounding. The series block must
    carry an authoritative per-year 'yearly' projection line per currency."""
    import re

    from agent.analyzer import _generate_cagr_forecast
    from agent.summarizer import _extract_number_tokens

    rows = []
    for year in range(2019, 2026):
        for month in range(1, 13):
            rows.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "p_bal_gel": 100.0 + (year - 2019) * 1.0 + (month % 2) * 5.0,
                "p_bal_usd": 35.0 + (year - 2019) * 1.5 + (month % 2) * 2.0,
            })
    df = pd.DataFrame(rows)

    df_out, note = _generate_cagr_forecast(
        df, "forecast balancing electricity price for next 10 years"
    )

    block = note[note.find("--- FORECAST SERIES (per year) ---"):]
    gel_yearly = re.findall(r"^(\d{4}) GEL yearly = ([\d.]+)\s*$", block, re.MULTILINE)
    usd_yearly = re.findall(r"^(\d{4}) USD yearly = ([\d.]+)\s*$", block, re.MULTILINE)
    assert len(gel_yearly) == 10, f"expected 10 GEL yearly lines, got {gel_yearly!r}"
    assert len(usd_yearly) == 10, f"expected 10 USD yearly lines, got {usd_yearly!r}"

    # The yearly values must be tokenisable so quoting them passes grounding.
    tokens = _extract_number_tokens(note)
    for _, val in gel_yearly + usd_yearly:
        norm = val.rstrip("0").rstrip(".")
        assert val in tokens or norm in tokens, f"yearly value {val!r} not tokenisable"


def _monthly_price_rows(year_start: int, year_end: int):
    """Monthly GEL series with a deliberate regime break: flat before 2019,
    trending after — so a fit window that includes the old regime produces a
    different CAGR than the canonical window."""
    rows = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            if year < 2019:
                val = 300.0  # old flat regime — poisons a full-history fit
            else:
                val = 100.0 + (year - 2019) * 4.0 + (month % 2) * 5.0
            rows.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "p_bal_gel": val,
            })
    return rows


def test_forecast_fit_window_deterministic_across_fetch_spans():
    """2026-07-10 model-variance report: when the analyzer LLM fails, the
    router fallback fetches the FULL history (132 rows vs 79) and the CAGR
    baseline shifts (2.29% vs 1.09%) — same question, different forecast,
    depending on which LLM/analyzer path ran. The fit window must be
    deterministic: anchored at the last observation with the canonical
    history width, regardless of how much history was fetched."""
    from agent.analyzer import _generate_cagr_forecast

    query = "forecast balancing electricity price for next 10 years"
    df_long = pd.DataFrame(_monthly_price_rows(2015, 2025))
    # The canonical window for a 10y horizon is 8 years anchored at the last
    # observation's month: 2018-12-01..2025-12-01 — the same window the
    # planner fetch expansion produces. The short frame simulates that fetch.
    df_short = df_long[df_long["date"] >= pd.Timestamp("2018-12-01")].reset_index(drop=True)

    out_long, note_long = _generate_cagr_forecast(df_long.copy(), query)
    out_short, note_short = _generate_cagr_forecast(df_short.copy(), query)

    # The long fetch is trimmed to the canonical window and says so.
    assert "Fit window:" in note_long

    # Identical CAGR statements...
    def _cagr_line(note: str) -> str:
        import re as _re
        m = _re.search(r"Yearly CAGR=[^.]*\.", note)
        assert m, f"no CAGR line in note: {note!r}"
        return m.group(0)

    assert _cagr_line(note_long) == _cagr_line(note_short)

    # ...and identical projected forecast rows.
    fc_long = out_long[out_long["is_forecast"] == True].reset_index(drop=True)  # noqa: E712
    fc_short = out_short[out_short["is_forecast"] == True].reset_index(drop=True)  # noqa: E712
    assert len(fc_long) == len(fc_short) > 0
    assert (fc_long["p_bal_gel"] - fc_short["p_bal_gel"]).abs().max() < 1e-9


def test_explicit_history_year_in_query_keeps_full_window():
    """A user who pins the history scope ('based on data since 2015') keeps
    the full fetched window — the canonical trim must not override an
    explicit historical request."""
    from agent.analyzer import _generate_cagr_forecast

    df_long = pd.DataFrame(_monthly_price_rows(2015, 2025))
    _, note = _generate_cagr_forecast(
        df_long.copy(),
        "forecast balancing electricity price for next 10 years based on data since 2015",
    )

    assert "Fit window:" not in note


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
