"""Retail answers must explain the stack and quote a growth rate, not derive one.

2026-08-15, from the domain owner: "should also explain that the end user
tariff consists of" and "not clear how cagr was calculated, based on which
categories?".

Both are addressed deterministically. The stack explanation rides in GUIDANCE,
which no truncation profile sheds, rather than in retrieved passages, which the
DATA profile drops first -- precisely on the questions where per-series
statistics grow. The growth rate is computed per series into stats_hint so the
model quotes a grounded number instead of computing one the grounding gate
then strips.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd  # noqa: E402

from models import QueryContext  # noqa: E402


def _frame(months=25, start=0.19, end=0.25):
    """Two series over a >1-year span so a CAGR is defined."""
    rows = []
    for i in range(months):
        for supplier, offset in (("telmico", 0.0), ("eps", 0.02)):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.DateOffset(months=i),
                    "supplier": supplier,
                    "category": "220/380|hh|cat2",
                    "final_price_net_gel_kwh": start
                    + (end - start) * i / (months - 1)
                    + offset,
                }
            )
    return pd.DataFrame(rows)


def test_growth_is_precomputed_per_series():
    from agent.analyzer import _append_column_aggregates

    df = _frame()
    ctx = QueryContext(query="t", df=df, cols=list(df.columns))
    _append_column_aggregates(ctx)

    hint = ctx.stats_hint
    assert "change_first_to_last=" in hint, hint[:400]
    assert "cagr=" in hint, "a >1-year span must carry an annualised rate:\n" + hint[:400]
    # One growth figure per series, not one for the pooled frame.
    assert hint.count("change_first_to_last=") == 2, hint


def test_a_short_series_gets_no_annualised_rate():
    """Annualising four months of data invents precision."""
    from agent.analyzer import _append_column_aggregates

    df = _frame(months=4)
    ctx = QueryContext(query="t", df=df, cols=list(df.columns))
    _append_column_aggregates(ctx)

    assert "change_first_to_last=" in ctx.stats_hint
    assert "cagr=" not in ctx.stats_hint


def test_a_zero_base_reports_no_growth_rate():
    """Percent change off zero is undefined, not infinite."""
    from agent.analyzer import _series_growth, _time_column

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2025-01-01"]),
            "final_price_net_gel_kwh": [0.0, 0.25],
        }
    )
    assert _series_growth(df, "final_price_net_gel_kwh", _time_column(df)) == ""


def test_growth_is_measured_in_time_order_not_row_order():
    """Rows arrive sorted by date DESC when no date filter was given."""
    from agent.analyzer import _series_growth, _time_column

    ascending = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2025-01-01"]),
            "final_price_net_gel_kwh": [0.20, 0.25],
        }
    )
    descending = ascending.iloc[::-1].reset_index(drop=True)

    forward = _series_growth(ascending, "final_price_net_gel_kwh", _time_column(ascending))
    reverse = _series_growth(descending, "final_price_net_gel_kwh", _time_column(descending))
    assert forward == reverse, "row order must not change the measured direction"
    assert "+25.0%" in forward


def test_the_retail_rules_reference_exists_and_states_the_stack():
    from skills.loader import load_reference

    rules = load_reference("energy-analyst", "retail-tariff-rules.md")

    assert rules, "retail-tariff-rules.md is missing"
    lowered = rules.lower()
    for required in ("transmission", "distribution", "supply", "guaranteed capacity"):
        assert required in lowered, f"the stack explanation omits {required!r}"
    assert "never average" in lowered or "do not average" in lowered
    assert "gel/kwh" in lowered and "gel/mwh" in lowered, "the unit contrast must be stated"


def test_the_retail_rules_are_registered_for_startup_validation():
    """An unregistered reference file fails silently at runtime, returning ''."""
    from skills.loader import _EXPECTED_FILES

    assert ("energy-analyst", "retail-tariff-rules.md") in _EXPECTED_FILES
