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


def test_each_component_share_of_the_final_price_is_precomputed():
    """"Distribution is 37% of the bill" is the natural thing to say about a
    stack, and the model says it whether or not the number exists in the
    corpus. On 2026-08-15 the same six tokens (37, 48, 74, 86, 111, 124) were
    rejected by strict-numeric grounding on two consecutive runs, gutting a
    2,571-character answer down to 537.
    """
    from agent.analyzer import _append_column_aggregates

    rows = []
    for month in range(1, 13):
        rows.append(
            {
                "date": pd.Timestamp(f"2026-{month:02d}-01"),
                "supplier": "telmico",
                "category": "220/380|hh|cat2",
                "transmission_tariff_gel_kwh": 0.010,
                "distribution_tariff_gel_kwh": 0.040,
                "supply_tariff_gel_kwh": 0.050,
                "final_price_net_gel_kwh": 0.100,
            }
        )
    df = pd.DataFrame(rows)
    ctx = QueryContext(query="t", df=df, cols=list(df.columns))
    _append_column_aggregates(ctx)

    hint = ctx.stats_hint
    assert "share of final price" in hint, hint[:400]
    # 0.040 / 0.100 = 40%, 0.050 / 0.100 = 50%, 0.010 / 0.100 = 10%
    assert "distribution=40.0%" in hint, hint[:400]
    assert "supply=50.0%" in hint, hint[:400]
    assert "transmission=10.0%" in hint, hint[:400]


def test_a_frame_without_a_component_stack_gets_no_share_line():
    from agent.analyzer import _append_column_aggregates

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "entity": ["a", "b"],
            "p_bal_gel": [100.0, 120.0],
        }
    )
    ctx = QueryContext(query="t", df=df, cols=list(df.columns))
    _append_column_aggregates(ctx)

    assert "share of final price" not in ctx.stats_hint


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


def test_regulated_tariffs_get_no_seasonal_statistics():
    """GNERC sets one tariff per regulatory period and it holds for every
    month in it. A summer-versus-winter split of that series measures the
    revision calendar, not seasonality -- and 20 such metrics in stats_hint
    is an invitation to build an answer on an artefact.
    """
    from agent.analyzer import _is_administered_price_frame, detect_monthly_timeseries

    rows = []
    for month in range(1, 13):
        rows.append(
            {
                "date": pd.Timestamp(f"2026-{month:02d}-01"),
                "supplier": "telmico",
                "category": "220/380|hh|cat2",
                "transmission_tariff_gel_kwh": 0.0067,
                # Vary one component so the detector genuinely engages --
                # otherwise this test would pass on a constant series for the
                # wrong reason and prove nothing.
                "distribution_tariff_gel_kwh": 0.0812 + 0.0001 * month,
                "supply_tariff_gel_kwh": 0.1104,
                "final_price_net_gel_kwh": 0.1983,
            }
        )
    df = pd.DataFrame(rows)

    # The detector WOULD fire on this frame; the guard is what stops it.
    assert detect_monthly_timeseries(df) is not None
    assert _is_administered_price_frame(df) is True


def test_market_prices_keep_their_seasonality():
    """The suppression must be specific: balancing prices ARE seasonal, and
    that asymmetry is the point of a retail-versus-wholesale comparison."""
    from agent.analyzer import _is_administered_price_frame

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-07-01"]),
            "p_bal_gel": [210.0, 160.0],
        }
    )
    assert _is_administered_price_frame(df) is False


def test_the_guidance_explains_why_there_is_no_season():
    from skills.loader import load_reference

    rules = load_reference("energy-analyst", "retail-tariff-rules.md").lower()
    assert "administered price" in rules
    assert "gnerc" in rules
    assert "step change" in rules
    # And that the wholesale side differs, which is what makes a comparison
    # worth writing.
    assert "balancing price is the opposite" in rules
