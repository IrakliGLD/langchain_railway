"""Column aggregates must not pool across the series a frame is split into.

The 2026-08-15 production complaint: a retail-price answer quoted "one
indicator when there are several categories and two distribution/supply
companies". Averaging a price across Telmico and EPS, or across household
cat 1 and commercial 35-110 kV, produces a number that is not any tariff
anyone pays -- and ``stats_hint`` feeds the grounding corpus, so the model
quotes it as if it were a level.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd  # noqa: E402


def _ctx(df):
    from models import QueryContext

    return QueryContext(query="test", df=df, cols=list(df.columns))


def _end_user_frame(suppliers=("telmico", "eps"), categories=("hh|cat1", "com|other")):
    """A frame shaped like get_end_user_prices output."""
    rows = []
    for month in range(1, 7):
        for supplier_index, supplier in enumerate(suppliers):
            for category_index, category in enumerate(categories):
                rows.append(
                    {
                        "date": pd.Timestamp(f"2026-{month:02d}-01"),
                        "supplier": supplier,
                        "category": category,
                        # Well separated per series so a pooled mean lands
                        # between them and matches none of them.
                        "final_price_net": 0.10
                        + 0.10 * supplier_index
                        + 0.40 * category_index,
                    }
                )
    return pd.DataFrame(rows)


def test_aggregates_are_reported_per_series_not_pooled():
    from agent.analyzer import _append_column_aggregates

    ctx = _ctx(_end_user_frame())
    _append_column_aggregates(ctx)

    hint = ctx.stats_hint

    # The pooled mean of the four series is 0.35, which is not any series'
    # value. Shipping it invites the answer to quote a price nobody pays.
    assert "mean=0.3500" not in hint, (
        "aggregates pooled across supplier and category:\n" + hint
    )

    # Each series must be identifiable and carry its own figures.
    for supplier in ("telmico", "eps"):
        assert supplier in hint, f"{supplier} missing from aggregates:\n{hint}"
    for value in ("0.1000", "0.2000", "0.5000", "0.6000"):
        assert value in hint, f"series level {value} missing from aggregates:\n{hint}"


def test_ungrouped_frames_keep_the_existing_pooled_behaviour():
    """A single-series frame has nothing to mix, so nothing should change."""
    from agent.analyzer import _append_column_aggregates

    df = pd.DataFrame(
        {
            "date": pd.to_datetime([f"2024-{m:02d}-01" for m in range(1, 13)]),
            "p_bal_gel": [float(95 + m) for m in range(12)],
        }
    )
    ctx = _ctx(df)
    _append_column_aggregates(ctx)

    assert "Balancing electricity price" in ctx.stats_hint
    assert "mean=100.5000" in ctx.stats_hint


def test_the_whole_retail_fleet_is_enumerated_not_refused():
    """Two supply companies x eight end-user categories = 16 series.

    The domain owner asked for every category to be shown, so the enumeration
    cap must clear the full fleet: a broad tariff question gets per-category
    figures, not a note explaining why it cannot answer.
    """
    from agent.analyzer import _append_column_aggregates

    categories = (
        "220/380|com|other",
        "220/380|com|small",
        "220/380|hh|cat1",
        "220/380|hh|cat2",
        "220/380|hh|cat3",
        "3.3-6-10|com|other",
        "3.3-6-10|hh|",
        "35-110|com|other",
    )
    ctx = _ctx(_end_user_frame(categories=categories))
    _append_column_aggregates(ctx)

    hint = ctx.stats_hint
    assert "16 distinct series" in hint, hint[:400]
    for category in categories:
        assert category in hint, f"category {category} not enumerated:\n{hint[:600]}"
    assert "no pooled average is reported" not in hint, (
        "the full retail fleet must be enumerated, not refused:\n" + hint[:400]
    )


def test_high_cardinality_frames_refuse_to_pool_rather_than_enumerate():
    """Too many series to list is still not a licence to average them."""
    from agent.analyzer import _append_column_aggregates

    rows = [
        {"date": pd.Timestamp("2026-01-01"), "entity": f"plant_{i}", "tariff_gel": 10.0 + i}
        for i in range(40)
    ]
    ctx = _ctx(pd.DataFrame(rows))
    _append_column_aggregates(ctx)

    hint = ctx.stats_hint
    # Pooled mean across 40 plants is 29.5 -- must not be presented as a level.
    assert "mean=29.5000" not in hint, "pooled a 40-series frame:\n" + hint
    assert "series" in hint.lower(), (
        "expected an explicit note that the frame spans multiple series:\n" + hint
    )
