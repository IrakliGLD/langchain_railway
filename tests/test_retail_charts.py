"""Retail price charts: one chart, every category, right units.

2026-08-15 production complaints, verbatim: "many charts for categories. one
line per chart. still not all categories. all are lines, should be all lines
on one chart" and "maybe need composition".

The planner emitted four chart_groups for one question, each carrying the same
single metric, so the reader got four near-identical panels instead of one
comparison. Retail charts are therefore built deterministically from the frame
rather than from planner suggestions.
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

_CATEGORIES = (
    "220/380|com|other",
    "220/380|com|small",
    "220/380|hh|cat1",
    "220/380|hh|cat2",
    "220/380|hh|cat3",
    "3.3-6-10|com|other",
    "3.3-6-10|hh|",
    "35-110|com|other",
)


def _retail_frame(suppliers=("telmico", "eps"), categories=_CATEGORIES, months=6):
    rows = []
    for month in range(1, months + 1):
        for si, supplier in enumerate(suppliers):
            for ci, category in enumerate(categories):
                rows.append(
                    {
                        "date": f"2026-{month:02d}-01",
                        "supplier": supplier,
                        "category": category,
                        "category_label": f"Label {category}",
                        "distribution_tariff_gel_kwh": 0.080 + 0.001 * ci,
                        "supply_tariff_gel_kwh": 0.110 + 0.002 * si,
                        "transmission_tariff_gel_kwh": 0.0067,
                        "final_price_net_gel_kwh": 0.190 + 0.001 * ci + 0.002 * si,
                    }
                )
    return pd.DataFrame(rows)


def _ctx_with(df, planned_groups=None):
    ctx = QueryContext(query="როგორია სამომხმარებლო ფასების დინამიკა?")
    ctx.df = df
    ctx.cols = list(df.columns)
    ctx.rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    ctx.plan = {"chart_groups": planned_groups or [], "chart_strategy": "single"}
    return ctx


def test_a_retail_frame_yields_one_trend_chart_not_four():
    """Four planner groups with the same metric became four panels."""
    from agent.chart_pipeline import _resolve_chart_groups

    planned = [
        {"metrics": ["final_price_net_gel_kwh"], "type": "line", "title": f"Panel {i}"}
        for i in range(4)
    ]
    ctx = _ctx_with(_retail_frame(), planned_groups=planned)
    groups = _resolve_chart_groups(ctx, list(ctx.df.select_dtypes("number").columns), None)

    assert len(groups) == 1, f"expected one retail chart, got {len(groups)}: {groups}"
    assert groups[0]["metrics"] == ["final_price_net_gel_kwh"]
    assert groups[0]["type"] == "line"


def test_the_gross_price_is_charted_when_vat_was_requested():
    from agent.chart_pipeline import _resolve_chart_groups

    df = _retail_frame()
    df["vat_gel_kwh"] = df["final_price_net_gel_kwh"] * 0.18
    df["total_gross_gel_kwh"] = df["final_price_net_gel_kwh"] * 1.18
    ctx = _ctx_with(df)
    groups = _resolve_chart_groups(ctx, list(df.select_dtypes("number").columns), None)

    assert groups[0]["metrics"] == ["total_gross_gel_kwh"]


def test_a_single_category_also_gets_a_component_composition():
    """The stack only means something for one company and one category."""
    from agent.chart_pipeline import _resolve_chart_groups

    df = _retail_frame(suppliers=("telmico",), categories=("220/380|hh|cat2",))
    ctx = _ctx_with(df)
    groups = _resolve_chart_groups(ctx, list(df.select_dtypes("number").columns), None)

    assert len(groups) == 2, f"expected trend + composition, got {groups}"
    composition = groups[1]
    assert composition["metrics"] == [
        "transmission_tariff_gel_kwh",
        "distribution_tariff_gel_kwh",
        "supply_tariff_gel_kwh",
    ], "components must stack in tariff order, and all three must be present"
    assert composition["type"] == "stackedbar"


def test_many_categories_get_no_composition_chart():
    """Stacking sixteen different stacks on one axis is not a composition."""
    from agent.chart_pipeline import _resolve_chart_groups

    ctx = _ctx_with(_retail_frame())
    groups = _resolve_chart_groups(ctx, list(ctx.df.select_dtypes("number").columns), None)

    assert all(g.get("type") != "stackedbar" for g in groups)


def test_the_composition_survives_the_force_to_line_rule():
    """Price/tariff dimensions are forced to 'line' so levels are not drawn as
    bars. The three components of ONE price genuinely sum to the total, which
    is a real part-to-whole -- it must be exempt or the composition is lost."""
    from agent.chart_pipeline import _choose_chart_type

    chart_type = _choose_chart_type(
        group={
            "type": "stackedbar",
            "metrics": [
                "transmission_tariff_gel_kwh",
                "distribution_tariff_gel_kwh",
                "supply_tariff_gel_kwh",
            ],
            "_component_composition": True,
        },
        visualization=None,
        has_time=True,
        has_categories=True,
        dimensions={"price_tariff"},
        category_count=1,
    )
    assert chart_type == "stackedbar"


def test_an_ordinary_tariff_chart_is_still_forced_to_line():
    """The exemption must not disarm the rule for everything else."""
    from agent.chart_pipeline import _choose_chart_type

    chart_type = _choose_chart_type(
        group={"type": "bar", "metrics": ["p_bal_gel"]},
        visualization=None,
        has_time=True,
        has_categories=True,
        dimensions={"price_tariff"},
        category_count=4,
    )
    assert chart_type == "line"


def test_per_kwh_columns_are_not_labelled_per_mwh():
    """Every other price in this system is GEL/MWh; these are GEL/kWh, a
    factor of 1000. Labelling the axis wrong invites exactly the scale
    confusion the tool exists to prevent."""
    from agent.chart_pipeline import unit_for_price

    assert unit_for_price(["final_price_net_gel_kwh"]) == "GEL/kWh"
    assert unit_for_price(["distribution_tariff_gel_kwh", "supply_tariff_gel_kwh"]) == "GEL/kWh"
    # Unchanged for the wholesale columns.
    assert unit_for_price(["p_bal_gel"]) == "GEL/MWh"
    assert unit_for_price(["p_bal_usd"]) == "USD/MWh"


def test_a_text_column_is_never_coerced_into_a_column_of_nothing():
    """General rule, not a retail one.

    ``_prepare_chart_source`` decides numeric-vs-categorical from a list of
    name hints, and anything unlisted goes through
    ``pd.to_numeric(errors='coerce')``. For a column of company names that
    yields all-NaN, and the series it identifies vanishes. Whatever the column
    is called, a coercion that erases every value is data loss.
    """
    import agent.chart_pipeline as cp

    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-02-01"],
            "some_unlisted_label": ["alpha", "beta"],
            "final_price_net_gel_kwh": [0.19, 0.20],
        }
    )
    ctx = _ctx_with(df)
    _, _, _, categorical_cols, num_cols = cp._prepare_chart_source(ctx)

    assert "some_unlisted_label" in categorical_cols
    assert "some_unlisted_label" not in num_cols


def test_a_genuinely_numeric_column_is_still_coerced():
    """The safeguard must not turn real numbers into strings."""
    import agent.chart_pipeline as cp

    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-02-01"],
            "some_unlisted_measure": ["1.5", "2.5"],
            "final_price_net_gel_kwh": [0.19, 0.20],
        }
    )
    ctx = _ctx_with(df)
    _, _, _, categorical_cols, num_cols = cp._prepare_chart_source(ctx)

    assert "some_unlisted_measure" in num_cols
    assert "some_unlisted_measure" not in categorical_cols


def test_every_category_reaches_the_chart_data():
    """The readability cap trims METRICS; it must never silently drop the
    sixteen supplier/category series the question is about."""
    from agent.chart_pipeline import build_chart
    from context import COLUMN_LABELS

    ctx = _ctx_with(_retail_frame())
    ctx = build_chart(ctx)

    assert ctx.chart_data, "no chart produced"
    # Chart rows carry display labels, not raw column names.
    supplier_key = COLUMN_LABELS["supplier"]
    category_key = COLUMN_LABELS["category"]
    charted = {(row.get(supplier_key), row.get(category_key)) for row in ctx.chart_data}
    assert len(charted) == 16, (
        f"expected 16 series in chart data, got {len(charted)}; "
        f"row keys={sorted(ctx.chart_data[0])}"
    )
