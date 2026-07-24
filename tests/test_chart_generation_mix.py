"""2026-07-09 chart fixes: 'period' time-key recognition (Bug A) and
generation-mix composition restricted to generation techs (Bug B)."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd

from agent.chart_pipeline import (
    _apply_composition_series_budget,
    _limit_series,
    _prepare_chart_source,
    build_chart,
)
from agent.derived_chart_builder import _build_decomposition_spec, _resolve_time_key
from models import QueryContext

# ---------------------------------------------------------------------------
# Bug A — "period" recognised as the time axis (not clobbered to nanoseconds)
# ---------------------------------------------------------------------------

def test_period_recognised_as_time_key_and_not_numeric():
    df = pd.DataFrame({
        "period": pd.to_datetime(["2015-01-01", "2015-02-01", "2015-03-01"]),
        "quantity_hydro": [800.0, 810.0, 790.0],
        "quantity_thermal": [200.0, 210.0, 190.0],
    })
    ctx = QueryContext(query="generation mix")
    ctx.df = df
    ctx.cols = list(df.columns)
    prepared, time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert time_key == "period"
    assert "period" not in num_cols  # never plotted as a data series
    # datetime preserved — NOT coerced to int64 nanosecond epoch.
    assert pd.api.types.is_datetime64_any_dtype(prepared["period"])


def test_resolve_time_key_finds_period():
    df = pd.DataFrame({"period": [1], "quantity_hydro": [1.0]})
    assert _resolve_time_key(df) == "period"


# ---------------------------------------------------------------------------
# Bug B — generation-mix decomposition keeps only generation techs
# ---------------------------------------------------------------------------

def _gen_mix_frame():
    return pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "share_hydro": [0.35, 0.40],
        "share_thermal": [0.10, 0.12],
        "share_wind": [0.02, 0.03],
        "share_export": [0.05, 0.06],   # demand-side — must be excluded
        "share_losses": [0.08, 0.07],   # demand-side — must be excluded
    })


def test_decomposition_excludes_demand_shares():
    df = _gen_mix_frame()
    num_cols = [c for c in df.columns if c != "period"]
    spec = _build_decomposition_spec(df, "period", num_cols, {c: c for c in df.columns})
    categories = {r["category"] for r in spec[0]["data"]}
    assert categories == {"share_hydro", "share_thermal", "share_wind"}
    # dates are formatted YYYY-MM, not nanoseconds
    assert spec[0]["data"][0]["date"] == "2024-01"


def test_non_generation_decomposition_is_untouched():
    # Balancing composition has no demand/transit share columns → unchanged.
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01"]),
        "share_import": [0.3],
        "share_regulated_hpp": [0.7],
    })
    spec = _build_decomposition_spec(
        df, "period", ["share_import", "share_regulated_hpp"],
        {c: c for c in df.columns},
    )
    categories = {r["category"] for r in spec[0]["data"]}
    assert categories == {"share_import", "share_regulated_hpp"}


def test_generation_aggregate_shares_are_kept():
    """Regression: the exclusion approach must NOT drop generation aggregates
    such as share_renewable — only demand/transit shares are removed."""
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "share_renewable": [0.30, 0.32],
        "share_thermal": [0.50, 0.48],
        "share_hydro": [0.20, 0.20],
        "share_export": [0.05, 0.06],   # demand — excluded
    })
    num_cols = [c for c in df.columns if c != "period"]
    spec = _build_decomposition_spec(df, "period", num_cols, {c: c for c in df.columns})
    categories = {r["category"] for r in spec[0]["data"]}
    assert categories == {"share_renewable", "share_thermal", "share_hydro"}


# ---------------------------------------------------------------------------
# Bug C - generic chart path must also respect supply/demand side intent
# ---------------------------------------------------------------------------

def _wide_supply_demand_frame():
    return pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "quantity_hydro": [300.0, 320.0, 340.0],
        "quantity_thermal": [180.0, 170.0, 160.0],
        "quantity_import": [90.0, 80.0, 70.0],
        "quantity_abkhazeti": [55.0, 56.0, 57.0],
        "quantity_direct customers": [120.0, 122.0, 124.0],
        "quantity_export": [40.0, 42.0, 44.0],
        "total_supply": [570.0, 570.0, 570.0],
        "total_demand": [215.0, 220.0, 225.0],
    })


def test_generation_mix_prepare_filters_to_generation_columns():
    """2026-07-09 follow-up: a generation-mix question narrows past the supply
    side to GENERATION techs — import/self-cons/total_supply are not part of
    the generation mix."""
    df = _wide_supply_demand_frame()
    ctx = QueryContext(query="what can you say about Georgian generation mix?")
    ctx.df = df
    ctx.cols = list(df.columns)

    _prepared, _time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert "quantity_hydro" in num_cols
    assert "quantity_thermal" in num_cols
    assert "quantity_import" not in num_cols
    assert "total_supply" not in num_cols
    assert "quantity_abkhazeti" not in num_cols
    assert "quantity_direct customers" not in num_cols
    assert "quantity_export" not in num_cols
    assert "total_demand" not in num_cols


def test_supply_scope_query_keeps_import_and_total_supply():
    """Supply-wide wording (import dependence, electricity supply) keeps the
    broad supply scope including import and total_supply."""
    df = _wide_supply_demand_frame()
    ctx = QueryContext(query="how dependent is Georgian electricity supply on import?")
    ctx.df = df
    ctx.cols = list(df.columns)

    _prepared, _time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert "quantity_hydro" in num_cols
    assert "quantity_import" in num_cols
    assert "total_supply" in num_cols
    assert "total_demand" not in num_cols


def test_demand_prepare_filters_supply_quantity_columns():
    df = _wide_supply_demand_frame()
    ctx = QueryContext(query="what can you say about electricity demand?")
    ctx.df = df
    ctx.cols = list(df.columns)

    _prepared, _time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert "quantity_abkhazeti" in num_cols
    assert "quantity_direct customers" in num_cols
    assert "quantity_export" in num_cols
    assert "total_demand" in num_cols
    assert "quantity_hydro" not in num_cols
    assert "quantity_thermal" not in num_cols
    assert "quantity_import" not in num_cols
    assert "total_supply" not in num_cols


def test_mixed_supply_and_demand_prepare_keeps_both_sides():
    df = _wide_supply_demand_frame()
    ctx = QueryContext(query="compare electricity supply and demand")
    ctx.df = df
    ctx.cols = list(df.columns)

    _prepared, _time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert "quantity_hydro" in num_cols
    assert "quantity_abkhazeti" in num_cols
    assert "total_supply" in num_cols
    assert "total_demand" in num_cols


def test_supply_adequacy_question_keeps_both_sides():
    """A yes/no adequacy question compares supply with demand; it is not a
    request for the supply components used to cover demand."""
    df = _wide_supply_demand_frame()
    ctx = QueryContext(query="does electricity supply meet demand?")
    ctx.df = df
    ctx.cols = list(df.columns)

    _prepared, _time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert "total_supply" in num_cols
    assert "total_demand" in num_cols


def test_generation_mix_prepare_filters_long_type_tech_rows():
    df = pd.DataFrame({
        "period": pd.to_datetime([
            "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01",
        ]),
        "type_tech": ["hydro", "thermal", "export", "direct customers"],
        "quantity_tech": [300.0, 180.0, 40.0, 120.0],
    })
    ctx = QueryContext(query="Georgian generation mix")
    ctx.df = df
    ctx.cols = list(df.columns)

    prepared, _time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert set(prepared["type_tech"]) == {"hydro", "thermal"}
    assert num_cols == ["quantity_tech"]


def test_generation_mix_rendered_chart_excludes_demand_series():
    df = _wide_supply_demand_frame()
    ctx = QueryContext(query="chart Georgian generation mix")
    ctx.df = df
    ctx.cols = list(df.columns)
    ctx.rows = list(df.itertuples(index=False, name=None))

    out = build_chart(ctx)

    assert out.chart_meta is not None
    source_metrics = set(out.chart_meta["sourceMetrics"])
    assert source_metrics <= {
        "quantity_hydro",
        "quantity_thermal",
    }
    assert not {
        "quantity_import",
        "total_supply",
        "quantity_abkhazeti",
        "quantity_direct customers",
        "quantity_export",
        "total_demand",
    } & source_metrics


# ---------------------------------------------------------------------------
# Side resolution must read the resolved/effective query, not only ctx.query
# ---------------------------------------------------------------------------

def test_chart_side_uses_effective_query_when_semantically_locked():
    """Intent tokens carried only by the resolved (English/canonical) query —
    e.g. a follow-up or a translated query — must still drive side filtering."""
    df = _wide_supply_demand_frame()
    ctx = QueryContext(query="show it")  # no side tokens in the raw query
    ctx.resolved_query = "Georgian generation mix"
    ctx.semantic_locked = True
    ctx.df = df
    ctx.cols = list(df.columns)

    _prepared, _time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert "quantity_hydro" in num_cols
    assert "quantity_thermal" in num_cols
    assert "quantity_import" not in num_cols
    assert "total_supply" not in num_cols
    assert "quantity_abkhazeti" not in num_cols
    assert "quantity_direct customers" not in num_cols
    assert "total_demand" not in num_cols


# ---------------------------------------------------------------------------
# Over-eager side resolution must not blank the chart
# ---------------------------------------------------------------------------

def test_supply_query_keeps_all_rows_when_every_row_is_demand():
    """If a supply-worded query returns an all-demand frame, the row filter
    must fall back to the full frame instead of emptying it (which would
    suppress the chart in build_chart)."""
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "type_tech": ["export", "direct customers"],
        "quantity_tech": [40.0, 120.0],
    })
    ctx = QueryContext(query="Georgian generation mix")
    ctx.df = df
    ctx.cols = list(df.columns)

    prepared, _time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert set(prepared["type_tech"]) == {"export", "direct customers"}
    assert num_cols == ["quantity_tech"]


def test_supply_query_keeps_metrics_when_every_column_is_demand():
    """Same fallback for the wide path: a supply query over an all-demand
    metric set must keep the metrics rather than return an empty list."""
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "quantity_abkhazeti": [55.0, 56.0],
        "quantity_export": [40.0, 42.0],
        "total_demand": [95.0, 98.0],
    })
    ctx = QueryContext(query="Georgian generation mix")
    ctx.df = df
    ctx.cols = list(df.columns)

    _prepared, _time_key, _labels, _cat, num_cols = _prepare_chart_source(ctx)

    assert set(num_cols) == {"quantity_abkhazeti", "quantity_export", "total_demand"}


# ---------------------------------------------------------------------------
# Bug D (2026-07-09 follow-up) — generation-mix questions must render the
# GENERATION composition: share columns only, generation techs only, stacked
# bars, and no 3-series cap chopping the component set.
# ---------------------------------------------------------------------------

def _canonicalized_mix_frame():
    """Shape of canonicalize_generation_mix_df output for an unfiltered fetch:
    quantity+share per supply tech plus derived aggregates. 12 monthly rows so
    the legacy chart gate (rows >= 10 for open-ended queries) is satisfied,
    matching real generation-mix responses (60+ months)."""
    n = 12
    periods = pd.date_range("2024-01-01", periods=n, freq="MS")
    base = {
        "quantity_hydro": 300.0,
        "quantity_import": 90.0,
        "quantity_self-cons": 10.0,
        "quantity_solar": 5.0,
        "quantity_thermal": 180.0,
        "quantity_wind": 15.0,
        "share_hydro": 0.50,
        "share_import": 0.15,
        "share_self-cons": 0.02,
        "share_solar": 0.01,
        "share_thermal": 0.30,
        "share_wind": 0.02,
        "total_supply": 600.0,
        "total_domestic_generation": 500.0,
        "local_generation": 320.0,
        "import_dependent_supply": 270.0,
        "import_dependency_ratio": 0.45,
    }
    data = {"period": periods}
    for col, start in base.items():
        step = 0.001 if start < 1 else 1.0
        data[col] = [start + i * step for i in range(n)]
    return pd.DataFrame(data)


def _build(ctx_query: str) -> QueryContext:
    df = _canonicalized_mix_frame()
    ctx = QueryContext(query=ctx_query)
    ctx.df = df
    ctx.cols = list(df.columns)
    ctx.rows = list(df.itertuples(index=False, name=None))
    return build_chart(ctx)


def test_generation_mix_chart_is_stacked_generation_shares():
    out = _build("what can you say about the generation mix of Georgia?")

    assert out.chart_meta is not None
    assert out.chart_type == "stackedbar"
    assert sorted(out.chart_meta["sourceMetrics"]) == [
        "share_hydro",
        "share_solar",
        "share_thermal",
        "share_wind",
    ]


def test_supply_mix_chart_keeps_all_supply_shares_stacked():
    out = _build("show the electricity supply mix over time")

    assert out.chart_meta is not None
    assert out.chart_type == "stackedbar"
    # All six supply components survive: composition groups are exempt from
    # the 3-series readability cap.
    assert sorted(out.chart_meta["sourceMetrics"]) == [
        "share_hydro",
        "share_import",
        "share_self-cons",
        "share_solar",
        "share_thermal",
        "share_wind",
    ]


def test_generation_quantities_stay_line_chart():
    out = _build("show electricity generation quantities in MWh by technology over time")

    assert out.chart_meta is not None
    assert out.chart_type == "line"


def test_time_sort_does_not_choose_which_series_survive():
    metrics = ["p_bal_gel", "xrate"]

    selected = _limit_series(
        metrics,
        "show the exchange rate over time",
        1,
        preserve_order=False,
        sort_rule="time_asc",
        top_n=None,
    )

    assert selected == ["xrate"]


def test_composition_budget_preserves_overflow_as_other():
    metrics = [f"quantity_component_{idx}" for idx in range(10)]
    df = pd.DataFrame(
        {
            metric: [float(idx + 1), float((idx + 1) * 2)]
            for idx, metric in enumerate(metrics)
        }
    )
    original_totals = df[metrics].sum(axis=1)

    budgeted_df, budgeted_metrics, budget_meta = _apply_composition_series_budget(
        df,
        metrics,
        max_series=8,
        top_n=None,
    )

    assert len(budgeted_metrics) == 8
    assert budgeted_metrics[-1] == "quantity_other"
    assert len(budget_meta["otherMembers"]) == 3
    pd.testing.assert_series_equal(
        budgeted_df[budgeted_metrics].sum(axis=1),
        original_totals,
        check_names=False,
    )


def test_explicit_top_n_is_the_only_composition_omission():
    metrics = [f"quantity_component_{idx}" for idx in range(6)]
    df = pd.DataFrame(
        {
            metric: [float(idx + 1), float(idx + 1)]
            for idx, metric in enumerate(metrics)
        }
    )

    _budgeted_df, budgeted_metrics, budget_meta = _apply_composition_series_budget(
        df,
        metrics,
        max_series=8,
        top_n=3,
    )

    assert budgeted_metrics == [
        "quantity_component_5",
        "quantity_component_4",
        "quantity_component_3",
    ]
    assert budget_meta == {"topN": 3}


def test_supply_coverage_chart_keeps_every_supply_component():
    """A demand-coverage question asks what supplies demand, not who consumes it.

    A stacked composition must also keep every component even when the
    analyzer's generic readability hint says max_series=3.
    """
    df = _canonicalized_mix_frame()
    demand_components = {
        "quantity_abkhazeti": 90.0,
        "quantity_supply-distribution": 210.0,
        "quantity_direct customers": 120.0,
        "quantity_losses": 35.0,
        "quantity_export": 45.0,
        "share_abkhazeti": 0.18,
        "share_supply-distribution": 0.42,
        "share_direct customers": 0.24,
        "share_losses": 0.07,
        "share_export": 0.09,
        "total_demand": 500.0,
    }
    for col, start in demand_components.items():
        step = 0.001 if start < 1 else 1.0
        df[col] = [start + i * step for i in range(len(df))]

    def enum(value):
        return SimpleNamespace(value=value)
    visualization = SimpleNamespace(
        chart_requested_by_user=False,
        chart_recommended=True,
        primary_presentation=None,
        # Exact production-log mismatch: the pre-data analyzer called this a
        # three-series trend even though the returned frame is a composition.
        preferred_chart_family=enum("line"),
        visual_goal=enum("trend"),
        measure_transform=enum("raw"),
        time_grain=None,
        series_split_mode=enum("single_chart"),
        max_series=3,
        sort_rule=enum("time_asc"),
        top_n=None,
    )
    def build_composition(query):
        ctx = QueryContext(query=query)
        ctx.question_analysis = SimpleNamespace(
            visualization=visualization,
            answer_kind=enum("timeseries"),
        )
        ctx.question_analysis_source = "llm_active"
        ctx.response_mode = "data_primary"
        ctx.used_tool = True
        ctx.tool_name = "get_generation_mix"
        ctx.tool_params = {"mode": "quantity", "types": []}
        ctx.df = df
        ctx.cols = list(df.columns)
        ctx.rows = list(df.itertuples(index=False, name=None))
        return build_chart(ctx)

    out = build_composition("How does Georgia meet its demand for electricity?")
    assert out.chart_type == "stackedbar"
    assert set(out.chart_meta["sourceMetrics"]) == {
        "quantity_hydro",
        "quantity_import",
        "quantity_self-cons",
        "quantity_solar",
        "quantity_thermal",
        "quantity_wind",
    }

    visualization.chart_requested_by_user = True
    generic_chart_out = build_composition(
        "Show a chart of how Georgia meets its demand for electricity"
    )
    assert generic_chart_out.chart_type == "stackedbar"

    explicit_line_out = build_composition(
        "Show as a line chart how Georgia meets its demand for electricity"
    )
    assert explicit_line_out.chart_type == "line"
    visualization.chart_requested_by_user = False

    demand_out = build_composition("Show the composition of electricity demand")
    assert set(demand_out.chart_meta["sourceMetrics"]) == {
        "share_abkhazeti",
        "share_supply-distribution",
        "share_direct customers",
        "share_losses",
        "share_export",
    }
