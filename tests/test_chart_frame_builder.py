"""Phase 13 (§16.4.2) tests for the long-form ChartFrame path.

Covers:
* ``classify_series`` role/transform inference (explicit hints + heuristic).
* ``from_wide`` dtype invariants (no strings in ``period``).
* ``build_chart_frame_long`` wide↔long round-trip.
* Season-grain long-form preserves the datetime invariant from Phase 12.
"""
import os

# Ensure config validation passes before importing project modules.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd
import pytest

from agent.chart_frame_builder import (  # noqa: E402
    ChartFrame,
    LONG_FRAME_COLUMNS,
    build_chart_frame,
    build_chart_frame_long,
    classify_series,
    from_wide,
    to_wide,
)
from contracts.question_analysis import MeasureTransform, SemanticRole  # noqa: E402


# ---------------------------------------------------------------------------
# classify_series
# ---------------------------------------------------------------------------


def test_classify_series_defaults_to_observed_raw():
    role, transform = classify_series("p_bal_gel")
    assert role == SemanticRole.OBSERVED
    assert transform == MeasureTransform.RAW


def test_classify_series_respects_explicit_hints():
    role, transform = classify_series(
        "p_bal_gel",
        role_hint=SemanticRole.REFERENCE,
        transform_hint=MeasureTransform.YOY_PCT,
    )
    assert role == SemanticRole.REFERENCE
    assert transform == MeasureTransform.YOY_PCT


def test_classify_series_derived_flag_bumps_role_when_no_hint():
    role, _ = classify_series("p_bal_gel_yoy", derived=True)
    assert role == SemanticRole.DERIVED


@pytest.mark.parametrize(
    "name,expected",
    [
        ("p_bal_gel_mom_pct", MeasureTransform.MOM_PCT),
        ("price_yoy_delta", MeasureTransform.YOY_DELTA),
        ("share_renewable", MeasureTransform.SHARE_OF_TOTAL),
        ("price_index_100", MeasureTransform.INDEX_100),
        ("cagr_balancing", MeasureTransform.CAGR),
    ],
)
def test_classify_series_heuristic_picks_up_suffix(name, expected):
    _, transform = classify_series(name)
    assert transform == expected


# ---------------------------------------------------------------------------
# from_wide dtype + shape invariants
# ---------------------------------------------------------------------------


def test_from_wide_long_form_has_stable_dtypes_and_columns():
    wide = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "p_bal_gel": [100.0, 110.0],
        }
    )
    frame = from_wide(
        wide,
        time_key="month",
        num_cols=["p_bal_gel"],
        dim_map={"p_bal_gel": "price_balancing"},
        label_map={"p_bal_gel": "Balancing price (GEL/MWh)"},
        measure_transform="raw",
    )
    assert isinstance(frame, ChartFrame)
    # Column order and names are stable.
    assert list(frame.long_df.columns) == list(LONG_FRAME_COLUMNS)
    # ``period`` must be datetime-typed (no strings), per Phase 12 invariant.
    assert pd.api.types.is_datetime64_any_dtype(frame.long_df["period"])
    # Value must be numeric.
    assert pd.api.types.is_float_dtype(frame.long_df["value"])
    # Role/transform populated.
    assert set(frame.long_df["role"].unique()) == {"observed"}
    assert set(frame.long_df["transform"].unique()) == {"raw"}
    # Series label comes from label_map, source_metric preserves the raw name.
    assert frame.long_df["series"].iloc[0] == "Balancing price (GEL/MWh)"
    assert frame.long_df["source_metric"].iloc[0] == "p_bal_gel"


def test_from_wide_empty_returns_empty_frame_with_typed_period():
    empty = from_wide(
        pd.DataFrame(),
        time_key="month",
        num_cols=["p_bal_gel"],
    )
    assert empty.is_empty()
    # Even empty, period must be datetime-typed so downstream dtype asserts hold.
    assert pd.api.types.is_datetime64_any_dtype(empty.long_df["period"])


def test_from_wide_without_time_key_fills_period_with_nat():
    wide = pd.DataFrame({"p_bal_gel": [50.0, 60.0]})
    frame = from_wide(
        wide,
        time_key=None,
        num_cols=["p_bal_gel"],
    )
    assert not frame.is_empty()
    assert frame.long_df["period"].isna().all()
    # Still datetime-typed.
    assert pd.api.types.is_datetime64_any_dtype(frame.long_df["period"])


def test_from_wide_marks_derived_columns():
    wide = pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "p_bal_gel": [100.0, 110.0],
            "p_bal_gel_yoy": [5.0, 8.0],
        }
    )
    frame = from_wide(
        wide,
        time_key="month",
        num_cols=["p_bal_gel", "p_bal_gel_yoy"],
        transform_hints={"p_bal_gel_yoy": MeasureTransform.YOY_PCT},
        derived_cols={"p_bal_gel_yoy"},
    )
    yoy_rows = frame.long_df[frame.long_df["source_metric"] == "p_bal_gel_yoy"]
    assert (yoy_rows["role"] == "derived").all()
    assert (yoy_rows["transform"] == "yoy_pct").all()
    assert yoy_rows["is_derived"].all()
    obs_rows = frame.long_df[frame.long_df["source_metric"] == "p_bal_gel"]
    assert (obs_rows["role"] == "observed").all()
    assert (obs_rows["is_derived"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# build_chart_frame_long wide↔long round-trip
# ---------------------------------------------------------------------------


def test_build_chart_frame_long_matches_wide_builder():
    raw = pd.DataFrame(
        {
            "month": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01"]
            ),
            "p_bal_gel": [100.0, 110.0, 120.0],
        }
    )
    wide_df, wide_meta = build_chart_frame(
        raw,
        time_key="month",
        category_cols=[],
        num_cols=["p_bal_gel"],
        dim_map={"p_bal_gel": "price_balancing"},
        time_grain="month",
        measure_transform="raw",
    )
    long_frame = build_chart_frame_long(
        raw,
        time_key="month",
        category_cols=[],
        num_cols=["p_bal_gel"],
        dim_map={"p_bal_gel": "price_balancing"},
        time_grain="month",
        measure_transform="raw",
    )
    # The long frame carries the same meta as the wide builder so downstream
    # ``chart_meta`` assembly is identical.
    assert long_frame.meta == wide_meta

    # Round-trip: long → wide should recover the wide builder's values.
    recovered = to_wide(long_frame, time_key="month")
    # Same row count and metric column.
    assert len(recovered) == len(wide_df)
    assert "p_bal_gel" in recovered.columns
    pd.testing.assert_series_equal(
        recovered.sort_values("month").reset_index(drop=True)["p_bal_gel"],
        wide_df.sort_values("month").reset_index(drop=True)["p_bal_gel"],
        check_names=False,
    )


def test_build_chart_frame_long_season_grain_keeps_period_datetime():
    raw = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2023-06-01", "2023-07-01", "2023-12-01",
                    "2024-06-01", "2024-12-01",
                ]
            ),
            "p_bal_gel": [100.0, 105.0, 120.0, 110.0, 130.0],
        }
    )
    frame = build_chart_frame_long(
        raw,
        time_key="month",
        category_cols=[],
        num_cols=["p_bal_gel"],
        dim_map={"p_bal_gel": "price_balancing"},
        time_grain="season",
        measure_transform="avg",
    )
    # Phase 12 + Phase 13 joint invariant: season buckets are datetimes, not
    # strings, in the long frame.
    assert pd.api.types.is_datetime64_any_dtype(frame.long_df["period"])
    assert not frame.long_df["period"].isna().all()
    # Season labels travel in meta so the renderer can format them.
    assert "seasonLabels" in frame.meta
    assert frame.meta["seasonLabels"]
