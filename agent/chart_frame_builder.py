"""Deterministic transforms for chart-specific frames.

Phase 13 (§16.4.2) introduces a long-form ``ChartFrame`` data structure as
an additive, shadow-mode companion to the existing wide builder. The
legacy ``build_chart_frame`` keeps its tuple signature unchanged so
current callers are not disturbed; the long-form path is built on top
via ``build_chart_frame_long`` and is only exposed on the wire when
``ENAI_CHART_LONGFORM`` is enabled (see ``agent/chart_pipeline.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config import SUMMER_MONTHS
from contracts.question_analysis import MeasureTransform, SemanticRole


def build_chart_frame(
    df: pd.DataFrame,
    *,
    time_key: Optional[str],
    category_cols: List[str],
    num_cols: List[str],
    dim_map: Dict[str, str],
    time_grain: Optional[str] = None,
    measure_transform: str = "raw",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return a chart-specific frame plus metadata about applied transforms."""

    if df.empty or not num_cols:
        return df.copy(), {}

    working = df.copy()
    meta: Dict[str, Any] = {}

    if time_key and time_key in working.columns:
        working[time_key] = pd.to_datetime(working[time_key], errors="coerce")

    if time_grain and time_grain != "raw":
        aggregation_mode = measure_transform if measure_transform in {"sum", "avg"} else None
        working, grain_meta = _apply_time_grain(
            working,
            time_key=time_key,
            category_cols=category_cols,
            num_cols=num_cols,
            dim_map=dim_map,
            time_grain=time_grain,
            aggregation_mode=aggregation_mode,
        )
        meta.update(grain_meta)

    if measure_transform in {"sum", "avg"}:
        meta["measureTransform"] = measure_transform
        return working, meta

    if measure_transform and measure_transform != "raw":
        working, transform_meta = _apply_measure_transform(
            working,
            time_key=time_key,
            category_cols=category_cols,
            num_cols=num_cols,
            measure_transform=measure_transform,
        )
        meta.update(transform_meta)

    return working, meta


def _apply_time_grain(
    df: pd.DataFrame,
    *,
    time_key: Optional[str],
    category_cols: List[str],
    num_cols: List[str],
    dim_map: Dict[str, str],
    time_grain: str,
    aggregation_mode: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not time_key or time_key not in df.columns:
        return df, {}

    working = df.copy()
    valid_time = working[time_key].notna()
    if not valid_time.any():
        return df, {}

    bucket = "_chart_bucket"
    season_labels: Dict[str, str] = {}

    if time_grain == "year":
        working[bucket] = working[time_key].dt.to_period("Y").dt.to_timestamp()
    elif time_grain == "quarter":
        working[bucket] = working[time_key].dt.to_period("Q").dt.to_timestamp()
    elif time_grain == "month":
        working[bucket] = working[time_key].dt.to_period("M").dt.to_timestamp()
    elif time_grain == "day":
        working[bucket] = working[time_key].dt.normalize()
    elif time_grain == "season":
        # Phase 12 fix: previously this branch wrote strings like "2023-summer"
        # into the bucket column, which then replaced the datetime `time_key`
        # on the grouped frame — any downstream pd.to_datetime on that column
        # silently failed. Now the bucket is a true Timestamp (first month of
        # the season within the bucket year), and the human-readable label
        # travels alongside in `_chart_bucket_label` plus a `seasonLabels` meta
        # map keyed by ISO timestamp.
        is_summer = working[time_key].dt.month.isin(list(SUMMER_MONTHS))
        bucket_year = working[time_key].dt.year.astype("Int64")
        bucket_ts = pd.to_datetime(
            pd.Series(
                [
                    # Summer → Jun-1 of the year; Winter → Dec-1 of the year.
                    f"{int(yr):04d}-{6 if summer else 12:02d}-01"
                    if pd.notna(yr)
                    else None
                    for yr, summer in zip(bucket_year, is_summer)
                ],
                index=working.index,
            ),
            errors="coerce",
        )
        working[bucket] = bucket_ts
        working["_chart_bucket_label"] = [
            f"{int(yr):04d}-summer" if summer else f"{int(yr):04d}-winter"
            if pd.notna(yr)
            else None
            for yr, summer in zip(bucket_year, is_summer)
        ]
        # Populate the labels map keyed by ISO timestamp for the renderer.
        for ts, label in zip(bucket_ts, working["_chart_bucket_label"]):
            if pd.notna(ts) and label:
                season_labels[ts.isoformat()] = label
    else:
        return df, {}

    group_keys = [bucket] + [col for col in category_cols if col in working.columns]
    if time_grain == "season":
        # Preserve the label through aggregation by including it in group keys.
        if "_chart_bucket_label" in working.columns:
            group_keys.append("_chart_bucket_label")
    agg_map = {
        col: (
            "sum"
            if aggregation_mode == "sum"
            else "mean"
            if aggregation_mode == "avg"
            else "sum"
            if dim_map.get(col) == "energy_qty"
            else "mean"
        )
        for col in num_cols
        if col in working.columns
    }
    if not agg_map:
        return df, {}

    grouped = (
        working.groupby(group_keys, dropna=False, as_index=False)
        .agg(agg_map)
        .rename(columns={bucket: time_key})
    )

    grouped = grouped.sort_values(time_key)

    meta: Dict[str, Any] = {"aggregation": time_grain}
    if time_grain == "season" and season_labels:
        meta["seasonLabels"] = season_labels

    return grouped.reset_index(drop=True), meta


def _apply_measure_transform(
    df: pd.DataFrame,
    *,
    time_key: Optional[str],
    category_cols: List[str],
    num_cols: List[str],
    measure_transform: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not time_key or time_key not in df.columns:
        return df, {}

    working = df.copy()
    group_cols = [col for col in category_cols if col in working.columns]
    sort_cols = group_cols + [time_key]
    working = working.sort_values(sort_cols)

    if measure_transform == "index_100":
        for col in num_cols:
            working[col] = _group_transform(
                working,
                group_cols,
                col,
                lambda series: _index_base_100(series),
            )
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "Index (base=100)",
        }

    if measure_transform == "cagr":
        cagr_meta: Dict[str, float] = {}
        for col in num_cols:
            working[col] = _group_transform(
                working,
                group_cols,
                col,
                lambda series: _index_base_100(series),
            )
            cagr_value = _estimate_cagr(working[col])
            if cagr_value is not None:
                cagr_meta[col] = cagr_value
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "Index (base=100)",
            "cagrBySeries": cagr_meta,
        }

    lag = _infer_period_lag(working[time_key])
    if measure_transform == "share_of_total":
        if len(num_cols) == 1 and group_cols:
            value_col = num_cols[0]
            partition_cols = [time_key] if time_key else []
            totals = working.groupby(partition_cols, dropna=False)[value_col].transform("sum")
            totals = totals.replace(0, pd.NA)
            working[value_col] = (pd.to_numeric(working[value_col], errors="coerce") / totals) * 100.0
        else:
            totals = working[num_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            totals = totals.replace(0, pd.NA)
            for col in num_cols:
                working[col] = (pd.to_numeric(working[col], errors="coerce") / totals) * 100.0
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "Share of total (%)",
        }

    if measure_transform == "mom_delta":
        for col in num_cols:
            working[col] = _group_transform(working, group_cols, col, lambda series: series.diff(1))
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "MoM change",
        }

    if measure_transform == "mom_pct":
        for col in num_cols:
            working[col] = _group_transform(working, group_cols, col, lambda series: series.pct_change(1) * 100.0)
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "MoM change (%)",
        }

    if measure_transform == "yoy_delta":
        for col in num_cols:
            working[col] = _group_transform(working, group_cols, col, lambda series: series.diff(lag))
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "YoY change",
        }

    if measure_transform == "yoy_pct":
        for col in num_cols:
            working[col] = _group_transform(working, group_cols, col, lambda series: series.pct_change(lag) * 100.0)
        working = working.dropna(subset=num_cols, how="all")
        return working.reset_index(drop=True), {
            "measureTransform": measure_transform,
            "yAxisTitle": "YoY change (%)",
        }

    return df, {}


def _group_transform(
    df: pd.DataFrame,
    group_cols: List[str],
    column: str,
    func,
) -> pd.Series:
    if group_cols:
        return df.groupby(group_cols, dropna=False)[column].transform(func)
    return func(df[column])


def _index_base_100(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    first_valid = numeric.dropna()
    if first_valid.empty:
        return pd.Series([None] * len(series), index=series.index, dtype="float64")
    base = float(first_valid.iloc[0])
    if base == 0.0:
        return pd.Series([None] * len(series), index=series.index, dtype="float64")
    return (numeric / base) * 100.0


def _estimate_cagr(series: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return None
    start = float(numeric.iloc[0])
    end = float(numeric.iloc[-1])
    periods = len(numeric) - 1
    if start <= 0.0 or end <= 0.0 or periods <= 0:
        return None
    return ((end / start) ** (1.0 / periods) - 1.0) * 100.0


def _infer_period_lag(series: pd.Series) -> int:
    dt = pd.to_datetime(series, errors="coerce").dropna().sort_values()
    if len(dt) < 3:
        return 1
    diffs = dt.diff().dropna().dt.days
    if diffs.empty:
        return 1
    median_days = float(diffs.median())
    if median_days <= 35:
        return 12
    if median_days <= 110:
        return 4
    return 1


# ---------------------------------------------------------------------------
# Phase 13 (§16.4.2): Long-form ChartFrame
#
# The wide-form path (``build_chart_frame``) above is preserved verbatim; the
# long-form path below is layered on top and behind a feature flag. The goal
# is to give every downstream renderer / override builder a single, typed
# representation of what each series means (``role``) and how it was derived
# (``transform``), without forcing a breaking change on existing callers.
# ---------------------------------------------------------------------------

LONG_FRAME_COLUMNS: Tuple[str, ...] = (
    "period",
    "series",
    "value",
    "unit",
    "role",
    "transform",
    "source_metric",
    "is_derived",
)


@dataclass
class ChartFrame:
    """Long-form representation of a chart-ready data frame.

    Columns of ``long_df`` (see ``LONG_FRAME_COLUMNS``):

    * ``period``       — ``datetime64[ns]`` (or ``NaT`` when the chart has no
      time axis). Never a string; season labels travel via ``meta``.
    * ``series``       — display label for the series (``str``).
    * ``value``        — ``float`` (or ``NaN``) after any transform.
    * ``unit``         — unit hint (e.g. ``"GEL/MWh"``, ``"%"``); ``""``
      when unknown.
    * ``role``         — ``SemanticRole`` value (string form).
    * ``transform``    — ``MeasureTransform`` value (string form).
    * ``source_metric``— raw column name this series came from before
      label rewriting.
    * ``is_derived``   — bool convenience mirror of
      ``role in {DERIVED, COMPONENT_*}`` OR ``transform != RAW``.

    ``meta`` carries chart-level metadata (``aggregation``, ``seasonLabels``,
    ``measureTransform``, ``yAxisTitle``, ...) — i.e. exactly what the wide
    builder returns as its second tuple element, so the two representations
    stay round-trip compatible.
    """

    long_df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return self.long_df is None or self.long_df.empty

    def series_names(self) -> List[str]:
        if self.is_empty():
            return []
        return [str(s) for s in self.long_df["series"].dropna().unique().tolist()]


def classify_series(
    name: str,
    *,
    derived: bool = False,
    role_hint: Optional[SemanticRole] = None,
    transform_hint: Optional[MeasureTransform] = None,
) -> Tuple[SemanticRole, MeasureTransform]:
    """Map a metric column name (plus hints) to (role, transform).

    Pure function: no DataFrame access. Called once per column during the
    wide→long conversion in :func:`from_wide`.

    Resolution order:
    1. Explicit ``role_hint`` / ``transform_hint`` win when provided — these
       come from override builders (§16.4.5) that already know the semantics
       of what they emitted.
    2. ``derived=True`` bumps the default role to ``DERIVED``.
    3. Heuristic name scan for transform suffixes (``_mom_pct``, ``_yoy_delta``,
       ``_index_100``, ``_cagr``, ``_share``) — best-effort only; the wide
       builder applies transforms in place on the same column name, so the
       suffix is usually absent and the caller is expected to pass
       ``transform_hint`` instead.
    """

    lowered = (name or "").lower()

    if role_hint is not None:
        role = role_hint
    elif derived:
        role = SemanticRole.DERIVED
    else:
        role = SemanticRole.OBSERVED

    if transform_hint is not None:
        return role, transform_hint

    # Best-effort name heuristic — only fires when the caller did not pass a
    # ``transform_hint``. Ordered most-specific-first so ``mom_pct`` does not
    # collide with ``mom_delta``.
    for suffix, transform in (
        ("mom_pct", MeasureTransform.MOM_PCT),
        ("yoy_pct", MeasureTransform.YOY_PCT),
        ("mom_delta", MeasureTransform.MOM_DELTA),
        ("yoy_delta", MeasureTransform.YOY_DELTA),
        ("index_100", MeasureTransform.INDEX_100),
        ("cagr", MeasureTransform.CAGR),
        ("share_of_total", MeasureTransform.SHARE_OF_TOTAL),
        ("share", MeasureTransform.SHARE_OF_TOTAL),
    ):
        if suffix in lowered:
            return role, transform

    return role, MeasureTransform.RAW


def from_wide(
    wide_df: pd.DataFrame,
    *,
    time_key: Optional[str],
    num_cols: List[str],
    dim_map: Optional[Dict[str, str]] = None,
    label_map: Optional[Dict[str, str]] = None,
    measure_transform: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    role_hints: Optional[Dict[str, SemanticRole]] = None,
    transform_hints: Optional[Dict[str, MeasureTransform]] = None,
    unit_map: Optional[Dict[str, str]] = None,
    derived_cols: Optional[Iterable[str]] = None,
) -> ChartFrame:
    """Melt a wide chart frame into a long-form :class:`ChartFrame`.

    * ``wide_df`` is the post-transform wide frame (output of
      :func:`build_chart_frame`). Column names are raw metric names, not
      display labels — the display labels are attached via ``label_map``.
    * ``time_key`` is the datetime column (if any). When missing, ``period``
      is filled with ``NaT`` so the long frame stays schema-stable.
    * ``measure_transform`` is the chart-wide transform applied by the wide
      builder (e.g. ``"mom_pct"``). Per-column hints in ``transform_hints``
      override this on a column-by-column basis (used by override builders
      that mix OBSERVED + DERIVED series in one frame).
    """

    dim_map = dim_map or {}
    label_map = label_map or {}
    role_hints = role_hints or {}
    transform_hints = transform_hints or {}
    unit_map = unit_map or {}
    derived_set = set(derived_cols or ())

    # Resolve the default transform from the string the wide builder used.
    try:
        default_transform = (
            MeasureTransform(measure_transform)
            if measure_transform
            else MeasureTransform.RAW
        )
    except ValueError:
        default_transform = MeasureTransform.RAW

    if wide_df is None or wide_df.empty or not num_cols:
        empty = pd.DataFrame(
            {col: pd.Series(dtype="object") for col in LONG_FRAME_COLUMNS}
        )
        # ``period`` must be datetime-typed even when empty so downstream
        # dtype assertions (tests, validators) do not trip on ``object``.
        empty["period"] = pd.to_datetime(empty["period"], errors="coerce")
        return ChartFrame(long_df=empty, meta=dict(meta or {}))

    present_cols = [col for col in num_cols if col in wide_df.columns]
    if not present_cols:
        empty = pd.DataFrame(
            {col: pd.Series(dtype="object") for col in LONG_FRAME_COLUMNS}
        )
        empty["period"] = pd.to_datetime(empty["period"], errors="coerce")
        return ChartFrame(long_df=empty, meta=dict(meta or {}))

    if time_key and time_key in wide_df.columns:
        period_series = pd.to_datetime(wide_df[time_key], errors="coerce")
    else:
        period_series = pd.Series(
            [pd.NaT] * len(wide_df), index=wide_df.index, dtype="datetime64[ns]"
        )

    records: List[Dict[str, Any]] = []
    for col in present_cols:
        role, transform = classify_series(
            col,
            derived=col in derived_set,
            role_hint=role_hints.get(col),
            transform_hint=transform_hints.get(col, default_transform)
            if col in transform_hints or measure_transform
            else None,
        )
        series_label = label_map.get(col, col)
        unit = unit_map.get(col) or dim_map.get(col) or ""
        values = pd.to_numeric(wide_df[col], errors="coerce")
        for period_val, value in zip(period_series, values):
            records.append(
                {
                    "period": period_val,
                    "series": series_label,
                    "value": None if pd.isna(value) else float(value),
                    "unit": str(unit),
                    "role": role.value,
                    "transform": transform.value,
                    "source_metric": col,
                    "is_derived": (
                        role
                        in {
                            SemanticRole.DERIVED,
                            SemanticRole.COMPONENT_PRIMARY,
                            SemanticRole.COMPONENT_SECONDARY,
                        }
                    )
                    or transform != MeasureTransform.RAW,
                }
            )

    long_df = pd.DataFrame(records, columns=list(LONG_FRAME_COLUMNS))
    # Force dtypes so downstream tests / renderers can rely on them.
    long_df["period"] = pd.to_datetime(long_df["period"], errors="coerce")
    long_df["series"] = long_df["series"].astype(str)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["unit"] = long_df["unit"].astype(str)
    long_df["role"] = long_df["role"].astype(str)
    long_df["transform"] = long_df["transform"].astype(str)
    long_df["source_metric"] = long_df["source_metric"].astype(str)
    long_df["is_derived"] = long_df["is_derived"].astype(bool)

    return ChartFrame(long_df=long_df, meta=dict(meta or {}))


def to_wide(frame: ChartFrame, time_key: Optional[str] = "period") -> pd.DataFrame:
    """Pivot a :class:`ChartFrame` back to wide form for legacy consumers.

    The wire payload sent to the frontend is still wide; this is the bridge
    that lets the pipeline treat ``ChartFrame`` as the source of truth
    internally while emitting the historical shape on the wire.
    """

    if frame is None or frame.is_empty():
        return pd.DataFrame()

    long_df = frame.long_df
    has_period = long_df["period"].notna().any()
    pivot_index: List[str] = []
    if has_period:
        pivot_index.append("period")
    wide = (
        long_df.pivot_table(
            index=pivot_index if pivot_index else None,
            columns="series",
            values="value",
            aggfunc="first",
            dropna=False,
        )
        if pivot_index
        else long_df.set_index("series")["value"].to_frame().T
    )
    wide = wide.reset_index() if pivot_index else wide.reset_index(drop=True)
    wide.columns.name = None
    if pivot_index and time_key and time_key != "period" and "period" in wide.columns:
        wide = wide.rename(columns={"period": time_key})
    return wide


def build_chart_frame_long(
    df: pd.DataFrame,
    *,
    time_key: Optional[str],
    category_cols: List[str],
    num_cols: List[str],
    dim_map: Dict[str, str],
    time_grain: Optional[str] = None,
    measure_transform: str = "raw",
    label_map: Optional[Dict[str, str]] = None,
    role_hints: Optional[Dict[str, SemanticRole]] = None,
    transform_hints: Optional[Dict[str, MeasureTransform]] = None,
    unit_map: Optional[Dict[str, str]] = None,
    derived_cols: Optional[Iterable[str]] = None,
) -> ChartFrame:
    """Build a long-form :class:`ChartFrame` from raw evidence data.

    Thin wrapper: delegates the dtype coercion, time-grain bucketing, and
    measure-transform application to the existing :func:`build_chart_frame`,
    then melts the result via :func:`from_wide`. Behaviour and numerics are
    identical to the wide path by construction — this path is purely a
    shape change.
    """

    wide_df, wide_meta = build_chart_frame(
        df,
        time_key=time_key,
        category_cols=category_cols,
        num_cols=num_cols,
        dim_map=dim_map,
        time_grain=time_grain,
        measure_transform=measure_transform,
    )
    return from_wide(
        wide_df,
        time_key=time_key,
        num_cols=num_cols,
        dim_map=dim_map,
        label_map=label_map,
        measure_transform=wide_meta.get("measureTransform", measure_transform),
        meta=wide_meta,
        role_hints=role_hints,
        transform_hints=transform_hints,
        unit_map=unit_map,
        derived_cols=derived_cols,
    )
