"""Per-tool adapters that normalize raw DataFrames into canonical evidence frames.

Each adapter is a write-once mapping from tool-specific columns to canonical
frame rows.  Domain logic (display names, unit labels, column prefixes) lives
here — not in Stage 4 regex.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from agent.metric_units import METRIC_UNITS, UnitCompatibilityError
from contracts.evidence_frames import (
    CanonicalFrame,
    ComparisonFrame,
    EntitySetFrame,
    ObservationFrame,
)
from contracts.question_analysis import AnswerKind, FilterCondition, FilterOperator

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# Display-name mappings (domain knowledge, written once per tool)
# ---------------------------------------------------------------------------

_PRICE_METRIC_LABELS = {
    "p_bal_gel": ("balancing_price", "GEL", "tetri/kWh"),
    "p_bal_usd": ("balancing_price", "USD", "USD¢/kWh"),
    "p_dereg_gel": ("deregulated_price", "GEL", "tetri/kWh"),
    "p_dereg_usd": ("deregulated_price", "USD", "USD¢/kWh"),
    "p_gcap_gel": ("guaranteed_capacity_price", "GEL", "tetri/kWh"),
    "p_gcap_usd": ("guaranteed_capacity_price", "USD", "USD¢/kWh"),
    "xrate": ("exchange_rate", "GEL/USD", "GEL/USD"),
}

_PRICE_UNIT_CONTRACTS = {
    "p_bal_gel": "price.gel",
    "p_bal_usd": "price.usd",
    "p_dereg_gel": "price.gel",
    "p_dereg_usd": "price.usd",
    "p_gcap_gel": "price.gel",
    "p_gcap_usd": "price.usd",
    "xrate": "exchange_rate.gel_usd",
}

_TARIFF_ENTITY_LABELS = {
    "enguri_hpp": "Enguri HPP",
    "enguri": "Enguri HPP",
    "vardnili_hpp": "Vardnili HPP",
    "vardnili": "Vardnili HPP",
    "dzevrula_hpp": "Dzevruli HPP",
    "dzevruli_hpp": "Dzevruli HPP",
    "gumati_hpp": "Gumati HPP",
    "shaori_hpp": "Shaori HPP",
    "rioni_hpp": "Rioni HPP",
    "lajanuri_hpp": "Lajanuri HPP",
    "zhinvali_hpp": "Zhinvali HPP",
    "vartsikhe_hpp": "Vartsikhe HPP",
    "khramhesi_i": "Khrami I HPP",
    "khramhesi_ii": "Khrami II HPP",
    "gardabani_tpp": "Gardabani TPP",
    "mtkvari_tpp": "Mtkvari Energy",
    "mktvari_tpp": "Mtkvari Energy",
    "tbilresi_tpp": "Tbilisi TPP",
    "tbilsresi_tpp": "Tbilisi TPP",
    "gpower_tpp": "G-POWER",
    "grouped_old_tpp": "Old TPP Group",
    "regulated_hpp": "Regulated HPPs",
    "regulated_new_tpp": "Regulated New TPPs",
    "regulated_old_tpp": "Regulated Old TPPs",
}

_COMPOSITION_ENTITY_LABELS = {
    "share_import": "Import",
    "share_deregulated_ren": "Deregulated Renewable",
    "share_regulated_hpp": "Regulated HPP",
    "share_regulated_new_tpp": "Regulated New TPP",
    "share_regulated_old_tpp": "Regulated Old TPP",
    "share_renewable_ppa": "Renewable PPA",
    "share_thermal_ppa": "Thermal PPA",
    "share_CfD_scheme": "CfD Scheme",
}

_TARIFF_VALUE_UNITS = {
    "gel": "GEL/MWh",
    "usd": "USD/MWh",
}


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

_FILTER_OPS = {
    FilterOperator.GT: lambda v, t: v > t,
    FilterOperator.GTE: lambda v, t: v >= t,
    FilterOperator.LT: lambda v, t: v < t,
    FilterOperator.LTE: lambda v, t: v <= t,
    FilterOperator.EQ: lambda v, t: v == t,
}

# Contract-vocabulary metric names (analyzer emits these in filter.metric) mapped
# to the set of frame-row metrics they match.  Frame rows use display-oriented
# names like "balancing_price"; the analyzer emits tool params vocabulary
# ("balancing").  Without this mapping, exact-equality comparison silently
# drops every threshold filter.
_CONTRACT_METRIC_ALIASES = {
    "balancing": {"balancing_price"},
    "deregulated": {"deregulated_price"},
    "guaranteed_capacity": {"guaranteed_capacity_price"},
    "exchange_rate": {"exchange_rate"},
}


def _metric_matches(row_metric: str, contract_metric: str) -> bool:
    """Return True if row_metric matches contract_metric (empty key matches all)."""
    if not contract_metric:
        return True
    if row_metric == contract_metric:
        return True
    return row_metric in _CONTRACT_METRIC_ALIASES.get(contract_metric, set())


def _apply_filter(rows: list[dict], filter_cond: Optional[FilterCondition]) -> list[dict]:
    """Apply a dimensional filter after canonical conversion.

    Rows for other metrics, and same-name rows in an incompatible currency,
    remain untouched. If no matching row can interpret the explicit unit, fail
    closed rather than comparing unrelated dimensions.
    """
    if filter_cond is None:
        return _without_internal_contracts(rows)
    op_fn = _FILTER_OPS.get(filter_cond.operator)
    if op_fn is None:
        return _without_internal_contracts(rows)
    metric_key = filter_cond.metric
    filtered = []
    matched = 0
    compatible = 0
    for row in rows:
        val = row.get("value")
        row_metric = row.get("metric", "")
        if _metric_matches(row_metric, metric_key):
            matched += 1
            contract_id = row.get("_unit_contract")
            try:
                threshold = (
                    METRIC_UNITS.get(contract_id).input_to_canonical(
                        filter_cond.value, filter_cond.unit
                    )
                    if contract_id
                    else float(filter_cond.value)
                )
            except UnitCompatibilityError:
                filtered.append(row)
                continue
            compatible += 1
            if val is not None and op_fn(val, threshold):
                filtered.append(row)
        else:
            # Keep rows for other metrics (filter applies only to specified metric).
            filtered.append(row)
    if matched and not compatible:
        raise UnitCompatibilityError(
            f"Filter unit {filter_cond.unit!r} is incompatible with metric {metric_key!r}"
        )
    return _without_internal_contracts(filtered)


def _without_internal_contracts(rows: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in row.items() if key != "_unit_contract"}
        for row in rows
    ]


def _canonical_value(metric_id: str, value: float) -> tuple[float, str]:
    definition = METRIC_UNITS.get(metric_id)
    return definition.storage_to_canonical(value), definition.canonical_unit


# ---------------------------------------------------------------------------
# Price adapter
# ---------------------------------------------------------------------------

def adapt_prices(
    df: pd.DataFrame,
    provenance_refs: Optional[List[str]] = None,
    filter_cond: Optional[FilterCondition] = None,
) -> ObservationFrame:
    """Convert get_prices DataFrame into ObservationFrame."""
    rows: list[dict] = []
    prov = provenance_refs or []
    date_col = "date" if "date" in df.columns else "year"

    for _, raw_row in df.iterrows():
        period = str(raw_row.get(date_col, ""))
        for col, (metric, currency, unit) in _PRICE_METRIC_LABELS.items():
            if col in df.columns:
                val = raw_row.get(col)
                if pd.notna(val):
                    contract_id = _PRICE_UNIT_CONTRACTS[col]
                    canonical_value, canonical_unit = _canonical_value(contract_id, val)
                    rows.append({
                        "period": period,
                        "entity_id": f"{metric}_{currency.lower()}",
                        "entity_label": f"{metric} ({currency})",
                        "metric": metric,
                        "value": canonical_value,
                        "unit": canonical_unit,
                        "_unit_contract": contract_id,
                    })

    rows = _apply_filter(rows, filter_cond)
    return ObservationFrame(rows=rows, provenance_refs=prov)


# ---------------------------------------------------------------------------
# Tariff adapter
# ---------------------------------------------------------------------------

def adapt_tariffs(
    df: pd.DataFrame,
    provenance_refs: Optional[List[str]] = None,
    filter_cond: Optional[FilterCondition] = None,
    answer_kind: Optional[AnswerKind] = None,
) -> CanonicalFrame:
    """Convert get_tariffs DataFrame into ObservationFrame or EntitySetFrame."""
    prov = provenance_refs or []

    # For LIST answer_kind, preserve per-entity snapshot values on the
    # EntitySetFrame rows so Stage 4 can deterministically render
    # single-period multi-entity tariff lookups without falling back to the LLM.
    if answer_kind == AnswerKind.LIST:
        entity_rows: list[dict] = []
        snapshot_period = _tariff_snapshot_period(df)
        for alias in _tariff_aliases(df):
            snapshot_values = _tariff_snapshot_values(df, alias, snapshot_period)
            has_any_observations = any(
                col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()
                for col in (
                    f"{alias}_tariff_gel",
                    f"{alias}_tariff_usd",
                )
            )
            if not has_any_observations:
                continue

            label = _TARIFF_ENTITY_LABELS.get(alias, alias)
            attributes = {}
            if snapshot_period is not None and snapshot_values:
                attributes = {
                    "period": snapshot_period,
                    "values": snapshot_values,
                }
            entity_rows.append({
                "entity_id": alias,
                "entity_label": label,
                "membership_reason": "observed_tariff",
                "attributes": attributes,
            })
        return EntitySetFrame(rows=entity_rows, provenance_refs=prov)

    # Default: ObservationFrame (timeseries/scalar)
    rows: list[dict] = []
    for _, raw_row in df.iterrows():
        period = str(raw_row.get("date", ""))
        for col in df.columns:
            if col == "date":
                continue
            val = raw_row.get(col)
            if pd.notna(val):
                alias = col.replace("_tariff_gel", "").replace("_tariff_usd", "")
                label = _TARIFF_ENTITY_LABELS.get(alias, alias)
                currency = "GEL" if "_gel" in col else "USD"
                contract_id = f"tariff.{currency.lower()}"
                canonical_value, canonical_unit = _canonical_value(contract_id, val)
                rows.append({
                    "period": period,
                    "entity_id": alias,
                    "entity_label": label,
                    "metric": "tariff",
                    "value": canonical_value,
                    "unit": canonical_unit,
                    "_unit_contract": contract_id,
                })

    rows = _apply_filter(rows, filter_cond)
    return ObservationFrame(rows=rows, provenance_refs=prov)


def _tariff_aliases(df: pd.DataFrame) -> list[str]:
    seen: dict[str, None] = {}
    for col in df.columns:
        alias = _tariff_alias_from_column(col)
        if alias and alias not in seen:
            seen[alias] = None
    return list(seen)


def _tariff_alias_from_column(col: str) -> Optional[str]:
    if col.endswith("_tariff_gel"):
        return col[: -len("_tariff_gel")]
    if col.endswith("_tariff_usd"):
        return col[: -len("_tariff_usd")]
    return None


def _tariff_snapshot_period(df: pd.DataFrame) -> Optional[str]:
    if "date" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    day_periods = list(dict.fromkeys(dates.dt.strftime("%Y-%m-%d")))
    if len(day_periods) == 1:
        return day_periods[0]

    month_periods = list(dict.fromkeys(dates.dt.strftime("%Y-%m")))
    if len(month_periods) == 1:
        return month_periods[0]

    year_periods = list(dict.fromkeys(dates.dt.strftime("%Y")))
    if len(year_periods) == 1:
        return year_periods[0]

    return None


def _tariff_snapshot_period_kind(snapshot_period: Optional[str]) -> Optional[str]:
    period = str(snapshot_period or "").strip()
    if not period:
        return None
    if len(period) == 10 and period[4] == "-" and period[7] == "-":
        return "day"
    if len(period) == 7 and period[4] == "-":
        return "month"
    if len(period) == 4 and period.isdigit():
        return "year"
    return None


def _tariff_snapshot_values(
    df: pd.DataFrame,
    alias: str,
    snapshot_period: Optional[str],
) -> list[dict]:
    period_kind = _tariff_snapshot_period_kind(snapshot_period)
    if snapshot_period is None or period_kind is None or "date" not in df.columns:
        return []

    working = df.copy()
    working["_snapshot_date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["_snapshot_date"])
    if working.empty:
        return []

    if period_kind == "day":
        working = working[
            working["_snapshot_date"].dt.strftime("%Y-%m-%d") == snapshot_period
        ]
    elif period_kind == "month":
        working = working[
            working["_snapshot_date"].dt.strftime("%Y-%m") == snapshot_period
        ]
    elif period_kind == "year":
        working = working[
            working["_snapshot_date"].dt.strftime("%Y") == snapshot_period
        ]
    if working.empty:
        return []

    values: list[dict] = []
    for currency, unit in _TARIFF_VALUE_UNITS.items():
        col = f"{alias}_tariff_{currency}"
        if col not in working.columns:
            continue
        value_rows = working[["_snapshot_date"]].copy()
        value_rows["value"] = pd.to_numeric(working[col], errors="coerce")
        value_rows = value_rows.dropna(subset=["value"])
        if value_rows.empty:
            continue

        unique_values = list(dict.fromkeys(float(v) for v in value_rows["value"].tolist()))
        if len(unique_values) == 1:
            values.append({
                "label": "tariff",
                "currency": currency.upper(),
                "value": unique_values[0],
                "unit": unit,
            })
            continue

        if period_kind == "month":
            sub_period_series = value_rows["_snapshot_date"].dt.strftime("%Y-%m-%d")
        elif period_kind == "year":
            sub_period_series = value_rows["_snapshot_date"].dt.strftime("%Y-%m")
        else:
            sub_period_series = pd.Series([""] * len(value_rows), index=value_rows.index)

        seen_entries: set[tuple[str, float]] = set()
        for (_, row), sub_period in zip(value_rows.iterrows(), sub_period_series.tolist()):
            numeric_value = float(row["value"])
            key = (sub_period, numeric_value)
            if key in seen_entries:
                continue
            seen_entries.add(key)
            entry = {
                "label": "tariff",
                "currency": currency.upper(),
                "value": numeric_value,
                "unit": unit,
            }
            if sub_period:
                entry["sub_period"] = sub_period
            values.append(entry)
    return values


# ---------------------------------------------------------------------------
# Generation mix adapter
# ---------------------------------------------------------------------------

def adapt_generation_mix(
    df: pd.DataFrame,
    provenance_refs: Optional[List[str]] = None,
    filter_cond: Optional[FilterCondition] = None,
) -> ObservationFrame:
    """Convert get_generation_mix DataFrame into ObservationFrame."""
    rows: list[dict] = []
    prov = provenance_refs or []
    canonical_metric_contracts = {
        "total_demand": "energy.quantity",
        "total_domestic_generation": "energy.quantity",
        "local_generation": "energy.quantity",
        "import_dependent_supply": "energy.quantity",
        "total_supply": "energy.quantity",
        "import_dependency_ratio": "ratio.share",
    }

    if "type_tech" not in df.columns:
        period_col = next((col for col in ("period", "date", "year", "month") if col in df.columns), None)
        if period_col is None:
            return ObservationFrame(rows=rows, provenance_refs=prov)

        for _, raw_row in df.iterrows():
            period = str(raw_row.get(period_col, ""))
            for col in df.columns:
                if col == period_col:
                    continue
                val = raw_row.get(col)
                if pd.isna(val):
                    continue
                if col in canonical_metric_contracts:
                    contract_id = canonical_metric_contracts[col]
                    canonical_value, canonical_unit = _canonical_value(contract_id, val)
                    rows.append({
                        "period": period,
                        "entity_id": "system",
                        "entity_label": "System",
                        "metric": col,
                        "value": canonical_value,
                        "unit": canonical_unit,
                        "_unit_contract": contract_id,
                    })
                elif col.startswith("quantity_"):
                    entity_id = col[len("quantity_"):]
                    canonical_value, canonical_unit = _canonical_value("energy.quantity", val)
                    rows.append({
                        "period": period,
                        "entity_id": entity_id,
                        "entity_label": entity_id.replace("_", " ").title(),
                        "metric": "generation_quantity",
                        "value": canonical_value,
                        "unit": canonical_unit,
                        "_unit_contract": "energy.quantity",
                    })
                elif col.startswith("share_"):
                    entity_id = col[len("share_"):]
                    canonical_value, canonical_unit = _canonical_value("ratio.share", val)
                    rows.append({
                        "period": period,
                        "entity_id": entity_id,
                        "entity_label": entity_id.replace("_", " ").title(),
                        "metric": "generation_share",
                        "value": canonical_value,
                        "unit": canonical_unit,
                        "_unit_contract": "ratio.share",
                    })
        rows = _apply_filter(rows, filter_cond)
        return ObservationFrame(rows=rows, provenance_refs=prov)

    for _, raw_row in df.iterrows():
        period = str(raw_row.get("period", ""))
        tech_type = str(raw_row.get("type_tech", "unknown"))
        qty = raw_row.get("quantity_tech")
        share = raw_row.get("share_tech")

        if pd.notna(qty):
            canonical_value, canonical_unit = _canonical_value("energy.quantity", qty)
            rows.append({
                "period": period,
                "entity_id": tech_type,
                "entity_label": tech_type.replace("_", " ").title(),
                "metric": "generation_quantity",
                "value": canonical_value,
                "unit": canonical_unit,
                "_unit_contract": "energy.quantity",
            })
        if pd.notna(share):
            canonical_value, canonical_unit = _canonical_value("ratio.share", share)
            rows.append({
                "period": period,
                "entity_id": tech_type,
                "entity_label": tech_type.replace("_", " ").title(),
                "metric": "generation_share",
                "value": canonical_value,
                "unit": canonical_unit,
                "_unit_contract": "ratio.share",
            })

    rows = _apply_filter(rows, filter_cond)
    return ObservationFrame(rows=rows, provenance_refs=prov)


# ---------------------------------------------------------------------------
# Balancing composition adapter
# ---------------------------------------------------------------------------

def adapt_balancing_composition(
    df: pd.DataFrame,
    provenance_refs: Optional[List[str]] = None,
    filter_cond: Optional[FilterCondition] = None,
) -> ObservationFrame:
    """Convert get_balancing_composition DataFrame into ObservationFrame."""
    rows: list[dict] = []
    prov = provenance_refs or []

    for _, raw_row in df.iterrows():
        period = str(raw_row.get("date", ""))
        for col in df.columns:
            if col in ("date", "segment"):
                continue
            val = raw_row.get(col)
            if pd.notna(val):
                label = _COMPOSITION_ENTITY_LABELS.get(col, col)
                entity_id = col.replace("share_", "")
                canonical_value, canonical_unit = _canonical_value("ratio.share", val)
                rows.append({
                    "period": period,
                    "entity_id": entity_id,
                    "entity_label": label,
                    "metric": "balancing_share",
                    "value": canonical_value,
                    "unit": canonical_unit,
                    "_unit_contract": "ratio.share",
                })

    rows = _apply_filter(rows, filter_cond)
    return ObservationFrame(rows=rows, provenance_refs=prov)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TOOL_ADAPTERS = {
    "get_prices": adapt_prices,
    "get_tariffs": adapt_tariffs,
    "get_generation_mix": adapt_generation_mix,
    "get_balancing_composition": adapt_balancing_composition,
}


def adapt_tool_result(
    tool_name: str,
    df: pd.DataFrame,
    provenance_refs: Optional[List[str]] = None,
    filter_cond: Optional[FilterCondition] = None,
    answer_kind: Optional[AnswerKind] = None,
) -> CanonicalFrame | None:
    """Normalize a tool's DataFrame into the appropriate canonical frame.

    Returns None if the tool is not recognized.
    """
    adapter = _TOOL_ADAPTERS.get(tool_name)
    if adapter is None:
        log.warning("No frame adapter for tool %s", tool_name)
        return None

    try:
        if tool_name == "get_tariffs":
            return adapter(df, provenance_refs=provenance_refs, filter_cond=filter_cond, answer_kind=answer_kind)
        return adapter(df, provenance_refs=provenance_refs, filter_cond=filter_cond)
    except Exception as exc:
        log.warning("Frame adapter failed for %s: %s", tool_name, exc)
        return None
