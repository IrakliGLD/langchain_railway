"""Per-tool adapters that normalize raw DataFrames into canonical evidence frames.

Each adapter is a write-once mapping from tool-specific columns to canonical
frame rows.  Domain logic (display names, unit labels, column prefixes) lives
here — not in Stage 4 regex.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

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
    "share_deregulated_hydro": "Deregulated Hydro",
    "share_regulated_hpp": "Regulated HPP",
    "share_regulated_new_tpp": "Regulated New TPP",
    "share_regulated_old_tpp": "Regulated Old TPP",
    "share_renewable_ppa": "Renewable PPA",
    "share_thermal_ppa": "Thermal PPA",
    "share_CfD_scheme": "CfD Scheme",
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
    """Apply a value-based filter to observation rows (after fetch, before framing)."""
    if filter_cond is None:
        return rows
    op_fn = _FILTER_OPS.get(filter_cond.operator)
    if op_fn is None:
        return rows
    metric_key = filter_cond.metric
    threshold = filter_cond.value
    filtered = []
    for row in rows:
        val = row.get("value")
        row_metric = row.get("metric", "")
        if _metric_matches(row_metric, metric_key):
            if val is not None and op_fn(val, threshold):
                filtered.append(row)
        else:
            # Keep rows for other metrics (filter applies only to specified metric).
            filtered.append(row)
    return filtered


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
                    rows.append({
                        "period": period,
                        "entity_id": f"{metric}_{currency.lower()}",
                        "entity_label": f"{metric} ({currency})",
                        "metric": metric,
                        "value": float(val),
                        "unit": unit,
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

    # For LIST answer_kind, produce an EntitySetFrame of tariff entities.
    if answer_kind == AnswerKind.LIST:
        entity_rows: list[dict] = []
        for col in df.columns:
            if col == "date":
                continue
            # Extract entity alias from column name: e.g. "enguri_tariff_gel" -> "enguri"
            alias = col.replace("_tariff_gel", "").replace("_tariff_usd", "")
            label = _TARIFF_ENTITY_LABELS.get(alias, alias)
            entity_rows.append({
                "entity_id": alias,
                "entity_label": label,
                "membership_reason": "regulated",
                "attributes": {},
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
                rows.append({
                    "period": period,
                    "entity_id": alias,
                    "entity_label": label,
                    "metric": "tariff",
                    "value": float(val),
                    "unit": f"{currency}/MWh",
                })

    rows = _apply_filter(rows, filter_cond)
    return ObservationFrame(rows=rows, provenance_refs=prov)


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
    canonical_metric_units = {
        "total_demand": "MWh",
        "total_domestic_generation": "MWh",
        "local_generation": "MWh",
        "import_dependent_supply": "MWh",
        "total_supply": "MWh",
        "import_dependency_ratio": "%",
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
                if col in canonical_metric_units:
                    rows.append({
                        "period": period,
                        "entity_id": "system",
                        "entity_label": "System",
                        "metric": col,
                        "value": float(val),
                        "unit": canonical_metric_units[col],
                    })
                elif col.startswith("quantity_"):
                    entity_id = col[len("quantity_"):]
                    rows.append({
                        "period": period,
                        "entity_id": entity_id,
                        "entity_label": entity_id.replace("_", " ").title(),
                        "metric": "generation_quantity",
                        "value": float(val),
                        "unit": "MWh",
                    })
                elif col.startswith("share_"):
                    entity_id = col[len("share_"):]
                    rows.append({
                        "period": period,
                        "entity_id": entity_id,
                        "entity_label": entity_id.replace("_", " ").title(),
                        "metric": "generation_share",
                        "value": float(val),
                        "unit": "%",
                    })
        rows = _apply_filter(rows, filter_cond)
        return ObservationFrame(rows=rows, provenance_refs=prov)

    for _, raw_row in df.iterrows():
        period = str(raw_row.get("period", ""))
        tech_type = str(raw_row.get("type_tech", "unknown"))
        qty = raw_row.get("quantity_tech")
        share = raw_row.get("share_tech")

        if pd.notna(qty):
            rows.append({
                "period": period,
                "entity_id": tech_type,
                "entity_label": tech_type.replace("_", " ").title(),
                "metric": "generation_quantity",
                "value": float(qty),
                "unit": "MWh",
            })
        if pd.notna(share):
            rows.append({
                "period": period,
                "entity_id": tech_type,
                "entity_label": tech_type.replace("_", " ").title(),
                "metric": "generation_share",
                "value": float(share),
                "unit": "%",
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
                rows.append({
                    "period": period,
                    "entity_id": entity_id,
                    "entity_label": label,
                    "metric": "balancing_share",
                    "value": float(val),
                    "unit": "%",
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
