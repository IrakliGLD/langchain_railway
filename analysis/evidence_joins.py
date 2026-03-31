"""Scoped analytical join helpers for multi-evidence merging.

Supports the 5 known join patterns between datasets produced by the
system's typed tools.  Not a generic framework -- each pattern explicitly
names the columns it selects and the join key it uses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def join_evidence(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    primary_tool: str,
    secondary_tool: str,
) -> pd.DataFrame:
    """Date-aligned left join using a known pattern.

    Returns a copy of ``primary_df`` with relevant columns from
    ``secondary_df`` merged in.  If the pattern is unknown or the join
    fails (e.g. no date columns), returns ``primary_df`` unchanged.
    """
    if primary_df.empty or secondary_df.empty:
        return primary_df

    date_primary = _find_date_column(primary_df)
    date_secondary = _find_date_column(secondary_df)
    if not date_primary or not date_secondary:
        log.warning(
            "evidence_join: could not find date column. primary=%s secondary=%s",
            date_primary, date_secondary,
        )
        return primary_df

    # Resolve which columns to select from the secondary dataset
    select_cols = _select_columns(secondary_df, secondary_tool, primary_tool)
    if not select_cols:
        log.info(
            "evidence_join: no columns to merge from %s into %s",
            secondary_tool, primary_tool,
        )
        return primary_df

    try:
        merged = _merge_on_date(
            primary_df, secondary_df,
            date_primary, date_secondary,
            select_cols,
        )
        log.info(
            "evidence_join: merged %d columns from %s into %s (%d rows)",
            len(select_cols), secondary_tool, primary_tool, len(merged),
        )
        return merged
    except Exception as exc:
        log.warning("evidence_join: merge failed: %s", exc)
        return primary_df


def join_evidence_with_provenance(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    primary_tool: str,
    secondary_tool: str,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Like ``join_evidence`` but also returns join provenance metadata.

    Returns ``(merged_df, provenance_dict)`` where *provenance_dict* is
    ``None`` when no merge was performed (empty inputs, missing date cols,
    or merge failure).
    """
    if primary_df.empty or secondary_df.empty:
        return primary_df, None

    date_primary = _find_date_column(primary_df)
    date_secondary = _find_date_column(secondary_df)
    if not date_primary or not date_secondary:
        return primary_df, None

    select_cols = _select_columns(secondary_df, secondary_tool, primary_tool)
    if not select_cols:
        return primary_df, None

    try:
        merged = _merge_on_date(
            primary_df, secondary_df,
            date_primary, date_secondary,
            select_cols,
        )
        provenance: Dict[str, Any] = {
            "primary_tool": primary_tool,
            "secondary_tool": secondary_tool,
            "join_key": date_primary,
            "join_type": "left",
            "columns_added": select_cols,
            "primary_rows": len(primary_df),
            "merged_rows": len(merged),
        }
        return merged, provenance
    except Exception as exc:
        log.warning("evidence_join: merge failed: %s", exc)
        return primary_df, None


# ---------------------------------------------------------------------------
# Column selection per join pattern
# ---------------------------------------------------------------------------

def _select_columns(
    secondary_df: pd.DataFrame,
    secondary_tool: str,
    primary_tool: str,
) -> List[str]:
    """Select relevant columns from the secondary dataset."""

    cols = list(secondary_df.columns)

    if secondary_tool == "get_balancing_composition":
        # Share columns for composition context
        return [c for c in cols if c.startswith("share_")]

    if secondary_tool == "get_prices":
        # Price + xrate columns for price context
        price_cols = [
            c for c in cols
            if c.startswith("p_bal_") or c.startswith("p_dereg_") or c.startswith("p_gcap_")
            or c == "xrate"
        ]
        return price_cols

    if secondary_tool == "get_tariffs":
        # Tariff columns
        tariff_cols = [
            c for c in cols
            if "tariff" in c.lower() or c.startswith("t_")
        ]
        return tariff_cols if tariff_cols else [c for c in cols if c.lower() != _find_date_column(secondary_df)]

    if secondary_tool == "get_generation_mix":
        # Generation columns (excluding date)
        date_col = _find_date_column(secondary_df)
        return [c for c in cols if c != date_col]

    return []


# ---------------------------------------------------------------------------
# Merge mechanics
# ---------------------------------------------------------------------------

def _merge_on_date(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    date_primary: str,
    date_secondary: str,
    select_cols: List[str],
) -> pd.DataFrame:
    """Left join primary with selected columns from secondary on date."""

    # Work on copies to avoid mutating originals
    primary = primary_df.copy()
    secondary = secondary_df.copy()

    # Normalize date columns to datetime
    primary[date_primary] = pd.to_datetime(primary[date_primary], errors="coerce")
    secondary[date_secondary] = pd.to_datetime(secondary[date_secondary], errors="coerce")

    # Drop rows with unparseable dates
    primary = primary.dropna(subset=[date_primary])
    secondary = secondary.dropna(subset=[date_secondary])

    # Select only the date column + desired columns from secondary
    merge_cols = [date_secondary] + [c for c in select_cols if c in secondary.columns]
    secondary_subset = secondary[merge_cols].rename(
        columns={date_secondary: date_primary},
    )

    # Drop duplicate date rows in secondary (keep first)
    secondary_subset = secondary_subset.drop_duplicates(subset=[date_primary], keep="first")

    merged = primary.merge(secondary_subset, on=date_primary, how="left")
    return merged


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """Find the date column in a DataFrame by name heuristic."""
    for col in df.columns:
        if col.lower() in ("date", "month", "period", "trade_date"):
            return col
        if "date" in col.lower():
            return col
    return None
