"""Canonical display and arithmetic units for analytical metric values."""

from __future__ import annotations


def metric_value_unit(metric: str) -> str:
    """Return the physical unit of one observation for a canonical metric."""

    normalized = str(metric or "").strip().lower()
    if normalized.endswith("_usd") or normalized == "tariff_usd":
        return "USD/MWh"
    if normalized.endswith("_gel") or normalized == "tariff_gel":
        return "GEL/MWh"
    if normalized in {"xrate", "exchange_rate"}:
        return "GEL/USD"
    if normalized.startswith("share_") or normalized.endswith("_ratio"):
        return "share (0-1)"
    if any(
        token in normalized
        for token in ("quantity", "generation", "demand", "consumption", "supply")
    ):
        return "MWh"
    return "value"


def metric_currency(metric: str) -> str | None:
    """Return the settlement currency encoded by a price metric."""

    normalized = str(metric or "").strip().lower()
    if normalized.endswith("_usd") or normalized == "tariff_usd":
        return "USD"
    if normalized.endswith("_gel") or normalized == "tariff_gel":
        return "GEL"
    return None


def metric_is_additive(metric: str) -> bool:
    """Whether observations can be meaningfully summed across periods."""

    normalized = str(metric or "").strip().lower()
    return any(
        token in normalized
        for token in ("quantity", "generation", "demand", "consumption", "supply")
    ) and not (
        normalized.startswith("share_")
        or normalized.endswith("_ratio")
        or normalized.startswith("p_")
    )
