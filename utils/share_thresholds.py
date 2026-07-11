"""Shared normalization helpers for balancing-share thresholds."""

from __future__ import annotations


def normalize_share_threshold(raw_value: float, matched_text: str) -> float:
    """Return a balancing-share threshold as a fraction in ``[0, 1]``.

    Values accompanied by an explicit percent sign are always percentages,
    including sub-one values such as ``0.2%``.  Without a percent sign, the
    legacy convention remains: values above one are percentages and values at
    or below one are already fractions.
    """

    value = float(raw_value)
    if "%" in str(matched_text or ""):
        return value / 100.0
    return value / 100.0 if value > 1 else value
