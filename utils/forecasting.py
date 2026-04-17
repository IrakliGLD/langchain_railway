"""Deterministic helpers for forecast query parsing."""

from __future__ import annotations

import re


_EXCLUDED_YEAR_PATTERN = re.compile(
    r"(?:exclude|excluding|without|except|omit|omitting|do not use|don't use|not use)"
    r"[^0-9]{0,30}"
    r"((?:19|20)\d{2}(?:[^0-9]+(?:19|20)\d{2})*)",
    re.IGNORECASE,
)


def extract_forecast_horizon_years(query: str) -> int:
    """Extract the requested forecast horizon in years from free text."""

    if not query:
        return 3

    match = re.search(r"(\d+)\s*-?year", query)
    if match:
        return min(max(int(match.group(1)), 1), 20)
    if "decade" in query.lower():
        return 10
    return 3


def extract_excluded_years(query: str) -> set[int]:
    """Return calendar years the user explicitly asked to exclude from modeling."""

    if not query:
        return set()

    years: set[int] = set()
    for block in _EXCLUDED_YEAR_PATTERN.findall(query):
        years.update(int(year) for year in re.findall(r"(?:19|20)\d{2}", block))
    return years
