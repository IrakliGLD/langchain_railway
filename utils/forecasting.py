"""Deterministic helpers for forecast query parsing."""

from __future__ import annotations

import re

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

_EXCLUDED_YEAR_PATTERN = re.compile(
    r"(?:exclude|excluding|without|except|omit|omitting|do not use|don't use|not use)"
    r"[^0-9]{0,30}"
    r"((?:19|20)\d{2}(?:[^0-9]+(?:19|20)\d{2})*)",
    re.IGNORECASE,
)


def extract_forecast_horizon_years(
    query: str,
    *,
    structured_horizon_years: int | None = None,
) -> int:
    """Resolve horizon with structured analysis authoritative over text fallback."""

    if structured_horizon_years is not None:
        return min(max(int(structured_horizon_years), 1), 20)

    if not query:
        return 3

    normalized = str(query).strip().lower()
    match = re.search(r"\b(\d+)\s*-?\s*years?\b", normalized, re.IGNORECASE)
    if match:
        return min(max(int(match.group(1)), 1), 20)
    word_pattern = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    word_match = re.search(rf"\b({word_pattern})\s*-?\s*years?\b", normalized, re.IGNORECASE)
    if word_match:
        return _NUMBER_WORDS[word_match.group(1).lower()]
    if "decade" in normalized:
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


def forecast_history_window_years(horizon_years: int, excluded_years_count: int = 0) -> int:
    """Canonical CAGR-fit history width in years for a given forecast horizon.

    Single policy shared by the fetch expansion (agent/planner.py
    ``_expand_forecast_history_window``) and the fit-window trim
    (agent/analyzer.py ``_trim_forecast_fit_window``) so every code path —
    analyzer-driven, evidence-plan, or router fallback — fits the same window
    and produces the same forecast for the same question (2026-07-10
    model-variance report: a fallback path fetched the full history and
    shifted the CAGR from 2.29% to 1.09%).
    """
    desired = max(5, min(8, int(horizon_years) + 2))
    if excluded_years_count:
        desired = min(10, max(8, desired + int(excluded_years_count)))
    return desired
