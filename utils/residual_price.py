"""Signals for deterministic residual-price calculations."""

from __future__ import annotations

import re

_NEGLIGIBLE_IMPORT_THRESHOLD_RE = re.compile(
    r"(?:less than|below|under|at most|no more than)\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def is_implied_ppa_cfd_price_query(query: str) -> bool:
    """Recognize the bounded, deterministic negligible-import approximation."""

    query_lower = str(query or "").strip().lower()
    if not query_lower:
        return False
    if not all(token in query_lower for token in ("balancing", "import", "ppa", "cfd")):
        return False
    if "share" not in query_lower:
        return False
    if not (re.search(r"\bregulated\b", query_lower) and re.search(r"\bderegulated\b", query_lower)):
        return False
    if not any(token in query_lower for token in ("weighted average", "average price", "weighted avg")):
        return False

    match = _NEGLIGIBLE_IMPORT_THRESHOLD_RE.search(query_lower)
    if not match:
        return False
    return float(match.group("value")) <= 1.0
