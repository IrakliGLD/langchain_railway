"""Fail-closed parsing for browser CORS origin configuration."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

log = logging.getLogger("Enai")


def _canonical_origin(candidate: str) -> str | None:
    try:
        parsed = urlsplit(candidate)
        # Accessing ``port`` also validates its syntax and range.
        parsed.port
    except (TypeError, ValueError):
        return None

    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if not parsed.hostname or parsed.username or parsed.password:
        return None
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        return None

    # Lower-case the host as well as the scheme. Browsers send ``Origin`` with a
    # lower-cased host, so an entry like ``https://Dashboard.Example.COM`` would
    # otherwise never match a real request -- a CORS failure that reads as the
    # configuration being ignored. Case-folding here also makes de-duplication
    # work across differently-cased spellings of one origin. Safe on the whole
    # netloc: userinfo is rejected above and ports are digits.
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def parse_allowed_origins(raw_origins: str) -> list[str]:
    """Return canonical HTTP(S) origins and discard URLs or unsafe entries."""
    origins: list[str] = []
    seen: set[str] = set()
    invalid_count = 0

    for raw_candidate in str(raw_origins or "").split(","):
        candidate = raw_candidate.strip()
        if not candidate:
            continue
        origin = _canonical_origin(candidate)
        if origin is None:
            invalid_count += 1
            continue
        if origin not in seen:
            seen.add(origin)
            origins.append(origin)

    if invalid_count:
        log.warning("Ignoring invalid CORS origins: count=%d", invalid_count)
    return origins
