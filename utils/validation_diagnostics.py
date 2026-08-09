"""Turn a pydantic ValidationError into telemetry that carries no data.

Error locations are field names and indices from our own contracts. The
messages and inputs beside them are the rejected values themselves, which is
why only ``loc`` is ever read.
"""

from __future__ import annotations

_MAXIMUM_REPORTED_LOCATIONS = 8


def validation_error_locations(exc: Exception) -> list[str]:
    """Return the dotted field paths a ValidationError rejected."""

    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return []
    try:
        raw = errors()
    except Exception:  # pragma: no cover - defensive
        return []
    located: list[str] = []
    for entry in list(raw)[:_MAXIMUM_REPORTED_LOCATIONS]:
        location = ".".join(
            str(part) for part in (entry or {}).get("loc", ()) if part != ""
        )
        if location and location not in located:
            located.append(location)
    return located
