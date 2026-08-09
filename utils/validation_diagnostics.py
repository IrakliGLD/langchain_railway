"""Turn a pydantic ValidationError into telemetry that carries no data.

Error locations are field names and indices from our own contracts. The
``input`` beside them is the rejected value itself, which is why it is never
read.

A field path is not always available. A ``model_validator`` rejection carries
an empty ``loc`` — pydantic has no field to blame — so a contract whose rules
all live in model validators, as ``ReportGenerationCheckpoint``'s identity
rules do, could only ever report an empty list. For those, the rule's own
message is the offender, and it is authored here rather than by the data:
pydantic's ``msg`` restates the constraint, never the input.
"""

from __future__ import annotations

_MAXIMUM_REPORTED_LOCATIONS = 8
_MAXIMUM_REPORTED_RULES = 4
_MAXIMUM_RULE_CHARS = 200
_VALUE_ERROR_PREFIX = "Value error, "


def _validation_entries(exc: Exception) -> list[dict]:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return []
    try:
        raw = errors()
    except Exception:  # pragma: no cover - defensive
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def _entry_location(entry: dict) -> str:
    return ".".join(
        str(part) for part in entry.get("loc", ()) if part != ""
    )


def validation_error_locations(exc: Exception) -> list[str]:
    """Return the dotted field paths a ValidationError rejected."""

    located: list[str] = []
    for entry in _validation_entries(exc)[:_MAXIMUM_REPORTED_LOCATIONS]:
        location = _entry_location(entry)
        if location and location not in located:
            located.append(location)
    return located


def validation_error_rules(exc: Exception) -> list[str]:
    """Return the model-level rules a ValidationError rejected.

    Only entries with no field path are reported: a located error is already
    named by :func:`validation_error_locations`, and repeating its message
    would add nothing. Bounded in count and length so one rejection cannot
    dominate a log line.
    """

    rules: list[str] = []
    for entry in _validation_entries(exc):
        if _entry_location(entry):
            continue
        message = " ".join(str(entry.get("msg") or "").split())
        if message.startswith(_VALUE_ERROR_PREFIX):
            message = message[len(_VALUE_ERROR_PREFIX):]
        message = message[:_MAXIMUM_RULE_CHARS]
        if message and message not in rules:
            rules.append(message)
        if len(rules) >= _MAXIMUM_REPORTED_RULES:
            break
    return rules
