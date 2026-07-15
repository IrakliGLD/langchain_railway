"""Structured trace logging helpers for per-request observability."""

from __future__ import annotations

import json
import logging
from typing import Any

from config import ENABLE_TRACE_DEBUG_ARTIFACTS
from utils.privacy_logging import hash_private_identifier, sanitize_observability_value


def _sanitize_trace_value(value: Any, *, debug: bool = False, depth: int = 0) -> Any:
    # Compatibility wrapper retained for callers/tests that imported the old
    # helper.  Debug mode no longer disables privacy controls.
    del debug, depth
    return sanitize_observability_value("value", value)


def trace_detail(log: logging.Logger, ctx, stage: str, event: str, *, debug: bool = False, **extra: Any) -> None:
    """Emit a structured JSON trace line tied to a request trace/session id."""

    if debug and not ENABLE_TRACE_DEBUG_ARTIFACTS:
        return

    payload = {
        "trace_id": getattr(ctx, "trace_id", ""),
        "session_id": hash_private_identifier(
            getattr(ctx, "session_id", ""), namespace="session"
        ),
        "stage": stage,
        "event": event,
        "debug": bool(debug),
    }
    if extra:
        payload["extra"] = {
            str(key): sanitize_observability_value(str(key), value)
            for key, value in extra.items()
        }
    log.info("TRACE_DETAIL %s", json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str))
