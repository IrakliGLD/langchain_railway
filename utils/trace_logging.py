"""Structured trace logging helpers for per-request observability."""

from __future__ import annotations

import json
import logging
from typing import Any

from config import ENABLE_TRACE_DEBUG_ARTIFACTS, TRACE_MAX_LIST_ITEMS, TRACE_TEXT_MAX_CHARS


def _truncate_text(value: str, *, max_chars: int = TRACE_TEXT_MAX_CHARS) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _sanitize_trace_value(value: Any, *, debug: bool = False, depth: int = 0) -> Any:
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json")
        except TypeError:
            value = value.model_dump()
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            value = value.to_dict()
        except Exception:
            value = str(value)

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        limit = TRACE_MAX_LIST_ITEMS if not debug else max(TRACE_MAX_LIST_ITEMS, 20)
        sanitized = [_sanitize_trace_value(item, debug=debug, depth=depth + 1) for item in items[:limit]]
        if len(items) > limit:
            sanitized.append(f"...(+{len(items) - limit} more)")
        return sanitized
    if isinstance(value, dict):
        items = list(value.items())
        limit = TRACE_MAX_LIST_ITEMS if not debug else max(TRACE_MAX_LIST_ITEMS, 20)
        sanitized = {
            str(key): _sanitize_trace_value(val, debug=debug, depth=depth + 1)
            for key, val in items[:limit]
        }
        if len(items) > limit:
            sanitized["__truncated__"] = f"+{len(items) - limit} more"
        return sanitized
    return _truncate_text(str(value))


def trace_detail(log: logging.Logger, ctx, stage: str, event: str, *, debug: bool = False, **extra: Any) -> None:
    """Emit a structured JSON trace line tied to a request trace/session id."""

    if debug and not ENABLE_TRACE_DEBUG_ARTIFACTS:
        return

    payload = {
        "trace_id": getattr(ctx, "trace_id", ""),
        "session_id": getattr(ctx, "session_id", ""),
        "stage": stage,
        "event": event,
        "debug": bool(debug),
    }
    if extra:
        payload["extra"] = {
            str(key): _sanitize_trace_value(value, debug=debug)
            for key, value in extra.items()
        }
    log.info("TRACE_DETAIL %s", json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str))
