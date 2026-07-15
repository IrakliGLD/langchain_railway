"""Privacy-preserving helpers for production observability.

The default policy is deliberately asymmetric: request/span correlation IDs
and code-owned categorical labels may remain readable, while actor/session
identifiers and arbitrary text are reduced to keyed fingerprints.  This keeps
logs useful for incident correlation without turning Railway/application logs
into a second store for user questions, answers, SQL, tokens, or PII.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from typing import Any

from config import SESSION_SIGNING_SECRET, TRACE_MAX_LIST_ITEMS

_SAFE_CORRELATION_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_./:-]{1,96}$")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.IGNORECASE)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_CONTENT_MARKER_RE = re.compile(
    r"(?i)(?P<prefix>\b(?:query|answer|prompt|sql|preview|history)\s*(?:=|:))(?P<value>[^\n]+)"
)

_PUBLIC_CORRELATION_FIELDS = frozenset(
    {"request_id", "trace_id", "span_id", "parent_span_id"}
)
_PRIVATE_IDENTIFIER_MARKERS = (
    "actor",
    "client_ip",
    "email",
    "rate_limit_key",
    "session",
    "subject",
    "token",
    "user",
)
_SAFE_LABEL_FIELDS = frozenset(
    {
        "action",
        "answer_kind",
        "candidate_topics",
        "chart_source",
        "classification",
        "data_source",
        "error_class",
        "error_type",
        "event",
        "event_type",
        "mode",
        "model",
        "operation",
        "preferred_path",
        "provider",
        "query_type",
        "render_style",
        "signals",
        "source",
        "stage",
        "summary_source",
        "tool",
        "tool_name",
        "trigger",
    }
)


def _hash_key() -> bytes:
    configured = os.getenv("ENAI_LOG_HASH_KEY", "").strip()
    material = configured or SESSION_SIGNING_SECRET or "enai-local-observability-only"
    return material.encode("utf-8")


def hash_private_identifier(value: object, *, namespace: str) -> str:
    """Return a stable deployment-local keyed fingerprint, never raw input."""

    text = str(value or "")
    if not text:
        return ""
    digest = hmac.new(
        _hash_key(),
        f"{namespace}\n{text}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:20]
    return f"hmac-sha256:{digest}"


def summarize_private_text(value: object, *, namespace: str) -> dict[str, Any]:
    """Describe arbitrary text without retaining any of its contents."""

    text = str(value or "")
    return {
        "redacted": True,
        "length": len(text),
        "fingerprint": hash_private_identifier(text, namespace=namespace),
    }


def _is_private_identifier_field(field: str) -> bool:
    normalized = field.strip().lower()
    return any(marker in normalized for marker in _PRIVATE_IDENTIFIER_MARKERS)


def sanitize_observability_value(field: str, value: Any, *, depth: int = 0) -> Any:
    """Project a value into the default-deny observability contract.

    Unknown strings are private text.  Only explicitly named correlation IDs
    and code-owned categorical fields remain readable.  Containers are bounded
    recursively so a nested DTO cannot bypass the policy.
    """

    field_name = str(field or "value").strip().lower() or "value"
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json")
        except TypeError:
            value = value.model_dump()
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            value = value.to_dict()
        except Exception:
            return summarize_private_text(type(value).__name__, namespace=field_name)

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if field_name in _PUBLIC_CORRELATION_FIELDS and _SAFE_CORRELATION_RE.fullmatch(value):
            return value
        if _is_private_identifier_field(field_name):
            namespace = next(
                (marker for marker in _PRIVATE_IDENTIFIER_MARKERS if marker in field_name),
                field_name,
            )
            return hash_private_identifier(value, namespace=namespace)
        if field_name in _SAFE_LABEL_FIELDS and _SAFE_LABEL_RE.fullmatch(value):
            return value
        return summarize_private_text(value, namespace=field_name)

    limit = TRACE_MAX_LIST_ITEMS
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        sanitized = [
            sanitize_observability_value(field_name, item, depth=depth + 1)
            for item in items[:limit]
        ]
        if len(items) > limit:
            sanitized.append({"truncated_items": len(items) - limit})
        return sanitized

    if isinstance(value, dict):
        items = list(value.items())
        sanitized = {
            str(key): sanitize_observability_value(str(key), item, depth=depth + 1)
            for key, item in items[:limit]
        }
        if len(items) > limit:
            sanitized["__truncated_items__"] = len(items) - limit
        return sanitized

    return summarize_private_text(type(value).__name__, namespace=field_name)


def sanitize_security_details(details: dict[str, Any]) -> dict[str, Any]:
    """Sanitize caller-provided details before serializing a security event."""

    return {
        str(key): sanitize_observability_value(str(key), value)
        for key, value in details.items()
    }


def redact_log_message(message: object) -> str:
    """Redact recognizable PII/secrets and content-labelled log fragments."""

    text = str(message or "")

    def _content_replacement(match: re.Match[str]) -> str:
        fingerprint = hash_private_identifier(match.group("value"), namespace="log_content")
        return f"{match.group('prefix')}[redacted:{fingerprint}]"

    text = _CONTENT_MARKER_RE.sub(_content_replacement, text)
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _JWT_RE.sub("[redacted-token]", text)
    text = _BEARER_RE.sub("Bearer [redacted-token]", text)
    text = _UUID_RE.sub("[redacted-uuid]", text)
    return text


class PrivacyLogFilter(logging.Filter):
    """Final safety net for the application logger.

    Call sites should still log categorical summaries/hashes directly.  This
    filter protects against a future regression or exception argument that
    contains recognizable PII, credentials, or explicitly labelled content.
    """

    def filter(self, record) -> bool:
        if isinstance(record.args, dict):
            record.args = {
                key: type(value).__name__ if isinstance(value, BaseException) else value
                for key, value in record.args.items()
            }
        elif isinstance(record.args, tuple):
            record.args = tuple(
                type(value).__name__ if isinstance(value, BaseException) else value
                for value in record.args
            )
        rendered = record.getMessage()
        if record.exc_info:
            error_class = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
            rendered = f"{rendered} error_class={error_class}"
            record.exc_info = None
            record.exc_text = None
        record.msg = redact_log_message(rendered)
        record.args = ()
        return True
