"""Actor-bound, content-free provider execution reconciliation."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from utils.request_deadline import current_request_execution_scope

log = logging.getLogger("Enai")


class ProviderDeliveryDisposition(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    TIMED_OUT = "timed_out"
    PERMANENT_FAILURE = "permanent_failure"


class ProviderExecutionError(RuntimeError):
    """Typed provider failure used to decide whether fallback is safe."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        stage: str,
        disposition: ProviderDeliveryDisposition,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.stage = stage
        self.disposition = disposition

    @property
    def safe_to_retry(self) -> bool:
        """Same-provider replay is safe only when delivery was rejected pre-send."""
        return self.disposition == ProviderDeliveryDisposition.REJECTED

    @property
    def safe_to_fallback(self) -> bool:
        """Cross-provider fallback safety (incident 2026-07-17).

        A locally-enforced timeout means OUR client abandoned the call: the
        original provider may still bill the attempt (operator cost), but a
        fresh attempt on a DIFFERENT provider is a distinct ledger claim, so
        the no-replay policy is preserved while the OpenAI safety net keeps
        the request answerable when the primary provider is slow. Truly
        ambiguous transport failures stay non-retryable.
        """
        return self.disposition in {
            ProviderDeliveryDisposition.REJECTED,
            ProviderDeliveryDisposition.TIMED_OUT,
        }


@dataclass(frozen=True, slots=True)
class ProviderAttemptToken:
    key: str
    provider: str
    stage: str
    request_id: str
    actor_binding: str
    bound: bool


@dataclass(slots=True)
class _AttemptRecord:
    state: ProviderDeliveryDisposition
    updated_monotonic: float


_ATTEMPT_TTL_SECONDS = 600.0
_ATTEMPT_MAX_ENTRIES = 4096
_ATTEMPTS: dict[str, _AttemptRecord] = {}
_ATTEMPT_LOCK = threading.Lock()


def _cleanup_locked(now: float) -> None:
    expired = [key for key, record in _ATTEMPTS.items() if now - record.updated_monotonic >= _ATTEMPT_TTL_SECONDS]
    for key in expired:
        _ATTEMPTS.pop(key, None)
    while len(_ATTEMPTS) >= _ATTEMPT_MAX_ENTRIES:
        oldest = min(_ATTEMPTS, key=lambda key: _ATTEMPTS[key].updated_monotonic)
        _ATTEMPTS.pop(oldest, None)


def claim_provider_attempt(provider: str, stage: str) -> ProviderAttemptToken:
    """Claim one provider/stage execution for the bound actor/request."""
    scope = current_request_execution_scope()
    request_id = scope.request_id if scope is not None else ""
    actor_binding = scope.actor_binding if scope is not None else ""
    bound = bool(request_id and actor_binding)
    if not bound:
        return ProviderAttemptToken("", provider, stage, request_id, actor_binding, False)

    key = f"{actor_binding}|{request_id}|{provider}|{stage}"
    now = time.monotonic()
    with _ATTEMPT_LOCK:
        _cleanup_locked(now)
        existing = _ATTEMPTS.get(key)
        if existing is not None:
            disposition = (
                ProviderDeliveryDisposition.AMBIGUOUS
                if existing.state
                in {
                    ProviderDeliveryDisposition.STARTED,
                    ProviderDeliveryDisposition.AMBIGUOUS,
                }
                else existing.state
            )
            raise ProviderExecutionError(
                "provider execution already recorded for actor-bound request",
                provider=provider,
                stage=stage,
                disposition=disposition,
            )
        _ATTEMPTS[key] = _AttemptRecord(ProviderDeliveryDisposition.STARTED, now)

    log.info(
        "Provider execution attempt started. provider=%s stage=%s request_id=%s actor_binding=%s",
        provider,
        stage,
        request_id,
        actor_binding,
    )
    return ProviderAttemptToken(key, provider, stage, request_id, actor_binding, True)


def finish_provider_attempt(
    token: ProviderAttemptToken,
    disposition: ProviderDeliveryDisposition,
    *,
    failure_reason: str = "",
) -> None:
    if token.bound:
        with _ATTEMPT_LOCK:
            record = _ATTEMPTS.get(token.key)
            if record is not None:
                record.state = disposition
                record.updated_monotonic = time.monotonic()
    if failure_reason:
        log.info(
            "Provider execution attempt finished. provider=%s stage=%s request_id=%s actor_binding=%s disposition=%s failure_reason=%s",
            token.provider,
            token.stage,
            token.request_id,
            token.actor_binding,
            disposition.value,
            failure_reason,
        )
    else:
        log.info(
            "Provider execution attempt finished. provider=%s stage=%s request_id=%s actor_binding=%s disposition=%s",
            token.provider,
            token.stage,
            token.request_id,
            token.actor_binding,
            disposition.value,
        )


# Exception type names that identify a locally-enforced client timeout across
# the SDKs in use (openai/httpx/google-genai/stdlib). Matched against the
# error and its __cause__ chain because langchain wraps SDK errors.
_TIMEOUT_ERROR_TYPE_NAMES = frozenset({
    "apitimeouterror",       # openai.APITimeoutError
    "timeoutexception",      # httpx.TimeoutException
    "connecttimeout",        # httpx.ConnectTimeout
    "readtimeout",           # httpx.ReadTimeout / requests.ReadTimeout
    "writetimeout",          # httpx.WriteTimeout
    "pooltimeout",           # httpx.PoolTimeout
    "timeouterror",          # stdlib TimeoutError / asyncio.TimeoutError
    "deadlineexceeded",      # google.api_core.exceptions.DeadlineExceeded
})


def _is_timeout_error(error: BaseException) -> bool:
    seen = 0
    current: BaseException | None = error
    while current is not None and seen < 5:
        if isinstance(current, TimeoutError):
            return True
        if type(current).__name__.lower() in _TIMEOUT_ERROR_TYPE_NAMES:
            return True
        current = current.__cause__ or current.__context__
        seen += 1
    return False


def classify_provider_failure(error: BaseException) -> ProviderDeliveryDisposition:
    """Conservatively distinguish rejected, timed-out, permanent, and ambiguous delivery."""
    if isinstance(error, ProviderExecutionError):
        return error.disposition
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None

    # A rate-limit response proves this attempt was rejected before execution.
    if status_code == 429:
        return ProviderDeliveryDisposition.REJECTED
    if status_code in {400, 401, 403, 404, 405, 422}:
        return ProviderDeliveryDisposition.PERMANENT_FAILURE
    # A locally-enforced timeout (incident 2026-07-17): our client gave up, so
    # a cross-provider fallback is safe even though the original provider may
    # still complete server-side. Distinct from generic ambiguity below.
    if status_code is None and _is_timeout_error(error):
        return ProviderDeliveryDisposition.TIMED_OUT
    # Network errors and 5xx responses can arrive after the provider accepted
    # work. Treat them as ambiguous and never blindly retry them.
    return ProviderDeliveryDisposition.AMBIGUOUS


# Enum-shaped tokens only (INVALID_ARGUMENT, API_KEY_INVALID, ...). Provider
# error MESSAGES can echo request content, so free text must never pass.
_SAFE_REASON_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,40}$")


def _error_info_reasons(details: Any) -> list[str]:
    """Pull google.rpc ErrorInfo ``reason`` constants out of an error payload."""
    if isinstance(details, dict):
        inner = details.get("error")
        details = inner.get("details") if isinstance(inner, dict) else details.get("details")
    reasons: list[str] = []
    if isinstance(details, list):
        for item in details[:8]:
            if isinstance(item, dict):
                reason = item.get("reason")
                if isinstance(reason, str) and _SAFE_REASON_RE.match(reason):
                    reasons.append(reason)
    return reasons


def extract_failure_reason(error: BaseException) -> str:
    """Best-effort content-free reason string for a provider failure.

    Added after the 2026-07-22 embedding outage: every Gemini query-embedding
    call was failing 400 for weeks of undiagnosable "provider call failed:
    ClientError" logs, because this module (correctly) refuses to log provider
    error messages. The compromise: emit only the HTTP status code plus
    strictly enum-shaped provider constants (e.g. ``INVALID_ARGUMENT``,
    ``API_KEY_INVALID``) validated by ``_SAFE_REASON_RE``, walking the cause
    chain the same bounded way as ``_is_timeout_error``. Returns "" when
    nothing safe is extractable.
    """
    status_code: int | None = None
    tokens: list[str] = []
    current: BaseException | None = error
    seen = 0
    while current is not None and seen < 5:
        if status_code is None:
            raw = getattr(current, "code", None)
            if raw is None:
                raw = getattr(current, "status_code", None)
            if raw is None:
                response = getattr(current, "response", None)
                raw = getattr(response, "status_code", None)
            try:
                status_code = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                status_code = None
        status = getattr(current, "status", None)
        if isinstance(status, str) and _SAFE_REASON_RE.match(status) and status not in tokens:
            tokens.append(status)
        for reason in _error_info_reasons(getattr(current, "details", None)):
            if reason not in tokens:
                tokens.append(reason)
        current = current.__cause__ or current.__context__
        seen += 1
    parts = ([str(status_code)] if status_code is not None else []) + tokens
    return "/".join(parts)


def wrap_provider_failure(
    error: BaseException,
    *,
    provider: str,
    stage: str,
    disposition: ProviderDeliveryDisposition | None = None,
) -> ProviderExecutionError:
    if isinstance(error, ProviderExecutionError):
        return error
    resolved = disposition or classify_provider_failure(error)
    reason = extract_failure_reason(error)
    suffix = f" [{reason}]" if reason else ""
    return ProviderExecutionError(
        f"provider call failed: {type(error).__name__}{suffix}",
        provider=provider,
        stage=stage,
        disposition=resolved,
    )


def reset_provider_attempts_for_tests() -> None:
    with _ATTEMPT_LOCK:
        _ATTEMPTS.clear()
