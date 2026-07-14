"""Gateway auth with optional public bearer-token support."""

from __future__ import annotations

import hmac
import logging
import re
import threading
import time
from typing import Literal, Optional

import jwt
from fastapi import HTTPException
from pydantic import BaseModel

from config import (
    ENABLE_PUBLIC_BEARER_AUTH,
    ENAI_AUTH_MODE,
    GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS,
    GATEWAY_ACTOR_ASSERTION_MODE,
    GATEWAY_SHARED_SECRET,
    SUPABASE_JWT_SECRET,
)

log = logging.getLogger("Enai")
_SAFE_ASSERTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HMAC_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ASSERTION_REPLAY_CACHE: dict[str, float] = {}
_ASSERTION_REPLAY_LOCK = threading.Lock()


class GatewayActorAssertion(BaseModel):
    """Verified identity context asserted by the trusted edge gateway."""

    actor_id: str
    session_id: str
    request_id: str
    issued_at: int


class CallerContext(BaseModel):
    """Normalized caller identity returned by authenticate_request()."""

    auth_mode: Literal["gateway", "public_bearer"]
    subject_id: str
    is_admin: bool = False
    actor_id: Optional[str] = None
    session_id: Optional[str] = None
    actor_assertion_verified: bool = False


def verify_gateway_actor_assertion(
    *,
    request_id: str,
    contract_version: str,
    actor_id: Optional[str],
    session_id: Optional[str],
    issued_at: Optional[str],
    signature: Optional[str],
    now_ts: Optional[float] = None,
) -> Optional[GatewayActorAssertion]:
    """Verify the P3 edge assertion, returning ``None`` only in rollout mode.

    The HMAC binds the contract version, authoritative request ID, actor,
    Supabase session, and issue time. A partial, malformed, stale, future, or
    tampered assertion always fails closed, including while rollout mode is
    ``optional``.
    """
    raw_fields = (actor_id, session_id, issued_at, signature)
    if all(value is None for value in raw_fields):
        if GATEWAY_ACTOR_ASSERTION_MODE == "required":
            raise HTTPException(status_code=401, detail="Gateway actor assertion required")
        return None
    if any(value is None for value in raw_fields):
        raise HTTPException(status_code=401, detail="Invalid gateway actor assertion")

    normalized_actor = str(actor_id).strip()
    normalized_session = str(session_id).strip()
    normalized_issued_at = str(issued_at).strip()
    normalized_signature = str(signature).strip()
    if (
        not _SAFE_ASSERTION_ID_RE.fullmatch(request_id)
        or not _SAFE_ASSERTION_ID_RE.fullmatch(normalized_actor)
        or not _SAFE_ASSERTION_ID_RE.fullmatch(normalized_session)
        or not _HMAC_SHA256_HEX_RE.fullmatch(normalized_signature)
    ):
        raise HTTPException(status_code=401, detail="Invalid gateway actor assertion")
    try:
        issued_at_seconds = int(normalized_issued_at)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid gateway actor assertion") from exc

    current_time = time.time() if now_ts is None else now_ts
    if abs(current_time - issued_at_seconds) > GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS:
        raise HTTPException(status_code=401, detail="Expired gateway actor assertion")

    payload = "\n".join(
        (
            contract_version,
            request_id,
            normalized_actor,
            normalized_session,
            str(issued_at_seconds),
        )
    )
    expected_signature = hmac.new(
        str(GATEWAY_SHARED_SECRET or "").encode("utf-8"),
        payload.encode("utf-8"),
        "sha256",
    ).hexdigest()
    if not GATEWAY_SHARED_SECRET or not hmac.compare_digest(
        expected_signature,
        normalized_signature.lower(),
    ):
        raise HTTPException(status_code=401, detail="Invalid gateway actor assertion")

    # The P3.B entitlement ledger is the cross-process idempotency authority.
    # This bounded local guard additionally stops the same signed operation
    # from starting twice on the currently required single backend replica.
    replay_key = hmac.new(
        str(GATEWAY_SHARED_SECRET).encode("utf-8"),
        f"{request_id}\n{normalized_actor}\n{normalized_session}".encode("utf-8"),
        "sha256",
    ).hexdigest()
    with _ASSERTION_REPLAY_LOCK:
        expired_keys = [
            key for key, expires_at in _ASSERTION_REPLAY_CACHE.items()
            if expires_at <= current_time
        ]
        for key in expired_keys:
            del _ASSERTION_REPLAY_CACHE[key]
        if replay_key in _ASSERTION_REPLAY_CACHE:
            raise HTTPException(status_code=409, detail="Gateway assertion replayed")
        _ASSERTION_REPLAY_CACHE[replay_key] = (
            current_time + GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS
        )

    return GatewayActorAssertion(
        actor_id=normalized_actor,
        session_id=normalized_session,
        request_id=request_id,
        issued_at=issued_at_seconds,
    )


def _verify_supabase_token(token: str) -> Optional[str]:
    """Verify a Supabase JWT and return the user's subject id (UUID), or None."""
    if not SUPABASE_JWT_SECRET:
        return None
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"require": ["sub", "exp"]},
            audience="authenticated",
        )
        subject = str(payload.get("sub") or "").strip()
        return subject if subject else None
    except jwt.ExpiredSignatureError:
        log.debug("Bearer token expired")
        return None
    except jwt.InvalidTokenError as exc:
        log.debug("Bearer token invalid: %s", exc)
        return None


def authenticate_request(
    *,
    x_app_key: Optional[str] = None,
    authorization: Optional[str] = None,
    request_id: str = "",
    contract_version: str = "chat-gateway-v1",
    x_actor_id: Optional[str] = None,
    x_actor_session_id: Optional[str] = None,
    x_actor_issued_at: Optional[str] = None,
    x_actor_signature: Optional[str] = None,
) -> CallerContext:
    """Authenticate a request via gateway secret or Supabase bearer token.

    Precedence:
        1. Valid gateway secret in X-App-Key → gateway caller
        2. Valid Supabase bearer token in Authorization → public caller
        3. Reject with 401

    Uses constant-time comparison for the gateway secret.
    """
    # 1. Gateway secret (constant-time comparison)
    if x_app_key and GATEWAY_SHARED_SECRET:
        if hmac.compare_digest(x_app_key, GATEWAY_SHARED_SECRET):
            assertion = verify_gateway_actor_assertion(
                request_id=request_id,
                contract_version=contract_version,
                actor_id=x_actor_id,
                session_id=x_actor_session_id,
                issued_at=x_actor_issued_at,
                signature=x_actor_signature,
            )
            return CallerContext(
                auth_mode="gateway",
                subject_id=(f"user:{assertion.actor_id}" if assertion else "gateway:internal"),
                actor_id=(assertion.actor_id if assertion else None),
                session_id=(assertion.session_id if assertion else None),
                actor_assertion_verified=assertion is not None,
            )

    # 2. Bearer token
    if ENABLE_PUBLIC_BEARER_AUTH and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            user_id = _verify_supabase_token(token.strip())
            if user_id:
                return CallerContext(
                    auth_mode="public_bearer",
                    subject_id=f"user:{user_id}",
                    actor_id=user_id,
                )
    elif authorization:
        log.debug("Bearer auth ignored because it is not enabled (ENAI_AUTH_MODE=%s)", ENAI_AUTH_MODE)

    raise HTTPException(status_code=401, detail="Unauthorized")
