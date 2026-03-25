"""Gateway auth with optional public bearer-token support."""

from __future__ import annotations

import hmac
import logging
from typing import Literal, Optional

import jwt
from fastapi import HTTPException
from pydantic import BaseModel

from config import (
    ENABLE_PUBLIC_BEARER_AUTH,
    ENAI_AUTH_MODE,
    GATEWAY_SHARED_SECRET,
    SUPABASE_JWT_SECRET,
)

log = logging.getLogger("Enai")


class CallerContext(BaseModel):
    """Normalized caller identity returned by authenticate_request()."""

    auth_mode: Literal["gateway", "public_bearer"]
    subject_id: str
    is_admin: bool = False


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
            return CallerContext(
                auth_mode="gateway",
                subject_id="gateway:internal",
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
                )
    elif authorization:
        log.debug("Bearer auth ignored because it is not enabled (ENAI_AUTH_MODE=%s)", ENAI_AUTH_MODE)

    raise HTTPException(status_code=401, detail="Unauthorized")
