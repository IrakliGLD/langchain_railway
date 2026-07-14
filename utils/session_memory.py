"""
Session-bound conversation memory utilities.

This module stores recent Q/A exchanges in-process and binds them to a
server-signed session token so caller-provided history is not trusted.
"""
from __future__ import annotations

import hashlib
import hmac
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

from config import SESSION_HISTORY_MAX_TURNS, SESSION_IDLE_TTL_SECONDS

_SESSION_STORE: Dict[str, Dict[str, object]] = {}
_SESSION_LOCK = threading.Lock()


def _actor_binding(actor_id: Optional[str]) -> str:
    normalized = str(actor_id or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _sign_session_id(session_id: str, secret: str, actor_id: Optional[str] = None) -> str:
    """Sign a session token, binding it to an actor whenever one is known."""
    normalized_actor = str(actor_id or "").strip()
    payload = (
        f"actor-session-v1\n{normalized_actor}\n{session_id}"
        if normalized_actor
        else session_id
    )
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _derive_authoritative_session_id(actor_id: str, external_session_id: str) -> str:
    """Create an opaque stable key for a verified edge actor/session pair."""
    payload = f"gateway-session-v1\n{actor_id}\n{external_session_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _new_session_payload(now_ts: float, actor_id: Optional[str]) -> Dict[str, object]:
    return {
        "updated_at": now_ts,
        "history": [],
        "actor_binding": _actor_binding(actor_id),
    }


def _cleanup_expired(now_ts: float) -> None:
    expired: List[str] = []
    for session_id, payload in list(_SESSION_STORE.items()):
        try:
            updated_at = float(payload.get("updated_at", 0.0))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            updated_at = 0.0
        if now_ts - updated_at > SESSION_IDLE_TTL_SECONDS:
            expired.append(session_id)
    for session_id in expired:
        _SESSION_STORE.pop(session_id, None)


def issue_session_token(secret: str, actor_id: Optional[str] = None) -> Tuple[str, str]:
    """Create a new signed session token and initialize storage."""
    session_id = uuid.uuid4().hex
    signature = _sign_session_id(session_id, secret, actor_id)
    token = f"{session_id}.{signature}"
    with _SESSION_LOCK:
        _SESSION_STORE[session_id] = _new_session_payload(time.time(), actor_id)
    return session_id, token


def resolve_session_token(
    token: Optional[str],
    secret: str,
    actor_id: Optional[str] = None,
) -> Optional[str]:
    """Return the session id only when its signature matches the actor."""
    if not token:
        return None
    try:
        session_id, signature = token.split(".", 1)
    except ValueError:
        return None
    if not session_id or not signature:
        return None
    expected = _sign_session_id(session_id, secret, actor_id)
    if not hmac.compare_digest(expected, signature):
        return None
    return session_id


def get_or_issue_session(
    token: Optional[str],
    secret: str,
    *,
    actor_id: Optional[str] = None,
    authoritative_session_id: Optional[str] = None,
) -> Tuple[str, str, bool]:
    """
    Resolve existing session token or create a new one.

    Returns:
        (session_id, session_token, reused_existing)
    """
    now_ts = time.time()
    with _SESSION_LOCK:
        _cleanup_expired(now_ts)

        if authoritative_session_id:
            normalized_actor = str(actor_id or "").strip()
            normalized_external_session = str(authoritative_session_id).strip()
            if not normalized_actor or not normalized_external_session:
                raise ValueError("An authoritative session requires a verified actor")
            session_id = _derive_authoritative_session_id(
                normalized_actor,
                normalized_external_session,
            )
            existing = _SESSION_STORE.get(session_id)
            if existing and existing.get("actor_binding") != _actor_binding(normalized_actor):
                raise ValueError("Session actor binding mismatch")
            payload = existing or _new_session_payload(now_ts, normalized_actor)
            payload["updated_at"] = now_ts
            _SESSION_STORE[session_id] = payload
            signature = _sign_session_id(session_id, secret, normalized_actor)
            return session_id, f"{session_id}.{signature}", existing is not None

        session_id = resolve_session_token(token, secret, actor_id)
        if session_id:
            expected_binding = _actor_binding(actor_id)
            payload = _SESSION_STORE.get(session_id)
            if payload is None:
                payload = _new_session_payload(now_ts, actor_id)
                _SESSION_STORE[session_id] = payload
            elif payload.get("actor_binding", "") != expected_binding:
                session_id = None
            if session_id:
                payload["updated_at"] = now_ts
                signature = _sign_session_id(session_id, secret, actor_id)
                return session_id, f"{session_id}.{signature}", True

        session_id = uuid.uuid4().hex
        signature = _sign_session_id(session_id, secret, actor_id)
        session_token = f"{session_id}.{signature}"
        _SESSION_STORE[session_id] = _new_session_payload(now_ts, actor_id)
        return session_id, session_token, False


def get_history(
    session_id: str,
    *,
    actor_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return stored history only when the optional actor binding matches."""
    with _SESSION_LOCK:
        payload = _SESSION_STORE.get(session_id)
        if not payload or payload.get("actor_binding", "") != _actor_binding(actor_id):
            return []
        history = payload.get("history", [])
        if not isinstance(history, list):
            return []
        return [dict(item) for item in history if isinstance(item, dict)]


def seed_history(
    session_id: str,
    turns: List[Dict[str, str]],
    *,
    actor_id: Optional[str] = None,
) -> None:
    """Write initial turns into a session that has no history yet.

    This is used to bridge edge-function-provided DB history into the
    in-process session store so that subsequent ``append_exchange`` calls
    accumulate *on top of* the seed rather than starting from scratch.

    If the session already has history, this is a no-op.
    """
    if not turns:
        return
    now_ts = time.time()
    with _SESSION_LOCK:
        payload = _SESSION_STORE.setdefault(
            session_id,
            _new_session_payload(now_ts, actor_id),
        )
        if payload.get("actor_binding", "") != _actor_binding(actor_id):
            raise ValueError("Session actor binding mismatch")
        existing = payload.get("history")
        if existing:
            return  # already populated — don't overwrite
        payload["history"] = [
            {"question": str(t.get("question", "")), "answer": str(t.get("answer", ""))}
            for t in turns[:SESSION_HISTORY_MAX_TURNS]
        ]
        payload["updated_at"] = now_ts


def append_exchange(
    session_id: str,
    question: str,
    answer: str,
    *,
    actor_id: Optional[str] = None,
) -> None:
    """Append Q/A pair and keep only the last N turns."""
    now_ts = time.time()
    with _SESSION_LOCK:
        payload = _SESSION_STORE.setdefault(
            session_id,
            _new_session_payload(now_ts, actor_id),
        )
        if payload.get("actor_binding", "") != _actor_binding(actor_id):
            raise ValueError("Session actor binding mismatch")
        history = payload.get("history")
        if not isinstance(history, list):
            history = []
            payload["history"] = history

        history.append({
            "question": str(question or ""),
            "answer": str(answer or ""),
        })
        if len(history) > SESSION_HISTORY_MAX_TURNS:
            del history[:-SESSION_HISTORY_MAX_TURNS]
        payload["updated_at"] = now_ts


# Contract continuity (architecture §3.2): the previous turn's routed-contract
# snapshot rides on the same session record and TTL as history.
_MAX_CONTRACT_SNAPSHOT_CHARS = 4000


def set_last_contract(
    session_id: str,
    snapshot_json: str,
    *,
    actor_id: Optional[str] = None,
) -> None:
    """Store the last authoritative routed-contract snapshot for the session."""
    snapshot = str(snapshot_json or "")
    if not session_id or not snapshot or len(snapshot) > _MAX_CONTRACT_SNAPSHOT_CHARS:
        return
    now_ts = time.time()
    with _SESSION_LOCK:
        payload = _SESSION_STORE.setdefault(
            session_id,
            _new_session_payload(now_ts, actor_id),
        )
        if payload.get("actor_binding", "") != _actor_binding(actor_id):
            raise ValueError("Session actor binding mismatch")
        payload["last_contract"] = snapshot
        payload["updated_at"] = now_ts


def get_last_contract(session_id: str, *, actor_id: Optional[str] = None) -> str:
    """Return the stored contract snapshot, or "" when absent."""
    if not session_id:
        return ""
    with _SESSION_LOCK:
        payload = _SESSION_STORE.get(session_id)
        if not payload or payload.get("actor_binding", "") != _actor_binding(actor_id):
            return ""
        return str(payload.get("last_contract", "") or "")
