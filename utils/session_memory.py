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


def _sign_session_id(session_id: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), session_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest


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


def issue_session_token(secret: str) -> Tuple[str, str]:
    """Create a new signed session token and initialize storage."""
    session_id = uuid.uuid4().hex
    signature = _sign_session_id(session_id, secret)
    token = f"{session_id}.{signature}"
    with _SESSION_LOCK:
        _SESSION_STORE[session_id] = {
            "updated_at": time.time(),
            "history": [],
        }
    return session_id, token


def resolve_session_token(token: Optional[str], secret: str) -> Optional[str]:
    """Return session_id only when token signature is valid."""
    if not token:
        return None
    try:
        session_id, signature = token.split(".", 1)
    except ValueError:
        return None
    if not session_id or not signature:
        return None
    expected = _sign_session_id(session_id, secret)
    if not hmac.compare_digest(expected, signature):
        return None
    return session_id


def get_or_issue_session(token: Optional[str], secret: str) -> Tuple[str, str, bool]:
    """
    Resolve existing session token or create a new one.

    Returns:
        (session_id, session_token, reused_existing)
    """
    now_ts = time.time()
    with _SESSION_LOCK:
        _cleanup_expired(now_ts)

        session_id = resolve_session_token(token, secret)
        if session_id:
            payload = _SESSION_STORE.setdefault(
                session_id,
                {"updated_at": now_ts, "history": []},
            )
            payload["updated_at"] = now_ts
            return session_id, f"{session_id}.{_sign_session_id(session_id, secret)}", True

        session_id = uuid.uuid4().hex
        signature = _sign_session_id(session_id, secret)
        session_token = f"{session_id}.{signature}"
        _SESSION_STORE[session_id] = {
            "updated_at": now_ts,
            "history": [],
        }
        return session_id, session_token, False


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return a copy of the stored history for a session."""
    with _SESSION_LOCK:
        payload = _SESSION_STORE.get(session_id)
        if not payload:
            return []
        history = payload.get("history", [])
        if not isinstance(history, list):
            return []
        return [dict(item) for item in history if isinstance(item, dict)]


def append_exchange(session_id: str, question: str, answer: str) -> None:
    """Append Q/A pair and keep only the last N turns."""
    now_ts = time.time()
    with _SESSION_LOCK:
        payload = _SESSION_STORE.setdefault(
            session_id,
            {"updated_at": now_ts, "history": []},
        )
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
