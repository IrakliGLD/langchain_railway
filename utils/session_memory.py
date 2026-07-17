"""Actor-bound, process-local conversation continuity and turn ownership."""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

from config import (
    SESSION_HISTORY_MAX_ITEM_CHARS,
    SESSION_HISTORY_MAX_TURNS,
    SESSION_IDLE_TTL_SECONDS,
    SESSION_MAX_ENTRIES,
    SESSION_TURN_WAIT_TIMEOUT_MS,
)

_SESSION_STORE: Dict[str, Dict[str, object]] = {}
_SESSION_LOCK = threading.RLock()
_SESSION_CAPACITY_EVICTIONS = 0
_SESSION_EXPIRATIONS = 0
_SESSION_STALE_TOKEN_RENEWALS = 0
_SESSION_CAPACITY_REJECTIONS = 0
_SESSION_TURN_TIMEOUTS = 0
_SESSION_TURN_WAIT_MAX_SECONDS = 0.0

_MAX_CONTRACT_SNAPSHOT_CHARS = 4000


class SessionExpiredError(RuntimeError):
    """Raised when a mutation or turn targets a missing/expired session."""


class SessionTurnBusyError(TimeoutError):
    """Raised when a prior turn still owns the actor-bound session."""


class SessionCapacityExceededError(RuntimeError):
    """Raised when every bounded session slot is pinned by active work."""


def _actor_binding(actor_id: Optional[str]) -> str:
    normalized = str(actor_id or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _normalize_auth_mode(auth_mode: Optional[str]) -> str:
    return str(auth_mode or "").strip().lower()


def _history_text(value: object) -> str:
    return str(value or "")[:SESSION_HISTORY_MAX_ITEM_CHARS]


def _sign_session_id(
    session_id: str,
    secret: str,
    actor_id: Optional[str] = None,
    auth_mode: Optional[str] = None,
) -> str:
    """Sign a token over session, actor, and authentication-mode ownership."""
    normalized_actor = str(actor_id or "").strip()
    normalized_auth_mode = _normalize_auth_mode(auth_mode)
    payload = (
        "\n".join(
            (
                "actor-session-v2",
                normalized_auth_mode,
                normalized_actor,
                session_id,
            )
        )
        if normalized_actor or normalized_auth_mode
        else session_id
    )
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _derive_authoritative_session_id(
    actor_id: str,
    external_session_id: str,
    auth_mode: str,
) -> str:
    """Create an opaque stable key for a verified actor/session/auth tuple."""
    payload = "\n".join(
        (
            "authoritative-session-v2",
            auth_mode,
            actor_id,
            external_session_id,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _new_session_payload(
    now_ts: float,
    actor_id: Optional[str],
    auth_mode: Optional[str],
) -> Dict[str, object]:
    return {
        "updated_at": now_ts,
        "history": [],
        "actor_binding": _actor_binding(actor_id),
        "auth_mode": _normalize_auth_mode(auth_mode),
        "turn_lock": threading.Lock(),
        "turn_participants": 0,
    }


def _updated_at(payload: Dict[str, object]) -> float:
    try:
        return float(payload.get("updated_at", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _turn_participants(payload: Dict[str, object]) -> int:
    try:
        return max(0, int(payload.get("turn_participants", 0)))
    except (TypeError, ValueError):
        return 0


def _binding_matches(
    payload: Dict[str, object],
    actor_id: Optional[str],
    auth_mode: Optional[str],
) -> bool:
    return (
        payload.get("actor_binding", "") == _actor_binding(actor_id)
        and payload.get("auth_mode", "") == _normalize_auth_mode(auth_mode)
    )


def _cleanup_expired(now_ts: float) -> None:
    """Remove idle, unpinned records. Caller must hold ``_SESSION_LOCK``."""
    global _SESSION_EXPIRATIONS
    expired = [
        session_id
        for session_id, payload in _SESSION_STORE.items()
        if _turn_participants(payload) == 0
        and now_ts - _updated_at(payload) > SESSION_IDLE_TTL_SECONDS
    ]
    for session_id in expired:
        _SESSION_STORE.pop(session_id, None)
    _SESSION_EXPIRATIONS += len(expired)


def _ensure_capacity_for_new_session(now_ts: float) -> None:
    """Evict oldest inactive records until one bounded slot is available."""
    global _SESSION_CAPACITY_EVICTIONS, _SESSION_CAPACITY_REJECTIONS
    _cleanup_expired(now_ts)
    while len(_SESSION_STORE) >= SESSION_MAX_ENTRIES:
        inactive = [
            (session_id, payload)
            for session_id, payload in _SESSION_STORE.items()
            if _turn_participants(payload) == 0
        ]
        if not inactive:
            _SESSION_CAPACITY_REJECTIONS += 1
            raise SessionCapacityExceededError("Session capacity is occupied by active turns")
        oldest_session_id, _ = min(
            inactive,
            key=lambda item: (_updated_at(item[1]), item[0]),
        )
        _SESSION_STORE.pop(oldest_session_id, None)
        _SESSION_CAPACITY_EVICTIONS += 1


def _create_session_locked(
    session_id: str,
    now_ts: float,
    actor_id: Optional[str],
    auth_mode: Optional[str],
) -> None:
    _ensure_capacity_for_new_session(now_ts)
    _SESSION_STORE[session_id] = _new_session_payload(
        now_ts,
        actor_id,
        auth_mode,
    )


def _payload_for_access_locked(
    session_id: str,
    *,
    actor_id: Optional[str],
    auth_mode: Optional[str],
    mutation: bool,
) -> Dict[str, object] | None:
    _cleanup_expired(time.time())
    payload = _SESSION_STORE.get(session_id)
    if payload is None:
        if mutation:
            raise SessionExpiredError("Session is missing or expired")
        return None
    if not _binding_matches(payload, actor_id, auth_mode):
        if mutation:
            raise ValueError("Session actor binding or authentication-mode binding mismatch")
        return None
    return payload


def issue_session_token(
    secret: str,
    actor_id: Optional[str] = None,
    auth_mode: Optional[str] = None,
) -> Tuple[str, str]:
    """Create a signed, owned session token and initialize bounded storage."""
    now_ts = time.time()
    with _SESSION_LOCK:
        session_id = uuid.uuid4().hex
        while session_id in _SESSION_STORE:
            session_id = uuid.uuid4().hex
        _create_session_locked(session_id, now_ts, actor_id, auth_mode)
    signature = _sign_session_id(session_id, secret, actor_id, auth_mode)
    return session_id, f"{session_id}.{signature}"


def resolve_session_token(
    token: Optional[str],
    secret: str,
    actor_id: Optional[str] = None,
    auth_mode: Optional[str] = None,
) -> Optional[str]:
    """Return the token session id only when actor/auth ownership matches."""
    if not token:
        return None
    try:
        session_id, signature = token.split(".", 1)
    except ValueError:
        return None
    if not session_id or not signature:
        return None
    expected = _sign_session_id(session_id, secret, actor_id, auth_mode)
    if not hmac.compare_digest(expected, signature):
        return None
    return session_id


def get_or_issue_session(
    token: Optional[str],
    secret: str,
    *,
    actor_id: Optional[str] = None,
    authoritative_session_id: Optional[str] = None,
    auth_mode: Optional[str] = None,
) -> Tuple[str, str, bool]:
    """Resolve a live owned session or issue a distinct new session."""
    global _SESSION_STALE_TOKEN_RENEWALS
    now_ts = time.time()
    normalized_actor = str(actor_id or "").strip()
    normalized_auth_mode = _normalize_auth_mode(auth_mode)

    with _SESSION_LOCK:
        _cleanup_expired(now_ts)

        candidate_session_id = ""
        if token:
            candidate_session_id = token.split(".", 1)[0].strip()
        candidate = _SESSION_STORE.get(candidate_session_id)
        if candidate is not None and not _binding_matches(
            candidate,
            normalized_actor,
            normalized_auth_mode,
        ):
            raise ValueError("Session actor binding or authentication-mode binding mismatch")

        if authoritative_session_id:
            normalized_external_session = str(authoritative_session_id).strip()
            if not normalized_actor or not normalized_external_session:
                raise ValueError("An authoritative session requires a verified actor")
            session_id = _derive_authoritative_session_id(
                normalized_actor,
                normalized_external_session,
                normalized_auth_mode,
            )
            existing = _SESSION_STORE.get(session_id)
            if existing is not None and not _binding_matches(
                existing,
                normalized_actor,
                normalized_auth_mode,
            ):
                raise ValueError("Session actor binding or authentication-mode binding mismatch")
            if existing is None:
                _create_session_locked(
                    session_id,
                    now_ts,
                    normalized_actor,
                    normalized_auth_mode,
                )
            else:
                existing["updated_at"] = now_ts
            signature = _sign_session_id(
                session_id,
                secret,
                normalized_actor,
                normalized_auth_mode,
            )
            return session_id, f"{session_id}.{signature}", existing is not None


        session_id = resolve_session_token(
            token,
            secret,
            normalized_actor,
            normalized_auth_mode,
        )
        if session_id:
            payload = _SESSION_STORE.get(session_id)
            if payload is not None and _binding_matches(
                payload,
                normalized_actor,
                normalized_auth_mode,
            ):
                payload["updated_at"] = now_ts
                signature = _sign_session_id(
                    session_id,
                    secret,
                    normalized_actor,
                    normalized_auth_mode,
                )
                return session_id, f"{session_id}.{signature}", True
            # A cryptographically valid token whose process-local record was
            # expired, evicted, or lost on restart may receive a new identity,
            # but must never resurrect its old session id or continuity.
            _SESSION_STALE_TOKEN_RENEWALS += 1

        new_session_id = uuid.uuid4().hex
        while new_session_id in _SESSION_STORE:
            new_session_id = uuid.uuid4().hex
        _create_session_locked(
            new_session_id,
            now_ts,
            normalized_actor,
            normalized_auth_mode,
        )
        signature = _sign_session_id(
            new_session_id,
            secret,
            normalized_actor,
            normalized_auth_mode,
        )
        return new_session_id, f"{new_session_id}.{signature}", False


class SessionTurnLease:
    """One acquired per-session turn lock with idempotent release."""

    def __init__(self, session_id: str, turn_lock: threading.Lock) -> None:
        self._session_id = session_id
        self._turn_lock = turn_lock
        self._release_lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._turn_lock.release()
            with _SESSION_LOCK:
                payload = _SESSION_STORE.get(self._session_id)
                if payload is not None and payload.get("turn_lock") is self._turn_lock:
                    payload["turn_participants"] = max(
                        0,
                        _turn_participants(payload) - 1,
                    )
                    payload["updated_at"] = time.time()
            self._released = True


def _finish_turn_acquisition(
    session_id: str,
    turn_lock,
    *,
    timeout_seconds: float | None,
) -> SessionTurnLease:
    global _SESSION_TURN_TIMEOUTS, _SESSION_TURN_WAIT_MAX_SECONDS
    timeout = (
        SESSION_TURN_WAIT_TIMEOUT_MS / 1000.0
        if timeout_seconds is None
        else max(0.0, float(timeout_seconds))
    )
    started = time.monotonic()
    acquired = turn_lock.acquire(timeout=timeout)
    waited = max(0.0, time.monotonic() - started)
    with _SESSION_LOCK:
        _SESSION_TURN_WAIT_MAX_SECONDS = max(
            _SESSION_TURN_WAIT_MAX_SECONDS,
            waited,
        )
        if not acquired:
            current = _SESSION_STORE.get(session_id)
            if current is not None and current.get("turn_lock") is turn_lock:
                current["turn_participants"] = max(
                    0,
                    _turn_participants(current) - 1,
                )
            _SESSION_TURN_TIMEOUTS += 1
            raise SessionTurnBusyError("A prior session turn is still running")
        current = _SESSION_STORE.get(session_id)
        if current is None or current.get("turn_lock") is not turn_lock:
            turn_lock.release()
            raise SessionExpiredError("Session expired while acquiring turn ownership")
        current["updated_at"] = time.time()
    return SessionTurnLease(session_id, turn_lock)


def _pin_turn_locked(
    session_id: str,
    *,
    actor_id: Optional[str],
    auth_mode: Optional[str],
):
    payload = _payload_for_access_locked(
        session_id,
        actor_id=actor_id,
        auth_mode=auth_mode,
        mutation=True,
    )
    assert payload is not None
    turn_lock = payload.get("turn_lock")
    if not hasattr(turn_lock, "acquire") or not hasattr(turn_lock, "release"):
        raise RuntimeError("Session turn state is invalid")
    payload["turn_participants"] = _turn_participants(payload) + 1
    return turn_lock


def acquire_session_turn(
    session_id: str,
    *,
    actor_id: Optional[str] = None,
    auth_mode: Optional[str] = None,
    timeout_seconds: float | None = None,
) -> SessionTurnLease:
    """Acquire bounded exclusive ownership for one complete session turn."""
    with _SESSION_LOCK:
        turn_lock = _pin_turn_locked(
            session_id,
            actor_id=actor_id,
            auth_mode=auth_mode,
        )
    return _finish_turn_acquisition(
        session_id,
        turn_lock,
        timeout_seconds=timeout_seconds,
    )


def get_or_issue_session_turn(
    token: Optional[str],
    secret: str,
    *,
    actor_id: Optional[str] = None,
    authoritative_session_id: Optional[str] = None,
    auth_mode: Optional[str] = None,
    timeout_seconds: float | None = None,
) -> Tuple[str, str, bool, SessionTurnLease]:
    """Resolve/create and pin a session atomically before waiting for its turn."""
    with _SESSION_LOCK:
        session_id, session_token, reused = get_or_issue_session(
            token,
            secret,
            actor_id=actor_id,
            authoritative_session_id=authoritative_session_id,
            auth_mode=auth_mode,
        )
        turn_lock = _pin_turn_locked(
            session_id,
            actor_id=actor_id,
            auth_mode=auth_mode,
        )
    lease = _finish_turn_acquisition(
        session_id,
        turn_lock,
        timeout_seconds=timeout_seconds,
    )
    return session_id, session_token, reused, lease


def get_history(
    session_id: str,
    *,
    actor_id: Optional[str] = None,
    auth_mode: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return stored history only when actor/auth ownership matches."""
    with _SESSION_LOCK:
        payload = _payload_for_access_locked(
            session_id,
            actor_id=actor_id,
            auth_mode=auth_mode,
            mutation=False,
        )
        if payload is None:
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
    auth_mode: Optional[str] = None,
) -> None:
    """Write initial turns into an existing, owned session with empty history."""
    if not turns:
        return
    now_ts = time.time()
    with _SESSION_LOCK:
        payload = _payload_for_access_locked(
            session_id,
            actor_id=actor_id,
            auth_mode=auth_mode,
            mutation=True,
        )
        assert payload is not None
        existing = payload.get("history")
        if existing:
            return
        payload["history"] = [
            {
                "question": _history_text(turn.get("question", "")),
                "answer": _history_text(turn.get("answer", "")),
            }
            for turn in turns[:SESSION_HISTORY_MAX_TURNS]
        ]
        payload["updated_at"] = now_ts


def append_exchange(
    session_id: str,
    question: str,
    answer: str,
    *,
    actor_id: Optional[str] = None,
    auth_mode: Optional[str] = None,
) -> None:
    """Append a Q/A pair to an existing owned session and bound history."""
    now_ts = time.time()
    with _SESSION_LOCK:
        payload = _payload_for_access_locked(
            session_id,
            actor_id=actor_id,
            auth_mode=auth_mode,
            mutation=True,
        )
        assert payload is not None
        history = payload.get("history")
        if not isinstance(history, list):
            history = []
            payload["history"] = history
        history.append(
            {
                "question": _history_text(question),
                "answer": _history_text(answer),
            }
        )
        if len(history) > SESSION_HISTORY_MAX_TURNS:
            del history[:-SESSION_HISTORY_MAX_TURNS]
        payload["updated_at"] = now_ts


def set_last_contract(
    session_id: str,
    snapshot_json: str,
    *,
    actor_id: Optional[str] = None,
    auth_mode: Optional[str] = None,
) -> None:
    """Store the last authoritative contract on an existing owned session."""
    snapshot = str(snapshot_json or "")
    if not session_id or not snapshot or len(snapshot) > _MAX_CONTRACT_SNAPSHOT_CHARS:
        return
    with _SESSION_LOCK:
        payload = _payload_for_access_locked(
            session_id,
            actor_id=actor_id,
            auth_mode=auth_mode,
            mutation=True,
        )
        assert payload is not None
        payload["last_contract"] = snapshot
        payload["updated_at"] = time.time()


def get_last_contract(
    session_id: str,
    *,
    actor_id: Optional[str] = None,
    auth_mode: Optional[str] = None,
) -> str:
    """Return an owned contract snapshot, or an empty string when absent."""
    if not session_id:
        return ""
    with _SESSION_LOCK:
        payload = _payload_for_access_locked(
            session_id,
            actor_id=actor_id,
            auth_mode=auth_mode,
            mutation=False,
        )
        if payload is None:
            return ""
        return str(payload.get("last_contract", "") or "")


def get_session_memory_snapshot() -> Dict[str, object]:
    """Return aggregate, content-free state for the protected metrics route."""
    with _SESSION_LOCK:
        _cleanup_expired(time.time())
        return {
            "storage": "process_local",
            "supported_replicas": 1,
            "current_sessions": len(_SESSION_STORE),
            "max_sessions": SESSION_MAX_ENTRIES,
            "idle_ttl_seconds": SESSION_IDLE_TTL_SECONDS,
            "history_max_turns": SESSION_HISTORY_MAX_TURNS,
            "history_max_item_chars": SESSION_HISTORY_MAX_ITEM_CHARS,
            "turn_wait_timeout_seconds": SESSION_TURN_WAIT_TIMEOUT_MS / 1000.0,
            "turn_participants": sum(
                _turn_participants(payload) for payload in _SESSION_STORE.values()
            ),
            "capacity_evictions": _SESSION_CAPACITY_EVICTIONS,
            "expired_sessions": _SESSION_EXPIRATIONS,
            "stale_token_renewals": _SESSION_STALE_TOKEN_RENEWALS,
            "capacity_rejections": _SESSION_CAPACITY_REJECTIONS,
            "turn_timeouts": _SESSION_TURN_TIMEOUTS,
            "max_turn_wait_seconds": round(_SESSION_TURN_WAIT_MAX_SECONDS, 6),
        }


def reset_session_memory_for_tests() -> None:
    """Reset process-local state and counters owned by isolated tests."""
    global _SESSION_CAPACITY_EVICTIONS
    global _SESSION_EXPIRATIONS
    global _SESSION_STALE_TOKEN_RENEWALS
    global _SESSION_CAPACITY_REJECTIONS
    global _SESSION_TURN_TIMEOUTS
    global _SESSION_TURN_WAIT_MAX_SECONDS
    with _SESSION_LOCK:
        _SESSION_STORE.clear()
        _SESSION_CAPACITY_EVICTIONS = 0
        _SESSION_EXPIRATIONS = 0
        _SESSION_STALE_TOKEN_RENEWALS = 0
        _SESSION_CAPACITY_REJECTIONS = 0
        _SESSION_TURN_TIMEOUTS = 0
        _SESSION_TURN_WAIT_MAX_SECONDS = 0.0
