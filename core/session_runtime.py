"""Actor-bound conversation turn orchestration over a session repository.

The repository owns storage, signing, expiry, capacity, and locking.  This
module owns the higher-level invariant that every history/contract operation in
one request uses the same actor and authentication-mode binding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from config import SESSION_HISTORY_MAX_ITEM_CHARS
from guardrails.firewall import inspect_query


class SessionRepository(Protocol):
    def get_or_issue_session_turn(self, token, secret, **kwargs): ...

    def get_history(self, session_id, **kwargs): ...

    def seed_history(self, session_id, turns, **kwargs): ...

    def get_last_contract(self, session_id, **kwargs): ...

    def append_exchange(self, session_id, question, answer, **kwargs): ...

    def set_last_contract(self, session_id, snapshot, **kwargs): ...

    def resolve_session_token(self, token, secret, **kwargs): ...

    def get_session_memory_snapshot(self): ...


class SessionOwnershipError(ValueError):
    """The presented session cannot be owned by the authenticated actor."""


def filter_caller_history(
    raw_history: Optional[list[Any]],
    *,
    is_bearer: Optional[bool] = None,
    max_item_chars: int = SESSION_HISTORY_MAX_ITEM_CHARS,
    max_items: int = 3,
) -> tuple[list[dict[str, str]], int]:
    """Normalize hostile transport or stored history into seedable turns.

    ``is_bearer`` remains an ignored compatibility argument for callers using
    the former helper signature from ``main``.
    """

    del is_bearer
    items: list[dict[str, str]] = []
    blocked = 0
    for turn in (raw_history or [])[:max_items]:
        if hasattr(turn, "model_dump"):
            turn = turn.model_dump()
        if not (isinstance(turn, dict) and turn.get("question")):
            continue
        question = str(turn.get("question", ""))[:max_item_chars]
        answer = str(turn.get("answer", ""))[:max_item_chars]
        question_decision = inspect_query(question)
        answer_decision = inspect_query(answer)
        if question_decision.action == "block" or answer_decision.action == "block":
            blocked += 1
            continue
        items.append(
            {
                "question": question_decision.sanitized_query,
                "answer": answer_decision.sanitized_query,
            }
        )
    return items, blocked


@dataclass
class SessionTurn:
    """One exclusively owned conversation turn and its continuity state."""

    session_id: str
    session_token: str
    reused_existing_session: bool
    history: list[dict[str, str]]
    blocked_stored_turns: int
    blocked_caller_turns: int
    seeded_from_caller: bool
    ignored_caller_history: bool
    actor_id: Optional[str]
    auth_mode: str
    _repository: SessionRepository
    _lease: Any

    @property
    def continuity_available(self) -> bool:
        return bool(self.reused_existing_session or self.history)

    def _owner(self) -> dict[str, Optional[str]]:
        return {"actor_id": self.actor_id, "auth_mode": self.auth_mode}

    def previous_contract(self) -> str:
        return self._repository.get_last_contract(self.session_id, **self._owner())

    def record_exchange(
        self,
        question: str,
        answer: str,
        *,
        contract_snapshot_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        owner = self._owner()
        self._repository.append_exchange(
            self.session_id,
            question,
            answer,
            **owner,
        )
        contract_snapshot = contract_snapshot_factory() if contract_snapshot_factory is not None else ""
        if contract_snapshot:
            self._repository.set_last_contract(
                self.session_id,
                contract_snapshot,
                **owner,
            )

    def release(self) -> None:
        self._lease.release()


class SessionRuntime:
    """Coordinate a complete request turn over one repository interface."""

    def __init__(
        self,
        *,
        repository: SessionRepository,
        signing_secret: str,
        caller_history_max_item_chars: int,
    ) -> None:
        self._repository = repository
        self._signing_secret = signing_secret
        self._caller_history_max_item_chars = caller_history_max_item_chars

    def begin_turn(
        self,
        *,
        token: Optional[str],
        actor_id: Optional[str],
        auth_mode: str,
        authoritative_session_id: Optional[str],
        caller_history: Optional[list[Any]],
        timeout_seconds: float,
    ) -> SessionTurn:
        try:
            session_id, session_token, reused, lease = self._repository.get_or_issue_session_turn(
                token,
                self._signing_secret,
                actor_id=actor_id,
                authoritative_session_id=authoritative_session_id,
                auth_mode=auth_mode,
                timeout_seconds=timeout_seconds,
            )
        except ValueError as exc:
            raise SessionOwnershipError(str(exc)) from exc
        owner = {"actor_id": actor_id, "auth_mode": auth_mode}
        try:
            stored_history = self._repository.get_history(session_id, **owner)
            bound_history, blocked_stored = filter_caller_history(stored_history)
            caller_history_items, blocked_caller = filter_caller_history(
                caller_history,
                max_item_chars=self._caller_history_max_item_chars,
            )
            seeded = False
            ignored = bool(caller_history_items and bound_history)
            if not bound_history and caller_history_items:
                bound_history = caller_history_items
                self._repository.seed_history(session_id, bound_history, **owner)
                seeded = True
            return SessionTurn(
                session_id=session_id,
                session_token=session_token,
                reused_existing_session=reused,
                history=bound_history,
                blocked_stored_turns=blocked_stored,
                blocked_caller_turns=blocked_caller,
                seeded_from_caller=seeded,
                ignored_caller_history=ignored,
                actor_id=actor_id,
                auth_mode=auth_mode,
                _repository=self._repository,
                _lease=lease,
            )
        except Exception:
            lease.release()
            raise

    def resolve_token(
        self,
        token: Optional[str],
        *,
        actor_id: Optional[str] = None,
        auth_mode: Optional[str] = None,
    ) -> Optional[str]:
        return self._repository.resolve_session_token(
            token,
            self._signing_secret,
            actor_id=actor_id,
            auth_mode=auth_mode,
        )

    def memory_snapshot(self) -> dict[str, object]:
        return self._repository.get_session_memory_snapshot()
