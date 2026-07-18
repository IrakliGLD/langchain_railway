"""Characterization tests for actor-bound conversation turn orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from core.session_runtime import (
    SessionOwnershipError,
    SessionRuntime,
    filter_caller_history,
)


@dataclass
class _Lease:
    releases: int = 0

    def release(self) -> None:
        self.releases += 1


class _Repository:
    def __init__(self) -> None:
        self.lease = _Lease()
        self.history = []
        self.contract = "prior-contract"
        self.calls = []

    def get_or_issue_session_turn(self, token, secret, **kwargs):
        self.calls.append(("acquire", token, secret, kwargs))
        return "session-1", "session-1.signed", token == "valid", self.lease

    def get_history(self, session_id, **kwargs):
        self.calls.append(("history", session_id, kwargs))
        return list(self.history)

    def seed_history(self, session_id, turns, **kwargs):
        self.calls.append(("seed", session_id, list(turns), kwargs))
        self.history = list(turns)

    def get_last_contract(self, session_id, **kwargs):
        self.calls.append(("get_contract", session_id, kwargs))
        return self.contract

    def append_exchange(self, session_id, question, answer, **kwargs):
        self.calls.append(("append", session_id, question, answer, kwargs))

    def set_last_contract(self, session_id, snapshot, **kwargs):
        self.calls.append(("set_contract", session_id, snapshot, kwargs))

    def resolve_session_token(self, token, secret, **kwargs):
        self.calls.append(("resolve", token, secret, kwargs))
        return "session-1" if token == "valid" else None

    def get_session_memory_snapshot(self):
        return {"storage": "process_local", "supported_replicas": 1}


def _runtime(repository=None):
    return SessionRuntime(
        repository=repository or _Repository(),
        signing_secret="signing-secret",
        caller_history_max_item_chars=2000,
    )


def test_history_filter_preserves_compatibility_and_blocks_hostile_turns(monkeypatch):
    decisions = {
        "safe": ("allow", "safe"),
        "rewrite": ("warn", "sanitized"),
        "hostile": ("block", ""),
    }

    class _Decision:
        def __init__(self, action, sanitized_query):
            self.action = action
            self.sanitized_query = sanitized_query

    monkeypatch.setattr(
        "core.session_runtime.inspect_query",
        lambda value: _Decision(*decisions.get(value, ("allow", value))),
    )

    history, blocked = filter_caller_history(
        [
            {"question": "safe", "answer": "rewrite"},
            {"question": "hostile", "answer": "safe"},
            {"answer": "missing question"},
        ],
        is_bearer=True,
    )

    assert history == [{"question": "safe", "answer": "sanitized"}]
    assert blocked == 1


def test_begin_turn_binds_every_repository_access_to_actor_and_auth_mode():
    repository = _Repository()
    repository.history = [{"question": "stored", "answer": "answer"}]
    runtime = _runtime(repository)

    turn = runtime.begin_turn(
        token="valid",
        actor_id="actor-1",
        auth_mode="gateway",
        authoritative_session_id="edge-session",
        caller_history=[{"question": "duplicate", "answer": "ignored"}],
        timeout_seconds=1.25,
    )

    assert turn.session_id == "session-1"
    assert turn.session_token == "session-1.signed"
    assert turn.reused_existing_session is True
    assert turn.history == [{"question": "stored", "answer": "answer"}]
    assert turn.ignored_caller_history is True
    assert turn.continuity_available is True
    assert repository.calls[0] == (
        "acquire",
        "valid",
        "signing-secret",
        {
            "actor_id": "actor-1",
            "authoritative_session_id": "edge-session",
            "auth_mode": "gateway",
            "timeout_seconds": 1.25,
        },
    )
    assert repository.calls[1][2] == {
        "actor_id": "actor-1",
        "auth_mode": "gateway",
    }


def test_begin_turn_seeds_only_filtered_caller_history_when_store_is_empty(monkeypatch):
    repository = _Repository()
    runtime = _runtime(repository)

    class _Decision:
        def __init__(self, value):
            self.action = "block" if value == "hostile" else "allow"
            self.sanitized_query = value

    monkeypatch.setattr("core.session_runtime.inspect_query", _Decision)
    turn = runtime.begin_turn(
        token=None,
        actor_id="actor-1",
        auth_mode="gateway",
        authoritative_session_id="edge-session",
        caller_history=[
            {"question": "safe", "answer": "answer"},
            {"question": "hostile", "answer": "answer"},
        ],
        timeout_seconds=0.5,
    )

    assert turn.history == [{"question": "safe", "answer": "answer"}]
    assert turn.blocked_caller_turns == 1
    assert turn.seeded_from_caller is True
    assert turn.continuity_available is True
    seed_call = next(call for call in repository.calls if call[0] == "seed")
    assert seed_call[2] == turn.history
    assert seed_call[3] == {"actor_id": "actor-1", "auth_mode": "gateway"}


def test_turn_contract_and_mutations_keep_the_same_owner_binding():
    repository = _Repository()
    runtime = _runtime(repository)
    turn = runtime.begin_turn(
        token=None,
        actor_id="actor-1",
        auth_mode="public_bearer",
        authoritative_session_id=None,
        caller_history=None,
        timeout_seconds=0.25,
    )

    assert turn.previous_contract() == "prior-contract"
    turn.record_exchange(
        "question",
        "answer",
        contract_snapshot_factory=lambda: "next-contract",
    )
    turn.release()

    owner = {"actor_id": "actor-1", "auth_mode": "public_bearer"}
    assert ("get_contract", "session-1", owner) in repository.calls
    assert ("append", "session-1", "question", "answer", owner) in repository.calls
    assert ("set_contract", "session-1", "next-contract", owner) in repository.calls
    assert repository.calls.index(("append", "session-1", "question", "answer", owner)) < repository.calls.index(
        ("set_contract", "session-1", "next-contract", owner)
    )
    assert repository.lease.releases == 1


def test_begin_turn_releases_the_lease_if_history_preparation_fails():
    repository = _Repository()

    def fail_history(*_args, **_kwargs):
        raise RuntimeError("history unavailable")

    repository.get_history = fail_history
    runtime = _runtime(repository)

    with pytest.raises(RuntimeError, match="history unavailable"):
        runtime.begin_turn(
            token=None,
            actor_id="actor-1",
            auth_mode="gateway",
            authoritative_session_id="edge-session",
            caller_history=None,
            timeout_seconds=0.25,
        )

    assert repository.lease.releases == 1


def test_only_acquisition_value_errors_are_classified_as_ownership_failures():
    repository = _Repository()
    repository.get_or_issue_session_turn = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ValueError("binding mismatch")
    )
    runtime = _runtime(repository)

    with pytest.raises(SessionOwnershipError, match="binding mismatch"):
        runtime.begin_turn(
            token="invalid",
            actor_id="actor-1",
            auth_mode="gateway",
            authoritative_session_id="edge-session",
            caller_history=None,
            timeout_seconds=0.25,
        )

    repository = _Repository()
    repository.get_history = lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("malformed history"))
    runtime = _runtime(repository)

    with pytest.raises(ValueError, match="malformed history") as exc_info:
        runtime.begin_turn(
            token=None,
            actor_id="actor-1",
            auth_mode="gateway",
            authoritative_session_id="edge-session",
            caller_history=None,
            timeout_seconds=0.25,
        )

    assert not isinstance(exc_info.value, SessionOwnershipError)
    assert repository.lease.releases == 1


def test_token_resolution_and_topology_snapshot_delegate_without_identifiers():
    repository = _Repository()
    runtime = _runtime(repository)

    assert runtime.resolve_token("valid", actor_id="actor-1", auth_mode="gateway") == "session-1"
    assert runtime.resolve_token("invalid", actor_id="actor-1", auth_mode="gateway") is None
    assert runtime.memory_snapshot() == {
        "storage": "process_local",
        "supported_replicas": 1,
    }
