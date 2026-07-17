"""
Tests for session-bound memory controls.
"""
import os
import threading
import time

import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from config import SESSION_HISTORY_MAX_TURNS  # noqa: E402
from utils import session_memory  # noqa: E402


def setup_function():
    reset = getattr(session_memory, "reset_session_memory_for_tests", None)
    if callable(reset):
        reset()
    else:
        session_memory._SESSION_STORE.clear()


def test_issue_and_resolve_signed_session_token():
    session_id, token = session_memory.issue_session_token("secret-1")
    assert session_id
    assert "." in token
    assert session_memory.resolve_session_token(token, "secret-1") == session_id


def test_invalid_signature_is_rejected():
    _, token = session_memory.issue_session_token("secret-1")
    tampered = token + "abc"
    assert session_memory.resolve_session_token(tampered, "secret-1") is None
    assert session_memory.resolve_session_token(token, "different-secret") is None


def test_actor_bound_session_token_rejects_cross_user_replay():
    session_id, token = session_memory.issue_session_token("secret-1", actor_id="actor-one")

    assert session_memory.resolve_session_token(token, "secret-1", actor_id="actor-one") == session_id
    assert session_memory.resolve_session_token(token, "secret-1", actor_id="actor-two") is None
    assert session_memory.resolve_session_token(token, "secret-1") is None


def test_verified_edge_session_is_stable_and_actor_isolated():
    first_id, first_token, first_reused = session_memory.get_or_issue_session(
        None,
        "secret-1",
        actor_id="actor-one",
        authoritative_session_id="supabase-session",
    )
    second_id, second_token, second_reused = session_memory.get_or_issue_session(
        None,
        "secret-1",
        actor_id="actor-one",
        authoritative_session_id="supabase-session",
    )
    other_actor_id, _, _ = session_memory.get_or_issue_session(
        None,
        "secret-1",
        actor_id="actor-two",
        authoritative_session_id="supabase-session",
    )

    assert first_reused is False
    assert second_reused is True
    assert (second_id, second_token) == (first_id, first_token)
    assert other_actor_id != first_id


def test_session_history_and_mutations_enforce_actor_binding():
    session_id, _ = session_memory.issue_session_token("secret-1", actor_id="actor-one")
    session_memory.append_exchange(
        session_id,
        "question",
        "answer",
        actor_id="actor-one",
    )

    assert session_memory.get_history(session_id, actor_id="actor-one")
    assert session_memory.get_history(session_id, actor_id="actor-two") == []
    with pytest.raises(ValueError, match="actor binding"):
        session_memory.append_exchange(
            session_id,
            "cross-user question",
            "cross-user answer",
            actor_id="actor-two",
        )


def test_authoritative_session_requires_verified_actor():
    with pytest.raises(ValueError, match="verified actor"):
        session_memory.get_or_issue_session(
            None,
            "secret-1",
            authoritative_session_id="unbound-session",
        )


def test_history_is_limited_to_configured_turns():
    session_id, _ = session_memory.issue_session_token("secret-1")
    for idx in range(10):
        session_memory.append_exchange(session_id, f"q{idx}", f"a{idx}")

    history = session_memory.get_history(session_id)
    assert len(history) <= SESSION_HISTORY_MAX_TURNS
    assert history[-1]["question"] == "q9"


def test_expired_token_gets_new_identity_instead_of_resurrecting_record(monkeypatch):
    monkeypatch.setattr(session_memory, "SESSION_IDLE_TTL_SECONDS", 60)
    session_id, token = session_memory.issue_session_token(
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
    )
    session_memory.append_exchange(
        session_id,
        "old question",
        "old answer",
        actor_id="actor-one",
        auth_mode="gateway",
    )
    with session_memory._SESSION_LOCK:
        session_memory._SESSION_STORE[session_id]["updated_at"] = time.time() - 61

    renewed_id, renewed_token, reused = session_memory.get_or_issue_session(
        token,
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
    )

    assert reused is False
    assert renewed_id != session_id
    assert session_id not in session_memory._SESSION_STORE
    assert session_memory.get_history(
        session_id,
        actor_id="actor-one",
        auth_mode="gateway",
    ) == []
    assert session_memory.resolve_session_token(
        renewed_token,
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
    ) == renewed_id
    assert session_memory.get_session_memory_snapshot()["stale_token_renewals"] == 1


def test_auth_mode_owns_token_history_and_contract_continuity():
    session_id, token = session_memory.issue_session_token(
        "secret-1",
        actor_id="same-actor",
        auth_mode="gateway",
    )
    session_memory.append_exchange(
        session_id,
        "gateway question",
        "gateway answer",
        actor_id="same-actor",
        auth_mode="gateway",
    )
    session_memory.set_last_contract(
        session_id,
        '{"query_type":"trend"}',
        actor_id="same-actor",
        auth_mode="gateway",
    )

    assert session_memory.resolve_session_token(
        token,
        "secret-1",
        actor_id="same-actor",
        auth_mode="public_bearer",
    ) is None
    assert session_memory.get_history(
        session_id,
        actor_id="same-actor",
        auth_mode="public_bearer",
    ) == []
    assert session_memory.get_last_contract(
        session_id,
        actor_id="same-actor",
        auth_mode="public_bearer",
    ) == ""


def test_mutation_cannot_recreate_an_expired_or_evicted_session():
    session_id, _ = session_memory.issue_session_token(
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
    )
    with session_memory._SESSION_LOCK:
        session_memory._SESSION_STORE.pop(session_id)

    with pytest.raises(session_memory.SessionExpiredError):
        session_memory.append_exchange(
            session_id,
            "must not",
            "recreate",
            actor_id="actor-one",
            auth_mode="gateway",
        )
    with pytest.raises(session_memory.SessionExpiredError):
        session_memory.seed_history(
            session_id,
            [{"question": "must not", "answer": "recreate"}],
            actor_id="actor-one",
            auth_mode="gateway",
        )
    with pytest.raises(session_memory.SessionExpiredError):
        session_memory.set_last_contract(
            session_id,
            "{}",
            actor_id="actor-one",
            auth_mode="gateway",
        )
    assert session_id not in session_memory._SESSION_STORE


def test_actor_bound_session_turns_are_serialized():
    session_id, _ = session_memory.issue_session_token(
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    order = []

    def first_turn():
        lease = session_memory.acquire_session_turn(
            session_id,
            actor_id="actor-one",
            auth_mode="gateway",
            timeout_seconds=0.5,
        )
        try:
            order.append("first")
            first_entered.set()
            assert release_first.wait(1.0)
        finally:
            lease.release()

    def second_turn():
        assert first_entered.wait(0.5)
        lease = session_memory.acquire_session_turn(
            session_id,
            actor_id="actor-one",
            auth_mode="gateway",
            timeout_seconds=0.5,
        )
        try:
            order.append("second")
            second_entered.set()
        finally:
            lease.release()

    first = threading.Thread(target=first_turn)
    second = threading.Thread(target=second_turn)
    first.start()
    second.start()
    assert first_entered.wait(0.5)
    assert second_entered.wait(0.03) is False
    release_first.set()
    first.join(1.0)
    second.join(1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert order == ["first", "second"]


def test_session_turn_wait_is_bounded_and_observable():
    session_id, _ = session_memory.issue_session_token(
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
    )
    first = session_memory.acquire_session_turn(
        session_id,
        actor_id="actor-one",
        auth_mode="gateway",
        timeout_seconds=0,
    )
    started = time.monotonic()
    try:
        with pytest.raises(session_memory.SessionTurnBusyError):
            session_memory.acquire_session_turn(
                session_id,
                actor_id="actor-one",
                auth_mode="gateway",
                timeout_seconds=0.02,
            )
    finally:
        first.release()

    assert time.monotonic() - started < 0.1
    snapshot = session_memory.get_session_memory_snapshot()
    assert snapshot["turn_timeouts"] == 1
    assert snapshot["turn_participants"] == 0


def test_store_evicts_oldest_inactive_session_at_hard_capacity(monkeypatch):
    monkeypatch.setattr(session_memory, "SESSION_MAX_ENTRIES", 2)
    first_id, _ = session_memory.issue_session_token("secret-1")
    second_id, _ = session_memory.issue_session_token("secret-1")
    now_ts = time.time()
    with session_memory._SESSION_LOCK:
        session_memory._SESSION_STORE[first_id]["updated_at"] = now_ts - 2
        session_memory._SESSION_STORE[second_id]["updated_at"] = now_ts - 1

    third_id, _ = session_memory.issue_session_token("secret-1")

    assert len(session_memory._SESSION_STORE) == 2
    assert first_id not in session_memory._SESSION_STORE
    assert {second_id, third_id} == set(session_memory._SESSION_STORE)
    assert session_memory.get_session_memory_snapshot()["capacity_evictions"] == 1


def test_store_never_evicts_an_active_turn_to_make_capacity(monkeypatch):
    monkeypatch.setattr(session_memory, "SESSION_MAX_ENTRIES", 1)
    session_id, _ = session_memory.issue_session_token(
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
    )
    lease = session_memory.acquire_session_turn(
        session_id,
        actor_id="actor-one",
        auth_mode="gateway",
        timeout_seconds=0,
    )
    try:
        with pytest.raises(session_memory.SessionCapacityExceededError):
            session_memory.issue_session_token("secret-1")
        assert set(session_memory._SESSION_STORE) == {session_id}
    finally:
        lease.release()

    replacement_id, _ = session_memory.issue_session_token("secret-1")
    assert set(session_memory._SESSION_STORE) == {replacement_id}


def test_cross_actor_replay_is_rejected_without_creating_continuity():
    original_id, token = session_memory.issue_session_token(
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
    )
    session_memory.append_exchange(
        original_id,
        "private question",
        "private answer",
        actor_id="actor-one",
        auth_mode="gateway",
    )

    with pytest.raises(ValueError, match="binding mismatch"):
        session_memory.get_or_issue_session(
            token,
            "secret-1",
            actor_id="actor-two",
            auth_mode="gateway",
        )

    assert set(session_memory._SESSION_STORE) == {original_id}
    assert session_memory.get_history(
        original_id,
        actor_id="actor-one",
        auth_mode="gateway",
    )[0]["question"] == "private question"


def test_atomic_session_turn_acquisition_pins_new_session(monkeypatch):
    monkeypatch.setattr(session_memory, "SESSION_MAX_ENTRIES", 1)
    session_id, _token, _reused, lease = session_memory.get_or_issue_session_turn(
        None,
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
        timeout_seconds=0,
    )
    try:
        with pytest.raises(session_memory.SessionCapacityExceededError):
            session_memory.issue_session_token(
                "secret-1",
                actor_id="actor-two",
                auth_mode="gateway",
            )
        assert set(session_memory._SESSION_STORE) == {session_id}
    finally:
        lease.release()

def test_history_fields_are_bounded_in_memory(monkeypatch):
    monkeypatch.setattr(session_memory, "SESSION_HISTORY_MAX_ITEM_CHARS", 8)
    session_id, _ = session_memory.issue_session_token("secret-1")

    session_memory.append_exchange(session_id, "q" * 20, "a" * 20)
    history = session_memory.get_history(session_id)

    assert history == [{"question": "q" * 8, "answer": "a" * 8}]


def test_idle_expiry_starts_after_active_turn_release(monkeypatch):
    monkeypatch.setattr(session_memory, "SESSION_IDLE_TTL_SECONDS", 60)
    session_id, token = session_memory.issue_session_token(
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
    )
    lease = session_memory.acquire_session_turn(
        session_id,
        actor_id="actor-one",
        auth_mode="gateway",
        timeout_seconds=0,
    )
    with session_memory._SESSION_LOCK:
        session_memory._SESSION_STORE[session_id]["updated_at"] = time.time() - 120

    lease.release()
    renewed_id, _, reused = session_memory.get_or_issue_session(
        token,
        "secret-1",
        actor_id="actor-one",
        auth_mode="gateway",
    )

    assert reused is True
    assert renewed_id == session_id

def test_session_metrics_contain_no_actor_or_session_identifiers():
    raw_actor = "private-actor-123"
    session_id, _ = session_memory.issue_session_token(
        "secret-1",
        actor_id=raw_actor,
        auth_mode="gateway",
    )

    snapshot_text = repr(session_memory.get_session_memory_snapshot())

    assert raw_actor not in snapshot_text
    assert session_id not in snapshot_text
