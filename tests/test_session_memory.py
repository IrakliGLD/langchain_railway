"""
Tests for session-bound memory controls.
"""
import os

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
