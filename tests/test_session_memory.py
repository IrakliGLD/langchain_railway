"""
Tests for session-bound memory controls.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from utils import session_memory  # noqa: E402
from config import SESSION_HISTORY_MAX_TURNS  # noqa: E402


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


def test_history_is_limited_to_configured_turns():
    session_id, _ = session_memory.issue_session_token("secret-1")
    for idx in range(10):
        session_memory.append_exchange(session_id, f"q{idx}", f"a{idx}")

    history = session_memory.get_history(session_id)
    assert len(history) <= SESSION_HISTORY_MAX_TURNS
    assert history[-1]["question"] == "q9"
