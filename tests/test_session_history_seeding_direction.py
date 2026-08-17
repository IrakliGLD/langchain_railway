"""Seeding and appending must truncate history in the same direction.

They did not:

    session_memory.py:537 (seed)   turns[:SESSION_HISTORY_MAX_TURNS]      oldest N
    session_memory.py:570 (append) del history[:-SESSION_HISTORY_MAX_TURNS]  newest N

So on the turn that seeds a session, a caller supplying more turns than the cap
had its MOST RECENT turn discarded and the oldest kept -- exactly the turn a
follow-up refers to. Observed 2026-08-17 with the cap at 2 and three caller
turns supplied.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from config import SESSION_HISTORY_MAX_TURNS

_ACTOR_SEQ = iter(range(1000))


def _turns(count: int):
    return [{"question": f"q{i}", "answer": f"a{i}"} for i in range(1, count + 1)]


def _fresh_session():
    """A session owned by a distinct actor, with no history yet."""
    import utils.session_memory as sm

    actor = f"actor-seeding-{next(_ACTOR_SEQ)}"
    session_id, _token = sm.issue_session_token(
        "secret-seeding", actor_id=actor, auth_mode="gateway"
    )
    return sm, session_id, actor


def _stored(sm, session_id, actor):
    snapshot = sm.get_history(session_id, actor_id=actor, auth_mode="gateway")
    return [turn["question"] for turn in snapshot]


def test_seeding_keeps_the_newest_turns_not_the_oldest():
    """The defect: with the cap at 2 and three turns supplied, q3 was dropped."""
    sm, session_id, actor = _fresh_session()
    supplied = SESSION_HISTORY_MAX_TURNS + 1

    sm.seed_history(
        session_id, _turns(supplied), actor_id=actor, auth_mode="gateway"
    )

    kept = _stored(sm, session_id, actor)
    assert len(kept) == SESSION_HISTORY_MAX_TURNS
    assert kept[-1] == f"q{supplied}", (
        f"the most recent turn was dropped; kept {kept}"
    )


def test_seeding_below_the_cap_keeps_everything_in_order():
    sm, session_id, actor = _fresh_session()

    sm.seed_history(session_id, _turns(2), actor_id=actor, auth_mode="gateway")

    assert _stored(sm, session_id, actor) == ["q1", "q2"]


def test_seeding_and_appending_agree_on_what_survives():
    """The same turns through either path must leave the same history."""
    sm, seeded_id, actor = _fresh_session()
    total = SESSION_HISTORY_MAX_TURNS + 2

    sm.seed_history(seeded_id, _turns(total), actor_id=actor, auth_mode="gateway")
    seeded = _stored(sm, seeded_id, actor)

    _sm2, appended_id, actor2 = _fresh_session()
    for turn in _turns(total):
        sm.append_exchange(
            appended_id, turn["question"], turn["answer"],
            actor_id=actor2, auth_mode="gateway",
        )
    appended = _stored(sm, appended_id, actor2)

    assert seeded == appended, f"seed kept {seeded}, append kept {appended}"
