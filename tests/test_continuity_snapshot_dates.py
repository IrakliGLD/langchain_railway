"""The continuity snapshot must not carry a date window into the next turn.

Enabling ENABLE_CONTRACT_CONTINUITY fixed the follow-up that could not reach its
data -- entity_scope now carries forward and a 0.15-confidence empty analysis
became 0.94 with topics and scope populated. That is worth keeping.

But the snapshot also carries ``params_hint.start_date`` / ``end_date``
(contract_continuity.py:41), so turn 2 inherits turn 1's window along with its
scope. On 2026-08-17 turn 1's analyzer had invented "August 2026" for a dateless
question; inheriting that is how the answer's coverage came to depend on which
question was asked first.

Scope is worth inheriting. A window is not: the next question may be about a
different period, and a period nobody asked for should not survive one turn, let
alone propagate.

Nothing is lost by dropping it. ``canonical_query_en`` is in the same snapshot and
states the period in the user's own terms when they named one ("monthly balancing
prices for 2023"), so the information still travels -- it just stops being applied
as a window the next turn never asked for.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.contract_continuity import continuity_snapshot_json
from tests.test_end_user_scope_clarification import _ctx


def _ctx_with_hint(**hint_fields):
    """A ctx whose top candidate tool carries a params_hint."""
    ctx = _ctx(
        "For Telmico commercial customers at 6-10 kV, is retail cheaper than wholesale?",
        topics=("network_supply_tariffs",),
        query_type="comparison",
    )
    qa = ctx.question_analysis
    qa.entity_scope = "Telmico commercial customers connected at 6–10 kV"
    payload = qa.model_dump(mode="json")
    payload["tooling"] = {
        "candidate_tools": [
            {
                "name": "get_end_user_prices",
                "score": 0.9,
                "params_hint": {**hint_fields},
            }
        ]
    }
    from contracts.question_analysis import QuestionAnalysis

    ctx.question_analysis = QuestionAnalysis(**payload)
    return ctx


def test_the_snapshot_drops_the_date_window():
    """The window is the part that must not survive the turn."""
    snapshot = continuity_snapshot_json(
        _ctx_with_hint(start_date="2026-08-01", end_date="2026-08-31", currency="gel")
    )

    assert snapshot, "snapshot should still be produced"
    payload = json.loads(snapshot)
    hint = payload.get("params_hint") or {}
    assert "start_date" not in hint
    assert "end_date" not in hint


def test_the_rest_of_the_hint_survives():
    """Currency, metric and granularity describe WHAT is asked, not WHEN."""
    snapshot = continuity_snapshot_json(
        _ctx_with_hint(
            start_date="2026-08-01",
            end_date="2026-08-31",
            currency="gel",
            granularity="monthly",
        )
    )

    hint = json.loads(snapshot)["params_hint"]
    assert hint["currency"] == "gel"
    assert hint["granularity"] == "monthly"


def test_a_hint_of_only_dates_leaves_no_empty_hint_behind():
    """params_hint must be absent, not an empty object."""
    snapshot = continuity_snapshot_json(
        _ctx_with_hint(start_date="2026-08-01", end_date="2026-08-31")
    )

    payload = json.loads(snapshot)
    assert "params_hint" not in payload


def test_the_scope_still_carries_forward():
    """The reason the flag was enabled at all -- do not regress it."""
    snapshot = continuity_snapshot_json(
        _ctx_with_hint(start_date="2026-08-01", end_date="2026-08-31")
    )

    payload = json.loads(snapshot)
    assert payload["entity_scope"] == "Telmico commercial customers connected at 6–10 kV"
    assert payload["canonical_query_en"]
