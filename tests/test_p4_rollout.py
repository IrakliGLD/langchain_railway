"""Tests for deterministic P4 request-cohort rollout assignment."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.p4_rollout import (
    GATE_EVIDENCE_FINALIZATION,
    GATE_PLAN_VALIDATION,
    RolloutDecision,
    build_rollout_decisions,
    decide_rollout,
    gate_is_active,
)
from models import QueryContext
from utils.metrics import metrics


def test_master_disabled_never_assigns_active():
    decision = decide_rollout(
        GATE_PLAN_VALIDATION,
        requested=False,
        percent=100,
        actor_id="actor-a",
    )
    assert decision == RolloutDecision(False, "disabled", 100, None, "none")


def test_zero_and_full_rollouts_do_not_require_identity():
    zero = decide_rollout(GATE_PLAN_VALIDATION, requested=True, percent=0)
    full = decide_rollout(GATE_PLAN_VALIDATION, requested=True, percent=100)
    assert zero.status == "holdback"
    assert zero.enabled is False
    assert full.status == "active"
    assert full.enabled is True


def test_invalid_gate_and_percentage_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="Unknown P4 rollout gate"):
        decide_rollout("typo", requested=True, percent=5, actor_id="actor")
    with pytest.raises(ValueError, match="from 0 to 100"):
        decide_rollout(GATE_PLAN_VALIDATION, requested=True, percent=101, actor_id="actor")
    with pytest.raises(ValueError, match="must be an integer"):
        decide_rollout(GATE_PLAN_VALIDATION, requested=True, percent="not-a-number", actor_id="actor")


def test_partial_rollout_without_identity_fails_closed():
    decision = decide_rollout(GATE_PLAN_VALIDATION, requested=True, percent=25)
    assert decision.status == "ineligible"
    assert decision.enabled is False
    assert decision.bucket is None


def test_actor_assignment_is_stable_and_takes_precedence():
    first = decide_rollout(
        GATE_EVIDENCE_FINALIZATION,
        requested=True,
        percent=25,
        actor_id="actor-a",
        session_id="session-a",
        request_id="request-a",
    )
    second = decide_rollout(
        GATE_EVIDENCE_FINALIZATION,
        requested=True,
        percent=25,
        actor_id="actor-a",
        session_id="different-session",
        request_id="different-request",
    )
    assert first == second
    assert first.key_source == "actor"
    assert first.bucket is not None


def test_percentage_steps_are_monotonic_for_one_gate_and_actor():
    decisions = [
        decide_rollout(
            GATE_PLAN_VALIDATION,
            requested=True,
            percent=percent,
            actor_id="actor-monotonic",
        )
        for percent in (5, 25, 100)
    ]
    assert not decisions[0].enabled or decisions[1].enabled
    assert decisions[2].enabled


def test_build_counts_only_privacy_safe_status_keys():
    before = dict(metrics.p4_rollout_events)
    decisions = build_rollout_decisions(
        {
            GATE_EVIDENCE_FINALIZATION: (True, 100),
            GATE_PLAN_VALIDATION: (False, 100),
        },
        actor_id="sensitive-actor-id",
    )
    assert decisions[GATE_EVIDENCE_FINALIZATION].enabled is True
    assert decisions[GATE_PLAN_VALIDATION].enabled is False
    assert metrics.p4_rollout_events.get(
        f"{GATE_EVIDENCE_FINALIZATION}:active", 0
    ) == before.get(f"{GATE_EVIDENCE_FINALIZATION}:active", 0) + 1
    assert metrics.p4_rollout_events.get(
        f"{GATE_PLAN_VALIDATION}:disabled", 0
    ) == before.get(f"{GATE_PLAN_VALIDATION}:disabled", 0) + 1
    assert "sensitive-actor-id" not in repr(metrics.p4_rollout_events)


def test_gate_reader_supports_typed_boolean_and_legacy_defaults():
    ctx = QueryContext(query="q")
    assert gate_is_active(ctx, GATE_PLAN_VALIDATION, default=True) is True
    ctx.p4_rollout_decisions[GATE_PLAN_VALIDATION] = False
    assert gate_is_active(ctx, GATE_PLAN_VALIDATION, default=True) is False
    ctx.p4_rollout_decisions[GATE_PLAN_VALIDATION] = RolloutDecision(
        True, "active", 5, 10, "actor"
    )
    assert gate_is_active(ctx, GATE_PLAN_VALIDATION, default=False) is True
    ctx.p4_rollout_decisions[GATE_PLAN_VALIDATION] = object()
    assert gate_is_active(ctx, GATE_PLAN_VALIDATION, default=True) is False
