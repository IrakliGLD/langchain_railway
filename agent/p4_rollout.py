"""Deterministic request-cohort assignment for P4 behavior activation.

Master flags still decide whether a behavior is eligible for activation. This
module only selects a stable percentage of eligible traffic. It prefers the
gateway-verified actor identifier, then the signed server session, then the
request identifier. Raw identifiers and their hashes are never returned,
logged, or stored in metrics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

from utils.metrics import metrics

GATE_EVIDENCE_FINALIZATION = "evidence_finalization"
GATE_PLAN_VALIDATION = "plan_validation"
GATE_HONEST_TERMINAL_OUTCOMES = "honest_terminal_outcomes"
GATE_EVIDENCE_REANALYSIS = "evidence_reanalysis"

_BUCKET_COUNT = 10_000
_HASH_NAMESPACE = "enai-p4-rollout-v1"
_VALID_GATES = frozenset({
    GATE_EVIDENCE_FINALIZATION,
    GATE_PLAN_VALIDATION,
    GATE_HONEST_TERMINAL_OUTCOMES,
    GATE_EVIDENCE_REANALYSIS,
})


@dataclass(frozen=True)
class RolloutDecision:
    """One privacy-safe rollout decision attached to a QueryContext."""

    enabled: bool
    status: str
    percent: int
    bucket: int | None
    key_source: str


def _identity_source(
    *, actor_id: str = "", session_id: str = "", request_id: str = "",
) -> tuple[str, str]:
    for source, value in (
        ("actor", actor_id),
        ("session", session_id),
        ("request", request_id),
    ):
        normalized = str(value or "").strip()
        if normalized:
            return source, normalized
    return "none", ""


def decide_rollout(
    gate: str,
    *,
    requested: bool,
    percent: int,
    actor_id: str = "",
    session_id: str = "",
    request_id: str = "",
) -> RolloutDecision:
    """Return a deterministic decision for one gate.

    requested is the existing master mode/flag. A partial rollout without any
    stable identity is deliberately ineligible rather than randomly changing
    behavior between calls. Zero and 100 percent require no identity.
    """
    if gate not in _VALID_GATES:
        raise ValueError(f"Unknown P4 rollout gate: {gate!r}")
    try:
        bounded_percent = int(percent)
    except (TypeError, ValueError) as exc:
        raise ValueError("P4 rollout percent must be an integer from 0 to 100") from exc
    if not 0 <= bounded_percent <= 100:
        raise ValueError("P4 rollout percent must be from 0 to 100")
    if not requested:
        return RolloutDecision(False, "disabled", bounded_percent, None, "none")
    if bounded_percent <= 0:
        return RolloutDecision(False, "holdback", bounded_percent, None, "none")
    if bounded_percent >= 100:
        return RolloutDecision(True, "active", bounded_percent, None, "all")

    key_source, identity = _identity_source(
        actor_id=actor_id, session_id=session_id, request_id=request_id,
    )
    if not identity:
        return RolloutDecision(False, "ineligible", bounded_percent, None, key_source)

    digest = hashlib.sha256(
        f"{_HASH_NAMESPACE}\0{gate}\0{key_source}\0{identity}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % _BUCKET_COUNT
    enabled = bucket < bounded_percent * 100
    return RolloutDecision(
        enabled,
        "active" if enabled else "holdback",
        bounded_percent,
        bucket,
        key_source,
    )


def build_rollout_decisions(
    gate_settings: Mapping[str, tuple[bool, int]],
    *,
    actor_id: str = "",
    session_id: str = "",
    request_id: str = "",
) -> dict[str, RolloutDecision]:
    """Resolve and count all requested P4 gates once at request creation."""
    decisions: dict[str, RolloutDecision] = {}
    for gate, (requested, percent) in gate_settings.items():
        decision = decide_rollout(
            gate,
            requested=requested,
            percent=percent,
            actor_id=actor_id,
            session_id=session_id,
            request_id=request_id,
        )
        decisions[gate] = decision
        metrics.log_p4_rollout(gate, decision.status)
    return decisions


def gate_is_active(ctx, gate: str, *, default: bool) -> bool:
    """Read a request decision, preserving direct-call/test compatibility."""
    decisions = getattr(ctx, "p4_rollout_decisions", None)
    if not decisions or gate not in decisions:
        return default
    decision = decisions[gate]
    if isinstance(decision, RolloutDecision):
        return decision.enabled
    # Accept a boolean for lightweight foreign contexts and focused tests.
    if isinstance(decision, bool):
        return decision
    # A present but malformed assignment must never broaden the cohort.
    return False
