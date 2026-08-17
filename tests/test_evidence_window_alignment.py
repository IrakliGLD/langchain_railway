"""Both sides of a comparison must span the same window.

Incident 2026-08-17. On the make-or-buy turns the retail primary fetched 66
monthly rows (its analyzer-invented window having been cleared, deliberately, by
``planner.resolve_make_or_buy_window``) while the wholesale companion fetched
11 -- because ``_resolve_secondary_params`` inherits the primary's window only
when the companion does not already have one of its own. The companion kept the
window the primary had just rejected, and the comparison was computed across
mismatched supports.

Alignment is the stated purpose of that inheritance block. These tests pin it
symmetrically: the primary's window governs, including when the primary
deliberately has none.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.evidence_planner import align_secondary_window  # noqa: E402


def test_companion_inherits_the_primary_window():
    """The existing behaviour, kept: an unset companion window is filled in."""
    params = align_secondary_window(
        {"metric": "balancing"},
        primary_params={"start_date": "2025-01-01", "end_date": "2025-12-31"},
    )

    assert params["start_date"] == "2025-01-01"
    assert params["end_date"] == "2025-12-31"


def test_companion_window_is_cleared_when_the_primary_has_none():
    """A window the primary rejected must not survive on the other side."""
    params = align_secondary_window(
        {"metric": "balancing", "start_date": "2026-08-01", "end_date": "2026-08-17"},
        primary_params={"currency": "gel"},
    )

    assert "start_date" not in params
    assert "end_date" not in params


def test_primary_window_overrides_a_differing_companion_window():
    """Two different windows are never both right for one comparison."""
    params = align_secondary_window(
        {"start_date": "2026-08-01", "end_date": "2026-08-17"},
        primary_params={"start_date": "2021-01-01", "end_date": "2026-07-31"},
    )

    assert params["start_date"] == "2021-01-01"
    assert params["end_date"] == "2026-07-31"


def test_alignment_leaves_non_window_params_alone():
    params = align_secondary_window(
        {"metric": "balancing", "currency": "usd", "start_date": "2026-08-01"},
        primary_params={"currency": "gel"},
    )

    assert params["metric"] == "balancing"
    assert params["currency"] == "usd"


def test_alignment_does_not_mutate_the_caller_dict():
    original = {"metric": "balancing", "start_date": "2026-08-01"}

    align_secondary_window(original, primary_params={})

    assert original["start_date"] == "2026-08-01"
