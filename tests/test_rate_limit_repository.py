"""Behavior contract for the P8.A in-memory rate-limit repository."""

from __future__ import annotations

from utils.rate_limits import InMemoryRateLimitRepository


def test_namespaces_have_independent_sliding_window_budgets():
    now = [100.0]
    repository = InMemoryRateLimitRepository(time_fn=lambda: now[0])

    assert repository.consume("preauth", "subject", max_requests=1) is True
    assert repository.consume("preauth", "subject", max_requests=1) is False
    assert repository.consume("gateway", "subject", max_requests=1) is True

    now[0] = 161.0
    assert repository.consume("preauth", "subject", max_requests=1) is True


def test_stale_subjects_are_evicted_on_the_bounded_sweep_schedule():
    now = [100.0]
    repository = InMemoryRateLimitRepository(
        time_fn=lambda: now[0],
        sweep_interval_seconds=300.0,
    )
    assert repository.consume("preauth", "stale", max_requests=2) is True

    now[0] = 401.0
    assert repository.consume("preauth", "fresh", max_requests=2) is True

    assert repository.protected_snapshot() == {
        "backend": "in_memory",
        "shared_across_processes": False,
        "subjects_by_namespace": {"gateway": 0, "preauth": 1, "user": 0},
    }


def test_clear_removes_subject_state_without_exposing_identifiers():
    repository = InMemoryRateLimitRepository(time_fn=lambda: 100.0)
    repository.consume("user", "user:private", max_requests=2)

    repository.clear()

    snapshot = repository.protected_snapshot()
    assert snapshot["subjects_by_namespace"]["user"] == 0
    assert "private" not in repr(snapshot)
