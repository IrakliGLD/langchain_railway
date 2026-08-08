"""Shared row-projection tests for report prompt and grounding scopes."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.report_projection import projected_row_indices
from contracts.report_evidence import ReportEvidenceManifest
from tests.fixtures_report_manifest import _manifest


def _wide_table(row_count: int):
    payload = _manifest().model_dump(mode="json")
    table = payload["items"][0]
    table["rows"] = [
        {"period": f"2026-{month:02d}", "price": 100.0 + month}
        for month in range(1, row_count + 1)
    ]
    table["total_row_count"] = row_count
    return ReportEvidenceManifest.model_validate(payload).items[0]


def test_projection_keeps_every_row_when_the_budget_allows():
    table = _manifest().items[0]

    assert projected_row_indices(table, budget_chars=30_000) == [0, 1]


def test_projection_returns_nothing_for_a_narrative_item():
    narrative = _manifest().items[1]

    assert projected_row_indices(narrative, budget_chars=30_000) == []


def test_projection_is_deterministic_and_sorted_under_budget_pressure():
    table = _wide_table(12)

    first = projected_row_indices(table, budget_chars=200)
    second = projected_row_indices(table, budget_chars=200)

    assert first == second
    assert first == sorted(first)
    assert 0 < len(first) < 12


def test_projection_keeps_both_boundaries_before_interior_rows():
    table = _wide_table(12)

    selected = projected_row_indices(table, budget_chars=200)

    assert selected[0] == 0
    assert selected[-1] == 11


def test_projection_admits_more_rows_as_the_budget_grows():
    table = _wide_table(12)

    narrow = projected_row_indices(table, budget_chars=200)
    wide = projected_row_indices(table, budget_chars=600)

    assert len(wide) > len(narrow)
    assert set(narrow) <= set(wide)
