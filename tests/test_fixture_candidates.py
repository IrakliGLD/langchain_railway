"""Tests for the self-growing golden set (design item 6): emission + harvest."""

from __future__ import annotations

import json
import logging
import os

import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from types import SimpleNamespace

from agent import fixture_candidates as fixture_candidates_module
from agent.fixture_candidates import MARKER, log_fixture_candidate, routed_fields_snapshot
from evaluation.harvest_fixture_candidates import parse_candidates, to_cases


def _emit_and_capture(caplog, trigger: str, ctx) -> list[str]:
    with caplog.at_level(logging.INFO, logger="Enai"):
        log_fixture_candidate(trigger, ctx)
    return [rec.getMessage() for rec in caplog.records if MARKER in rec.getMessage()]


@pytest.fixture(autouse=True)
def _enable_local_fixture_capture(monkeypatch):
    monkeypatch.setattr(fixture_candidates_module, "FIXTURE_CAPTURE_MODE", "raw")
    monkeypatch.setattr(fixture_candidates_module, "FIXTURE_CAPTURE_SAMPLE_RATE", 1.0)


def test_fixture_capture_is_off_by_default(monkeypatch, caplog):
    monkeypatch.setattr(fixture_candidates_module, "FIXTURE_CAPTURE_MODE", "off")
    lines = _emit_and_capture(
        caplog,
        "cross_check_disagreement",
        SimpleNamespace(query="must stay private", question_analysis=None, trace_id="trace-off"),
    )
    assert lines == []


def test_emission_round_trips_through_harvester(caplog):
    ctx = SimpleNamespace(
        query="Why did the balancing price change?\nsecond line",
        question_analysis=None,
        trace_id="t-123",
    )
    lines = _emit_and_capture(caplog, "cross_check_disagreement", ctx)
    assert len(lines) == 1

    candidates = parse_candidates(lines)
    assert len(candidates) == 1
    assert candidates[0]["trigger"] == "cross_check_disagreement"
    # Newlines collapsed so the log line stays single-line parseable.
    assert candidates[0]["query"] == "Why did the balancing price change? second line"
    assert candidates[0]["trace_id"] == "t-123"

    cases = to_cases(candidates)
    assert cases[0]["id"] == "cand_001"
    assert cases[0]["expected"] == {}
    assert "VERIFY expected before adopting" in cases[0]["note"]


def test_non_ascii_query_survives(caplog):
    ctx = SimpleNamespace(query="რა იყო ფასი 2024 წელს?", question_analysis=None, trace_id="")
    lines = _emit_and_capture(caplog, "provenance_gate_failure", ctx)
    candidates = parse_candidates(lines)
    assert candidates[0]["query"] == "რა იყო ფასი 2024 წელს?"


def test_harvester_dedupes_by_query():
    payload = json.dumps({"trigger": "x", "query": "same q", "routed": {}, "trace_id": ""})
    lines = [f"prefix {MARKER} {payload}", f"{MARKER} {payload}", "unrelated line"]
    assert len(parse_candidates(lines)) == 1


def test_harvester_skips_malformed_json():
    lines = [f"{MARKER} not-json", f"{MARKER} " + json.dumps({"query": "ok"})]
    candidates = parse_candidates(lines)
    assert len(candidates) == 1
    assert candidates[0]["query"] == "ok"


def test_routed_fields_snapshot_none_is_empty():
    assert routed_fields_snapshot(None) == {}


def test_emission_never_raises_on_broken_ctx(caplog):
    class Exploding:
        @property
        def query(self):
            raise RuntimeError("boom")

    log_fixture_candidate("cross_check_disagreement", Exploding())  # must not raise
