"""Per-stage provider timeouts.

Report generation runs in the durable worker, not behind a waiting user, and
its repair prompts carry the rejected draft on top of the full evidence packet.
The synchronous /ask timeout is the wrong budget for that stage.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import core.llm as llm
from utils.request_deadline import (
    bind_request_execution_scope,
    cap_request_deadline,
)


def test_report_stages_receive_a_longer_floor_than_the_synchronous_path(monkeypatch):
    monkeypatch.setattr(llm, "_configured_provider_timeout_seconds", lambda _p: 45.0)

    for report_stage in (
        "report_planner",
        "report_plan_repair",
        "report_section_writer_exec_summary",
        "report_section_repair_exec_summary_attempt_2",
    ):
        assert llm._effective_provider_timeout_seconds("nvidia", report_stage) == 120.0


def test_synchronous_stages_keep_their_configured_timeout(monkeypatch):
    monkeypatch.setattr(llm, "_configured_provider_timeout_seconds", lambda _p: 45.0)

    for interactive_stage in ("question_analyzer", "summarizer", "router", "llm"):
        assert llm._effective_provider_timeout_seconds("nvidia", interactive_stage) == 45.0


def test_a_short_job_deadline_still_bounds_the_report_floor(monkeypatch):
    monkeypatch.setattr(llm, "_configured_provider_timeout_seconds", lambda _p: 45.0)

    deadline = cap_request_deadline(maximum_seconds=20, source="test_report_job")
    with bind_request_execution_scope(
        deadline=deadline,
        request_id="report:test",
        actor_id="actor",
    ):
        bounded = llm._effective_provider_timeout_seconds("nvidia", "report_planner")

    assert 0 < bounded <= 20.0
