import time

import pytest

from agent import pipeline
from models import QueryContext
from utils.request_deadline import RequestDeadline, RequestDeadlineExceeded


def test_evidence_loop_receives_remaining_request_budget(monkeypatch):
    captured = {}
    deadline = RequestDeadline.from_budget_ms(
        budget_ms=5000,
        now_monotonic=time.monotonic(),
        source="test",
    )
    ctx = QueryContext(
        query="Show balancing price trend in 2024.",
        request_deadline=deadline,
    )
    ctx.evidence_plan = [
        {
            "role": "primary_data",
            "tool_name": "get_prices",
            "params": {},
            "satisfied": False,
        }
    ]

    def _capture_evidence_loop(ctx_arg, timeout_seconds=None):
        captured["timeout_seconds"] = timeout_seconds
        ctx_arg.evidence_plan[0]["satisfied"] = True
        ctx_arg.evidence_plan_complete = True
        return ctx_arg

    monkeypatch.setattr(pipeline, "ENABLE_TYPED_TOOLS", False)
    monkeypatch.setattr(pipeline, "ENABLE_EVIDENCE_PLANNER", True)
    monkeypatch.setattr(
        pipeline.evidence_planner,
        "execute_remaining_evidence",
        _capture_evidence_loop,
    )

    pipeline._execute_evidence_plan(ctx)

    assert captured["timeout_seconds"] is not None
    assert 0 < captured["timeout_seconds"] <= 5


def test_process_query_rejects_expired_budget_before_prepare(monkeypatch):
    deadline = RequestDeadline.from_budget_ms(
        budget_ms=0,
        now_monotonic=time.monotonic(),
        source="test",
    )
    monkeypatch.setattr(
        pipeline.planner,
        "prepare_context",
        lambda _ctx: pytest.fail("prepare must not run after deadline exhaustion"),
    )

    with pytest.raises(RequestDeadlineExceeded) as exc:
        pipeline.process_query("Show balancing price trend.", request_deadline=deadline)

    assert exc.value.stage == "stage_0_prepare_context"
