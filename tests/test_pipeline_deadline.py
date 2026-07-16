import threading
import time

import pandas as pd
import pytest

from agent import pipeline
from models import QueryContext
from utils.request_deadline import (
    RequestDeadline,
    RequestDeadlineExceeded,
    bind_request_execution_scope,
    current_request_execution_scope,
)


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


def test_parallel_secondary_calls_inherit_identity_and_narrower_deadline(monkeypatch):
    parent_deadline = RequestDeadline.from_budget_ms(
        budget_ms=5000,
        now_monotonic=time.monotonic(),
        source="test-parent",
    )
    observed = []
    observed_lock = threading.Lock()

    def _capture(invocation):
        scope = current_request_execution_scope()
        assert scope is not None
        assert scope.deadline is not None
        with observed_lock:
            observed.append(
                (
                    scope.request_id,
                    scope.actor_binding,
                    scope.deadline.source,
                    scope.deadline.deadline_monotonic,
                )
            )
        frame = pd.DataFrame(
            {"date": ["2024-01-01"], f"value_{invocation.name}": [1.0]}
        )
        return frame, list(frame.columns), [tuple(frame.iloc[0])]

    monkeypatch.setattr(pipeline, "EVIDENCE_PARALLEL_SECONDARY", True)
    monkeypatch.setattr(pipeline.evidence_planner, "execute_tool", _capture)
    ctx = QueryContext(query="test", request_deadline=parent_deadline)
    ctx.tool_name = "get_prices"
    ctx.df = pd.DataFrame({"date": ["2024-01-01"], "price": [50.0]})
    ctx.cols = list(ctx.df.columns)
    ctx.rows = [tuple(ctx.df.iloc[0])]
    ctx.evidence_plan = [
        {
            "role": "primary_data",
            "tool_name": "get_prices",
            "params": {},
            "satisfied": True,
        },
        {
            "role": "secondary_one",
            "tool_name": "get_tariffs",
            "params": {},
            "satisfied": False,
        },
        {
            "role": "secondary_two",
            "tool_name": "get_generation_mix",
            "params": {},
            "satisfied": False,
        },
    ]

    with bind_request_execution_scope(
        deadline=parent_deadline,
        request_id="request-child-deadline",
        actor_id="actor-child-deadline",
    ) as parent_scope:
        pipeline.evidence_planner.execute_remaining_evidence(
            ctx,
            timeout_seconds=0.5,
        )

    assert len(observed) == 2
    assert all(item[0] == "request-child-deadline" for item in observed)
    assert all(item[1] == parent_scope.actor_binding for item in observed)
    assert all(item[2] == "secondary_evidence" for item in observed)
    assert all(item[3] <= parent_deadline.deadline_monotonic for item in observed)
