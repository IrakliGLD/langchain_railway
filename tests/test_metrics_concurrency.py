"""Tests for P5.5 thread-safe metrics and per-source tool observability (M22, L1).

Concurrent requests mutate the shared Metrics aggregates; updates must not be
lost and snapshots must be internally consistent. Stage 0.8 secondary tool
calls must be observable per tool without double-counting the legacy total.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import threading

import pandas as pd
import pytest

from utils.metrics import Metrics, metrics

# ---------------------------------------------------------------------------
# M22: exact totals under concurrency
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_counter_increments_are_exact(self):
        m = Metrics()
        threads = 8
        per_thread = 500

        def _hammer():
            for _ in range(per_thread):
                m.log_tool_error()
                m.log_plan_validation("rule_x", "warn")
                m.log_llm_usage(
                    model_name="test-model",
                    prompt_tokens=3,
                    completion_tokens=2,
                    total_tokens=5,
                    estimated_cost_usd=0.001,
                )

        workers = [threading.Thread(target=_hammer) for _ in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        expected = threads * per_thread
        assert m.tool_error_count == expected
        assert m.plan_validation_events["rule_x:warn"] == expected
        assert m.llm_total_tokens == expected * 5
        assert m.llm_usage_by_model["test-model"]["calls"] == expected
        assert m.llm_usage_by_model["test-model"]["total_tokens"] == expected * 5

    def test_snapshot_is_consistent_under_concurrent_mutation(self):
        m = Metrics()
        stop = threading.Event()
        errors: list[BaseException] = []

        def _mutate():
            while not stop.is_set():
                m.log_stage("stage_x", 1.0)
                m.log_terminal_outcome("data_answer")

        def _snapshot():
            try:
                for _ in range(200):
                    stats = m.get_stats()
                    # Internal consistency: every stage key present in counts
                    # has an average computable from the same snapshot.
                    for key in stats["stage_counts"]:
                        assert key in stats["stage_avg_ms"]
            except BaseException as exc:  # noqa: BLE001 - surface to main thread
                errors.append(exc)

        mutators = [threading.Thread(target=_mutate) for _ in range(4)]
        reader = threading.Thread(target=_snapshot)
        for t in mutators:
            t.start()
        reader.start()
        reader.join()
        stop.set()
        for t in mutators:
            t.join()

        assert errors == []

    def test_module_singleton_has_lock(self):
        assert isinstance(metrics._lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# L1: per-source tool observability
# ---------------------------------------------------------------------------


class TestToolCallSource:
    def test_counter_and_latency_keyed_by_tool_and_source(self):
        m = Metrics()
        m.log_tool_call_source("get_prices", "primary", 0.5)
        m.log_tool_call_source("get_prices", "secondary", 0.25)
        m.log_tool_call_source("get_prices", "secondary", 0.25)

        assert m.tool_calls_by_source["get_prices:primary"] == 1
        assert m.tool_calls_by_source["get_prices:secondary"] == 2
        assert m.tool_time_by_source["get_prices:secondary"] == pytest.approx(0.5)
        stats = m.get_stats()
        assert stats["tool_calls_by_source"]["get_prices:secondary"] == 2

    def test_secondary_execution_counts_source_but_not_legacy_total(self):
        from agent import pipeline
        from agent.tools.types import ToolInvocation
        from models import QueryContext

        ctx = QueryContext(query="secondary evidence")
        ctx.evidence_plan = [{
            "role": "composition_context",
            "tool_name": "get_balancing_composition",
            "params": {},
            "satisfied": False,
            "source": "planner",
        }]
        invocation = ToolInvocation(
            name="get_balancing_composition", params={}, confidence=0.9, reason="test",
        )
        df = pd.DataFrame({"date": ["2024-01-01"], "share_import": [0.2]})

        def _executor(_invocation):
            return df, list(df.columns), [tuple(r) for r in df.itertuples(index=False, name=None)]

        before_total = metrics.tool_call_count
        before_secondary = metrics.tool_calls_by_source.get(
            "get_balancing_composition:secondary", 0,
        )

        stored = pipeline._execute_evidence_step(
            ctx,
            invocation,
            plan_step=ctx.evidence_plan[0],
            is_primary=False,
            is_explanation=False,
            stage_label="stage_0_8",
            validate_relevance=False,
            emit_tool_call_metric=False,
            emit_tool_execute_trace=False,
            executor=_executor,
        )

        assert stored is True
        assert metrics.tool_call_count == before_total  # legacy total untouched
        assert metrics.tool_calls_by_source.get(
            "get_balancing_composition:secondary", 0,
        ) == before_secondary + 1

    def test_primary_execution_counts_both(self):
        from agent import pipeline
        from agent.tools.types import ToolInvocation
        from models import QueryContext

        ctx = QueryContext(query="plot recent balancing prices")
        invocation = ToolInvocation(
            name="get_prices", params={"metric": "balancing"}, confidence=0.9, reason="test",
        )
        df = pd.DataFrame({"date": ["2024-01-01"], "p_bal_gel": [10.0]})

        def _executor(_invocation):
            return df, list(df.columns), [tuple(r) for r in df.itertuples(index=False, name=None)]

        before_total = metrics.tool_call_count
        before_primary = metrics.tool_calls_by_source.get("get_prices:primary", 0)

        pipeline._execute_evidence_step(
            ctx,
            invocation,
            plan_step=None,
            is_primary=True,
            is_explanation=False,
            stage_label="stage_0_5",
            executor=_executor,
        )

        assert metrics.tool_call_count == before_total + 1
        assert metrics.tool_calls_by_source.get("get_prices:primary", 0) == before_primary + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
