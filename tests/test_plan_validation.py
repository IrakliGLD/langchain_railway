"""Tests for P4.2 typed plan validation and its enforcement stage (finding M3).

The validator must return a typed PlanValidationResult (rejects vs warnings),
keep its existing warning logs, and never mutate the plan. Enforcement is the
pipeline's job: pass-through in the default ``warn`` mode, terminal clarify
BEFORE any tool/DB call in ``enforce`` mode.
"""

from __future__ import annotations

import os

# Ensure config validation passes before importing project modules.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import inspect

import pytest

from agent import pipeline
from agent.evidence_planner import (
    SEVERITY_REJECT,
    SEVERITY_WARN,
    PlanValidationResult,
    _validate_plan_against_answer_kind,
    build_evidence_plan,
)
from contracts.question_analysis import AnswerKind, QuestionAnalysis, RenderStyle
from models import QueryContext, ResolutionPolicy
from utils.metrics import metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_qa_payload(
    query_type: str = "data_retrieval",
    tools: list | None = None,
    answer_kind: str | None = None,
    derived_metrics: list | None = None,
) -> dict:
    if tools is None:
        tools = [{"name": "get_prices", "score": 0.9, "reason": "price data"}]
    payload = {
        "version": "question_analysis_v1",
        "raw_query": "test query",
        "canonical_query_en": "test query in English",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": query_type,
            "analysis_mode": "light",
            "intent": "test intent",
            "needs_clarification": False,
            "confidence": 0.9,
        },
        "routing": {
            "preferred_path": "tool",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": True,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {},
        "tooling": {"candidate_tools": tools},
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": derived_metrics or [],
        },
    }
    if answer_kind is not None:
        payload["answer_kind"] = answer_kind
    return payload


def _qa(answer_kind: str | None = None, **kwargs) -> QuestionAnalysis:
    return QuestionAnalysis(**_make_qa_payload(answer_kind=answer_kind, **kwargs))


def _primary_step(tool_name: str = "get_prices", params: dict | None = None) -> dict:
    return {
        "role": "primary_data",
        "tool_name": tool_name,
        "params": params or {},
        "satisfied": False,
        "source": "planner",
    }


# ---------------------------------------------------------------------------
# Typed validator results
# ---------------------------------------------------------------------------


class TestValidatorTypedResult:
    def test_comparison_single_source_without_range_warns(self):
        # No explicit range: the tool's default recent-history window can
        # still produce multi-period rows, so this must NOT reject.
        qa = _qa(answer_kind="comparison")
        steps = [_primary_step(params={"metric": "balancing"})]
        result = _validate_plan_against_answer_kind(steps, qa, "compare balancing price")
        assert [i.rule for i in result.warnings] == ["comparison_single_source"]
        assert result.rejects == []

    def test_comparison_single_point_range_is_reject(self):
        # An explicit degenerate range pins one period for one comparand —
        # provably not comparable.
        qa = _qa(answer_kind="comparison")
        steps = [_primary_step(params={
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
        })]
        result = _validate_plan_against_answer_kind(steps, qa, "compare january to january")
        assert [i.rule for i in result.rejects] == ["comparison_single_point_range"]
        assert result.rejects[0].severity == SEVERITY_REJECT

    def test_comparison_multi_period_range_passes(self):
        qa = _qa(answer_kind="comparison")
        steps = [_primary_step(params={"start_date": "2024-01-01", "end_date": "2024-06-30"})]
        result = _validate_plan_against_answer_kind(steps, qa, "jan vs june prices")
        assert result.rejects == []

    def test_comparison_two_entities_passes(self):
        qa = _qa(answer_kind="comparison")
        steps = [_primary_step(tool_name="get_tariffs", params={"entities": ["enguri_hpp", "vardnili_hpp"]})]
        result = _validate_plan_against_answer_kind(steps, qa, "enguri vs vardnili tariff")
        assert result.rejects == []

    def test_comparison_two_sources_passes(self):
        qa = _qa(answer_kind="comparison")
        steps = [
            _primary_step(params={"metric": "balancing"}),
            {
                "role": "correlation_driver",
                "tool_name": "get_balancing_composition",
                "params": {},
                "satisfied": False,
                "source": "planner",
            },
        ]
        result = _validate_plan_against_answer_kind(steps, qa, "price vs composition")
        assert result.rejects == []

    def test_timeseries_missing_range_warns_not_rejects(self):
        qa = _qa(answer_kind="timeseries")
        steps = [_primary_step(params={"metric": "balancing"})]
        result = _validate_plan_against_answer_kind(steps, qa, "recent balancing prices")
        assert [i.rule for i in result.warnings] == ["timeseries_missing_range"]
        assert result.rejects == []

    def test_timeseries_single_point_range_warns(self):
        qa = _qa(answer_kind="timeseries")
        steps = [_primary_step(params={"start_date": "2024-01-01", "end_date": "2024-01-01"})]
        result = _validate_plan_against_answer_kind(steps, qa, "prices in january")
        assert [i.rule for i in result.warnings] == ["timeseries_single_point_range"]
        assert result.rejects == []

    def test_forecast_missing_range_warns(self):
        qa = _qa(answer_kind="forecast")
        steps = [_primary_step(params={"metric": "balancing"})]
        result = _validate_plan_against_answer_kind(steps, qa, "forecast balancing price")
        assert [i.rule for i in result.warnings] == ["forecast_missing_range"]
        assert result.rejects == []

    def test_scenario_missing_metric_is_reject(self):
        qa = _qa(answer_kind="scenario")
        steps = [_primary_step(params={"metric": "balancing"})]
        result = _validate_plan_against_answer_kind(steps, qa, "what if prices double")
        assert [i.rule for i in result.rejects] == ["scenario_missing_metric"]

    def test_scenario_with_metric_passes(self):
        qa = _qa(
            answer_kind="scenario",
            derived_metrics=[{
                "metric_name": "scenario_scale",
                "metric": "p_bal_gel",
                "scenario_factor": 1.2,
            }],
        )
        steps = [_primary_step(params={"metric": "balancing"})]
        result = _validate_plan_against_answer_kind(steps, qa, "what if prices rise 20%")
        assert result.rejects == []

    def test_list_not_enumerated_warns(self):
        qa = _qa(answer_kind="list")
        steps = [_primary_step(params={"metric": "balancing"})]
        result = _validate_plan_against_answer_kind(steps, qa, "list the prices")
        assert "list_not_enumerated" in [i.rule for i in result.warnings]
        assert result.rejects == []

    def test_list_tariffs_default_subset_warns(self):
        # get_tariffs with no entities after repair: inherently enumerated,
        # but the silent default subset deserves its own signal.
        qa = _qa(answer_kind="list")
        steps = [_primary_step(tool_name="get_tariffs", params={})]
        result = _validate_plan_against_answer_kind(steps, qa, "list regulated tariffs")
        rules = [i.rule for i in result.warnings]
        assert "list_tariffs_default_subset" in rules
        assert "list_not_enumerated" not in rules
        assert result.rejects == []

    def test_empty_steps_return_empty_result(self):
        qa = _qa(answer_kind="comparison")
        result = _validate_plan_against_answer_kind([], qa, "anything")
        assert result.issues == []

    def test_validator_never_mutates_steps(self):
        qa = _qa(answer_kind="comparison")
        steps = [_primary_step(params={"metric": "balancing"})]
        before = [dict(s) for s in steps]
        _validate_plan_against_answer_kind(steps, qa, "compare")
        assert steps == before

    def test_issue_messages_contain_no_query_text(self):
        marker = "SECRET-USER-QUERY-MARKER"
        qa = _qa(answer_kind="comparison")
        steps = [_primary_step(params={"metric": "balancing"})]
        result = _validate_plan_against_answer_kind(steps, qa, marker)
        assert result.issues, "expected at least one issue"
        for issue in result.issues:
            assert marker not in issue.message

    def test_deterministic_with_narrative_steps_warns(self):
        qa = _qa(answer_kind="scalar")
        qa.render_style = RenderStyle.DETERMINISTIC
        steps = [
            _primary_step(params={"metric": "balancing"}),
            {
                "role": "composition_context",
                "tool_name": "get_balancing_composition",
                "params": {},
                "satisfied": False,
                "source": "planner",
            },
        ]
        result = _validate_plan_against_answer_kind(steps, qa, "current price")
        assert "deterministic_with_narrative_steps" in [i.rule for i in result.warnings]
        assert result.rejects == []


# ---------------------------------------------------------------------------
# Stage 0.4 integration: build_evidence_plan publishes the typed result
# ---------------------------------------------------------------------------


class TestBuildPlanPublishesValidation:
    def test_result_stored_on_ctx_and_counted(self):
        payload = _make_qa_payload(answer_kind="timeseries")
        ctx = QueryContext(query="recent balancing prices")
        ctx.question_analysis = QuestionAnalysis(**payload)
        ctx.question_analysis_source = "llm_active"

        before = dict(metrics.plan_validation_events)
        ctx = build_evidence_plan(ctx)

        assert isinstance(ctx.plan_validation, PlanValidationResult)
        rules = [i.rule for i in ctx.plan_validation.issues]
        assert "timeseries_missing_range" in rules
        key = f"timeseries_missing_range:{SEVERITY_WARN}"
        assert metrics.plan_validation_events.get(key, 0) == before.get(key, 0) + 1

    def test_non_authoritative_analysis_leaves_validation_none(self):
        ctx = QueryContext(query="test")
        ctx = build_evidence_plan(ctx)
        assert ctx.plan_validation is None


# ---------------------------------------------------------------------------
# Enforcement stage
# ---------------------------------------------------------------------------


def _reject_result() -> PlanValidationResult:
    result = PlanValidationResult()
    result.add("comparison_single_point_range", SEVERITY_REJECT, "single pinned period")
    return result


def _warn_result() -> PlanValidationResult:
    result = PlanValidationResult()
    result.add("timeseries_missing_range", SEVERITY_WARN, "no range")
    return result


class TestEnforcementStage:
    def test_warn_mode_passes_through_rejects(self, monkeypatch):
        monkeypatch.setattr(pipeline, "PLAN_VALIDATION_MODE", "warn")
        ctx = QueryContext(query="compare balancing price")
        ctx.plan_validation = _reject_result()
        res = pipeline._enforce_plan_validation_stage(ctx)
        assert res.terminal is False
        assert ctx.summary == ""

    def test_enforce_mode_clarifies_before_execution(self, monkeypatch):
        monkeypatch.setattr(pipeline, "PLAN_VALIDATION_MODE", "enforce")
        ctx = QueryContext(query="compare balancing price")
        ctx.plan_validation = _reject_result()

        before = dict(metrics.plan_validation_events)
        res = pipeline._enforce_plan_validation_stage(ctx)

        assert res.terminal is True
        assert res.ctx.summary_source == "clarification_request"
        assert res.ctx.clarify_reason == "plan_validation_comparison_single_point_range"
        assert res.ctx.resolution_policy == ResolutionPolicy.CLARIFY
        assert res.ctx.summary
        key = "comparison_single_point_range:enforced"
        assert metrics.plan_validation_events.get(key, 0) == before.get(key, 0) + 1

    def test_enforce_mode_holdback_remains_warning_only(self, monkeypatch):
        monkeypatch.setattr(pipeline, "PLAN_VALIDATION_MODE", "enforce")
        ctx = QueryContext(query="compare balancing price")
        ctx.plan_validation = _reject_result()
        ctx.p4_rollout_decisions["plan_validation"] = False

        res = pipeline._enforce_plan_validation_stage(ctx)

        assert res.terminal is False
        assert ctx.summary == ""

    def test_enforce_mode_ignores_warn_only_result(self, monkeypatch):
        monkeypatch.setattr(pipeline, "PLAN_VALIDATION_MODE", "enforce")
        ctx = QueryContext(query="recent prices")
        ctx.plan_validation = _warn_result()
        res = pipeline._enforce_plan_validation_stage(ctx)
        assert res.terminal is False

    def test_missing_validation_result_passes_through(self, monkeypatch):
        monkeypatch.setattr(pipeline, "PLAN_VALIDATION_MODE", "enforce")
        ctx = QueryContext(query="anything")
        res = pipeline._enforce_plan_validation_stage(ctx)
        assert res.terminal is False

    def test_enforcement_runs_before_any_execution_in_process_query(self):
        """Zero-tool-call guarantee: enforcement precedes evidence execution
        and the SQL fallback in the process_query source order."""
        source = inspect.getsource(pipeline.process_query)
        enforce_at = source.index("_enforce_plan_validation_stage")
        execute_at = source.index("_execute_evidence_plan(")
        sql_at = source.index("_run_generate_sql_stage")
        assert enforce_at < execute_at
        assert enforce_at < sql_at

    def test_default_mode_is_warn(self):
        import config

        assert config.PLAN_VALIDATION_MODE == "warn"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
