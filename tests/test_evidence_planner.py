"""Tests for agent.evidence_planner (Stage 0.4 / 0.8)."""

from __future__ import annotations

import os

# Ensure config validation passes before importing project modules.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

import pandas as pd

from contracts.question_analysis import QuestionAnalysis
from models import QueryContext
from agent.evidence_planner import (
    build_evidence_plan,
    execute_remaining_evidence,
    has_unsatisfied_steps,
    merge_evidence_into_context,
    next_unsatisfied_step,
)
from agent.planner import resolve_tool_params
from agent.summarizer import _add_evidence_record_tokens, _tokenize_cell_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qa_payload(
    query_type: str = "data_retrieval",
    preferred_path: str = "tool",
    tools: list | None = None,
    needs_driver: bool = False,
    needs_correlation: bool = False,
    needs_multi_tool: bool = False,
    evidence_roles: list | None = None,
    derived_metrics: list | None = None,
    period: dict | None = None,
) -> dict:
    """Build a minimal valid QuestionAnalysis payload for testing."""
    if tools is None:
        tools = [{"name": "get_prices", "score": 0.9, "reason": "price data"}]

    return {
        "version": "question_analysis_v1",
        "raw_query": "test query",
        "canonical_query_en": "test query in English",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": query_type,
            "analysis_mode": "analyst" if needs_driver else "light",
            "intent": "test intent",
            "needs_clarification": False,
            "confidence": 0.9,
        },
        "routing": {
            "preferred_path": preferred_path,
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": True,
            "needs_multi_tool": needs_multi_tool,
            "evidence_roles": evidence_roles or [],
        },
        "knowledge": {},
        "tooling": {"candidate_tools": tools},
        "sql_hints": {"period": period} if period else {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
        },
        "analysis_requirements": {
            "needs_driver_analysis": needs_driver,
            "needs_correlation_context": needs_correlation,
            "derived_metrics": derived_metrics or [],
        },
    }


def _ctx_with_qa(payload: dict) -> QueryContext:
    ctx = QueryContext(query=payload.get("raw_query", "test"))
    ctx.question_analysis = QuestionAnalysis(**payload)
    ctx.question_analysis_source = "llm_active"
    return ctx


# ---------------------------------------------------------------------------
# Stage 0.4: build_evidence_plan
# ---------------------------------------------------------------------------

class TestBuildEvidencePlan:
    def test_no_analyzer_returns_empty_plan(self):
        ctx = QueryContext(query="test")
        ctx = build_evidence_plan(ctx)

        assert ctx.evidence_plan == []
        assert ctx.evidence_plan_source == ""

    def test_shadow_analyzer_returns_empty_plan(self):
        payload = _make_qa_payload()
        ctx = _ctx_with_qa(payload)
        ctx.question_analysis_source = "llm_shadow"  # not authoritative
        ctx = build_evidence_plan(ctx)

        assert ctx.evidence_plan == []

    def test_single_tool_simple_query(self):
        payload = _make_qa_payload(
            query_type="data_retrieval",
            tools=[{"name": "get_prices", "score": 0.9, "reason": "price data"}],
        )
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 1
        assert ctx.evidence_plan[0]["tool_name"] == "get_prices"
        assert ctx.evidence_plan[0]["role"] == "primary_data"
        assert ctx.evidence_plan[0]["satisfied"] is False
        assert ctx.evidence_plan_source == "deterministic"


class TestResolveToolParams:
    def test_expands_single_month_explanation_window_for_derived_metrics(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            needs_driver=True,
            derived_metrics=[
                {"metric_name": "mom_absolute_change", "metric": "balancing"},
                {"metric_name": "mom_percent_change", "metric": "balancing"},
            ],
            period={
                "kind": "month",
                "start_date": "2024-05-01",
                "end_date": "2024-05-31",
                "granularity": "month",
                "raw_text": "May 2024",
            },
        )
        payload["raw_query"] = "Why did balancing price change in May 2024?"
        payload["canonical_query_en"] = "Explain the reasons for the change in balancing electricity price in May 2024."
        qa = QuestionAnalysis.model_validate(payload)

        params = resolve_tool_params(qa, "get_prices", payload["raw_query"])

        assert params is not None
        assert params["start_date"] == "2019-05-01"
        assert params["end_date"] == "2024-05-31"

    def test_keeps_single_month_window_without_derived_metrics(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            needs_driver=True,
            derived_metrics=[],
            period={
                "kind": "month",
                "start_date": "2024-05-01",
                "end_date": "2024-05-31",
                "granularity": "month",
                "raw_text": "May 2024",
            },
        )
        payload["raw_query"] = "Show balancing price in May 2024."
        payload["canonical_query_en"] = "Show balancing price in May 2024."
        qa = QuestionAnalysis.model_validate(payload)

        params = resolve_tool_params(qa, "get_prices", payload["raw_query"])

        assert params is not None
        assert params["start_date"] == "2024-05-01"
        assert params["end_date"] == "2024-05-31"

    def test_balancing_prices_with_driver_analysis_adds_composition_and_tariffs(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {"name": "get_prices", "score": 0.9, "reason": "price data"},
                {"name": "get_balancing_composition", "score": 0.7, "reason": "shares"},
                {"name": "get_tariffs", "score": 0.6, "reason": "tariffs"},
            ],
            needs_driver=True,
        )
        payload["raw_query"] = "Why did balancing electricity price change in 2023?"
        payload["canonical_query_en"] = payload["raw_query"]
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 3
        assert ctx.evidence_plan[0]["tool_name"] == "get_prices"
        assert ctx.evidence_plan[0]["role"] == "primary_data"
        assert ctx.evidence_plan[1]["tool_name"] == "get_balancing_composition"
        assert ctx.evidence_plan[1]["role"] == "composition_context"
        assert ctx.evidence_plan[2]["tool_name"] == "get_tariffs"
        assert ctx.evidence_plan[2]["role"] == "tariff_context"

    def test_prices_with_correlation_adds_composition_and_tariffs(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {"name": "get_prices", "score": 0.9, "reason": "prices"},
                {"name": "get_balancing_composition", "score": 0.7, "reason": "shares"},
                {"name": "get_tariffs", "score": 0.5, "reason": "tariffs"},
            ],
            needs_correlation=True,
        )
        payload["raw_query"] = "How are balancing electricity prices correlated with source costs?"
        payload["canonical_query_en"] = payload["raw_query"]
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        tool_names = [s["tool_name"] for s in ctx.evidence_plan]
        assert "get_prices" in tool_names
        assert "get_balancing_composition" in tool_names
        assert "get_tariffs" in tool_names
        assert len(ctx.evidence_plan) == 3

    def test_composition_with_driver_adds_prices(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {"name": "get_balancing_composition", "score": 0.9, "reason": "shares"},
                {"name": "get_prices", "score": 0.6, "reason": "price context"},
            ],
            needs_driver=True,
        )
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 2
        assert ctx.evidence_plan[0]["tool_name"] == "get_balancing_composition"
        assert ctx.evidence_plan[1]["tool_name"] == "get_prices"
        assert ctx.evidence_plan[1]["role"] == "correlation_driver"

    def test_generation_mix_with_driver_adds_prices(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {"name": "get_generation_mix", "score": 0.85, "reason": "generation"},
                {"name": "get_prices", "score": 0.6, "reason": "prices"},
            ],
            needs_driver=True,
        )
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 2
        assert ctx.evidence_plan[0]["tool_name"] == "get_generation_mix"
        assert ctx.evidence_plan[1]["tool_name"] == "get_prices"

    def test_explicit_multi_tool_with_roles(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {"name": "get_prices", "score": 0.9, "reason": "prices"},
                {"name": "get_balancing_composition", "score": 0.8, "reason": "drivers"},
            ],
            needs_multi_tool=True,
            evidence_roles=["primary_data", "composition_context"],
        )
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 2
        roles = [s["role"] for s in ctx.evidence_plan]
        assert "primary_data" in roles
        assert "composition_context" in roles

    def test_comparison_with_two_strong_candidates(self):
        payload = _make_qa_payload(
            query_type="comparison",
            tools=[
                {"name": "get_prices", "score": 0.85, "reason": "prices"},
                {"name": "get_tariffs", "score": 0.75, "reason": "tariffs"},
            ],
        )
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 2
        tool_names = [s["tool_name"] for s in ctx.evidence_plan]
        assert "get_prices" in tool_names
        assert "get_tariffs" in tool_names

    def test_comparison_weak_secondary_excluded(self):
        payload = _make_qa_payload(
            query_type="comparison",
            tools=[
                {
                    "name": "get_prices",
                    "score": 0.85,
                    "reason": "prices",
                    "params_hint": {"metric": "exchange_rate"},
                },
                {"name": "get_tariffs", "score": 0.30, "reason": "tariffs"},
            ],
        )
        payload["raw_query"] = "Compare exchange rate in 2023 and 2024"
        payload["canonical_query_en"] = payload["raw_query"]
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        # Weak secondary candidate should be excluded
        assert len(ctx.evidence_plan) == 1

    def test_no_candidate_tools_returns_empty(self):
        payload = _make_qa_payload(tools=[])
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert ctx.evidence_plan == []
        assert ctx.evidence_plan_source == ""

    def test_secondary_inherits_dates_from_primary(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {
                    "name": "get_prices",
                    "score": 0.9,
                    "reason": "prices",
                    "params_hint": {
                        "start_date": "2023-01-01",
                        "end_date": "2023-12-31",
                    },
                },
                {"name": "get_balancing_composition", "score": 0.7, "reason": "shares"},
                {"name": "get_tariffs", "score": 0.6, "reason": "tariffs"},
            ],
            needs_driver=True,
        )
        payload["raw_query"] = "Why did balancing price change in 2023?"
        payload["canonical_query_en"] = payload["raw_query"]
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 3
        for secondary in ctx.evidence_plan[1:]:
            assert secondary["params"]["start_date"] == "2023-01-01"
            assert secondary["params"]["end_date"] == "2023-12-31"

    def test_balancing_tariff_secondary_inherits_currency_and_category_entities(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {
                    "name": "get_prices",
                    "score": 0.9,
                    "reason": "prices",
                    "params_hint": {
                        "metric": "balancing",
                        "currency": "usd",
                        "start_date": "2023-01-01",
                        "end_date": "2023-12-31",
                    },
                },
                {"name": "get_balancing_composition", "score": 0.7, "reason": "shares"},
                {"name": "get_tariffs", "score": 0.6, "reason": "tariffs"},
            ],
            needs_driver=True,
        )
        payload["raw_query"] = "Why did balancing price change in 2023?"
        payload["canonical_query_en"] = payload["raw_query"]
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        tariff_step = next(s for s in ctx.evidence_plan if s["tool_name"] == "get_tariffs")
        assert tariff_step["params"]["currency"] == "usd"
        assert tariff_step["params"]["entities"] == [
            "regulated_hpp",
            "regulated_new_tpp",
            "regulated_old_tpp",
        ]

    def test_balancing_composition_secondary_does_not_inherit_currency(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {
                    "name": "get_prices",
                    "score": 0.9,
                    "reason": "prices",
                    "params_hint": {
                        "metric": "balancing",
                        "currency": "usd",
                        "start_date": "2023-12-01",
                        "end_date": "2024-01-31",
                    },
                },
                {"name": "get_balancing_composition", "score": 0.7, "reason": "shares"},
            ],
            needs_driver=True,
        )
        payload["raw_query"] = "Explain the reasons for the change in balancing price."
        payload["canonical_query_en"] = payload["raw_query"]
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        composition_step = next(s for s in ctx.evidence_plan if s["tool_name"] == "get_balancing_composition")
        assert composition_step["params"]["start_date"] == "2023-12-01"
        assert composition_step["params"]["end_date"] == "2024-01-31"
        assert "currency" not in composition_step["params"]

    def test_all_steps_marked_unsatisfied(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {"name": "get_prices", "score": 0.9, "reason": "prices"},
                {"name": "get_balancing_composition", "score": 0.8, "reason": "drivers"},
            ],
            needs_driver=True,
        )
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        for step in ctx.evidence_plan:
            assert step["satisfied"] is False

    def test_mismatched_tool_does_not_corrupt_plan_state(self):
        """Regression: Stage 0.5 matching a different tool than plan[0] must not
        mark the wrong step as satisfied."""
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {"name": "get_prices", "score": 0.9, "reason": "prices"},
                {"name": "get_balancing_composition", "score": 0.7, "reason": "shares"},
                {"name": "get_tariffs", "score": 0.6, "reason": "tariffs"},
            ],
            needs_driver=True,
        )
        payload["raw_query"] = "Why did balancing electricity price change?"
        payload["canonical_query_en"] = payload["raw_query"]
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 3
        assert ctx.evidence_plan[0]["tool_name"] == "get_prices"

        # Simulate Stage 0.5 matching get_generation_mix (not in the plan)
        # The plan should NOT have any step marked satisfied
        mismatched_step = next(
            (s for s in ctx.evidence_plan if s["tool_name"] == "get_generation_mix" and not s.get("satisfied")),
            None,
        )
        assert mismatched_step is None  # no plan step for get_generation_mix

        # Simulate Stage 0.5 matching get_balancing_composition (in plan but not primary)
        matched_step = next(
            (s for s in ctx.evidence_plan if s["tool_name"] == "get_balancing_composition" and not s.get("satisfied")),
            None,
        )
        assert matched_step is not None
        assert matched_step["role"] == "composition_context"


# ---------------------------------------------------------------------------
# Stage 0.8: execute_remaining_evidence
# ---------------------------------------------------------------------------

def _mock_execute_tool(results_by_name):
    """Return a mock that maps tool name to (df, cols, rows)."""
    import agent.evidence_planner as ep
    original = ep.execute_tool

    def _mock(invocation):
        name = invocation.name
        if name in results_by_name:
            return results_by_name[name]
        raise RuntimeError(f"Unexpected tool call: {name}")

    return _mock


class TestExecuteRemainingEvidence:
    def test_executes_unsatisfied_steps(self, monkeypatch):
        comp_df = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "share_import": [0.3, 0.25],
            "share_thermal_ppa": [0.4, 0.45],
        })
        comp_result = (comp_df, list(comp_df.columns), [tuple(r) for r in comp_df.itertuples(index=False)])

        monkeypatch.setattr(
            "agent.evidence_planner.execute_tool",
            _mock_execute_tool({"get_balancing_composition": comp_result}),
        )

        ctx = QueryContext(query="test")
        ctx.tool_name = "get_prices"
        ctx.df = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "p_bal_gel": [50.0, 55.0],
        })
        ctx.cols = list(ctx.df.columns)
        ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False)]
        ctx.evidence_plan = [
            {"role": "primary_data", "tool_name": "get_prices", "params": {}, "satisfied": True},
            {"role": "composition_context", "tool_name": "get_balancing_composition", "params": {}, "satisfied": False},
        ]

        ctx = execute_remaining_evidence(ctx)

        assert ctx.evidence_plan_complete is True
        assert "composition_context" in ctx.evidence_collected
        # Merged DataFrame should have share columns
        assert "share_import" in ctx.cols

    def test_failed_step_logged_and_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "agent.evidence_planner.execute_tool",
            lambda inv: (_ for _ in ()).throw(RuntimeError("connection refused")),
        )

        ctx = QueryContext(query="test")
        ctx.tool_name = "get_prices"
        ctx.df = pd.DataFrame({"date": ["2023-01-01"], "p_bal_gel": [50.0]})
        ctx.cols = list(ctx.df.columns)
        ctx.rows = [(("2023-01-01", 50.0))]
        ctx.evidence_plan = [
            {"role": "primary_data", "tool_name": "get_prices", "params": {}, "satisfied": True},
            {"role": "composition_context", "tool_name": "get_balancing_composition", "params": {}, "satisfied": False},
        ]

        ctx = execute_remaining_evidence(ctx)

        assert ctx.evidence_plan_complete is False
        assert "composition_context" not in ctx.evidence_collected
        assert "error" in ctx.evidence_plan[1]

    def test_cap_at_three_steps(self, monkeypatch):
        call_count = {"n": 0}

        def counting_mock(inv):
            call_count["n"] += 1
            df = pd.DataFrame({"date": ["2023-01-01"], "val": [1.0]})
            return df, list(df.columns), [tuple(r) for r in df.itertuples(index=False)]

        monkeypatch.setattr("agent.evidence_planner.execute_tool", counting_mock)

        ctx = QueryContext(query="test")
        ctx.tool_name = "get_prices"
        ctx.df = pd.DataFrame({"date": ["2023-01-01"], "p_bal_gel": [50.0]})
        ctx.cols = list(ctx.df.columns)
        ctx.rows = []
        # 4 unsatisfied steps
        ctx.evidence_plan = [
            {"role": "primary_data", "tool_name": "get_prices", "params": {}, "satisfied": True},
            {"role": "r1", "tool_name": "get_balancing_composition", "params": {}, "satisfied": False},
            {"role": "r2", "tool_name": "get_tariffs", "params": {}, "satisfied": False},
            {"role": "r3", "tool_name": "get_generation_mix", "params": {}, "satisfied": False},
            {"role": "r4", "tool_name": "get_prices", "params": {}, "satisfied": False},
        ]

        ctx = execute_remaining_evidence(ctx)

        assert call_count["n"] == 3  # capped at 3


class TestMergeEvidenceIntoContext:
    def test_merge_composition_into_prices(self):
        ctx = QueryContext(query="test")
        ctx.tool_name = "get_prices"
        ctx.df = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "p_bal_gel": [50.0, 55.0],
        })
        ctx.evidence_plan = [
            {"role": "primary_data", "tool_name": "get_prices", "params": {}, "satisfied": True},
        ]
        ctx.evidence_collected = {
            "composition_context": {
                "tool": "get_balancing_composition",
                "df": pd.DataFrame({
                    "date": ["2023-01-01", "2023-02-01"],
                    "share_import": [0.3, 0.25],
                }),
                "cols": ["date", "share_import"],
                "rows": [],
            },
        }

        ctx = merge_evidence_into_context(ctx)

        assert "share_import" in ctx.cols
        assert "p_bal_gel" in ctx.cols
        assert len(ctx.rows) == 2

    def test_empty_df_skips_merge(self):
        ctx = QueryContext(query="test")
        ctx.df = pd.DataFrame()
        ctx.evidence_collected = {"some_role": {"tool": "x", "df": pd.DataFrame({"a": [1]})}}

        ctx = merge_evidence_into_context(ctx)

        assert ctx.df.empty

    def test_primary_promotion_updates_tool_params_for_post_plan_enrichment(self):
        ctx = QueryContext(query="test")
        ctx.used_tool = False
        ctx.tool_name = "get_balancing_composition"
        ctx.tool_params = {"entities": ["import"]}
        ctx.df = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "share_import": [0.3, 0.25],
        })
        ctx.evidence_plan = [
            {
                "role": "primary_data",
                "tool_name": "get_prices",
                "params": {
                    "metric": "balancing",
                    "currency": "usd",
                    "start_date": "2023-01-01",
                    "end_date": "2023-02-01",
                },
                "satisfied": True,
            },
        ]
        primary_df = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "p_bal_usd": [18.0, 19.0],
        })
        ctx.evidence_collected = {
            "primary_data": {
                "tool": "get_prices",
                "params": {
                    "metric": "balancing",
                    "currency": "usd",
                    "start_date": "2023-01-01",
                    "end_date": "2023-02-01",
                },
                "df": primary_df,
                "cols": list(primary_df.columns),
                "rows": [tuple(r) for r in primary_df.itertuples(index=False, name=None)],
            },
        }

        ctx = merge_evidence_into_context(ctx)

        assert ctx.used_tool is True
        assert ctx.tool_name == "get_prices"
        assert ctx.tool_params["metric"] == "balancing"
        assert ctx.tool_params["currency"] == "usd"
        assert "p_bal_usd" in ctx.cols


# ---------------------------------------------------------------------------
# Plan-driven helpers
# ---------------------------------------------------------------------------

class TestPlanDrivenHelpers:
    def test_next_unsatisfied_step_returns_first(self):
        plan = [
            {"role": "primary_data", "tool_name": "get_prices", "satisfied": True},
            {"role": "composition_context", "tool_name": "get_balancing_composition", "satisfied": False},
            {"role": "tariff_context", "tool_name": "get_tariffs", "satisfied": False},
        ]
        step = next_unsatisfied_step(plan)
        assert step is not None
        assert step["role"] == "composition_context"

    def test_next_unsatisfied_step_all_satisfied(self):
        plan = [
            {"role": "primary_data", "tool_name": "get_prices", "satisfied": True},
        ]
        assert next_unsatisfied_step(plan) is None

    def test_next_unsatisfied_step_empty_plan(self):
        assert next_unsatisfied_step([]) is None

    def test_has_unsatisfied_steps_true(self):
        plan = [
            {"role": "primary_data", "tool_name": "get_prices", "satisfied": True},
            {"role": "composition_context", "tool_name": "get_balancing_composition", "satisfied": False},
        ]
        assert has_unsatisfied_steps(plan) is True

    def test_has_unsatisfied_steps_false(self):
        plan = [
            {"role": "primary_data", "tool_name": "get_prices", "satisfied": True},
        ]
        assert has_unsatisfied_steps(plan) is False

    def test_has_unsatisfied_steps_empty(self):
        assert has_unsatisfied_steps([]) is False

    def test_plan_driven_invocation_uses_first_step(self):
        """When evidence plan is non-empty, the first unsatisfied step
        should be used for invocation, not keyword routing."""
        plan = [
            {"role": "primary_data", "tool_name": "get_prices", "params": {"start_date": "2023-01-01"}, "satisfied": False},
            {"role": "composition_context", "tool_name": "get_balancing_composition", "params": {}, "satisfied": False},
        ]
        step = next_unsatisfied_step(plan)
        assert step["tool_name"] == "get_prices"
        assert step["params"]["start_date"] == "2023-01-01"

        # After marking first satisfied, next should be composition
        step["satisfied"] = True
        step2 = next_unsatisfied_step(plan)
        assert step2["tool_name"] == "get_balancing_composition"


# ---------------------------------------------------------------------------
# Join provenance
# ---------------------------------------------------------------------------

class TestJoinProvenance:
    def test_merge_records_join_provenance(self):
        ctx = QueryContext(query="test")
        ctx.tool_name = "get_prices"
        ctx.df = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "p_bal_gel": [50.0, 55.0],
        })
        ctx.evidence_plan = [
            {"role": "primary_data", "tool_name": "get_prices", "params": {}, "satisfied": True},
        ]
        ctx.evidence_collected = {
            "composition_context": {
                "tool": "get_balancing_composition",
                "df": pd.DataFrame({
                    "date": ["2023-01-01", "2023-02-01"],
                    "share_import": [0.3, 0.25],
                }),
                "cols": ["date", "share_import"],
                "rows": [],
            },
        }

        ctx = merge_evidence_into_context(ctx)

        assert len(ctx.join_provenance) == 1
        prov = ctx.join_provenance[0]
        assert prov["primary_tool"] == "get_prices"
        assert prov["secondary_tool"] == "get_balancing_composition"
        assert prov["role"] == "composition_context"
        assert prov["join_key"] == "date"
        assert "share_import" in prov["columns_added"]

    def test_no_provenance_when_no_evidence(self):
        ctx = QueryContext(query="test")
        ctx.df = pd.DataFrame({"date": ["2023-01-01"], "p_bal_gel": [50.0]})
        ctx.evidence_collected = {}

        ctx = merge_evidence_into_context(ctx)

        assert ctx.join_provenance == []


# ---------------------------------------------------------------------------
# Evidence record grounding tokens
# ---------------------------------------------------------------------------

class TestEvidenceRecordTokens:
    def test_derived_metric_values_become_grounding_tokens(self):
        ctx = QueryContext(query="test")
        ctx.analysis_evidence = [
            {
                "record_type": "derived",
                "derived_metric_name": "mom_absolute_change",
                "current_value": 55.0,
                "previous_value": 43.0,
                "absolute_change": 12.0,
                "percent_change": 27.907,
                "source_cells": [
                    {"column": "p_bal_gel", "period": "2024-01", "value": 55.0, "role": "current"},
                    {"column": "p_bal_gel", "period": "2023-12", "value": 43.0, "role": "previous"},
                ],
            },
        ]

        tokens: set = set()
        _add_evidence_record_tokens(tokens, ctx)

        # Computed values should be present
        assert "12" in tokens or "12.0" in tokens
        assert "55" in tokens or "55.0" in tokens
        assert "43" in tokens or "43.0" in tokens

    def test_scenario_values_become_grounding_tokens(self):
        ctx = QueryContext(query="test")
        ctx.analysis_evidence = [
            {
                "record_type": "scenario",
                "aggregate_result": 1500.0,
                "baseline_aggregate": 1000.0,
                "delta_aggregate": 500.0,
                "delta_percent": 50.0,
                "min_period_value": 100.0,
                "max_period_value": 200.0,
            },
        ]

        tokens: set = set()
        _add_evidence_record_tokens(tokens, ctx)

        assert "1500" in tokens or "1500.0" in tokens
        assert "500" in tokens or "500.0" in tokens

    def test_empty_evidence_produces_no_tokens(self):
        ctx = QueryContext(query="test")
        ctx.analysis_evidence = []

        tokens: set = set()
        _add_evidence_record_tokens(tokens, ctx)

        assert len(tokens) == 0

    def test_source_cells_min_max_become_grounding_tokens(self):
        """source_cells with min_value/max_value (trend, scenario) should
        be tokenized for grounding."""
        ctx = QueryContext(query="test")
        ctx.analysis_evidence = [
            {
                "record_type": "derived",
                "derived_metric_name": "trend_slope",
                "trend_slope": 5.0,
                "source_cells": [
                    {"column": "p_bal_gel", "role": "trend_series",
                     "row_count": 12, "min_value": 42.5, "max_value": 97.3},
                ],
            },
        ]
        tokens: set = set()
        _add_evidence_record_tokens(tokens, ctx)

        assert "42.5" in tokens or "42" in tokens
        assert "97.3" in tokens or "97" in tokens


# ---------------------------------------------------------------------------
# Error-skip behavior: failed steps should be skipped, not retried
# ---------------------------------------------------------------------------

class TestErrorSkipBehavior:
    def test_next_unsatisfied_step_skips_errored(self):
        plan = [
            {"role": "primary_data", "tool_name": "get_prices", "satisfied": False, "error": "stage_0_5:timeout"},
            {"role": "composition_context", "tool_name": "get_balancing_composition", "satisfied": False},
        ]
        step = next_unsatisfied_step(plan)
        assert step is not None
        assert step["tool_name"] == "get_balancing_composition"

    def test_next_unsatisfied_step_all_errored(self):
        plan = [
            {"role": "primary_data", "tool_name": "get_prices", "satisfied": False, "error": "failed"},
        ]
        assert next_unsatisfied_step(plan) is None

    def test_has_unsatisfied_steps_ignores_errored(self):
        plan = [
            {"role": "primary_data", "tool_name": "get_prices", "satisfied": True},
            {"role": "composition_context", "tool_name": "get_balancing_composition", "satisfied": False, "error": "timeout"},
        ]
        assert has_unsatisfied_steps(plan) is False

    def test_evidence_loop_skips_errored_steps(self, monkeypatch):
        """Stage 0.8 should not retry a step that already has an error from
        Stage 0.5 or 0.7."""
        calls = []

        def _mock(invocation):
            calls.append(invocation.name)
            df = pd.DataFrame({"date": ["2023-01-01"], "share_import": [0.3]})
            return df, list(df.columns), [tuple(r) for r in df.itertuples(index=False)]

        monkeypatch.setattr("agent.evidence_planner.execute_tool", _mock)

        ctx = QueryContext(query="test")
        ctx.tool_name = "get_prices"
        ctx.df = pd.DataFrame({"date": ["2023-01-01"], "p_bal_gel": [50.0]})
        ctx.cols = list(ctx.df.columns)
        ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False)]
        ctx.evidence_plan = [
            {"role": "primary_data", "tool_name": "get_prices", "params": {}, "satisfied": True},
            # This step failed in Stage 0.5 — should NOT be retried
            {"role": "composition_context", "tool_name": "get_balancing_composition", "params": {}, "satisfied": False, "error": "stage_0_5:connection refused"},
            # This step should still execute
            {"role": "tariff_context", "tool_name": "get_tariffs", "params": {}, "satisfied": False},
        ]

        ctx = execute_remaining_evidence(ctx)

        # Only get_tariffs should have been called (composition was skipped)
        assert calls == ["get_tariffs"]
        # Plan completeness: primary satisfied, composition errored (not retried), tariffs succeeded
        assert ctx.evidence_plan[1].get("error") == "stage_0_5:connection refused"
        assert ctx.evidence_plan[2].get("satisfied") is True


# ---------------------------------------------------------------------------
# Join provenance grounding tokens
# ---------------------------------------------------------------------------

class TestJoinProvenanceGrounding:
    def test_join_provenance_tokens_added(self):
        from agent.summarizer import _add_join_provenance_tokens, _tokenize_cell_value

        ctx = QueryContext(query="test")
        ctx.join_provenance = [
            {
                "primary_tool": "get_prices",
                "secondary_tool": "get_balancing_composition",
                "primary_rows": 24,
                "merged_rows": 24,
            },
        ]
        tokens: set = set()
        _add_join_provenance_tokens(tokens, ctx)

        assert "24" in tokens or "24.0" in tokens

    def test_empty_provenance_adds_nothing(self):
        from agent.summarizer import _add_join_provenance_tokens

        ctx = QueryContext(query="test")
        ctx.join_provenance = []
        tokens: set = set()
        _add_join_provenance_tokens(tokens, ctx)
        assert len(tokens) == 0
