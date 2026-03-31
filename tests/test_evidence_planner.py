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
from agent.evidence_planner import build_evidence_plan, execute_remaining_evidence, merge_evidence_into_context


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
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
        },
        "analysis_requirements": {
            "needs_driver_analysis": needs_driver,
            "needs_correlation_context": needs_correlation,
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

    def test_prices_with_driver_analysis_adds_composition(self):
        payload = _make_qa_payload(
            query_type="data_explanation",
            tools=[
                {"name": "get_prices", "score": 0.9, "reason": "price data"},
                {"name": "get_balancing_composition", "score": 0.7, "reason": "shares"},
            ],
            needs_driver=True,
        )
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 2
        assert ctx.evidence_plan[0]["tool_name"] == "get_prices"
        assert ctx.evidence_plan[0]["role"] == "primary_data"
        assert ctx.evidence_plan[1]["tool_name"] == "get_balancing_composition"
        assert ctx.evidence_plan[1]["role"] == "composition_context"

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
                {"name": "get_prices", "score": 0.85, "reason": "prices"},
                {"name": "get_tariffs", "score": 0.30, "reason": "tariffs"},
            ],
        )
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
            ],
            needs_driver=True,
        )
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 2
        secondary = ctx.evidence_plan[1]
        assert secondary["params"]["start_date"] == "2023-01-01"
        assert secondary["params"]["end_date"] == "2023-12-31"

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
            ],
            needs_driver=True,
        )
        ctx = _ctx_with_qa(payload)
        ctx = build_evidence_plan(ctx)

        assert len(ctx.evidence_plan) == 2
        assert ctx.evidence_plan[0]["tool_name"] == "get_prices"

        # Simulate Stage 0.5 matching get_tariffs (not in the plan)
        # The plan should NOT have any step marked satisfied
        mismatched_step = next(
            (s for s in ctx.evidence_plan if s["tool_name"] == "get_tariffs" and not s.get("satisfied")),
            None,
        )
        assert mismatched_step is None  # no plan step for get_tariffs

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
