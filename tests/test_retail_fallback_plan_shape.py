"""The retail evidence fallback must emit a step the pipeline can execute.

2026-08-17 production, HTTP 500, error_class=KeyError, immediately after:

    Evidence plan: retail question with no analyzer tool candidates --
    planning get_end_user_prices

Every other step the planner builds carries ``tool_name`` / ``params`` /
``satisfied`` (evidence_planner.py:327-333), and ``_expand_evidence_steps``
reads ``steps[0]["tool_name"]`` directly. The retail fallback returned ``tool``
and ``params_hint`` instead, so the first consumer to touch it raised KeyError
and the request died with a 500.

Latent since the fallback was added on 2026-08-15: it only fires when a retail
question comes back with ``candidate_tools=[]``, which no earlier trace hit. The
first request in the same session had two candidate tools and answered fine; the
second had none and 500'd.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.evidence_planner import build_evidence_plan
from contracts.question_analysis import ToolName
from tests.test_end_user_scope_clarification import _ctx

# The 2026-08-17 question, canonicalised. EPS at 6-10 kV, retail versus wholesale.
QUERY = (
    "For EP Georgia Supply commercial customers connected at 6-10 kV, is the "
    "retail price cheaper than buying on the wholesale market?"
)


def _retail_ctx():
    """A retail question the analyzer nominated no tools for."""
    ctx = _ctx(
        QUERY,
        topics=("network_supply_tariffs", "market_structure", "exchange_transition"),
        query_type="ambiguous",
        preferred_path="knowledge",
    )
    ctx.question_analysis.entity_scope = (
        "EP Georgia Supply commercial customers connected at 6–10 kV"
    )
    assert not ctx.question_analysis.tooling.candidate_tools, "fixture must have no tools"
    return ctx


def test_the_retail_fallback_does_not_crash_the_request():
    """The 500. Any consumer of the plan raised KeyError on the first key it read."""
    ctx = build_evidence_plan(_retail_ctx())

    assert ctx.evidence_plan, "fallback planned nothing"
    step = ctx.evidence_plan[0]
    # The keys every consumer reads: pipeline.py:1559-1562, 1902-1905,
    # evidence_planner.py:387, 750.
    assert step["tool_name"] == ToolName.GET_END_USER_PRICES.value
    assert step["role"] == "primary_data"
    assert isinstance(step["params"], dict)
    assert step["satisfied"] is False


def test_the_fallback_step_carries_resolved_params_not_a_hint():
    """``params_hint: None`` gave the executor nothing to call the tool with.

    Resolving them here also means the fallback path gets the make-or-buy
    benchmark and the widened window, which is what the non-fallback request in
    the same session got and this one did not.
    """
    ctx = build_evidence_plan(_retail_ctx())
    params = ctx.evidence_plan[0]["params"]

    assert params.get("supplier") == "eps"
    assert params.get("include_wholesale_benchmark") is True, (
        "no benchmark means no comparison and no annual block"
    )


def test_a_non_retail_question_still_plans_nothing_here():
    """The fallback must not start planning retail tools for other questions."""
    ctx = _ctx(
        "What is the balancing price trend?",
        topics=("balancing_price",),
        query_type="ambiguous",
    )
    out = build_evidence_plan(ctx)

    tools = [s.get("tool_name") for s in (out.evidence_plan or [])]
    assert ToolName.GET_END_USER_PRICES.value not in tools
