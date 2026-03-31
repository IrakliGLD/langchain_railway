"""Stage 0.4 / 0.8: Deterministic multi-evidence planning and execution.

Stage 0.4 expands a QuestionAnalysis into an ordered list of evidence steps.
Stage 0.8 executes any remaining unsatisfied steps after normal tool routing.

Gated by ``ENABLE_EVIDENCE_PLANNER``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from contracts.question_analysis import (
    EvidenceRole,
    QuestionAnalysis,
    QueryType,
    ToolName,
)
from models import QueryContext
from agent.planner import resolve_tool_params
from agent.tools import execute_tool
from agent.tools.types import ToolInvocation
from analysis.evidence_joins import join_evidence

log = logging.getLogger("Enai")

# Score threshold for a secondary candidate tool to be included in the plan.
_SECONDARY_TOOL_MIN_SCORE = 0.50


# ---------------------------------------------------------------------------
# Stage 0.4: Build evidence plan
# ---------------------------------------------------------------------------

def build_evidence_plan(ctx: QueryContext) -> QueryContext:
    """Deterministic expansion of QuestionAnalysis into evidence steps.

    Reads ``ctx.question_analysis`` and produces ``ctx.evidence_plan`` -- a
    list of dicts, each describing one tool invocation needed to satisfy the
    query.  The plan is *ordered*: the first step is the primary dataset,
    subsequent steps are supporting context.

    When the analyzer is absent or not authoritative, the plan is empty and
    the pipeline falls through to existing routing.
    """
    qa = ctx.question_analysis
    if qa is None or ctx.question_analysis_source != "llm_active":
        ctx.evidence_plan = []
        ctx.evidence_plan_source = ""
        return ctx

    steps = _expand_evidence_steps(qa, ctx.query)

    ctx.evidence_plan = steps
    ctx.evidence_plan_source = "deterministic" if steps else ""
    return ctx


def _expand_evidence_steps(
    qa: QuestionAnalysis,
    raw_query: str,
) -> List[Dict[str, Any]]:
    """Apply deterministic expansion rules to produce evidence steps."""

    candidates = qa.tooling.candidate_tools
    if not candidates:
        return []

    top = candidates[0]
    top_name = top.name.value
    top_hint = top.params_hint

    ar = qa.analysis_requirements
    needs_driver = ar.needs_driver_analysis
    needs_correlation = ar.needs_correlation_context
    query_type = qa.classification.query_type

    # Explicit multi-tool signal from analyzer (Phase 1.5 prompt)
    explicit_multi = qa.routing.needs_multi_tool
    explicit_roles = [r.value for r in qa.routing.evidence_roles]

    # --- Resolve params for the primary tool ---
    try:
        primary_params = resolve_tool_params(qa, top_name, raw_query, hint=top_hint)
    except ValueError:
        primary_params = None
    if primary_params is None:
        return []

    primary_step: Dict[str, Any] = {
        "role": EvidenceRole.PRIMARY_DATA.value,
        "tool_name": top_name,
        "params": primary_params,
        "satisfied": False,
        "source": "planner",
    }

    # --- Single-tool fast path ---
    if not explicit_multi and not needs_driver and not needs_correlation and query_type != QueryType.COMPARISON:
        return [primary_step]

    # --- Multi-evidence expansion ---
    steps: List[Dict[str, Any]] = [primary_step]
    added_tools: set = {top_name}

    # Strategy 1: Explicit evidence_roles from analyzer
    if explicit_multi and explicit_roles:
        _add_steps_from_roles(
            steps, added_tools, explicit_roles, qa, raw_query, candidates, primary_params,
        )

    # Strategy 2: Deterministic rules based on primary tool + analysis requirements
    if not explicit_multi or len(steps) < 2:
        _add_steps_from_rules(
            steps, added_tools, top_name, needs_driver, needs_correlation,
            query_type, qa, raw_query, candidates, primary_params,
        )

    # Strategy 3: Comparison queries with multiple strong candidates
    if query_type == QueryType.COMPARISON and len(steps) < 2:
        _add_steps_from_comparison(
            steps, added_tools, qa, raw_query, candidates, primary_params,
        )

    return steps


def _add_steps_from_roles(
    steps: List[Dict[str, Any]],
    added_tools: set,
    explicit_roles: List[str],
    qa: QuestionAnalysis,
    raw_query: str,
    candidates: list,
    primary_params: dict,
) -> None:
    """Add steps for explicit evidence_roles emitted by the analyzer."""
    role_to_tool = _role_to_default_tool(steps[0]["tool_name"])

    for role in explicit_roles:
        if role == EvidenceRole.PRIMARY_DATA.value:
            continue
        tool_name = role_to_tool.get(role)
        if not tool_name or tool_name in added_tools:
            continue
        params = _resolve_secondary_params(qa, tool_name, raw_query, candidates, primary_params)
        if params is not None:
            steps.append({
                "role": role,
                "tool_name": tool_name,
                "params": params,
                "satisfied": False,
                "source": "planner_role",
            })
            added_tools.add(tool_name)


def _add_steps_from_rules(
    steps: List[Dict[str, Any]],
    added_tools: set,
    primary_tool: str,
    needs_driver: bool,
    needs_correlation: bool,
    query_type: QueryType,
    qa: QuestionAnalysis,
    raw_query: str,
    candidates: list,
    primary_params: dict,
) -> None:
    """Add steps based on deterministic rules: primary tool + analysis flags."""

    # get_prices + driver analysis → add composition
    if primary_tool == ToolName.GET_PRICES.value and needs_driver:
        if ToolName.GET_BALANCING_COMPOSITION.value not in added_tools:
            params = _resolve_secondary_params(
                qa, ToolName.GET_BALANCING_COMPOSITION.value, raw_query, candidates, primary_params,
            )
            if params is not None:
                steps.append({
                    "role": EvidenceRole.COMPOSITION_CONTEXT.value,
                    "tool_name": ToolName.GET_BALANCING_COMPOSITION.value,
                    "params": params,
                    "satisfied": False,
                    "source": "planner_rule",
                })
                added_tools.add(ToolName.GET_BALANCING_COMPOSITION.value)

    # get_prices + correlation context → add composition + tariffs
    if primary_tool == ToolName.GET_PRICES.value and needs_correlation:
        if ToolName.GET_BALANCING_COMPOSITION.value not in added_tools:
            params = _resolve_secondary_params(
                qa, ToolName.GET_BALANCING_COMPOSITION.value, raw_query, candidates, primary_params,
            )
            if params is not None:
                steps.append({
                    "role": EvidenceRole.COMPOSITION_CONTEXT.value,
                    "tool_name": ToolName.GET_BALANCING_COMPOSITION.value,
                    "params": params,
                    "satisfied": False,
                    "source": "planner_rule",
                })
                added_tools.add(ToolName.GET_BALANCING_COMPOSITION.value)
        if ToolName.GET_TARIFFS.value not in added_tools:
            params = _resolve_secondary_params(
                qa, ToolName.GET_TARIFFS.value, raw_query, candidates, primary_params,
            )
            if params is not None:
                steps.append({
                    "role": EvidenceRole.TARIFF_CONTEXT.value,
                    "tool_name": ToolName.GET_TARIFFS.value,
                    "params": params,
                    "satisfied": False,
                    "source": "planner_rule",
                })
                added_tools.add(ToolName.GET_TARIFFS.value)

    # get_balancing_composition + driver/correlation → add prices
    if primary_tool == ToolName.GET_BALANCING_COMPOSITION.value and (needs_driver or needs_correlation):
        if ToolName.GET_PRICES.value not in added_tools:
            params = _resolve_secondary_params(
                qa, ToolName.GET_PRICES.value, raw_query, candidates, primary_params,
            )
            if params is not None:
                steps.append({
                    "role": EvidenceRole.CORRELATION_DRIVER.value,
                    "tool_name": ToolName.GET_PRICES.value,
                    "params": params,
                    "satisfied": False,
                    "source": "planner_rule",
                })
                added_tools.add(ToolName.GET_PRICES.value)

    # get_generation_mix + driver → add prices
    if primary_tool == ToolName.GET_GENERATION_MIX.value and (needs_driver or needs_correlation):
        if ToolName.GET_PRICES.value not in added_tools:
            params = _resolve_secondary_params(
                qa, ToolName.GET_PRICES.value, raw_query, candidates, primary_params,
            )
            if params is not None:
                steps.append({
                    "role": EvidenceRole.CORRELATION_DRIVER.value,
                    "tool_name": ToolName.GET_PRICES.value,
                    "params": params,
                    "satisfied": False,
                    "source": "planner_rule",
                })
                added_tools.add(ToolName.GET_PRICES.value)


def _add_steps_from_comparison(
    steps: List[Dict[str, Any]],
    added_tools: set,
    qa: QuestionAnalysis,
    raw_query: str,
    candidates: list,
    primary_params: dict,
) -> None:
    """For comparison queries, include strong secondary candidate tools."""
    for candidate in candidates[1:]:
        if candidate.score < _SECONDARY_TOOL_MIN_SCORE:
            break
        cand_name = candidate.name.value
        if cand_name in added_tools:
            continue
        params = _resolve_secondary_params(
            qa, cand_name, raw_query, candidates, primary_params,
        )
        if params is not None:
            steps.append({
                "role": EvidenceRole.CORRELATION_DRIVER.value,
                "tool_name": cand_name,
                "params": params,
                "satisfied": False,
                "source": "planner_comparison",
            })
            added_tools.add(cand_name)
            break  # at most one secondary for comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _role_to_default_tool(primary_tool: str) -> Dict[str, str]:
    """Map evidence roles to the most likely supporting tool given a primary."""
    base = {
        EvidenceRole.COMPOSITION_CONTEXT.value: ToolName.GET_BALANCING_COMPOSITION.value,
        EvidenceRole.TARIFF_CONTEXT.value: ToolName.GET_TARIFFS.value,
        EvidenceRole.CORRELATION_DRIVER.value: ToolName.GET_PRICES.value,
    }
    # If the primary tool is prices, the correlation driver should be composition
    if primary_tool == ToolName.GET_PRICES.value:
        base[EvidenceRole.CORRELATION_DRIVER.value] = ToolName.GET_BALANCING_COMPOSITION.value
    return base


def _resolve_secondary_params(
    qa: QuestionAnalysis,
    tool_name: str,
    raw_query: str,
    candidates: list,
    primary_params: dict,
) -> Optional[dict]:
    """Resolve params for a secondary tool.

    Tries the candidate's own hint first.  Falls back to inheriting date
    range from the primary tool so the datasets are temporally aligned.
    """
    # Find the candidate's hint if present
    hint = None
    for c in candidates:
        if c.name.value == tool_name:
            hint = c.params_hint
            break

    try:
        params = resolve_tool_params(qa, tool_name, raw_query, hint=hint)
    except ValueError:
        params = None

    if params is None:
        return None

    # Inherit date range from primary if secondary didn't resolve its own
    if "start_date" not in params and "start_date" in primary_params:
        params["start_date"] = primary_params["start_date"]
    if "end_date" not in params and "end_date" in primary_params:
        params["end_date"] = primary_params["end_date"]

    return params


# ---------------------------------------------------------------------------
# Stage 0.8: Execute remaining evidence steps
# ---------------------------------------------------------------------------

_EVIDENCE_LOOP_MAX_STEPS = 3


def execute_remaining_evidence(ctx: QueryContext) -> QueryContext:
    """Execute unsatisfied evidence plan steps after tool routing.

    Iterates remaining plan steps, executes each tool, stores results in
    ``ctx.evidence_collected``, then merges secondary datasets into
    ``ctx.df`` via date-aligned joins.

    Max iterations: ``min(len(remaining), 3)``.
    Failed steps are logged and skipped, not retried.
    """
    remaining = [s for s in ctx.evidence_plan if not s.get("satisfied")]
    cap = min(len(remaining), _EVIDENCE_LOOP_MAX_STEPS)

    for step in remaining[:cap]:
        invocation = ToolInvocation(
            name=step["tool_name"],
            params=step["params"],
            confidence=0.85,
            reason=f"evidence_plan:{step['role']}",
        )
        try:
            df, cols, rows = execute_tool(invocation)
            ctx.evidence_collected[step["role"]] = {
                "tool": invocation.name,
                "df": df,
                "cols": list(cols),
                "rows": [tuple(r) for r in rows],
            }
            step["satisfied"] = True
            log.info(
                "Evidence loop: fetched %s via %s (%d rows)",
                step["role"], invocation.name, len(rows),
            )
        except Exception as exc:
            step["error"] = str(exc)
            log.warning(
                "Evidence loop: step failed. role=%s tool=%s err=%s",
                step["role"], step["tool_name"], exc,
            )

    # Merge secondary evidence into the primary DataFrame
    ctx = merge_evidence_into_context(ctx)
    ctx.evidence_plan_complete = all(s.get("satisfied") for s in ctx.evidence_plan)
    return ctx


def merge_evidence_into_context(ctx: QueryContext) -> QueryContext:
    """Merge secondary evidence datasets into ``ctx.df``.

    The primary dataset remains as the base of ``ctx.df``.  Each secondary
    dataset is joined via ``join_evidence`` using the known patterns.
    """
    if not ctx.evidence_collected:
        return ctx

    # Identify the primary tool from the plan or from ctx.tool_name
    primary_tool = ctx.tool_name or ""
    if ctx.evidence_plan:
        primary_step = next(
            (s for s in ctx.evidence_plan if s["role"] == EvidenceRole.PRIMARY_DATA.value),
            None,
        )
        if primary_step:
            primary_tool = primary_step["tool_name"]

    # If the evidence loop fetched the primary dataset (e.g. Stage 0.5
    # matched a different tool), promote it to ctx.df so the merge base
    # is correct.
    primary_evidence = ctx.evidence_collected.get(EvidenceRole.PRIMARY_DATA.value)
    if primary_evidence and primary_evidence.get("tool") == primary_tool:
        primary_df = primary_evidence.get("df")
        if primary_df is not None and not primary_df.empty:
            if ctx.tool_name != primary_tool:
                log.info(
                    "Evidence merge: promoting %s from evidence_collected to ctx.df "
                    "(Stage 0.5 had matched %s)",
                    primary_tool, ctx.tool_name,
                )
                ctx.df = primary_df
                ctx.tool_name = primary_tool

    if ctx.df.empty:
        return ctx

    for role, evidence in ctx.evidence_collected.items():
        if role == EvidenceRole.PRIMARY_DATA.value:
            continue  # primary is already ctx.df (either original or promoted)
        secondary_df = evidence.get("df")
        if secondary_df is None or (isinstance(secondary_df, pd.DataFrame) and secondary_df.empty):
            continue
        secondary_tool = evidence.get("tool", "")

        ctx.df = join_evidence(ctx.df, secondary_df, primary_tool, secondary_tool)

    # Refresh cols/rows after merge
    ctx.cols = list(ctx.df.columns)
    ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False, name=None)]

    return ctx
