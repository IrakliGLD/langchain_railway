"""Stage 0.4 / 0.8: Deterministic multi-evidence planning and execution.

Stage 0.4 expands a QuestionAnalysis into an ordered list of evidence steps.
Stage 0.8 executes any remaining unsatisfied steps after normal tool routing.

Gated by ``ENABLE_EVIDENCE_PLANNER``.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from agent.plan_validation import (
    SEVERITY_REJECT,
    SEVERITY_WARN,
    PlanValidationIssue,
    PlanValidationResult,
    _cross_check_visualization,
    validate_plan_against_answer_kind,
)
from agent.planner import resolve_tool_params
from agent.provenance import build_provenance_refs, sql_query_hash
from agent.tools import execute_tool
from agent.tools.tariff_tools import TARIFF_ENTITY_ALIASES
from agent.tools.types import ToolInvocation
from analysis.evidence_joins import join_evidence, join_evidence_with_provenance
from analysis.system_quantities import normalize_tool_dataframe
from contracts.question_analysis import (
    AnswerKind,
    EvidenceRole,
    QueryType,
    QuestionAnalysis,
    RenderStyle,
    ToolName,
)
from models import QueryContext
from utils.metrics import metrics

log = logging.getLogger("Enai")

# Score threshold for a secondary candidate tool to be included in the plan.
_SECONDARY_TOOL_MIN_SCORE = 0.50

# Answer shapes that inherently need multi-source or enumerated evidence.
# These block the single-tool fast path regardless of query_type, because the
# shape signal (COMPARISON etc.) is a stronger commitment than query_type.
_SHAPE_BLOCKS_FAST_PATH = frozenset({
    AnswerKind.COMPARISON,
    AnswerKind.FORECAST,
    AnswerKind.SCENARIO,
})

# "%" or spelled-out "percent"/"pct" (audit 2026-07-08 finding #6: the
# spelled-out form previously never matched). \b keeps "percentage" excluded.
# KA/RU threshold phrasings stay fixture-driven (§5.9), not guessed here.
_PCT = r"\s*(?:%|percent\b|pct\b)"
_SHARE_THRESHOLD_PATTERNS = (
    rf"(more than|above|over|exceed(?:ed|s|ing)?|greater than)\s+\d+(?:\.\d+)?{_PCT}",
    rf"(at least|not less than|minimum of)\s+\d+(?:\.\d+)?{_PCT}",
    rf"(less than|below|under|fewer than)\s+\d+(?:\.\d+)?{_PCT}",
    rf"(at most|no more than|maximum of)\s+\d+(?:\.\d+)?{_PCT}",
    rf"\d+(?:\.\d+)?{_PCT}\s+or\s+more",
    rf"\d+(?:\.\d+)?{_PCT}\s+or\s+less",
)
_SHARE_MONTH_LIST_HINTS = (
    "months where",
    "which months",
    "those months",
    "during these months",
    "for those months",
)

_GENERATION_CORRELATION_TARGETS = {
    "demand",
    "consumption",
    "generation",
    "local_generation",
    "import_dependency",
    "import dependence",
    "energy_security",
    "energy security",
    "self-sufficiency",
    "total_demand",
    "total_domestic_generation",
    "import_dependency_ratio",
}


def _is_threshold_share_query(raw_query: str, primary_tool: str) -> bool:
    if primary_tool != ToolName.GET_BALANCING_COMPOSITION.value:
        return False
    query_lower = str(raw_query or "").lower()
    if not any(token in query_lower for token in ("share", "composition", "contribute", "contribution")):
        return False
    return any(re.search(pattern, query_lower) for pattern in _SHARE_THRESHOLD_PATTERNS)


def _share_query_requests_price_context(raw_query: str, primary_tool: str) -> bool:
    if not _is_threshold_share_query(raw_query, primary_tool):
        return False
    query_lower = str(raw_query or "").lower()
    asks_price = any(token in query_lower for token in ("price", "gel", "usd"))
    return asks_price


def _correlation_needs_generation_mix(
    qa: QuestionAnalysis,
    raw_query: str,
    primary_tool: str,
) -> bool:
    """Return True when a correlation/explanation needs system quantity context."""
    if primary_tool not in {ToolName.GET_PRICES.value, ToolName.GET_GENERATION_MIX.value}:
        return False

    query_lower = str(raw_query or "").lower()
    if any(
        token in query_lower
        for token in (
            "demand",
            "consumption",
            "generation mix",
            "generation",
            "import dependency",
            "import dependence",
            "energy security",
            "self-sufficiency",
        )
    ):
        return True

    for metric in qa.analysis_requirements.derived_metrics or []:
        metric_name = getattr(metric.metric_name, "value", str(metric.metric_name or "")).strip()
        target_metric = str(getattr(metric, "target_metric", "") or "").strip().lower()
        if metric_name == "correlation_to_target" and target_metric in _GENERATION_CORRELATION_TARGETS:
            return True
    return False


def _repair_list_tariff_entities(
    qa: QuestionAnalysis,
    primary_tool: str,
    primary_params: Dict[str, Any],
) -> None:
    """Inject the entity alias from entity_scope when a LIST plan would otherwise run empty.

    ``get_tariffs`` falls back to ``DEFAULT_TARIFF_ENTITY_ALIASES`` (a narrow
    12-plant subset) when called with no entities.  For a LIST question such
    as "which plants are regulated?" that default silently omits three regulated
    plants.  If the analyzer emitted an ``entity_scope`` that matches a known
    alias (e.g. ``regulated_plants``), promote it into ``params.entities`` so
    the tool returns the complete enumeration.
    """
    if qa.answer_kind != AnswerKind.LIST:
        return
    if primary_tool != ToolName.GET_TARIFFS.value:
        return
    existing = primary_params.get("entities") or []
    if existing:
        return
    scope = (qa.entity_scope or "").strip().lower()
    if not scope:
        return
    if scope not in TARIFF_ENTITY_ALIASES:
        return
    primary_params["entities"] = [scope]
    log.info(
        "LIST plan repair: injected entities=[%s] for get_tariffs (entity_scope alias).",
        scope,
    )


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
    # P4.2 (finding M3): validate the expanded plan against the answer
    # contract and publish the typed result on the context. Deterministic
    # repairs (e.g. _repair_list_tariff_entities) run during expansion, so
    # this single validation pass always sees the repaired plan. Enforcement
    # of reject-severity issues happens in the pipeline stage gated by
    # ENAI_PLAN_VALIDATION_MODE; this function never mutates the plan.
    validation = validate_plan_against_answer_kind(steps, qa, ctx.query)
    ctx.plan_validation = validation
    for issue in validation.issues:
        metrics.log_plan_validation(issue.rule, issue.severity)
    return ctx


# Compatibility for callers that used the former private validator location.
_validate_plan_against_answer_kind = validate_plan_against_answer_kind


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
    threshold_share_with_prices = _share_query_requests_price_context(raw_query, top_name)

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

    # Repair LIST-shape plans whose primary params lack the full entity set.
    # Without this, "which plants are regulated?" would run get_tariffs with
    # DEFAULT_TARIFF_ENTITY_ALIASES (12 plants) and silently omit three.
    _repair_list_tariff_entities(qa, top_name, primary_params)

    primary_step: Dict[str, Any] = {
        "role": EvidenceRole.PRIMARY_DATA.value,
        "tool_name": top_name,
        "params": primary_params,
        "satisfied": False,
        "source": "planner",
    }

    # --- Single-tool fast path ---
    # Shape signals (answer_kind COMPARISON/FORECAST/SCENARIO) inherently need
    # richer evidence than a single primary tool, so they force the multi-step
    # expansion even when query_type and driver/correlation flags say otherwise.
    if (
        not explicit_multi
        and not needs_driver
        and not needs_correlation
        and query_type != QueryType.COMPARISON
        and not threshold_share_with_prices
        and qa.answer_kind not in _SHAPE_BLOCKS_FAST_PATH
    ):
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

    # Validation runs in build_evidence_plan (P4.2) so single-step fast-path
    # plans are covered too and the typed result lands on the context.
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
    needs_generation_mix = _correlation_needs_generation_mix(qa, raw_query, steps[0]["tool_name"])

    for role in explicit_roles:
        if role == EvidenceRole.PRIMARY_DATA.value:
            continue
        tool_name = role_to_tool.get(role)
        if (
            role == EvidenceRole.CORRELATION_DRIVER.value
            and needs_generation_mix
            and steps[0]["tool_name"] == ToolName.GET_PRICES.value
        ):
            tool_name = ToolName.GET_GENERATION_MIX.value
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
    is_balancing_prices_primary = (
        primary_tool == ToolName.GET_PRICES.value
        and str(primary_params.get("metric") or "").strip().lower() == "balancing"
    )
    threshold_share_with_prices = _share_query_requests_price_context(raw_query, primary_tool)
    needs_generation_mix = _correlation_needs_generation_mix(qa, raw_query, primary_tool)
    # Forecast narratives on balancing price must cite composition shifts and
    # policy events (see ``knowledge/balancing_price.md`` "For Forecasting"
    # and ``skills/answer-composer/references/forecast-caveats.md``).  A
    # primary-only plan leaves the summarizer with nothing but the CAGR
    # number — the same defect surfaced by the ``Plan validation`` warning
    # in production traces for forecast queries.
    forecast_narrative_needs_composition = (
        is_balancing_prices_primary
        and qa.answer_kind == AnswerKind.FORECAST
        and qa.render_style == RenderStyle.NARRATIVE
    )

    # get_prices + driver analysis → add composition
    # Also: balancing-price forecast narratives unconditionally need
    # composition context, even when ``needs_driver`` is False.
    if primary_tool == ToolName.GET_PRICES.value and (
        needs_driver or forecast_narrative_needs_composition
    ):
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

    # Balancing price driver/comparison analysis also needs tariff context so
    # answers can cite regulated source-price layers, not just composition.
    if (
        is_balancing_prices_primary
        and (needs_driver or needs_correlation or query_type == QueryType.COMPARISON)
        and not needs_generation_mix
        and ToolName.GET_TARIFFS.value not in added_tools
    ):
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

    # get_prices + correlation context → add composition + tariffs
    if primary_tool == ToolName.GET_PRICES.value and needs_correlation:
        secondary_tool = (
            ToolName.GET_GENERATION_MIX.value
            if needs_generation_mix
            else ToolName.GET_BALANCING_COMPOSITION.value
        )
        secondary_role = (
            EvidenceRole.CORRELATION_DRIVER.value
            if needs_generation_mix
            else EvidenceRole.COMPOSITION_CONTEXT.value
        )
        if secondary_tool not in added_tools:
            params = _resolve_secondary_params(
                qa, secondary_tool, raw_query, candidates, primary_params,
            )
            if params is not None:
                steps.append({
                    "role": secondary_role,
                    "tool_name": secondary_tool,
                    "params": params,
                    "satisfied": False,
                    "source": "planner_rule",
                })
                added_tools.add(secondary_tool)
        if (
            not is_balancing_prices_primary
            and ToolName.GET_TARIFFS.value not in added_tools
            and not needs_generation_mix
        ):
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
    if primary_tool == ToolName.GET_BALANCING_COMPOSITION.value and (
        needs_driver or needs_correlation or threshold_share_with_prices
    ):
        if threshold_share_with_prices:
            # Ontology-migration shadow (§5 slice 1): measure whether the
            # analyzer already emits get_prices for threshold-share queries.
            # The planner rule stays authoritative until sustained agreement
            # clears the cutover bar; only then is this rule deletable and
            # the contract becomes the sole owner of the evidence ontology.
            _analyzer_emitted = any(
                t.name == ToolName.GET_PRICES.value
                for t in (qa.tooling.candidate_tools or [])[1:]
            )
            metrics.log_evidence_rule_agreement(
                "threshold_share_price_context", agree=_analyzer_emitted,
            )
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

    # Inherit the primary time window so cross-tool evidence lines up on the same periods.
    if "start_date" not in params and "start_date" in primary_params:
        params["start_date"] = primary_params["start_date"]
    if "end_date" not in params and "end_date" in primary_params:
        params["end_date"] = primary_params["end_date"]
    if (
        tool_name in {ToolName.GET_PRICES.value, ToolName.GET_TARIFFS.value}
        and primary_params.get("currency")
        and not params.get("currency")
    ):
        params["currency"] = primary_params["currency"]
    if (
        tool_name in {ToolName.GET_PRICES.value, ToolName.GET_GENERATION_MIX.value}
        and primary_params.get("granularity") == "yearly"
    ):
        params["granularity"] = "yearly"

    # Balancing-price enrichment defaults to grouped tariff buckets used by the analyzer templates.
    if (
        tool_name == ToolName.GET_TARIFFS.value
        and primary_params.get("metric") == "balancing"
    ):
        if primary_params.get("currency"):
            params["currency"] = primary_params["currency"]
        if not params.get("entities"):
            params["entities"] = [
                "regulated_hpp",
                "regulated_new_tpp",
                "regulated_old_tpp",
            ]

    return params


def next_unsatisfied_step(plan: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the first unsatisfied step that hasn't already failed, or None."""
    for step in plan:
        if not step.get("satisfied") and not step.get("error"):
            return step
    return None


def has_unsatisfied_steps(plan: List[Dict[str, Any]]) -> bool:
    """Return True when the evidence plan has actionable work remaining.

    Steps that already failed (have an ``error`` key) are not actionable.
    """
    return any(not s.get("satisfied") and not s.get("error") for s in plan)


# ---------------------------------------------------------------------------
# Stage 0.8: Execute remaining evidence steps
# ---------------------------------------------------------------------------

_EVIDENCE_LOOP_MAX_STEPS = 3
EVIDENCE_LOOP_BUDGET_SECONDS = float(os.getenv("EVIDENCE_LOOP_BUDGET_SECONDS", "30"))


def execute_remaining_evidence(ctx: QueryContext, timeout_seconds: Optional[float] = None) -> QueryContext:
    """Stage 0.8 entry: execute unsatisfied evidence plan steps after tool routing.

    Thin delegate to ``pipeline._run_secondary_evidence_loop``. The body
    moved to ``pipeline.py`` in Phase F.5.1.a to remove the circular-import
    workaround that previously forced a lazy import. The delegate stays
    here so that tests that monkey-patch ``agent.evidence_planner.execute_tool``
    continue to intercept the tool calls — this function passes its local
    ``execute_tool`` binding down to the moved loop body via ``executor=``.

    External callers (only ``pipeline.process_query`` today, via this name)
    keep their import path; internal logic lives next to the helper it
    uses.
    """
    # Lazy import here to avoid module-load-time circular import; safe at
    # call time because pipeline.py has finished initialising by then.
    from agent.pipeline import _run_secondary_evidence_loop
    return _run_secondary_evidence_loop(
        ctx,
        executor=execute_tool,
        timeout_seconds=timeout_seconds,
    )


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
            primary_params = dict(
                primary_evidence.get("params")
                or (primary_step.get("params") if primary_step else {})
                or {},
            )
            if ctx.tool_name != primary_tool or ctx.df.empty:
                log.info(
                    "Evidence merge: restoring primary %s from evidence_collected to ctx.df "
                    "(Stage 0.5 matched %s, current rows=%d)",
                    primary_tool, ctx.tool_name, len(ctx.df),
                )
                ctx.df = primary_df
                ctx.tool_name = primary_tool
            ctx.used_tool = True
            ctx.tool_params = primary_params
            if not ctx.tool_match_reason:
                ctx.tool_match_reason = "evidence_plan:primary_data"

    if ctx.df.empty:
        return ctx

    parent_refs = list(ctx.provenance_refs)
    if primary_evidence:
        parent_refs.extend(primary_evidence.get("provenance_refs", []))

    for role, evidence in ctx.evidence_collected.items():
        if role == EvidenceRole.PRIMARY_DATA.value:
            continue  # primary is already ctx.df (either original or promoted)
        secondary_df = evidence.get("df")
        if secondary_df is None or (isinstance(secondary_df, pd.DataFrame) and secondary_df.empty):
            continue
        secondary_tool = evidence.get("tool", "")
        secondary_refs = list(evidence.get("provenance_refs", []))
        parent_refs.extend(secondary_refs)

        ctx.df, provenance = join_evidence_with_provenance(
            ctx.df, secondary_df, primary_tool, secondary_tool,
        )
        if provenance is not None:
            provenance["role"] = role
            provenance["provenance_refs"] = list(dict.fromkeys(secondary_refs))
            ctx.join_provenance.append(provenance)

    # Refresh cols/rows after merge
    ctx.cols = list(ctx.df.columns)
    ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False, name=None)]
    merge_hash = sql_query_hash("|".join(sorted(set(parent_refs))) + "|evidence_join")
    ctx.provenance_refs = build_provenance_refs(
        ctx.cols,
        ctx.rows,
        source="derived_join",
        query_hash=merge_hash,
        parent_refs=parent_refs,
    )
    if ctx.evidence_frame is not None:
        ctx.evidence_frame.provenance_refs = list(ctx.provenance_refs)

    return ctx
