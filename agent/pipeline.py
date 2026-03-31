"""
Pipeline Orchestrator

Coordinates request handling with optional fast tool path:
  0. planner.prepare_context            -> cheap language/mode/conceptual detection
  1. router.match_tool (optional)       -> pre-LLM deterministic routing
  2. orchestrator.run_agent_loop        -> bounded typed-tool loop (optional)
  3. planner.generate_plan              -> legacy LLM plan path (fallback only)
  4. sql_executor.validate_and_execute  -> SQL validation/execution fallback
  5. analyzer.enrich
  6. summarizer.summarize_data
  7. chart_pipeline.build_chart
"""
import json
import logging
import time

import pandas as pd

from config import (
    ANALYZER_CONFIDENCE_OVERRIDE_THRESHOLD,
    ENABLE_AGENT_LOOP,
    ENABLE_EVIDENCE_PLANNER,
    ENABLE_TYPED_TOOLS,
    ENABLE_QUESTION_ANALYZER_HINTS,
    ENABLE_QUESTION_ANALYZER_SHADOW,
    ENABLE_VECTOR_KNOWLEDGE_HINTS,
    ENABLE_VECTOR_KNOWLEDGE_SHADOW,
)
from models import QueryContext, ResponseMode, ResolutionPolicy
from utils.metrics import metrics
from utils.query_validation import validate_tool_relevance
from agent import planner, sql_executor, analyzer, summarizer, chart_pipeline, orchestrator, evidence_planner
from agent.provenance import clear_provenance, sql_query_hash, stamp_provenance, tool_invocation_hash
from agent.router import match_tool, ROUTER_ENABLE_SEMANTIC_FALLBACK, _last_semantic_scores
from agent.tools import execute_tool
from agent.tools.types import ToolInvocation
from contracts.question_analysis import PreferredPath
from contracts.vector_knowledge import VectorKnowledgeMode
from knowledge.vector_retrieval import format_vector_knowledge_for_prompt, retrieve_vector_knowledge
from utils.trace_logging import trace_detail

log = logging.getLogger("Enai")

_EXPLANATION_ROUTING_SIGNALS = (
    "why",
    "explain",
    "reason",
    "cause",
    "because",
    "driver",
    "drivers",
    "factor",
    "factors",
    "what does this mean",
    "what does that mean",
    "რატომ",
    "ახსენი",
    "почему",
    "объясни",
)
_SCENARIO_DERIVED_METRICS = {
    "scenario_payoff",
    "scenario_scale",
    "scenario_offset",
}

# --- Response-mode derivation constants ---
# Types where the answer mode is unambiguous regardless of preferred_path.
_ALWAYS_KNOWLEDGE_TYPES = {"conceptual_definition", "regulatory_procedure"}
_ALWAYS_DATA_TYPES = {"data_retrieval", "data_explanation", "factual_lookup"}


def _derive_response_mode(ctx: QueryContext) -> str:
    """Derive response_mode once from question-analysis or heuristic fallback.

    Rules:
    - _ALWAYS_KNOWLEDGE_TYPES → knowledge_primary regardless of preferred_path
    - _ALWAYS_DATA_TYPES → data_primary regardless of preferred_path
    - For ambiguous types (comparison, forecast, ambiguous, unsupported):
      preferred_path is the tie-breaker.  knowledge → knowledge_primary,
      anything else → data_primary.
    - When no analyzer is available, fall back to is_conceptual_question()
      which was already computed in prepare_context().
    """
    if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active":
        qa_type = ctx.question_analysis.classification.query_type.value
        qa_path = ctx.question_analysis.routing.preferred_path.value
        if qa_type in _ALWAYS_KNOWLEDGE_TYPES:
            return ResponseMode.KNOWLEDGE_PRIMARY
        if qa_type in _ALWAYS_DATA_TYPES:
            return ResponseMode.DATA_PRIMARY
        # Ambiguous types: comparison, forecast, ambiguous, unsupported
        if qa_path == "knowledge":
            return ResponseMode.KNOWLEDGE_PRIMARY
        return ResponseMode.DATA_PRIMARY
    # No analyzer — use the heuristic already computed in prepare_context()
    return (
        ResponseMode.KNOWLEDGE_PRIMARY if ctx.is_conceptual
        else ResponseMode.DATA_PRIMARY
    )


def _derive_resolution_policy(ctx: QueryContext) -> str:
    """Derive whether the pipeline should answer or request clarification."""

    if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active":
        if ctx.question_analysis.routing.preferred_path == PreferredPath.CLARIFY:
            return ResolutionPolicy.CLARIFY
    return ResolutionPolicy.ANSWER


def _requested_derived_metric_names(ctx: QueryContext) -> list[str]:
    """Return active analyzer requested derived metrics in stable order."""

    if ctx.question_analysis is None or ctx.question_analysis_source != "llm_active":
        return []

    names: list[str] = []
    seen: set[str] = set()
    for metric in ctx.question_analysis.analysis_requirements.derived_metrics or []:
        name = getattr(metric.metric_name, "value", str(metric.metric_name or "")).strip()
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def _missing_requested_evidence(ctx: QueryContext) -> list[str]:
    """Return requested derived metrics that Stage 3 did not materialize."""

    requested = list(ctx.requested_derived_metrics or [])
    if not requested:
        return []

    evidence_names = {
        str(record.get("derived_metric_name") or "").strip()
        for record in (ctx.analysis_evidence or [])
        if str(record.get("derived_metric_name") or "").strip()
    }
    return [name for name in requested if name not in evidence_names]


def _should_block_data_summary_for_missing_evidence(ctx: QueryContext) -> bool:
    """Return True when missing analytical evidence should stop Stage 4."""

    if not ctx.missing_evidence_for_metrics:
        return False
    if ctx.question_analysis is None or ctx.question_analysis_source != "llm_active":
        return False

    query_type = ctx.question_analysis.classification.query_type.value
    if ctx.question_analysis.routing.preferred_path == PreferredPath.CLARIFY:
        return True
    if query_type in {"forecast", "comparison"}:
        return True
    # Comparison-shaped data_explanation (e.g., "compare Jan vs Feb prices") also
    # needs its derived metrics to produce a meaningful answer.
    if query_type == "data_explanation" and ctx.question_analysis.analysis_requirements.derived_metrics:
        return True
    return False


def _enrich_prices_with_composition(
    ctx: QueryContext,
    invocation: ToolInvocation,
    is_explanation: bool,
) -> QueryContext:
    """Auto-fetch balancing composition shares and merge into price data.

    Called for 'why' queries routed to ``get_prices`` so the summarizer
    can explain structural drivers behind price movements.  Safe to call
    from any routing path (keyword, analyzer, agent).
    """
    if (
        not is_explanation
        or invocation.name != "get_prices"
        or ctx.df.empty
        or any(c.startswith("share_") for c in ctx.cols)
    ):
        return ctx

    try:
        comp_invocation = ToolInvocation(
            name="get_balancing_composition",
            params={
                "start_date": invocation.params.get("start_date"),
                "end_date": invocation.params.get("end_date"),
            },
        )
        comp_df, comp_cols, comp_rows = execute_tool(comp_invocation)
        if not comp_df.empty:
            date_col_price = next(
                (c for c in ctx.df.columns if "date" in c.lower()), None,
            )
            date_col_comp = next(
                (c for c in comp_df.columns if "date" in c.lower()), None,
            )
            if date_col_price and date_col_comp:
                ctx.df[date_col_price] = pd.to_datetime(
                    ctx.df[date_col_price], errors="coerce",
                )
                comp_df[date_col_comp] = pd.to_datetime(
                    comp_df[date_col_comp], errors="coerce",
                )
                share_cols = [
                    c for c in comp_df.columns if c.startswith("share_")
                ]
                merge_cols = [date_col_comp] + share_cols
                merged = ctx.df.merge(
                    comp_df[merge_cols].rename(
                        columns={date_col_comp: date_col_price},
                    ),
                    on=date_col_price,
                    how="left",
                )
                ctx.df = merged
                ctx.cols = list(merged.columns)
                ctx.rows = [
                    tuple(r)
                    for r in merged.itertuples(index=False, name=None)
                ]
                log.info(
                    "Enriched price data with %d composition columns for why-query",
                    len(share_cols),
                )
                trace_detail(
                    log, ctx, "composition_enrichment", "result",
                    attempted=True, success=True,
                    share_cols_added=len(share_cols),
                )
                return ctx
        # comp_df was empty
        trace_detail(
            log, ctx, "composition_enrichment", "result",
            attempted=True, success=False, share_cols_added=0,
        )
    except Exception as enrich_err:
        log.warning(
            "Composition enrichment for why-query failed: %s", enrich_err,
        )
        trace_detail(
            log, ctx, "composition_enrichment", "result",
            attempted=True, success=False, share_cols_added=0,
            error=str(enrich_err),
        )
    return ctx


def _should_route_tool_as_explanation(ctx: QueryContext) -> bool:
    """Return True when tool routing should use explanation-style handling."""
    query_lower = (ctx.query or "").strip().lower()
    explanation_signal_hit = any(signal in query_lower for signal in _EXPLANATION_ROUTING_SIGNALS)

    if ctx.question_analysis:
        qa_type = ctx.question_analysis.classification.query_type.value
        if qa_type == "conceptual_definition":
            return True
        if qa_type != "data_explanation":
            return False

        # Authoritative signal: analyzer explicitly requests driver analysis
        if ctx.question_analysis.analysis_requirements.needs_driver_analysis:
            return True

        intent_text = str(ctx.question_analysis.classification.intent or "").strip().lower()
        intent_signal_hit = any(
            signal in intent_text
            for signal in ("why", "explain", "reason", "cause", "driver", "factor")
        )
        derived_metrics = list(ctx.question_analysis.analysis_requirements.derived_metrics or [])
        has_scenario_metric = any(
            getattr(metric.metric_name, "value", str(metric.metric_name)) in _SCENARIO_DERIVED_METRICS
            for metric in derived_metrics
        )
        if has_scenario_metric and not explanation_signal_hit and not intent_signal_hit:
            return False
        return True

    return explanation_signal_hit


def _derive_resolved_query(ctx: QueryContext) -> tuple[str, str]:
    """Return the downstream resolved query and its authority source.

    Only active analyzer output may influence runtime behavior. Shadow-mode
    canonicalization remains useful for observability but must not change
    routing or fallback prompts.
    """
    if (
        ctx.question_analysis is not None
        and ctx.question_analysis_source == "llm_active"
        and ctx.question_analysis.canonical_query_en.strip()
    ):
        return ctx.question_analysis.canonical_query_en, "llm_active_canonical"
    return ctx.query, "raw_query"


def _apply_tool_result(
    ctx: QueryContext,
    invocation: ToolInvocation,
    df: pd.DataFrame,
    cols: list,
    rows: list,
    *,
    is_explanation: bool,
    relevance_query: str | None = None,
) -> QueryContext:
    """Attach a successful tool result to ctx and run shared post-processing."""
    ctx.df = df
    ctx.cols = list(cols)
    ctx.rows = [tuple(r) for r in rows]
    ctx.used_tool = True
    ctx.tool_name = invocation.name
    ctx.tool_params = dict(invocation.params)
    ctx.tool_match_reason = invocation.reason
    ctx.tool_confidence = invocation.confidence
    ctx.tool_fallback_reason = ""
    stamp_provenance(
        ctx,
        ctx.cols,
        ctx.rows,
        source="tool",
        query_hash=tool_invocation_hash(invocation.name, invocation.params),
    )
    ctx.plan.setdefault("intent", "tool_query")
    ctx.plan.setdefault("target", invocation.name)

    tool_relevant, tool_reason = validate_tool_relevance(
        (relevance_query or ctx.query),
        invocation.name,
    )
    if not tool_relevant:
        metrics.log_relevance_block()
        ctx.used_tool = False
        ctx.tool_fallback_reason = f"tool_relevance_blocked:{tool_reason}"
        ctx.df = ctx.df.iloc[0:0]
        ctx.cols = []
        ctx.rows = []
        clear_provenance(ctx)
        log.warning("Recovered tool path blocked by relevance policy. reason=%s", tool_reason)
        return ctx

    log.info("Recovered tool relevance validated. reason=%s", tool_reason)
    if ENABLE_EVIDENCE_PLANNER:
        # Composition enrichment is handled by the evidence loop
        # (Stage 0.8 via COMPOSITION_CONTEXT role); skip inline enrichment.
        return ctx
    return _enrich_prices_with_composition(ctx, invocation, is_explanation)


def _attempt_analyzer_tool_recovery(
    ctx: QueryContext,
    *,
    failed_invocation: ToolInvocation | None,
    is_explanation: bool,
) -> tuple[QueryContext, bool]:
    """Try broader but still deterministic recovery before agent fallback.

    Scope intentionally stays narrow:
    - first try the known safe why-query recovery from composition -> prices
    - then try one deterministic route using the active resolved query
    - if none succeed, let the normal planner/SQL path take over
    """

    candidates: list[ToolInvocation] = []
    if failed_invocation is not None and failed_invocation.name == "get_balancing_composition" and is_explanation:
        candidates.append(
            ToolInvocation(
                name="get_prices",
                params={
                    "start_date": failed_invocation.params.get("start_date"),
                    "end_date": failed_invocation.params.get("end_date"),
                    "metric": "balancing",
                    "currency": "gel",
                    "granularity": "monthly",
                },
                confidence=failed_invocation.confidence * 0.9,
                reason=f"composition_fallback_from:{failed_invocation.reason}",
            )
        )

    recovered_query = (ctx.resolved_query or ctx.query).strip()
    if recovered_query:
        generic_invocation = match_tool(recovered_query, is_explanation=is_explanation)
        if generic_invocation:
            same_as_failed = (
                failed_invocation is not None
                and generic_invocation.name == failed_invocation.name
                and generic_invocation.params == failed_invocation.params
            )
            unsafe_broad_composition = (
                generic_invocation.name == "get_balancing_composition"
                and not generic_invocation.params.get("entities")
            )
            if not same_as_failed and not unsafe_broad_composition:
                candidates.append(generic_invocation)

    seen: set[tuple[str, str]] = set()
    for invocation in candidates:
        dedupe_key = (invocation.name, json.dumps(invocation.params, sort_keys=True))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        try:
            t_tool = time.time()
            df, cols, rows = execute_tool(invocation)
            metrics.log_tool_call(time.time() - t_tool)
            ctx = _apply_tool_result(
                ctx,
                invocation,
                df,
                cols,
                rows,
                is_explanation=is_explanation,
                relevance_query=recovered_query,
            )
            if ctx.used_tool:
                log.info(
                    "Analyzer tool recovery succeeded. recovered_tool=%s rows=%d",
                    invocation.name,
                    len(ctx.rows),
                )
                return ctx, True
        except Exception as exc:
            log.warning("Analyzer tool recovery candidate failed. tool=%s err=%s", invocation.name, exc)

    return ctx, False


def process_query(
    query: str,
    conversation_history=None,
    trace_id: str = "",
    session_id: str = "",
) -> QueryContext:
    """Run the full query pipeline and return a populated QueryContext."""
    ctx = QueryContext(
        query=query,
        conversation_history=conversation_history,
        trace_id=trace_id,
        session_id=session_id,
    )

    def _trace_stage(stage_name: str, started_at: float, **extra):
        elapsed_ms = max(0.0, (time.time() - started_at) * 1000.0)
        ctx.stage_timings_ms[stage_name] = round(elapsed_ms, 2)
        metrics.log_stage(stage_name, elapsed_ms)
        payload = {
            "trace_id": ctx.trace_id,
            "session_id": ctx.session_id,
            "stage": stage_name,
            "duration_ms": round(elapsed_ms, 2),
        }
        if extra:
            payload["extra"] = extra
        log.info("TRACE %s", json.dumps(payload, ensure_ascii=True, sort_keys=True))

    # Stage 0: cheap preparation
    t_stage = time.time()
    ctx = planner.prepare_context(ctx)
    _trace_stage("stage_0_prepare_context", t_stage, conceptual=ctx.is_conceptual, lang=ctx.lang_code)

    # Stage 0.2: structured question analysis
    if ENABLE_QUESTION_ANALYZER_SHADOW or ENABLE_QUESTION_ANALYZER_HINTS:
        t_stage = time.time()
        analyzer_mode = "active" if ENABLE_QUESTION_ANALYZER_HINTS else "shadow"
        if ENABLE_QUESTION_ANALYZER_HINTS:
            ctx = planner.analyze_question_active(ctx)
        else:
            ctx = planner.analyze_question_shadow(ctx)
        qa_type = ""
        qa_path = ""
        qa_conf = 0.0
        analyzer_conceptual = False
        conceptual_disagree = False
        mode_disagree = False
        if ctx.question_analysis is not None:
            qa_type = ctx.question_analysis.classification.query_type.value
            qa_path = ctx.question_analysis.routing.preferred_path.value
            qa_conf = ctx.question_analysis.classification.confidence
            
            # Compute analyzer_conceptual for tracing only — the authoritative
            # response_mode derivation happens after Stage 0.3 via
            # _derive_response_mode() which uses a stricter rule set.
            analyzer_conceptual = (
                qa_type in _ALWAYS_KNOWLEDGE_TYPES
                or (qa_path == "knowledge" and qa_type not in _ALWAYS_DATA_TYPES)
            )

            conceptual_disagree = analyzer_conceptual != bool(ctx.is_conceptual)
            mode_disagree = ctx.question_analysis.classification.analysis_mode.value != str(ctx.mode)
        _trace_stage(
            "stage_0_2_question_analyzer",
            t_stage,
            mode=analyzer_mode,
            ok=bool(ctx.question_analysis),
            error=bool(ctx.question_analysis_error),
            query_type=qa_type,
            preferred_path=qa_path,
            confidence=qa_conf,
            heuristic_conceptual=bool(ctx.is_conceptual),
            analyzer_conceptual=analyzer_conceptual,
            conceptual_disagree=conceptual_disagree,
            heuristic_mode=str(ctx.mode),
            analyzer_mode=(ctx.question_analysis.classification.analysis_mode.value if ctx.question_analysis else ""),
            mode_disagree=mode_disagree,
        )

    routing_query, routing_query_source = _derive_resolved_query(ctx)
    ctx.resolved_query = routing_query
    ctx.resolved_query_source = routing_query_source
    if routing_query_source == "llm_active_canonical":
        ctx.semantic_locked = True

    # Stage 0.3: vector-backed knowledge retrieval (shadow/active collection only)
    if ENABLE_VECTOR_KNOWLEDGE_SHADOW or ENABLE_VECTOR_KNOWLEDGE_HINTS:
        t_stage = time.time()
        retrieval_mode = "active" if ENABLE_VECTOR_KNOWLEDGE_HINTS else "shadow"
        bundle = retrieve_vector_knowledge(
            routing_query,
            retrieval_mode=(
                VectorKnowledgeMode.active
                if ENABLE_VECTOR_KNOWLEDGE_HINTS
                else VectorKnowledgeMode.shadow
            ),
            question_analysis=ctx.question_analysis,
        )
        ctx.vector_knowledge = bundle
        ctx.vector_knowledge_source = f"vector_{retrieval_mode}"
        ctx.vector_knowledge_error = bundle.error
        ctx.vector_knowledge_prompt = format_vector_knowledge_for_prompt(bundle) if not bundle.error else ""
        top_sources = [chunk.document_title or chunk.source_key for chunk in bundle.chunks[:3]]
        top_sections = [
            f"{chunk.document_title or chunk.source_key} | {chunk.section_title or chunk.section_path or f'chunk_{chunk.chunk_index}'}"
            for chunk in bundle.chunks[:3]
        ]
        trace_detail(
            log,
            ctx,
            "stage_0_3_vector_knowledge",
            "validated",
            mode=retrieval_mode,
            chunk_count=bundle.chunk_count,
            strategy=bundle.strategy.value,
            preferred_topics=bundle.filters.preferred_topics,
            top_sources=top_sources,
            top_sections=top_sections,
            error=bundle.error,
        )
        _trace_stage(
            "stage_0_3_vector_knowledge",
            t_stage,
            mode=retrieval_mode,
            chunk_count=bundle.chunk_count,
            error=bool(bundle.error),
            strategy=bundle.strategy.value,
        )

    # --- Derive response_mode (single source of truth for answer mode) ---
    ctx.response_mode = _derive_response_mode(ctx)
    ctx.resolution_policy = _derive_resolution_policy(ctx)
    ctx.requested_derived_metrics = _requested_derived_metric_names(ctx)
    # Keep is_conceptual in sync for backward compatibility with stages that
    # still read it, but no stage should ever re-derive this independently.
    ctx.is_conceptual = ctx.response_mode == ResponseMode.KNOWLEDGE_PRIMARY
    log.info(
        "Response mode derived: %s | resolution_policy=%s",
        ctx.response_mode,
        ctx.resolution_policy,
    )

    # Set policy-blocked flags for observability before the short-circuit return.
    if ctx.response_mode == ResponseMode.KNOWLEDGE_PRIMARY or ctx.resolution_policy == ResolutionPolicy.CLARIFY:
        if ENABLE_TYPED_TOOLS:
            ctx.tool_blocked_by_policy = True
        if ENABLE_AGENT_LOOP:
            ctx.agent_loop_blocked_by_policy = True

    trace_detail(
        log, ctx, "response_mode_derivation", "result",
        response_mode=ctx.response_mode,
        resolution_policy=ctx.resolution_policy,
        is_conceptual=ctx.is_conceptual,
        tool_blocked_by_policy=ctx.tool_blocked_by_policy,
        agent_loop_blocked_by_policy=ctx.agent_loop_blocked_by_policy,
        clarify_reason=ctx.clarify_reason,
        requested_derived_metrics=list(ctx.requested_derived_metrics or []),
        analyzer_available=ctx.question_analysis is not None,
        analyzer_source=ctx.question_analysis_source,
        semantic_locked=ctx.semantic_locked,
    )

    if ctx.resolution_policy == ResolutionPolicy.CLARIFY:
        if not ctx.clarify_reason:
            ctx.clarify_reason = "analyzer_preferred_path_clarify"
        t_stage = time.time()
        ctx = summarizer.answer_clarify(ctx)
        _trace_stage("stage_4_clarify_summary", t_stage, reason=ctx.clarify_reason)
        return ctx

    # Conceptual short-circuit
    if ctx.response_mode == ResponseMode.KNOWLEDGE_PRIMARY:
        t_stage = time.time()
        ctx = summarizer.answer_conceptual(ctx)
        _trace_stage("stage_4_conceptual_summary", t_stage)
        return ctx

    # Stage 0.4: Evidence planning (multi-tool expansion)
    if ENABLE_EVIDENCE_PLANNER:
        t_stage = time.time()
        ctx = evidence_planner.build_evidence_plan(ctx)
        _trace_stage(
            "stage_0_4_evidence_plan", t_stage,
            steps=len(ctx.evidence_plan),
            source=ctx.evidence_plan_source,
            tools=[s["tool_name"] for s in ctx.evidence_plan],
        )

    # Stage 0.5: pre-LLM typed tool routing
    if ENABLE_TYPED_TOOLS:
        t_stage = time.time()

        is_exp = _should_route_tool_as_explanation(ctx)

        # Plan-driven invocation: when evidence plan exists, execute the
        # first unsatisfied step instead of keyword-matching.  Keyword
        # routing remains as fallback when the plan is empty.
        _plan_step_used = None
        if ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan:
            _plan_step = evidence_planner.next_unsatisfied_step(ctx.evidence_plan)
            if _plan_step:
                invocation = ToolInvocation(
                    name=_plan_step["tool_name"],
                    params=_plan_step["params"],
                    confidence=0.85,
                    reason=f"evidence_plan:{_plan_step['role']}",
                )
                _plan_step_used = _plan_step
                _trace_stage(
                    "stage_0_5_plan_driven", t_stage,
                    plan_tool=_plan_step["tool_name"],
                    plan_role=_plan_step["role"],
                )
            else:
                # All plan steps already satisfied (e.g. by earlier stages)
                invocation = None
        else:
            invocation = match_tool(ctx.query, is_explanation=is_exp)
        _trace_stage("stage_0_5_router_match", t_stage, matched=bool(invocation), plan_driven=bool(_plan_step_used))
        if invocation:
            if _plan_step_used:
                metrics.log_router_match("plan_driven")
            elif "semantic fallback" in (invocation.reason or "").lower():
                metrics.log_router_match("semantic")
            else:
                metrics.log_router_match("deterministic")
            ctx.used_tool = True
            ctx.tool_name = invocation.name
            ctx.tool_params = dict(invocation.params)
            ctx.tool_match_reason = invocation.reason
            ctx.tool_confidence = invocation.confidence

            try:
                t_tool = time.time()
                df, cols, rows = execute_tool(invocation)
                metrics.log_tool_call(time.time() - t_tool)
                _trace_stage("stage_0_6_tool_execute", t_tool, tool=invocation.name, rows=len(rows))

                ctx.df = df
                ctx.cols = list(cols)
                ctx.rows = [tuple(r) for r in rows]
                stamp_provenance(
                    ctx,
                    ctx.cols,
                    ctx.rows,
                    source="tool",
                    query_hash=tool_invocation_hash(invocation.name, invocation.params),
                )
                ctx.plan.setdefault("intent", "tool_query")
                ctx.plan.setdefault("target", invocation.name)
                _relevance_q = ctx.resolved_query or ctx.query
                tool_relevant, tool_reason = validate_tool_relevance(_relevance_q, invocation.name)
                if not tool_relevant:
                    metrics.log_relevance_block()
                    ctx.used_tool = False
                    ctx.tool_fallback_reason = f"tool_relevance_blocked:{tool_reason}"
                    metrics.log_tool_fallback_intent(ctx.query, "tool_relevance_blocked")
                    ctx.df = ctx.df.iloc[0:0]
                    ctx.cols = []
                    ctx.rows = []
                    clear_provenance(ctx)
                    log.warning("Typed tool relevance blocked. reason=%s", tool_reason)
                else:
                    log.info("Typed tool relevance validated. reason=%s", tool_reason)
                    if not ENABLE_EVIDENCE_PLANNER:
                        ctx = _enrich_prices_with_composition(ctx, invocation, is_exp)

                    # Multi-evidence: store result under the matching plan step.
                    # When plan-driven, _plan_step_used is the exact step;
                    # otherwise fall back to name-based matching.
                    if ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan:
                        _matched_step = _plan_step_used or next(
                            (s for s in ctx.evidence_plan if s["tool_name"] == invocation.name and not s.get("satisfied")),
                            None,
                        )
                        if _matched_step:
                            ctx.evidence_collected[_matched_step["role"]] = {
                                "tool": invocation.name,
                                "df": ctx.df.copy(),
                                "cols": list(ctx.cols),
                                "rows": list(ctx.rows),
                            }
                            _matched_step["satisfied"] = True
                            log.info(
                                "Evidence plan: stored Stage 0.5 result as %s, %d steps remaining",
                                _matched_step["role"],
                                sum(1 for s in ctx.evidence_plan if not s.get("satisfied")),
                            )
                        else:
                            log.info(
                                "Evidence plan: Stage 0.5 tool %s not in plan; result stored on ctx.df only",
                                invocation.name,
                            )

                log.info(
                    "Typed tool route hit: tool=%s confidence=%.2f reason=%s",
                    invocation.name,
                    invocation.confidence,
                    invocation.reason,
                )
            except Exception as exc:
                metrics.log_tool_error()
                ctx.used_tool = False
                ctx.tool_fallback_reason = str(exc)
                metrics.log_tool_fallback_intent(ctx.query, "tool_execution_error")
                clear_provenance(ctx)
                # Mark the plan step as failed so Stage 0.8 skips it
                # instead of retrying the same deterministic failure.
                if _plan_step_used:
                    _plan_step_used["error"] = f"stage_0_5:{exc}"
                log.warning(
                    "Typed tool failed; falling back to SQL path. tool=%s err=%s",
                    invocation.name,
                    exc,
                )
        else:
            metrics.log_router_match("miss")
            trace_detail(
                log, ctx, "stage_0_5_router_match", "miss_detail",
                semantic_fallback_enabled=ROUTER_ENABLE_SEMANTIC_FALLBACK,
                semantic_scores=_last_semantic_scores.copy(),
            )

            # --- Stage 0.7: LLM analyzer-driven tool routing (fallback) ---
            # Fires when Stage 0.5 keyword/plan-driven routing misses.
            # When ENABLE_EVIDENCE_PLANNER is on, this is a second chance
            # to satisfy the primary plan step; secondary steps are handled
            # by Stage 0.8's evidence loop.  Legacy recovery via
            # _attempt_analyzer_tool_recovery() only fires when the planner
            # has no actionable steps remaining (all satisfied or errored).
            # Only use analyzer output when running in active/hints mode —
            # shadow mode must never influence routing decisions.
            t_stage = time.time()
            analyzer_invocation = None
            analyzer_build_error = ""
            if ctx.question_analysis and ENABLE_QUESTION_ANALYZER_HINTS:
                try:
                    analyzer_invocation = planner.build_tool_invocation_from_analysis(
                        ctx.question_analysis, ctx.query,
                    )
                except Exception as exc:
                    analyzer_build_error = str(exc)
                    ctx.tool_fallback_reason = f"analyzer_tool_build_error:{exc}"
                    log.warning("Analyzer tool invocation build failed: %s", exc)

            # Trace the analyzer routing decision regardless of outcome
            qa = ctx.question_analysis
            trace_detail(
                log, ctx, "stage_0_7_analyzer_route", "decision",
                invocation_built=bool(analyzer_invocation),
                preferred_path=qa.routing.preferred_path.value if qa else "",
                prefer_tool=qa.routing.prefer_tool if qa else False,
                top_tool=analyzer_invocation.name if analyzer_invocation else None,
                top_score=analyzer_invocation.confidence if analyzer_invocation else None,
                build_error=analyzer_build_error,
                analyzer_available=bool(qa),
                hints_enabled=ENABLE_QUESTION_ANALYZER_HINTS,
            )

            if analyzer_invocation:
                _trace_stage(
                    "stage_0_7_analyzer_route", t_stage,
                    tool=analyzer_invocation.name,
                    confidence=analyzer_invocation.confidence,
                )
                metrics.log_router_match("analyzer")
                ctx.used_tool = True
                ctx.tool_name = analyzer_invocation.name
                ctx.tool_params = dict(analyzer_invocation.params)
                ctx.tool_match_reason = analyzer_invocation.reason
                ctx.tool_confidence = analyzer_invocation.confidence

                try:
                    t_tool = time.time()
                    df, cols, rows = execute_tool(analyzer_invocation)
                    metrics.log_tool_call(time.time() - t_tool)
                    _trace_stage(
                        "stage_0_7_analyzer_tool_execute", t_tool,
                        tool=analyzer_invocation.name, rows=len(rows),
                    )
                    ctx.df = df
                    ctx.cols = list(cols)
                    ctx.rows = [tuple(r) for r in rows]
                    stamp_provenance(
                        ctx, ctx.cols, ctx.rows, source="tool",
                        query_hash=tool_invocation_hash(analyzer_invocation.name, analyzer_invocation.params),
                    )
                    ctx.plan.setdefault("intent", "tool_query")
                    ctx.plan.setdefault("target", analyzer_invocation.name)
                    _relevance_q = ctx.resolved_query or ctx.query
                    tool_relevant, tool_reason = validate_tool_relevance(_relevance_q, analyzer_invocation.name)
                    if not tool_relevant:
                        metrics.log_relevance_block()
                        ctx.used_tool = False
                        ctx.tool_fallback_reason = f"analyzer_tool_relevance_blocked:{tool_reason}"
                        ctx.df = ctx.df.iloc[0:0]
                        ctx.cols = []
                        ctx.rows = []
                        clear_provenance(ctx)
                        log.warning("Analyzer tool relevance blocked. reason=%s", tool_reason)
                    else:
                        log.info(
                            "Analyzer tool route hit: tool=%s confidence=%.2f reason=%s",
                            analyzer_invocation.name,
                            analyzer_invocation.confidence,
                            analyzer_invocation.reason,
                        )
                        if not ENABLE_EVIDENCE_PLANNER:
                            ctx = _enrich_prices_with_composition(ctx, analyzer_invocation, is_exp)

                        # Multi-evidence: store result under matching plan step
                        if ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan:
                            _matched_step = next(
                                (s for s in ctx.evidence_plan
                                 if s["tool_name"] == analyzer_invocation.name and not s.get("satisfied")),
                                None,
                            )
                            if _matched_step:
                                ctx.evidence_collected[_matched_step["role"]] = {
                                    "tool": analyzer_invocation.name,
                                    "df": ctx.df.copy(),
                                    "cols": list(ctx.cols),
                                    "rows": list(ctx.rows),
                                }
                                _matched_step["satisfied"] = True
                except Exception as exc:
                    # --- Fallback: when evidence planner is on, Stage 0.8
                    # handles remaining steps; skip ad-hoc recovery.  When
                    # off, fall back to the composition→prices recovery. ---
                    if ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan and evidence_planner.has_unsatisfied_steps(ctx.evidence_plan):
                        # Mark the matching plan step as failed so Stage 0.8
                        # advances past it instead of retrying.
                        _failed_step = next(
                            (s for s in ctx.evidence_plan
                             if s["tool_name"] == analyzer_invocation.name and not s.get("satisfied")),
                            None,
                        )
                        if _failed_step:
                            _failed_step["error"] = f"stage_0_7:{exc}"
                        log.info(
                            "Analyzer tool failed; evidence plan has remaining steps. tool=%s err=%s",
                            analyzer_invocation.name, exc,
                        )
                        ctx.used_tool = False
                        ctx.tool_fallback_reason = f"analyzer_tool_execution_error:{exc}"
                        clear_provenance(ctx)
                    else:
                        ctx, recovered = _attempt_analyzer_tool_recovery(
                            ctx,
                            failed_invocation=analyzer_invocation,
                            is_explanation=is_exp,
                        )

                        if not recovered:
                            metrics.log_tool_error()
                            ctx.used_tool = False
                            ctx.tool_fallback_reason = f"analyzer_tool_execution_error:{exc}"
                            metrics.log_tool_fallback_intent(ctx.query, "analyzer_tool_execution_error")
                            clear_provenance(ctx)
                            log.warning(
                                "Analyzer tool failed; falling back. tool=%s err=%s",
                                analyzer_invocation.name, exc,
                            )
            else:
                if analyzer_build_error:
                    if ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan and evidence_planner.has_unsatisfied_steps(ctx.evidence_plan):
                        log.info("Analyzer tool build failed; evidence plan has remaining steps.")
                    else:
                        ctx, recovered = _attempt_analyzer_tool_recovery(
                            ctx,
                            failed_invocation=None,
                            is_explanation=is_exp,
                        )
                        if recovered:
                            log.info("Analyzer tool build failure recovered via resolved-query fallback.")
                        else:
                            metrics.log_tool_fallback_intent(ctx.query, "analyzer_tool_build_error")
                else:
                    metrics.log_tool_fallback_intent(ctx.query, "router_no_match")

    # Stage 0.8: Evidence completeness loop
    if ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan and not all(
        s.get("satisfied") for s in ctx.evidence_plan
    ):
        t_stage = time.time()
        ctx = evidence_planner.execute_remaining_evidence(ctx)
        _trace_stage(
            "stage_0_8_evidence_loop", t_stage,
            complete=ctx.evidence_plan_complete,
            collected=len(ctx.evidence_collected),
            satisfied=[s["tool_name"] for s in ctx.evidence_plan if s.get("satisfied")],
        )

    # Stage 1/2: fallback SQL path when no tool route was used
    # When the analyzer understood the query but its tool invocation failed,
    # prefer the planner+SQL path (which consumes question_analysis) over the
    # context-blind agent loop.  The agent loop remains the last resort.
    _analyzer_tool_failed = (
        not ctx.used_tool
        and ctx.tool_fallback_reason
        and ctx.tool_fallback_reason.startswith("analyzer_tool_")
    )
    _evidence_plan_satisfied = (
        ENABLE_EVIDENCE_PLANNER
        and ctx.evidence_plan
        and ctx.evidence_plan_complete
    )
    if not ctx.used_tool and not _evidence_plan_satisfied and ENABLE_AGENT_LOOP and not _analyzer_tool_failed:
        t_stage = time.time()
        ctx = orchestrator.run_agent_loop(ctx)
        _trace_stage("stage_agent_loop", t_stage, outcome=ctx.agent_outcome, rounds=ctx.agent_rounds)
        if ctx.agent_outcome == "conceptual_exit":
            log.info("Agent conceptual exit | rounds=%s", ctx.agent_rounds)
            return ctx
        if ctx.agent_outcome == "data_exit":
            _relevance_q = ctx.resolved_query or ctx.query
            tool_relevant, tool_reason = validate_tool_relevance(_relevance_q, ctx.tool_name or "")
            if not tool_relevant:
                metrics.log_relevance_block()
                ctx.used_tool = False
                ctx.agent_outcome = "fallback_exit"
                ctx.agent_fallback_reason = "agent_tool_relevance_blocked"
                ctx.tool_fallback_reason = f"agent_tool_relevance_blocked:{tool_reason}"
                metrics.log_tool_fallback_intent(ctx.query, "agent_tool_relevance_blocked")
                ctx.df = ctx.df.iloc[0:0]
                ctx.cols = []
                ctx.rows = []
                clear_provenance(ctx)
                log.warning("Agent data exit blocked by relevance policy. reason=%s", tool_reason)
            else:
                log.info("Agent tool relevance validated. reason=%s", tool_reason)
            log.info("Agent data exit | tool=%s | rows=%s", ctx.tool_name, len(ctx.rows))
        elif ctx.agent_outcome == "fallback_exit":
            log.info("Agent fallback exit | reason=%s", ctx.agent_fallback_reason)

    # Stage 1/2: hard-coded legacy fallback path when no tool route was used
    if not ctx.used_tool:
        if ctx.tool_fallback_reason:
            metrics.log_tool_fallback_intent(ctx.query, f"tool_fallback:{ctx.tool_fallback_reason}")
        t_stage = time.time()
        ctx = planner.generate_plan(ctx)
        _trace_stage("stage_1_generate_plan", t_stage, conceptual=ctx.is_conceptual, skip_sql=ctx.skip_sql)
        log.info("Stage 1 complete | conceptual=%s | skip_sql=%s", ctx.is_conceptual, ctx.skip_sql)

        if ctx.is_conceptual or ctx.skip_sql:
            if ctx.skip_sql and not ctx.is_conceptual:
                log.info("Skipping SQL path: %s", ctx.skip_sql_reason)
            t_stage = time.time()
            ctx = summarizer.answer_conceptual(ctx)
            _trace_stage("stage_4_conceptual_summary", t_stage)
            return ctx

        t_stage = time.time()
        ctx = sql_executor.validate_and_execute(ctx)
        _trace_stage("stage_2_sql_execute", t_stage, rows=len(ctx.rows), cols=len(ctx.cols))
        log.info("Stage 2 complete | rows=%s | cols=%s", len(ctx.rows), len(ctx.cols))
        if ctx.skip_sql:
            log.info("Stage 2 blocked by policy: %s", ctx.skip_sql_reason)
            t_stage = time.time()
            ctx = summarizer.answer_conceptual(ctx)
            _trace_stage("stage_4_conceptual_summary", t_stage)
            return ctx
    else:
        log.info("Stage 2 bypassed | tool=%s | rows=%s", ctx.tool_name, len(ctx.rows))

    # Snapshot the source tabular output before analyzer mutates/augments rows.
    if ctx.rows and ctx.cols and not ctx.provenance_rows:
        inferred_source = str(ctx.provenance_source or "")
        inferred_hash = str(ctx.provenance_query_hash or "")
        if not inferred_hash:
            if ctx.used_tool and ctx.tool_name:
                inferred_source = inferred_source or "tool"
                inferred_hash = tool_invocation_hash(ctx.tool_name, ctx.tool_params)
            elif ctx.safe_sql:
                inferred_source = inferred_source or "sql"
                inferred_hash = sql_query_hash(ctx.safe_sql)
        stamp_provenance(
            ctx,
            ctx.cols,
            ctx.rows,
            source=inferred_source,
            query_hash=inferred_hash,
        )

    # Stage 3: enrich
    t_stage = time.time()
    ctx = analyzer.enrich(ctx)
    _trace_stage(
        "stage_3_analyzer_enrich",
        t_stage,
        share_override=bool(ctx.share_summary_override),
        correlation_keys=list(ctx.correlation_results.keys()),
    )
    log.info("Stage 3 complete | analysis enrichment done")

    ctx.missing_evidence_for_metrics = _missing_requested_evidence(ctx)
    trace_detail(
        log,
        ctx,
        "stage_3_analyzer_enrich",
        "evidence_readiness",
        requested_derived_metrics=list(ctx.requested_derived_metrics or []),
        missing_evidence_for_metrics=list(ctx.missing_evidence_for_metrics or []),
    )

    if _should_block_data_summary_for_missing_evidence(ctx):
        # If some evidence was materialized, allow Stage 4 with partial data
        # rather than forcing a clarification that discards all available context.
        if ctx.analysis_evidence:
            log.info(
                "Partial evidence available (%d records); proceeding to Stage 4 "
                "despite missing: %s",
                len(ctx.analysis_evidence), ctx.missing_evidence_for_metrics,
            )
            ctx.data_summary_blocked_reason = (
                "partial_evidence:" + ",".join(ctx.missing_evidence_for_metrics)
            )
            # Fall through to Stage 4 summarize_data below
        else:
            ctx.resolution_policy = ResolutionPolicy.CLARIFY
            ctx.clarify_reason = "missing_requested_analysis_evidence"
            ctx.data_summary_blocked_reason = (
                "missing_derived_evidence:" + ",".join(ctx.missing_evidence_for_metrics)
            )
            ctx.tool_blocked_by_policy = ctx.tool_blocked_by_policy or ENABLE_TYPED_TOOLS
            ctx.agent_loop_blocked_by_policy = ctx.agent_loop_blocked_by_policy or ENABLE_AGENT_LOOP
            log.info(
                "Blocking data summarization due to missing derived evidence: %s",
                ctx.missing_evidence_for_metrics,
            )
            t_stage = time.time()
            ctx = summarizer.answer_clarify(ctx)
            _trace_stage(
                "stage_4_clarify_summary",
                t_stage,
                reason=ctx.clarify_reason,
                missing_evidence=",".join(ctx.missing_evidence_for_metrics),
            )
            return ctx

    # Append structured evidence summary when evidence planner contributed data
    if ENABLE_EVIDENCE_PLANNER and ctx.evidence_collected:
        evidence_summary_parts = []
        for role, evidence in ctx.evidence_collected.items():
            tool = evidence.get("tool", "unknown")
            row_count = len(evidence.get("rows", []))
            col_names = evidence.get("cols", [])
            evidence_summary_parts.append(
                f"  {role}: {tool} ({row_count} rows, columns: {', '.join(col_names[:8])})"
            )
        if evidence_summary_parts:
            ctx.stats_hint = (
                (ctx.stats_hint or "")
                + "\n\nEVIDENCE SOURCES:\n"
                + "\n".join(evidence_summary_parts)
            )

    # Stage 4: summarize
    t_stage = time.time()
    ctx = summarizer.summarize_data(ctx)
    _trace_stage(
        "stage_4_summarize_data",
        t_stage,
        summary_source=ctx.summary_source,
        gate_passed=ctx.summary_provenance_gate_passed,
        gate_reason=ctx.summary_provenance_gate_reason,
        coverage=ctx.summary_provenance_coverage,
    )
    log.info("Stage 4 complete | summary generated")

    # Stage 5: chart
    t_stage = time.time()
    ctx = chart_pipeline.build_chart(ctx)
    _trace_stage("stage_5_chart_build", t_stage, chart_type=ctx.chart_type or "")
    log.info("Stage 5 complete | chart_type=%s", ctx.chart_type)

    return ctx
