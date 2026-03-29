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
    ENABLE_TYPED_TOOLS,
    ENABLE_QUESTION_ANALYZER_HINTS,
    ENABLE_QUESTION_ANALYZER_SHADOW,
    ENABLE_VECTOR_KNOWLEDGE_HINTS,
    ENABLE_VECTOR_KNOWLEDGE_SHADOW,
)
from models import QueryContext, ResponseMode
from utils.metrics import metrics
from utils.query_validation import validate_tool_relevance
from agent import planner, sql_executor, analyzer, summarizer, chart_pipeline, orchestrator
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

    # Stage 0.3: vector-backed knowledge retrieval (shadow/active collection only)
    if ENABLE_VECTOR_KNOWLEDGE_SHADOW or ENABLE_VECTOR_KNOWLEDGE_HINTS:
        t_stage = time.time()
        retrieval_mode = "active" if ENABLE_VECTOR_KNOWLEDGE_HINTS else "shadow"
        routing_query = ctx.query
        if ctx.question_analysis is not None and ctx.question_analysis.canonical_query_en.strip():
            routing_query = ctx.question_analysis.canonical_query_en
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
    # Keep is_conceptual in sync for backward compatibility with stages that
    # still read it, but no stage should ever re-derive this independently.
    ctx.is_conceptual = ctx.response_mode == ResponseMode.KNOWLEDGE_PRIMARY
    log.info("Response mode derived: %s", ctx.response_mode)

    # Set policy-blocked flags for observability before the short-circuit return.
    if ctx.response_mode == ResponseMode.KNOWLEDGE_PRIMARY:
        if ENABLE_TYPED_TOOLS:
            ctx.tool_blocked_by_policy = True
        if ENABLE_AGENT_LOOP:
            ctx.agent_loop_blocked_by_policy = True

    trace_detail(
        log, ctx, "response_mode_derivation", "result",
        response_mode=ctx.response_mode,
        is_conceptual=ctx.is_conceptual,
        tool_blocked_by_policy=ctx.tool_blocked_by_policy,
        agent_loop_blocked_by_policy=ctx.agent_loop_blocked_by_policy,
        analyzer_available=ctx.question_analysis is not None,
        analyzer_source=ctx.question_analysis_source,
    )

    # Conceptual short-circuit
    if ctx.response_mode == ResponseMode.KNOWLEDGE_PRIMARY:
        t_stage = time.time()
        ctx = summarizer.answer_conceptual(ctx)
        _trace_stage("stage_4_conceptual_summary", t_stage)
        return ctx

    # Stage 0.5: pre-LLM typed tool routing
    if ENABLE_TYPED_TOOLS:
        t_stage = time.time()

        is_exp = False
        if ctx.question_analysis:
            qa_type = ctx.question_analysis.classification.query_type.value
            if qa_type in ("data_explanation", "conceptual_definition"):
                is_exp = True
        elif any(w in ctx.query.lower() for w in ["why", "explain", "reason", "რატომ", "ახსენი", "почему", "объясни"]):
            is_exp = True
            
        is_exp = _should_route_tool_as_explanation(ctx)
        invocation = match_tool(ctx.query, is_explanation=is_exp)
        _trace_stage("stage_0_5_router_match", t_stage, matched=bool(invocation))
        if invocation:
            if "semantic fallback" in (invocation.reason or "").lower():
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
                tool_relevant, tool_reason = validate_tool_relevance(ctx.query, invocation.name)
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
                    ctx = _enrich_prices_with_composition(ctx, invocation, is_exp)

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
            # When the keyword router misses, check whether the question
            # analyzer (which sees conversation history) identified a tool.
            # Only use analyzer output when running in active/hints mode —
            # shadow mode must never influence routing decisions.
            t_stage = time.time()
            analyzer_invocation = None
            if ctx.question_analysis and ENABLE_QUESTION_ANALYZER_HINTS:
                try:
                    analyzer_invocation = planner.build_tool_invocation_from_analysis(
                        ctx.question_analysis, ctx.query,
                    )
                except Exception as exc:
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
                    tool_relevant, tool_reason = validate_tool_relevance(ctx.query, analyzer_invocation.name)
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
                        ctx = _enrich_prices_with_composition(ctx, analyzer_invocation, is_exp)
                except Exception as exc:
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
                metrics.log_tool_fallback_intent(ctx.query, "router_no_match")

    # Stage 1/2: fallback SQL path when no tool route was used
    if not ctx.used_tool and ENABLE_AGENT_LOOP:
        t_stage = time.time()
        ctx = orchestrator.run_agent_loop(ctx)
        _trace_stage("stage_agent_loop", t_stage, outcome=ctx.agent_outcome, rounds=ctx.agent_rounds)
        if ctx.agent_outcome == "conceptual_exit":
            log.info("Agent conceptual exit | rounds=%s", ctx.agent_rounds)
            return ctx
        if ctx.agent_outcome == "data_exit":
            tool_relevant, tool_reason = validate_tool_relevance(ctx.query, ctx.tool_name or "")
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
