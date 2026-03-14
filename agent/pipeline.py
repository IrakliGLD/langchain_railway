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

from config import (
    ENABLE_AGENT_LOOP,
    ENABLE_TYPED_TOOLS,
    ENABLE_QUESTION_ANALYZER_HINTS,
    ENABLE_QUESTION_ANALYZER_SHADOW,
)
from models import QueryContext
from utils.metrics import metrics
from utils.query_validation import validate_tool_relevance
from agent import planner, sql_executor, analyzer, summarizer, chart_pipeline, orchestrator
from agent.provenance import clear_provenance, sql_query_hash, stamp_provenance, tool_invocation_hash
from agent.router import match_tool
from agent.tools import execute_tool

log = logging.getLogger("Enai")
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
            analyzer_conceptual = qa_type == "conceptual_definition"
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

    # Conceptual short-circuit
    if ctx.is_conceptual:
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
        why_override=bool(ctx.why_summary_override),
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
