"""
Pipeline Orchestrator — analyzer-first flow with evidence-based rendering.

Stages:
  0.1  planner.prepare_context           -> language / mode / conceptual detection
  0.2  llm.analyze_question (active)     -> QuestionAnalysis with answer_kind / render_style
       answer_kind cross-check           -> safety override vs query_type-derived kind
  0.3  vector knowledge retrieval        -> skipped for deterministic data paths
  0.4  evidence plan derivation          -> tool invocations needed for the answer
  0.5  evidence collection loop          -> frame adapters + canonical evidence frames
  0.6  evidence validation               -> gap detection, render_style degradation
  0.7  router.match_tool (legacy)        -> fallback pre-LLM deterministic routing
  0.8  orchestrator.run_agent_loop       -> bounded typed-tool loop (legacy fallback)
  1/2  planner / sql_executor            -> legacy LLM plan + SQL fallback
  3    analyzer.enrich                   -> stats, correlation, trendlines
  4    summarizer.summarize_data         -> generic renderer or LLM narrative
  5    chart_pipeline.build_chart        -> chart from evidence or SQL results
"""
import json
import logging
import re
import time

import pandas as pd
from sqlalchemy import text

from config import (
    ANALYZER_CONFIDENCE_OVERRIDE_THRESHOLD,
    ENABLE_AGENT_LOOP,
    ENABLE_EVIDENCE_PLANNER,
    ENABLE_TYPED_TOOLS,
    ENABLE_QUESTION_ANALYZER_HINTS,
    ENABLE_QUESTION_ANALYZER_SHADOW,
    ENABLE_VECTOR_KNOWLEDGE_HINTS,
    ENABLE_VECTOR_KNOWLEDGE_SHADOW,
    PIPELINE_MODE,
)
from analysis.shares import compute_entity_price_contributions
from core.query_executor import ENGINE
from models import QueryContext, ResponseMode, ResolutionPolicy
from utils.metrics import metrics
from utils.query_validation import validate_tool_relevance
from agent import planner, sql_executor, analyzer, summarizer, chart_pipeline, orchestrator, evidence_planner
from agent.provenance import clear_provenance, sql_query_hash, stamp_provenance, tool_invocation_hash
from agent.router import match_tool, ROUTER_ENABLE_SEMANTIC_FALLBACK, _last_semantic_scores
from agent.evidence_validator import validate_evidence
from agent.frame_adapters import adapt_tool_result
from agent.tools import execute_tool
from agent.tools.types import ToolInvocation
from analysis.system_quantities import normalize_tool_dataframe
from contracts.question_analysis import AnswerKind, PreferredPath, RenderStyle, _SCENARIO_METRIC_NAMES
from contracts.vector_knowledge import VectorKnowledgeMode, VectorRetrievalTier
from knowledge.vector_retrieval import (
    pack_vector_knowledge_for_prompt,
    retrieve_vector_knowledge,
)
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
_SCENARIO_DERIVED_METRICS = _SCENARIO_METRIC_NAMES
# answer_kind values eligible for scenario-metric override.  Strong structural
# shapes (COMPARISON, LIST, KNOWLEDGE, CLARIFY) are never overridden.
_SCENARIO_OVERRIDE_ELIGIBLE = frozenset({
    AnswerKind.EXPLANATION,
    AnswerKind.SCALAR,
    AnswerKind.TIMESERIES,
})

# --- Response-mode derivation constants ---
# Types where the answer mode is unambiguous regardless of preferred_path.
_ALWAYS_KNOWLEDGE_TYPES = {"conceptual_definition", "regulatory_procedure"}
_ALWAYS_DATA_TYPES = {"data_retrieval", "data_explanation", "factual_lookup"}

_TECHNICAL_CONCEPT_TOKENS = (
    "generation mix",
    "generation",
    "demand",
    "consumption",
    "import dependency",
    "import dependence",
    "energy security",
    "self-sufficiency",
)

_TECHNICAL_CONCEPT_EXPLORATION_TOKENS = (
    "what can you say",
    "tell me about",
    "describe",
    "overview",
    "characteristic",
    "trend",
    "mix",
    "dependency",
    "security",
)

_REGULATORY_CONCEPT_TOKENS = (
    "regulation",
    "law",
    "procedure",
    "eligibility",
    "license",
    "licence",
    "registration",
    "rule",
)


def _should_attempt_authoritative_router_fallback(ctx: QueryContext) -> bool:
    """Allow a narrow raw-router fallback after Stage 0.2 for technical concepts.

    This keeps the general rule intact: authoritative QA should drive routing.
    But when the analyzer classifies a query as conceptual while response-mode
    derivation has already promoted it to DATA_PRIMARY, we still need a second
    chance to reach a typed tool if the analyzer did not nominate one.
    """
    if not ctx.has_authoritative_question_analysis:
        return False
    if ctx.response_mode != ResponseMode.DATA_PRIMARY:
        return False

    qa = ctx.question_analysis
    if qa.classification.query_type.value != "conceptual_definition":
        return False

    query_text = " ".join(
        part for part in (str(ctx.query or ""), str(ctx.effective_query or "")) if part
    ).lower()
    return any(token in query_text for token in _TECHNICAL_CONCEPT_TOKENS)


# Clarification/evidence helpers decide whether later stages can answer safely.
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
    if ctx.has_authoritative_question_analysis:
        qa_type = ctx.question_analysis.classification.query_type.value
        qa_path = ctx.question_analysis.routing.preferred_path.value
        query_text = " ".join(
            part for part in (str(ctx.query or ""), str(ctx.effective_query or "")) if part
        )
        query_lower = query_text.lower()
        if (
            qa_type == "conceptual_definition"
            and any(token in query_lower for token in _TECHNICAL_CONCEPT_TOKENS)
            and any(token in query_lower for token in _TECHNICAL_CONCEPT_EXPLORATION_TOKENS)
            and not any(token in query_lower for token in _REGULATORY_CONCEPT_TOKENS)
        ):
            return ResponseMode.DATA_PRIMARY
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

    if (
        ctx.clarify_selection_override
        and ctx.has_authoritative_question_analysis
        and ctx.question_analysis.routing.preferred_path == PreferredPath.CLARIFY
    ):
        # The user already chose one of the offered clarify branches, so this turn
        # should continue with that interpretation instead of re-asking.
        return ResolutionPolicy.ANSWER

    if ctx.has_authoritative_question_analysis:
        if ctx.question_analysis.routing.preferred_path in (
            PreferredPath.CLARIFY,
            PreferredPath.REJECT,
        ):
            return ResolutionPolicy.CLARIFY
    return ResolutionPolicy.ANSWER


# --- answer_kind cross-check: derive from query_type, compare with LLM-emitted ---

_QUERY_TYPE_TO_ANSWER_KIND: dict[str, AnswerKind] = {
    "conceptual_definition": AnswerKind.KNOWLEDGE,
    "regulatory_procedure": AnswerKind.KNOWLEDGE,
    "factual_lookup": AnswerKind.SCALAR,
    "data_explanation": AnswerKind.EXPLANATION,
    "comparison": AnswerKind.COMPARISON,
    "forecast": AnswerKind.FORECAST,
    "ambiguous": AnswerKind.CLARIFY,
    "unsupported": AnswerKind.CLARIFY,
}

# `data_retrieval` is intentionally omitted: the query_type is too coarse to
# safely infer whether the answer shape should be SCALAR, LIST, or TIMESERIES.
# Overriding an authoritative LIST/SCALAR answer_kind to TIMESERIES corrupts the
# deterministic path for single-period snapshot questions such as "which
# entities had tariffs in July 2023 and what were they?".
_AMBIGUOUS_QUERY_TYPES_FOR_ANSWER_KIND = frozenset({"data_retrieval"})

# answer_kind values considered "safe" — can display any shape without data loss.
_SAFE_ANSWER_KINDS = frozenset({AnswerKind.TIMESERIES, AnswerKind.EXPLANATION, AnswerKind.KNOWLEDGE})


def _derive_answer_kind_from_query_type(ctx) -> AnswerKind | None:
    """Deterministic answer_kind derivation from query_type (fallback + cross-check)."""
    if not ctx.has_authoritative_question_analysis:
        return None
    qa_type = ctx.question_analysis.classification.query_type.value
    if qa_type in _AMBIGUOUS_QUERY_TYPES_FOR_ANSWER_KIND:
        return None
    return _QUERY_TYPE_TO_ANSWER_KIND.get(qa_type)


# Keyword signals for the analyzer-absent fallback.  Kept aligned with
# `_EXPLANATION_ROUTING_SIGNALS` at line ~60 and `_has_comparison_signal` at
# line ~467.  See F1 in the Phase-A/B/C audit plan: when the analyzer is
# shadow/failed we must still route Stage 3 enrichments (share / forecast /
# why) via `ctx.effective_answer_kind` instead of silently degrading.
_FORECAST_ROUTING_SIGNALS = (
    "forecast",
    "predict",
    "projection",
    "will be",
    "next month",
    "next year",
    "estimate",
    "პროგნოზ",
    "прогноз",
)
_SHARE_ROUTING_SIGNALS = (
    "share",
    "composition",
    "contribute",
    "contribution",
    "breakdown",
    "структур",
    "сост",
    "წილ",
)


def _resolve_effective_answer_kind(ctx) -> AnswerKind | None:
    """Resolve answer_kind for Stage 3 routing regardless of analyzer state.

    When the analyzer is authoritative, prefer its emitted `answer_kind`.
    Otherwise deterministically derive one from the raw query so Stage 3
    enrichments (share composition, forecast, driver/why, etc.) still fire on
    analyzer failure or when shadow-mode is active.  This is the single
    source of truth consumed by analyzer.py's shape-gated branches.
    """
    if ctx.has_authoritative_question_analysis:
        qa = ctx.question_analysis
        if qa is not None and qa.answer_kind is not None:
            return qa.answer_kind

    # Keyword-based fallback.  Mirrors the legacy routing fallback at
    # pipeline.py:706 so enrichment and routing stay in lockstep.
    query_lower = str(ctx.query or "").strip().lower()
    if not query_lower:
        return None

    if any(signal in query_lower for signal in _FORECAST_ROUTING_SIGNALS):
        return AnswerKind.FORECAST

    # Comparison must win over explanation for phrases like "why did X
    # compared to Y" — COMPARISON shape carries more downstream semantics.
    if _has_comparison_signal(ctx.query):
        return AnswerKind.COMPARISON

    if any(signal in query_lower for signal in _EXPLANATION_ROUTING_SIGNALS):
        return AnswerKind.EXPLANATION

    if any(signal in query_lower for signal in _SHARE_ROUTING_SIGNALS):
        # Share/composition queries render best as COMPARISON shape — they
        # need entity rows and secondary framing (contribution columns).
        return AnswerKind.COMPARISON

    # classify_query_type gives a coarse routing signal aligned with
    # deterministic fallbacks used elsewhere in the pipeline.
    try:
        from core.llm import classify_query_type
        qtype = classify_query_type(ctx.query or "")
    except Exception:
        qtype = "unknown"

    if qtype == "comparison":
        return AnswerKind.COMPARISON
    if qtype == "list":
        return AnswerKind.LIST
    if qtype == "single_value":
        return AnswerKind.SCALAR
    if qtype == "trend" or qtype == "table":
        return AnswerKind.TIMESERIES
    if qtype == "regulatory_procedure":
        return AnswerKind.KNOWLEDGE

    # Heuristic conceptual classifier already ran in prepare_context().
    if ctx.is_conceptual:
        return AnswerKind.KNOWLEDGE

    return None


def _resolve_vector_retrieval_tier(
    answer_kind: AnswerKind | None,
    render_style: RenderStyle | None,
    *,
    is_conceptual: bool = False,
) -> VectorRetrievalTier:
    """Decide how much vector-retrieval effort a query warrants.

    Policy (see Phase D spec §15):

    * KNOWLEDGE / EXPLANATION → ``FULL`` — the retrieved passages *are*
      the answer (knowledge) or the primary explanatory backing.
    * CLARIFY → ``SKIP`` — no data to ground, no knowledge to cite.
    * Data shapes (SCALAR / LIST / TIMESERIES / COMPARISON / FORECAST /
      SCENARIO) with ``DETERMINISTIC`` render → ``SKIP`` — the generic
      renderer bypasses the LLM and never consumes the vector prompt.
    * Data shapes with ``NARRATIVE`` render → ``LIGHT`` — the summarizer
      may sprinkle in background context, but ``top_k=6`` + re-rank is
      wasted work for a one-or-two-passage sprinkle.
    * No authoritative answer_kind (analyzer absent / failed): fall back
      on ``is_conceptual`` — True → ``FULL``, False → ``LIGHT`` (keep
      retrieval warm for narrative answers, but avoid the full-K cost
      when we don't even know the shape).
    * A data-shape ``answer_kind`` with ``render_style=None`` (keyword-
      derived answer_kind but analyzer non-authoritative, so no render
      hint) → ``LIGHT``.  Treated explicitly so a future edit to the
      DETERMINISTIC branch can't silently degrade this path.
    """
    if PIPELINE_MODE == "fast":
        return VectorRetrievalTier.SKIP

    if answer_kind == AnswerKind.CLARIFY:
        return VectorRetrievalTier.SKIP

    if answer_kind in (AnswerKind.KNOWLEDGE, AnswerKind.EXPLANATION):
        return VectorRetrievalTier.FULL

    if answer_kind is not None:
        # All remaining AnswerKind members are data shapes.
        if render_style == RenderStyle.DETERMINISTIC:
            return VectorRetrievalTier.SKIP
        if render_style == RenderStyle.NARRATIVE:
            return VectorRetrievalTier.LIGHT
        # render_style is None: authoritative analyzer always populates
        # render_style (pipeline.py defaults it to NARRATIVE before calling
        # this function), so a None here means the answer_kind was derived
        # from keyword fallback.  Keep retrieval warm (LIGHT) — same as
        # the NARRATIVE default, but expressed explicitly.
        return VectorRetrievalTier.LIGHT

    # --- Analyzer-absent fallback ---
    if is_conceptual:
        return VectorRetrievalTier.FULL
    return VectorRetrievalTier.LIGHT


def _cross_check_answer_kind(ctx) -> None:
    """Compare LLM-emitted answer_kind against query_type-derived value.

    If they disagree, log a warning and prefer the safer option.
    This runs as an active check even when the analyzer succeeds.
    """
    if not ctx.has_authoritative_question_analysis:
        return
    qa = ctx.question_analysis
    llm_kind = qa.answer_kind
    derived_kind = _derive_answer_kind_from_query_type(ctx)

    if llm_kind is None or derived_kind is None:
        return

    if llm_kind == derived_kind:
        return

    # Disagreement detected — prefer the safer option.
    # "Safer" means: narrative-friendly or broader shape that can still display correctly.
    if llm_kind in _SAFE_ANSWER_KINDS:
        chosen = llm_kind
    elif derived_kind in _SAFE_ANSWER_KINDS:
        chosen = derived_kind
    else:
        # Neither is in the safe set — trust the LLM (it has more context).
        chosen = llm_kind

    log.warning(
        "answer_kind cross-check disagreement: llm=%s derived=%s chosen=%s "
        "(query_type=%s, query=%.80s)",
        llm_kind.value if llm_kind else None,
        derived_kind.value if derived_kind else None,
        chosen.value,
        qa.classification.query_type.value,
        ctx.query,
    )
    # Update the analysis in-place if the chosen value differs from what the LLM emitted.
    if chosen != llm_kind:
        qa.answer_kind = chosen


def _detect_clarify_selection(query: str, conversation_history) -> str | None:
    """If the query is a single digit or 'option N' and the last assistant
    message was a clarification, return the selected option text. Otherwise None."""
    if not conversation_history:
        return None
    q = query.strip()
    m = re.match(r'^(?:option\s+)?(\d)$', q, re.IGNORECASE)
    if not m:
        return None
    option_num = int(m.group(1))

    # Find the last assistant answer
    last_answer = None
    for turn in reversed(conversation_history):
        if turn.get("answer"):
            last_answer = turn["answer"]
            break
    if not last_answer or "Please choose one of these directions:" not in last_answer:
        return None

    # Extract option text
    for line in last_answer.splitlines():
        line = line.strip()
        if line.startswith(f"{option_num}."):
            return line[len(f"{option_num}."):].strip()
    return None


def _rewrite_query_for_clarify_selection(selected: str, conversation_history) -> str:
    """Preserve the original question context when applying a clarify option."""
    if not conversation_history:
        return selected

    original_question = ""
    for turn in reversed(conversation_history):
        answer = str(turn.get("answer") or "")
        question = str(turn.get("question") or "").strip()
        if answer and "Please choose one of these directions:" in answer:
            original_question = question
            break

    if not original_question:
        return selected
    return f"{original_question}\nSelected interpretation: {selected}"


def _requested_derived_metric_names(ctx: QueryContext) -> list[str]:
    """Return active analyzer requested derived metrics in stable order."""

    if not ctx.has_authoritative_question_analysis:
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

    if ctx.clarify_selection_override:
        return False
    if not ctx.missing_evidence_for_metrics:
        return False
    if not ctx.has_authoritative_question_analysis:
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


# Inline enrichment helpers augment tool results with supporting datasets when the evidence planner is off.
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


def _has_comparison_signal(query: str) -> bool:
    query_lower = (query or "").strip().lower()
    return any(
        signal in query_lower
        for signal in (
            "compare",
            "comparison",
            "versus",
            " vs ",
            "year over year",
            "month over month",
            "difference between",
            "შედარ",
            "сравн",
        )
    )


def _has_residual_weighted_price_signal(query: str) -> bool:
    """Return True when the query asks for a residual/remaining weighted price."""
    query_lower = (query or "").strip().lower()
    if not query_lower:
        return False
    calc_hit = any(
        signal in query_lower
        for signal in ("weighted average", "average price", "weighted avg", "mean price")
    )
    scope_hit = any(
        signal in query_lower
        for signal in ("remaining", "residual", "other electricity", "excluding", "except")
    )
    explicit_residual_components = (
        "renewable ppa" in query_lower
        and "import" in query_lower
        and ("thermal generation ppa" in query_lower or "thermal ppa" in query_lower)
        and "cfd" in query_lower
    )
    balancing_hit = "balancing" in query_lower
    context_hit = any(
        signal in query_lower
        for signal in ("tariff", "tariffs", "regulated", "deregulated")
    )
    return calc_hit and balancing_hit and (
        (scope_hit and context_hit) or explicit_residual_components
    )


def _should_enrich_balancing_driver_context(
    ctx: QueryContext,
    invocation: ToolInvocation,
    is_explanation: bool,
) -> bool:
    """Return True when balancing price results need source-price context."""
    if invocation.name != "get_prices" or ctx.df.empty:
        return False

    metric = str(invocation.params.get("metric") or "").strip().lower()
    if metric != "balancing":
        return False

    if any(
        col in ctx.df.columns
        for col in (
            "price_deregulated_hydro_gel",
            "price_regulated_hpp_gel",
            "contribution_regulated_hpp_gel",
        )
    ):
        return False

    if is_explanation:
        return True

    if _has_residual_weighted_price_signal(ctx.query):
        return True

    if ctx.has_authoritative_question_analysis:
        qa_type = ctx.question_analysis.classification.query_type.value
        reqs = ctx.question_analysis.analysis_requirements
        return (
            qa_type == "comparison"
            or reqs.needs_driver_analysis
            or reqs.needs_correlation_context
        )

    return _has_comparison_signal(ctx.query)


def _merge_frame_into_context_by_date(
    ctx: QueryContext,
    secondary_df: pd.DataFrame,
    *,
    allowed_columns: set[str] | None = None,
    secondary_tool: str = "",
    secondary_role: str = "",
) -> tuple[int, list[str]]:
    """Merge selected columns from secondary_df into ctx.df using the date column."""
    if ctx.df.empty or secondary_df.empty:
        return 0, []

    date_col_primary = next(
        (c for c in ctx.df.columns if "date" in c.lower()),
        None,
    )
    date_col_secondary = next(
        (c for c in secondary_df.columns if "date" in c.lower()),
        None,
    )
    if not date_col_primary or not date_col_secondary:
        return 0, []

    # Only merge genuinely new columns so the primary result shape stays stable.
    candidate_cols = []
    for col in secondary_df.columns:
        if col == date_col_secondary:
            continue
        if col in ctx.df.columns:
            continue
        if allowed_columns is not None and col not in allowed_columns:
            continue
        candidate_cols.append(col)

    if not candidate_cols:
        return 0, []

    primary_df = ctx.df.copy()
    merge_df = secondary_df.copy()
    primary_df[date_col_primary] = pd.to_datetime(
        primary_df[date_col_primary], errors="coerce",
    )
    merge_df[date_col_secondary] = pd.to_datetime(
        merge_df[date_col_secondary], errors="coerce",
    )

    merged = primary_df.merge(
        merge_df[[date_col_secondary] + candidate_cols].rename(
            columns={date_col_secondary: date_col_primary},
        ),
        on=date_col_primary,
        how="left",
    )
    ctx.df = merged
    ctx.cols = list(merged.columns)
    ctx.rows = [tuple(r) for r in merged.itertuples(index=False, name=None)]
    # Record the join metadata so Stage 4 can ground merged-evidence explanations.
    ctx.join_provenance.append(
        {
            "primary_tool": ctx.tool_name or "",
            "secondary_tool": secondary_tool,
            "role": secondary_role,
            "join_type": "left",
            "join_keys": [date_col_primary],
            "primary_rows": len(primary_df),
            "secondary_rows": len(merge_df),
            "merged_rows": len(merged),
            "columns_added": list(candidate_cols),
        }
    )
    return len(candidate_cols), candidate_cols


def _enrich_prices_with_balancing_driver_context(
    ctx: QueryContext,
    invocation: ToolInvocation,
    is_explanation: bool,
) -> QueryContext:
    """Attach source-price and contribution context for balancing price analysis.

    Falls back to composition-only enrichment when the richer contribution panel
    is unavailable. This keeps existing explanation behavior safe while giving
    balancing price comparisons and driver analyses better evidence.
    """
    should_add_driver_context = _should_enrich_balancing_driver_context(
        ctx, invocation, is_explanation,
    )
    if not should_add_driver_context:
        return _enrich_prices_with_composition(ctx, invocation, is_explanation)

    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            driver_df = compute_entity_price_contributions(
                conn,
                start_date=invocation.params.get("start_date"),
                end_date=invocation.params.get("end_date"),
            )
        cols_added, added_columns = _merge_frame_into_context_by_date(
            ctx,
            driver_df,
            secondary_tool="compute_entity_price_contributions",
            secondary_role="balancing_driver_context",
        )
        if cols_added > 0:
            ctx.evidence_collected["balancing_driver_context"] = {
                "tool": "compute_entity_price_contributions",
                "params": {
                    "start_date": invocation.params.get("start_date"),
                    "end_date": invocation.params.get("end_date"),
                    "currency": invocation.params.get("currency"),
                },
                "df": driver_df,
                "cols": list(driver_df.columns),
                "rows": [tuple(r) for r in driver_df.itertuples(index=False, name=None)],
            }
            log.info(
                "Enriched balancing price data with %d driver-context columns",
                cols_added,
            )
            trace_detail(
                log, ctx, "balancing_driver_enrichment", "result",
                attempted=True,
                success=True,
                columns_added=cols_added,
                column_names=added_columns,
                start_date=invocation.params.get("start_date"),
                end_date=invocation.params.get("end_date"),
            )
            return ctx

        trace_detail(
            log, ctx, "balancing_driver_enrichment", "result",
            attempted=True,
            success=False,
            columns_added=0,
            reason="no_new_columns",
        )
    except Exception as enrich_err:
        log.warning(
            "Balancing driver enrichment failed: %s",
            enrich_err,
        )
        trace_detail(
            log, ctx, "balancing_driver_enrichment", "result",
            attempted=True,
            success=False,
            columns_added=0,
            error=str(enrich_err),
        )

    fallback_is_explanation = is_explanation or _has_comparison_signal(ctx.query)
    return _enrich_prices_with_composition(
        ctx,
        invocation,
        fallback_is_explanation,
    )


def _should_route_tool_as_explanation(ctx: QueryContext) -> bool:
    """Return True when tool routing should use explanation-style handling.

    When the analyzer is authoritative, answer_kind is the single signal.
    Keyword detection is a legacy fallback that only fires when the analyzer
    is absent (shadow-mode, disabled, or failed).
    """
    if ctx.has_authoritative_question_analysis:
        qa = ctx.question_analysis
        # answer_kind is the authoritative signal when present.
        if qa.answer_kind is not None:
            return qa.answer_kind == AnswerKind.EXPLANATION
        # answer_kind is None but analyzer is authoritative — derive from
        # query_type as a safe fallback within the contract.
        qa_type = qa.classification.query_type.value
        return qa_type in ("conceptual_definition", "data_explanation")

    # Legacy keyword fallback: only when analyzer is absent.
    query_lower = (ctx.query or "").strip().lower()
    return any(signal in query_lower for signal in _EXPLANATION_ROUTING_SIGNALS)


def _derive_resolved_query(ctx: QueryContext) -> tuple[str, str]:
    """Return the downstream resolved query and its authority source.

    Only active analyzer output may influence runtime behavior. Shadow-mode
    canonicalization remains useful for observability but must not change
    routing or fallback prompts.
    """
    if ctx.has_authoritative_question_analysis and ctx.question_analysis.canonical_query_en.strip():
        return ctx.question_analysis.canonical_query_en, "llm_active_canonical"
    return ctx.query, "raw_query"


# Shared attachment logic keeps router, analyzer, and recovery tool paths consistent.
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
    df = normalize_tool_dataframe(invocation.name, df)
    ctx.df = df
    ctx.cols = list(df.columns)
    ctx.rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
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
    if not ctx.has_authoritative_question_analysis:
        ctx.plan.setdefault("intent", "tool_query")
        ctx.plan.setdefault("target", invocation.name)

    tool_relevant, tool_reason = validate_tool_relevance(
        (relevance_query or ctx.query),
        invocation.name,
        question_analysis=ctx.question_analysis if ctx.has_authoritative_question_analysis else None,
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

    # --- Phase 2: Build canonical evidence frame alongside raw df ---
    _build_and_attach_evidence_frame(ctx, invocation)

    if ENABLE_EVIDENCE_PLANNER:
        # Composition enrichment is handled by the evidence loop
        # (Stage 0.8 via COMPOSITION_CONTEXT role); skip inline enrichment.
        return ctx
    return _enrich_prices_with_balancing_driver_context(
        ctx, invocation, is_explanation,
    )


def _build_and_attach_evidence_frame(ctx: QueryContext, invocation: ToolInvocation) -> None:
    """Build a canonical evidence frame from the tool result and attach to ctx.

    The frame is stored on ctx.evidence_frame for use by the generic renderer
    in Stage 4.  This is additive — the raw df/cols/rows remain untouched for
    backward compatibility with Stage 3 enrichment and existing summarizer paths.
    """
    if ctx.df is None or ctx.df.empty:
        return

    answer_kind = None
    filter_cond = None
    if ctx.has_authoritative_question_analysis:
        qa = ctx.question_analysis
        answer_kind = qa.answer_kind
        # Extract filter from the matching tool candidate's params_hint
        for tc in qa.tooling.candidate_tools:
            if tc.name.value == invocation.name and tc.params_hint is not None:
                filter_cond = tc.params_hint.filter
                break

    prov_refs = []
    if hasattr(ctx, "provenance") and ctx.provenance:
        prov_refs = [p.get("query_hash", "") for p in ctx.provenance if isinstance(p, dict)]

    frame = adapt_tool_result(
        tool_name=invocation.name,
        df=ctx.df,
        provenance_refs=prov_refs,
        filter_cond=filter_cond,
        answer_kind=answer_kind,
    )
    if frame is not None:
        ctx.evidence_frame = frame
        log.info(
            "Built canonical evidence frame: type=%s rows=%d (tool=%s)",
            type(frame).__name__,
            len(frame.rows),
            invocation.name,
        )

        # Phase 3: validate evidence against answer_kind requirements.
        gap = validate_evidence(frame, answer_kind)
        if gap is not None:
            if gap.correctable:
                log.warning(
                    "Evidence gap (correctable): %s — downstream may re-plan or degrade",
                    gap,
                )
                ctx.evidence_gap = gap
                trace_detail(log, ctx, "evidence", "evidence_gap_correctable",
                             answer_kind=str(gap.answer_kind), reason=gap.reason)
            else:
                log.warning(
                    "Evidence gap (not correctable): %s — degrading render_style to narrative",
                    gap,
                )
                ctx.evidence_gap = gap
                trace_detail(log, ctx, "evidence", "evidence_gap_not_correctable",
                             answer_kind=str(gap.answer_kind), reason=gap.reason)
                # Degrade: let LLM narrative handle the mismatch.
                if ctx.has_authoritative_question_analysis:
                    ctx.question_analysis.render_style = RenderStyle.NARRATIVE


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

    # Try a tiny set of safe alternatives instead of reopening the full search space.
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
    # Detect clarification-selection replies (e.g. "1", "option 2")
    selected = _detect_clarify_selection(query, conversation_history)
    if selected:
        query = _rewrite_query_for_clarify_selection(selected, conversation_history)
        log.info("Clarification selection detected; rewriting query to: %s", query)

    ctx = QueryContext(
        query=query,
        conversation_history=conversation_history,
        trace_id=trace_id,
        session_id=session_id,
        clarify_selection_override=selected is not None,
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

    # --- answer_kind cross-check (before Stage 0.3 so we can skip retrieval) ---
    _cross_check_answer_kind(ctx)
    if ctx.has_authoritative_question_analysis:
        qa = ctx.question_analysis
        # Fallback: if LLM did not emit answer_kind, derive it from query_type.
        if qa.answer_kind is None:
            qa.answer_kind = _derive_answer_kind_from_query_type(ctx)
        # Override: scenario-family derived metrics signal SCENARIO when the
        # LLM misclassified a scenario query as data_explanation or similar.
        # Strong structural answer_kinds (COMPARISON, LIST, KNOWLEDGE, CLARIFY)
        # are never overridden — they represent a deliberate shape choice.
        if qa.answer_kind in _SCENARIO_OVERRIDE_ELIGIBLE:
            derived = qa.analysis_requirements.derived_metrics or []
            if any(m.metric_name in _SCENARIO_DERIVED_METRICS for m in derived):
                log.info(
                    "answer_kind override: %s → SCENARIO (scenario derived metrics present)",
                    qa.answer_kind.value if qa.answer_kind else None,
                )
                qa.answer_kind = AnswerKind.SCENARIO
        # Fallback: if LLM did not emit render_style, default to narrative (safer).
        if qa.render_style is None:
            qa.render_style = RenderStyle.NARRATIVE
        log.info(
            "answer_kind=%s render_style=%s grouping=%s entity_scope=%s (query=%.80s)",
            qa.answer_kind.value if qa.answer_kind else None,
            qa.render_style.value if qa.render_style else None,
            qa.grouping.value if qa.grouping else None,
            qa.entity_scope,
            ctx.query,
        )

    # Populate ctx.effective_answer_kind once so Stage 3 enrichment dispatch
    # works whether the analyzer was authoritative, shadow, or failed.  This
    # restores forecast/why/share behaviors on analyzer failure (F1).
    ctx.effective_answer_kind = _resolve_effective_answer_kind(ctx)

    # Three-tier vector retrieval policy (Phase D):
    #   FULL  → knowledge / explanation answers that consume passages directly
    #   LIGHT → narrative data answers that sprinkle in background
    #   SKIP  → deterministic data paths, clarify, etc.
    _qa_for_tier = ctx.question_analysis if ctx.has_authoritative_question_analysis else None
    _retrieval_tier = _resolve_vector_retrieval_tier(
        answer_kind=ctx.effective_answer_kind,
        render_style=(_qa_for_tier.render_style if _qa_for_tier is not None else None),
        is_conceptual=bool(ctx.is_conceptual),
    )
    ctx.vector_retrieval_tier = _retrieval_tier
    if _retrieval_tier == VectorRetrievalTier.SKIP:
        log.info(
            "Skipping vector retrieval: tier=SKIP (answer_kind=%s render_style=%s)",
            getattr(ctx.effective_answer_kind, "value", None),
            getattr(_qa_for_tier.render_style, "value", None) if _qa_for_tier is not None else None,
        )
    elif _retrieval_tier == VectorRetrievalTier.LIGHT:
        log.info(
            "Vector retrieval: tier=LIGHT (answer_kind=%s render_style=%s)",
            getattr(ctx.effective_answer_kind, "value", None),
            getattr(_qa_for_tier.render_style, "value", None) if _qa_for_tier is not None else None,
        )

    # Stage 0.3: vector-backed knowledge retrieval (shadow/active collection only)
    if (
        (ENABLE_VECTOR_KNOWLEDGE_SHADOW or ENABLE_VECTOR_KNOWLEDGE_HINTS)
        and _retrieval_tier != VectorRetrievalTier.SKIP
    ):
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
            tier=_retrieval_tier,
        )
        ctx.vector_knowledge = bundle
        ctx.vector_knowledge_source = f"vector_{retrieval_mode}"
        ctx.vector_knowledge_error = bundle.error
        
        # Cross-notify circuit breaker on DB-layer failures from vector store.
        # Match broadly: psycopg wraps as "ConnectionTimeout", but SQLAlchemy
        # may surface "OperationalError" with varied messages like "timeout expired",
        # "connection timed out", "could not connect", etc.
        if bundle.error:
            _err_lower = str(bundle.error).lower()
            _is_db_failure = any(kw in _err_lower for kw in (
                "connectiontimeout", "operationalerror", "timeout",
                "could not connect", "connection refused", "connection reset",
            ))
            if _is_db_failure:
                from utils.resilience import db_circuit_breaker
                db_circuit_breaker.record_failure()
                log.warning(
                    "Stage 0.3 DB failure → circuit breaker notified (failures=%d/%d): %.120s",
                    db_circuit_breaker._failure_count,
                    db_circuit_breaker.failure_threshold,
                    bundle.error,
                )

        packed_vector_knowledge = (
            pack_vector_knowledge_for_prompt(bundle)
            if not bundle.error
            else None
        )
        ctx.vector_knowledge_prompt = (
            packed_vector_knowledge.prompt
            if packed_vector_knowledge is not None
            else ""
        )
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
            tier=_retrieval_tier.value,
            chunk_count=bundle.chunk_count,
            strategy=bundle.strategy.value,
            preferred_topics=bundle.filters.preferred_topics,
            top_sources=top_sources,
            top_sections=top_sections,
            packed_chunk_count=(len(packed_vector_knowledge.headers) if packed_vector_knowledge is not None else 0),
            packed_sections=(packed_vector_knowledge.headers[:3] if packed_vector_knowledge is not None else []),
            packed_truncated=(packed_vector_knowledge.truncated if packed_vector_knowledge is not None else False),
            error=bundle.error,
        )
        _trace_stage(
            "stage_0_3_vector_knowledge",
            t_stage,
            mode=retrieval_mode,
            tier=_retrieval_tier.value,
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
            if (
                ctx.has_authoritative_question_analysis
                and ctx.question_analysis.routing.preferred_path == PreferredPath.REJECT
            ):
                ctx.clarify_reason = "request_not_supported_as_phrased"
            else:
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

    # Stage 0.4: expand the authoritative analyzer output into the exact datasets we still need.
    if ENABLE_EVIDENCE_PLANNER:
        t_stage = time.time()
        ctx = evidence_planner.build_evidence_plan(ctx)
        _trace_stage(
            "stage_0_4_evidence_plan", t_stage,
            steps=len(ctx.evidence_plan),
            source=ctx.evidence_plan_source,
            tools=[s["tool_name"] for s in ctx.evidence_plan],
        )

    # Stage 0.5: prefer deterministic tool execution before falling back to heavier planners.
    # PRE-FLIGHT: check circuit breaker before attempting tools
    from utils.resilience import db_circuit_breaker
    _cb_allowed, _cb_reason = db_circuit_breaker.allow_request()
    if not _cb_allowed:
        log.warning(
            "Skipping tool execution: circuit breaker is %s (%s). "
            "Pipeline will fall through to Stage 1/2 or CLARIFY.",
            db_circuit_breaker._state, _cb_reason,
        )

    if ENABLE_TYPED_TOOLS and _cb_allowed:
        t_stage = time.time()

        is_exp = _should_route_tool_as_explanation(ctx)

        # Plan-driven invocation: when evidence plan exists, execute the
        # first unsatisfied step instead of keyword-matching. Raw-query
        # routing remains fallback-only when Stage 0.2 is absent/failed.
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
            invocation = (
                None
                if ctx.has_authoritative_question_analysis
                else match_tool(ctx.query, is_explanation=is_exp)
            )
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

                df = normalize_tool_dataframe(invocation.name, df)
                ctx.df = df
                ctx.cols = list(df.columns)
                ctx.rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
                stamp_provenance(
                    ctx,
                    ctx.cols,
                    ctx.rows,
                    source="tool",
                    query_hash=tool_invocation_hash(invocation.name, invocation.params),
                )
                if not ctx.has_authoritative_question_analysis:
                    ctx.plan.setdefault("intent", "tool_query")
                    ctx.plan.setdefault("target", invocation.name)
                _relevance_q = ctx.resolved_query or ctx.query
                tool_relevant, tool_reason = validate_tool_relevance(
                    _relevance_q,
                    invocation.name,
                    question_analysis=ctx.question_analysis if ctx.has_authoritative_question_analysis else None,
                )
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
                        ctx = _enrich_prices_with_balancing_driver_context(
                            ctx, invocation, is_exp,
                        )

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
                                "params": dict(invocation.params),
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

            # Stage 0.7: if keyword/plan routing misses, let the active analyzer nominate one tool.
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
            if ctx.has_authoritative_question_analysis and ENABLE_QUESTION_ANALYZER_HINTS:
                try:
                    analyzer_invocation = planner.build_tool_invocation_from_analysis(
                        ctx.question_analysis, ctx.query,
                    )
                except Exception as exc:
                    analyzer_build_error = str(exc)
                    ctx.tool_fallback_reason = f"analyzer_tool_build_error:{exc}"
                    log.warning("Analyzer tool invocation build failed: %s", exc)

            if (
                analyzer_invocation is None
                and _should_attempt_authoritative_router_fallback(ctx)
            ):
                analyzer_invocation = match_tool(ctx.query, is_explanation=is_exp)
                if analyzer_invocation is not None:
                    analyzer_invocation.reason = (
                        f"authoritative_data_primary_router_fallback:{analyzer_invocation.reason}"
                    )

            # Trace the analyzer routing decision regardless of outcome
            qa = ctx.question_analysis if ctx.has_authoritative_question_analysis else None
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
                    df = normalize_tool_dataframe(analyzer_invocation.name, df)
                    ctx.df = df
                    ctx.cols = list(df.columns)
                    ctx.rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
                    stamp_provenance(
                        ctx, ctx.cols, ctx.rows, source="tool",
                        query_hash=tool_invocation_hash(analyzer_invocation.name, analyzer_invocation.params),
                    )
                    if not ctx.has_authoritative_question_analysis:
                        ctx.plan.setdefault("intent", "tool_query")
                        ctx.plan.setdefault("target", analyzer_invocation.name)
                    _relevance_q = ctx.resolved_query or ctx.query
                    tool_relevant, tool_reason = validate_tool_relevance(
                        _relevance_q,
                        analyzer_invocation.name,
                        question_analysis=ctx.question_analysis if ctx.has_authoritative_question_analysis else None,
                    )
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
                            ctx = _enrich_prices_with_balancing_driver_context(
                                ctx, analyzer_invocation, is_exp,
                            )

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
                                    "params": dict(analyzer_invocation.params),
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

    # Stage 0.8: finish any remaining evidence-plan steps and merge them into the main frame.
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

    if ENABLE_EVIDENCE_PLANNER and ctx.used_tool and ctx.tool_name:
        post_plan_invocation = ToolInvocation(
            name=ctx.tool_name,
            params=dict(ctx.tool_params),
            confidence=ctx.tool_confidence,
            reason=ctx.tool_match_reason,
        )
        ctx = _enrich_prices_with_balancing_driver_context(
            ctx,
            post_plan_invocation,
            _should_route_tool_as_explanation(ctx),
        )

    # Stage 1/2: agent loop or SQL fallback only runs after deterministic tool paths are exhausted.
    # When Stage 0.2 is authoritative, it owns route selection, so the
    # context-blind agent loop must not run ahead of planner/SQL fallback.
    # Without authoritative Stage 0.2, the agent loop remains the last resort.
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
    if (
        not ctx.used_tool
        and not _evidence_plan_satisfied
        and ENABLE_AGENT_LOOP
        and not _analyzer_tool_failed
        and not ctx.has_authoritative_question_analysis
    ):
        t_stage = time.time()
        ctx = orchestrator.run_agent_loop(ctx)
        _trace_stage("stage_agent_loop", t_stage, outcome=ctx.agent_outcome, rounds=ctx.agent_rounds)
        if ctx.agent_outcome == "conceptual_exit":
            log.info("Agent conceptual exit | rounds=%s", ctx.agent_rounds)
            return ctx
        if ctx.agent_outcome == "data_exit":
            _relevance_q = ctx.resolved_query or ctx.query
            tool_relevant, tool_reason = validate_tool_relevance(
                _relevance_q,
                ctx.tool_name or "",
                question_analysis=ctx.question_analysis if ctx.has_authoritative_question_analysis else None,
            )
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

    # Final legacy fallback: generate SQL, execute it, then continue through analysis/summarization.
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

    # Snapshot the source tabular output before analyzer mutates or augments the evidence.
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
    if ctx.rows and ctx.cols and set(ctx.cols) - set(ctx.provenance_cols or []):
        inferred_source = str(ctx.provenance_source or ("tool" if ctx.used_tool else "sql"))
        inferred_hash = str(ctx.provenance_query_hash or "")
        if not inferred_hash:
            if ctx.used_tool and ctx.tool_name:
                inferred_hash = tool_invocation_hash(ctx.tool_name, ctx.tool_params)
            elif ctx.safe_sql:
                inferred_hash = sql_query_hash(ctx.safe_sql)
        stamp_provenance(
            ctx,
            ctx.cols,
            ctx.rows,
            source=inferred_source or "tool",
            query_hash=inferred_hash or sql_query_hash(f"{ctx.query}|stage3_enriched"),
        )
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
