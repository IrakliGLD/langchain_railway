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
  1/2  planner / sql_executor            -> legacy LLM plan + SQL fallback
  3    analyzer.enrich                   -> stats, correlation, trendlines
  4    summarizer.summarize_data         -> generic renderer or LLM narrative
  5    chart_pipeline.build_chart        -> chart from evidence or SQL results
"""
import json
import logging
import re
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from dataclasses import dataclass
from datetime import date
from functools import wraps

import pandas as pd
from sqlalchemy import text

from agent import analyzer, chart_pipeline, evidence_finalizer, evidence_planner, planner, sql_executor, summarizer
from agent.evidence_derivation import (
    derive_evidence,
)
from agent.evidence_derivation import (
    missing_requested_evidence as _missing_requested_evidence,
)
from agent.evidence_derivation import (
    requested_derived_metric_names as _requested_derived_metric_names,
)
from agent.fixture_candidates import log_fixture_candidate
from agent.p4_rollout import (
    GATE_EVIDENCE_FINALIZATION,
    GATE_EVIDENCE_REANALYSIS,
    GATE_HONEST_TERMINAL_OUTCOMES,
    GATE_PLAN_VALIDATION,
    build_rollout_decisions,
    gate_is_active,
)
from agent.provenance import (
    build_provenance_refs,
    clear_provenance,
    sql_query_hash,
    stamp_provenance,
    tool_invocation_hash,
)
from agent.render_fitness import df_date_span, period_bounds_from_hint
from agent.router import ROUTER_ENABLE_SEMANTIC_FALLBACK, _last_semantic_scores, match_tool
from agent.stage_orchestrator import PipelineStageOrchestrator
from agent.tools import execute_tool
from agent.tools.types import ToolInvocation
from analysis.shares import compute_entity_price_contributions
from analysis.system_quantities import normalize_tool_dataframe
from config import (
    ANALYZER_CONFIDENCE_OVERRIDE_THRESHOLD,
    DB_SECONDARY_DRAIN_TIMEOUT_MS,
    ENABLE_EVIDENCE_PLANNER,
    ENABLE_EVIDENCE_REANALYSIS,
    ENABLE_HONEST_TERMINAL_OUTCOMES,
    ENABLE_QUESTION_ANALYZER_HINTS,
    ENABLE_QUESTION_ANALYZER_SHADOW,
    ENABLE_TYPED_TOOLS,
    ENABLE_VECTOR_KNOWLEDGE_HINTS,
    ENABLE_VECTOR_KNOWLEDGE_SHADOW,
    EVIDENCE_FINALIZATION_ENFORCE_PERCENT,
    EVIDENCE_PARALLEL_SECONDARY,
    EVIDENCE_REANALYSIS_PERCENT,
    HONEST_TERMINAL_OUTCOMES_PERCENT,
    PIPELINE_MODE,
    PLAN_VALIDATION_ENFORCE_PERCENT,
    PLAN_VALIDATION_MODE,
)
from contracts.question_analysis import (
    _SCENARIO_METRIC_NAMES,
    AnswerKind,
    KnowledgeTopicName,
    PreferredPath,
    RenderStyle,
)
from contracts.vector_knowledge import VectorKnowledgeMode, VectorRetrievalTier
from core.db_gateway import database_connection
from core.db_work_coordinator import (
    DatabaseWorkCapacityExceeded,
    db_work_coordinator,
)
from core.query_executor import ENGINE
from knowledge.vector_retrieval import (
    pack_vector_knowledge_for_prompt,
    retrieve_vector_knowledge,
)
from models import QueryContext, ResolutionPolicy, ResponseMode, TerminalOutcome
from utils.metrics import metrics
from utils.privacy_logging import hash_private_identifier
from utils.query_validation import extract_query_topics, validate_tool_relevance
from utils.request_deadline import (
    RequestDeadlineExceeded,
    bind_request_execution_scope,
    bind_request_execution_scope_snapshot,
    cap_request_deadline,
    current_request_execution_scope,
)
from utils.residual_price import is_implied_ppa_cfd_price_query
from utils.trace_logging import trace_detail

log = logging.getLogger("Enai")


def _remaining_request_seconds(ctx: QueryContext, stage: str) -> float | None:
    deadline = getattr(ctx, "request_deadline", None)
    if deadline is None:
        return None
    deadline.ensure_remaining(stage)
    return deadline.remaining_seconds()


def _require_request_budget(ctx: QueryContext, stage: str) -> None:
    _remaining_request_seconds(ctx, stage)


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

# --- Scenario quantitative-anchor gate ---
# Production logs (2026-05-13) showed a query "if more ppa will be signed, how
# this will affect the price?" routed to SCENARIO because the analyzer emitted
# a fabricated scenario_factor=1.34 from a query containing no number.  The
# generic renderer then produced a meaningless "Result: 24511.48 (× 1.34)"
# from summing 132 historical balancing prices and scaling.
#
# The override now requires the user query to contain an explicit quantitative
# anchor — a digit, a percentage, or a multiplicative/directional word — before
# upgrading EXPLANATION/SCALAR/TIMESERIES → SCENARIO.  When the anchor is
# absent, the request stays on the narrative path (LLM summarizer with full
# evidence frames), which is the safer fallback.
#
# Conservative by design: any digit anywhere in the query counts, even dates,
# to avoid false-positives suppressing legitimate scenario queries.  Multilingual
# support is digit-based today; non-Latin multiplicative vocabulary can be
# added when a real failure case emerges.
_QUANTITATIVE_ANCHOR_RE = re.compile(
    r"\b("
    r"\d+(?:[.,]\d+)?\s*(?:%|percent|gel|usd|eur|mwh|gwh|mw|gw|kwh)?"
    r"|double|doubles|doubling"
    r"|halve|halves|halving|half"
    r"|triple|triples|tripling"
    r"|quadruple|twice"
    r"|(?:increases?|increasing|decreases?|decreasing"
    r"|raises?|raising|cuts?|cutting"
    r"|grows?|growing|shrinks?|shrinking"
    r"|rises?|rising|falls?|falling"
    r"|drops?|dropping)\s+by\b"
    r")\b",
    re.IGNORECASE,
)


def _query_has_quantitative_anchor(query: str) -> bool:
    """Return True iff *query* contains a numeric or multiplicative anchor.

    Used as the SCENARIO override gate at the answer_kind derivation site.
    See ``_QUANTITATIVE_ANCHOR_RE`` for the failure case that motivated it.
    """
    return bool(_QUANTITATIVE_ANCHOR_RE.search(query or ""))


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

_TARGET_MODEL_KNOWLEDGE_TOKENS = (
    "target model",
    "target market",
    "market reform",
    "market model",
    "self-dispatch",
    "self dispatch",
    "hourly imbalance",
    "brp",
    "balance responsible",
)


def _is_target_model_knowledge_query(query_text: str) -> bool:
    query_lower = str(query_text or "").lower()
    return bool(query_lower) and any(
        token in query_lower for token in _TARGET_MODEL_KNOWLEDGE_TOKENS
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
            qa_type in _ALWAYS_DATA_TYPES
            and ctx.is_conceptual
            and "market_structure" in extract_query_topics(query_text)
        ):
            # Deterministic policy-intent guard: retrieval availability must not
            # decide whether a liberalization/status question becomes a tariff
            # data lookup when the raw query is conceptual.
            return ResponseMode.KNOWLEDGE_PRIMARY
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
            # Disagreement-rescue mirror to _resolve_vector_retrieval_tier:
            # when the analyzer chose a data shape but the heuristic flagged
            # the query as conceptual AND vector retrieval actually pulled
            # regulation chunks, route to KNOWLEDGE_PRIMARY so STRICT_NUMERIC
            # grounding doesn't reject the regulation-grounded answer
            # downstream.  Trace: 2026-05-14 incident on
            # "what is a price of electricity esco paying to sellers of
            # balancing electricity?" — analyzer chose factual_lookup →
            # DATA_PRIMARY → STRICT_NUMERIC grounding rejected the prose
            # answer grounded in transitory_market_rules.md Article 14
            # (claims_count=0, conservative fallback fired).
            if (
                ctx.is_conceptual
                and (
                    getattr(ctx.vector_knowledge, "chunk_count", 0) > 0
                    or (
                        qa_type == "factual_lookup"
                        and bool(getattr(ctx, "vector_knowledge_error", ""))
                    )
                )
            ):
                return ResponseMode.KNOWLEDGE_PRIMARY
            return ResponseMode.DATA_PRIMARY
        # Ambiguous types: comparison, forecast, ambiguous, unsupported
        if qa_path == "knowledge":
            return ResponseMode.KNOWLEDGE_PRIMARY
        return ResponseMode.DATA_PRIMARY
    query_text = " ".join(
        part for part in (str(ctx.query or ""), str(ctx.effective_query or "")) if part
    )
    if _is_target_model_knowledge_query(query_text):
        return ResponseMode.KNOWLEDGE_PRIMARY
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

# Query types where the source is typically a legal/regulatory text and an
# answer_kind=LIST should be trusted over the deterministic KNOWLEDGE fallback.
# Narrative summarisation of an enumerated legal source (eligible parties,
# requirements, documents) tends to drop or merge items; LIST forces complete
# enumeration. Gated on high confidence to avoid clobbering low-quality LLM
# guesses.
_LEGAL_LIST_QUERY_TYPES = frozenset({"regulatory_procedure", "conceptual_definition"})
_LEGAL_LIST_MIN_CONFIDENCE = 0.85


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


# Topics where the answer depends on market-structure / settlement-path
# semantics that the LLM cannot derive from raw column values alone (e.g.
# the ESCO buy-vs-sell asymmetry, the 14 named regulated plants, the
# three settlement paths: regulated tariff / p_dereg / confidential PPA).
# When the analyzer flags any of these as a candidate topic, we keep
# vector retrieval warm at LIGHT even for ``DETERMINISTIC`` data shapes
# so ``balancing_price.md`` / ``tariffs.md`` / ``market_structure.md``
# reach Stage 4.
#
# Q2 production trace c995f0c7 (2026-05-17): analyzer emitted
# candidate_topics=["balancing_price", "generation_mix", "market_structure"]
# but render_style=DETERMINISTIC → tier=SKIP → no knowledge in prompt →
# LLM applied the wrong column mapping (Fix C's baked-in mapping was
# incorrect; the right mapping lives in balancing_price.md but never
# reached the LLM).
_MARKET_STRUCTURE_TOPICS = frozenset({
    KnowledgeTopicName.BALANCING_PRICE,
    KnowledgeTopicName.TARIFFS,
    KnowledgeTopicName.MARKET_STRUCTURE,
    KnowledgeTopicName.DIRECT_CONTRACTS,
    KnowledgeTopicName.CROSS_BORDER_TRADE,
    KnowledgeTopicName.EXCHANGE_TRANSITION,
    KnowledgeTopicName.CFD_PPA,
    KnowledgeTopicName.PSO_TRADING,
})


def _resolve_vector_retrieval_tier(
    answer_kind: AnswerKind | None,
    render_style: RenderStyle | None,
    *,
    is_conceptual: bool = False,
    topics=None,
) -> VectorRetrievalTier:
    """Decide how much vector-retrieval effort a query warrants.

    Policy (see Phase D spec §15):

    * KNOWLEDGE / EXPLANATION → ``FULL`` — the retrieved passages *are*
      the answer (knowledge) or the primary explanatory backing.
    * CLARIFY → ``SKIP`` — no data to ground, no knowledge to cite.
    * Data shapes (SCALAR / LIST / TIMESERIES / COMPARISON / FORECAST /
      SCENARIO) with ``DETERMINISTIC`` render → ``SKIP`` — the generic
      renderer bypasses the LLM and never consumes the vector prompt.
      Two rescues bump this back to ``LIGHT``:
        - ``is_conceptual=True``: heuristic / analyzer disagreement,
          keep regulation context warm (Trace 2026-05-14 SCALAR rescue).
        - ``topics`` contains any of ``_MARKET_STRUCTURE_TOPICS``: the
          analyzer flagged the query as touching balancing-price /
          tariffs / market-structure / CFD-PPA / PSO-trading semantics.
          The LLM needs ``balancing_price.md`` / ``tariffs.md`` content
          to map vernacular labels onto the right columns and to
          respect the ESCO buy-vs-sell asymmetry. Trace 2026-05-17
          c995f0c7 — ``answer_kind=comparison render_style=deterministic``
          ran with ``domain_knowledge_in_prompt=0 chars`` and the LLM
          applied the wrong column mapping for "small hydro sellers".
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

    topic_set = (
        set(topics) if topics is not None else set()
    )
    needs_market_structure_knowledge = bool(
        topic_set & _MARKET_STRUCTURE_TOPICS
    )

    if answer_kind is not None:
        # All remaining AnswerKind members are data shapes.
        if render_style == RenderStyle.DETERMINISTIC:
            # Disagreement rescue: if the heuristic flagged the query as
            # conceptual but the analyzer landed on a deterministic data
            # shape, keep retrieval warm at LIGHT.  The analyzer still wins
            # on answer shape (we render deterministically), but a few
            # regulation chunks reach the summarizer so it can ground the
            # answer in the law when the question is genuinely ambiguous
            # between "what's the current value" and "how is this defined".
            # Trace: 2026-05-14 incident on
            # "what is a price of electricity esco paying to sellers of
            # balancing electricity?" — SCALAR+DETERMINISTIC bypassed
            # transitory_market_rules.md Article 14 entirely.
            if is_conceptual:
                return VectorRetrievalTier.LIGHT
            # Market-structure rescue (2026-05-18, Q2 trace c995f0c7):
            # see _MARKET_STRUCTURE_TOPICS comment above.
            if needs_market_structure_knowledge:
                return VectorRetrievalTier.LIGHT
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

    # Legal-list exception: when the LLM emits LIST with high confidence for a
    # regulatory or conceptual question, the source is almost always a legal
    # text that enumerates items (eligible parties, requirements, documents).
    # Forcing such answers into KNOWLEDGE narrative loses items to paraphrase.
    # See skills/pipeline-failure-diagnostics/references/failure-taxonomy.md
    # Pattern B for the post-mortem.
    if (
        llm_kind == AnswerKind.LIST
        and qa.classification.query_type.value in _LEGAL_LIST_QUERY_TYPES
        and qa.classification.confidence >= _LEGAL_LIST_MIN_CONFIDENCE
    ):
        metrics.log_analyzer_cross_check("legal_list_exception")
        log.info(
            "answer_kind cross-check: trusting LLM list shape for legal/conceptual question "
            "(llm=list derived=%s confidence=%.2f query_type=%s, query=%.80s)",
            derived_kind.value,
            qa.classification.confidence,
            qa.classification.query_type.value,
            ctx.query,
        )
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

    metrics.log_analyzer_cross_check("disagreement")
    log_fixture_candidate("cross_check_disagreement", ctx)
    log.warning(
        "answer_kind cross-check disagreement: llm=%s derived=%s chosen=%s "
        "(query_type=%s, query=%.80s)",
        llm_kind.value if llm_kind else None,
        derived_kind.value if derived_kind else None,
        chosen.value,
        qa.classification.query_type.value,
        ctx.query,
    )
    trace_detail(
        log,
        ctx,
        "stage_0_2_question_analyzer",
        "answer_kind_cross_check",
        llm_kind=llm_kind.value if llm_kind else None,
        derived_kind=derived_kind.value if derived_kind else None,
        chosen=chosen.value,
        query_type=qa.classification.query_type.value,
    )
    # Update the analysis in-place if the chosen value differs from what the LLM emitted.
    if chosen != llm_kind:
        metrics.log_analyzer_cross_check("override_applied")
        qa.answer_kind = chosen


# Production trace 190c2893 (2026-05-13) showed a user reply of literally "1."
# fall through to a full 25-second pipeline ending in a grounding-failure
# fallback.  The previous regex ^(?:option\s+)?(\d)$ rejected trailing
# punctuation, multi-digit options ("10"), and the marker check only
# accepted one specific clarify phrasing.  Both are loosened below.
_CLARIFY_OPTION_QUERY_RE = re.compile(
    r"^\s*(?:option\s+)?(\d+)\s*[.)]?\s*$",
    re.IGNORECASE,
)
# Matches lines like "1. text", "1) text", "**1.** text", "  1) **text**"
# inside an assistant turn that already passed the clarify-marker check.
_CLARIFY_OPTION_LINE_TEMPLATE = (
    r"^\s*\**\s*{num}\s*[.)]\s*\**\s*(.+?)\s*\**\s*$"
)


def _detect_clarify_selection(query: str, conversation_history) -> str | None:
    """Detect a numeric reply ("1", "1.", "1)", "option 2", "10.") to a
    prior clarification turn and return the selected option text.

    Returns None if the query is not a numeric option, the history is
    empty, the last assistant turn shows no clarify marker, or the
    matching option line cannot be located.
    """
    if not conversation_history:
        return None
    m = _CLARIFY_OPTION_QUERY_RE.match(query or "")
    if not m:
        return None
    option_num = int(m.group(1))

    # Find the last assistant answer.
    last_answer = None
    for turn in reversed(conversation_history):
        if turn.get("answer"):
            last_answer = turn["answer"]
            break
    if not last_answer:
        return None

    # Reuse the shared clarify-marker list from core.llm so this gate stays
    # in sync with the other "is this a clarify turn?" check sites.  Lazy
    # import to avoid a circular dependency at module load.
    from core.llm import _CLARIFY_ASSISTANT_MARKERS

    last_lower = last_answer.lower()
    if not any(marker in last_lower for marker in _CLARIFY_ASSISTANT_MARKERS):
        return None

    # Extract option text — tolerant of "N.", "N)", and surrounding bold/whitespace.
    option_line_re = re.compile(
        _CLARIFY_OPTION_LINE_TEMPLATE.format(num=option_num)
    )
    for line in last_answer.splitlines():
        m_line = option_line_re.match(line)
        if m_line:
            return m_line.group(1).strip()
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
    if is_implied_ppa_cfd_price_query(query_lower):
        return True
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
            "price_deregulated_ren_gel",
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
    # Driver-enrichment frames come directly from SQL (e.g.
    # ``compute_entity_price_contributions``) where PostgreSQL ``numeric``
    # columns arrive as ``Decimal`` with ``object`` dtype. Coerce here so
    # downstream ``select_dtypes(include="number")`` consumers (per-column
    # aggregates, grounding-token extraction, chart builders) see proper
    # float64 dtypes. See ``coerce_decimal_columns_to_float`` for the
    # full root-cause explanation.
    from analysis.system_quantities import coerce_decimal_columns_to_float
    merged, _ = coerce_decimal_columns_to_float(merged)
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
        with database_connection(ENGINE, operation="evidence_enrichment", read_only=True) as conn:
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
    df = normalize_tool_dataframe(invocation.name, df, params=invocation.params)
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
        (relevance_query or ctx.resolved_query or ctx.query),
        invocation.name,
        question_analysis=ctx.question_analysis if ctx.has_authoritative_question_analysis else None,
        raw_query=ctx.query,
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
    """Attach a canonical evidence frame from a recovered tool result.

    Legacy call site (analyzer tool recovery). Since P4.1 this delegates to the
    single ``evidence_finalizer.finalize_evidence`` routine with
    ``legacy_attach=True`` so it attaches in every finalization mode — the
    recovery path built frames in production before P4.1, and ``off``/``shadow``
    must remain byte-identical to that deployed behavior. Normal execution now
    finalizes through the same routine (see ``_execute_evidence_plan``).
    """
    evidence_finalizer.finalize_evidence(
        ctx,
        stage="recovery_apply_tool_result",
        tool_name=invocation.name,
        tool_params=dict(invocation.params),
        legacy_attach=True,
    )


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


def _pick_primary_invocation(
    ctx: QueryContext, is_explanation: bool,
) -> tuple[ToolInvocation | None, dict | None, str, str]:
    """Choose the primary tool invocation from a fixed strategy chain.

    Tries four strategies in order; the first that yields a non-None
    invocation wins. This consolidates what was previously split across
    Stages 0.5 and 0.7. All per-strategy observability fires from the
    orchestration code in process_query() based on the returned `source`
    label so dashboards see the same trace and metric shapes as before.

    Strategies (in order):

    1. ``plan_driven`` - if an evidence plan exists, use the next
       unsatisfied step's tool + params.
    2. ``keyword_router`` - legacy raw-query keyword router. Only
       attempted when no authoritative analyzer is available.
    3. ``analyzer_built`` - ``planner.build_tool_invocation_from_analysis``
       on the authoritative QuestionAnalysis. Only attempted when
       Stages 1+2 are unsatisfied (i.e. no earlier strategy already
       produced an invocation).
    4. ``router_fallback`` - authoritative router fallback via
       ``match_tool``, gated on ``_should_attempt_authoritative_router_fallback``.

    Returns:
        invocation: The chosen ToolInvocation, or None if all strategies
            missed.
        plan_step: The evidence-plan step the invocation satisfies, when
            sourced from the plan. None for non-plan strategies; the
            executor falls back to name-match against the plan.
        source: One of "plan_driven", "keyword_router", "analyzer_built",
            "router_fallback", or "" when nothing matched. Drives
            stage-specific bookkeeping in the orchestration block.
        build_error: Non-empty string when `analyzer_built` raised. Stays
            empty for other sources. Carried back so the orchestration
            can emit `analyzer_tool_build_error` traces and fallback
            intents.
    """
    # Strategy 1: plan-driven (from the evidence plan)
    if ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan:
        plan_step = evidence_planner.next_unsatisfied_step(ctx.evidence_plan)
        if plan_step:
            invocation = ToolInvocation(
                name=plan_step["tool_name"],
                params=plan_step["params"],
                confidence=0.85,
                reason=f"evidence_plan:{plan_step['role']}",
            )
            return invocation, plan_step, "plan_driven", ""

    # Strategy 2: legacy keyword router (only when no authoritative analyzer)
    if not (ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan):
        if not ctx.has_authoritative_question_analysis:
            invocation = match_tool(ctx.query, is_explanation=is_explanation)
            if invocation:
                return invocation, None, "keyword_router", ""

    # Strategy 3: analyzer-built invocation
    build_error = ""
    if ctx.has_authoritative_question_analysis and ENABLE_QUESTION_ANALYZER_HINTS:
        try:
            invocation = planner.build_tool_invocation_from_analysis(
                ctx.question_analysis, ctx.query,
            )
            if invocation:
                return invocation, None, "analyzer_built", ""
        except Exception as exc:
            build_error = str(exc)
            log.warning("Analyzer tool invocation build failed: %s", exc)

    # Strategy 4: authoritative router fallback
    if _should_attempt_authoritative_router_fallback(ctx):
        invocation = match_tool(ctx.query, is_explanation=is_explanation)
        if invocation is not None:
            invocation.reason = (
                f"authoritative_data_primary_router_fallback:{invocation.reason}"
            )
            return invocation, None, "router_fallback", build_error

    return None, None, "", build_error


def _emit_trace_stage(ctx: QueryContext, stage_name: str, started_at: float, **extra) -> None:
    """Module-level stage-trace emitter.

    Encapsulates the trace emission logic used by `_trace_stage` (the local
    closure inside `process_query`) and `_execute_evidence_step` (the
    extracted helper at module scope). The closure version remains so that
    every existing call site inside `process_query` keeps its current
    signature; this function is what both wrappers ultimately call.
    """
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


def _execute_evidence_step(
    ctx: QueryContext,
    invocation: ToolInvocation,
    *,
    plan_step: dict | None,
    is_primary: bool,
    is_explanation: bool,
    stage_label: str,
    tool_execute_trace: str = "stage_0_6_tool_execute",
    relevance_log_prefix: str = "Typed",
    relevance_fallback_reason_prefix: str = "tool",
    emit_fallback_intent_on_relevance_block: bool = True,
    validate_relevance: bool = True,
    emit_tool_call_metric: bool = True,
    emit_tool_execute_trace: bool = True,
    executor: callable = None,
) -> bool:
    """Execute one evidence step end-to-end and store the result.

    Common implementation across Stage 0.5 / 0.7 / 0.8 (see Phase F.5 in
    docs/active/query_pipeline_architecture.md). Caller wraps in try/except
    for stage-specific failure recovery; this helper handles the happy path
    plus relevance-gate cleanup. Tool-execution exceptions bubble out.

    Behaviour by mode:

    ``is_primary=True`` (Stage 0.5/0.7 primary execution):
      - Stamps ctx.df / cols / rows / provenance from the tool result.
      - Optionally seeds ctx.plan defaults when the analyzer is absent.
      - On relevance block: clears ctx.df / cols / rows + provenance,
        sets ``ctx.used_tool=False`` with a fallback reason, and returns
        ``False`` so the caller can fall through to the next strategy.

    ``is_primary=False`` (Stage 0.8 secondary loop):
      - Does not touch ctx.df / cols / rows.
      - Only stores the fetched DataFrame in ``ctx.evidence_collected``.

    For both modes, when a plan_step is supplied (or, for primary mode, when
    the result name-matches an unsatisfied plan step), the result is recorded
    in ``ctx.evidence_collected`` and the step is marked satisfied.

    Args:
        ctx: The query context being mutated.
        invocation: The ToolInvocation to execute.
        plan_step: The evidence plan step being satisfied. Required for
            secondary executions; optional for primary (will fall back to
            name-match on the plan).
        is_primary: Whether to set ctx.df / tool_* / provenance from the
            result. The first execution in a request is primary; secondary
            evidence-loop steps are not.
        is_explanation: Used for the legacy driver-context enrichment that
            still runs when ``ENABLE_EVIDENCE_PLANNER`` is False.
        stage_label: Tag included in fallback reasons and log lines (e.g.
            "stage_0_5") so events from different call sites stay
            distinguishable in traces.
        tool_execute_trace: Name of the `_trace_stage` event emitted right
            after the tool runs. Defaults to "stage_0_6_tool_execute"
            (Stage 0.5's name); Stage 0.7 passes
            "stage_0_7_analyzer_tool_execute" to preserve its trace shape.
        relevance_log_prefix: Word prefixing relevance-validation log lines
            ("Typed tool relevance validated/blocked" by default, vs
            "Analyzer tool relevance ..." for Stage 0.7).
        relevance_fallback_reason_prefix: Prefix for the relevance-block
            fallback reason string. "tool" → "tool_relevance_blocked" by
            default; Stage 0.7 passes "analyzer_tool" →
            "analyzer_tool_relevance_blocked".
        emit_fallback_intent_on_relevance_block: Whether to call
            `metrics.log_tool_fallback_intent` when relevance is blocked.
            True for Stage 0.5 (current behaviour); False for Stage 0.7
            (which historically did not emit this counter on its own
            relevance block).
        validate_relevance: Whether to run `validate_tool_relevance` and
            its short-circuit on block. True for primary executions
            (Stage 0.5 / 0.7); False for Stage 0.8 secondary loop, which
            historically trusted the evidence planner's tool choices
            without re-validating relevance per step.
        emit_tool_call_metric: Whether to call `metrics.log_tool_call`
            with the tool execution duration. True for primary
            executions (Stage 0.5 / 0.7); False for Stage 0.8 to avoid
            double-counting secondary loop tool calls in current
            dashboards.
        emit_tool_execute_trace: Whether to emit a per-step
            `_trace_stage(tool_execute_trace, ...)` event. True for
            primary executions; False for Stage 0.8 which currently
            only emits the outer `stage_0_8_evidence_loop` aggregate
            trace.

    Returns:
        True  - the tool ran, relevance passed, the result was stored.
        False - relevance was blocked. Caller should treat the step as
                unsatisfied and try the next strategy.

    Raises:
        Any exception from ``execute_tool``. The caller catches and decides
        whether to fall through to SQL or attempt a recovery candidate.
    """
    t_tool = time.time()
    # `executor` lets a caller substitute its own binding of `execute_tool` so
    # tests that patch `<caller_module>.execute_tool` still take effect; this
    # is how `evidence_planner.execute_remaining_evidence` preserves
    # backward-compatibility with its monkeypatched test suite.
    _execute = executor or execute_tool
    df, cols, rows = _execute(invocation)
    _tool_duration = time.time() - t_tool
    if emit_tool_call_metric:
        metrics.log_tool_call(_tool_duration)
    # P5.5 (finding L1): per-source observability. Secondary Stage 0.8 calls
    # keep the legacy total untouched (emit_tool_call_metric=False) but are no
    # longer invisible — calls and latency are counted per tool under a
    # primary/secondary dimension.
    metrics.log_tool_call_source(
        invocation.name,
        "primary" if is_primary else "secondary",
        _tool_duration,
    )
    if emit_tool_execute_trace:
        _emit_trace_stage(ctx, tool_execute_trace, t_tool, tool=invocation.name, rows=len(rows))

    df = normalize_tool_dataframe(invocation.name, df, params=invocation.params)

    if is_primary:
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

    if validate_relevance:
        _relevance_q = ctx.resolved_query or ctx.query
        tool_relevant, tool_reason = validate_tool_relevance(
            _relevance_q,
            invocation.name,
            question_analysis=ctx.question_analysis if ctx.has_authoritative_question_analysis else None,
            raw_query=ctx.query,
        )

        if not tool_relevant:
            metrics.log_relevance_block()
            _block_reason_tag = f"{relevance_fallback_reason_prefix}_relevance_blocked"
            if is_primary:
                ctx.used_tool = False
                ctx.tool_fallback_reason = f"{_block_reason_tag}:{tool_reason}"
                if emit_fallback_intent_on_relevance_block:
                    metrics.log_tool_fallback_intent(ctx.query, _block_reason_tag)
                ctx.df = ctx.df.iloc[0:0]
                ctx.cols = []
                ctx.rows = []
                clear_provenance(ctx)
            log.warning("%s tool relevance blocked. reason=%s", relevance_log_prefix, tool_reason)
            return False

        log.info("%s tool relevance validated. reason=%s", relevance_log_prefix, tool_reason)

    if is_primary and not ENABLE_EVIDENCE_PLANNER:
        # Legacy driver-context enrichment runs immediately when the
        # evidence planner is disabled. With the planner on, enrichment
        # runs in the post-loop step in process_query().
        _enrich_prices_with_balancing_driver_context(ctx, invocation, is_explanation)

    if ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan:
        matched_step = plan_step
        if matched_step is None and is_primary:
            matched_step = next(
                (s for s in ctx.evidence_plan
                 if s["tool_name"] == invocation.name and not s.get("satisfied")),
                None,
            )
        if matched_step:
            stored_cols = list(ctx.cols if is_primary else df.columns)
            stored_rows = (
                list(ctx.rows) if is_primary
                else [tuple(r) for r in df.itertuples(index=False, name=None)]
            )
            query_hash = tool_invocation_hash(invocation.name, invocation.params)
            refs = build_provenance_refs(
                stored_cols,
                stored_rows,
                source="tool",
                query_hash=query_hash,
                parent_refs=(ctx.provenance_refs if is_primary else ()),
            )
            ctx.evidence_collected[matched_step["role"]] = {
                "tool": invocation.name,
                "params": dict(invocation.params),
                "df": (ctx.df.copy() if is_primary else df),
                "cols": stored_cols,
                "rows": stored_rows,
                "query_hash": query_hash,
                "provenance_refs": refs,
            }
            matched_step["satisfied"] = True
            remaining = sum(1 for s in ctx.evidence_plan if not s.get("satisfied"))
            log.info(
                "Evidence plan: stored %s result as %s, %d steps remaining",
                stage_label, matched_step["role"], remaining,
            )
        elif is_primary:
            log.info(
                "Evidence plan: %s tool %s not in plan; result stored on ctx.df only",
                stage_label, invocation.name,
            )

    return True


def _prefetched_result_executor(result):
    """Executor stand-in that returns an already-fetched tool result.

    Lets the prefetch pass hand a concurrently-fetched result to
    ``_execute_evidence_step`` through its ``executor=`` parameter, so the
    normalise/store logic (and every log line it emits) runs unchanged.
    """
    def _executor(_invocation):
        return result
    return _executor


def _execute_secondary_call(executor, invocation, child_deadline):
    """Run one secondary call with its parent identity and bounded deadline."""
    with bind_request_execution_scope_snapshot(deadline=child_deadline):
        return executor(invocation)


def _run_secondary_evidence_loop(
    ctx: QueryContext,
    *,
    executor: callable,
    timeout_seconds: float | None = None,
) -> QueryContext:
    """Run Stage 0.8 secondary evidence loop.

    Iterates remaining unsatisfied plan steps (capped at
    ``_EVIDENCE_LOOP_MAX_STEPS`` and ``EVIDENCE_LOOP_BUDGET_SECONDS``),
    executes each via the shared ``_execute_evidence_step`` helper with
    secondary semantics (``is_primary=False``; no relevance validation;
    no per-step tool-call metric or trace; result merged into
    ``ctx.evidence_collected`` only), and finally joins secondary frames
    into ``ctx.df`` via ``merge_evidence_into_context``.

    When ``EVIDENCE_PARALLEL_SECONDARY`` is on and two or more steps
    remain, pure-I/O calls are prefetched through the process-wide bounded
    DB-work coordinator. Every call inherits an actor/request-bound child
    deadline, while ctx mutation remains serial and plan-ordered. Evidence
    storage order, log order, trace shapes, and metric counters are therefore
    unchanged. The loop budget bounds waiting; queued work is cancelled at
    exit and PostgreSQL ``statement_timeout`` bounds running DB statements.
    A non-cooperative overrun is explicitly surfaced in coordinator metrics.
    Phase F.5.1.a moved this body out of ``evidence_planner`` into
    pipeline.py to remove the circular-import workaround that previously
    forced a lazy ``from agent.pipeline import _execute_evidence_step``
    inside the loop. ``evidence_planner.execute_remaining_evidence`` is
    preserved as a thin delegate that calls this function and passes its
    local ``execute_tool`` binding, so tests that monkey-patch
    ``agent.evidence_planner.execute_tool`` continue to work — the
    shared secondary executor submits that same binding.
    """
    remaining = [
        s for s in ctx.evidence_plan
        if not s.get("satisfied") and not s.get("error")
    ]
    cap = min(len(remaining), evidence_planner._EVIDENCE_LOOP_MAX_STEPS)
    steps = remaining[:cap]

    configured_budget = float(evidence_planner.EVIDENCE_LOOP_BUDGET_SECONDS)
    budget = (
        configured_budget
        if timeout_seconds is None
        else min(configured_budget, max(0.0, timeout_seconds))
    )
    deadline = time.monotonic() + budget

    invocations = [
        ToolInvocation(
            name=step["tool_name"],
            params=step["params"],
            confidence=0.85,
            reason=f"evidence_plan:{step['role']}",
        )
        for step in steps
    ]

    # Every secondary call receives a child deadline that cannot outlive the
    # request. Parallel work uses the one process-wide bounded executor; ctx
    # mutation still happens strictly in plan order below.
    child_deadline = cap_request_deadline(
        maximum_seconds=budget,
        source="secondary_evidence",
    )
    request_scope = current_request_execution_scope()
    request_key_seed = (
        f"{request_scope.actor_binding}:{request_scope.request_id}"
        if request_scope is not None and request_scope.request_id
        else ctx.trace_id or f"local-{id(ctx)}-{time.monotonic_ns()}"
    )
    request_key = hash_private_identifier(
        request_key_seed,
        namespace="secondary-request",
    )
    parallel = EVIDENCE_PARALLEL_SECONDARY and len(steps) >= 2
    futures: list[Future | None] = [None] * len(steps)
    if parallel:
        futures = []
        for invocation in invocations:
            captured_context = copy_context()

            def _call(
                captured_context=captured_context,
                invocation=invocation,
            ):
                return captured_context.run(
                    _execute_secondary_call,
                    executor,
                    invocation,
                    child_deadline,
                )

            try:
                future = db_work_coordinator.submit_secondary(
                    request_key,
                    _call,
                    timeout_seconds=max(0.0, deadline - time.monotonic()),
                )
            except DatabaseWorkCapacityExceeded as exc:
                future = Future()
                future.set_exception(exc)
            futures.append(future)

    try:
        for step, invocation, future in zip(steps, invocations, futures):
            step_executor = executor
            if future is None:
                # Serial path: the budget gates starting a step and the DB
                # gateway enforces the same child deadline during execution.
                if time.monotonic() > deadline:
                    step["error"] = f"evidence_loop_budget_exceeded_{budget}s"
                    log.warning(
                        "Evidence loop budget exhausted (%.1fs); skipping step: tool=%s role=%s",
                        budget, step["tool_name"], step.get("role"),
                    )
                    continue

                def step_executor(inv):
                    return _execute_secondary_call(
                        executor,
                        inv,
                        child_deadline,
                    )
            else:
                # Prefetched path: the budget bounds waiting for the result.
                try:
                    result = future.result(timeout=max(0.0, deadline - time.monotonic()))
                except FuturesTimeoutError as exc:
                    if future.done():
                        # The tool itself raised a TimeoutError (alias of the
                        # futures timeout) — treat as a step failure, not budget.
                        step["error"] = str(exc)
                        log.warning(
                            "Evidence loop: step failed. role=%s tool=%s err=%s",
                            step["role"], step["tool_name"], exc,
                        )
                    else:
                        step["error"] = f"evidence_loop_budget_exceeded_{budget}s"
                        log.warning(
                            "Evidence loop budget exhausted (%.1fs); skipping step: tool=%s role=%s",
                            budget, step["tool_name"], step.get("role"),
                        )
                    continue
                except Exception as exc:
                    step["error"] = str(exc)
                    log.warning(
                        "Evidence loop: step failed. role=%s tool=%s err=%s",
                        step["role"], step["tool_name"], exc,
                    )
                    continue
                step_executor = _prefetched_result_executor(result)

            try:
                _execute_evidence_step(
                    ctx,
                    invocation,
                    plan_step=step,
                    is_primary=False,
                    is_explanation=False,
                    stage_label="stage_0_8",
                    validate_relevance=False,
                    emit_tool_call_metric=False,
                    emit_tool_execute_trace=False,
                    executor=step_executor,
                )
                stored_rows = len(ctx.evidence_collected.get(step["role"], {}).get("rows", []))
                log.info(
                    "Evidence loop: fetched %s via %s (%d rows)",
                    step["role"], invocation.name, stored_rows,
                )
            except Exception as exc:
                step["error"] = str(exc)
                log.warning(
                    "Evidence loop: step failed. role=%s tool=%s err=%s",
                    step["role"], step["tool_name"], exc,
                )
    finally:
        if parallel:
            drain = db_work_coordinator.drain_secondary(
                request_key,
                timeout_seconds=DB_SECONDARY_DRAIN_TIMEOUT_MS / 1000.0,
            )
            if drain.remaining:
                log.error(
                    "Secondary evidence work exceeded its parent deadline: remaining=%d",
                    drain.remaining,
                )

    ctx = evidence_planner.merge_evidence_into_context(ctx)
    ctx.evidence_plan_complete = all(s.get("satisfied") for s in ctx.evidence_plan)
    return ctx


def _execute_evidence_plan(ctx: QueryContext) -> QueryContext:
    """Run primary execution + secondary evidence loop + driver-context enrichment.

    Phase §5.1.b. Consolidates the previously-separate Stages 0.5/0.6/0.7
    (primary tool execution via the four-strategy chain in
    ``_pick_primary_invocation``) and Stage 0.8 (secondary plan-step loop)
    into one function with one entry point from ``process_query``.

    Three logical passes, in order:

    1. **Pass 1 - primary execution**: ``_pick_primary_invocation``
       returns the first non-None invocation from the strategy chain;
       per-source traces and metrics fire; ``_execute_evidence_step``
       runs with ``is_primary=True``. On failure, recovery depends on
       the source label (analyzer-built / router-fallback get
       ``_attempt_analyzer_tool_recovery``; plan / keyword-router get
       a log + fall-through to SQL).

    2. **Pass 2 - secondary loop**: when the evidence plan still has
       unsatisfied steps, ``execute_remaining_evidence`` (delegating
       to ``_run_secondary_evidence_loop``) iterates over them.

    3. **Driver-context enrichment**: when the primary execution
       produced a usable result (``ctx.used_tool`` and
       ``ctx.tool_name`` set), ``_enrich_prices_with_balancing_driver_context``
       attaches balancing-specific contribution / source-price columns.

    Behaviour preservation: every trace shape
    (``stage_0_5_plan_driven``, ``stage_0_5_router_match``,
    ``stage_0_5_router_match`` miss_detail, ``stage_0_6_tool_execute``,
    ``stage_0_7_analyzer_route`` decision + timed,
    ``stage_0_7_analyzer_tool_execute``, ``stage_0_8_evidence_loop``),
    every metric counter (``log_router_match``, ``log_stage_0_7``,
    ``log_tool_fallback_intent``, ``log_tool_error``,
    ``log_relevance_block``), and every recovery path
    (``_attempt_analyzer_tool_recovery`` for analyzer-source failures,
    SQL fall-through for plan/router-source failures) fires under the
    same conditions as before. Pure refactor.
    """
    _require_request_budget(ctx, "stage_0_5_evidence")
    from utils.resilience import db_circuit_breaker
    # Advisory peek only. The real DB probe is owned by core.query_executor's
    # guarded allow_request()/record_* pair; calling allow_request() here would
    # consume the half-open probe slot without ever recording an outcome, which
    # wedges the breaker in half_open (probe forever "in flight") until restart.
    _cb_allowed, _cb_reason = db_circuit_breaker.would_allow()
    if not _cb_allowed:
        log.warning(
            "Skipping tool execution: circuit breaker is %s (%s). "
            "Pipeline will fall through to Stage 1/2 or CLARIFY.",
            db_circuit_breaker.state, _cb_reason,
        )

    # ---- Pass 1: primary execution via the strategy chain ----
    if ENABLE_TYPED_TOOLS and _cb_allowed:
        t_stage_primary = time.time()

        is_exp = _should_route_tool_as_explanation(ctx)

        primary_invocation, primary_plan_step, primary_source, primary_build_error = (
            _pick_primary_invocation(ctx, is_exp)
        )
        primary_is_analyzer_source = primary_source in {"analyzer_built", "router_fallback"}
        primary_is_plan_or_router_source = primary_source in {"plan_driven", "keyword_router"}

        # Capture build error on ctx for downstream observability when the
        # analyzer-built strategy raised (matches Stage 0.7's prior behaviour).
        if primary_build_error:
            ctx.tool_fallback_reason = f"analyzer_tool_build_error:{primary_build_error}"

        # Stage 0.5 traces (always emitted — preserves backward compat)
        if primary_source == "plan_driven":
            _emit_trace_stage(
                ctx, "stage_0_5_plan_driven", t_stage_primary,
                plan_tool=primary_plan_step["tool_name"],
                plan_role=primary_plan_step["role"],
            )
        _emit_trace_stage(
            ctx, "stage_0_5_router_match", t_stage_primary,
            matched=primary_is_plan_or_router_source,
            plan_driven=(primary_source == "plan_driven"),
        )

        # Stage 0.7 begins whenever Stage 0.5's strategies (plan-driven,
        # keyword router) did not produce an invocation — i.e. whenever
        # we're in the analyzer-source half of the chain or we missed
        # entirely.
        primary_entered_stage_0_7 = not primary_is_plan_or_router_source

        if primary_is_plan_or_router_source:
            if primary_source == "plan_driven":
                metrics.log_router_match("plan_driven")
            elif "semantic fallback" in (primary_invocation.reason or "").lower():
                metrics.log_router_match("semantic")
            else:
                metrics.log_router_match("deterministic")
        else:
            # Stage 0.5's strategies missed. The original code emitted miss +
            # miss_detail unconditionally — even when Stage 0.7 later
            # succeeded — so dashboards see the same "miss" count regardless
            # of whether an analyzer-built strategy rescued the request.
            metrics.log_router_match("miss")
            trace_detail(
                log, ctx, "stage_0_5_router_match", "miss_detail",
                semantic_fallback_enabled=ROUTER_ENABLE_SEMANTIC_FALLBACK,
                semantic_scores=_last_semantic_scores.copy(),
            )

        if primary_entered_stage_0_7:
            metrics.log_stage_0_7("entered")
            t_stage_0_7 = time.time()

            qa = ctx.question_analysis if ctx.has_authoritative_question_analysis else None
            trace_detail(
                log, ctx, "stage_0_7_analyzer_route", "decision",
                invocation_built=primary_is_analyzer_source,
                preferred_path=qa.routing.preferred_path.value if qa else "",
                prefer_tool=qa.routing.prefer_tool if qa else False,
                top_tool=primary_invocation.name if primary_is_analyzer_source else None,
                top_score=primary_invocation.confidence if primary_is_analyzer_source else None,
                build_error=primary_build_error,
                analyzer_available=bool(qa),
                hints_enabled=ENABLE_QUESTION_ANALYZER_HINTS,
            )
            if primary_is_analyzer_source:
                metrics.log_stage_0_7("invocation_built")
                _emit_trace_stage(
                    ctx, "stage_0_7_analyzer_route", t_stage_0_7,
                    tool=primary_invocation.name,
                    confidence=primary_invocation.confidence,
                )
                metrics.log_router_match("analyzer")

        if primary_invocation is not None:
            ctx.used_tool = True
            ctx.tool_name = primary_invocation.name
            ctx.tool_params = dict(primary_invocation.params)
            ctx.tool_match_reason = primary_invocation.reason
            ctx.tool_confidence = primary_invocation.confidence

            try:
                if primary_is_analyzer_source:
                    _primary_used = _execute_evidence_step(
                        ctx,
                        primary_invocation,
                        plan_step=None,
                        is_primary=True,
                        is_explanation=is_exp,
                        stage_label="stage_0_7",
                        tool_execute_trace="stage_0_7_analyzer_tool_execute",
                        relevance_log_prefix="Analyzer",
                        relevance_fallback_reason_prefix="analyzer_tool",
                        emit_fallback_intent_on_relevance_block=False,
                    )
                    if _primary_used:
                        metrics.log_stage_0_7("used_result")
                        log.info(
                            "Analyzer tool route hit: tool=%s confidence=%.2f reason=%s",
                            primary_invocation.name,
                            primary_invocation.confidence,
                            primary_invocation.reason,
                        )
                else:
                    _execute_evidence_step(
                        ctx,
                        primary_invocation,
                        plan_step=primary_plan_step,
                        is_primary=True,
                        is_explanation=is_exp,
                        stage_label="stage_0_5",
                    )
                    log.info(
                        "Typed tool route hit: tool=%s confidence=%.2f reason=%s",
                        primary_invocation.name,
                        primary_invocation.confidence,
                        primary_invocation.reason,
                    )
            except Exception as exc:
                if primary_is_analyzer_source:
                    if (
                        ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan
                        and evidence_planner.has_unsatisfied_steps(ctx.evidence_plan)
                    ):
                        _failed_step = next(
                            (s for s in ctx.evidence_plan
                             if s["tool_name"] == primary_invocation.name
                             and not s.get("satisfied")),
                            None,
                        )
                        if _failed_step:
                            _failed_step["error"] = f"stage_0_7:{exc}"
                        log.info(
                            "Analyzer tool failed; evidence plan has remaining steps. "
                            "tool=%s err=%s",
                            primary_invocation.name, exc,
                        )
                        ctx.used_tool = False
                        ctx.tool_fallback_reason = f"analyzer_tool_execution_error:{exc}"
                        clear_provenance(ctx)
                    else:
                        ctx, recovered = _attempt_analyzer_tool_recovery(
                            ctx,
                            failed_invocation=primary_invocation,
                            is_explanation=is_exp,
                        )
                        if not recovered:
                            metrics.log_tool_error()
                            ctx.used_tool = False
                            ctx.tool_fallback_reason = (
                                f"analyzer_tool_execution_error:{exc}"
                            )
                            metrics.log_tool_fallback_intent(
                                ctx.query, "analyzer_tool_execution_error"
                            )
                            clear_provenance(ctx)
                            log.warning(
                                "Analyzer tool failed; falling back. tool=%s err=%s",
                                primary_invocation.name, exc,
                            )
                else:
                    metrics.log_tool_error()
                    ctx.used_tool = False
                    ctx.tool_fallback_reason = str(exc)
                    metrics.log_tool_fallback_intent(ctx.query, "tool_execution_error")
                    clear_provenance(ctx)
                    if primary_plan_step:
                        primary_plan_step["error"] = f"stage_0_5:{exc}"
                    log.warning(
                        "Typed tool failed; falling back to SQL path. tool=%s err=%s",
                        primary_invocation.name, exc,
                    )
        elif primary_entered_stage_0_7:
            # No invocation from any strategy. Stage 0.7's miss handling
            # decides whether to attempt the legacy recovery path.
            if primary_build_error:
                if (
                    ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan
                    and evidence_planner.has_unsatisfied_steps(ctx.evidence_plan)
                ):
                    log.info("Analyzer tool build failed; evidence plan has remaining steps.")
                else:
                    ctx, recovered = _attempt_analyzer_tool_recovery(
                        ctx,
                        failed_invocation=None,
                        is_explanation=is_exp,
                    )
                    if recovered:
                        log.info(
                            "Analyzer tool build failure recovered via resolved-query fallback."
                        )
                    else:
                        metrics.log_tool_fallback_intent(
                            ctx.query, "analyzer_tool_build_error"
                        )
            else:
                metrics.log_tool_fallback_intent(ctx.query, "router_no_match")

    # ---- Pass 2: secondary evidence loop (Stage 0.8) ----
    if ENABLE_EVIDENCE_PLANNER and ctx.evidence_plan and not all(
        s.get("satisfied") for s in ctx.evidence_plan
    ):
        t_stage_0_8 = time.time()
        remaining_seconds = _remaining_request_seconds(ctx, "stage_0_8_evidence_loop")
        loop_timeout_seconds = (
            None
            if remaining_seconds is None
            else min(evidence_planner.EVIDENCE_LOOP_BUDGET_SECONDS, remaining_seconds)
        )
        ctx = evidence_planner.execute_remaining_evidence(
            ctx,
            timeout_seconds=loop_timeout_seconds,
        )
        _emit_trace_stage(
            ctx, "stage_0_8_evidence_loop", t_stage_0_8,
            complete=ctx.evidence_plan_complete,
            collected=len(ctx.evidence_collected),
            satisfied=[s["tool_name"] for s in ctx.evidence_plan if s.get("satisfied")],
        )

    # ---- Post-loop: balancing driver-context enrichment ----
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

    # P4.1 (finding H1): finalize the canonical evidence frame from the settled
    # primary + secondary + balancing-enrichment state. This is the normal-path
    # counterpart to the recovery-path attachment, superseding any stale frame a
    # mid-execution recovery may have left. Gated by ENAI_EVIDENCE_FINALIZATION_MODE.
    evidence_finalizer.safe_finalize(ctx, stage="stage_0_8_settled")

    return ctx


_DATA_SHAPED_ANSWER_KINDS = {
    AnswerKind.SCALAR,
    AnswerKind.LIST,
    AnswerKind.TIMESERIES,
    AnswerKind.COMPARISON,
    AnswerKind.SCENARIO,
    AnswerKind.FORECAST,
}


def _detect_evidence_anomaly(ctx: QueryContext) -> str | None:
    """Detect surprising primary evidence after plan execution (§3.6).

    Shadow signal for every request (counter + trace); the flag-gated
    re-analysis consumes it. Only fires for authoritative, data-shaped
    contracts — knowledge/clarify paths legitimately carry no tabular
    evidence, and SQL-fallback requests are judged by the fallback, not here.
    """
    if not ctx.has_authoritative_question_analysis:
        return None
    qa = ctx.question_analysis
    if qa is None or qa.answer_kind not in _DATA_SHAPED_ANSWER_KINDS:
        return None
    if not ctx.used_tool:
        return None
    if not ctx.rows:
        return "primary_empty"
    bounds = period_bounds_from_hint(qa)
    if bounds:
        span = df_date_span(ctx.df)
        if span:
            req_start = date.fromisoformat(bounds[0])
            req_end = date.fromisoformat(bounds[1])
            if span[1] < req_start or span[0] > req_end:
                return "period_gap"
    return None


def _attempt_evidence_reanalysis(ctx: QueryContext, anomaly: str) -> "StageResult":
    """One flag-gated re-analysis on surprising evidence (§3.6, design item 1).

    P4.5 (finding M4): re-analysis re-runs Stage 0.2 with the anomaly as trusted
    context AND re-runs the *entire* post-analysis dependent slice — resolved
    query, answer-kind finalization, retrieval tier + vector knowledge, response
    mode + derived-metric setup, the knowledge/clarify short-circuits, plan
    build + validation, and evidence execution — mirroring the same ordered
    sequence ``process_query`` runs after Stage 0.2. This is the correctness
    fix over the earlier slice, which reran only analysis + plan: a
    reclassification to KNOWLEDGE/CLARIFY previously continued in stale
    data-mode state and never reached the conceptual/clarify terminal.

    Returns a :class:`StageResult`. ``terminal=True`` when the fresh
    classification short-circuits (knowledge answer, clarification, or a
    plan-validation reject); the caller returns that answer instead of
    continuing the data path. ``reanalysis_attempted`` guarantees this runs at
    most once per request.
    """
    t_stage = time.time()
    ctx.reanalysis_attempted = True
    ctx.evidence_anomaly = (
        f"Prior execution anomaly: {anomaly}. The first interpretation "
        "produced no usable evidence for the interpreted tool/period; "
        "reconsider the interpretation."
    )
    prev_tool = str(ctx.tool_name or "")
    prev_kind = (
        ctx.question_analysis.answer_kind.value
        if ctx.question_analysis and ctx.question_analysis.answer_kind
        else ""
    )
    prev_mode = str(ctx.response_mode or "")

    # 1. Re-run the Stage 0.2 interpreter with the anomaly as trusted context.
    ctx = planner.analyze_question_active(ctx)

    # 2. Reset per-attempt evidence AND terminal/mode state so the second pass
    #    starts exactly like a fresh request and cannot inherit a stale data-mode
    #    decision (skip_sql, resolution policy, terminal outcome).
    ctx.used_tool = False
    ctx.tool_name = None
    ctx.tool_params = {}
    ctx.tool_match_reason = ""
    ctx.tool_confidence = 0.0
    ctx.tool_fallback_reason = ""
    ctx.df = pd.DataFrame()
    ctx.rows = []
    ctx.cols = []
    ctx.evidence_collected = {}
    ctx.evidence_plan = []
    ctx.evidence_plan_complete = False
    ctx.plan_validation = None
    ctx.skip_sql = False
    ctx.skip_sql_reason = ""
    ctx.resolution_policy = ""
    ctx.clarify_reason = ""
    ctx.terminal_outcome = ""
    clear_provenance(ctx)
    # The first attempt's evidence is discarded — drop any frame it built so a
    # stale frame can never survive into the re-executed plan.
    evidence_finalizer.safe_invalidate(ctx, stage="stage_0_9_reanalysis", reason="reanalysis_reset")

    def _reanalysis_trace(terminal_reason: str) -> None:
        new_kind = (
            ctx.question_analysis.answer_kind.value
            if ctx.question_analysis and ctx.question_analysis.answer_kind
            else ""
        )
        _emit_trace_stage(
            ctx, "stage_0_9_reanalysis", t_stage,
            anomaly=anomaly,
            tool_changed=(str(ctx.tool_name or "") != prev_tool),
            kind_changed=(new_kind != prev_kind),
            mode_changed=(str(ctx.response_mode or "") != prev_mode),
            terminal_reason=terminal_reason,
        )

    # 3. Re-run the post-analysis dependent slice (mirrors process_query after
    #    Stage 0.2). Recomputing resolved query, retrieval tier, vector
    #    knowledge, and response mode is exactly what lets a data->knowledge or
    #    data->clarify reclassification take effect (M4).
    routing_query, routing_query_source = _derive_resolved_query(ctx)
    ctx.resolved_query = routing_query
    ctx.resolved_query_source = routing_query_source
    if routing_query_source == "llm_active_canonical":
        ctx.semantic_locked = True
    _finalize_answer_kind(ctx)

    _retrieval_tier = _resolve_vector_tier(ctx)
    ctx.vector_retrieval_tier = _retrieval_tier.value
    _require_request_budget(ctx, "stage_0_9_reanalysis_vector")
    _run_vector_knowledge_stage(ctx, _retrieval_tier, routing_query)

    _apply_response_mode(ctx)

    # 4. Honor the short-circuits the fresh classification may now trigger.
    _res = _early_answer_clarify(ctx)
    ctx = _res.ctx
    if _res.terminal:
        _reanalysis_trace("clarify")
        return StageResult(ctx, terminal=True)
    _res = _early_answer_conceptual(ctx)
    ctx = _res.ctx
    if _res.terminal:
        _reanalysis_trace("knowledge")
        return StageResult(ctx, terminal=True)

    # 5. Rebuild + validate the plan, then re-execute evidence.
    if ENABLE_EVIDENCE_PLANNER:
        ctx = evidence_planner.build_evidence_plan(ctx)
        _res = _enforce_plan_validation_stage(ctx)
        ctx = _res.ctx
        if _res.terminal:
            _reanalysis_trace("plan_validation_reject")
            return StageResult(ctx, terminal=True)
    ctx = _execute_evidence_plan(ctx)

    _reanalysis_trace("continue")
    return StageResult(ctx, terminal=False)


def _finalize_answer_kind(ctx: QueryContext) -> None:
    """Cross-check + finalize answer_kind/render_style and effective_answer_kind.

    Extracted verbatim from process_query (audit P0-4a). Pure ctx mutation; no
    early return.
    """
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
        #
        # Gated on a quantitative anchor in the user query: see comment near
        # ``_QUANTITATIVE_ANCHOR_RE``.  Without an anchor, the analyzer is
        # likely hallucinating ``scenario_factor`` (production log 2026-05-13);
        # we stay on the narrative path instead of computing a garbage number.
        if qa.answer_kind in _SCENARIO_OVERRIDE_ELIGIBLE:
            derived = qa.analysis_requirements.derived_metrics or []
            has_scenario_metric = any(
                m.metric_name in _SCENARIO_DERIVED_METRICS for m in derived
            )
            if has_scenario_metric and _query_has_quantitative_anchor(ctx.query):
                metrics.log_analyzer_cross_check("scenario_override_applied")
                log.info(
                    "answer_kind override: %s → SCENARIO (scenario derived metrics present)",
                    qa.answer_kind.value if qa.answer_kind else None,
                )
                qa.answer_kind = AnswerKind.SCENARIO
            elif has_scenario_metric:
                metrics.log_analyzer_cross_check("scenario_override_gated")
                log.info(
                    "answer_kind override suppressed: scenario derived metrics present "
                    "but no quantitative anchor in query (kind=%s, query=%.80s)",
                    qa.answer_kind.value if qa.answer_kind else None,
                    ctx.query,
                )
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


def _resolve_vector_tier(ctx: QueryContext) -> "VectorRetrievalTier":
    """Resolve + record the 3-tier vector-retrieval policy for this query.

    Extracted verbatim from process_query (audit P0-4a). Sets
    ctx.vector_retrieval_tier and returns the tier for local use by Stage 0.3.
    """
    # Three-tier vector retrieval policy (Phase D):
    #   FULL  → knowledge / explanation answers that consume passages directly
    #   LIGHT → narrative data answers that sprinkle in background
    #   SKIP  → deterministic data paths, clarify, etc.
    _qa_for_tier = ctx.question_analysis if ctx.has_authoritative_question_analysis else None
    _candidate_topic_names = (
        [tc.name for tc in (_qa_for_tier.knowledge.candidate_topics or [])]
        if _qa_for_tier is not None
        else None
    )
    _retrieval_tier = _resolve_vector_retrieval_tier(
        answer_kind=ctx.effective_answer_kind,
        render_style=(_qa_for_tier.render_style if _qa_for_tier is not None else None),
        is_conceptual=bool(ctx.is_conceptual),
        topics=_candidate_topic_names,
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
    return _retrieval_tier


def _apply_response_mode(ctx: QueryContext) -> None:
    """Derive + record response_mode/resolution_policy and policy-blocked flags.

    Extracted verbatim from process_query (audit P0-4a). Pure ctx mutation; no
    early return.
    """
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

    # Set the policy-blocked flag for observability before the short-circuit return.
    if ctx.response_mode == ResponseMode.KNOWLEDGE_PRIMARY or ctx.resolution_policy == ResolutionPolicy.CLARIFY:
        if ENABLE_TYPED_TOOLS:
            ctx.tool_blocked_by_policy = True

    trace_detail(
        log, ctx, "response_mode_derivation", "result",
        response_mode=ctx.response_mode,
        resolution_policy=ctx.resolution_policy,
        is_conceptual=ctx.is_conceptual,
        tool_blocked_by_policy=ctx.tool_blocked_by_policy,
        clarify_reason=ctx.clarify_reason,
        requested_derived_metrics=list(ctx.requested_derived_metrics or []),
        analyzer_available=ctx.question_analysis is not None,
        analyzer_source=ctx.question_analysis_source,
        semantic_locked=ctx.semantic_locked,
    )


def _run_vector_knowledge_stage(
    ctx: QueryContext, _retrieval_tier: "VectorRetrievalTier", routing_query: str
) -> None:
    """Stage 0.3 - vector-backed knowledge retrieval (shadow/active).

    No-op when the feature is disabled or the tier is SKIP. Extracted from
    process_query (audit P0-4b); uses the module-level _emit_trace_stage rather
    than the process_query-local _trace_stage closure.
    """
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
        _emit_trace_stage(
            ctx,
            "stage_0_3_vector_knowledge",
            t_stage,
            mode=retrieval_mode,
            tier=_retrieval_tier.value,
            chunk_count=bundle.chunk_count,
            error=bool(bundle.error),
            strategy=bundle.strategy.value,
        )
        # Phase A.2: adjacency observability. When
        # ``VECTOR_ADJACENCY_MODE != "off"``, the bundle carries any
        # adjacent chunks the retriever fetched. Emit a separate trace
        # event so the hit rate is visible in production logs without
        # changing the prompt content (A.3 owns the pack cutover).
        if bundle.adjacent_chunks:
            from knowledge.vector_retrieval import get_adjacency_mode

            _adj_sections = [
                f"{c.document_title or c.source_key} | "
                f"{c.section_title or c.section_path or f'chunk_{c.chunk_index}'}"
                for c in bundle.adjacent_chunks[:6]
            ]
            # Estimate the pack cost an A.3 cutover would incur — text + header.
            _adj_packed_chars = sum(
                len(c.text_content or "") + 80 for c in bundle.adjacent_chunks
            )
            trace_detail(
                log,
                ctx,
                "stage_0_3_vector_knowledge_adjacency",
                "validated",
                adjacency_mode=get_adjacency_mode(),
                adjacent_chunk_count=len(bundle.adjacent_chunks),
                adjacent_sections=_adj_sections,
                would_be_packed_chars=_adj_packed_chars,
            )

        # Phase B.3: reference-expansion observability. Same shadow pattern
        # as adjacency — emit the resolved-chunk metadata so the operator
        # can observe hit rate before flipping to "on" in B.4.
        if bundle.reference_chunks:
            from knowledge.vector_retrieval import get_reference_expansion_mode

            _ref_sections = [
                f"{c.document_title or c.source_key} | "
                f"{c.section_title or c.section_path or f'chunk_{c.chunk_index}'}"
                for c in bundle.reference_chunks[:6]
            ]
            _ref_packed_chars = sum(
                len(c.text_content or "") + 80 for c in bundle.reference_chunks
            )
            # Also surface the union of outgoing-ref article numbers across
            # the primary set, so an operator can spot when the resolver
            # silently dropped refs (e.g. budget cap, missing target).
            _attempted_article_numbers: set[str] = set()
            for c in bundle.chunks:
                for ref in c.outgoing_refs or []:
                    kind_value = getattr(ref.kind, "value", str(ref.kind or ""))
                    if kind_value == "article":
                        num = str(ref.number or "").strip()
                        if num:
                            _attempted_article_numbers.add(num)
            trace_detail(
                log,
                ctx,
                "stage_0_3_vector_knowledge_references",
                "validated",
                reference_mode=get_reference_expansion_mode(),
                reference_chunk_count=len(bundle.reference_chunks),
                reference_sections=_ref_sections,
                attempted_article_numbers=sorted(_attempted_article_numbers),
                would_be_packed_chars=_ref_packed_chars,
            )


@dataclass
class StageResult:
    """Outcome of a process_query stage that may short-circuit the pipeline.

    ``ctx`` is the (possibly replaced) context to continue with. ``terminal``
    True means process_query should return ``ctx`` immediately because an early
    answer (clarify / conceptual / agent-exit) was produced. (audit P0-4c)
    """
    ctx: QueryContext
    terminal: bool = False


def _early_answer_clarify(ctx: QueryContext) -> StageResult:
    """Produce the clarify answer when resolution policy is CLARIFY (terminal).

    Extracted from process_query (audit P0-4c). Non-CLARIFY policies pass through
    unchanged (terminal=False). Uses the module-level _emit_trace_stage.
    """
    if ctx.resolution_policy != ResolutionPolicy.CLARIFY:
        return StageResult(ctx, terminal=False)
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
    ctx.terminal_outcome = TerminalOutcome.CLARIFICATION_REQUIRED.value
    metrics.log_terminal_outcome(TerminalOutcome.CLARIFICATION_REQUIRED.value)
    _emit_trace_stage(ctx, "stage_4_clarify_summary", t_stage, reason=ctx.clarify_reason)
    return StageResult(ctx, terminal=True)


def _early_answer_conceptual(ctx: QueryContext) -> StageResult:
    """Produce the conceptual answer when response mode is KNOWLEDGE_PRIMARY (terminal).

    Extracted from process_query (audit P0-4c). Other response modes pass through
    unchanged (terminal=False). Uses the module-level _emit_trace_stage.
    """
    if ctx.response_mode != ResponseMode.KNOWLEDGE_PRIMARY:
        return StageResult(ctx, terminal=False)
    t_stage = time.time()
    ctx = summarizer.answer_conceptual(ctx)
    ctx.terminal_outcome = TerminalOutcome.CONCEPTUAL_ANSWER.value
    metrics.log_terminal_outcome(TerminalOutcome.CONCEPTUAL_ANSWER.value)
    _emit_trace_stage(ctx, "stage_4_conceptual_summary", t_stage)
    return StageResult(ctx, terminal=True)


def _enforce_plan_validation_stage(ctx: QueryContext) -> StageResult:
    """P4.2 (finding M3): act on reject-severity plan-validation issues.

    Runs immediately after Stage 0.4 so an unsatisfiable plan makes ZERO
    tool/DB calls. In the default ``warn`` mode this stage is a pass-through
    — the typed result and counters exist, behavior is unchanged. In
    ``enforce`` mode a plan whose evidence provably cannot satisfy the answer
    contract terminal-clarifies instead of executing: the user picks a
    direction rather than receiving an answer shaped like something the
    evidence cannot support. Never loops: validation ran exactly once during
    Stage 0.4 (after plan repairs) and this stage only reads the result.
    """
    validation = getattr(ctx, "plan_validation", None)
    rejects = list(getattr(validation, "rejects", []) or [])
    if not rejects:
        return StageResult(ctx, terminal=False)
    if (
        PLAN_VALIDATION_MODE != "enforce"
        or not gate_is_active(ctx, GATE_PLAN_VALIDATION, default=True)
    ):
        # Shadow visibility: counters were already logged per-issue by
        # build_evidence_plan; nothing else to do in warn mode.
        return StageResult(ctx, terminal=False)

    first = rejects[0]
    for issue in rejects:
        metrics.log_plan_validation(issue.rule, "enforced")
    ctx.resolution_policy = ResolutionPolicy.CLARIFY
    ctx.clarify_reason = f"plan_validation_{first.rule}"
    t_stage = time.time()
    ctx = summarizer.answer_clarify(ctx)
    ctx.terminal_outcome = TerminalOutcome.CLARIFICATION_REQUIRED.value
    metrics.log_terminal_outcome(TerminalOutcome.CLARIFICATION_REQUIRED.value)
    _emit_trace_stage(
        ctx, "stage_0_4_plan_validation_clarify", t_stage,
        rule=first.rule,
        rejects=[issue.rule for issue in rejects],
    )
    log.info(
        "Plan validation enforcement: clarifying before execution. rule=%s rejects=%d",
        first.rule, len(rejects),
    )
    return StageResult(ctx, terminal=True)


def _run_generate_sql_stage(ctx: QueryContext) -> StageResult:
    """Legacy fallback: generate + execute SQL (or short-circuit to a conceptual answer).

    Extracted from process_query (audit P0-4d). terminal=True when a conceptual answer
    was produced (is_conceptual / skip_sql); otherwise SQL ran (or a tool already did)
    and the pipeline continues. Uses the module-level _emit_trace_stage.
    """
    # Final legacy fallback: generate SQL, execute it, then continue through analysis/summarization.
    if not ctx.used_tool:
        if ctx.tool_fallback_reason:
            metrics.log_tool_fallback_intent(ctx.query, f"tool_fallback:{ctx.tool_fallback_reason}")
        t_stage = time.time()
        ctx = planner.generate_plan(ctx)
        _emit_trace_stage(ctx, "stage_1_generate_plan", t_stage, conceptual=ctx.is_conceptual, skip_sql=ctx.skip_sql)
        log.info("Stage 1 complete | conceptual=%s | skip_sql=%s", ctx.is_conceptual, ctx.skip_sql)

        if ctx.is_conceptual or ctx.skip_sql:
            # Pre-execution skip: the planner's should_skip_sql_execution
            # decided this query needs no SQL — a legitimate conceptual route,
            # not a data failure.
            if ctx.skip_sql and not ctx.is_conceptual:
                log.info("Skipping SQL path: %s", ctx.skip_sql_reason)
            t_stage = time.time()
            ctx = summarizer.answer_conceptual(ctx)
            ctx.terminal_outcome = ctx.terminal_outcome or TerminalOutcome.CONCEPTUAL_ANSWER.value
            _emit_trace_stage(ctx, "stage_4_conceptual_summary", t_stage)
            return StageResult(ctx, terminal=True)

        t_stage = time.time()
        ctx = sql_executor.validate_and_execute(ctx)
        _emit_trace_stage(ctx, "stage_2_sql_execute", t_stage, rows=len(ctx.rows), cols=len(ctx.cols))
        log.info("Stage 2 complete | rows=%s | cols=%s", len(ctx.rows), len(ctx.cols))
        if ctx.skip_sql:
            # Post-execution skip: validate_and_execute set skip_sql because the
            # generated SQL failed validation or relevance — a data failure
            # (finding H12). On a data-primary request this must not be dressed
            # up as a conceptual success.
            log.info("Stage 2 blocked by policy: %s", ctx.skip_sql_reason)
            t_stage = time.time()
            ctx = _answer_skipped_sql_data_failure(ctx)
            _emit_trace_stage(
                ctx, "stage_4_conceptual_summary", t_stage,
                terminal_outcome=ctx.terminal_outcome,
            )
            return StageResult(ctx, terminal=True)
    else:
        log.info("Stage 2 bypassed | tool=%s | rows=%s", ctx.tool_name, len(ctx.rows))
    return StageResult(ctx, terminal=False)


def _answer_skipped_sql_data_failure(ctx: QueryContext) -> QueryContext:
    """Answer a request whose generated SQL was skipped after a failed attempt.

    P4.4 (finding H12). A data-primary request that reaches this point wanted
    data, but the generated SQL failed validation/relevance. The honest outcome
    is evidence-unavailable, not a conceptual narrative with invented numbers.
    The classification and its shadow counter are always recorded; the
    user-facing routing change is gated by ENABLE_HONEST_TERMINAL_OUTCOMES so
    the operator can size the blast radius (evidence_unavailable_shadow) before
    cutover. Anti-retry-storm behavior is unchanged either way (HTTP 200).
    """
    is_data_primary = ctx.response_mode != ResponseMode.KNOWLEDGE_PRIMARY
    if is_data_primary:
        metrics.log_terminal_outcome(f"{TerminalOutcome.EVIDENCE_UNAVAILABLE.value}_shadow")
        if (
            ENABLE_HONEST_TERMINAL_OUTCOMES
            and gate_is_active(ctx, GATE_HONEST_TERMINAL_OUTCOMES, default=True)
        ):
            return summarizer.answer_evidence_unavailable(ctx)
    ctx = summarizer.answer_conceptual(ctx)
    ctx.terminal_outcome = ctx.terminal_outcome or TerminalOutcome.CONCEPTUAL_ANSWER.value
    metrics.log_terminal_outcome(ctx.terminal_outcome)
    return ctx


def _check_missing_evidence_stage(ctx: QueryContext) -> StageResult:
    """Block data summarization when requested derived evidence is missing.

    Extracted from process_query (audit P0-4d). terminal=True only when there is no
    partial evidence and we must clarify; partial-evidence and not-blocked both pass
    through. Uses the module-level _emit_trace_stage.
    """
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
            log.info(
                "Blocking data summarization due to missing derived evidence: %s",
                ctx.missing_evidence_for_metrics,
            )
            t_stage = time.time()
            ctx = summarizer.answer_clarify(ctx)
            ctx.terminal_outcome = TerminalOutcome.CLARIFICATION_REQUIRED.value
            metrics.log_terminal_outcome(TerminalOutcome.CLARIFICATION_REQUIRED.value)
            _emit_trace_stage(
                ctx,
                "stage_4_clarify_summary",
                t_stage,
                reason=ctx.clarify_reason,
                missing_evidence=",".join(ctx.missing_evidence_for_metrics),
            )
            return StageResult(ctx, terminal=True)
    return StageResult(ctx, terminal=False)


def _process_query_impl(
    query: str,
    conversation_history=None,
    trace_id: str = "",
    session_id: str = "",
    previous_contract_snapshot: str = "",
    request_deadline=None,
    actor_id: str = "",
    request_id: str = "",
) -> QueryContext:
    """Run the full query pipeline and return a populated QueryContext."""
    # Detect clarification-selection replies (e.g. "1", "option 2")
    selected = _detect_clarify_selection(query, conversation_history)
    if selected:
        query = _rewrite_query_for_clarify_selection(selected, conversation_history)
        log.info(
            "Clarification selection detected. rewritten_query_hash=%s",
            hash_private_identifier(query, namespace="query"),
        )

    ctx = QueryContext(
        query=query,
        conversation_history=conversation_history,
        trace_id=trace_id,
        session_id=session_id,
        request_deadline=request_deadline,
        previous_contract_snapshot=previous_contract_snapshot,
        clarify_selection_override=selected is not None,
        p4_rollout_decisions=build_rollout_decisions(
            {
                GATE_EVIDENCE_FINALIZATION: (
                    evidence_finalizer.EVIDENCE_FINALIZATION_MODE == "enforce",
                    EVIDENCE_FINALIZATION_ENFORCE_PERCENT,
                ),
                GATE_PLAN_VALIDATION: (
                    PLAN_VALIDATION_MODE == "enforce",
                    PLAN_VALIDATION_ENFORCE_PERCENT,
                ),
                GATE_HONEST_TERMINAL_OUTCOMES: (
                    ENABLE_HONEST_TERMINAL_OUTCOMES,
                    HONEST_TERMINAL_OUTCOMES_PERCENT,
                ),
                GATE_EVIDENCE_REANALYSIS: (
                    ENABLE_EVIDENCE_REANALYSIS,
                    EVIDENCE_REANALYSIS_PERCENT,
                ),
            },
            actor_id=actor_id,
            session_id=session_id,
            request_id=request_id,
        ),
    )

    # `_trace_stage` is a thin wrapper around the module-level helper that
    # implicitly passes the current ctx. Keeping it as a local closure
    # preserves every call site inside this function without needing a
    # `ctx` argument; the module-level `_emit_trace_stage` is what the
    # extracted `_execute_evidence_step` helper uses.
    def _trace_stage(stage_name: str, started_at: float, **extra):
        _emit_trace_stage(ctx, stage_name, started_at, **extra)

    stages = PipelineStageOrchestrator(
        ctx,
        require_budget=_require_request_budget,
    )

    # Stage 0: cheap preparation
    t_stage = time.time()
    ctx = stages.run("stage_0_prepare_context", planner.prepare_context)
    _trace_stage("stage_0_prepare_context", t_stage, conceptual=ctx.is_conceptual, lang=ctx.lang_code)

    # Stage 0.2: structured question analysis
    if ENABLE_QUESTION_ANALYZER_SHADOW or ENABLE_QUESTION_ANALYZER_HINTS:
        t_stage = time.time()
        analyzer_mode = "active" if ENABLE_QUESTION_ANALYZER_HINTS else "shadow"
        analyzer_stage = (
            planner.analyze_question_active
            if ENABLE_QUESTION_ANALYZER_HINTS
            else planner.analyze_question_shadow
        )
        ctx = stages.run("stage_0_2_question_analyzer", analyzer_stage)
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

    _finalize_answer_kind(ctx)

    _retrieval_tier = _resolve_vector_tier(ctx)
    ctx.vector_retrieval_tier = _retrieval_tier.value

    stages.run_effect(
        "stage_0_3_vector_knowledge",
        lambda stage_ctx: _run_vector_knowledge_stage(
            stage_ctx,
            _retrieval_tier,
            routing_query,
        ),
    )

    _apply_response_mode(ctx)

    if stages.run_terminal("stage_4_early_answer", _early_answer_clarify):
        return stages.context
    ctx = stages.context

    # Conceptual short-circuit
    if stages.run_terminal(
        "stage_4_early_conceptual_answer",
        _early_answer_conceptual,
    ):
        return stages.context
    ctx = stages.context

    # Stage 0.4: expand the authoritative analyzer output into the exact datasets we still need.
    if ENABLE_EVIDENCE_PLANNER:
        t_stage = time.time()
        ctx = stages.run(
            "stage_0_4_evidence_plan",
            evidence_planner.build_evidence_plan,
        )
        _trace_stage(
            "stage_0_4_evidence_plan", t_stage,
            steps=len(ctx.evidence_plan),
            source=ctx.evidence_plan_source,
            tools=[s["tool_name"] for s in ctx.evidence_plan],
        )

        # P4.2 (M3): reject-severity contract mismatches clarify BEFORE any
        # tool/DB call when ENAI_PLAN_VALIDATION_MODE=enforce; pass-through
        # (typed result + counters only) in the default warn mode.
        _res = _enforce_plan_validation_stage(ctx)
        ctx = stages.adopt(_res.ctx)
        if _res.terminal:
            return ctx

    # Stage 0.5 / 0.6 / 0.7 / 0.8: primary execution (strategy chain) +
    # secondary evidence loop + driver-context enrichment, all in one
    # consolidated function. See _execute_evidence_plan above for the
    # full body — every trace shape, metric counter, and recovery path
    # is identical to the pre-§5.1.b form.
    ctx = stages.adopt(_execute_evidence_plan(ctx))

    # Stage 0.9: surprising-evidence detection (always) + flag-gated single
    # re-analysis (§3.6, design item 1).
    _anomaly = _detect_evidence_anomaly(ctx)
    if _anomaly:
        metrics.log_evidence_anomaly(_anomaly)
        _emit_trace_stage(ctx, "stage_0_9_evidence_anomaly", time.time(), tag=_anomaly)
        if (
            ENABLE_EVIDENCE_REANALYSIS
            and gate_is_active(ctx, GATE_EVIDENCE_REANALYSIS, default=True)
            and not ctx.reanalysis_attempted
        ):
            should_stop = stages.run_terminal(
                "stage_0_9_evidence_reanalysis",
                lambda stage_ctx: _attempt_evidence_reanalysis(stage_ctx, _anomaly),
            )
            ctx = stages.context
            # A reclassification to knowledge/clarify (or a plan-validation
            # reject) during re-analysis is terminal — the fresh mode owns the
            # answer instead of the pipeline continuing in stale data mode (M4).
            if should_stop:
                return ctx

    # Legacy agent loop removed (audit): with Stage 0.2 authoritative it never ran,
    # and analyzer-failure / no-tool cases fall through to the generate-plan/SQL
    # fallback below (a more capable path than the retired keyword-driven loop).
    if stages.run_terminal("stage_1_generate_sql", _run_generate_sql_stage):
        return stages.context
    ctx = stages.context

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
    ctx = stages.run("stage_3_analyzer_enrich", derive_evidence)
    _trace_stage(
        "stage_3_analyzer_enrich",
        t_stage,
        share_override=bool(ctx.share_summary_override),
        correlation_keys=list(ctx.correlation_results.keys()),
    )
    log.info("Stage 3 complete | analysis enrichment done")

    # P4.1 (finding H1): re-finalize after Stage 3 enrichment, which is the last
    # step that mutates the tabular evidence before Stage 4 rendering. The frame
    # the generic renderer sees now reflects the fully enriched evidence.
    evidence_finalizer.safe_finalize(ctx, stage="stage_3_enriched")

    ctx.missing_evidence_for_metrics = _missing_requested_evidence(ctx)
    trace_detail(
        log,
        ctx,
        "stage_3_analyzer_enrich",
        "evidence_readiness",
        requested_derived_metrics=list(ctx.requested_derived_metrics or []),
        missing_evidence_for_metrics=list(ctx.missing_evidence_for_metrics or []),
    )

    if stages.run_terminal(
        "stage_4_missing_evidence_answer",
        _check_missing_evidence_stage,
    ):
        return stages.context
    ctx = stages.context

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
    ctx = stages.run("stage_4_summarize_data", summarizer.summarize_data)
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
    ctx = stages.run("stage_5_chart_build", chart_pipeline.build_chart)
    _trace_stage("stage_5_chart_build", t_stage, chart_type=ctx.chart_type or "")
    log.info("Stage 5 complete | chart_type=%s", ctx.chart_type)

    # P4.4 (H12): a grounded answer produced from evidence is the data_answer
    # terminal outcome (unless a Stage-4 fallback already set a distinct one).
    if not ctx.terminal_outcome:
        ctx.terminal_outcome = TerminalOutcome.DATA_ANSWER.value
        metrics.log_terminal_outcome(TerminalOutcome.DATA_ANSWER.value)

    return ctx

@wraps(_process_query_impl)
def process_query(
    query: str,
    conversation_history=None,
    trace_id: str = "",
    session_id: str = "",
    previous_contract_snapshot: str = "",
    request_deadline=None,
    actor_id: str = "",
    request_id: str = "",
) -> QueryContext:
    """Bind the trusted request identity/deadline around every deep I/O call."""
    with bind_request_execution_scope(
        deadline=request_deadline,
        request_id=request_id,
        actor_id=actor_id,
    ):
        return _process_query_impl(
            query,
            conversation_history=conversation_history,
            trace_id=trace_id,
            session_id=session_id,
            previous_contract_snapshot=previous_contract_snapshot,
            request_deadline=request_deadline,
            actor_id=actor_id,
            request_id=request_id,
        )
