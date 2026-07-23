"""
Pipeline Stage 4: LLM Summarization

Generates natural language answers — either from domain knowledge (conceptual)
or from SQL query results (data summarization).
"""
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from agent.analyzer import _extract_forecast_horizon
from agent.generic_renderer import render as generic_render
from agent.render_fitness import evaluate_render_fitness
from agent.summary_grounding import (
    _MAX_NORMALIZED_NUMBER_TOKEN_CHARS,
    _MAX_REFS_PER_CLAIM,
    GroundingComparison,
    _add_aggregate_tokens,
    _add_decimal_rounding_variants,
    _add_evidence_record_tokens,
    _add_join_provenance_tokens,
    _add_rounded_source_variants,
    _apply_absence_claim_guardrail,
    _attach_claim_provenance,
    _build_claim_provenance,
    _build_grounding_corpus,
    _build_grounding_tokens,
    _build_grounding_tokens_candidate,
    _build_row_context,
    _canonicalize_finite_decimal,
    _derive_claims_from_text,
    _enforce_provenance_gate,
    _expand_text_number_token,
    _extract_number_tokens,
    _has_unsupported_absence_claims,
    _hashed_number_token,
    _is_ratio_column,
    _is_year_like_integer,
    _normalize_number_token,
    _normalized_decimal_token,
    _rounded_match_variants,
    _serialize_scalar,
    _tokenize_cell_value,
    compare_grounding_policies,
)

# Deterministic Stage-4 sources subject to the shadow fitness check (§3.9).
# Keep in sync with agent/answer_provenance.py's path map.
_DETERMINISTIC_SUMMARY_SOURCES = {
    "generic_renderer",
    "deterministic_share_summary",
    "deterministic_regulated_tariff_list_direct",
    "deterministic_residual_weighted_price_direct",
}
from config import PIPELINE_MODE
from context import scrub_schema_mentions, strip_inline_citation_markers
from contracts.evidence_frames import EntitySetFrame, ForecastFrame, ObservationFrame, ScenarioFrame
from contracts.question_analysis import AnswerKind, RenderStyle
from core.llm import (
    SummaryEnvelope,
    classify_query_type,
    get_query_focus,
    get_relevant_domain_knowledge,
    llm_summarize,
    llm_summarize_structured,
)
from models import GroundingPolicy, QueryContext, ResolutionPolicy, TerminalOutcome
from utils.language import get_evidence_unavailable_message, get_grounding_fallback_message
from utils.metrics import metrics
from utils.share_thresholds import normalize_share_threshold
from utils.trace_logging import trace_detail

log = logging.getLogger("Enai")

# The sign only counts when NOT directly preceded by a digit: in range prose
# ("2020-2025", "60-65%") and ISO dates ("2024-01-15") the hyphen is a
# separator, and parsing it as a minus minted ungrounded negative tokens
# ('-2025', '-65') that failed the provenance gate (2026-07-10 report,
# trace 872ca85f). Genuine negatives ("-1.4", "(-0.77)") keep their sign.
# Scientific notation is one token so values such as 1e9 cannot degrade into
# individually ignored digits.
_SUMMARY_ENERGY_DOMAIN_FOCUSES = {"balancing", "generation", "trade", "energy_security"}
_REGULATED_TARIFF_ALIAS_COLUMN_PREFIXES = {
    "regulated_hpp": ("regulated_hpp_tariff_",),
    "regulated_new_tpp": ("regulated_new_tpp_tariff_", "gardabani_tpp_tariff_"),
    "regulated_old_tpp": ("regulated_old_tpp_tariff_", "grouped_old_tpp_tariff_"),
    "enguri_hpp": ("enguri_hpp_tariff_",),
    "enguri": ("enguri_tariff_",),
    "vardnili_hpp": ("vardnili_hpp_tariff_",),
    "vardnili": ("vardnili_tariff_",),
    "dzevrula_hpp": ("dzevrula_hpp_tariff_",),
    "dzevruli_hpp": ("dzevruli_hpp_tariff_",),
    "gumati_hpp": ("gumati_hpp_tariff_",),
    "shaori_hpp": ("shaori_hpp_tariff_",),
    "rioni_hpp": ("rioni_hpp_tariff_",),
    "lajanuri_hpp": ("lajanuri_hpp_tariff_",),
    "zhinvali_hpp": ("zhinvali_hpp_tariff_",),
    "vartsikhe_hpp": ("vartsikhe_hpp_tariff_",),
    "khramhesi_i": ("khramhesi_i_tariff_",),
    "khramhesi_ii": ("khramhesi_ii_tariff_",),
    "gardabani_tpp": ("gardabani_tpp_tariff_",),
    "mtkvari_tpp": ("mtkvari_tpp_tariff_",),
    "mktvari_tpp": ("mktvari_tpp_tariff_",),
    "tbilresi_tpp": ("tbilresi_tpp_tariff_",),
    "tbilsresi_tpp": ("tbilsresi_tpp_tariff_",),
    "gpower_tpp": ("gpower_tpp_tariff_",),
    "old_tpp_group": ("grouped_old_tpp_tariff_",),
}
_REGULATED_TARIFF_ALIAS_GROUP_LABELS = {
    "regulated_hpp": "Regulated HPPs",
    "regulated_new_tpp": "Regulated new TPP",
    "regulated_old_tpp": "Regulated old TPPs",
    "enguri_hpp": "Enguri HPP",
    "enguri": "Enguri HPP",
    "vardnili_hpp": "Vardnili HPP",
    "vardnili": "Vardnili HPP",
    "dzevrula_hpp": "Dzevruli HPP",
    "dzevruli_hpp": "Dzevruli HPP",
    "gumati_hpp": "Gumati HPP",
    "shaori_hpp": "Shaori HPP",
    "rioni_hpp": "Rioni HPP",
    "lajanuri_hpp": "Lajanuri HPP",
    "zhinvali_hpp": "Zhinvali HPP",
    "vartsikhe_hpp": "Vartsikhe HPP",
    "khramhesi_i": "Khrami I HPP",
    "khramhesi_ii": "Khrami II HPP",
    "gardabani_tpp": "Gardabani TPP",
    "mtkvari_tpp": "Mtkvari Energy",
    "mktvari_tpp": "Mtkvari Energy",
    "tbilresi_tpp": "Tbilisi TPP",
    "tbilsresi_tpp": "Tbilisi TPP",
    "gpower_tpp": "G-POWER",
    "old_tpp_group": "Old TPP Group",
}
_AGGREGATE_TARIFF_LIST_ALIASES = frozenset({
    "regulated_hpp",
    "regulated_new_tpp",
    "regulated_old_tpp",
    "regulated_plants",
    "old_tpp_group",
})
_REGULATED_TARIFF_ENTITY_DISPLAY_NAMES = {
    "enguri hpp": "Enguri HPP",
    "vardnili hpp": "Vardnili HPP",
    "dzevrula hpp": "Dzevruli HPP",
    "gumati hpp": "Gumati HPP",
    "shaori hpp": "Shaori HPP",
    "rioni hpp": "Rioni HPP",
    "lajanuri hpp": "Lajanuri HPP",
    "zhinvali hpp": "Zhinvali HPP",
    "vartsikhe hpp": "Vartsikhe HPP",
    "khramhesi I": "Khrami I HPP",
    "khramhesi II": "Khrami II HPP",
    "gardabani tpp": "Gardabani TPP",
    "gpower tpp": "G-POWER",
    "mktvari tpp": "Mtkvari Energy",
    "tbilsresi tpp": "Tbilisi TPP",
}
_TOTAL_INCOME_QUERY_SIGNALS = (
    "total income",
    "total revenue",
    "combined income",
    "combined revenue",
    "overall income",
    "overall revenue",
    "market sale",
    "market sales",
    "sell",
    "sales",
)
_CFD_COMPONENT_QUERY_SIGNALS = (
    "cfd",
    "compensation",
    "payoff",
)
_MIXED_EVIDENCE_QUERY_TYPES = {"forecast", "comparison"}
# answer_kind values whose summaries inherently contain derived values
# (MoM changes, percentages, projections, scenario payoffs) that strict
# numeric grounding rejects as false positives.
_ANALYTICAL_ANSWER_KINDS = frozenset({
    AnswerKind.EXPLANATION,
    AnswerKind.FORECAST,
    AnswerKind.SCENARIO,
    AnswerKind.COMPARISON,
})

# Fix F (2026-05-17): the DATA-SHAPE RULE in the summarizer prompt
# (see core/llm.py:_data_shape_rule) explicitly instructs the LLM to
# disclose when a user-asked category has no dedicated column — e.g.
# "wind is grouped under renewable_ppa" / "no pure-wind column exists".
# Q2 production retest 5a00ee06 (2026-05-17) showed the LLM following
# the rule correctly: it mapped "small hydro" → ``price_regulated_hpp_*``,
# "thermal" → ``price_regulated_*_tpp_*``, and honestly said the data
# does not contain a dedicated wind column. The absence-claim guardrail
# pattern ``\bno\b ... \brecorded|available|present\b`` matched the
# legitimate disclaimer and replaced the entire useful answer with a
# generic refusal. This whitelist skips the guardrail when the summary
# shows clear signals of transparent equivalence-mapping (referencing
# the DATA-SHAPE RULE, citing specific data column names with
# backticks, or using mapping/proxy language).
# Detects column-name citations like ``price_regulated_hpp_gel`` or
# ``share_thermal_ppa`` wrapped in backticks — strong signal that the
# LLM is doing transparent column-level mapping rather than asserting
# unsupported absence.


# Numeric grounding helpers normalize every number before provenance matching.
















# Grounding-token builders widen the evidence space to include aggregates and derived metrics.




# ---------------------------------------------------------------------------
# Grounding shadow harness (audit Phase 2/7). SHADOW-ONLY: nothing below is
# wired to the live grounding gate. It builds a *candidate* grounding corpus
# that scopes the ratio->percent (x100) token expansion to genuine ratio/share
# columns, plus a comparator that reports where the candidate would disagree
# with the current "x100 for every abs<=1 cell" policy. Purpose: gather
# disagreement evidence before deciding any cutover (docs/architecture-audit.md
# P1-7). Keep this together with the grounding code when grounding.py is split.
# ---------------------------------------------------------------------------

# Column names whose [0,1] values are genuinely ratios/shares and legitimately
# render as percentages (share_import, import_dependency_ratio, *_percent, ...).






# Off by default: enabling GROUNDING_SHADOW_LOG makes the live gate additionally
# log (never act on) candidate-vs-current disagreements over real traffic.
_GROUNDING_SHADOW_LOG = os.getenv("GROUNDING_SHADOW_LOG", "").strip().lower() in {"1", "true", "yes", "on"}


def _maybe_log_grounding_shadow(envelope: SummaryEnvelope, ctx: QueryContext) -> None:
    """Shadow-log where the candidate ratio-scoping policy would disagree with the
    live grounding decision. Never changes the gate; see docs/architecture-audit.md
    P1-7 for the disagreement-review cutover process.
    """
    if not _GROUNDING_SHADOW_LOG:
        return
    try:
        comparison = compare_grounding_policies(envelope, ctx)
    except Exception:  # pragma: no cover - shadow must never break the gate
        return
    if comparison.disagree:
        log.warning(
            "🕵️ grounding shadow DISAGREE: live_passed=%s candidate_passed=%s "
            "(live_ratio=%.2f candidate_ratio=%.2f threshold=%.2f) removed_tokens=%s",
            comparison.current_passed,
            comparison.candidate_passed,
            comparison.current_ratio,
            comparison.candidate_ratio,
            comparison.threshold,
            comparison.divergent_tokens[:20],
        )


def _is_summary_grounded(envelope: SummaryEnvelope, ctx: QueryContext) -> bool:
    # Fix D (2026-05-17): Q3 production trace 9b9b28b4 ran into a silent
    # threshold-comparison bug. ``GroundingPolicy(str, Enum)`` — in
    # Python 3.11+ ``str(StrEnum.MEMBER)`` returns the FQN
    # ``"GroundingPolicy.EVIDENCE_AWARE"`` instead of the value
    # ``"evidence_aware"``, so the old ``grounding_policy == GroundingPolicy.X``
    # comparisons silently mismatched and EVIDENCE_AWARE fell through to
    # the 0.9 STRICT_NUMERIC threshold. Q3's 0.78 ratio would have passed
    # under the intended 0.7 EVIDENCE_AWARE threshold but instead got
    # rejected, producing the conservative fallback answer.
    # Fix: normalise to the enum's ``.value`` (a stable lowercase string)
    # and compare in string-space throughout.
    raw_policy = ctx.grounding_policy or GroundingPolicy.STRICT_NUMERIC
    grounding_policy = getattr(raw_policy, "value", str(raw_policy))
    if grounding_policy == GroundingPolicy.NOT_APPLICABLE.value:
        return True

    claim_text = "\n".join(envelope.claims or [])
    answer_tokens = _extract_number_tokens((envelope.answer or "") + "\n" + claim_text)
    if not answer_tokens:
        return True

    source_tokens = _build_grounding_tokens(ctx)
    if not source_tokens:
        log.warning(
            "🔬 Grounding fail (no source tokens): answer has %d number tokens but evidence "
            "corpus is empty. policy=%s",
            len(answer_tokens),
            grounding_policy,
        )
        return False

    matched = sum(1 for t in answer_tokens if t in source_tokens)
    match_ratio = matched / max(1, len(answer_tokens))
    # Evidence-aware queries produce derived values (percentages, deltas)
    # that legitimately extend beyond raw data; use a lower threshold.
    # Comparison normalised to ``.value`` to fix the Python-3.11+ StrEnum
    # FQN-stringification bug (see top-of-function comment).
    min_ratio = 0.7 if grounding_policy == GroundingPolicy.EVIDENCE_AWARE.value else 0.9
    passed = match_ratio >= min_ratio
    if not passed:
        # Diagnostic: when grounding fails, log the matched vs missing tokens
        # so we can distinguish hallucination from derivation from rounding.
        # Triggered only on failures (not on every pass) to keep logs quiet.
        # The first 20 missing tokens (sorted) usually reveal the pattern —
        # rounded values, percentages computed from ratios, hallucinated
        # absolute numbers, or year/date strings outside the data window.
        missing = [t for t in answer_tokens if t not in source_tokens]
        log.warning(
            "🔬 Grounding fail: matched %d/%d tokens (ratio=%.2f, threshold=%.2f, policy=%s). "
            "missing_token_count=%d",
            matched,
            len(answer_tokens),
            match_ratio,
            min_ratio,
            grounding_policy,
            len(missing),
        )
    _maybe_log_grounding_shadow(envelope, ctx)
    return passed








def _merge_topic_preferences(*topic_lists: Optional[List[str]]) -> Optional[List[str]]:
    merged: List[str] = []
    seen: Set[str] = set()
    for topic_list in topic_lists:
        for topic in topic_list or []:
            normalized = str(topic or "").strip()
            if not normalized or normalized in seen:
                continue
            merged.append(normalized)
            seen.add(normalized)
    return merged or None
























def _extract_preferred_topics(ctx: QueryContext) -> Optional[List[str]]:
    """Extract preferred topic names from question analysis if active.

    Returns the top 3 topics with score >= 0.25, or None if the analyzer
    is not active.  Used by both answer_conceptual() and summarize_data()
    to drive domain-knowledge selection via the analyzer instead of the
    keyword-based TOPIC_MAP.
    """
    if (
        ctx.question_analysis is not None
        and ctx.question_analysis_source == "llm_active"
    ):
        ranked = sorted(
            ctx.question_analysis.knowledge.candidate_topics,
            key=lambda c: c.score,
            reverse=True,
        )
        topics = [c.name.value for c in ranked if c.score >= 0.25][:3]
        return topics or None
    return None


def _extract_vector_preferred_topics(ctx: QueryContext) -> Optional[List[str]]:
    """Extract preferred topics from active vector retrieval results."""

    if ctx.vector_knowledge is None or ctx.vector_knowledge_source != "vector_active":
        return None
    preferred = list(ctx.vector_knowledge.filters.preferred_topics or [])
    return preferred[:5] or None


# Topic and policy helpers decide how much non-tabular evidence Stage 4 may rely on.
def _build_data_summary_topic_preferences(ctx: QueryContext) -> Optional[List[str]]:
    """Merge analyzer and vector topic preferences for data summaries."""

    return _merge_topic_preferences(
        _extract_vector_preferred_topics(ctx),
        _extract_preferred_topics(ctx),
    )


def _resolve_summary_answer_kind(ctx: QueryContext) -> Optional[AnswerKind]:
    """Return the authoritative answer_kind for Stage 4 policy decisions."""

    if (
        ctx.question_analysis is not None
        and ctx.question_analysis_source == "llm_active"
        and ctx.question_analysis.answer_kind is not None
    ):
        return ctx.question_analysis.answer_kind
    return getattr(ctx, "effective_answer_kind", None)


def _select_data_summary_domain_topics(
    ctx: QueryContext,
    answer_kind: Optional[AnswerKind],
) -> Optional[List[str]]:
    """Choose the narrowest topic slice that still matches the answer shape."""

    if answer_kind in (AnswerKind.FORECAST, AnswerKind.SCENARIO):
        return ["seasonal_patterns"]
    return _build_data_summary_topic_preferences(ctx)


def _should_load_data_summary_domain_knowledge(
    ctx: QueryContext,
    *,
    routing_query: str,
    answer_kind: Optional[AnswerKind],
    query_type: str,
    render_deterministic: bool,
) -> bool:
    """Apply the Chapter 14.7 Stage 4 domain-knowledge policy."""

    if PIPELINE_MODE == "fast" or render_deterministic:
        return False
    if ctx.resolution_policy == ResolutionPolicy.CLARIFY:
        return False
    if answer_kind == AnswerKind.CLARIFY:
        return False
    if answer_kind in (AnswerKind.KNOWLEDGE, AnswerKind.EXPLANATION):
        return True
    if answer_kind in (AnswerKind.FORECAST, AnswerKind.SCENARIO):
        return True
    if answer_kind in (AnswerKind.SCALAR, AnswerKind.LIST):
        return False
    if query_type in ("single_value", "list"):
        return False

    focus = get_query_focus(routing_query)
    return focus in _SUMMARY_ENERGY_DOMAIN_FOCUSES or bool(_build_data_summary_topic_preferences(ctx))


def _derive_data_summary_grounding_policy(ctx: QueryContext, query_type: str) -> str:
    """Choose the appropriate grounding policy for Stage 4 summaries."""

    if ctx.resolution_policy == ResolutionPolicy.CLARIFY:
        return GroundingPolicy.NOT_APPLICABLE
    if ctx.response_mode == "knowledge_primary":
        return GroundingPolicy.NOT_APPLICABLE
    analyzer_active = (
        ctx.question_analysis is not None
        and ctx.question_analysis_source == "llm_active"
    )
    needs_knowledge = bool(
        analyzer_active
        and getattr(ctx.question_analysis.routing, "needs_knowledge", False)
    )
    needs_driver_analysis = bool(
        analyzer_active
        and getattr(ctx.question_analysis.analysis_requirements, "needs_driver_analysis", False)
    )
    has_non_tabular_evidence = bool(
        ctx.vector_knowledge_prompt
        and ctx.vector_knowledge_source == "vector_active"
    )
    has_derived_metrics = bool(
        analyzer_active
        and ctx.question_analysis.analysis_requirements.derived_metrics
    )
    # When domain knowledge is injected into the prompt, the answer is expected
    # to weave in domain figures (e.g. support-scheme USD prices) that are not in
    # the data corpus — strict numeric grounding over-rejects them. Must be read
    # AFTER ctx.summary_domain_knowledge is set by the caller (see summarize_data).
    has_domain_knowledge = bool((getattr(ctx, "summary_domain_knowledge", "") or "").strip())
    # answer_kind-aware: analytical answer shapes produce derived values
    # (MoM deltas, percentages, projections) that strict grounding rejects.
    answer_kind = (
        ctx.question_analysis.answer_kind
        if analyzer_active
        else None
    )
    if (
        answer_kind in _ANALYTICAL_ANSWER_KINDS
        and (
            needs_knowledge or needs_driver_analysis or has_non_tabular_evidence
            or has_derived_metrics or has_domain_knowledge
        )
    ):
        return GroundingPolicy.EVIDENCE_AWARE
    # Legacy query_type fallback when answer_kind is unavailable.
    if (
        query_type in _MIXED_EVIDENCE_QUERY_TYPES
        and has_non_tabular_evidence
        and (needs_knowledge or needs_driver_analysis)
    ):
        return GroundingPolicy.EVIDENCE_AWARE
    return GroundingPolicy.STRICT_NUMERIC


def _should_use_comparison_first_guidance(ctx: QueryContext, query_type: str) -> bool:
    """Return True when Stage 4 should frame the answer as a period comparison first."""

    if query_type != "data_explanation":
        return False

    requested = list(ctx.requested_derived_metrics or [])
    if not requested and ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active":
        requested = [
            getattr(metric.metric_name, "value", str(metric.metric_name or "")).strip()
            for metric in (ctx.question_analysis.analysis_requirements.derived_metrics or [])
            if getattr(metric.metric_name, "value", str(metric.metric_name or "")).strip()
        ]

    return any(name.startswith(("mom_", "yoy_")) for name in requested)


def _build_clarification_options(ctx: QueryContext) -> List[str]:
    """Build a concise set of clarification options for ambiguous queries."""

    query_lower = (ctx.query or "").strip().lower()
    clarify_reason = str(ctx.clarify_reason or "")
    options: List[str] = []

    def _add(option: str) -> None:
        if option not in options:
            options.append(option)

    if clarify_reason == "request_not_supported_as_phrased":
        _add("Restate the request as a direct data retrieval or calculation with the exact entity groups, periods, and formula you want.")
        _add("Ask for the observable monthly shares and balancing prices first, then request the calculation on top of those results.")
        _add("If you want a derived proxy, state the exact proxy formula so I can calculate only what the available data supports.")
        return options[:3]

    if clarify_reason == "underdefined_computed_target":
        if any(signal in query_lower for signal in ("remaining", "residual", "excluding", "except")):
            _add("Treat remaining energy as the residual after excluding regulated hydro, regulated thermals, and deregulated renewable.")
            _add("Treat remaining energy as the existing PPA/CfD/import residual layer.")
            _add("List the exact components that should be included in the remaining-energy bucket, and I will calculate it for the requested months.")
            return options[:3]

    if any(signal in query_lower for signal in ("trend", "history", "historical", "past", "change")):
        _add("Summarize the historical trend in observed electricity prices in Georgia.")
    if any(signal in query_lower for signal in ("target model", "market model", "reform", "liberalization")):
        _add("Explain how price formation is expected to work under the target market model.")
    if any(signal in query_lower for signal in ("factor", "factors", "influencing", "driver", "drivers", "cause", "why")):
        _add("Explain the main factors that have influenced price changes in the historical data.")

    if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active":
        query_type = ctx.question_analysis.classification.query_type.value
        if query_type == "forecast":
            _add("Give a cautious forward-looking view based on the available regulatory context and recent data.")
        elif query_type == "comparison":
            _add("Compare the current market model with the target model rather than summarizing historical prices.")

    _add("Answer with the closest grounded interpretation the current evidence supports.")
    return options[:3]


def answer_clarify(ctx: QueryContext) -> QueryContext:
    """Ask the user to pick the intended interpretation instead of guessing."""

    options = _build_clarification_options(ctx)
    option_lines = [f"{idx}. {option}" for idx, option in enumerate(options, 1)]
    clarify_reason = str(ctx.clarify_reason or "the request can be interpreted in more than one valid way")
    ctx.summary = (
        "I can answer this, but I want to make sure we take the right interpretation first.\n\n"
        f"Reason: {clarify_reason.replace('_', ' ')}.\n\n"
        "Please choose one of these directions:\n"
        f"{chr(10).join(option_lines)}\n\n"
        "Reply with the option number, or restate the question in the direction you want."
    )
    ctx.summary_source = "clarification_request"
    ctx.summary_claims = []
    ctx.summary_citations = ["clarification_request"]
    ctx.summary_confidence = 0.85
    ctx.summary_claim_provenance = []
    ctx.summary_provenance_coverage = 0.0
    ctx.summary_provenance_gate_passed = True
    ctx.summary_provenance_gate_reason = "not_applicable_clarify"
    ctx.grounding_policy = GroundingPolicy.NOT_APPLICABLE
    ctx.summary_domain_knowledge = ""
    ctx.summary = strip_inline_citation_markers(scrub_schema_mentions(ctx.summary))
    return ctx


# Clarification/conceptual answers bypass numeric grounding and cite background knowledge only.
def answer_evidence_unavailable(ctx: QueryContext) -> QueryContext:
    """Terminal degraded answer for a data request whose evidence is unavailable.

    P4.4 (finding H12): when a data-primary request's generated SQL fails
    validation or relevance, the pipeline used to call ``answer_conceptual``,
    dressing an unavailable dataset as a plausible domain narrative. This
    deterministic path instead states the limitation honestly and, because the
    message is a fixed template rather than LLM output, structurally cannot
    invent numeric claims. Anti-retry-storm behavior is preserved: this is a
    normal HTTP 200 answer, not a 4xx that would trigger client retries.

    Reads: ctx.lang_code, ctx.skip_sql_reason
    Writes: ctx.summary, summary_source, summary_claims, grounding/gate state,
            ctx.terminal_outcome.
    """
    ctx.grounding_policy = GroundingPolicy.NOT_APPLICABLE
    ctx.summary_domain_knowledge = ""
    ctx.summary = get_evidence_unavailable_message(getattr(ctx, "lang_code", "") or "en")
    ctx.summary_source = "evidence_unavailable"
    ctx.summary_claims = []
    ctx.summary_citations = ["evidence_unavailable"]
    ctx.summary_confidence = 0.2
    ctx.summary_claim_provenance = []
    ctx.summary_provenance_coverage = 0.0
    ctx.summary_provenance_gate_passed = True
    ctx.summary_provenance_gate_reason = "not_applicable_evidence_unavailable"
    ctx.terminal_outcome = TerminalOutcome.EVIDENCE_UNAVAILABLE.value
    metrics.log_terminal_outcome(TerminalOutcome.EVIDENCE_UNAVAILABLE.value)
    log.info(
        "Terminal outcome: evidence_unavailable (data-primary request, reason=%s)",
        ctx.skip_sql_reason or "unspecified",
    )
    return ctx


def answer_conceptual(ctx: QueryContext) -> QueryContext:
    """Generate an answer for conceptual/definitional questions (no SQL).

    Reads: ctx.query, ctx.lang_instruction, ctx.conversation_history
    Writes: ctx.summary
    """
    analyzer_active = ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active"
    ctx.grounding_policy = GroundingPolicy.NOT_APPLICABLE
    ctx.summary_domain_knowledge = ""
    routing_query = (
        ctx.question_analysis.canonical_query_en
        if analyzer_active and ctx.question_analysis is not None
        else ctx.query
    )
    preferred_topics = _extract_preferred_topics(ctx)
    vector_evidence_active = bool(
        ctx.vector_knowledge is not None
        and ctx.vector_knowledge_source == "vector_active"
        and ctx.vector_knowledge.chunk_count > 0
    )
    if PIPELINE_MODE == "fast":
        vector_evidence_active = False
    vector_preferred_topics = (
        list(ctx.vector_knowledge.filters.preferred_topics)
        if vector_evidence_active and ctx.vector_knowledge is not None
        else []
    )
    domain_background_topics = _merge_topic_preferences(vector_preferred_topics, preferred_topics)
    query_lower = routing_query.lower()
    domain_knowledge = ""
    if PIPELINE_MODE != "fast":
        domain_knowledge = get_relevant_domain_knowledge(
            routing_query,
            use_cache=False,
            preferred_topics=domain_background_topics if vector_evidence_active else preferred_topics,
        )
    if analyzer_active and preferred_topics:
        log.info("Using active question-analyzer topics for conceptual answer: %s", preferred_topics)
    if vector_evidence_active and vector_preferred_topics:
        log.info("Using vector-aware background topics for conceptual answer: %s", domain_background_topics)
    try:
        knowledge_payload = json.loads(domain_knowledge) if domain_knowledge else {}
    except json.JSONDecodeError:
        knowledge_payload = {}
    matched_topics = list(knowledge_payload.keys()) if isinstance(knowledge_payload, dict) else []
    has_filtered_knowledge = bool(matched_topics)
    has_domain_specific_knowledge = any(topic != "general_definitions" for topic in matched_topics)

    # General energy terms
    general_terms = [
        "renewable energy", "განახლებადი ენერგია", "возобновляемая энергия",
        "electricity market", "ელექტროენერგიის ბაზარი", "рынок электроэнергии",
        "balancing market", "საბალანსო ბაზარი", "балансирующий рынок",
        "tariff", "ტარიფი", "тариф",
        "ppa", "power purchase agreement",
        "cfd", "contract for difference",
        "hydropower", "ჰიდროენერგია", "гидроэнергия",
        "thermal power", "თერმული ენერგია", "тепловая энергия",
        "import", "export", "იმპორტი", "ექსპორტი",
        "demand", "მოთხოვნა", "generation mix", "გენერაციის სტრუქტურა",
        "capacity", "სიმძლავრე", "regulated", "deregulated",
        "exchange rate", "გაცვლითი კურსი", "обменный курс"
    ]

    domain_terms = [
        "enguri", "vardnili", "gardabani", "gnerc", "esco", "gse",
        "ენგური", "ვარდნილი", "გარდაბანი",
        "საქართველო", "georgia"
    ]

    is_general_question = any(term in query_lower for term in general_terms)
    is_domain_specific = any(term in query_lower for term in domain_terms) or has_domain_specific_knowledge

    if is_general_question and not is_domain_specific:
        conceptual_hint = (
            "NOTE: This is a GENERAL conceptual/definitional question about energy terminology. "
            "No database query was executed. "
            "\n\n"
            "RESPONSE FORMAT (MANDATORY):\n"
            "1. **General Definition**: Start with a clear, universal definition of the concept "
            "(2-3 sentences explaining what it is, how it works generally).\n"
            "2. **Georgia Context**: Then provide Georgia-specific context showing how this concept "
            "applies in the Georgian electricity market (2-3 sentences).\n"
            "\n"
            "Use the GeneralDefinitions section from domain knowledge if the term is defined there. "
            "Structure your answer with these two clear sections."
        )
        log.info("📖 General conceptual question - will provide definition + Georgia context")
    elif is_domain_specific or has_filtered_knowledge:
        conceptual_hint = (
            "NOTE: This is a domain-specific conceptual question about the Georgian electricity market. "
            "No database query was executed. "
            "Answer using the provided domain knowledge about Georgia's energy sector. "
            "If the concept appears in the domain knowledge, define it directly from that material."
        )
        log.info("🇬🇪 Domain-specific conceptual question - will use Georgia domain knowledge")
    else:
        conceptual_hint = (
            "NOTE: This is a conceptual/definitional question. "
            "No database query was executed. "
            "\n\n"
            "If this topic is covered in domain knowledge, provide a clear explanation. "
            "If NOT covered, acknowledge the limitation: "
            "'This specific topic is not currently in my domain knowledge base. "
            "For accurate information, I recommend consulting official sources.' "
            "Then provide what general context you can."
        )
        log.info("❓ Conceptual question - topic may be outside domain scope")

    if vector_evidence_active:
        conceptual_hint = (
            "NOTE: This question is answered using both official regulations and curated market insights. "
            "No database query was executed.\n\n"
            "SOURCE INTEGRATION RULES:\n"
            "- EXTERNAL_SOURCE_PASSAGES contain primary evidence from official regulations (best for procedural rules and registration steps).\n"
            "- DOMAIN_KNOWLEDGE contains curated market context and analytical insights (best for factors, drivers, and practical trends).\n"
            "- Synthesize the final answer by integrating both sources. If they cover different aspects (e.g., regulatory mechanics vs. real-world factors), explain both clearly.\n"
            "- Use DOMAIN_KNOWLEDGE as a peer source to provide a complete analytical picture alongside the regulatory requirements."
        )
        log.info(
            "Active vector evidence present for conceptual answer; synthesizing peer domain knowledge"
        )

    # Retrieval-failure resilience: when vector retrieval errored (e.g. the
    # embeddings API is down) the official-document passages are MISSING, not
    # proven absent. Without this the model answers "the provided sources do not
    # contain..." even though the regulations were ingested (prod traces
    # d62c2134 / b0aef6fd, where stage_0_3 returned a 400).
    if getattr(ctx, "vector_knowledge_error", ""):
        conceptual_hint += (
            "\n\nIMPORTANT — RETRIEVAL UNAVAILABLE: Retrieval of official "
            "document/regulation passages FAILED for this query due to a technical "
            "error, so any such passages are MISSING (not proven absent). Do NOT "
            "state that the information does not exist or is not in the sources. "
            "Answer from DOMAIN_KNOWLEDGE for whatever it covers; for anything it "
            "does not cover, say the official documents could not be retrieved right "
            "now and suggest retrying shortly."
        )
        log.info("Conceptual answer: vector retrieval unavailable — added do-not-claim-absence caveat")

    vector_knowledge = ctx.vector_knowledge_prompt if vector_evidence_active and PIPELINE_MODE != "fast" else ""
    domain_knowledge_for_summary = domain_knowledge
    if vector_evidence_active:
        # When vector evidence is present, the inline domain-knowledge layer
        # is treated as a "peer source" and capped to avoid prompt bloat.
        # Default 12000 chars; configurable via env so deployments with
        # expanded inline knowledge (e.g. detailed Articles-13/14/36 rule
        # sets in balancing_price.md) can lift the cap without a code change.
        # See 2026-05-15 trace 1b132a9b: at 12000 cap, sections F (LLM
        # disambiguation rules + mandatory completeness checklist) got
        # truncated, producing a shallow answer despite full retrieval.
        _dk_cap_raw = os.getenv("DOMAIN_KNOWLEDGE_MAX_CHARS_WITH_VECTOR", "12000").strip()
        try:
            _dk_cap = max(1000, int(_dk_cap_raw))
        except ValueError:
            _dk_cap = 12000
        if len(domain_knowledge_for_summary) > _dk_cap:
            domain_knowledge_for_summary = domain_knowledge_for_summary[:_dk_cap]
            log.info(
                "Capped domain_knowledge to %d chars (synthesizing with vector evidence)",
                _dk_cap,
            )
    # Diagnostic: log prompt composition BEFORE the LLM call so we can correlate
    # input volume / shape with output quality.  Critical for debugging
    # "answer is shallow despite all rescue layers" cases like trace 226a56ef
    # (2026-05-15) — without this we can't tell whether the LLM saw the full
    # comprehensive content or whether something truncated upstream.
    log.info(
        "🔬 LLM input composition (conceptual): domain_kb=%d chars, vector_passages=%d chars, "
        "history_turns=%d, response_mode=%s, query_type=%s",
        len(domain_knowledge_for_summary),
        len(vector_knowledge),
        len(ctx.conversation_history or []),
        ctx.response_mode,
        (
            ctx.question_analysis.classification.query_type.value
            if ctx.question_analysis is not None
            else "none"
        ),
    )
    try:
        envelope = llm_summarize_structured(
            ctx.effective_query,
            data_preview="",
            stats_hint=conceptual_hint,
            lang_instruction=ctx.lang_instruction,
            conversation_history=ctx.conversation_history,
            domain_knowledge=domain_knowledge_for_summary,
            vector_knowledge=vector_knowledge,
            question_analysis=ctx.question_analysis,
            effective_answer_kind=getattr(ctx, "effective_answer_kind", None),
            vector_knowledge_bundle=ctx.vector_knowledge,
            response_mode=ctx.response_mode,
        )
        ctx.summary = envelope.answer
        ctx.summary_source = "structured_conceptual_summary"
        ctx.summary_claims = list(envelope.claims)
        ctx.summary_citations = list(envelope.citations)
        ctx.summary_confidence = float(envelope.confidence)
        ctx.summary_claim_provenance = []
        ctx.summary_provenance_coverage = 0.0
        ctx.summary_provenance_gate_passed = True
        ctx.summary_provenance_gate_reason = "not_applicable_conceptual"
        # Diagnostic: log answer shape so we can correlate output length /
        # structure with input composition.  ``summary_preview`` is the first
        # 600 chars verbatim — enough to see whether the LLM enumerated
        # sub-points, cited articles, or just gave a brief summary.
        trace_detail(
            log, ctx, "stage_4_conceptual_summary", "answer_generated",
            summary_length=len(envelope.answer),
            summary_preview=envelope.answer[:600],
            citations=list(envelope.citations),
            claims_count=len(envelope.claims),
            confidence=float(envelope.confidence),
            summary_source="structured_conceptual_summary",
        )
    except Exception as exc:
        # Same repair as the structured data path (incident 2026-07-17): the
        # former `except RuntimeError: raise` breaker sentinel also re-raised
        # F8's ProviderExecutionError here, turning conceptual answers into
        # 500s. All provider failures degrade to the legacy conceptual
        # fallback; a true provider-tier outage re-raises from the call below.
        metrics.log_summary_schema_failure()
        log.warning(
            "Structured conceptual summarization failed (%s): %s",
            type(exc).__name__, exc,
        )
        ctx.summary = llm_summarize(
            ctx.effective_query,
            data_preview="",
            stats_hint=conceptual_hint,
            lang_instruction=ctx.lang_instruction,
            conversation_history=ctx.conversation_history,
            domain_knowledge=domain_knowledge,
            vector_knowledge=vector_knowledge,
        )
        ctx.summary_source = "legacy_conceptual_text_fallback"
        ctx.summary_claims = []
        ctx.summary_citations = ["legacy_text_fallback"]
        ctx.summary_confidence = 0.5
        ctx.summary_claim_provenance = []
        ctx.summary_provenance_coverage = 0.0
        ctx.summary_provenance_gate_passed = True
        ctx.summary_provenance_gate_reason = "not_applicable_conceptual"
    ctx.summary = strip_inline_citation_markers(scrub_schema_mentions(ctx.summary))
    log.info("✅ Conceptual answer generated")
    return ctx

# Deterministic direct answers avoid LLM summarization when the analytical evidence is already complete.
def _build_scenario_fallback_answer(ctx: QueryContext) -> Optional[str]:
    """Build a deterministic answer from scenario evidence when LLM grounding fails.

    Returns None if no scenario evidence is available, allowing the caller
    to fall back to the generic grounding failure message.
    """
    evidence = ctx.analysis_evidence or []
    scenario_records = [r for r in evidence if r.get("record_type") == "scenario"]
    if not scenario_records:
        return None
    rec = scenario_records[0]
    query_lower = (ctx.query or "").strip().lower()

    metric_name = rec.get("derived_metric_name", "")
    factor = rec.get("scenario_factor")
    volume = rec.get("scenario_volume")
    agg_result = rec.get("aggregate_result")
    if agg_result is None:
        return None

    row_count = rec.get("row_count")
    period_range = rec.get("period_range", "")
    min_val = rec.get("min_period_value")
    max_val = rec.get("max_period_value")
    mean_val = rec.get("mean_period_value")
    formula = rec.get("formula", "")

    parts: list[str] = []

    if metric_name == "scenario_payoff":
        positive_sum = rec.get("positive_sum", 0.0)
        negative_sum = rec.get("negative_sum", 0.0)
        positive_count = rec.get("positive_count", 0)
        negative_count = rec.get("negative_count", 0)
        market_component = rec.get("market_component_aggregate")
        combined_total = rec.get("combined_total_aggregate")
        metric_key = str(rec.get("metric") or "").strip()
        market_label = (
            "Balancing market sales income"
            if metric_key in {"p_bal_usd", "p_bal_gel"}
            else "Market sales income at observed prices"
        )
        include_income_breakdown = (
            market_component is not None
            and combined_total is not None
            and (
                any(signal in query_lower for signal in _TOTAL_INCOME_QUERY_SIGNALS)
                and any(signal in query_lower for signal in _CFD_COMPONENT_QUERY_SIGNALS)
            )
        )

        parts.append(
            f"**{'CfD Payoff and Income Analysis' if include_income_breakdown else 'CfD Payoff Analysis'}** "
            f"(strike: {factor} USD/MWh"
            + (f", volume: {volume} MW" if volume is not None else "")
            + ")"
        )
        if period_range:
            parts.append(f"**Period:** {period_range} ({row_count} months)")
        parts.append(f"**Formula:** {formula}")
        if include_income_breakdown:
            parts.append(f"**{market_label}:** {market_component} USD")
            parts.append(f"**CfD financial compensation:** {agg_result} USD")
            parts.append(f"**Total combined income:** {combined_total} USD")
        parts.append(f"**Net total payoff:** {agg_result} USD")
        if positive_sum and positive_sum != 0:
            parts.append(
                f"**Income from favorable periods** (market price below strike): "
                f"{positive_sum} USD across {positive_count} months"
            )
        if negative_sum and negative_sum != 0:
            parts.append(
                f"**Compensation cost in unfavorable periods** (market price above strike): "
                f"{negative_sum} USD across {negative_count} months"
            )
        parts.append(
            f"**Per-period range:** min {min_val} to max {max_val} USD "
            f"(average {mean_val} USD/month)"
        )

    elif metric_name in ("scenario_scale", "scenario_offset"):
        baseline = rec.get("baseline_aggregate")
        delta = rec.get("delta_aggregate")
        delta_pct = rec.get("delta_percent")
        op = "\u00d7" if metric_name == "scenario_scale" else "+"
        parts.append(f"**Scenario Analysis** ({op} {factor})")
        if period_range:
            parts.append(f"**Period:** {period_range} ({row_count} periods)")
        parts.append(f"**Result:** {agg_result}")
        if baseline is not None:
            parts.append(f"**Baseline:** {baseline}")
        if delta is not None:
            parts.append(
                f"**Change:** {delta}"
                + (f" ({delta_pct}%)" if delta_pct is not None else "")
            )
        parts.append(f"**Range:** {min_val} to {max_val} (mean {mean_val})")

    else:
        return None

    return "\n\n".join(parts)


def _tariff_alias_has_observations(df: pd.DataFrame, alias: str) -> bool:
    """Return True when the retrieved tariff frame includes data for an alias."""
    prefixes = _REGULATED_TARIFF_ALIAS_COLUMN_PREFIXES.get(alias, ())
    if not prefixes:
        return False

    for col in df.columns:
        if not any(col.startswith(prefix) for prefix in prefixes):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            return True
    return False


def _build_regulated_tariff_list_direct_answer(ctx: QueryContext) -> str | None:
    """Build a deterministic list of regulated plants from tariff evidence.

    Gated by answer_kind=LIST + tool_name=get_tariffs (no regex).
    """
    if ctx.df.empty:
        return None
    # Structural gate: must be a LIST question using the tariff tool.
    if (ctx.tool_name or "").strip() != "get_tariffs":
        return None
    if (
        ctx.has_authoritative_question_analysis
        and ctx.question_analysis.answer_kind is not None
        and ctx.question_analysis.answer_kind != AnswerKind.LIST
    ):
        return None

    from agent.tools.tariff_tools import resolve_tariff_alias_entities

    selected_aliases = [
        str(entity).strip()
        for entity in (ctx.tool_params.get("entities") or [])
        if str(entity).strip() in _REGULATED_TARIFF_ALIAS_COLUMN_PREFIXES
    ]
    if not selected_aliases:
        selected_aliases = ["regulated_hpp", "regulated_new_tpp", "regulated_old_tpp"]

    active_aliases = [
        alias for alias in selected_aliases if _tariff_alias_has_observations(ctx.df, alias)
    ]
    if not active_aliases:
        return None

    date_col = next((col for col in ctx.df.columns if "date" in col.lower()), None)
    period_line = ""
    if date_col:
        dates = pd.to_datetime(ctx.df[date_col], errors="coerce").dropna()
        if not dates.empty:
            period_line = (
                f"Retrieved tariff coverage: {dates.min().strftime('%B %Y')} to "
                f"{dates.max().strftime('%B %Y')}."
            )

    group_lines: list[str] = []
    as_of_date = None
    if date_col:
        dates = pd.to_datetime(ctx.df[date_col], errors="coerce").dropna()
        if not dates.empty:
            as_of_date = dates.max().date()

    for alias in active_aliases:
        raw_entities = resolve_tariff_alias_entities(alias, as_of=as_of_date)
        display_names = [
            _REGULATED_TARIFF_ENTITY_DISPLAY_NAMES.get(raw_entity, raw_entity)
            for raw_entity in raw_entities
        ]
        if not display_names:
            continue
        group_label = _REGULATED_TARIFF_ALIAS_GROUP_LABELS.get(alias, alias.replace("_", " ").title())
        group_lines.append(f"- {group_label}: {', '.join(display_names)}")

    if not group_lines:
        return None

    parts = ["The power plants under price regulation in the retrieved tariff data are:"]
    if period_line:
        parts.append(period_line)
    parts.append("")
    parts.extend(group_lines)
    return "\n".join(parts)


_RESIDUAL_MONTH_NAME_TO_NUMBER = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


_RESIDUAL_THRESHOLD_RULES: list[tuple[str, str, str]] = [
    (r"(more than|above|over|exceed(?:ed|s|ing)?|greater than)\s+(\d+(?:\.\d+)?)\s*%?", "gt", "exceeded"),
    (r"(at least|not less than|minimum of)\s+(\d+(?:\.\d+)?)\s*%?", "ge", "was at least"),
    (r"(less than|below|under|fewer than)\s+(\d+(?:\.\d+)?)\s*%?", "lt", "was below"),
    (r"(at most|no more than|maximum of)\s+(\d+(?:\.\d+)?)\s*%?", "le", "was at most"),
    (r"(\d+(?:\.\d+)?)\s*%\s+or\s+more", "ge", "was at least"),
    (r"(\d+(?:\.\d+)?)\s*%\s+or\s+less", "le", "was at most"),
]


_RESIDUAL_DIRECT_INTENTS = frozenset({
    "residual_weighted_price_calculation",
    "residual_weighted_price_followup",
    "implied_ppa_cfd_price_approximation",
})
_RESIDUAL_DIRECT_ANSWER_KINDS = frozenset({
    AnswerKind.SCALAR,
    AnswerKind.TIMESERIES,
    AnswerKind.COMPARISON,
})


def _residual_direct_answer_is_authorized(ctx: QueryContext) -> bool:
    """Return True only when Stage 0.2 explicitly authorized the residual shortcut."""
    if not ctx.has_authoritative_question_analysis:
        return False

    qa = ctx.question_analysis
    if qa is None or qa.render_style != RenderStyle.DETERMINISTIC:
        return False
    if qa.answer_kind not in _RESIDUAL_DIRECT_ANSWER_KINDS:
        return False

    classification = getattr(qa, "classification", None)
    intent = str(getattr(classification, "intent", "") or "").strip().lower()
    return intent in _RESIDUAL_DIRECT_INTENTS


# Residual weighted-price helpers turn decomposition outputs into a direct answer when possible.
def _has_explicit_residual_component_query_signal(query: str) -> bool:
    query_lower = (query or "").strip().lower()
    return (
        "renewable ppa" in query_lower
        and "import" in query_lower
        and ("thermal generation ppa" in query_lower or "thermal ppa" in query_lower)
        and "cfd" in query_lower
    )


def _extract_residual_share_threshold(query: str) -> tuple[str, float, str] | None:
    query_lower = (query or "").strip().lower()
    if not query_lower or "%" not in query_lower:
        return None
    if not any(token in query_lower for token in ("share", "composition", "contribute", "contribution")):
        return None

    for pattern, operator, phrase in _RESIDUAL_THRESHOLD_RULES:
        match = re.search(pattern, query_lower)
        if not match:
            continue
        raw_group = match.group(match.lastindex or 1)
        try:
            raw_value = float(raw_group)
        except (TypeError, ValueError):
            continue
        threshold = normalize_share_threshold(raw_value, match.group(0))
        return operator, threshold, phrase
    return None


def _extract_month_list_from_query(query: str) -> list[pd.Timestamp]:
    """Extract explicit month-year mentions from the raw query in stable order."""
    pattern = re.compile(
        r"\b("
        + "|".join(sorted(_RESIDUAL_MONTH_NAME_TO_NUMBER.keys(), key=len, reverse=True))
        + r")\s+(20\d{2})\b",
        re.IGNORECASE,
    )
    months: list[pd.Timestamp] = []
    seen: set[tuple[int, int]] = set()
    for match in pattern.finditer(query or ""):
        month_token = match.group(1).lower()
        year = int(match.group(2))
        month_num = _RESIDUAL_MONTH_NAME_TO_NUMBER.get(month_token)
        if not month_num:
            continue
        key = (year, month_num)
        if key in seen:
            continue
        seen.add(key)
        months.append(pd.Timestamp(year=year, month=month_num, day=1))
    return months


def _build_residual_weighted_price_direct_answer(ctx: QueryContext) -> str | None:
    """Build a deterministic answer for residual weighted-price calculations.

    Gated by an authoritative deterministic residual-computation contract plus
    required column presence.
    """
    if ctx.df.empty:
        return None
    if not _residual_direct_answer_is_authorized(ctx):
        return None

    intent = str(getattr(getattr(ctx.question_analysis, "classification", None), "intent", "") or "").lower()
    is_ppa_cfd_approximation = intent == "implied_ppa_cfd_price_approximation"

    required_cols = {
        "share_ppa_import_total",
        "residual_contribution_ppa_import_gel",
        "residual_contribution_ppa_import_usd",
    }
    if is_ppa_cfd_approximation:
        required_cols.update({
            "known_price_coverage_ok",
            "share_import",
            "share_renewable_ppa",
            "share_thermal_ppa",
            "share_cfd_scheme",
        })
    if not required_cols.issubset(set(ctx.df.columns)):
        return None

    date_col = next((c for c in ctx.df.columns if "date" in c.lower()), None)
    if not date_col:
        return None

    df = ctx.df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["share_ppa_import_total"] = pd.to_numeric(df["share_ppa_import_total"], errors="coerce")
    df["residual_contribution_ppa_import_gel"] = pd.to_numeric(
        df["residual_contribution_ppa_import_gel"], errors="coerce"
    )
    df["residual_contribution_ppa_import_usd"] = pd.to_numeric(
        df["residual_contribution_ppa_import_usd"], errors="coerce"
    )
    if is_ppa_cfd_approximation:
        approximation_share_cols = [
            "share_import",
            "share_renewable_ppa",
            "share_thermal_ppa",
            "share_cfd_scheme",
        ]
        for col in approximation_share_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["share_ppa_cfd_total"] = df[
            ["share_renewable_ppa", "share_thermal_ppa", "share_cfd_scheme"]
        ].sum(axis=1, min_count=3)
        coverage_ok = df["known_price_coverage_ok"].map(
            lambda value: value is True or str(value).strip().lower() in {"true", "1", "yes"}
        )
        df = df[coverage_ok]
        if df.empty:
            return (
                "No months could be calculated because the retrieved data did not contain complete prices "
                "for every positive-share regulated and deregulated bucket."
            )
    df = df.dropna(subset=[date_col, "share_ppa_import_total"])
    df = df[df["share_ppa_import_total"] > 0]
    if df.empty:
        return None

    requested_months = _extract_month_list_from_query(ctx.query)
    if requested_months:
        requested_keys = {(ts.year, ts.month) for ts in requested_months}
        df = df[df[date_col].map(lambda ts: (ts.year, ts.month) in requested_keys if pd.notna(ts) else False)]
    if df.empty:
        return None

    threshold_rule = _extract_residual_share_threshold(ctx.query)
    if threshold_rule:
        operator, threshold, phrase = threshold_rule
        threshold_col = "share_import" if is_ppa_cfd_approximation else "share_ppa_import_total"
        if operator == "gt":
            df = df[df[threshold_col] > threshold]
        elif operator == "ge":
            df = df[df[threshold_col] >= threshold]
        elif operator == "lt":
            df = df[df[threshold_col] < threshold]
        else:
            df = df[df[threshold_col] <= threshold]
        if df.empty:
            threshold_pct = threshold * 100 if threshold <= 1 else threshold
            label = (
                "import"
                if is_ppa_cfd_approximation
                else
                "renewable PPA + import + thermal PPA + CfD scheme"
                if _has_explicit_residual_component_query_signal(ctx.query)
                else "the residual PPA/CfD/import layer"
            )
            return (
                f"No requested months were found where **{label.title()}** {phrase} "
                f"**{threshold_pct:.1f}%** of balancing electricity."
            )

    df = df.sort_values(date_col).copy()
    if is_ppa_cfd_approximation:
        df = df.dropna(subset=[
            "share_import",
            "share_ppa_cfd_total",
            "residual_contribution_ppa_import_gel",
            "residual_contribution_ppa_import_usd",
        ])
        df = df[df["share_ppa_cfd_total"] > 0]
        if df.empty:
            return None
        df["implied_ppa_cfd_price_gel"] = (
            df["residual_contribution_ppa_import_gel"] / df["share_ppa_cfd_total"]
        )
        df["implied_ppa_cfd_price_usd"] = (
            df["residual_contribution_ppa_import_usd"] / df["share_ppa_cfd_total"]
        )
        lines = [
            "**Approximate Weighted Average PPA/CfD Price**",
            "",
            "This is a monthly implied price for Renewable PPA + Thermal PPA + CfD Scheme. "
            "It assumes the import contribution is zero in months passing the requested import-share filter.",
            "",
            "Formula: approximate PPA/CfD price = residual PPA/CfD/import contribution / PPA/CfD share.",
            "Exact conditional formula: (residual contribution - import share × import price) / PPA/CfD share.",
            "",
        ]
        if threshold_rule:
            _operator, threshold, _phrase = threshold_rule
            lines.append(f"Applied filter: import share {_phrase} {threshold * 100:.3g}%.")
            lines.append("")
        for _, row in df.iterrows():
            period_str = pd.to_datetime(row[date_col]).strftime("%B %Y")
            lines.append(
                f"- {period_str}: import share {float(row['share_import']) * 100:.3f}%; "
                f"PPA/CfD share {float(row['share_ppa_cfd_total']) * 100:.1f}%; "
                f"approximate PPA/CfD price {float(row['implied_ppa_cfd_price_gel']):.1f} GEL/MWh "
                f"({float(row['implied_ppa_cfd_price_usd']):.1f} USD/MWh)"
            )
        return "\n".join(lines)

    # Recover the implied weighted price by dividing the residual contribution by the residual share.
    df["remaining_weighted_price_gel"] = (
        df["residual_contribution_ppa_import_gel"] / df["share_ppa_import_total"]
    )
    df["remaining_weighted_price_usd"] = (
        df["residual_contribution_ppa_import_usd"] / df["share_ppa_import_total"]
    )

    explicit_components = _has_explicit_residual_component_query_signal(ctx.query)
    title = (
        "**Weighted Average Price of Renewable PPA + Import + Thermal PPA + CfD Scheme**"
        if explicit_components
        else "**Weighted Average Price of the Remaining Balancing Electricity**"
    )
    lines = [
        title,
        "",
        (
            "Target bucket = Renewable PPA + Import + Thermal PPA + CfD Scheme."
            if explicit_components
            else "Remaining bucket = balancing electricity excluding regulated hydro, regulated thermals, and deregulated renewable."
        ),
        "This corresponds to the residual PPA/CfD/import layer in the current balancing-price decomposition.",
        "",
        "Formula used: residual weighted price = residual contribution / remaining share.",
        "",
    ]
    if threshold_rule:
        _operator, threshold, _phrase = threshold_rule
        threshold_pct = threshold * 100 if threshold <= 1 else threshold
        lines.insert(4, f"Applied filter: remaining share had to satisfy the requested {threshold_pct:.1f}% threshold.")
        lines.insert(5, "")
    for _, row in df.iterrows():
        period_str = pd.to_datetime(row[date_col]).strftime("%B %Y")
        share_pct = float(row["share_ppa_import_total"]) * 100
        lines.append(
            f"- {period_str}: remaining share {share_pct:.1f}%; "
            f"weighted average remaining price {float(row['remaining_weighted_price_gel']):.1f} GEL/MWh "
            f"({float(row['remaining_weighted_price_usd']):.1f} USD/MWh)"
        )
    return "\n".join(lines)


# Forecast helpers parse the precomputed trendline block produced in Stage 3.
def _extract_first_float(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_trendline_forecasts(stats_hint: str) -> tuple[str | None, list[dict[str, Any]]]:
    if not stats_hint:
        return None, []

    marker = "--- TRENDLINE FORECASTS"
    start_idx = stats_hint.find(marker)
    if start_idx < 0:
        return None, []

    section = stats_hint[start_idx:].splitlines()
    target_date: str | None = None
    current: dict[str, Any] | None = None
    entries: list[dict[str, Any]] = []

    for raw_line in section[1:]:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("--- ") and "TRENDLINE FORECASTS" not in line:
            break
        if line.lower().startswith("target date:"):
            target_date = line.split(":", 1)[1].strip()
            continue
        if line.endswith(":") and not line.startswith("-"):
            if current:
                entries.append(current)
            label = line[:-1].strip()
            metric = label
            season = None
            season_match = re.match(r"^(.*?)\s+\(([^)]+)\)$", label)
            if season_match:
                metric = season_match.group(1).strip()
                season = season_match.group(2).strip()
            current = {"label": label, "metric": metric, "season": season}
            continue
        if current is None or not line.startswith("-"):
            continue
        if "forecast value" in line.lower():
            current["forecast_value"] = _extract_first_float(line)
        elif "equation" in line.lower():
            current["equation"] = line.split(":", 1)[1].strip() if ":" in line else ""
        elif (
            "goodness of fit" in line.lower()
            or "r²" in line.lower()
            or "râ²" in line.lower()
            or "r^2" in line.lower()
        ):
            current["r_squared"] = _extract_first_float(line)

    if current:
        entries.append(current)
    return target_date, [entry for entry in entries if entry.get("forecast_value") is not None]


def _filter_relevant_forecast_entries(ctx: QueryContext, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not entries:
        return []

    query_lower = (ctx.query or "").strip().lower()
    selected = list(entries)

    if "balancing" in query_lower:
        selected = [entry for entry in selected if str(entry.get("metric", "")).startswith("p_bal_")]
    elif any(token in query_lower for token in ("deregulated", "power plant")):
        selected = [entry for entry in selected if str(entry.get("metric", "")).startswith("p_dereg_")]
    elif "guaranteed capacity" in query_lower:
        selected = [entry for entry in selected if str(entry.get("metric", "")).startswith("p_gcap_")]
    elif any(token in query_lower for token in ("exchange rate", "xrate", "gel per usd")):
        selected = [entry for entry in selected if str(entry.get("metric", "")) == "xrate"]

    if not selected:
        selected = list(entries)

    wants_gel = "gel" in query_lower
    wants_usd = "usd" in query_lower
    if wants_gel and not wants_usd:
        gel_selected = [entry for entry in selected if str(entry.get("metric", "")).endswith("_gel")]
        if gel_selected:
            selected = gel_selected
    elif wants_usd and not wants_gel:
        usd_selected = [entry for entry in selected if str(entry.get("metric", "")).endswith("_usd")]
        if usd_selected:
            selected = usd_selected

    non_xrate = [entry for entry in selected if str(entry.get("metric", "")) != "xrate"]
    if non_xrate:
        selected = non_xrate

    return selected


def _build_scenario_frame(ctx: QueryContext) -> ScenarioFrame | None:
    """Build a ScenarioFrame from Stage 3 scenario evidence records."""
    evidence = ctx.analysis_evidence or []
    scenario_records = [r for r in evidence if r.get("record_type") == "scenario"]
    if not scenario_records:
        return None
    rows = []
    for rec in scenario_records:
        if rec.get("aggregate_result") is None:
            continue
        rows.append({
            "metric_name": rec.get("derived_metric_name", ""),
            "scenario_factor": rec.get("scenario_factor"),
            "scenario_volume": rec.get("scenario_volume"),
            "aggregate_result": rec.get("aggregate_result"),
            "row_count": rec.get("row_count"),
            "period_range": rec.get("period_range", ""),
            "min_period_value": rec.get("min_period_value"),
            "max_period_value": rec.get("max_period_value"),
            "mean_period_value": rec.get("mean_period_value"),
            "formula": rec.get("formula", ""),
            "metric": rec.get("metric", ""),
            "positive_sum": rec.get("positive_sum", 0.0),
            "negative_sum": rec.get("negative_sum", 0.0),
            "positive_count": rec.get("positive_count", 0),
            "negative_count": rec.get("negative_count", 0),
            "market_component_aggregate": rec.get("market_component_aggregate"),
            "combined_total_aggregate": rec.get("combined_total_aggregate"),
            "baseline_aggregate": rec.get("baseline_aggregate"),
            "delta_aggregate": rec.get("delta_aggregate"),
            "delta_percent": rec.get("delta_percent"),
        })
    return ScenarioFrame(rows=rows, provenance_refs=list(ctx.provenance_refs)) if rows else None


def _build_forecast_frame_from_cagr_rows(ctx: QueryContext) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract forecast entries from CAGR-projected ``is_forecast=True`` rows
    in ``ctx.df``.

    Used when ``_generate_cagr_forecast`` produced forecast rows but
    ``_precalculate_trendlines`` was skipped (Phase 18 behaviour — linear
    trendlines are suppressed to avoid visual conflict with the CAGR-driven
    forecast series). Without this bridge the generic renderer's FORECAST
    path would see no frame, fall through to the LLM summarizer, and fail
    grounding because the LLM has no deterministic anchor for projected
    values.

    Returns ``(target_date_iso, entries)`` where ``entries`` matches the
    ``ForecastFrame.rows`` schema: ``metric``, ``forecast_value``,
    ``r_squared`` (None for CAGR), ``equation`` (CAGR % if parsable),
    ``season`` (summer/winter/None).
    """
    import pandas as pd  # local to avoid import-order issues
    df = getattr(ctx, "df", None)
    if df is None or df.empty or "is_forecast" not in df.columns:
        return None, []
    fc_mask = df["is_forecast"].fillna(False).astype(bool)
    if not fc_mask.any():
        return None, []
    forecast_rows = df.loc[fc_mask].copy()

    # Resolve time column: prefer a datetime-like column with a clear name.
    time_col = None
    for col in forecast_rows.columns:
        if str(col).lower() in {"date", "period", "time"} or "date" in str(col).lower():
            time_col = col
            break
    if time_col is None:
        for col in forecast_rows.columns:
            if pd.api.types.is_datetime64_any_dtype(forecast_rows[col]):
                time_col = col
                break
    if time_col is None:
        return None, []
    forecast_rows[time_col] = pd.to_datetime(forecast_rows[time_col], errors="coerce")
    forecast_rows = forecast_rows.dropna(subset=[time_col])
    if forecast_rows.empty:
        return None, []

    # Pick the metric column(s): every numeric column that isn't a marker,
    # scratch, reference, or year/month integer.  Multiple columns are
    # expected for balancing-price forecasts (``p_bal_gel`` AND ``p_bal_usd``)
    # so the renderer can show separate GEL and USD entries per season.
    def _is_candidate_metric(col_name: str) -> bool:
        lower = str(col_name).lower()
        if lower in {"is_forecast", "xrate"}:
            return False
        if str(col_name).startswith("__"):
            return False
        if lower in {"year", "month"} or re.search(r"\b(year|month)\b", lower):
            return False
        return True

    metric_cols: list[str] = []
    for col in forecast_rows.columns:
        if col == time_col:
            continue
        if not _is_candidate_metric(str(col)):
            continue
        if not pd.api.types.is_numeric_dtype(forecast_rows[col]):
            continue
        # Skip columns that are all-NaN in forecast rows so we never emit
        # an empty entry just because the column existed historically.
        if pd.to_numeric(forecast_rows[col], errors="coerce").dropna().empty:
            continue
        metric_cols.append(col)
    if not metric_cols:
        return None, []

    # Target date = latest forecast date.
    target_ts = forecast_rows[time_col].max()
    target_date = target_ts.strftime("%Y-%m-%d") if pd.notna(target_ts) else None

    # Parse CAGR% from stats_hint note, if present, for a human-readable
    # ``equation`` field. Best-effort — absence does not block the frame.
    # For multi-currency notes (e.g. ``GEL Yearly CAGR=X%, Summer=Y%, ...;
    # USD Yearly CAGR=A%, ...``) match per-currency labels; fall back to the
    # legacy unlabeled form for single-currency notes.
    stats_hint = str(getattr(ctx, "stats_hint", "") or "")

    def _match_cagr_for_metric(metric_col_name: str, kind: str) -> re.Match[str] | None:
        """``kind`` ∈ {"yearly", "summer", "winter"}.

        Looks for a currency-prefixed match first (``GEL Yearly CAGR=…``) and
        falls back to the legacy unprefixed form (``Yearly CAGR=…``).
        """
        currency_prefix = None
        lower = metric_col_name.lower()
        if lower.endswith("_gel"):
            currency_prefix = "GEL"
        elif lower.endswith("_usd"):
            currency_prefix = "USD"
        label_map = {"yearly": "Yearly CAGR", "summer": "Summer", "winter": "Winter"}
        label = label_map[kind]
        if currency_prefix:
            prefixed = re.search(
                rf"{currency_prefix}[^;]*?{label}=([-0-9.]+)%",
                stats_hint,
            )
            if prefixed:
                return prefixed
        return re.search(rf"{label}=([-0-9.]+)%", stats_hint)

    # Restrict to the target year so we emit one entry per metric/season,
    # not every forecast year.
    target_year = target_ts.year
    target_year_rows = forecast_rows[forecast_rows[time_col].dt.year == target_year]

    entries: list[dict[str, Any]] = []
    if "season" in target_year_rows.columns:
        season_series = target_year_rows["season"].astype("object").fillna("").str.lower()
        for season_label, season_tag in (("summer", "summer"), ("winter", "winter")):
            season_rows = target_year_rows[season_series == season_label]
            if season_rows.empty:
                continue
            for metric_col in metric_cols:
                val = pd.to_numeric(season_rows[metric_col], errors="coerce").dropna()
                if val.empty:
                    continue
                match = _match_cagr_for_metric(metric_col, season_tag)
                yearly_match = _match_cagr_for_metric(metric_col, "yearly")
                yearly_equation = (
                    f"CAGR={yearly_match.group(1)}% per year" if yearly_match else None
                )
                entries.append(
                    {
                        "metric": str(metric_col),
                        "forecast_value": float(val.iloc[-1]),
                        "r_squared": None,
                        "equation": (
                            f"CAGR={match.group(1)}% per year" if match else yearly_equation
                        ),
                        "season": season_tag,
                    }
                )

    if not entries:
        # Non-seasonal (quantity / yearly-price) path — one entry per metric.
        for metric_col in metric_cols:
            val = pd.to_numeric(target_year_rows[metric_col], errors="coerce").dropna()
            if val.empty:
                continue
            yearly_match = _match_cagr_for_metric(metric_col, "yearly")
            yearly_equation = (
                f"CAGR={yearly_match.group(1)}% per year" if yearly_match else None
            )
            entries.append(
                {
                    "metric": str(metric_col),
                    "forecast_value": float(val.iloc[-1]),
                    "r_squared": None,
                    "equation": yearly_equation,
                    "season": None,
                }
            )

    return target_date, entries


def _build_forecast_frame(ctx: QueryContext) -> ForecastFrame | None:
    """Build a ForecastFrame for the generic renderer.

    Two sources, tried in order:
    1. ``_precalculate_trendlines`` output embedded in ``stats_hint`` (legacy
       linear-regression path — still used when CAGR produced nothing and
       trendlines ran).
    2. CAGR-projected ``is_forecast=True`` rows in ``ctx.df`` (current path
       whenever ``_generate_cagr_forecast`` succeeded — Phase 18 suppresses
       the trendline pre-calc in this case).

    Having both sources ensures the deterministic FORECAST path in the
    generic renderer always has a frame to render, which keeps the
    summarizer off the LLM retry + grounding-fallback path.
    """
    target_date, entries = _parse_trendline_forecasts(ctx.stats_hint or "")
    if not entries:
        # Fallback: extract CAGR forecast rows from ctx.df.
        target_date, entries = _build_forecast_frame_from_cagr_rows(ctx)
    if not entries:
        return None
    entries = _filter_relevant_forecast_entries(ctx, entries)
    if not entries:
        return None
    return ForecastFrame(
        rows=entries,
        target_date=target_date,
        provenance_refs=list(ctx.provenance_refs),
    )


def _allow_single_period_tariff_snapshot_render(
    ctx: QueryContext,
    frame: ObservationFrame | EntitySetFrame | ForecastFrame | ScenarioFrame,
) -> bool:
    """Permit direct rendering when a tariff snapshot was misclassified as TIMESERIES."""
    gap = getattr(ctx, "evidence_gap", None)
    if gap is None or not getattr(gap, "correctable", False):
        return False
    if getattr(gap, "answer_kind", None) != AnswerKind.TIMESERIES:
        return False
    if (ctx.tool_name or "").strip() != "get_tariffs":
        return False
    if not isinstance(frame, ObservationFrame):
        return False
    if len(frame.periods) != 1 or len(frame.entities) == 0:
        return False
    return True


def _try_generic_renderer(ctx: QueryContext) -> str | None:
    """Attempt the generic renderer when answer_kind + evidence frame are available.

    Handles all standard deterministic answer_kinds: SCALAR, LIST, TIMESERIES,
    COMPARISON, SCENARIO.  FORECAST is intentionally excluded — see below.
    Returns the rendered answer string, or None if not applicable (caller
    should fall through to legacy dispatch).
    """
    if not ctx.has_authoritative_question_analysis:
        return None
    qa = ctx.question_analysis
    if qa.answer_kind is None:
        return None

    # Phase C (2026-05-22): FORECAST answers go through the LLM, not the
    # deterministic renderer. The trendline values are already in
    # ``ctx.stats_hint`` from Stage 3 enrichment (``--- TRENDLINE FORECASTS
    # ---`` block in ``agent/analyzer.py``), so the LLM can cite them
    # verbatim. What the deterministic renderer cannot do — and what the
    # LLM with ``forecast-caveats.md`` in skill guidance CAN — is apply
    # judgment about:
    #   - R²-tiered reliability caveats (< 0.5 vs 0.5-0.7 vs > 0.7);
    #   - the "long-horizon → focus on structural drivers, not linear
    #     extrapolation" rule for 5+ year forecasts;
    #   - the July 2027 target-market regime-break warning;
    #   - summer/winter separation with different driver mixes.
    # Trace c507e4d7 + b7d1a493 (2026-05-22) showed the deterministic
    # renderer shipped "Winter (GEL): 12.49 GEL/MWh" from a one-year
    # data window (Phase B fixes the data window; Phase C ensures the
    # LLM applies judgment to the trendline output).
    if qa.answer_kind == AnswerKind.FORECAST:
        return None

    selected_tariff_aliases = {
        str(alias).strip()
        for alias in (ctx.tool_params.get("entities") or [])
        if str(alias).strip()
    }
    evidence_frame = getattr(ctx, "evidence_frame", None)
    if isinstance(evidence_frame, EntitySetFrame):
        selected_tariff_aliases.update(
            str(row.get("entity_id") or "").strip()
            for row in evidence_frame.rows
            if str(row.get("entity_id") or "").strip()
        )

    if (
        qa.answer_kind == AnswerKind.LIST
        and (ctx.tool_name or "").strip() == "get_tariffs"
        and any(alias in _AGGREGATE_TARIFF_LIST_ALIASES for alias in selected_tariff_aliases)
    ):
        # Keep the existing regulated-plant expansion path for grouped tariff
        # aliases. The generic renderer is used for exact-entity snapshot lists,
        # while aggregated aliases still need the domain mapping in the direct
        # regulated-tariff formatter.
        return None

    # For SCENARIO, build typed frame from Stage 3 enrichment data.  This
    # answer_kind is inherently deterministic when evidence is available,
    # so it bypasses the render_style gate.  (FORECAST was treated the same
    # way until Phase C 2026-05-22; it now early-returns above so the LLM
    # can apply forecast-caveats.md judgment to the trendline values.)
    frame = evidence_frame
    if qa.answer_kind == AnswerKind.SCENARIO:
        scenario_frame = _build_scenario_frame(ctx)
        if scenario_frame is not None and not scenario_frame.is_empty():
            frame = scenario_frame
    else:
        # Non-scenario shapes require explicit DETERMINISTIC render_style.
        if qa.render_style != RenderStyle.DETERMINISTIC:
            return None

    # Correctable evidence gaps mean data is incomplete — fall through to legacy
    # paths that can handle partial data or re-plan.
    gap = getattr(ctx, "evidence_gap", None)
    if gap is not None and getattr(gap, "correctable", False):
        if _allow_single_period_tariff_snapshot_render(ctx, frame):
            log.info(
                "Allowing generic renderer for single-period tariff snapshot despite "
                "TIMESERIES evidence gap."
            )
        else:
            return None

    if frame is None or frame.is_empty():
        return None

    # Extract metric_hint from the top candidate tool's params_hint so that
    # scalar rendering selects the correct metric row (e.g. balancing_price
    # instead of xrate).
    metric_hint: str | None = None
    try:
        top_tool = (qa.tooling.candidate_tools or [None])[0]
        if top_tool and top_tool.params_hint:
            metric_hint = top_tool.params_hint.metric
    except (AttributeError, IndexError):
        pass

    result = generic_render(
        frame=frame,
        answer_kind=qa.answer_kind,
        grouping=qa.grouping,
        entity_scope=qa.entity_scope,
        language_code=qa.language.answer_language.value if qa.language else "en",
        metric_hint=metric_hint,
    )
    if result and result.strip():
        return result
    return None


def summarize_data(ctx: QueryContext) -> QueryContext:
    """Generate an answer from SQL query results.

    Reads: ctx.query, ctx.preview, ctx.stats_hint, ctx.lang_instruction,
           ctx.conversation_history, ctx.share_summary_override
    Writes: ctx.summary
    """
    # Choose the strongest deterministic answer path first, then fall back to LLM summarization.
    grounding_guardrail_triggered = False
    ctx.summary_source = ""
    ctx.grounding_policy = ""
    ctx.summary_domain_knowledge = ""

    # Evidence precedence: discard share override if semantic intent is not share/composition.
    # Trust only structured analyzer signals, not the free-form intent string.
    if ctx.share_summary_override and ctx.semantic_locked:
        if not ctx.analyzer_indicates_share_intent:
            log.warning(
                "Discarding share_summary_override: structured analyzer signals "
                "do not indicate share/composition",
            )
            ctx.share_summary_override = None

    # Explicitly authorized calculations take precedence over the generic
    # evidence-frame renderer, which can only display retrieved observations
    # and cannot derive the requested residual price.
    _residual_answer = _build_residual_weighted_price_direct_answer(ctx)

    # --- Generic renderer path (answer_kind + evidence frame) ---
    # Attempt the generic renderer FIRST when we have a canonical evidence frame
    # and a deterministic render_style.  Handles SCALAR, LIST, TIMESERIES,
    # COMPARISON, SCENARIO, and FORECAST answer kinds — no regex detection.
    _generic_answer = None if _residual_answer is not None else _try_generic_renderer(ctx)
    if _residual_answer is not None:
        ctx.summary = _residual_answer
        ctx.summary_source = "deterministic_residual_weighted_price_direct"
        ctx.summary_claims = []
        ctx.summary_citations = ["deterministic_residual_weighted_price_direct"]
        ctx.summary_confidence = 0.95
        metrics.log_deterministic_skip(ctx.summary_source)
        log.info("Deterministic residual weighted-price answer eligible; skipping Stage 4 LLM.")
    elif _generic_answer is not None:
        ctx.summary = _generic_answer
        ctx.summary_source = "generic_renderer"
        # The generic renderer is already deterministic over the evidence frame.
        # Treat it as a direct-evidence answer, not as narrative prose that must
        # pass the LLM claim/provenance gate.
        ctx.summary_claims = []
        ctx.summary_citations = ["generic_renderer"]
        ctx.summary_confidence = 0.97
        metrics.log_deterministic_skip("generic_renderer")
        log.info("Generic renderer produced answer (answer_kind=%s); skipping legacy dispatch.",
                 ctx.question_analysis.answer_kind.value if ctx.question_analysis and ctx.question_analysis.answer_kind else "?")
    elif ctx.share_summary_override:
        # Permanent specialized formatter for share-intent LIST/SCALAR queries
        # (renewable vs thermal PPA decomposition + per-period price join).
        # Sits in the dispatch chain AFTER the generic renderer so that a
        # canonical ObservationFrame/EntitySetFrame is preferred when it can
        # satisfy the answer shape; the override only fires when the generic
        # renderer cannot produce a domain-decomposed answer.
        # Analogous to SCENARIO/FORECAST specialized formatters per §3.4.
        ctx.summary = ctx.share_summary_override
        ctx.summary_source = "deterministic_share_summary"
        ctx.summary_claims = _derive_claims_from_text(ctx.summary)
        ctx.summary_citations = ["deterministic_share_summary"]
        ctx.summary_confidence = 1.0
    elif (regulated_tariff_answer := _build_regulated_tariff_list_direct_answer(ctx)) is not None:
        ctx.summary = regulated_tariff_answer
        ctx.summary_source = "deterministic_regulated_tariff_list_direct"
        ctx.summary_claims = []
        ctx.summary_citations = ["deterministic_regulated_tariff_list_direct"]
        ctx.summary_confidence = 0.98
        metrics.log_deterministic_skip(ctx.summary_source)
        log.info("Deterministic regulated tariff list answer eligible; skipping Stage 4 LLM.")
    else:
        # Load domain knowledge for complex queries so the LLM can explain
        # causal mechanisms, not just describe data patterns.
        # Prefer analyzer query_type; fall back to heuristic only when unavailable.
        # Only use active analyzer output — shadow must not influence behavior.
        if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active":
            query_type = ctx.question_analysis.classification.query_type.value
            routing_query = ctx.question_analysis.canonical_query_en or ctx.query
        else:
            query_type = classify_query_type(ctx.query)
            routing_query = ctx.query
        comparison_focus = _should_use_comparison_first_guidance(ctx, query_type)
        render_deterministic = (
            ctx.question_analysis is not None
            and ctx.question_analysis_source == "llm_active"
            and ctx.question_analysis.render_style == RenderStyle.DETERMINISTIC
        )
        answer_kind = _resolve_summary_answer_kind(ctx)
        domain_knowledge = ""
        if _should_load_data_summary_domain_knowledge(
            ctx,
            routing_query=routing_query,
            answer_kind=answer_kind,
            query_type=query_type,
            render_deterministic=render_deterministic,
        ):
            preferred_topics = _select_data_summary_domain_topics(ctx, answer_kind)
            domain_knowledge = get_relevant_domain_knowledge(
                routing_query, use_cache=False, preferred_topics=preferred_topics,
            )
            if preferred_topics:
                log.info("Using merged topics for data summary domain knowledge: %s", preferred_topics)
        ctx.summary_domain_knowledge = domain_knowledge
        # Decide grounding policy AFTER domain knowledge is loaded so an
        # explanatory answer that carries domain knowledge uses EVIDENCE_AWARE
        # (its domain figures count as grounded) rather than STRICT_NUMERIC.
        ctx.grounding_policy = _derive_data_summary_grounding_policy(ctx, query_type)
        vector_knowledge = (
            ctx.vector_knowledge_prompt
            if PIPELINE_MODE != "fast"
            and ctx.vector_knowledge is not None
            and ctx.vector_knowledge_source == "vector_active"
            else ""
        )

        try:
            envelope = llm_summarize_structured(
                routing_query,
                ctx.preview,
                ctx.stats_hint,
                ctx.lang_instruction,
                conversation_history=ctx.conversation_history,
                domain_knowledge=domain_knowledge,
                vector_knowledge=vector_knowledge,
                question_analysis=ctx.question_analysis,
                effective_answer_kind=getattr(ctx, "effective_answer_kind", None),
                vector_knowledge_bundle=ctx.vector_knowledge,
                response_mode=ctx.response_mode,
                resolution_policy=ctx.resolution_policy,
                grounding_policy=ctx.grounding_policy,
                comparison_focus=comparison_focus,
            )
            if not _is_summary_grounded(envelope, ctx):
                grounding_guardrail_triggered = True
                metrics.log_summary_grounding_failure()
                # Try deterministic scenario fallback before generic message.
                scenario_answer = _build_scenario_fallback_answer(ctx)
                if scenario_answer is not None:
                    log.info("Summary grounding failed; using deterministic scenario fallback.")
                    envelope = SummaryEnvelope(
                        answer=scenario_answer,
                        claims=[],
                        citations=["deterministic_scenario_fallback"],
                        confidence=0.95,
                    )
                else:
                    log.warning("Summary grounding check failed; using conservative fallback answer.")
                    envelope = SummaryEnvelope(
                        answer=get_grounding_fallback_message(getattr(ctx, "lang_code", "") or "en"),
                        claims=[],
                        citations=["guardrail_grounding_fallback"],
                        confidence=0.2,
                    )

            ctx.summary = envelope.answer
            _fallback_citations = set(envelope.citations or [])
            if "guardrail_grounding_fallback" in _fallback_citations:
                ctx.summary_source = "structured_summary_grounding_fallback"
            elif "deterministic_scenario_fallback" in _fallback_citations:
                ctx.summary_source = "deterministic_scenario_fallback"
            else:
                ctx.summary_source = "structured_summary"
            ctx.summary_claims = list(envelope.claims)
            ctx.summary_citations = list(envelope.citations)
            ctx.summary_confidence = float(envelope.confidence)
            trace_detail(
                log,
                ctx,
                "stage_4_summarize_data",
                "artifact",
                debug=True,
                summary_envelope=envelope,
            )
        except Exception as e:
            # Incident 2026-07-17: this block previously kept an
            # `except RuntimeError: raise` clause as the breaker-open sentinel.
            # F8's ProviderExecutionError subclasses RuntimeError, so every
            # provider failure (including ambiguous delivery) re-raised and
            # turned /ask into a 500 instead of degrading. Breaker-open now
            # surfaces as ProviderExecutionError(REJECTED) with its own
            # core-level OpenAI fallback; if the provider tier is truly down,
            # the legacy call below re-raises immediately from its own breaker
            # check — so provider failures of every disposition degrade here.
            metrics.log_summary_schema_failure()
            log.warning(
                "Structured summarization failed (%s): %s",
                type(e).__name__, e,
            )
            ctx.summary = llm_summarize(
                routing_query,
                ctx.preview,
                ctx.stats_hint,
                ctx.lang_instruction,
                conversation_history=ctx.conversation_history,
                domain_knowledge=domain_knowledge,
                vector_knowledge=vector_knowledge,
            )
            ctx.summary_source = "legacy_text_fallback"
            ctx.summary_claims = _derive_claims_from_text(ctx.summary)
            ctx.summary_citations = ["legacy_text_fallback"]
            ctx.summary_confidence = 0.5

    ctx.summary = strip_inline_citation_markers(scrub_schema_mentions(ctx.summary))
    _apply_absence_claim_guardrail(ctx)

    # Shadow fitness check on deterministic renders (§3.9): observe-only —
    # the provenance gate below is a no-op for these paths, so this is the
    # only visibility into "right shape, wrong rows" failures. Never allowed
    # to raise into the pipeline.
    if ctx.summary_source in _DETERMINISTIC_SUMMARY_SOURCES:
        try:
            _fitness_tags = evaluate_render_fitness(ctx)
        except Exception:
            log.debug("render-fitness check failed", exc_info=True)
            _fitness_tags = []
        if _fitness_tags:
            for _tag in _fitness_tags:
                metrics.log_render_fitness(_tag)
            trace_detail(
                log, ctx, "stage_4_render_fitness", "violations",
                tags=_fitness_tags, summary_source=ctx.summary_source,
            )

    trace_detail(
        log,
        ctx,
        "stage_4_summarize_data",
        "pre_gate",
        summary_source=ctx.summary_source,
        grounding_guardrail_triggered=grounding_guardrail_triggered,
        grounding_policy=ctx.grounding_policy,
        claims_count=len(ctx.summary_claims or []),
        citations=list(ctx.summary_citations or []),
        confidence=ctx.summary_confidence,
        summary_preview=ctx.summary,
    )
    _attach_claim_provenance(ctx)
    _enforce_provenance_gate(ctx)
    return ctx
