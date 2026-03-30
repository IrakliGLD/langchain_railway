"""
Pipeline Stage 4: LLM Summarization

Generates natural language answers — either from domain knowledge (conceptual)
or from SQL query results (data summarization).
"""
import json
import logging
import re
import hashlib
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Set

from models import QueryContext, ResolutionPolicy, GroundingPolicy
from core.llm import (
    llm_summarize,
    llm_summarize_structured,
    SummaryEnvelope,
    classify_query_type,
    get_relevant_domain_knowledge,
)
from context import scrub_schema_mentions
from config import PROVENANCE_MIN_COVERAGE
from utils.metrics import metrics
from utils.trace_logging import trace_detail

log = logging.getLogger("Enai")

_NUMBER_PATTERN = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_PROVENANCE_CONTEXT_COLS = ("date", "month", "year", "entity", "type_tech", "segment")
_MAX_REFS_PER_TOKEN = 4
_MAX_REFS_PER_CLAIM = 12
_MAX_PROVENANCE_CITATIONS = 30
_SUPPORTED_DETERMINISTIC_SCENARIOS = {
    "scenario_payoff",
    "scenario_scale",
    "scenario_offset",
}
_SUPPORTED_DETERMINISTIC_QUERY_TYPES = {
    "data_retrieval",
    "data_explanation",
}
_EXPLANATION_QUERY_SIGNALS = (
    "why",
    "explain",
    "interpret",
    "reason",
    "cause",
    "what does this mean",
    "what does that mean",
    "meaning",
    "implication",
    "imply",
    "compare",
    "comparison",
    "versus",
)
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


def _normalize_number_token(raw_token: str) -> Optional[str]:
    token = (raw_token or "").strip().replace(",", "")
    if token.endswith("%"):
        token = token[:-1]
    if not token:
        return None
    try:
        normalized = format(Decimal(token).normalize(), "f")
    except (InvalidOperation, ValueError):
        normalized = token
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if not normalized or normalized in {"-", "+", "."}:
        return None
    return normalized


def _extract_number_tokens(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for match in _NUMBER_PATTERN.finditer(text or ""):
        normalized = _normalize_number_token(match.group(0))
        if not normalized or len(normalized) <= 1:
            continue
        tokens.add(normalized)
    return tokens


def _build_grounding_corpus(ctx: QueryContext) -> str:
    parts = [ctx.preview or "", ctx.stats_hint or ""]
    if str(ctx.grounding_policy or "") == GroundingPolicy.EVIDENCE_AWARE:
        parts.append(ctx.summary_domain_knowledge or "")
        parts.append(ctx.vector_knowledge_prompt or "")
    if ctx.df is not None and not ctx.df.empty:
        try:
            parts.append(ctx.df.head(200).to_string(index=False))
        except Exception:  # pragma: no cover - defensive
            pass
    return "\n".join(parts)


def _add_aggregate_tokens(tokens: Set[str], ctx: QueryContext) -> None:
    """Add column-level aggregates to grounding tokens for analyst-mode queries.

    CfD and similar calculations produce derived values (strike * volume, etc.)
    that don't exist in raw data rows.  By adding sum/mean/min/max/count of
    each numeric column, we give the grounding check legitimate computed values
    to match against — preventing false-positive failures while keeping the 90%
    threshold intact for non-analyst queries.
    """
    if ctx.df is None or ctx.df.empty:
        return
    qa = ctx.question_analysis
    is_analyst = (
        (qa is not None and qa.classification.analysis_mode.value == "analyst")
        or ctx.mode == "analyst"
    )
    if not is_analyst:
        return
    for col in ctx.df.select_dtypes(include="number").columns:
        series = ctx.df[col].dropna()
        if series.empty:
            continue
        for val in [series.sum(), series.mean(), series.min(), series.max(), len(series)]:
            tokens.update(_tokenize_cell_value(val))


def _build_grounding_tokens(ctx: QueryContext) -> Set[str]:
    tokens = _extract_number_tokens(_build_grounding_corpus(ctx))
    # Expand text-extracted numbers with cell-level tokenization so that
    # ratios in stats_hint (e.g. 0.0666) also register as percentages (6.66).
    expanded: Set[str] = set()
    for t in tokens:
        expanded.update(_tokenize_cell_value(t))
    tokens.update(expanded)
    if ctx.df is not None and not ctx.df.empty:
        for _, series in ctx.df.head(200).items():
            for value in series.tolist():
                tokens.update(_tokenize_cell_value(value))
    elif ctx.rows:
        for row in ctx.rows[:200]:
            for value in row:
                tokens.update(_tokenize_cell_value(value))
    _add_aggregate_tokens(tokens, ctx)
    return tokens


def _is_summary_grounded(envelope: SummaryEnvelope, ctx: QueryContext) -> bool:
    grounding_policy = str(ctx.grounding_policy or GroundingPolicy.STRICT_NUMERIC)
    if grounding_policy == GroundingPolicy.NOT_APPLICABLE:
        return True

    claim_text = "\n".join(envelope.claims or [])
    answer_tokens = _extract_number_tokens((envelope.answer or "") + "\n" + claim_text)
    if not answer_tokens:
        return True

    source_tokens = _build_grounding_tokens(ctx)
    if not source_tokens:
        return False

    matched = sum(1 for t in answer_tokens if t in source_tokens)
    match_ratio = matched / max(1, len(answer_tokens))
    return match_ratio >= 0.9


def _serialize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    return str(value)


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


def _tokenize_cell_value(value: Any) -> Set[str]:
    tokens = _extract_number_tokens(str(value))
    if value is None or isinstance(value, bool):
        return tokens
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return tokens
    if not numeric.is_finite():
        return tokens

    # Support rounding and truncation for all numeric values
    # This ensures that raw data like 37.9913 can be matched by LLM-rounded "37.99" or "38"
    
    # 1. Direct percentage support for ratio cells (abs <= 1)
    if abs(numeric) <= 1:
        percent_raw = numeric * Decimal("100")
        for val in [percent_raw, round(percent_raw, 1), round(percent_raw, 2)]:
            t = _normalize_number_token(str(val))
            if t:
                tokens.add(t)
        
        # Truncation for ratios
        pr_str = str(percent_raw)
        if "." in pr_str:
            dec_idx = pr_str.find(".")
            for i in [1, 2]: # Truncate at 1 or 2 decimals
                if len(pr_str) > dec_idx + i:
                    t = _normalize_number_token(pr_str[:dec_idx + i + 1])
                    if t:
                        tokens.add(t)

    # 2. General rounding support for all numbers (including the primary value)
    for val in [numeric, round(numeric, 1), round(numeric, 2)]:
        t = _normalize_number_token(str(val))
        if t:
            tokens.add(t)

    # 3. General truncation
    num_str = str(numeric)
    if "." in num_str:
        dec_idx = num_str.find(".")
        for i in [1, 2]:
            if len(num_str) > dec_idx + i:
                t = _normalize_number_token(num_str[:dec_idx + i + 1])
                if t:
                    tokens.add(t)

    # 4. Unsigned versions for ALL generated tokens (e.g., "dropped by 5" matching "-5")
    unsigned_extra = set()
    for t in tokens:
        if t.startswith("-"):
            unsigned_extra.add(t[1:])
    tokens.update(unsigned_extra)

    return tokens


def _build_row_context(cols: List[Any], row: tuple) -> Dict[str, Any]:
    normalized = [str(c).lower() for c in cols]
    context: Dict[str, Any] = {}
    for key in _PROVENANCE_CONTEXT_COLS:
        if key in normalized:
            idx = normalized.index(key)
            if idx < len(row):
                context[str(cols[idx])] = _serialize_scalar(row[idx])
    if context:
        return context
    for idx, col in enumerate(cols[:3]):
        if idx < len(row):
            context[str(col)] = _serialize_scalar(row[idx])
    return context


def _build_claim_provenance(
    claims: List[str],
    cols: List[Any],
    rows: List[tuple],
    query_hash: str = "",
    source: str = "",
    stats_hint: str = "",
    domain_knowledge: str = "",
    external_source_passages: str = "",
) -> tuple[List[Dict[str, Any]], float, List[str]]:
    token_index: Dict[str, List[Dict[str, Any]]] = {}
    source_priority = {
        "sql": 0,
        "tool": 0,
        "derived_analysis": 1,
        "external_source_passages": 2,
        "domain_knowledge": 3,
        "unknown": 4,
    }

    def _index_text_source(text: str, *, source_name: str, column_label: str, coordinate: str) -> None:
        if not text:
            return
        for token in _extract_number_tokens(text):
            refs = token_index.setdefault(token, [])
            refs.append({
                "source": source_name,
                "row_number": 0,
                "row_index": -1,
                "column": column_label,
                "value": token,
                "cell_id": f"{source_name}:{hashlib.md5(token.encode()).hexdigest()[:8]}",
                "row_context": {"Type": column_label},
                "coordinate": coordinate,
                "query_hash": query_hash,
            })

    # --- Index statistics and derived analysis ---
    if stats_hint:
        for token in _extract_number_tokens(stats_hint):
            refs = token_index.setdefault(token, [])
            refs.append({
                "source": "derived_analysis",
                "row_number": 0,
                "row_index": -1,
                "column": "Statistics/Causal Evidence",
                "value": token,
                "cell_id": f"derived:{hashlib.md5(token.encode()).hexdigest()[:8]}",
                "row_context": {"Type": "System Analysis"},
                "coordinate": "stats_hint",
                "query_hash": query_hash,
            })
    _index_text_source(
        domain_knowledge,
        source_name="domain_knowledge",
        column_label="Domain Knowledge",
        coordinate="domain_knowledge",
    )
    _index_text_source(
        external_source_passages,
        source_name="external_source_passages",
        column_label="External Source Passages",
        coordinate="external_source_passages",
    )
    for row_idx, row in enumerate(rows):
        row_context = _build_row_context(cols, row)
        for col_idx, col_name in enumerate(cols):
            if col_idx >= len(row):
                continue
            value = row[col_idx]
            for token in _tokenize_cell_value(value):
                refs = token_index.setdefault(token, [])
                refs.append(
                    {
                        "row_index": row_idx,
                        "row_number": row_idx + 1,
                        "column": str(col_name),
                        "value": _serialize_scalar(value),
                        "coordinate": f"r{row_idx + 1}.c[{col_name}]",
                        "query_hash": query_hash,
                        "source": source or "unknown",
                        "row_fingerprint": hashlib.sha256(
                            f"{query_hash}|{row_idx + 1}|{repr(row)}".encode("utf-8")
                        ).hexdigest()[:16],
                        "cell_id": f"{query_hash}:r{row_idx + 1}:c[{col_name}]",
                        "row_context": row_context,
                    }
                )

    claim_entries: List[Dict[str, Any]] = []
    citation_anchors: List[str] = []
    numeric_claims = 0
    grounded_numeric_claims = 0

    for claim_idx, claim in enumerate(claims):
        claim_tokens = sorted(_extract_number_tokens(claim))
        if claim_tokens:
            numeric_claims += 1

        matched_tokens: List[str] = []
        unmatched_tokens: List[str] = []
        claim_refs: List[Dict[str, Any]] = []
        seen_refs = set()

        for token in claim_tokens:
            refs = sorted(
                token_index.get(token, []),
                key=lambda ref: (
                    source_priority.get(str(ref.get("source") or "unknown"), 4),
                    int(ref.get("row_index", -1)),
                ),
            )
            if not refs:
                unmatched_tokens.append(token)
                continue
            matched_tokens.append(token)
            for ref in refs[:_MAX_REFS_PER_TOKEN]:
                ref_key = (ref["row_number"], ref["column"], str(ref["value"]))
                if ref_key in seen_refs:
                    continue
                seen_refs.add(ref_key)
                claim_refs.append(ref)
                citation_anchors.append(f"claim_{claim_idx}:{ref['cell_id']}")
                if len(claim_refs) >= _MAX_REFS_PER_CLAIM:
                    break
            if len(claim_refs) >= _MAX_REFS_PER_CLAIM:
                break

        if claim_tokens and not unmatched_tokens and claim_refs:
            grounded_numeric_claims += 1

        claim_entries.append(
            {
                "claim_index": claim_idx,
                "claim_text": claim,
                "tokens": claim_tokens,
                "matched_tokens": matched_tokens,
                "unmatched_tokens": unmatched_tokens,
                "cell_refs": claim_refs,
                "is_fully_grounded": bool(claim_tokens) and not unmatched_tokens and bool(claim_refs),
            }
        )

    coverage = 1.0 if numeric_claims == 0 else grounded_numeric_claims / max(1, numeric_claims)
    deduped_anchors = list(dict.fromkeys(citation_anchors))
    return claim_entries, coverage, deduped_anchors


def _derive_claims_from_text(summary_text: str) -> List[str]:
    claims = [line.strip(" -*\t") for line in (summary_text or "").splitlines() if line.strip()]
    if not claims and summary_text.strip():
        claims = [summary_text.strip()]
    return claims[:8]


def _attach_claim_provenance(ctx: QueryContext) -> None:
    claims = [str(c).strip() for c in (ctx.summary_claims or []) if str(c).strip()]
    source_cols = list(ctx.provenance_cols or ctx.cols or [])
    source_rows = [tuple(r) for r in (ctx.provenance_rows or ctx.rows or [])]

    if not claims:
        ctx.summary_claim_provenance = []
        ctx.summary_provenance_coverage = 0.0
        return
    if not source_cols or not source_rows:
        ctx.summary_claim_provenance = []
        ctx.summary_provenance_coverage = 0.0
        return

    claim_prov, coverage, anchors = _build_claim_provenance(
        claims,
        source_cols,
        source_rows,
        query_hash=ctx.provenance_query_hash or "unknown",
        source=ctx.provenance_source or "unknown",
        stats_hint=ctx.stats_hint or "",
        domain_knowledge=ctx.summary_domain_knowledge or "",
        external_source_passages=ctx.vector_knowledge_prompt or "",
    )
    ctx.summary_claim_provenance = claim_prov
    ctx.summary_provenance_coverage = round(float(coverage), 4)
    if anchors:
        merged = list(dict.fromkeys(list(ctx.summary_citations or []) + anchors))
        ctx.summary_citations = merged[:_MAX_PROVENANCE_CITATIONS]


def _enforce_provenance_gate(ctx: QueryContext) -> None:
    claim_entries = list(ctx.summary_claim_provenance or [])
    numeric_claims = [entry for entry in claim_entries if entry.get("tokens")]
    if not claim_entries:
        ctx.summary_provenance_gate_passed = True
        ctx.summary_provenance_gate_reason = "no_claims"
        trace_detail(
            log,
            ctx,
            "stage_4_summarize_data",
            "provenance_gate",
            gate_passed=True,
            gate_reason=ctx.summary_provenance_gate_reason,
            numeric_claims=0,
            coverage=0.0,
        )
        return
    if not numeric_claims:
        ctx.summary_provenance_gate_passed = True
        ctx.summary_provenance_gate_reason = "no_numeric_claims"
        trace_detail(
            log,
            ctx,
            "stage_4_summarize_data",
            "provenance_gate",
            gate_passed=True,
            gate_reason=ctx.summary_provenance_gate_reason,
            numeric_claims=0,
            coverage=float(ctx.summary_provenance_coverage or 0.0),
        )
        return

    has_ungrounded_claim = any(not bool(entry.get("is_fully_grounded")) for entry in numeric_claims)
    coverage = float(ctx.summary_provenance_coverage or 0.0)
    # Use coverage-only gating: the binary has_ungrounded_claim check was
    # too strict for analytical queries where the LLM legitimately derives
    # values (MoM changes, percentages) from raw data.  Coverage already
    # captures grounding quality proportionally.  has_ungrounded_claim is
    # still computed and logged for observability.
    gate_passed = coverage >= PROVENANCE_MIN_COVERAGE
    if gate_passed:
        ctx.summary_provenance_gate_passed = True
        ctx.summary_provenance_gate_reason = "ok"
        trace_detail(
            log,
            ctx,
            "stage_4_summarize_data",
            "provenance_gate",
            gate_passed=True,
            gate_reason=ctx.summary_provenance_gate_reason,
            numeric_claims=len(numeric_claims),
            coverage=coverage,
        )
        return

    metrics.log_summary_grounding_failure()
    if hasattr(metrics, "log_provenance_gate_failure"):
        metrics.log_provenance_gate_failure()
    ctx.summary_provenance_gate_passed = False
    ctx.summary_provenance_gate_reason = (
        f"coverage={coverage:.4f}, min={PROVENANCE_MIN_COVERAGE:.4f}, "
        f"ungrounded_numeric_claims={int(has_ungrounded_claim)}"
    )
    ctx.summary = (
        "I could not produce citation-grade grounding for all numeric claims from the retrieved dataset. "
        "Please refine the query scope (period/entity/metric) and retry."
    )
    ctx.summary_source = "citation_gate_fallback"
    ctx.summary_claims = []
    ctx.summary_citations = ["citation_gate_fallback"]
    ctx.summary_confidence = min(float(ctx.summary_confidence or 0.0), 0.2)
    ctx.summary_claim_provenance = []
    ctx.summary_provenance_coverage = 0.0
    unmatched = [
        {
            "claim_index": entry.get("claim_index"),
            "unmatched_tokens": list(entry.get("unmatched_tokens") or []),
        }
        for entry in numeric_claims
        if entry.get("unmatched_tokens")
    ]
    trace_detail(
        log,
        ctx,
        "stage_4_summarize_data",
        "provenance_gate",
        gate_passed=False,
        gate_reason=ctx.summary_provenance_gate_reason,
        numeric_claims=len(numeric_claims),
        coverage=coverage,
        unmatched_tokens=unmatched,
    )
    trace_detail(
        log,
        ctx,
        "stage_4_summarize_data",
        "artifact",
        debug=True,
        summary_claim_provenance=claim_entries,
    )


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


def _build_data_summary_topic_preferences(ctx: QueryContext) -> Optional[List[str]]:
    """Merge analyzer and vector topic preferences for data summaries."""

    return _merge_topic_preferences(
        _extract_vector_preferred_topics(ctx),
        _extract_preferred_topics(ctx),
    )


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
    options: List[str] = []

    def _add(option: str) -> None:
        if option not in options:
            options.append(option)

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
    ctx.summary = scrub_schema_mentions(ctx.summary)
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
    vector_preferred_topics = (
        list(ctx.vector_knowledge.filters.preferred_topics)
        if vector_evidence_active and ctx.vector_knowledge is not None
        else []
    )
    domain_background_topics = _merge_topic_preferences(vector_preferred_topics, preferred_topics)
    query_lower = routing_query.lower()
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
            "NOTE: This question is answered from OFFICIAL REGULATORY SOURCE PASSAGES. "
            "No database query was executed. "
            "EXTERNAL_SOURCE_PASSAGES contain the primary evidence retrieved from official regulations.\n\n"
            "PRIMARY EVIDENCE RULES (MANDATORY):\n"
            "- Build the answer PRIMARILY from EXTERNAL_SOURCE_PASSAGES content.\n"
            "- Use DOMAIN_KNOWLEDGE only as secondary background to clarify terms or provide brief Georgia-specific context.\n"
            "- If EXTERNAL_SOURCE_PASSAGES do not contain a procedural or regulatory detail, say that directly instead of filling the gap from background knowledge.\n"
            "- For eligibility, registration, compliance, and process questions, extract the wording and constraints from EXTERNAL_SOURCE_PASSAGES, then use DOMAIN_KNOWLEDGE only to synthesize the answer clearly.\n"
            "- DO NOT give a generic answer when specific regulatory content is available in EXTERNAL_SOURCE_PASSAGES."
        )
        log.info(
            "Active vector evidence present for conceptual answer; preserving topic-filtered domain knowledge as secondary background"
        )

    vector_knowledge = ctx.vector_knowledge_prompt if vector_evidence_active else ""
    domain_knowledge_for_summary = domain_knowledge
    if vector_evidence_active and len(domain_knowledge_for_summary) > 4000:
        domain_knowledge_for_summary = domain_knowledge_for_summary[:4000]
        log.info("Capped domain_knowledge to 4000 chars (vector evidence is primary)")
    try:
        envelope = llm_summarize_structured(
            ctx.query,
            data_preview="",
            stats_hint=conceptual_hint,
            lang_instruction=ctx.lang_instruction,
            conversation_history=ctx.conversation_history,
            domain_knowledge=domain_knowledge_for_summary,
            vector_knowledge=vector_knowledge,
            question_analysis=ctx.question_analysis,
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
    except RuntimeError:
        # Circuit breaker open — system-level issue, don't mask with fallback
        raise
    except Exception as exc:
        metrics.log_summary_schema_failure()
        log.warning(
            "Structured conceptual summarization failed (%s): %s",
            type(exc).__name__, exc,
        )
        ctx.summary = llm_summarize(
            ctx.query,
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
    ctx.summary = scrub_schema_mentions(ctx.summary)
    log.info("✅ Conceptual answer generated")
    return ctx


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


def _is_deterministic_scenario_eligible(ctx: QueryContext) -> bool:
    """Return True when scenario evidence is complete enough to skip Stage 4 LLM."""
    evidence = ctx.analysis_evidence or []
    scenario_records = [r for r in evidence if r.get("record_type") == "scenario"]
    if len(scenario_records) != 1:
        return False

    rec = scenario_records[0]
    metric_name = str(rec.get("derived_metric_name") or "").strip()
    if metric_name not in _SUPPORTED_DETERMINISTIC_SCENARIOS:
        return False
    if rec.get("aggregate_result") is None:
        return False

    if ctx.question_analysis is not None:
        query_type = getattr(ctx.question_analysis.classification.query_type, "value", "")
        if query_type and query_type not in _SUPPORTED_DETERMINISTIC_QUERY_TYPES:
            return False

    query_lower = (ctx.query or "").strip().lower()
    if any(signal in query_lower for signal in _EXPLANATION_QUERY_SIGNALS):
        return False

    return _build_scenario_fallback_answer(ctx) is not None


def summarize_data(ctx: QueryContext) -> QueryContext:
    """Generate an answer from SQL query results.

    Reads: ctx.query, ctx.preview, ctx.stats_hint, ctx.lang_instruction,
           ctx.conversation_history, ctx.share_summary_override
    Writes: ctx.summary
    """
    strict_grounding_retry = False
    ctx.summary_source = ""
    ctx.grounding_policy = ""
    ctx.summary_domain_knowledge = ""

    if ctx.share_summary_override:
        ctx.summary = ctx.share_summary_override
        ctx.summary_source = "deterministic_share_summary"
        ctx.summary_claims = _derive_claims_from_text(ctx.summary)
        ctx.summary_citations = ["deterministic_share_summary"]
        ctx.summary_confidence = 1.0
    elif _is_deterministic_scenario_eligible(ctx):
        ctx.summary = _build_scenario_fallback_answer(ctx) or ""
        ctx.summary_source = "deterministic_scenario_direct"
        ctx.summary_claims = []
        ctx.summary_citations = ["deterministic_scenario_direct"]
        ctx.summary_confidence = 0.95
        metrics.log_deterministic_skip(ctx.summary_source)
        log.info("Deterministic scenario answer eligible; skipping Stage 4 LLM.")
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
        ctx.grounding_policy = _derive_data_summary_grounding_policy(ctx, query_type)
        comparison_focus = _should_use_comparison_first_guidance(ctx, query_type)
        domain_knowledge = ""
        if query_type not in ("single_value", "list"):
            preferred_topics = _build_data_summary_topic_preferences(ctx)
            domain_knowledge = get_relevant_domain_knowledge(
                routing_query, use_cache=False, preferred_topics=preferred_topics,
            )
            if preferred_topics:
                log.info("Using merged topics for data summary domain knowledge: %s", preferred_topics)
        ctx.summary_domain_knowledge = domain_knowledge
        vector_knowledge = (
            ctx.vector_knowledge_prompt
            if ctx.vector_knowledge is not None and ctx.vector_knowledge_source == "vector_active"
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
                vector_knowledge_bundle=ctx.vector_knowledge,
                response_mode=ctx.response_mode,
                resolution_policy=ctx.resolution_policy,
                grounding_policy=ctx.grounding_policy,
                comparison_focus=comparison_focus,
            )
            if not _is_summary_grounded(envelope, ctx):
                strict_grounding_retry = True
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
                        answer="I could not fully ground a detailed narrative from the provided data preview. "
                        "Please refine the query or narrow the period for a more precise grounded answer.",
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
        except RuntimeError:
            # Circuit breaker open — system-level issue, don't mask with fallback
            raise
        except Exception as e:
            metrics.log_summary_schema_failure()
            log.warning(
                "Structured summarization failed (%s): %s",
                type(e).__name__, e,
            )
            ctx.summary = llm_summarize(
                ctx.query,
                ctx.preview,
                ctx.stats_hint,
                ctx.lang_instruction,
                conversation_history=ctx.conversation_history,
                domain_knowledge=domain_knowledge,
                vector_knowledge=vector_knowledge,
            )
            ctx.summary_source = "legacy_text_fallback"
            ctx.summary_claims = []
            ctx.summary_citations = ["legacy_text_fallback"]
            ctx.summary_confidence = 0.5

    ctx.summary = scrub_schema_mentions(ctx.summary)
    trace_detail(
        log,
        ctx,
        "stage_4_summarize_data",
        "pre_gate",
        summary_source=ctx.summary_source,
        strict_grounding_retry=strict_grounding_retry,
        grounding_policy=ctx.grounding_policy,
        claims_count=len(ctx.summary_claims or []),
        citations=list(ctx.summary_citations or []),
        confidence=ctx.summary_confidence,
        summary_preview=ctx.summary,
    )
    _attach_claim_provenance(ctx)
    _enforce_provenance_gate(ctx)
    return ctx
