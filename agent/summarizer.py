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

from models import QueryContext
from core.llm import (
    llm_summarize,
    llm_summarize_structured,
    SummaryEnvelope,
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
    if ctx.df is not None and not ctx.df.empty:
        try:
            parts.append(ctx.df.head(200).to_string(index=False))
        except Exception:  # pragma: no cover - defensive
            pass
    return "\n".join(parts)


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
    return tokens


def _is_summary_grounded(envelope: SummaryEnvelope, ctx: QueryContext) -> bool:
    claim_text = "\n".join(envelope.claims or [])
    answer_tokens = _extract_number_tokens((envelope.answer or "") + "\n" + claim_text)
    if not answer_tokens:
        return True

    source_tokens = _build_grounding_tokens(ctx)
    if not source_tokens:
        return False

    matched = sum(1 for t in answer_tokens if t in source_tokens)
    match_ratio = matched / max(1, len(answer_tokens))
    return match_ratio >= 0.7


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
) -> tuple[List[Dict[str, Any]], float, List[str]]:
    token_index: Dict[str, List[Dict[str, Any]]] = {}
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
            refs = token_index.get(token, [])
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
    gate_passed = (not has_ungrounded_claim) and coverage >= PROVENANCE_MIN_COVERAGE
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


def answer_conceptual(ctx: QueryContext) -> QueryContext:
    """Generate an answer for conceptual/definitional questions (no SQL).

    Reads: ctx.query, ctx.lang_instruction, ctx.conversation_history
    Writes: ctx.summary
    """
    analyzer_active = ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active"
    routing_query = (
        ctx.question_analysis.canonical_query_en
        if analyzer_active and ctx.question_analysis is not None
        else ctx.query
    )
    preferred_topics = None
    if analyzer_active and ctx.question_analysis is not None:
        ranked_topics = sorted(
            ctx.question_analysis.knowledge.candidate_topics,
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        preferred_topics = [
            candidate.name.value
            for candidate in ranked_topics
            if candidate.score >= 0.25
        ][:3]
    query_lower = routing_query.lower()
    domain_knowledge = get_relevant_domain_knowledge(
        routing_query,
        use_cache=False,
        preferred_topics=preferred_topics,
    )
    if analyzer_active and preferred_topics:
        log.info("Using active question-analyzer topics for conceptual answer: %s", preferred_topics)
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

    try:
        envelope = llm_summarize_structured(
            ctx.query,
            data_preview="",
            stats_hint=conceptual_hint,
            lang_instruction=ctx.lang_instruction,
            conversation_history=ctx.conversation_history,
            domain_knowledge=domain_knowledge,
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
    except Exception as exc:
        metrics.log_summary_schema_failure()
        log.warning("Structured conceptual summarization failed: %s", exc)
        ctx.summary = llm_summarize(
            ctx.query,
            data_preview="",
            stats_hint=conceptual_hint,
            lang_instruction=ctx.lang_instruction,
            conversation_history=ctx.conversation_history,
            domain_knowledge=domain_knowledge,
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


def summarize_data(ctx: QueryContext) -> QueryContext:
    """Generate an answer from SQL query results.

    Reads: ctx.query, ctx.preview, ctx.stats_hint, ctx.lang_instruction,
           ctx.conversation_history, ctx.share_summary_override
    Writes: ctx.summary
    """
    strict_grounding_retry = False
    ctx.summary_source = ""

    if ctx.share_summary_override:
        ctx.summary = ctx.share_summary_override
        ctx.summary_source = "deterministic_share_summary"
        ctx.summary_claims = _derive_claims_from_text(ctx.summary)
        ctx.summary_citations = ["deterministic_share_summary"]
        ctx.summary_confidence = 1.0
    else:
        try:
            envelope = llm_summarize_structured(
                ctx.query,
                ctx.preview,
                ctx.stats_hint,
                ctx.lang_instruction,
                conversation_history=ctx.conversation_history,
            )
            if not _is_summary_grounded(envelope, ctx):
                strict_grounding_retry = True
                metrics.log_summary_grounding_failure()
                log.warning("Summary grounding check failed; retrying with strict grounding.")
                envelope = llm_summarize_structured(
                    ctx.query,
                    ctx.preview,
                    ctx.stats_hint,
                    ctx.lang_instruction,
                    conversation_history=ctx.conversation_history,
                    strict_grounding=True,
                )
                if not _is_summary_grounded(envelope, ctx):
                    metrics.log_summary_grounding_failure()
                    log.warning("Strict grounding retry still failed; using conservative fallback answer.")
                    envelope = SummaryEnvelope(
                        answer="I could not fully ground a detailed narrative from the provided data preview. "
                        "Please refine the query or narrow the period for a more precise grounded answer.",
                        claims=[],
                        citations=["guardrail_grounding_fallback"],
                        confidence=0.2,
                    )

            ctx.summary = envelope.answer
            ctx.summary_source = (
                "structured_summary_grounding_fallback"
                if "guardrail_grounding_fallback" in list(envelope.citations or [])
                else "structured_summary"
            )
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
            metrics.log_summary_schema_failure()
            log.warning(f"Structured summarization failed: {e}")
            ctx.summary = llm_summarize(
                ctx.query,
                ctx.preview,
                ctx.stats_hint,
                ctx.lang_instruction,
                conversation_history=ctx.conversation_history,
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
        claims_count=len(ctx.summary_claims or []),
        citations=list(ctx.summary_citations or []),
        confidence=ctx.summary_confidence,
        summary_preview=ctx.summary,
    )
    _attach_claim_provenance(ctx)
    _enforce_provenance_gate(ctx)
    return ctx
