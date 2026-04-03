"""
Pipeline Stage 1: Query Planning

Detects query type (conceptual vs data), analysis mode, language,
and generates the LLM plan + raw SQL.
"""
import json
import logging
import re
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from contracts.question_analysis import QuestionAnalysis, PreferredPath, QueryType, ToolName
from models import QueryContext
from core.llm import (
    llm_cache,
    make_gemini,
    llm_generate_plan_and_sql,
    get_relevant_domain_knowledge,
    llm_analyze_question,
)
from utils.language import detect_language, get_language_instruction
from utils.query_validation import is_conceptual_question, should_skip_sql_execution
from utils.trace_logging import trace_detail
from agent.aggregation import detect_aggregation_intent
from agent.tools.types import ToolInvocation
from agent.router import (
    extract_date_range,
    extract_currency,
    extract_price_metric,
    extract_balancing_entities,
    extract_tariff_entities,
    extract_generation_types,
)
from agent.analyzer import BALANCING_SHARE_METADATA
from agent.tools.composition_tools import ALLOWED_BALANCING_ENTITIES

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# Balancing entity normalization
# ---------------------------------------------------------------------------

# Build inverted indexes from BALANCING_SHARE_METADATA (e.g. analyzer.py:37-47).
# _COST_TO_ENTITIES: {"cheap": ["regulated_hpp", "deregulated_hydro"], ...}
# _LABEL_TO_ENTITY: {"regulated hpp": "regulated_hpp", "deregulated hydro": "deregulated_hydro", ...}
_ALLOWED_LOWER = {e.lower() for e in ALLOWED_BALANCING_ENTITIES}

_COST_TO_ENTITIES: dict[str, list[str]] = {}
_LABEL_TO_ENTITY: dict[str, str] = {}


def _normalize_balancing_entity_text(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[_/\\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


_ENTITY_ALIAS_TO_ENTITY: dict[str, str] = {
    "thermal generation ppa": "thermal_ppa",
    "thermal generation ppas": "thermal_ppa",
    "thermal ppa": "thermal_ppa",
    "thermal ppas": "thermal_ppa",
    "renewable ppas": "renewable_ppa",
    "old regulated tpps": "regulated_old_tpp",
    "new regulated tpps": "regulated_new_tpp",
    "regulated hpps": "regulated_hpp",
}

for _share_key, _meta in BALANCING_SHARE_METADATA.items():
    _entity = _share_key.removeprefix("share_")
    if _entity.lower() in _ALLOWED_LOWER:
        _COST_TO_ENTITIES.setdefault(_meta["cost"], []).append(_entity)
        _LABEL_TO_ENTITY[_normalize_balancing_entity_text(_meta["label"])] = _entity


def normalize_balancing_entities(raw_entities: list[str]) -> list[str] | None:
    """Expand semantic/cost-based entity names to valid tool identifiers.

    Returns:
        - A deduplicated list in ``ALLOWED_BALANCING_ENTITIES`` order when
          all requested entities resolve.
        - An empty list when *input* is empty (caller did not specify
          entities; tool should use its default = all).
        - ``None`` when input had entities but one or more could not be
          resolved, signalling *unresolved_concept*: the caller should not
          invoke the tool with a partial or implicit "all" fallback.
    """
    if not raw_entities:
        return []

    resolved: list[str] = []
    unresolved_seen = False
    for raw in raw_entities:
        val = _normalize_balancing_entity_text(raw)
        # 1. Already a valid entity?
        if val in _ALLOWED_LOWER:
            resolved.append(val)
            continue
        alias_entity = _ENTITY_ALIAS_TO_ENTITY.get(val)
        if alias_entity:
            resolved.append(alias_entity)
            continue
        # 2. Underscore-normalized form?
        val_under = val.replace(" ", "_")
        if val_under in _ALLOWED_LOWER:
            resolved.append(val_under)
            continue
        # 3. Cost-tier expansion ("cheap energy", "expensive sources", …)
        matched_cost = False
        for cost_key, entities in _COST_TO_ENTITIES.items():
            if cost_key in val:
                resolved.extend(entities)
                matched_cost = True
                break
        if matched_cost:
            continue
        # 4. Label substring match ("hydro" -> deregulated_hydro, etc.)
        matched_label = False
        for label, entity in _LABEL_TO_ENTITY.items():
            if val in label or label in val:
                resolved.append(entity)
                matched_label = True
        if matched_label:
            continue
        # 5. Unresolvable — log for observability
        log.warning("normalize_balancing_entities: dropping unresolvable entity %r", raw)
        unresolved_seen = True

    # Deduplicate while preserving ALLOWED order
    seen = set(resolved)
    out = [e for e in ALLOWED_BALANCING_ENTITIES if e.lower() in seen]
    if unresolved_seen or not out:
        # Input had entities but one or more did not resolve -> fail closed
        log.warning(
            "normalize_balancing_entities: unresolved entities detected, returning None. input=%r resolved=%r",
            raw_entities,
            out,
        )
        return None
    return out


_PRICE_METRIC_HINT_MAP: dict[str, tuple[str, str | None]] = {
    "balancing": ("balancing", None),
    "balancing_price": ("balancing", None),
    "balancing_price_gel": ("balancing", "gel"),
    "balancing_price_usd": ("balancing", "usd"),
    "p_bal_gel": ("balancing", "gel"),
    "p_bal_usd": ("balancing", "usd"),
    "deregulated": ("deregulated", None),
    "deregulated_price": ("deregulated", None),
    "deregulated_price_gel": ("deregulated", "gel"),
    "deregulated_price_usd": ("deregulated", "usd"),
    "p_dereg_gel": ("deregulated", "gel"),
    "p_dereg_usd": ("deregulated", "usd"),
    "guaranteed_capacity": ("guaranteed_capacity", None),
    "guaranteed_capacity_price": ("guaranteed_capacity", None),
    "guaranteed_capacity_price_gel": ("guaranteed_capacity", "gel"),
    "guaranteed_capacity_price_usd": ("guaranteed_capacity", "usd"),
    "p_gcap_gel": ("guaranteed_capacity", "gel"),
    "p_gcap_usd": ("guaranteed_capacity", "usd"),
    "exchange_rate": ("exchange_rate", None),
    "xrate": ("exchange_rate", None),
}


def normalize_price_metric_hint(raw_metric: str | None) -> tuple[str | None, str | None]:
    """Map analyzer metric hints to strict get_prices enums.

    The analyzer may emit semantic names or raw DB/alias names such as
    ``p_bal_gel`` or ``balancing_price_gel``. ``get_prices`` accepts only the
    strict tool enums, so planner must repair these hints deterministically.
    """
    if not raw_metric:
        return None, None

    key = str(raw_metric).strip().lower().replace(" ", "_")
    return _PRICE_METRIC_HINT_MAP.get(key, (None, None))


_TOOL_GRANULARITY_HINT_MAP: dict[str, str | None] = {
    "month": "monthly",
    "monthly": "monthly",
    "year": "yearly",
    "yearly": "yearly",
    # These values describe period shape, not typed-tool aggregation.
    # Treat them as "no explicit aggregation hint" and let the tool default apply.
    "range": None,
    "relative": None,
}


def normalize_tool_granularity_hint(raw_granularity: str | None) -> str | None:
    """Map analyzer period-granularity hints to typed-tool enum values.

    The question-analysis contract uses period words such as ``month`` and
    ``year`` while typed tools expect ``monthly`` / ``yearly``.
    Unsupported explicit aggregations (for example ``day`` or ``quarter``)
    must fail closed instead of silently downgrading to ``monthly``.
    """
    if not raw_granularity:
        return None

    key = str(raw_granularity).strip().lower().replace(" ", "_")
    if key not in _TOOL_GRANULARITY_HINT_MAP:
        raise ValueError(f"unsupported_tool_granularity_hint:{raw_granularity}")
    return _TOOL_GRANULARITY_HINT_MAP.get(key)


# ---------------------------------------------------------------------------
# Constants (moved from main.py)
# ---------------------------------------------------------------------------

ANALYTICAL_KEYWORDS = {
    "trend", "change", "growth", "increase", "decrease", "compare", "impact",
    "volatility", "pattern", "season", "relationship", "correlation", "evolution",
    "driver", "cause", "effect", "factor", "reason", "influence", "depend", "why", "behind",
    "payoff", "hypothetical", "scenario",
}


# ---------------------------------------------------------------------------
# Helpers (moved from main.py)
# ---------------------------------------------------------------------------

def detect_analysis_mode(user_query: str) -> str:
    """Detect if query requires analytical mode based on keywords.

    Priority: analyst keywords checked FIRST so that queries like
    "What is the trend in balancing price?" get analyst mode even
    though they also contain simple patterns like "what is".
    """
    query_lower = user_query.lower()

    # Deep analysis keywords -> analyst mode (HIGHEST PRIORITY)
    analyst_keywords = [
        "trend over time", "correlation", "driver", "impact on",
        "relationship between", "explain the dynamics", "analyze",
        "what drives", "what causes", "why does", "why did",
        # Scenario / what-if
        "what if", "hypothetical", "calculate payoff", "calculate income",
        "if price were", "if prices were", "contract for difference",
        "strike price of", "strike price sensitivity", "with strike",
        "cfd payoff", "cfd income", "cfd calculation", "cfd contract",
        "ppa contract", "what would be my income", "what would be my payoff",
        # Georgian
        "რამ გამოიწვია", "ტენდენცია", "კორელაცია", "დინამიკა", "ანალიზი",
        "რატომ", "რა იწვევს",
        "რა იქნებოდა თუ", "სცენარი",
        # Russian
        "что вызвало", "тренд", "корреляция", "динамика", "анализ",
        "почему", "что влияет",
        "что если", "сценарий", "рассчитать доход",
    ]
    if any(k in query_lower for k in analyst_keywords):
        return "analyst"

    # Broader analytical keywords (single-word triggers)
    if any(kw in query_lower for kw in ANALYTICAL_KEYWORDS):
        return "analyst"

    # Default: simple/factual queries
    return "light"



def _extract_plan_and_sql(combined_output: str) -> tuple[dict, str]:
    separator = "---SQL---"
    if separator not in combined_output:
        raise ValueError("LLM output malformed: missing '---SQL---' separator")

    plan_text, raw_sql = combined_output.split(separator, 1)
    normalized_sql = raw_sql.strip()
    if not normalized_sql:
        raise ValueError("LLM output malformed: SQL part is empty")

    parsed_plan = json.loads(plan_text.strip())
    if not isinstance(parsed_plan, dict):
        raise ValueError("LLM output malformed: plan is not a JSON object")
    normalized_plan = {
        "intent": str(parsed_plan.get("intent", "general")),
        "target": str(parsed_plan.get("target", "")),
        "period": str(parsed_plan.get("period", "")),
    }
    if "chart_strategy" in parsed_plan:
        normalized_plan["chart_strategy"] = str(parsed_plan.get("chart_strategy", ""))
    if "chart_groups" in parsed_plan and isinstance(parsed_plan.get("chart_groups"), list):
        normalized_plan["chart_groups"] = parsed_plan.get("chart_groups")
    return normalized_plan, normalized_sql


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4), reraise=True)
def _generate_plan_and_sql_with_retry(
    user_query: str,
    analysis_mode: str,
    lang_instruction: str,
    question_analysis: Optional[QuestionAnalysis] = None,
    vector_knowledge: str = "",
) -> tuple[dict, str]:
    kwargs = dict(
        user_query=user_query,
        analysis_mode=analysis_mode,
        lang_instruction=lang_instruction,
        question_analysis=question_analysis,
    )
    if vector_knowledge:
        kwargs["vector_knowledge"] = vector_knowledge
    combined_output = llm_generate_plan_and_sql(**kwargs)
    return _extract_plan_and_sql(combined_output)


def _build_plan_from_question_analysis(qa: QuestionAnalysis) -> dict[str, object]:
    """Project authoritative Stage 0.2 semantics into the legacy plan shape."""
    target = ""
    if qa.sql_hints.metric:
        target = qa.sql_hints.metric
    elif qa.analysis_requirements.derived_metrics:
        first_metric = qa.analysis_requirements.derived_metrics[0]
        target = str(first_metric.target_metric or first_metric.metric or "").strip()
    elif qa.sql_hints.entities:
        target = ", ".join(qa.sql_hints.entities[:3])
    elif qa.tooling.candidate_tools and qa.tooling.candidate_tools[0].params_hint is not None:
        params_hint = qa.tooling.candidate_tools[0].params_hint
        if params_hint.metric:
            target = params_hint.metric
        elif params_hint.entities:
            target = ", ".join(params_hint.entities[:3])
        elif params_hint.types:
            target = ", ".join(params_hint.types[:3])
    else:
        target = qa.classification.intent

    if not target:
        target = qa.classification.intent

    period = ""
    if qa.sql_hints.period is not None:
        period_info = qa.sql_hints.period
        period = (
            str(period_info.raw_text or "").strip()
            or f"{period_info.start_date} to {period_info.end_date}"
        )

    return {
        "intent": qa.classification.intent,
        "target": target,
        "period": period,
    }


def _merge_non_semantic_plan_fields(
    authoritative_plan: dict[str, object],
    generated_plan: Optional[dict[str, object]],
) -> dict[str, object]:
    """Preserve Stage 0.2 semantics while allowing legacy chart metadata through."""
    merged = dict(authoritative_plan)
    if not isinstance(generated_plan, dict):
        return merged

    for key in ("chart_strategy", "chart_groups"):
        if key in generated_plan:
            merged[key] = generated_plan[key]
    return merged


# ---------------------------------------------------------------------------
# Main pipeline stage
# ---------------------------------------------------------------------------

def prepare_context(ctx: QueryContext) -> QueryContext:
    """Stage 0: Fast context preparation before any heavy LLM call."""
    ctx.mode = detect_analysis_mode(ctx.query)
    log.info(f"Selected mode: {ctx.mode}")

    ctx.lang_code = detect_language(ctx.query)
    ctx.lang_instruction = get_language_instruction(ctx.lang_code)
    log.info(f"Detected language: {ctx.lang_code}")

    ctx.is_conceptual = is_conceptual_question(ctx.query)
    if ctx.is_conceptual:
        log.info("Conceptual question detected - skipping plan+SQL generation")

    return ctx


def analyze_question(ctx: QueryContext, *, source: str) -> QueryContext:
    """Stage 0.2: Run structured question analysis and stamp its source."""

    try:
        ctx.question_analysis = llm_analyze_question(
            user_query=ctx.query,
            conversation_history=ctx.conversation_history,
        )
        ctx.question_analysis_error = ""
        ctx.question_analysis_source = source
        log.info(
            "Question analyzer result | source=%s type=%s path=%s confidence=%.2f",
            source,
            ctx.question_analysis.classification.query_type.value,
            ctx.question_analysis.routing.preferred_path.value,
            ctx.question_analysis.classification.confidence,
        )
        trace_detail(
            log,
            ctx,
            "stage_0_2_question_analyzer",
            "validated",
            source=source,
            query_type=ctx.question_analysis.classification.query_type.value,
            preferred_path=ctx.question_analysis.routing.preferred_path.value,
            confidence=ctx.question_analysis.classification.confidence,
            candidate_topics=[topic.name.value for topic in ctx.question_analysis.knowledge.candidate_topics],
            candidate_tools=[tool.name.value for tool in ctx.question_analysis.tooling.candidate_tools],
            canonical_query_en=ctx.question_analysis.canonical_query_en,
            derived_metrics=[
                m.model_dump(mode="json")
                for m in ctx.question_analysis.analysis_requirements.derived_metrics
            ],
        )
        trace_detail(
            log,
            ctx,
            "stage_0_2_question_analyzer",
            "artifact",
            debug=True,
            question_analysis=ctx.question_analysis,
        )
    except Exception as exc:
        ctx.question_analysis = None
        ctx.question_analysis_error = str(exc)
        ctx.question_analysis_source = f"{source}_error"
        log.warning("Question analyzer failed | source=%s error=%s", source, exc)
        trace_detail(
            log,
            ctx,
            "stage_0_2_question_analyzer",
            "error",
            source=source,
            error=str(exc),
        )
    return ctx


def analyze_question_shadow(ctx: QueryContext) -> QueryContext:
    """Stage 0.2: Run structured question analysis without changing routing behavior."""
    return analyze_question(ctx, source="llm_shadow")


def analyze_question_active(ctx: QueryContext) -> QueryContext:
    """Stage 0.2: Run structured question analysis for downstream hint consumption."""
    return analyze_question(ctx, source="llm_active")


def generate_plan(ctx: QueryContext) -> QueryContext:
    """Stage 1: Generate plan + SQL using LLM fallback path.

    Expects prepare_context() to already have been called in pipeline.
    """
    if not ctx.lang_instruction or not ctx.mode:
        ctx = prepare_context(ctx)
    if ctx.is_conceptual:
        return ctx

    authoritative_qa = ctx.has_authoritative_question_analysis
    authoritative_plan = (
        _build_plan_from_question_analysis(ctx.question_analysis)
        if authoritative_qa and ctx.question_analysis is not None
        else None
    )

    # Generate plan + SQL in one LLM call
    try:
        retry_kwargs = dict(
            user_query=ctx.query,
            analysis_mode=ctx.mode,
            lang_instruction=ctx.lang_instruction,
            question_analysis=(
                ctx.question_analysis
                if authoritative_qa
                else None
            ),
        )
        vector_knowledge = (
            ctx.vector_knowledge_prompt
            if ctx.vector_knowledge is not None and ctx.vector_knowledge_source == "vector_active"
            else ""
        )
        if vector_knowledge:
            retry_kwargs["vector_knowledge"] = vector_knowledge
        generated_plan, ctx.raw_sql = _generate_plan_and_sql_with_retry(**retry_kwargs)
        ctx.plan = (
            _merge_non_semantic_plan_fields(authoritative_plan, generated_plan)
            if authoritative_plan is not None
            else generated_plan
        )

    except Exception as exc:
        log.warning("Strict plan parsing failed after retries, attempting SQL salvage: %s", exc)
        kwargs = dict(
            user_query=ctx.query,
            analysis_mode=ctx.mode,
            lang_instruction=ctx.lang_instruction,
            question_analysis=(
                ctx.question_analysis
                if authoritative_qa
                else None
            ),
        )
        vector_knowledge = (
            ctx.vector_knowledge_prompt
            if ctx.vector_knowledge is not None and ctx.vector_knowledge_source == "vector_active"
            else ""
        )
        if vector_knowledge:
            kwargs["vector_knowledge"] = vector_knowledge
        combined_output = llm_generate_plan_and_sql(**kwargs)
        separator = "---SQL---"
        if separator not in combined_output:
            log.exception("Combined Plan/SQL generation failed (missing separator)")
            raise
        plan_text, raw_sql = combined_output.split(separator, 1)
        ctx.raw_sql = raw_sql.strip()
        if not ctx.raw_sql:
            log.exception("Combined Plan/SQL generation failed (empty SQL)")
            raise ValueError("LLM output malformed: SQL part is empty")
        try:
            parsed_plan = json.loads(plan_text.strip())
            generated_plan = (
                parsed_plan
                if isinstance(parsed_plan, dict)
                else {"intent": "general", "target": "", "period": ""}
            )
            ctx.plan = (
                _merge_non_semantic_plan_fields(authoritative_plan, generated_plan)
                if authoritative_plan is not None
                else generated_plan
            )
        except json.JSONDecodeError:
            log.warning("Plan JSON decoding failed, defaulting to general plan.")
            ctx.plan = (
                dict(authoritative_plan)
                if authoritative_plan is not None
                else {"intent": "general", "target": "", "period": ""}
            )

    log.info(f"Plan: {ctx.plan}")
    trace_detail(
        log,
        ctx,
        "stage_1_generate_plan",
        "plan_ready",
        question_analysis_used=authoritative_qa,
        question_analysis_source=ctx.question_analysis_source,
        plan=ctx.plan,
        raw_sql_present=bool(ctx.raw_sql),
        raw_sql_len=len(ctx.raw_sql or ""),
    )
    trace_detail(
        log,
        ctx,
        "stage_1_generate_plan",
        "artifact",
        debug=True,
        plan=ctx.plan,
        raw_sql=ctx.raw_sql or "",
    )

    # Check if SQL should be skipped
    ctx.skip_sql, ctx.skip_sql_reason = should_skip_sql_execution(ctx.query, ctx.plan)
    if ctx.skip_sql:
        log.info(f"Skipping SQL execution: {ctx.skip_sql_reason}")

    # Detect aggregation intent
    ctx.aggregation_intent = detect_aggregation_intent(ctx.query)
    log.info(f"Aggregation intent: {ctx.aggregation_intent}")

    return ctx


# -----------------------------------------------------------------------
# QuestionAnalysis → ToolInvocation bridge
# -----------------------------------------------------------------------

# Minimum score on the top tool candidate for the analyzer to drive routing.
from config import ANALYZER_TOOL_MIN_SCORE as _ANALYZER_TOOL_MIN_SCORE


def build_tool_invocation_from_analysis(
    qa: QuestionAnalysis,
    raw_query: str,
) -> Optional[ToolInvocation]:
    """Convert the LLM question-analyzer output into a concrete ToolInvocation.

    Returns ``None`` when:
    - ``preferred_path`` is not ``tool`` or ``prefer_tool`` is False,
    - no candidate tool meets the minimum score threshold,
    - parameter resolution fails for the chosen tool.

    The function reuses the deterministic parameter extractors from
    ``agent.router`` so that dates, entities, metrics, and currency are
    resolved identically regardless of whether the keyword router or the
    LLM analyzer drove the routing decision.
    """
    # Primary gate: require preferred_path == TOOL.  The prefer_tool flag
    # acts as a soft boost only for ambiguous paths — it must NOT override
    # an explicit SQL or KNOWLEDGE recommendation.
    if qa.routing.preferred_path != PreferredPath.TOOL:
        if not qa.routing.prefer_tool or qa.routing.preferred_path in (
            PreferredPath.SQL,
            PreferredPath.KNOWLEDGE,
        ):
            return None

    candidates = qa.tooling.candidate_tools
    if not candidates:
        return None

    top = candidates[0]
    if top.score < _ANALYZER_TOOL_MIN_SCORE:
        log.info(
            "Analyzer top tool score too low: tool=%s score=%.2f (min=%.2f)",
            top.name.value, top.score, _ANALYZER_TOOL_MIN_SCORE,
        )
        return None

    tool_name = top.name.value
    # Let ValueError propagate — the pipeline catches it and sets
    # _analyzer_tool_failed to prevent the agent loop from running.
    params = resolve_tool_params(qa, tool_name, raw_query, hint=top.params_hint)

    if params is None:
        return None

    reason = f"analyzer:{top.reason or top.name.value} (score={top.score:.2f})"
    return ToolInvocation(
        name=tool_name,
        params=params,
        confidence=top.score,
        reason=reason,
    )


def resolve_tool_params(
    qa: QuestionAnalysis,
    tool_name: str,
    raw_query: str,
    *,
    hint: "Optional[object]" = None,
) -> Optional[dict]:
    """Resolve concrete tool parameters from analyzer output and regex extractors.

    Shared by ``build_tool_invocation_from_analysis`` and
    ``evidence_planner.build_evidence_plan`` so that parameter resolution is
    identical regardless of the call site.

    Returns ``None`` only for unknown tools.  Raises ``ValueError`` for
    unresolvable entity references so the caller can decide on fallback.
    """
    # Use the canonical English query for parameter extraction so that
    # keyword extractors work reliably even for non-English input.
    effective_query = (qa.canonical_query_en or raw_query).lower()

    # --- Resolve dates ---
    # Prefer the analyzer's structured period; fall back to regex extraction.
    start_date = None
    end_date = None
    if hint and getattr(hint, "start_date", None):
        start_date = hint.start_date
    if hint and getattr(hint, "end_date", None):
        end_date = hint.end_date
    if (not start_date or not end_date) and qa.sql_hints.period:
        start_date = start_date or qa.sql_hints.period.start_date
        end_date = end_date or qa.sql_hints.period.end_date
    if not start_date or not end_date:
        regex_start, regex_end = extract_date_range(effective_query)
        start_date = start_date or regex_start
        end_date = end_date or regex_end

    # Sanity check: if the query is a comparison or trend but dates collapsed
    # to a single point, fall back to regex for a wider range.
    _RANGE_QUERY_TYPES = {QueryType.COMPARISON, QueryType.DATA_EXPLANATION, QueryType.DATA_RETRIEVAL}
    if qa.classification.query_type in _RANGE_QUERY_TYPES and start_date and start_date == end_date:
        regex_start, regex_end = extract_date_range(effective_query)
        if regex_start and regex_end and regex_start != regex_end:
            log.info(
                "Analyzer returned single-point dates (%s) for %s query; "
                "expanding to regex range %s–%s",
                start_date, qa.classification.query_type.value,
                regex_start, regex_end,
            )
            start_date, end_date = regex_start, regex_end

    params: dict = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    # --- Tool-specific parameter resolution ---
    if tool_name == ToolName.GET_PRICES.value:
        hint_metric = hint.metric if hint and getattr(hint, "metric", None) else None
        normalized_metric, implied_currency = normalize_price_metric_hint(hint_metric)
        metric = normalized_metric or extract_price_metric(effective_query)
        hint_currency = str(hint.currency).strip().lower() if hint and getattr(hint, "currency", None) else None
        if implied_currency and hint_currency and hint_currency != implied_currency:
            log.warning(
                "Analyzer get_prices hint had contradictory metric/currency pair; "
                "using alias-implied currency. metric=%r hint_currency=%r implied_currency=%r",
                hint_metric,
                hint_currency,
                implied_currency,
            )
        currency = implied_currency or hint_currency or extract_currency(effective_query)
        granularity = normalize_tool_granularity_hint(getattr(hint, "granularity", None) if hint else None) or "monthly"
        params.update({"metric": metric, "currency": currency, "granularity": granularity})

    elif tool_name == ToolName.GET_TARIFFS.value:
        entities = (hint.entities if hint and getattr(hint, "entities", None) else []) or extract_tariff_entities(effective_query)
        currency = (hint.currency if hint and getattr(hint, "currency", None) else None) or extract_currency(effective_query)
        if entities:
            params["entities"] = entities
        params["currency"] = currency

    elif tool_name == ToolName.GET_GENERATION_MIX.value:
        types = (hint.types if hint and getattr(hint, "types", None) else []) or extract_generation_types(effective_query)
        mode = (getattr(hint, "mode", None) if hint else None) or "quantity"
        granularity = normalize_tool_granularity_hint(getattr(hint, "granularity", None) if hint else None) or "monthly"
        if types:
            params["types"] = types
        params.update({"mode": mode, "granularity": granularity})

    elif tool_name == ToolName.GET_BALANCING_COMPOSITION.value:
        raw_entities = (hint.entities if hint and getattr(hint, "entities", None) else []) or extract_balancing_entities(effective_query)
        entities = normalize_balancing_entities(raw_entities)
        if entities is None:
            # Entities were specified but none resolved → unresolved_concept.
            # Abort rather than silently broadening to "all entities".
            raise ValueError(
                "unresolved_balancing_entities:"
                + (",".join(str(entity).strip() for entity in raw_entities if str(entity).strip()) or "unknown")
            )
        if entities:
            params["entities"] = entities
        # empty list (no entities specified) → tool fetches all (safe default)

    else:
        log.warning("Unknown tool from analyzer: %s", tool_name)
        return None

    return params
