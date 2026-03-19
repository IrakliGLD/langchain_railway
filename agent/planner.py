"""
Pipeline Stage 1: Query Planning

Detects query type (conceptual vs data), analysis mode, language,
and generates the LLM plan + raw SQL.
"""
import json
import logging
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

log = logging.getLogger("Enai")


# ---------------------------------------------------------------------------
# Constants (moved from main.py)
# ---------------------------------------------------------------------------

ANALYTICAL_KEYWORDS = {
    "trend", "change", "growth", "increase", "decrease", "compare", "impact",
    "volatility", "pattern", "season", "relationship", "correlation", "evolution",
    "driver", "cause", "effect", "factor", "reason", "influence", "depend", "why", "behind"
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
        # Georgian
        "რამ გამოიწვია", "ტენდენცია", "კორელაცია", "დინამიკა", "ანალიზი",
        "რატომ", "რა იწვევს",
        # Russian
        "что вызвало", "тренд", "корреляция", "динамика", "анализ",
        "почему", "что влияет",
    ]
    if any(k in query_lower for k in analyst_keywords):
        return "analyst"

    # Broader analytical keywords (single-word triggers)
    if any(kw in query_lower for kw in ANALYTICAL_KEYWORDS):
        return "analyst"

    # Default: simple/factual queries
    return "light"


def llm_analyze_with_domain_knowledge(user_query: str, lang_instruction: str) -> str:
    """First LLM call: Pure reasoning using domain knowledge.
    Forces the model to think like an energy analyst BEFORE writing SQL.
    """
    cache_input = f"domain_reasoning|{user_query}|{lang_instruction}"
    cached_response = llm_cache.get(cache_input)
    if cached_response:
        log.info("Domain Reasoning: (cached)")
        return cached_response

    system = (
        "You are a senior energy market analyst for Georgia. "
        "Interpret the user's question using ONLY the domain knowledge. "
        "Do not write SQL. Do not mention tables. "
        "Answer in 3 parts: "
        "1. Intent: What is the user asking? (e.g., price trend, share, driver) "
        "2. Key Concepts: List domain concepts involved "
        "3. Reasoning: Explain using domain knowledge "
        f"{lang_instruction}"
    )
    domain_json = get_relevant_domain_knowledge(user_query, use_cache=False)
    prompt = f"""
User question: {user_query}

Domain Knowledge:
{domain_json}

Respond in structured text only.
"""
    try:
        llm = make_gemini()
        response = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        log.info(f"Domain Reasoning:\n{response}")
        llm_cache.set(cache_input, response)
        return response
    except Exception as e:
        log.warning(f"Domain reasoning failed: {e}. Using fallback.")
        fallback = "Intent: general\nKey Concepts: balancing price\nReasoning: Use xrate and entity shares from trade_derived_entities."
        llm_cache.set(cache_input, fallback)
        return fallback


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

    # Generate plan + SQL in one LLM call
    try:
        retry_kwargs = dict(
            user_query=ctx.query,
            analysis_mode=ctx.mode,
            lang_instruction=ctx.lang_instruction,
            question_analysis=(
                ctx.question_analysis
                if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active"
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
        ctx.plan, ctx.raw_sql = _generate_plan_and_sql_with_retry(**retry_kwargs)

    except Exception as exc:
        log.warning("Strict plan parsing failed after retries, attempting SQL salvage: %s", exc)
        kwargs = dict(
            user_query=ctx.query,
            analysis_mode=ctx.mode,
            lang_instruction=ctx.lang_instruction,
            question_analysis=(
                ctx.question_analysis
                if ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active"
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
            ctx.plan = parsed_plan if isinstance(parsed_plan, dict) else {"intent": "general", "target": "", "period": ""}
        except json.JSONDecodeError:
            log.warning("Plan JSON decoding failed, defaulting to general plan.")
            ctx.plan = {"intent": "general", "target": "", "period": ""}

    log.info(f"Plan: {ctx.plan}")
    trace_detail(
        log,
        ctx,
        "stage_1_generate_plan",
        "plan_ready",
        question_analysis_used=bool(
            ctx.question_analysis is not None and ctx.question_analysis_source == "llm_active"
        ),
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
_ANALYZER_TOOL_MIN_SCORE = 0.55


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
    hint = top.params_hint
    # Use the canonical English query for parameter extraction so that
    # keyword extractors work reliably even for non-English input.
    effective_query = (qa.canonical_query_en or raw_query).lower()

    # --- Resolve dates ---
    # Prefer the analyzer's structured period; fall back to regex extraction.
    start_date = None
    end_date = None
    if hint and hint.start_date:
        start_date = hint.start_date
    if hint and hint.end_date:
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
        metric = (hint.metric if hint and hint.metric else None) or extract_price_metric(effective_query)
        currency = (hint.currency if hint and hint.currency else None) or extract_currency(effective_query)
        granularity = (hint.granularity if hint and hint.granularity else None) or "monthly"
        params.update({"metric": metric, "currency": currency, "granularity": granularity})

    elif tool_name == ToolName.GET_TARIFFS.value:
        entities = (hint.entities if hint and hint.entities else []) or extract_tariff_entities(effective_query)
        currency = (hint.currency if hint and hint.currency else None) or extract_currency(effective_query)
        if entities:
            params["entities"] = entities
        params["currency"] = currency

    elif tool_name == ToolName.GET_GENERATION_MIX.value:
        types = (hint.types if hint and hint.types else []) or extract_generation_types(effective_query)
        mode = (hint.mode if hint and hint.mode else None) or "quantity"
        granularity = (hint.granularity if hint and hint.granularity else None) or "monthly"
        if types:
            params["types"] = types
        params.update({"mode": mode, "granularity": granularity})

    elif tool_name == ToolName.GET_BALANCING_COMPOSITION.value:
        entities = (hint.entities if hint and hint.entities else []) or extract_balancing_entities(effective_query)
        if entities:
            params["entities"] = entities

    else:
        log.warning("Unknown tool from analyzer: %s", tool_name)
        return None

    reason = f"analyzer:{top.reason or top.name.value} (score={top.score:.2f})"
    return ToolInvocation(
        name=tool_name,
        params=params,
        confidence=top.score,
        reason=reason,
    )
