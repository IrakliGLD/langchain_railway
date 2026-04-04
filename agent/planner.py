"""
Pipeline Stage 1: Query Planning

Detects query type (conceptual vs data), analysis mode, language,
and generates the LLM plan + raw SQL.
"""
import json
import logging
import re
from calendar import monthrange
from datetime import date
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


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    """Parse an ISO date string safely."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _expand_single_month_explanation_window(
    qa: QuestionAnalysis,
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Widen single-month explanation queries so derived metrics can be computed.

    For questions such as "Explain the change in balancing price in May 2024",
    the analyzer often requests MoM/YoY metrics plus historical same-month
    context.  A single-month fetch cannot satisfy those requests, so expand the
    start date to five years before the target month while preserving the same
    end date.
    """
    period = getattr(qa.sql_hints, "period", None)
    derived_metrics = list(getattr(qa.analysis_requirements, "derived_metrics", []) or [])
    if (
        qa.classification.query_type != QueryType.DATA_EXPLANATION
        or not derived_metrics
        or period is None
        or getattr(period, "kind", None) != "month"
    ):
        return start_date, end_date

    start_dt = _parse_iso_date(start_date)
    end_dt = _parse_iso_date(end_date)
    if start_dt is None or end_dt is None:
        return start_date, end_date
    if (start_dt.year, start_dt.month) != (end_dt.year, end_dt.month):
        return start_date, end_date
    if start_dt.day != 1:
        return start_date, end_date

    expanded_start = date(start_dt.year - 5, start_dt.month, 1).isoformat()
    if expanded_start == start_date:
        return start_date, end_date

    log.info(
        "Expanding single-month data explanation window for derived metrics: %s–%s -> %s–%s",
        start_date,
        end_date,
        expanded_start,
        end_date,
    )
    return expanded_start, end_date


_BALANCING_PRICE_EXPLANATION_TOKENS = (
    "why",
    "reason",
    "reasons",
    "explain",
    "cause",
    "caused",
    "changed",
    "change",
    "changes",
    "რატომ",
    "მიზეზ",
    "ახსენ",
    "почему",
    "причин",
    "объясн",
)

_BALANCING_PRICE_BALANCING_TOKENS = (
    "balancing",
    "balance market",
    "საბალანს",
    "баланс",
)

_BALANCING_PRICE_PRICE_TOKENS = (
    "price",
    "prices",
    "ფას",
    "цен",
)

_BALANCING_PRICE_FORECAST_TOKENS = (
    "forecast",
    "forecasting",
    "predict",
    "prediction",
    "projection",
    "projected",
    "outlook",
    "extrapolat",
)

_BALANCING_PRICE_FORECAST_CONCEPTUAL_TOKENS = (
    "why",
    "how ",
    "how?",
    "difficult",
    "difficulty",
    "reliable",
    "reliability",
    "uncertain",
    "uncertainty",
    "caveat",
    "caveats",
    "assumption",
    "assumptions",
    "method",
    "methodology",
)

_NUMERIC_CALCULATION_TOKENS = (
    "calculate",
    "weighted average",
    "average price",
    "weighted avg",
    "mean price",
)

_UNDERDEFINED_SCOPE_TOKENS = (
    "remaining",
    "residual",
    "leftover",
    "everything except",
    "excluding",
    "except",
)

_NUMERIC_PRICE_CONTEXT_TOKENS = (
    "price",
    "prices",
    "tariff",
    "tariffs",
    "gel",
    "usd",
    "share",
    "shares",
)

_RESIDUAL_SCOPE_TOKENS = _UNDERDEFINED_SCOPE_TOKENS + (
    "other electricity",
    "other energy",
)

_RESIDUAL_HYDRO_TOKENS = (
    "regulated hydro",
    "regulated hpp",
    "regulated hydro generation",
)

_RESIDUAL_THERMAL_TOKENS = (
    "regulated thermal",
    "regulated thermals",
    "regulated tpp",
    "regulated old tpp",
    "regulated new tpp",
    "thermal generation",
    "thermal tariffs",
    "thermal",
)

_RESIDUAL_DEREGULATED_TOKENS = (
    "deregulated hydro",
    "deregulated power plant",
    "deregulated power plants",
    "deregulated hpp",
)

_MONTH_PATTERN_BY_NUMBER = {
    1: r"\b(january|jan|январ[ьяею]?)\b|იანვ",
    2: r"\b(february|feb|феврал[ьяею]?)\b|თებ",
    3: r"\b(march|mar|март[аеу]?)\b|მარტ",
    4: r"\b(april|apr|апрел[ьяею]?)\b|აპრ",
    5: r"\b(may|май|мая)\b|მაის",
    6: r"\b(june|jun|июн[ьяею]?)\b|ივნ",
    7: r"\b(july|jul|июл[ьяею]?)\b|ივლ",
    8: r"\b(august|aug|август[аеу]?)\b|აგვ",
    9: r"\b(september|sept|sep|сентябр[ьяею]?)\b|სექტ",
    10: r"\b(october|oct|октябр[ьяею]?)\b|ოქტ",
    11: r"\b(november|nov|ноябр[ьяею]?)\b|ნოემბ",
    12: r"\b(december|dec|декабр[ьяею]?)\b|დეკ",
}


def _extract_explicit_month_period(raw_query: str) -> tuple[Optional[str], Optional[str]]:
    """Extract a concrete month-year period directly from the raw query text."""
    query_lower = str(raw_query or "").lower()
    year_match = re.search(r"\b(20\d{2})\b", query_lower)
    if not year_match:
        return None, None
    year = int(year_match.group(1))

    matched_months = [
        month_num
        for month_num, pattern in _MONTH_PATTERN_BY_NUMBER.items()
        if re.search(pattern, query_lower)
    ]
    if len(matched_months) != 1:
        return None, None
    month = matched_months[0]

    last_day = monthrange(year, month)[1]
    return date(year, month, 1).isoformat(), date(year, month, last_day).isoformat()


def _is_month_specific_balancing_price_explanation(raw_query: str) -> bool:
    """Detect month-specific balancing-price why/explanation questions.

    This guardrail catches phrasing such as "why balancing electricity prices
    changed in November 2024" so the runtime analyzer cannot misroute it to
    conceptual/knowledge mode.
    """
    query_lower = str(raw_query or "").lower()
    if not query_lower:
        return False
    if not any(token in query_lower for token in _BALANCING_PRICE_EXPLANATION_TOKENS):
        return False
    if not any(token in query_lower for token in _BALANCING_PRICE_BALANCING_TOKENS):
        return False
    if not any(token in query_lower for token in _BALANCING_PRICE_PRICE_TOKENS):
        return False

    start_date, end_date = _extract_explicit_month_period(raw_query)
    if not start_date or not end_date:
        start_date, end_date = extract_date_range(raw_query)
    start_dt = _parse_iso_date(start_date)
    end_dt = _parse_iso_date(end_date)
    if start_dt is None or end_dt is None:
        return False
    return (start_dt.year, start_dt.month) == (end_dt.year, end_dt.month)


def _apply_balancing_month_explanation_guardrail(
    qa: QuestionAnalysis,
    raw_query: str,
) -> tuple[QuestionAnalysis, bool]:
    """Coerce obvious month-specific balancing-price why-queries to tool mode."""
    if not _is_month_specific_balancing_price_explanation(raw_query):
        return qa, False

    already_supported = (
        qa.classification.query_type == QueryType.DATA_EXPLANATION
        and qa.routing.preferred_path == PreferredPath.TOOL
    )
    if already_supported:
        return qa, False

    start_date, end_date = _extract_explicit_month_period(raw_query)
    if not start_date or not end_date:
        start_date, end_date = extract_date_range(raw_query)
    payload = qa.model_dump(mode="json")

    payload["classification"]["query_type"] = QueryType.DATA_EXPLANATION.value
    payload["classification"]["analysis_mode"] = "analyst"
    payload["classification"]["needs_clarification"] = False
    payload["classification"]["ambiguities"] = []
    payload["classification"]["confidence"] = max(
        float(payload["classification"].get("confidence", 0.0)),
        0.95,
    )

    payload["routing"].update({
        "preferred_path": PreferredPath.TOOL.value,
        "needs_sql": False,
        "needs_knowledge": False,
        "prefer_tool": True,
        "needs_multi_tool": True,
        "evidence_roles": [
            "primary_data",
            "composition_context",
        ],
    })

    payload["tooling"]["candidate_tools"] = [
        {
            "name": ToolName.GET_PRICES.value,
            "score": 1.0,
            "reason": "balancing price history for monthly driver explanation",
            "params_hint": {
                "metric": "balancing",
                "currency": "both",
                "granularity": "monthly",
                "start_date": start_date,
                "end_date": end_date,
                "entities": [],
                "types": [],
            },
        },
        {
            "name": ToolName.GET_BALANCING_COMPOSITION.value,
            "score": 0.95,
            "reason": "balancing composition needed to explain price changes",
            "params_hint": {
                "start_date": start_date,
                "end_date": end_date,
                "entities": [],
                "types": [],
            },
        },
    ]

    period_payload = payload.get("sql_hints", {}).get("period")
    start_dt = _parse_iso_date(start_date)
    end_dt = _parse_iso_date(end_date)
    if (
        not isinstance(period_payload, dict)
        and start_dt is not None
        and end_dt is not None
        and (start_dt.year, start_dt.month) == (end_dt.year, end_dt.month)
    ):
        payload.setdefault("sql_hints", {})
        payload["sql_hints"]["period"] = {
            "kind": "month",
            "start_date": start_date,
            "end_date": end_date,
            "granularity": "month",
            "raw_text": f"{start_dt.strftime('%B')} {start_dt.year}",
        }
    payload.setdefault("sql_hints", {})
    if not payload["sql_hints"].get("metric"):
        payload["sql_hints"]["metric"] = "balancing"

    payload.setdefault("knowledge", {})
    topic_rows = [
        topic
        for topic in payload["knowledge"].get("candidate_topics", [])
        if isinstance(topic, dict) and topic.get("name")
    ]
    topic_names = {topic.get("name") for topic in topic_rows}
    for name, score in (
        ("balancing_price", 1.0),
        ("currency_influence", 0.7),
        ("seasonal_patterns", 0.55),
    ):
        if name not in topic_names:
            topic_rows.append({
                "name": name,
                "score": score,
            })
    topic_rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    payload["knowledge"]["candidate_topics"] = topic_rows[:5]

    forced_metrics = [
        {"metric_name": "mom_absolute_change", "metric": "balancing"},
        {"metric_name": "mom_percent_change", "metric": "balancing"},
        {"metric_name": "yoy_absolute_change", "metric": "balancing"},
        {"metric_name": "yoy_percent_change", "metric": "balancing"},
        {"metric_name": "mom_absolute_change", "metric": "exchange_rate"},
        {"metric_name": "share_delta_mom", "metric": "share_import"},
        {"metric_name": "share_delta_mom", "metric": "share_regulated_hpp"},
        {"metric_name": "share_delta_mom", "metric": "share_regulated_old_tpp"},
        {"metric_name": "share_delta_mom", "metric": "share_renewable_ppa"},
        {"metric_name": "share_delta_mom", "metric": "share_thermal_ppa"},
    ]
    existing_metrics = payload["analysis_requirements"].get("derived_metrics", []) or []
    merged_metrics = []
    seen_metric_keys: set[tuple[str, str | None]] = set()
    for item in list(existing_metrics) + forced_metrics:
        if not isinstance(item, dict):
            continue
        metric_name = str(item.get("metric_name") or "").strip()
        metric = str(item.get("metric") or "").strip() or None
        if not metric_name:
            continue
        key = (metric_name, metric)
        if key in seen_metric_keys:
            continue
        seen_metric_keys.add(key)
        merged_metrics.append(item)
    payload["analysis_requirements"]["needs_driver_analysis"] = True
    payload["analysis_requirements"]["needs_correlation_context"] = False
    payload["analysis_requirements"]["derived_metrics"] = merged_metrics

    return QuestionAnalysis.model_validate(payload), True


def _is_simple_balancing_price_forecast_query(
    qa: QuestionAnalysis,
    raw_query: str,
) -> bool:
    """Return True when a plain balancing-price forecast should use tools."""

    if qa.routing.preferred_path != PreferredPath.KNOWLEDGE:
        return False
    if qa.classification.query_type not in (
        QueryType.AMBIGUOUS,
        QueryType.UNSUPPORTED,
        QueryType.FORECAST,
    ):
        return False

    query_lower = str(raw_query or "").lower()
    if not query_lower:
        return False
    if not any(token in query_lower for token in _BALANCING_PRICE_FORECAST_TOKENS):
        return False
    if not any(token in query_lower for token in _BALANCING_PRICE_BALANCING_TOKENS):
        return False
    if not any(token in query_lower for token in _BALANCING_PRICE_PRICE_TOKENS):
        return False
    if any(token in query_lower for token in _BALANCING_PRICE_FORECAST_CONCEPTUAL_TOKENS):
        return False

    tool_names = {
        getattr(tool.name, "value", str(tool.name or ""))
        for tool in (qa.tooling.candidate_tools or [])
    }
    if tool_names and ToolName.GET_PRICES.value not in tool_names:
        return False

    topic_names = {
        getattr(topic.name, "value", str(topic.name or ""))
        for topic in (qa.knowledge.candidate_topics or [])
    }
    if topic_names and "balancing_price" not in topic_names:
        return False

    return True


def _apply_balancing_price_forecast_guardrail(
    qa: QuestionAnalysis,
    raw_query: str,
) -> tuple[QuestionAnalysis, bool]:
    """Coerce obvious balancing-price forecasts to tool mode."""

    if not _is_simple_balancing_price_forecast_query(qa, raw_query):
        return qa, False

    payload = qa.model_dump(mode="json")
    payload["classification"]["query_type"] = QueryType.FORECAST.value
    payload["classification"]["analysis_mode"] = "analyst"
    payload["classification"]["needs_clarification"] = False
    payload["classification"]["ambiguities"] = []
    payload["classification"]["confidence"] = max(
        float(payload["classification"].get("confidence", 0.0)),
        0.85,
    )

    payload["routing"].update({
        "preferred_path": PreferredPath.TOOL.value,
        "needs_sql": False,
        "needs_knowledge": False,
        "prefer_tool": True,
        "needs_multi_tool": False,
        "evidence_roles": ["primary_data"],
    })

    payload["tooling"]["candidate_tools"] = [
        {
            "name": ToolName.GET_PRICES.value,
            "score": 1.0,
            "reason": "historical balancing prices needed for forecast extrapolation",
            "params_hint": {
                "metric": "balancing",
                "currency": "both",
                "granularity": "monthly",
                "start_date": None,
                "end_date": None,
                "entities": [],
                "types": [],
            },
        }
    ]

    payload.setdefault("sql_hints", {})
    if not payload["sql_hints"].get("metric"):
        payload["sql_hints"]["metric"] = "balancing"

    payload.setdefault("knowledge", {})
    topic_rows = [
        topic
        for topic in payload["knowledge"].get("candidate_topics", [])
        if isinstance(topic, dict) and topic.get("name")
    ]
    topic_names = {topic.get("name") for topic in topic_rows}
    for name, score in (
        ("balancing_price", 1.0),
        ("seasonal_patterns", 0.7),
    ):
        if name not in topic_names:
            topic_rows.append({
                "name": name,
                "score": score,
            })
    topic_rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    payload["knowledge"]["candidate_topics"] = topic_rows[:5]

    forecast_metric = {"metric_name": "trend_slope", "metric": "balancing"}
    existing_metrics = payload["analysis_requirements"].get("derived_metrics", []) or []
    merged_metrics = []
    seen_metric_keys: set[tuple[str, str | None]] = set()
    for item in list(existing_metrics) + [forecast_metric]:
        if not isinstance(item, dict):
            continue
        metric_name = str(item.get("metric_name") or "").strip()
        metric = str(item.get("metric") or "").strip() or None
        if not metric_name:
            continue
        key = (metric_name, metric)
        if key in seen_metric_keys:
            continue
        seen_metric_keys.add(key)
        merged_metrics.append(item)
    payload["analysis_requirements"]["needs_trend_context"] = True
    payload["analysis_requirements"]["derived_metrics"] = merged_metrics

    return QuestionAnalysis.model_validate(payload), True


def _is_underdefined_numeric_computation_query(
    qa: QuestionAnalysis,
    raw_query: str,
) -> bool:
    """Return True when an underdefined numeric calculation should clarify, not use knowledge."""

    if (
        qa.classification.query_type not in (QueryType.AMBIGUOUS, QueryType.UNSUPPORTED)
        or qa.routing.preferred_path != PreferredPath.KNOWLEDGE
    ):
        return False

    query_lower = str(raw_query or "").lower()
    if not query_lower:
        return False
    if not any(token in query_lower for token in _NUMERIC_CALCULATION_TOKENS):
        return False
    if not any(token in query_lower for token in _UNDERDEFINED_SCOPE_TOKENS):
        return False
    if not any(token in query_lower for token in _NUMERIC_PRICE_CONTEXT_TOKENS):
        return False

    tool_names = {
        getattr(tool.name, "value", str(tool.name or ""))
        for tool in (qa.tooling.candidate_tools or [])
    }
    if ToolName.GET_PRICES.value not in tool_names:
        return False
    if not (
        ToolName.GET_TARIFFS.value in tool_names
        or ToolName.GET_BALANCING_COMPOSITION.value in tool_names
    ):
        return False
    return True


def _flatten_recent_conversation_text(conversation_history) -> str:
    """Return the last few question/answer turns as one normalized string."""

    if not conversation_history:
        return ""

    parts: list[str] = []
    for turn in conversation_history[-3:]:
        if not isinstance(turn, dict):
            continue
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if question:
            parts.append(question)
        if answer:
            parts.append(answer)
    return " ".join(parts).lower()


def _has_explicit_residual_bucket_definition(text: str) -> bool:
    """Return True when text defines the priced residual bucket explicitly."""

    text_lower = str(text or "").lower()
    if not text_lower:
        return False
    scope_hit = any(token in text_lower for token in _RESIDUAL_SCOPE_TOKENS)
    hydro_hit = any(token in text_lower for token in _RESIDUAL_HYDRO_TOKENS)
    thermal_hit = any(token in text_lower for token in _RESIDUAL_THERMAL_TOKENS)
    deregulated_hit = any(token in text_lower for token in _RESIDUAL_DEREGULATED_TOKENS)
    return scope_hit and hydro_hit and thermal_hit and deregulated_hit


def _is_history_resolved_numeric_computation_query(
    qa: QuestionAnalysis,
    raw_query: str,
    conversation_history,
) -> bool:
    """Return True when recent history resolves an otherwise ambiguous residual bucket."""

    if (
        qa.classification.query_type not in (QueryType.AMBIGUOUS, QueryType.UNSUPPORTED)
        or qa.routing.preferred_path not in (PreferredPath.CLARIFY, PreferredPath.KNOWLEDGE)
    ):
        return False

    query_lower = str(raw_query or "").lower()
    if not query_lower:
        return False
    if not any(token in query_lower for token in _NUMERIC_CALCULATION_TOKENS):
        return False
    if not any(token in query_lower for token in _RESIDUAL_SCOPE_TOKENS):
        return False
    if "balancing" not in query_lower:
        return False
    if not any(token in query_lower for token in _NUMERIC_PRICE_CONTEXT_TOKENS):
        return False

    history_text = _flatten_recent_conversation_text(conversation_history)
    combined_text = " ".join(part for part in (history_text, query_lower) if part).strip()
    if not _has_explicit_residual_bucket_definition(combined_text):
        return False

    tool_names = {
        getattr(tool.name, "value", str(tool.name or ""))
        for tool in (qa.tooling.candidate_tools or [])
    }
    if tool_names and ToolName.GET_PRICES.value not in tool_names:
        return False
    return True


def _apply_history_resolved_numeric_computation_guardrail(
    qa: QuestionAnalysis,
    raw_query: str,
    conversation_history,
) -> tuple[QuestionAnalysis, bool]:
    """Promote history-resolved residual calculations back to tool-based retrieval."""

    if not _is_history_resolved_numeric_computation_query(qa, raw_query, conversation_history):
        return qa, False

    start_date, end_date = extract_date_range(raw_query)
    payload = qa.model_dump(mode="json")
    payload["classification"]["query_type"] = QueryType.DATA_RETRIEVAL.value
    payload["classification"]["needs_clarification"] = False
    payload["classification"]["ambiguities"] = []
    payload["classification"]["confidence"] = max(
        float(payload["classification"].get("confidence", 0.0)),
        0.85,
    )
    payload["routing"].update({
        "preferred_path": PreferredPath.TOOL.value,
        "needs_sql": False,
        "needs_knowledge": False,
        "prefer_tool": True,
        "needs_multi_tool": True,
        "evidence_roles": [
            "primary_data",
            "composition_context",
            "tariff_context",
        ],
    })
    payload.setdefault("sql_hints", {})
    if not payload["sql_hints"].get("metric"):
        payload["sql_hints"]["metric"] = "balancing"

    payload["tooling"]["candidate_tools"] = [
        {
            "name": ToolName.GET_PRICES.value,
            "score": 1.0,
            "reason": "balancing price and deregulated source price needed for residual weighted-price calculation",
            "params_hint": {
                "metric": "balancing",
                "currency": "both",
                "granularity": "monthly",
                "start_date": start_date,
                "end_date": end_date,
                "entities": [],
                "types": [],
            },
        },
        {
            "name": ToolName.GET_BALANCING_COMPOSITION.value,
            "score": 0.97,
            "reason": "composition shares needed to recover the remaining share",
            "params_hint": {
                "start_date": start_date,
                "end_date": end_date,
                "entities": [],
                "types": [],
            },
        },
        {
            "name": ToolName.GET_TARIFFS.value,
            "score": 0.96,
            "reason": "regulated tariffs needed for the excluded source-price layers",
            "params_hint": {
                "currency": "both",
                "granularity": "monthly",
                "start_date": start_date,
                "end_date": end_date,
                "entities": [],
                "types": [],
            },
        },
    ]

    return QuestionAnalysis.model_validate(payload), True


def _apply_underdefined_numeric_clarify_guardrail(
    qa: QuestionAnalysis,
    raw_query: str,
) -> tuple[QuestionAnalysis, bool]:
    """Coerce unresolved numeric computation requests to clarification."""

    if not _is_underdefined_numeric_computation_query(qa, raw_query):
        return qa, False

    payload = qa.model_dump(mode="json")
    payload["classification"]["needs_clarification"] = True
    payload["classification"]["ambiguities"] = [
        "computed target definition is underdefined",
    ]
    payload["classification"]["confidence"] = max(
        float(payload["classification"].get("confidence", 0.0)),
        0.8,
    )
    payload["routing"].update({
        "preferred_path": PreferredPath.CLARIFY.value,
        "needs_sql": False,
        "needs_knowledge": False,
        "prefer_tool": False,
        "needs_multi_tool": False,
        "evidence_roles": [],
    })

    return QuestionAnalysis.model_validate(payload), True


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
        analyzed = llm_analyze_question(
            user_query=ctx.query,
            conversation_history=ctx.conversation_history,
        )
        ctx.question_analysis, guardrail_applied = _apply_balancing_month_explanation_guardrail(
            analyzed,
            ctx.query,
        )
        forecast_guardrail_applied = False
        clarify_guardrail_applied = False
        history_resolution_applied = False
        if ctx.question_analysis is not None:
            ctx.question_analysis, forecast_guardrail_applied = _apply_balancing_price_forecast_guardrail(
                ctx.question_analysis,
                ctx.query,
            )
            ctx.question_analysis, clarify_guardrail_applied = _apply_underdefined_numeric_clarify_guardrail(
                ctx.question_analysis,
                ctx.query,
            )
            ctx.question_analysis, history_resolution_applied = _apply_history_resolved_numeric_computation_guardrail(
                ctx.question_analysis,
                ctx.query,
                ctx.conversation_history,
            )
        ctx.question_analysis_error = ""
        ctx.question_analysis_source = source
        if guardrail_applied:
            log.info(
                "Applied balancing month explanation guardrail: coerced analyzer output to data_explanation/tool for query=%r",
                ctx.query,
            )
        if forecast_guardrail_applied:
            log.info(
                "Applied balancing price forecast guardrail: coerced analyzer output to forecast/tool for query=%r",
                ctx.query,
            )
        if clarify_guardrail_applied:
            ctx.clarify_reason = "underdefined_computed_target"
            log.info(
                "Applied underdefined numeric computation guardrail: coerced analyzer output to clarify for query=%r",
                ctx.query,
            )
        if history_resolution_applied:
            ctx.clarify_reason = ""
            log.info(
                "Applied history-resolved numeric computation guardrail: coerced analyzer output to data_retrieval/tool for query=%r",
                ctx.query,
            )
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

    start_date, end_date = _expand_single_month_explanation_window(
        qa, start_date, end_date,
    )

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
