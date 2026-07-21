"""Finalize structured question interpretations into authoritative answer contracts."""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from typing import Optional

from agent.router import (
    extract_balancing_entities,
    extract_date_range,
    extract_generation_mode,
)
from contracts.question_analysis import (
    AnswerKind,
    PreferredPath,
    QueryType,
    QuestionAnalysis,
    RenderStyle,
    ToolName,
)
from models import QueryContext
from utils.language import detect_language


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    """Parse an ISO date string safely."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None

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

_TECHNICAL_QUANTITY_CONCEPT_TOKENS = (
    "generation mix",
    "generation",
    "demand",
    "consumption",
    "import dependency",
    "import dependence",
    "energy security",
    "self-sufficiency",
)

_TECHNICAL_EXPLORATION_TOKENS = (
    "what can you say",
    "tell me about",
    "describe",
    "overview",
    "characteristic",
    "characteristics",
    "explain",
)

_GEOGRAPHY_SCOPE_TOKENS = (
    "georgia",
    "georgian",
    "in georgia",
    "of georgia",
    "georgia's",
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

_TREND_QUERY_TOKENS = (
    "trend",
    "trends",
    "historical trend",
    "evolution",
    "evolve",
    "trajectory",
    "change over time",
)

_CORRELATION_QUERY_TOKENS = (
    "correlation",
    "relationship",
    "association",
    "correlat",
)

_PRICE_CONTEXT_TOKENS = (
    "price",
    "prices",
    "tariff",
    "tariffs",
    "gel/mwh",
    "usd/mwh",
)


def _query_mentions_any(query_lower: str, tokens: tuple[str, ...]) -> bool:
    return any(token in query_lower for token in tokens)


def _primary_query_surface(raw_query: str) -> str:
    query_text = str(raw_query or "").strip()
    if not query_text:
        return ""
    marker = "\nSelected interpretation:"
    if marker in query_text:
        return query_text.split(marker, 1)[0].strip()
    return query_text


def _resolved_quantity_metric_from_query(raw_query: str) -> str | None:
    query_lower = _primary_query_surface(raw_query).lower()
    if "energy security" in query_lower:
        return "energy_security"
    if "self-sufficiency" in query_lower:
        return "self-sufficiency"
    if "import dependency" in query_lower or "import dependence" in query_lower:
        return "import_dependency"
    if "local generation" in query_lower or "domestic generation" in query_lower:
        return "local_generation"
    if "generation mix" in query_lower:
        return "generation"
    if "consumption" in query_lower:
        return "consumption"
    if "demand" in query_lower:
        return "demand"
    if "generation" in query_lower:
        return "generation"
    return None


def _set_answer_contract(
    payload: dict[str, object],
    *,
    answer_kind: str,
    render_style: str,
    grouping: str = "none",
    entity_scope: str | None = None,
) -> None:
    payload["answer_kind"] = answer_kind
    payload["render_style"] = render_style
    payload["grouping"] = grouping
    payload["entity_scope"] = entity_scope


def _merge_derived_metric_dicts(
    existing_metrics: list[dict[str, object]] | None,
    forced_metrics: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged_metrics: list[dict[str, object]] = []
    seen_metric_keys: set[tuple[str, str | None, str | None]] = set()
    for item in list(existing_metrics or []) + list(forced_metrics):
        if not isinstance(item, dict):
            continue
        metric_name = str(item.get("metric_name") or "").strip()
        metric = str(item.get("metric") or "").strip() or None
        target_metric = str(item.get("target_metric") or "").strip() or None
        if not metric_name:
            continue
        key = (metric_name, metric, target_metric)
        if key in seen_metric_keys:
            continue
        seen_metric_keys.add(key)
        merged_metrics.append(item)
    return merged_metrics


def _is_technical_indicator_bundle_query(
    qa: QuestionAnalysis,
    raw_query: str,
) -> bool:
    query_lower = _primary_query_surface(raw_query).lower()
    if not query_lower:
        return False
    if qa.classification.query_type != QueryType.CONCEPTUAL_DEFINITION:
        return False
    if qa.routing.preferred_path not in (PreferredPath.KNOWLEDGE, PreferredPath.CLARIFY):
        return False
    if _query_mentions_any(query_lower, _REGULATORY_CONCEPT_TOKENS):
        return False
    if _query_mentions_any(query_lower, _TREND_QUERY_TOKENS):
        return False
    if _query_mentions_any(query_lower, _CORRELATION_QUERY_TOKENS):
        return False
    if not _query_mentions_any(query_lower, _TECHNICAL_QUANTITY_CONCEPT_TOKENS):
        return False

    concept_hits = sum(token in query_lower for token in _TECHNICAL_QUANTITY_CONCEPT_TOKENS)
    has_scope = _query_mentions_any(query_lower, _GEOGRAPHY_SCOPE_TOKENS)
    has_exploration = _query_mentions_any(query_lower, _TECHNICAL_EXPLORATION_TOKENS)
    return concept_hits >= 2 or (has_scope and has_exploration)


def _apply_technical_indicator_bundle_guardrail(
    qa: QuestionAnalysis,
    raw_query: str,
) -> tuple[QuestionAnalysis, bool]:
    """Promote broad technical concept queries to generation-mix evidence."""
    if not _is_technical_indicator_bundle_query(qa, raw_query):
        return qa, False

    primary_query = _primary_query_surface(raw_query)
    resolved_metric = _resolved_quantity_metric_from_query(primary_query) or "generation"
    payload = qa.model_dump(mode="json")
    payload["canonical_query_en"] = primary_query or payload.get("canonical_query_en") or raw_query
    payload["classification"]["query_type"] = QueryType.DATA_EXPLANATION.value
    payload["classification"]["analysis_mode"] = "analyst"
    payload["classification"]["needs_clarification"] = False
    payload["classification"]["ambiguities"] = []
    payload["classification"]["confidence"] = max(
        float(payload["classification"].get("confidence", 0.0)),
        0.92,
    )

    payload["routing"].update({
        "preferred_path": PreferredPath.TOOL.value,
        "needs_sql": False,
        "needs_knowledge": True,
        "prefer_tool": True,
        "needs_multi_tool": False,
        "evidence_roles": ["primary_data"],
    })
    _set_answer_contract(
        payload,
        answer_kind="explanation",
        render_style="narrative",
        grouping="none",
        entity_scope="system",
    )

    payload["tooling"]["candidate_tools"] = [
        {
            "name": ToolName.GET_GENERATION_MIX.value,
            "score": 1.0,
            "reason": "system quantity evidence needed for technical concept assessment",
            "params_hint": {
                "metric": resolved_metric,
                "granularity": "yearly",
                "start_date": None,
                "end_date": None,
                "entities": [],
                "types": [],
                # Composition wording (generation mix/structure) keeps share
                # mode; pinning "quantity" here would undo the mix routing
                # (2026-07-09 report). Single authority: agent/router.py.
                "mode": extract_generation_mode(primary_query.lower()),
            },
        }
    ]

    payload.setdefault("sql_hints", {})
    payload["sql_hints"]["metric"] = resolved_metric

    topic_rows = [
        topic
        for topic in payload.get("knowledge", {}).get("candidate_topics", [])
        if isinstance(topic, dict) and topic.get("name")
    ]
    topic_names = {topic.get("name") for topic in topic_rows}
    for name, score in (
        ("generation_mix", 1.0),
        ("seasonal_patterns", 0.55),
        ("general_definitions", 0.45),
    ):
        if name not in topic_names:
            topic_rows.append({"name": name, "score": score})
    topic_rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    payload["knowledge"]["candidate_topics"] = topic_rows[:5]

    return QuestionAnalysis.model_validate(payload), True


def _is_quantity_trend_query(raw_query: str) -> bool:
    query_lower = _primary_query_surface(raw_query).lower()
    if not query_lower:
        return False
    if not _query_mentions_any(query_lower, _TREND_QUERY_TOKENS):
        return False
    if not _query_mentions_any(query_lower, _TECHNICAL_QUANTITY_CONCEPT_TOKENS):
        return False
    if _query_mentions_any(query_lower, _PRICE_CONTEXT_TOKENS):
        return False
    if _query_mentions_any(query_lower, _CORRELATION_QUERY_TOKENS):
        return False
    if "forecast" in query_lower or "predict" in query_lower:
        return False
    return True


def _apply_quantity_trend_guardrail(
    qa: QuestionAnalysis,
    raw_query: str,
) -> tuple[QuestionAnalysis, bool]:
    """Coerce explicit quantity-trend questions to canonical system quantities."""
    if not _is_quantity_trend_query(raw_query):
        return qa, False

    primary_query = _primary_query_surface(raw_query)
    resolved_metric = _resolved_quantity_metric_from_query(primary_query)
    if resolved_metric is None:
        return qa, False

    existing_tool_names = {
        getattr(tool.name, "value", str(tool.name or ""))
        for tool in (qa.tooling.candidate_tools or [])
    }
    payload_metrics = qa.model_dump(mode="json")["analysis_requirements"].get("derived_metrics", []) or []
    has_trend_metric = any(
        isinstance(item, dict)
        and str(item.get("metric_name") or "").strip() == "trend_slope"
        and str(item.get("metric") or "").strip().lower() in {resolved_metric, "demand", "consumption"}
        for item in payload_metrics
    )
    already_supported = (
        qa.routing.preferred_path == PreferredPath.TOOL
        and ToolName.GET_GENERATION_MIX.value in existing_tool_names
        and has_trend_metric
    )
    if already_supported:
        return qa, False

    payload = qa.model_dump(mode="json")
    payload["canonical_query_en"] = primary_query or payload.get("canonical_query_en") or raw_query
    payload["classification"]["query_type"] = QueryType.DATA_EXPLANATION.value
    payload["classification"]["analysis_mode"] = "analyst"
    payload["classification"]["needs_clarification"] = False
    payload["classification"]["ambiguities"] = []
    payload["classification"]["confidence"] = max(
        float(payload["classification"].get("confidence", 0.0)),
        0.93,
    )

    payload["routing"].update({
        "preferred_path": PreferredPath.TOOL.value,
        "needs_sql": False,
        "needs_knowledge": False,
        "prefer_tool": True,
        "needs_multi_tool": False,
        "evidence_roles": ["primary_data"],
    })
    _set_answer_contract(
        payload,
        answer_kind="explanation",
        render_style="narrative",
        grouping="none",
        entity_scope="system",
    )

    payload["tooling"]["candidate_tools"] = [
        {
            "name": ToolName.GET_GENERATION_MIX.value,
            "score": 1.0,
            "reason": "system quantity time series needed for trend assessment",
            "params_hint": {
                "metric": resolved_metric,
                "granularity": "monthly",
                "start_date": None,
                "end_date": None,
                "entities": [],
                "types": [],
                # Composition wording ("generation mix evolution") keeps share
                # mode so the trend renders as a stacked composition; plain
                # quantity trends stay "quantity". Single authority: router.
                "mode": extract_generation_mode(primary_query.lower()),
            },
        }
    ]
    payload.setdefault("sql_hints", {})
    payload["sql_hints"]["metric"] = resolved_metric

    payload["analysis_requirements"]["needs_driver_analysis"] = False
    payload["analysis_requirements"]["needs_trend_context"] = True
    payload["analysis_requirements"]["needs_correlation_context"] = False
    payload["analysis_requirements"]["derived_metrics"] = _merge_derived_metric_dicts(
        payload["analysis_requirements"].get("derived_metrics", []) or [],
        [{"metric_name": "trend_slope", "metric": resolved_metric}],
    )

    topic_rows = [
        topic
        for topic in payload.get("knowledge", {}).get("candidate_topics", [])
        if isinstance(topic, dict) and topic.get("name")
    ]
    topic_names = {topic.get("name") for topic in topic_rows}
    for name, score in (
        ("generation_mix", 1.0),
        ("seasonal_patterns", 0.7),
    ):
        if name not in topic_names:
            topic_rows.append({"name": name, "score": score})
    topic_rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    payload["knowledge"]["candidate_topics"] = topic_rows[:5]

    return QuestionAnalysis.model_validate(payload), True


def _supported_pairwise_metrics(raw_query: str) -> list[str]:
    query_lower = _primary_query_surface(raw_query).lower()
    metrics: list[str] = []
    if any(term in query_lower for term in ("balancing price", "balancing electricity price", "balancing electricity", "p_bal")):
        metrics.append("balancing")
    if "consumption" in query_lower:
        metrics.append("consumption")
    elif "demand" in query_lower:
        metrics.append("demand")
    if "local generation" in query_lower or "domestic generation" in query_lower:
        metrics.append("local_generation")
    elif "generation" in query_lower:
        metrics.append("generation")
    if "energy security" in query_lower:
        metrics.append("energy_security")
    elif "self-sufficiency" in query_lower:
        metrics.append("self-sufficiency")
    elif "import dependency" in query_lower or "import dependence" in query_lower:
        metrics.append("import_dependency")
    deduped: list[str] = []
    seen: set[str] = set()
    for metric in metrics:
        if metric in seen:
            continue
        deduped.append(metric)
        seen.add(metric)
    return deduped


def _is_supported_pairwise_correlation_query(raw_query: str) -> bool:
    query_lower = _primary_query_surface(raw_query).lower()
    if not _query_mentions_any(query_lower, _CORRELATION_QUERY_TOKENS):
        return False
    return len(_supported_pairwise_metrics(raw_query)) >= 2


def _apply_pairwise_correlation_guardrail(
    qa: QuestionAnalysis,
    raw_query: str,
) -> tuple[QuestionAnalysis, bool]:
    """Coerce supported pairwise technical correlations to a deterministic multi-tool contract."""
    if not _is_supported_pairwise_correlation_query(raw_query):
        return qa, False

    primary_query = _primary_query_surface(raw_query)
    pair = _supported_pairwise_metrics(primary_query)
    if len(pair) < 2:
        return qa, False

    primary_metric = next((metric for metric in pair if metric != "balancing"), pair[0])
    target_metric = "balancing" if "balancing" in pair else pair[1]

    payload = qa.model_dump(mode="json")
    payload["canonical_query_en"] = primary_query or payload.get("canonical_query_en") or raw_query
    existing_metrics = payload["analysis_requirements"].get("derived_metrics", []) or []
    has_supported_corr = any(
        isinstance(item, dict)
        and str(item.get("metric_name") or "").strip() == "correlation_to_target"
        and str(item.get("metric") or "").strip().lower() == primary_metric
        and str(item.get("target_metric") or "").strip().lower() == target_metric
        for item in existing_metrics
    )
    existing_tool_names = {
        getattr(tool.name, "value", str(tool.name or ""))
        for tool in (qa.tooling.candidate_tools or [])
    }
    already_supported = (
        qa.routing.preferred_path == PreferredPath.TOOL
        and has_supported_corr
        and (
            {ToolName.GET_PRICES.value, ToolName.GET_GENERATION_MIX.value}.issubset(existing_tool_names)
            or (target_metric != "balancing" and ToolName.GET_GENERATION_MIX.value in existing_tool_names)
        )
    )
    if already_supported:
        return qa, False

    needs_prices = "balancing" in pair
    payload["classification"]["query_type"] = QueryType.DATA_EXPLANATION.value
    payload["classification"]["analysis_mode"] = "analyst"
    payload["classification"]["needs_clarification"] = False
    payload["classification"]["ambiguities"] = []
    payload["classification"]["confidence"] = max(
        float(payload["classification"].get("confidence", 0.0)),
        0.94,
    )

    payload["routing"].update({
        "preferred_path": PreferredPath.TOOL.value,
        "needs_sql": False,
        "needs_knowledge": False,
        "prefer_tool": True,
        "needs_multi_tool": needs_prices,
        "evidence_roles": ["primary_data", "correlation_driver"] if needs_prices else ["primary_data"],
    })
    _set_answer_contract(
        payload,
        answer_kind="explanation",
        render_style="narrative",
        grouping="none",
        entity_scope="system",
    )

    candidate_tools = []
    if needs_prices:
        candidate_tools.append(
            {
                "name": ToolName.GET_PRICES.value,
                "score": 1.0,
                "reason": "price series required as the reference side of the requested correlation",
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
        )
    candidate_tools.append(
        {
            "name": ToolName.GET_GENERATION_MIX.value,
            "score": 0.97 if needs_prices else 1.0,
            "reason": "system quantity series required as the comparison side of the requested correlation",
            "params_hint": {
                "metric": primary_metric,
                "granularity": "monthly",
                "start_date": None,
                "end_date": None,
                "entities": [],
                "types": [],
                "mode": "quantity",
            },
        }
    )
    payload["tooling"]["candidate_tools"] = candidate_tools

    payload.setdefault("sql_hints", {})
    payload["sql_hints"]["metric"] = primary_metric
    payload["analysis_requirements"]["needs_driver_analysis"] = False
    payload["analysis_requirements"]["needs_trend_context"] = False
    payload["analysis_requirements"]["needs_correlation_context"] = True
    payload["analysis_requirements"]["derived_metrics"] = _merge_derived_metric_dicts(
        existing_metrics,
        [{"metric_name": "correlation_to_target", "metric": primary_metric, "target_metric": target_metric}],
    )

    topic_rows = [
        topic
        for topic in payload.get("knowledge", {}).get("candidate_topics", [])
        if isinstance(topic, dict) and topic.get("name")
    ]
    topic_names = {topic.get("name") for topic in topic_rows}
    for name, score in (
        ("balancing_price", 1.0 if needs_prices else 0.0),
        ("generation_mix", 0.9),
    ):
        if score > 0 and name not in topic_names:
            topic_rows.append({"name": name, "score": score})
    topic_rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    payload["knowledge"]["candidate_topics"] = topic_rows[:5]

    return QuestionAnalysis.model_validate(payload), True

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
    "deregulated renewable",
    "deregulated renewables",
    "deregulated plant",
    "deregulated plants",
    "deregulated power plant",
    "deregulated power plants",
)

_RESIDUAL_HYDRO_ENTITIES = frozenset({"regulated_hpp"})
_RESIDUAL_THERMAL_ENTITIES = frozenset({"regulated_old_tpp", "regulated_new_tpp"})
_RESIDUAL_DEREGULATED_ENTITIES = frozenset({"deregulated_ren"})

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

    # Resolve a unique month mention so the guardrail can force a precise analysis window.
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

    # Rewrite the analyzer payload so later stages see a deterministic multi-tool explanation task.
    payload["classification"]["query_type"] = QueryType.DATA_EXPLANATION.value
    payload["classification"]["analysis_mode"] = "analyst"
    payload["classification"]["needs_clarification"] = False
    payload["classification"]["ambiguities"] = []
    payload["classification"]["confidence"] = max(
        float(payload["classification"].get("confidence", 0.0)),
        0.95,
    )

    # Force downstream stages to recognize this as an analytical EXPLANATION.
    payload["answer_kind"] = AnswerKind.EXPLANATION.value
    payload["render_style"] = RenderStyle.NARRATIVE.value

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


# Forecast guardrails catch plain balancing-price forecasts that should stay on tool data.
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


# Clarify/history guardrails protect ambiguous residual weighted-price calculations.
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
    extracted_entities = {
        str(entity).lower()
        for entity in extract_balancing_entities(text_lower)
    }
    scope_hit = any(token in text_lower for token in _RESIDUAL_SCOPE_TOKENS)
    hydro_hit = (
        any(token in text_lower for token in _RESIDUAL_HYDRO_TOKENS)
        or bool(extracted_entities & _RESIDUAL_HYDRO_ENTITIES)
    )
    thermal_hit = (
        any(token in text_lower for token in _RESIDUAL_THERMAL_TOKENS)
        or bool(extracted_entities & _RESIDUAL_THERMAL_ENTITIES)
    )
    deregulated_hit = (
        any(token in text_lower for token in _RESIDUAL_DEREGULATED_TOKENS)
        or bool(extracted_entities & _RESIDUAL_DEREGULATED_ENTITIES)
    )
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

def _build_implied_ppa_cfd_question_analysis(ctx: QueryContext) -> QuestionAnalysis:
    """Build the authoritative contract for the negligible-import approximation."""

    start_date, end_date = extract_date_range(ctx.query)
    lang_code = ctx.lang_code or detect_language(ctx.query)
    payload = {
        "version": "question_analysis_v1",
        "raw_query": ctx.query,
        "canonical_query_en": (
            "Find months where import is below the requested negligible-share threshold and calculate "
            "the approximate weighted average PPA/CfD price from the balancing-price residual."
        ),
        "language": {"input_language": lang_code, "answer_language": lang_code},
        "classification": {
            "query_type": "data_retrieval",
            "analysis_mode": "light",
            "intent": "implied_ppa_cfd_price_approximation",
            "needs_clarification": False,
            "confidence": 1.0,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "tool",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": True,
            "needs_multi_tool": False,
            "evidence_roles": ["primary_data"],
        },
        "knowledge": {"candidate_topics": []},
        "tooling": {
            "candidate_tools": [{
                "name": "get_prices",
                "score": 1.0,
                "reason": "balancing prices anchor the deterministic PPA/CfD residual calculation",
                "params_hint": {
                    "metric": "balancing",
                    "currency": "both",
                    "granularity": "monthly",
                    "start_date": start_date,
                    "end_date": end_date,
                    "entities": [],
                    "types": [],
                },
            }],
        },
        "sql_hints": {
            "metric": "balancing",
            "entities": [],
            "aggregation": "monthly",
            "dimensions": ["period", "price", "share"],
            "period": None,
            "filter": None,
        },
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
            "primary_presentation": "table",
            "visual_goal": "threshold_scan",
            "measure_transform": "weighted_avg",
            "time_grain": "month",
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_trend_context": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
        "answer_kind": "timeseries",
        "render_style": "deterministic",
        "grouping": "by_period",
        "entity_scope": "ppa_cfd_excluding_import_approximation",
    }
    return QuestionAnalysis.model_validate(payload)

@dataclass(frozen=True)
class QuestionContractFinalization:
    """A finalized contract plus explicit, ordered guardrail decisions."""

    contract: QuestionAnalysis
    applied_guardrails: tuple[str, ...]
    clarify_reason_update: str | None = None

    def applied(self, guardrail: str) -> bool:
        return guardrail in self.applied_guardrails


def finalize_question_contract(
    analyzed: QuestionAnalysis,
    *,
    raw_query: str,
    conversation_history,
) -> QuestionContractFinalization:
    """Apply deterministic contract guardrails once in their established order."""
    contract, balancing_month = _apply_balancing_month_explanation_guardrail(
        analyzed,
        raw_query,
    )
    contract, balancing_forecast = _apply_balancing_price_forecast_guardrail(
        contract,
        raw_query,
    )
    contract, underdefined_clarify = _apply_underdefined_numeric_clarify_guardrail(
        contract,
        raw_query,
    )
    contract, history_resolved = _apply_history_resolved_numeric_computation_guardrail(
        contract,
        raw_query,
        conversation_history,
    )
    contract, technical_bundle = _apply_technical_indicator_bundle_guardrail(
        contract,
        raw_query,
    )
    contract, quantity_trend = _apply_quantity_trend_guardrail(
        contract,
        raw_query,
    )
    contract, pairwise_correlation = _apply_pairwise_correlation_guardrail(
        contract,
        raw_query,
    )

    applied = tuple(
        name
        for name, was_applied in (
            ("balancing_month_explanation", balancing_month),
            ("balancing_price_forecast", balancing_forecast),
            ("underdefined_numeric_clarify", underdefined_clarify),
            ("history_resolved_numeric_computation", history_resolved),
            ("technical_indicator_bundle", technical_bundle),
            ("quantity_trend", quantity_trend),
            ("pairwise_correlation", pairwise_correlation),
        )
        if was_applied
    )
    clarify_reason_update = None
    if underdefined_clarify:
        clarify_reason_update = "underdefined_computed_target"
    if history_resolved:
        clarify_reason_update = ""

    return QuestionContractFinalization(
        contract=contract,
        applied_guardrails=applied,
        clarify_reason_update=clarify_reason_update,
    )
