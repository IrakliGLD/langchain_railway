"""
Phase 4 agent loop orchestration.

Runs a bounded tool-calling loop with typed tools only.
SQL generation is intentionally NOT exposed as an LLM-callable tool.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from config import (
    AGENT_MAX_ROUNDS,
    AGENT_TOOL_PREVIEW_MAX_CHARS,
    AGENT_TOOL_PREVIEW_ROWS,
    AGENT_TOOL_TIMEOUT_SECONDS,
    GEMINI_MODEL,
    MODEL_TYPE,
    OPENAI_MODEL,
)
from core.llm import _log_usage_for_message, make_gemini, make_openai
from models import QueryContext
from utils.metrics import metrics
from agent.provenance import stamp_provenance, tool_invocation_hash
from agent.tool_adapter import (
    ToolExecutionResult,
    execute_tool_for_agent,
    format_tool_preview_message,
)

log = logging.getLogger("Enai")


ALLOWED_AGENT_TOOL_NAMES = {
    "get_prices",
    "get_balancing_composition",
    "get_tariffs",
    "get_generation_mix",
}


AGENT_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "get_prices",
        "description": "Fetch balancing, deregulated, guaranteed-capacity prices or exchange rate time series.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "currency": {"type": "string", "enum": ["gel", "usd", "both"]},
                "metric": {"type": "string", "enum": ["balancing", "deregulated", "guaranteed_capacity", "exchange_rate"]},
                "granularity": {"type": "string", "enum": ["monthly", "yearly"]},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "get_balancing_composition",
        "description": "Fetch balancing electricity composition shares by entity.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "get_tariffs",
        "description": "Fetch key tariff series for Enguri, Gardabani TPP, and grouped old TPP entities.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}},
                "currency": {"type": "string", "enum": ["gel", "usd"]},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "get_generation_mix",
        "description": "Fetch generation or demand quantities/shares by technology.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "types": {"type": "array", "items": {"type": "string"}},
                "mode": {"type": "string", "enum": ["quantity", "share"]},
                "granularity": {"type": "string", "enum": ["monthly", "yearly"]},
                "limit": {"type": "integer"},
            },
        },
    },
]


def _build_agent_model():
    llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
    try:
        return llm.bind_tools(AGENT_TOOL_SCHEMAS)
    except Exception as exc:  # pragma: no cover - provider-specific behavior
        log.warning("Agent tool binding failed, using raw model: %s", exc)
        return llm


def _agent_system_prompt(lang_instruction: str) -> str:
    return (
        "You are an energy data assistant. Use ONLY the provided typed tools for data retrieval. "
        "Never write SQL and never request a SQL-generation tool. "
        "When enough data is available, provide a concise final answer. "
        "If no tool is needed, answer directly.\n"
        f"{lang_instruction}"
    )


def _extract_content(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return " ".join(parts).strip()
    return str(content or "").strip()


def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    direct = getattr(response, "tool_calls", None)
    if isinstance(direct, list) and direct:
        return direct

    additional = getattr(response, "additional_kwargs", {}) or {}
    raw_calls = additional.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []

    parsed: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_calls, start=1):
        fn = raw.get("function", {}) if isinstance(raw, dict) else {}
        name = fn.get("name")
        args_raw = fn.get("arguments", "{}")
        args: Dict[str, Any]
        if isinstance(args_raw, str):
            try:
                loaded = json.loads(args_raw)
                args = loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                args = {}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        parsed.append({
            "id": raw.get("id", f"call_{idx}") if isinstance(raw, dict) else f"call_{idx}",
            "name": name,
            "args": args,
        })
    return parsed


def _normalize_tool_call(raw_call: Dict[str, Any], round_idx: int, call_idx: int) -> Tuple[str, str, Dict[str, Any]]:
    call_id = str(raw_call.get("id") or f"r{round_idx}_c{call_idx}")
    name = str(raw_call.get("name") or raw_call.get("tool") or "").strip()
    args = raw_call.get("args", raw_call.get("arguments", {}))
    if isinstance(args, str):
        try:
            loaded = json.loads(args)
            args = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return call_id, name, args


def _query_prefers_data_exit(query: str) -> bool:
    q = query.lower()
    data_markers = [
        "trend",
        "over time",
        "from ",
        "between ",
        "compare",
        "chart",
        "plot",
        "table",
        "monthly",
        "yearly",
        "history",
        "show",
        "list",
    ]
    return any(marker in q for marker in data_markers)


def _score_dataset_for_query(query: str, result: ToolExecutionResult) -> int:
    q = query.lower()
    score = 0
    keyword_map = {
        "get_prices": ["price", "xrate", "exchange", "balancing"],
        "get_tariffs": ["tariff", "enguri", "gardabani"],
        "get_balancing_composition": ["share", "composition", "mix", "balancing"],
        "get_generation_mix": ["generation", "demand", "technology", "type_tech"],
    }
    for token in keyword_map.get(result.name, []):
        if token in q:
            score += 2
    if result.row_count > 0:
        score += 1
    return score


def _select_primary_dataset(
    query: str,
    datasets: Dict[str, ToolExecutionResult],
) -> Optional[Tuple[str, ToolExecutionResult]]:
    if not datasets:
        return None
    if len(datasets) == 1:
        return next(iter(datasets.items()))

    scored: List[Tuple[int, str, ToolExecutionResult]] = []
    for dataset_id, result in datasets.items():
        scored.append((_score_dataset_for_query(query, result), dataset_id, result))
    scored.sort(key=lambda x: x[0], reverse=True)
    if len(scored) >= 2 and scored[0][0] == scored[1][0]:
        return None
    _, dataset_id, result = scored[0]
    return dataset_id, result


def _attach_dataset(ctx: QueryContext, dataset_id: str, result: ToolExecutionResult) -> QueryContext:
    ctx.used_tool = True
    ctx.tool_name = result.name
    ctx.tool_params = dict(result.params)
    ctx.tool_match_reason = f"agent_selected:{dataset_id}"
    ctx.tool_confidence = 0.80
    ctx.df = result.df if result.df is not None else pd.DataFrame()
    ctx.cols = list(result.cols or [])
    ctx.rows = [tuple(r) for r in (result.rows or [])]
    stamp_provenance(
        ctx,
        ctx.cols,
        ctx.rows,
        source="tool",
        query_hash=tool_invocation_hash(result.name, result.params),
    )
    ctx.plan.setdefault("intent", "agent_tool_query")
    ctx.plan.setdefault("target", result.name)
    return ctx


def _resolve_agent_model_name(llm: Any, response: Any) -> str:
    """Best-effort model name for agent-loop token telemetry."""
    response_meta = getattr(response, "response_metadata", None)
    if isinstance(response_meta, dict):
        for key in ("model_name", "model"):
            value = response_meta.get(key)
            if value:
                return str(value)

    for attr in ("model_name", "model", "model_id"):
        value = getattr(llm, attr, None)
        if value:
            return str(value)

    return GEMINI_MODEL if MODEL_TYPE == "gemini" else OPENAI_MODEL


def _set_fallback(ctx: QueryContext, reason: str) -> QueryContext:
    ctx.agent_outcome = "fallback_exit"
    ctx.agent_fallback_reason = reason
    metrics.log_agent_exit("fallback_exit")
    return ctx


def available_agent_tools() -> List[str]:
    return sorted(ALLOWED_AGENT_TOOL_NAMES)


def run_agent_loop(ctx: QueryContext, llm: Any = None) -> QueryContext:
    """Run bounded typed-tool agent loop and set terminal agent outcome on ctx."""
    ctx.agent_outcome = ""
    ctx.agent_fallback_reason = ""
    ctx.agent_rounds = 0
    ctx.agent_tool_calls = 0

    if llm is None:
        try:
            llm = _build_agent_model()
        except Exception as exc:
            return _set_fallback(ctx, f"agent_model_init_failed: {exc}")

    # Build context-enriched user message so the agent sees resolved intent,
    # prior conversation, and what was already attempted.
    user_content = ctx.query
    _is_followup = ctx.resolved_query and ctx.resolved_query != ctx.query
    if _is_followup:
        user_content = (
            f"[Resolved query: {ctx.resolved_query}]\n\n"
            f"Original user message: {ctx.query}"
        )
    # Only inject conversation history when the query looks like a follow-up,
    # to avoid dragging prior-topic context into unrelated fallback queries.
    if _is_followup and ctx.conversation_history:
        recent = ctx.conversation_history[-3:]
        history_lines = "\n".join(
            f"Q: {turn.get('question', '')}\nA: {turn.get('answer', '')}"
            for turn in recent
        )
        user_content = f"Previous conversation:\n{history_lines}\n\n{user_content}"
    if ctx.tool_fallback_reason:
        user_content += f"\n\n[Note: A previous tool attempt failed: {ctx.tool_fallback_reason}]"

    messages: List[Any] = [
        SystemMessage(content=_agent_system_prompt(ctx.lang_instruction)),
        HumanMessage(content=user_content),
    ]
    datasets: Dict[str, ToolExecutionResult] = {}

    for round_idx in range(1, AGENT_MAX_ROUNDS + 1):
        ctx.agent_rounds = round_idx
        metrics.log_agent_round()

        t_llm = time.time()
        try:
            response = llm.invoke(messages)
            _log_usage_for_message(response, model_name=_resolve_agent_model_name(llm, response))
            metrics.log_llm_call(time.time() - t_llm)
        except Exception as exc:
            return _set_fallback(ctx, f"agent_model_error: {exc}")

        content = _extract_content(response)
        tool_calls = _extract_tool_calls(response)

        if tool_calls:
            if isinstance(response, AIMessage):
                messages.append(response)
            else:
                messages.append(AIMessage(content=content or ""))

            for call_idx, raw_call in enumerate(tool_calls, start=1):
                call_id, tool_name, args = _normalize_tool_call(raw_call, round_idx, call_idx)
                ctx.agent_tool_calls += 1

                if tool_name not in ALLOWED_AGENT_TOOL_NAMES:
                    result = ToolExecutionResult(
                        name=tool_name or "unknown",
                        params=args,
                        success=False,
                        error=f"Unknown or disallowed tool: {tool_name}",
                        cols=[],
                        rows=[],
                    )
                    metrics.log_tool_error()
                else:
                    t_tool = time.time()
                    result = execute_tool_for_agent(
                        tool_name=tool_name,
                        params=args,
                        preview_rows=AGENT_TOOL_PREVIEW_ROWS,
                        preview_max_chars=AGENT_TOOL_PREVIEW_MAX_CHARS,
                        timeout_seconds=AGENT_TOOL_TIMEOUT_SECONDS,
                    )
                    if result.success:
                        metrics.log_tool_call(time.time() - t_tool)
                    else:
                        metrics.log_tool_error()

                dataset_id = f"ds_{len(datasets) + 1}"
                if result.success:
                    datasets[dataset_id] = result

                preview_message = format_tool_preview_message(dataset_id, result)
                metrics.log_agent_preview(len(preview_message))
                messages.append(ToolMessage(content=preview_message, tool_call_id=call_id))
                ctx.agent_trace.append({
                    "round": round_idx,
                    "tool": tool_name,
                    "dataset_id": dataset_id,
                    "success": result.success,
                    "row_count": result.row_count,
                    "error": result.error,
                })

            continue

        # No tool call: model is likely returning final answer.
        if content:
            if datasets and _query_prefers_data_exit(ctx.query):
                selected = _select_primary_dataset(ctx.query, datasets)
                if selected:
                    dataset_id, result = selected
                    _attach_dataset(ctx, dataset_id, result)
                    ctx.agent_outcome = "data_exit"
                    metrics.log_agent_exit("data_exit")
                    return ctx
                return _set_fallback(ctx, "agent_dataset_ambiguous")

            ctx.summary = content
            ctx.is_conceptual = True
            ctx.agent_outcome = "conceptual_exit"
            metrics.log_agent_exit("conceptual_exit")
            return ctx

        # Empty response with no tool calls.
        selected = _select_primary_dataset(ctx.query, datasets)
        if selected:
            dataset_id, result = selected
            _attach_dataset(ctx, dataset_id, result)
            ctx.agent_outcome = "data_exit"
            metrics.log_agent_exit("data_exit")
            return ctx
        return _set_fallback(ctx, "agent_empty_response")

    # Max rounds reached.
    selected = _select_primary_dataset(ctx.query, datasets)
    if selected and len(datasets) == 1:
        dataset_id, result = selected
        _attach_dataset(ctx, dataset_id, result)
        ctx.agent_outcome = "data_exit"
        metrics.log_agent_exit("data_exit")
        return ctx
    return _set_fallback(ctx, "agent_max_rounds_exceeded")
