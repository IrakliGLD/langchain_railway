"""
Metrics tracking for observability.

Simple in-memory metrics for monitoring application performance.
"""
import logging
import hashlib
import re
from contextvars import ContextVar
from typing import Any, Dict

log = logging.getLogger("Enai")

_request_usage_var: ContextVar[Dict[str, Any] | None] = ContextVar("request_usage_var", default=None)


class Metrics:
    """Simple metrics tracker for observability."""

    def __init__(self):
        self.request_count = 0
        self.llm_call_count = 0
        self.sql_query_count = 0
        self.tool_call_count = 0
        self.tool_error_count = 0
        self.agent_round_count = 0
        self.agent_data_exit_count = 0
        self.agent_conceptual_exit_count = 0
        self.agent_fallback_exit_count = 0
        self.total_agent_preview_chars = 0
        self.stage_count_by_name = {}
        self.stage_total_time_ms = {}
        self.security_event_count = 0
        self.security_events_by_type = {}
        self.firewall_allow_count = 0
        self.firewall_warn_count = 0
        self.firewall_block_count = 0
        self.summary_schema_failure_count = 0
        self.summary_grounding_failure_count = 0
        self.provenance_gate_failure_count = 0
        self.deterministic_summary_skip_count = 0
        self.deterministic_summary_skips_by_source = {}
        self.relevance_block_count = 0
        self.load_shed_count = 0
        self.circuit_open_events = {}
        self.router_deterministic_match_count = 0
        self.router_semantic_match_count = 0
        self.router_miss_count = 0
        self.tool_fallback_intents = {}
        self.llm_prompt_tokens = 0
        self.llm_completion_tokens = 0
        self.llm_total_tokens = 0
        self.llm_estimated_cost_usd = 0.0
        self.llm_usage_by_model = {}
        self.session_history_with_count = 0
        self.session_history_without_count = 0
        self.session_history_total_turns = 0
        self.error_count = 0
        self.total_llm_time = 0.0
        self.total_sql_time = 0.0
        self.total_tool_time = 0.0
        self.total_request_time = 0.0

    def log_request(self, duration: float):
        """Log a request with its duration."""
        self.request_count += 1
        self.total_request_time += duration
        avg_time = self.total_request_time / self.request_count
        log.info(f"📊 Metrics: requests={self.request_count}, avg_time={avg_time:.2f}s")

    def log_llm_call(self, duration: float):
        """Log an LLM API call with its duration."""
        self.llm_call_count += 1
        self.total_llm_time += duration

    def log_sql_query(self, duration: float):
        """Log a SQL query with its duration."""
        self.sql_query_count += 1
        self.total_sql_time += duration

    def log_tool_call(self, duration: float):
        """Log a typed tool execution with its duration."""
        self.tool_call_count += 1
        self.total_tool_time += duration

    def log_tool_error(self):
        """Log an error in typed tool execution."""
        self.tool_error_count += 1

    def log_agent_round(self):
        """Log a single agent loop round."""
        self.agent_round_count += 1

    def log_agent_exit(self, outcome: str):
        """Log terminal outcome for the agent loop."""
        normalized = (outcome or "").strip().lower()
        if normalized == "data_exit":
            self.agent_data_exit_count += 1
        elif normalized == "conceptual_exit":
            self.agent_conceptual_exit_count += 1
        elif normalized == "fallback_exit":
            self.agent_fallback_exit_count += 1

    def log_agent_preview(self, char_count: int):
        """Track prompt payload size added by tool previews."""
        self.total_agent_preview_chars += max(0, int(char_count))

    def log_stage(self, stage_name: str, duration_ms: float):
        """Track per-stage timings for distributed tracing views."""
        key = (stage_name or "unknown").strip().lower() or "unknown"
        duration_ms = max(0.0, float(duration_ms))
        self.stage_count_by_name[key] = self.stage_count_by_name.get(key, 0) + 1
        self.stage_total_time_ms[key] = self.stage_total_time_ms.get(key, 0.0) + duration_ms

    def log_security_event(self, event_type: str):
        """Track structured security/audit events."""
        key = (event_type or "unknown").strip().lower() or "unknown"
        self.security_event_count += 1
        self.security_events_by_type[key] = self.security_events_by_type.get(key, 0) + 1

    def start_request_telemetry(self, trace_id: str = ""):
        """Initialize per-request token/cost telemetry context."""
        _request_usage_var.set(
            {
                "trace_id": str(trace_id or ""),
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "models": {},
            }
        )

    def log_llm_usage(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost_usd: float,
    ):
        """Track aggregate and per-request LLM token/cost usage."""
        model_key = (model_name or "unknown").strip()
        prompt_tokens = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
        total_tokens = max(0, int(total_tokens or 0))
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        cost = max(0.0, float(estimated_cost_usd or 0.0))

        self.llm_prompt_tokens += prompt_tokens
        self.llm_completion_tokens += completion_tokens
        self.llm_total_tokens += total_tokens
        self.llm_estimated_cost_usd += cost

        bucket = self.llm_usage_by_model.setdefault(
            model_key,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["total_tokens"] += total_tokens
        bucket["estimated_cost_usd"] += cost

        current = _request_usage_var.get()
        if current is not None:
            current["llm_calls"] += 1
            current["prompt_tokens"] += prompt_tokens
            current["completion_tokens"] += completion_tokens
            current["total_tokens"] += total_tokens
            current["estimated_cost_usd"] += cost
            req_models = current.setdefault("models", {})
            req_bucket = req_models.setdefault(
                model_key,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                },
            )
            req_bucket["calls"] += 1
            req_bucket["prompt_tokens"] += prompt_tokens
            req_bucket["completion_tokens"] += completion_tokens
            req_bucket["total_tokens"] += total_tokens
            req_bucket["estimated_cost_usd"] += cost
            _request_usage_var.set(current)

    def get_current_request_telemetry(self) -> Dict[str, Any]:
        """Return current request token/cost telemetry snapshot."""
        current = _request_usage_var.get()
        if current is None:
            return {
                "trace_id": "",
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "models": {},
            }
        return {
            "trace_id": current.get("trace_id", ""),
            "llm_calls": int(current.get("llm_calls", 0)),
            "prompt_tokens": int(current.get("prompt_tokens", 0)),
            "completion_tokens": int(current.get("completion_tokens", 0)),
            "total_tokens": int(current.get("total_tokens", 0)),
            "estimated_cost_usd": float(current.get("estimated_cost_usd", 0.0)),
            "models": dict(current.get("models", {})),
        }

    def finalize_request_telemetry(self) -> Dict[str, Any]:
        """Return and clear per-request telemetry context."""
        snapshot = self.get_current_request_telemetry()
        _request_usage_var.set(None)
        return snapshot

    def log_firewall_decision(self, action: str):
        """Track Stage-0 firewall outcomes."""
        normalized = (action or "").strip().lower()
        if normalized == "block":
            self.firewall_block_count += 1
        elif normalized == "warn":
            self.firewall_warn_count += 1
        else:
            self.firewall_allow_count += 1

    def log_summary_schema_failure(self):
        """Track summarizer schema validation failures."""
        self.summary_schema_failure_count += 1

    def log_summary_grounding_failure(self):
        """Track ungrounded summary outputs."""
        self.summary_grounding_failure_count += 1

    def log_provenance_gate_failure(self):
        """Track citation-grade provenance gate failures."""
        self.provenance_gate_failure_count += 1

    def log_deterministic_skip(self, source: str):
        """Track deterministic answer paths that skip Stage 4 LLM."""
        key = (source or "unknown").strip().lower() or "unknown"
        self.deterministic_summary_skip_count += 1
        self.deterministic_summary_skips_by_source[key] = (
            self.deterministic_summary_skips_by_source.get(key, 0) + 1
        )

    def log_router_match(self, match_type: str):
        """Track router coverage by match type."""
        normalized = (match_type or "").strip().lower()
        if normalized == "deterministic":
            self.router_deterministic_match_count += 1
        elif normalized == "semantic":
            self.router_semantic_match_count += 1
        else:
            self.router_miss_count += 1

    def log_tool_fallback_intent(self, query: str, reason: str):
        """Track long-tail fallback intents for typed-tool coverage expansion."""
        text = (query or "").strip().lower()
        if not text:
            text = "empty_query"
        sanitized = re.sub(r"\b20\d{2}\b", "<year>", text)
        sanitized = re.sub(r"\d+(?:\.\d+)?", "<num>", sanitized)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()[:200]
        signature_hash = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()[:12]
        signature = f"{signature_hash}|{sanitized}"
        key = f"{(reason or 'unknown').strip().lower()}|{signature}"
        self.tool_fallback_intents[key] = self.tool_fallback_intents.get(key, 0) + 1

    def log_relevance_block(self):
        """Track hard relevance guardrail blocks."""
        self.relevance_block_count += 1

    def log_load_shed(self):
        """Track rejected requests due to backpressure gate."""
        self.load_shed_count += 1

    def log_circuit_open(self, component: str):
        """Track fail-fast events due to open circuit breaker."""
        key = (component or "unknown").strip().lower() or "unknown"
        self.circuit_open_events[key] = self.circuit_open_events.get(key, 0) + 1

    def log_session_history_context(self, turns: int):
        """Track whether requests arrive with conversation history."""
        turns = max(0, int(turns or 0))
        if turns > 0:
            self.session_history_with_count += 1
            self.session_history_total_turns += turns
        else:
            self.session_history_without_count += 1

    def log_error(self):
        """Log an error occurrence."""
        self.error_count += 1

    def get_stats(self) -> dict:
        """
        Get current metrics statistics.

        Returns:
            Dictionary with all metrics
        """
        return {
            "requests": self.request_count,
            "llm_calls": self.llm_call_count,
            "sql_queries": self.sql_query_count,
            "tool_calls": self.tool_call_count,
            "tool_errors": self.tool_error_count,
            "agent_rounds": self.agent_round_count,
            "agent_data_exits": self.agent_data_exit_count,
            "agent_conceptual_exits": self.agent_conceptual_exit_count,
            "agent_fallback_exits": self.agent_fallback_exit_count,
            "avg_agent_preview_chars": self.total_agent_preview_chars / max(1, self.agent_round_count),
            "stage_counts": dict(self.stage_count_by_name),
            "stage_avg_ms": {
                key: self.stage_total_time_ms.get(key, 0.0) / max(1, self.stage_count_by_name.get(key, 0))
                for key in self.stage_count_by_name
            },
            "security_events": self.security_event_count,
            "security_events_by_type": dict(self.security_events_by_type),
            "firewall_allow": self.firewall_allow_count,
            "firewall_warn": self.firewall_warn_count,
            "firewall_block": self.firewall_block_count,
            "summary_schema_failures": self.summary_schema_failure_count,
            "summary_grounding_failures": self.summary_grounding_failure_count,
            "provenance_gate_failures": self.provenance_gate_failure_count,
            "deterministic_summary_skips": self.deterministic_summary_skip_count,
            "deterministic_summary_skips_by_source": dict(self.deterministic_summary_skips_by_source),
            "router_deterministic_matches": self.router_deterministic_match_count,
            "router_semantic_matches": self.router_semantic_match_count,
            "router_misses": self.router_miss_count,
            "tool_fallback_intents_top": dict(
                sorted(
                    self.tool_fallback_intents.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:20]
            ),
            "session_history_with": self.session_history_with_count,
            "session_history_without": self.session_history_without_count,
            "session_history_avg_turns": (
                self.session_history_total_turns / max(1, self.session_history_with_count)
            ),
            "relevance_blocks": self.relevance_block_count,
            "load_shed_events": self.load_shed_count,
            "circuit_open_events": dict(self.circuit_open_events),
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "llm_total_tokens": self.llm_total_tokens,
            "llm_estimated_cost_usd": self.llm_estimated_cost_usd,
            "llm_usage_by_model": dict(self.llm_usage_by_model),
            "errors": self.error_count,
            "avg_request_time": self.total_request_time / max(1, self.request_count),
            "avg_llm_time": self.total_llm_time / max(1, self.llm_call_count),
            "avg_sql_time": self.total_sql_time / max(1, self.sql_query_count),
            "avg_tool_time": self.total_tool_time / max(1, self.tool_call_count),
        }


# Global metrics instance
metrics = Metrics()
