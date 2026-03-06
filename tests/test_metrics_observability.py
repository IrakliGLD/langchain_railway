"""
Tests for observability telemetry additions (stage tracing + request token usage).
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("APP_SECRET_KEY", "test-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from utils.metrics import Metrics


def test_stage_metrics_are_aggregated():
    m = Metrics()
    m.log_stage("stage_1_generate_plan", 10.0)
    m.log_stage("stage_1_generate_plan", 30.0)
    m.log_stage("stage_2_sql_execute", 50.0)

    stats = m.get_stats()
    assert stats["stage_counts"]["stage_1_generate_plan"] == 2
    assert stats["stage_counts"]["stage_2_sql_execute"] == 1
    assert abs(stats["stage_avg_ms"]["stage_1_generate_plan"] - 20.0) < 1e-6


def test_request_telemetry_tracks_llm_tokens_and_cost():
    m = Metrics()
    m.start_request_telemetry("trace-123")
    m.log_llm_usage(
        model_name="gpt-4o-mini",
        prompt_tokens=120,
        completion_tokens=30,
        total_tokens=150,
        estimated_cost_usd=0.0012,
    )
    m.log_llm_usage(
        model_name="gpt-4o-mini",
        prompt_tokens=80,
        completion_tokens=20,
        total_tokens=100,
        estimated_cost_usd=0.0008,
    )

    snapshot = m.finalize_request_telemetry()
    assert snapshot["trace_id"] == "trace-123"
    assert snapshot["llm_calls"] == 2
    assert snapshot["prompt_tokens"] == 200
    assert snapshot["completion_tokens"] == 50
    assert snapshot["total_tokens"] == 250
    assert abs(snapshot["estimated_cost_usd"] - 0.002) < 1e-9
    assert snapshot["models"]["gpt-4o-mini"]["calls"] == 2


def test_cost_estimation_uses_actual_model_provider(monkeypatch):
    from core import llm

    monkeypatch.setattr(llm, "OPENAI_INPUT_COST_PER_1K_USD", 1.0)
    monkeypatch.setattr(llm, "OPENAI_OUTPUT_COST_PER_1K_USD", 2.0)
    monkeypatch.setattr(llm, "GEMINI_INPUT_COST_PER_1K_USD", 3.0)
    monkeypatch.setattr(llm, "GEMINI_OUTPUT_COST_PER_1K_USD", 4.0)
    monkeypatch.setattr(llm, "MODEL_TYPE", "gemini")

    # Even when MODEL_TYPE is gemini, OpenAI model usage must use OpenAI rates.
    openai_cost = llm._estimate_cost_usd(1000, 1000, model_name="gpt-4o-mini")
    gemini_cost = llm._estimate_cost_usd(1000, 1000, model_name="gemini-2.5-flash")

    assert abs(openai_cost - 3.0) < 1e-9
    assert abs(gemini_cost - 7.0) < 1e-9


def test_router_and_fallback_intent_metrics_are_tracked():
    m = Metrics()
    m.log_router_match("deterministic")
    m.log_router_match("semantic")
    m.log_router_match("miss")
    m.log_tool_fallback_intent("How did balancing electricity cost evolve in 2024?", "router_no_match")
    m.log_tool_fallback_intent("How did balancing electricity cost evolve in 2025?", "router_no_match")
    m.log_provenance_gate_failure()

    stats = m.get_stats()
    assert stats["router_deterministic_matches"] == 1
    assert stats["router_semantic_matches"] == 1
    assert stats["router_misses"] == 1
    assert stats["provenance_gate_failures"] == 1
    assert len(stats["tool_fallback_intents_top"]) >= 1
