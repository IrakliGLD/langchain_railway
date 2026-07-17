"""Regression: typed provider failures must degrade, not 500 (incident 2026-07-17).

F8 wrapped every provider failure in ProviderExecutionError, which subclasses
RuntimeError. The summarizer's two `except RuntimeError: raise` clauses —
written pre-F8 to re-raise only the breaker-open bare RuntimeError — then
re-raised every provider failure, skipping the P0.7-hardened legacy fallback
and turning an ambiguous NVIDIA delivery into a pipeline 500 on /ask.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from agent import summarizer
from models import QueryContext
from utils.provider_attempts import (
    ProviderDeliveryDisposition,
    ProviderExecutionError,
)


def _ambiguous_failure(*_args, **_kwargs):
    raise ProviderExecutionError(
        "provider call failed: APIConnectionError",
        provider="nvidia",
        stage="structured_summarize",
        disposition=ProviderDeliveryDisposition.AMBIGUOUS,
    )


def _rejected_failure(*_args, **_kwargs):
    raise ProviderExecutionError(
        "LLM circuit breaker open for provider=nvidia reason=open",
        provider="nvidia",
        stage="structured_summarize",
        disposition=ProviderDeliveryDisposition.REJECTED,
    )


def _data_ctx() -> QueryContext:
    return QueryContext(
        query="Show the observed value",
        preview="date value\n2024-01-01 10",
        stats_hint="Rows: 1",
        cols=["date", "value"],
        rows=[("2024-01-01", 10)],
        provenance_cols=["date", "value"],
        provenance_rows=[("2024-01-01", 10)],
        provenance_query_hash="fallback123",
        provenance_source="sql",
    )


class TestStructuredDataPathDegrades:
    def test_ambiguous_provider_failure_uses_legacy_fallback(self, monkeypatch):
        monkeypatch.setattr(summarizer, "llm_summarize_structured", _ambiguous_failure)
        monkeypatch.setattr(
            summarizer, "llm_summarize", lambda *a, **k: "The observed value was 10.",
        )

        out = summarizer.summarize_data(_data_ctx())

        assert out.summary_source == "legacy_text_fallback"
        assert "10" in out.summary

    def test_rejected_provider_failure_also_degrades(self, monkeypatch):
        # Breaker-open reaching the summarizer means core-level OpenAI fallback
        # was unavailable; the legacy path still owns the degradation attempt.
        monkeypatch.setattr(summarizer, "llm_summarize_structured", _rejected_failure)
        monkeypatch.setattr(
            summarizer, "llm_summarize", lambda *a, **k: "The observed value was 10.",
        )

        out = summarizer.summarize_data(_data_ctx())

        assert out.summary_source == "legacy_text_fallback"

    def test_total_provider_outage_still_propagates(self, monkeypatch):
        # When the legacy fallback's own call fails too (provider tier down),
        # the error propagates — no infinite retry, no silent empty answer.
        monkeypatch.setattr(summarizer, "llm_summarize_structured", _rejected_failure)
        monkeypatch.setattr(summarizer, "llm_summarize", _rejected_failure)

        with pytest.raises(ProviderExecutionError):
            summarizer.summarize_data(_data_ctx())


class TestConceptualPathDegrades:
    def test_ambiguous_provider_failure_uses_legacy_conceptual_fallback(self, monkeypatch):
        monkeypatch.setattr(
            summarizer, "llm_summarize_structured_conceptual", _ambiguous_failure,
            raising=False,
        )
        # The conceptual path may use the same structured entry point; patch both.
        monkeypatch.setattr(summarizer, "llm_summarize_structured", _ambiguous_failure)
        monkeypatch.setattr(
            summarizer, "llm_summarize", lambda *a, **k: "A market definition.",
        )

        ctx = QueryContext(query="What is the balancing market?")
        ctx.is_conceptual = True

        out = summarizer.answer_conceptual(ctx)

        assert out.summary
        assert out.summary_source in (
            "legacy_conceptual_text_fallback",
            "conceptual_summary",  # pre-structured path if knowledge empty
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
