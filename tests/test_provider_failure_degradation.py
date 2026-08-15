"""Regression: typed provider failures must degrade, not 500 (incident 2026-07-17).

F8 wrapped every provider failure in ProviderExecutionError, which subclasses
RuntimeError. The summarizer's two `except RuntimeError: raise` clauses —
written pre-F8 to re-raise only the breaker-open bare RuntimeError — then
re-raised every provider failure, skipping the P0.7-hardened legacy fallback
and turning an ambiguous NVIDIA delivery into a pipeline 500 on /ask.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from agent import summarizer
from contracts.question_analysis import AnswerKind
from models import QueryContext, TerminalOutcome
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
    def test_ambiguous_provider_failure_returns_transient_outcome_without_second_call(
        self, monkeypatch
    ):
        monkeypatch.setattr(summarizer, "llm_summarize_structured", _ambiguous_failure)
        legacy_calls = []
        monkeypatch.setattr(summarizer, "llm_summarize", lambda *a, **k: legacy_calls.append(1))

        out = summarizer.summarize_data(_data_ctx())

        assert out.summary_source == "transient_failure"
        assert out.terminal_outcome == TerminalOutcome.TRANSIENT_FAILURE.value
        assert out.summary_claims == []
        assert "try again" in out.summary.lower()
        assert legacy_calls == []

    def test_rejected_provider_failure_also_avoids_same_provider_replay(self, monkeypatch):
        monkeypatch.setattr(summarizer, "llm_summarize_structured", _rejected_failure)
        legacy_calls = []
        monkeypatch.setattr(summarizer, "llm_summarize", lambda *a, **k: legacy_calls.append(1))

        out = summarizer.summarize_data(_data_ctx())

        assert out.summary_source == "transient_failure"
        assert out.terminal_outcome == TerminalOutcome.TRANSIENT_FAILURE.value
        assert legacy_calls == []

    def test_non_provider_structured_failure_keeps_legacy_parser_fallback(self, monkeypatch):
        def _schema_failure(*_args, **_kwargs):
            raise ValueError("malformed structured response")

        monkeypatch.setattr(summarizer, "llm_summarize_structured", _schema_failure)
        monkeypatch.setattr(
            summarizer, "llm_summarize", lambda *a, **k: "The observed value was 10.",
        )

        out = summarizer.summarize_data(_data_ctx())

        assert out.summary_source == "legacy_text_fallback"
        assert "10" in out.summary


class TestTimeoutClassification:
    """Incident follow-up: locally-enforced timeouts are TIMED_OUT (fallback-safe)."""

    def test_stdlib_timeout_error_is_timed_out(self):
        from utils.provider_attempts import classify_provider_failure

        assert classify_provider_failure(TimeoutError("elapsed")) is (
            ProviderDeliveryDisposition.TIMED_OUT
        )

    def test_sdk_timeout_type_name_is_timed_out(self):
        from utils.provider_attempts import classify_provider_failure

        APITimeoutError = type("APITimeoutError", (Exception,), {})
        assert classify_provider_failure(APITimeoutError("request timed out")) is (
            ProviderDeliveryDisposition.TIMED_OUT
        )

    def test_wrapped_timeout_cause_is_timed_out(self):
        from utils.provider_attempts import classify_provider_failure

        wrapper = RuntimeError("langchain wrapper")
        wrapper.__cause__ = TimeoutError("inner timeout")
        assert classify_provider_failure(wrapper) is ProviderDeliveryDisposition.TIMED_OUT

    def test_connection_reset_stays_ambiguous(self):
        from utils.provider_attempts import classify_provider_failure

        assert classify_provider_failure(ConnectionResetError("reset")) is (
            ProviderDeliveryDisposition.AMBIGUOUS
        )

    def test_fallback_safety_matrix(self):
        def _err(disposition):
            return ProviderExecutionError(
                "x", provider="nvidia", stage="s", disposition=disposition,
            )

        assert _err(ProviderDeliveryDisposition.REJECTED).safe_to_fallback is True
        assert _err(ProviderDeliveryDisposition.TIMED_OUT).safe_to_fallback is True
        assert _err(ProviderDeliveryDisposition.AMBIGUOUS).safe_to_fallback is False
        assert _err(ProviderDeliveryDisposition.PERMANENT_FAILURE).safe_to_fallback is False
        # Same-provider replay stays REJECTED-only.
        assert _err(ProviderDeliveryDisposition.TIMED_OUT).safe_to_retry is False
        assert _err(ProviderDeliveryDisposition.REJECTED).safe_to_retry is True


class TestConceptualPathDegrades:
    def test_ambiguous_provider_failure_returns_transient_outcome_without_second_call(
        self, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            summarizer, "llm_summarize_structured_conceptual", _ambiguous_failure,
            raising=False,
        )
        # The conceptual path may use the same structured entry point; patch both.
        monkeypatch.setattr(summarizer, "llm_summarize_structured", _ambiguous_failure)
        legacy_calls = []
        monkeypatch.setattr(summarizer, "llm_summarize", lambda *a, **k: legacy_calls.append(1))

        ctx = QueryContext(query="What is the balancing market?")
        ctx.is_conceptual = True

        with caplog.at_level(logging.INFO, logger="Enai"):
            out = summarizer.answer_conceptual(ctx)

        assert out.summary_source == "transient_failure"
        assert out.terminal_outcome == TerminalOutcome.TRANSIENT_FAILURE.value
        assert legacy_calls == []
        messages = [record.getMessage() for record in caplog.records]
        assert not any("Conceptual answer generated" in message for message in messages)
        assert any(
            "Conceptual answer completed with terminal outcome" in message
            for message in messages
        )

    def test_atomic_list_renderer_skips_terminal_outcome_without_warning(self, caplog):
        ctx = QueryContext(query="List the eligible participants")
        ctx.question_analysis = SimpleNamespace(answer_kind=AnswerKind.LIST)
        ctx.question_analysis_source = "llm_active"
        ctx.summary_source = "transient_failure"
        ctx.terminal_outcome = TerminalOutcome.TRANSIENT_FAILURE.value
        ctx.summary_claims = []

        with caplog.at_level(logging.WARNING, logger="Enai"):
            summarizer._enforce_atomic_conceptual_list(ctx)

        assert not any(
            "Atomic list contract could not render" in record.getMessage()
            for record in caplog.records
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Content-free failure reasons (2026-07-22 embedding outage)
# ---------------------------------------------------------------------------

import types


class _FakeGoogleClientError(Exception):
    """Shape-faithful google.genai.errors.ClientError double (probe-verified)."""

    def __init__(self):
        super().__init__("400 INVALID_ARGUMENT. {'error': {...}}")
        self.code = 400
        self.status = "INVALID_ARGUMENT"
        self.message = "API key not valid. Please pass a valid API key."
        self.details = {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "API_KEY_INVALID",
                        "domain": "googleapis.com",
                    }
                ],
            }
        }
        self.response = types.SimpleNamespace(status_code=400)


def test_extract_failure_reason_google_client_error():
    """The 2026-07-22 outage was undiagnosable because the wrapped message
    collapsed to 'provider call failed: ClientError'. The extractor must
    surface the enum-shaped reason chain without any free text."""
    from utils.provider_attempts import extract_failure_reason

    assert extract_failure_reason(_FakeGoogleClientError()) == "400/INVALID_ARGUMENT/API_KEY_INVALID"


def test_extract_failure_reason_rejects_free_text_tokens():
    """Only strictly enum-shaped tokens may reach logs — provider messages
    can echo request content and this module is content-free by design."""
    from utils.provider_attempts import extract_failure_reason

    err = _FakeGoogleClientError()
    err.status = "Bad key sk-abc123 leaked"
    err.details = {"error": {"details": [{"reason": "not enum shaped!"}]}}
    reason = extract_failure_reason(err)
    assert reason == "400"
    assert "sk-abc123" not in reason


def test_extract_failure_reason_empty_for_plain_error():
    from utils.provider_attempts import extract_failure_reason

    assert extract_failure_reason(RuntimeError("boom")) == ""


def test_wrap_provider_failure_carries_reason_not_content():
    from utils.provider_attempts import (
        ProviderDeliveryDisposition,
        wrap_provider_failure,
    )

    wrapped = wrap_provider_failure(
        _FakeGoogleClientError(), provider="gemini", stage="query_embedding",
    )
    msg = str(wrapped)
    assert "[400/INVALID_ARGUMENT/API_KEY_INVALID]" in msg
    assert "API key not valid" not in msg
    assert wrapped.disposition == ProviderDeliveryDisposition.PERMANENT_FAILURE


def test_finish_provider_attempt_logs_failure_reason(caplog):
    import logging

    from utils.provider_attempts import (
        ProviderAttemptToken,
        ProviderDeliveryDisposition,
        finish_provider_attempt,
    )

    token = ProviderAttemptToken("", "gemini", "query_embedding", "", "", False)
    with caplog.at_level(logging.INFO, logger="Enai"):
        finish_provider_attempt(
            token,
            ProviderDeliveryDisposition.PERMANENT_FAILURE,
            failure_reason="400/INVALID_ARGUMENT/API_KEY_INVALID",
        )
        finish_provider_attempt(token, ProviderDeliveryDisposition.COMPLETED)

    lines = [r.getMessage() for r in caplog.records if "attempt finished" in r.getMessage()]
    assert any("failure_reason=400/INVALID_ARGUMENT/API_KEY_INVALID" in line for line in lines)
    assert not any("failure_reason" in line for line in lines if "disposition=completed" in line)
