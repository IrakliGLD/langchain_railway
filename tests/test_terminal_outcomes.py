"""Tests for P4.4 honest terminal outcomes (finding H12).

A data-primary request whose generated SQL fails validation/relevance must not
be answered as a conceptual narrative. The evidence-unavailable path is
deterministic (no invented numbers), the outcome taxonomy is always recorded,
and the user-facing routing change is gated by ENABLE_HONEST_TERMINAL_OUTCOMES.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import re

from agent import pipeline, summarizer
from models import QueryContext, ResponseMode, TerminalOutcome
from utils.language import get_evidence_unavailable_message
from utils.metrics import metrics

# ---------------------------------------------------------------------------
# summarizer.answer_evidence_unavailable
# ---------------------------------------------------------------------------


class TestAnswerEvidenceUnavailable:
    def test_sets_transparent_state_and_no_claims(self):
        ctx = QueryContext(query="what was the average balancing price in Ruritania")
        ctx.lang_code = "en"
        ctx.skip_sql_reason = "sql_validation_failed:unwhitelisted table"

        before = dict(metrics.terminal_outcome_events)
        out = summarizer.answer_evidence_unavailable(ctx)

        assert out.summary_source == "evidence_unavailable"
        assert out.summary_claims == []
        assert out.terminal_outcome == TerminalOutcome.EVIDENCE_UNAVAILABLE.value
        # No numeric claims can appear — message is a fixed template.
        assert not re.search(r"\d", out.summary)
        assert out.summary == get_evidence_unavailable_message("en")
        # Gate passes (no claims to ground) but is explicitly tagged.
        assert out.summary_provenance_gate_passed is True
        assert out.summary_provenance_gate_reason == "not_applicable_evidence_unavailable"
        key = TerminalOutcome.EVIDENCE_UNAVAILABLE.value
        assert metrics.terminal_outcome_events.get(key, 0) == before.get(key, 0) + 1

    def test_localized_message(self):
        for lang in ("en", "ka", "ru"):
            ctx = QueryContext(query="q")
            ctx.lang_code = lang
            out = summarizer.answer_evidence_unavailable(ctx)
            assert out.summary == get_evidence_unavailable_message(lang)


# ---------------------------------------------------------------------------
# _answer_skipped_sql_data_failure routing
# ---------------------------------------------------------------------------


class TestSkippedSqlRouting:
    def _data_primary_ctx(self) -> QueryContext:
        ctx = QueryContext(query="average balancing price last quarter")
        ctx.lang_code = "en"
        ctx.response_mode = ResponseMode.DATA_PRIMARY.value
        ctx.skip_sql = True
        ctx.skip_sql_reason = "sql_relevance_blocked:irrelevant"
        ctx.is_conceptual = False
        return ctx

    def test_gated_off_uses_conceptual_but_records_shadow(self, monkeypatch):
        monkeypatch.setattr(pipeline, "ENABLE_HONEST_TERMINAL_OUTCOMES", False)
        # Avoid a real LLM call: stub answer_conceptual.
        def _fake_conceptual(ctx):
            ctx.summary = "A conceptual narrative."
            ctx.summary_source = "conceptual"
            return ctx
        monkeypatch.setattr(pipeline.summarizer, "answer_conceptual", _fake_conceptual)

        before = dict(metrics.terminal_outcome_events)
        ctx = pipeline._answer_skipped_sql_data_failure(self._data_primary_ctx())

        assert ctx.summary_source == "conceptual"
        assert ctx.terminal_outcome == TerminalOutcome.CONCEPTUAL_ANSWER.value
        shadow_key = f"{TerminalOutcome.EVIDENCE_UNAVAILABLE.value}_shadow"
        assert metrics.terminal_outcome_events.get(shadow_key, 0) == before.get(shadow_key, 0) + 1

    def test_gated_on_uses_evidence_unavailable(self, monkeypatch):
        monkeypatch.setattr(pipeline, "ENABLE_HONEST_TERMINAL_OUTCOMES", True)
        ctx = pipeline._answer_skipped_sql_data_failure(self._data_primary_ctx())
        assert ctx.summary_source == "evidence_unavailable"
        assert ctx.terminal_outcome == TerminalOutcome.EVIDENCE_UNAVAILABLE.value
        assert not re.search(r"\d", ctx.summary)

    def test_knowledge_primary_stays_conceptual_even_when_enabled(self, monkeypatch):
        monkeypatch.setattr(pipeline, "ENABLE_HONEST_TERMINAL_OUTCOMES", True)
        def _fake_conceptual(ctx):
            ctx.summary = "A conceptual narrative."
            ctx.summary_source = "conceptual"
            return ctx
        monkeypatch.setattr(pipeline.summarizer, "answer_conceptual", _fake_conceptual)

        ctx = self._data_primary_ctx()
        ctx.response_mode = ResponseMode.KNOWLEDGE_PRIMARY.value
        out = pipeline._answer_skipped_sql_data_failure(ctx)

        assert out.summary_source == "conceptual"
        assert out.terminal_outcome == TerminalOutcome.CONCEPTUAL_ANSWER.value


class TestConfigDefault:
    def test_honest_terminal_outcomes_default_off(self):
        import config

        assert config.ENABLE_HONEST_TERMINAL_OUTCOMES is False


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
