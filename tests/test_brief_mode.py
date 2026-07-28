from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import core.llm as llm_core
from agent import pipeline, summarizer
from agent.answer_mode_policy import (
    BRIEF_MAX_WORDS,
    apply_answer_mode_policy,
    count_words,
)
from models import QueryContext
from skills.loader import get_answer_length_guidance


def _long_answer() -> str:
    return " ".join(
        f"Finding {index} is grounded in the supplied evidence and includes useful context."
        for index in range(1, 30)
    )


def test_brief_policy_enforces_a_hard_word_cap_without_touching_charts() -> None:
    chart = {"type": "line", "data": [{"date": "2024-01-01", "value": 10}]}
    ctx = QueryContext(
        query="Explain the trend.",
        answer_mode="brief",
        summary=_long_answer(),
        summary_source="structured_summary",
        charts=[chart],
    )

    changed = apply_answer_mode_policy(ctx)

    assert changed is True
    assert count_words(ctx.summary) <= BRIEF_MAX_WORDS
    assert ctx.charts == [chart]


def test_standard_policy_is_byte_for_byte_unchanged() -> None:
    answer = _long_answer()
    ctx = QueryContext(
        query="Explain the trend.",
        answer_mode="standard",
        summary=answer,
        summary_source="structured_summary",
    )

    assert apply_answer_mode_policy(ctx) is False
    assert ctx.summary == answer


def test_brief_policy_does_not_truncate_required_clarification() -> None:
    answer = _long_answer()
    ctx = QueryContext(
        query="Ambiguous question",
        answer_mode="brief",
        summary=answer,
        summary_source="clarification_request",
    )

    assert apply_answer_mode_policy(ctx) is False
    assert ctx.summary == answer


def test_brief_policy_preserves_complete_regulatory_enumeration() -> None:
    answer = _long_answer()
    ctx = QueryContext(
        query="List every eligibility requirement.",
        answer_mode="brief",
        summary=answer,
        summary_source="structured_conceptual_summary",
        question_analysis=SimpleNamespace(
            classification=SimpleNamespace(
                query_type=SimpleNamespace(value="regulatory_procedure")
            ),
            answer_kind=None,
        ),
    )

    assert apply_answer_mode_policy(ctx) is False
    assert ctx.summary == answer


def test_brief_guidance_reference_remains_available_for_non_semantic_consumers() -> None:
    guidance = get_answer_length_guidance("brief")
    assert "BRIEF MODE" in guidance
    assert str(BRIEF_MAX_WORDS) in guidance
    assert get_answer_length_guidance("standard") == ""


def test_structured_summarizer_shares_mode_neutral_prompt_and_cache(monkeypatch) -> None:
    captured_prompts = []
    captured_keys = []
    values = {}

    class _DummyCache:
        def get(self, key):
            captured_keys.append(key)
            return values.get(key)

        def set(self, key, value):
            values[key] = value
            return None

    class _DummyMessage:
        content = '{"answer":"ok","claims":[],"citations":[],"confidence":0.9}'
        response_metadata = {}

    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    def _capture_invoke(_llm, messages, _model_name):
        captured_prompts.append(messages[1][1])
        return _DummyMessage()

    monkeypatch.setattr(llm_core, "_invoke_with_resilience", _capture_invoke)

    llm_core.llm_summarize_structured(
        user_query="Explain the balancing price trend.",
        data_preview="date,value\n2024-01-01,10",
        stats_hint="The value increased.",
        answer_mode="brief",
    )
    llm_core.llm_summarize_structured(
        user_query="Explain the balancing price trend.",
        data_preview="date,value\n2024-01-01,10",
        stats_hint="The value increased.",
        answer_mode="standard",
    )

    assert len(captured_prompts) == 1
    assert "BRIEF MODE" not in captured_prompts[0]
    assert f"Hard maximum: {BRIEF_MAX_WORDS} words" not in captured_prompts[0]
    assert len(captured_keys) == 2
    assert captured_keys[0] == captured_keys[1]
    assert "|am=" not in captured_keys[0]


def test_structured_summarizer_logs_content_free_mode_and_cache_status(
    monkeypatch, caplog
) -> None:
    values = {}

    class _DummyCache:
        def get(self, key):
            return values.get(key)

        def set(self, key, value):
            values[key] = value
            return None

    class _DummyMessage:
        content = '{"answer":"ok","claims":[],"citations":[],"confidence":0.9}'
        response_metadata = {}

    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        llm_core,
        "_invoke_with_resilience",
        lambda *_args, **_kwargs: _DummyMessage(),
    )
    caplog.set_level("INFO", logger="Enai")

    for answer_mode in ("brief", "standard"):
        llm_core.llm_summarize_structured(
            user_query="PRIVATE QUESTION MUST NOT APPEAR",
            data_preview="date,value\n2024-01-01,10",
            stats_hint="The value increased.",
            answer_mode=answer_mode,
        )

    events = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Structured summary cache lookup:")
    ]
    assert any("answer_mode=brief" in event and "cache_status=miss" in event for event in events)
    assert any("answer_mode=standard" in event and "cache_status=hit" in event for event in events)
    assert all("PRIVATE QUESTION MUST NOT APPEAR" not in event for event in events)


def test_standard_summarizer_cache_key_has_no_answer_mode_suffix(monkeypatch) -> None:
    captured_keys = []

    class _RecordingCache:
        def get(self, key):
            captured_keys.append(key)
            return None

        def set(self, _key, _value):
            return None

    class _DummyMessage:
        content = '{"answer":"ok","claims":[],"citations":[],"confidence":0.9}'
        response_metadata = {}

    monkeypatch.setattr(llm_core, "llm_cache", _RecordingCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        llm_core,
        "_invoke_with_resilience",
        lambda *_args, **_kwargs: _DummyMessage(),
    )

    llm_core.llm_summarize_structured(
        user_query="Explain the balancing price trend.",
        data_preview="date,value\n2024-01-01,10",
        stats_hint="The value increased.",
    )

    assert captured_keys
    assert "|am=standard" not in captured_keys[0]
    assert "|am=" not in captured_keys[0]


def test_data_summarizer_applies_brief_policy_after_deterministic_render(monkeypatch) -> None:
    chart = {"type": "line", "data": [{"date": "2024-01-01", "value": 10}]}
    monkeypatch.setattr(summarizer, "_try_generic_renderer", lambda _ctx: _long_answer())
    monkeypatch.setattr(summarizer, "evaluate_render_fitness", lambda _ctx: [])

    ctx = QueryContext(
        query="Explain the trend.",
        answer_mode="brief",
        charts=[chart],
    )
    result = summarizer.summarize_data(ctx)

    assert result.summary_source == "generic_renderer"
    assert count_words(result.summary) <= BRIEF_MAX_WORDS
    assert result.charts == [chart]


def test_brief_pipeline_uses_one_knowledge_call_and_skips_every_heavy_stage(
    monkeypatch,
) -> None:
    calls = []

    def _brief_answer(ctx):
        calls.append(ctx.query)
        ctx.summary = "Brief knowledge answer."
        ctx.summary_source = "brief_knowledge"
        return ctx

    def _forbidden(stage):
        def _raise(*_args, **_kwargs):
            raise AssertionError(f"Brief must skip {stage}")

        return _raise

    monkeypatch.setattr(
        pipeline.summarizer,
        "answer_brief_knowledge",
        _brief_answer,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline.planner,
        "analyze_question_active",
        _forbidden("question analysis"),
    )
    monkeypatch.setattr(
        pipeline.planner,
        "analyze_question_shadow",
        _forbidden("question analysis"),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_vector_knowledge_stage",
        _forbidden("vector retrieval"),
    )
    monkeypatch.setattr(
        pipeline.evidence_planner,
        "build_evidence_plan",
        _forbidden("evidence planning"),
    )
    monkeypatch.setattr(pipeline, "match_tool", _forbidden("typed tools"))
    monkeypatch.setattr(pipeline, "execute_tool", _forbidden("typed tools"))
    monkeypatch.setattr(pipeline, "derive_evidence", _forbidden("analysis"))
    monkeypatch.setattr(
        pipeline.chart_pipeline,
        "build_chart",
        _forbidden("chart generation"),
    )

    result = pipeline._process_query_impl(
        "Why did the balancing price change in July 2025?",
        answer_mode="brief",
    )

    assert calls == ["Why did the balancing price change in July 2025?"]
    assert result.summary == "Brief knowledge answer."
    assert result.mode == "light"
    assert result.vector_retrieval_tier == "skip"
    assert result.skip_sql is True
    assert result.used_tool is False
    assert result.charts == []
    assert set(result.stage_timings_ms) == {
        "stage_0_prepare_context",
        "stage_0_1_brief_knowledge",
    }


def test_brief_knowledge_answer_sets_conceptual_terminal_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        summarizer,
        "llm_answer_brief_knowledge",
        lambda *_args, **_kwargs: "A balancing market settles real-time deviations.",
        raising=False,
    )
    ctx = QueryContext(
        query="What is a balancing market?",
        lang_code="en",
        lang_instruction="Respond in English.",
        answer_mode="brief",
    )

    result = summarizer.answer_brief_knowledge(ctx)

    assert result.summary == "A balancing market settles real-time deviations."
    assert result.summary_source == "brief_knowledge"
    assert result.summary_citations == []
    assert result.summary_provenance_gate_reason == "not_applicable_brief_knowledge"
    assert result.terminal_outcome == "conceptual_answer"
    assert count_words(result.summary) <= BRIEF_MAX_WORDS


def test_brief_model_call_is_single_compact_unstructured_invocation(monkeypatch) -> None:
    captured = {"bind": [], "calls": []}

    class _DummyCache:
        def get(self, _key):
            return None

        def set(self, _key, _value):
            return None

    class _DummyLlm:
        def bind(self, **kwargs):
            captured["bind"].append(kwargs)
            return self

    class _DummyMessage:
        content = "Use Standard mode for an evidence-based July 2025 analysis."
        response_metadata = {}

    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_primary_model_name", lambda: "openai/test")
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: _DummyLlm())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    def _invoke(llm, messages, model_name, stage):
        captured["calls"].append((llm, messages, model_name, stage))
        return _DummyMessage()

    monkeypatch.setattr(llm_core, "_invoke_at_stage", _invoke)
    monkeypatch.setattr(
        llm_core,
        "_fallback_to_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Brief must not make a fallback model call")
        ),
    )

    answer = llm_core.llm_answer_brief_knowledge(
        "Why did the balancing price change in July 2025?",
        lang_instruction="Respond in English.",
        conversation_history=[{"role": "user", "content": "Earlier question"}],
    )

    assert answer == "Use Standard mode for an evidence-based July 2025 analysis."
    assert captured["bind"] == [{"max_tokens": 256}]
    assert len(captured["calls"]) == 1
    _, messages, _, stage = captured["calls"][0]
    assert stage == "brief_knowledge"
    assert len(messages) == 2
    assert "UNTRUSTED_DATA_PREVIEW" not in messages[1][1]
    assert "UNTRUSTED_EXTERNAL_SOURCE_PASSAGES" not in messages[1][1]
    assert "at most 120 words" in messages[0][1]


def test_brief_model_failure_is_not_replayed_on_another_provider(monkeypatch) -> None:
    failure = llm_core.ProviderExecutionError(
        "temporary timeout",
        provider="nvidia",
        stage="brief_knowledge",
        disposition=llm_core.ProviderDeliveryDisposition.TIMED_OUT,
    )
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        llm_core,
        "_invoke_at_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        llm_core,
        "_fallback_to_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Brief must not make a fallback model call")
        ),
    )

    with pytest.raises(llm_core.ProviderExecutionError) as caught:
        llm_core.llm_answer_brief_knowledge("What is a balancing market?")

    assert caught.value is failure
