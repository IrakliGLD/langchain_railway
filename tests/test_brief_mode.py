from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import core.llm as llm_core
from agent import summarizer
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
