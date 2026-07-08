"""Tests for contract continuity slice 1 (design item 2, flag-gated OFF)."""

from __future__ import annotations

import json
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from types import SimpleNamespace

from agent.contract_continuity import continuity_snapshot_json
from contracts.question_analysis import QuestionAnalysis
from models import QueryContext
from utils import session_memory


def _qa_payload() -> dict:
    return {
        "version": "question_analysis_v1",
        "raw_query": "monthly balancing prices for 2023",
        "canonical_query_en": "monthly balancing prices for 2023",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "data_retrieval",
            "analysis_mode": "light",
            "intent": "price series",
            "needs_clarification": False,
            "confidence": 0.9,
        },
        "routing": {
            "preferred_path": "tool",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": True,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {},
        "tooling": {
            "candidate_tools": [{
                "name": "get_prices", "score": 0.9, "reason": "price data",
                "params_hint": {
                    "metric": "balancing", "currency": "GEL",
                    "start_date": "2023-01-01", "end_date": "2023-12-31",
                },
            }],
        },
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }


def _authoritative_ctx() -> QueryContext:
    ctx = QueryContext(query="monthly balancing prices for 2023")
    ctx.question_analysis = QuestionAnalysis(**_qa_payload())
    ctx.question_analysis_source = "llm_active"
    return ctx


# --- snapshot builder ------------------------------------------------------

def test_snapshot_contains_routed_fields_and_hint():
    snapshot = json.loads(continuity_snapshot_json(_authoritative_ctx()))
    assert snapshot["query_type"] == "data_retrieval"
    assert snapshot["top_tool"] == "get_prices"
    assert snapshot["params_hint"]["start_date"] == "2023-01-01"
    assert snapshot["canonical_query_en"] == "monthly balancing prices for 2023"


def test_snapshot_empty_without_authoritative_analysis():
    assert continuity_snapshot_json(QueryContext(query="q")) == ""


def test_snapshot_never_raises_on_fake_ctx():
    fake = SimpleNamespace(has_authoritative_question_analysis=True,
                           question_analysis=SimpleNamespace())  # no .tooling
    assert continuity_snapshot_json(fake) == ""


def test_snapshot_dropped_when_over_cap(monkeypatch):
    monkeypatch.setattr("agent.contract_continuity._MAX_SNAPSHOT_CHARS", 10)
    assert continuity_snapshot_json(_authoritative_ctx()) == ""


# --- session storage -------------------------------------------------------

def test_session_last_contract_round_trip():
    session_memory.set_last_contract("s-cc-1", '{"a":1}')
    assert session_memory.get_last_contract("s-cc-1") == '{"a":1}'
    assert session_memory.get_last_contract("s-cc-absent") == ""
    session_memory.set_last_contract("s-cc-2", "x" * 5000)  # over cap → rejected
    assert session_memory.get_last_contract("s-cc-2") == ""


# --- analyzer prompt block -------------------------------------------------

def test_block_inserted_after_user_question_when_provided():
    from core.llm import _ANALYZER_BLOCK_PREVIOUS_CONTRACT, _build_analyzer_prompt_blocks

    blocks = _build_analyzer_prompt_blocks(
        "and for 2023?", "", "single_value", "default",
        previous_contract='{"top_tool":"get_prices"}',
    )
    names = [name for name, _ in blocks]
    assert names[0] == "UNTRUSTED_USER_QUESTION"
    assert names[1] == _ANALYZER_BLOCK_PREVIOUS_CONTRACT
    body = dict(blocks)[_ANALYZER_BLOCK_PREVIOUS_CONTRACT]
    assert '{"top_tool":"get_prices"}' in body


def test_block_absent_when_not_provided():
    from core.llm import _ANALYZER_BLOCK_PREVIOUS_CONTRACT, _build_analyzer_prompt_blocks

    blocks = _build_analyzer_prompt_blocks("and for 2023?", "", "single_value", "default")
    assert _ANALYZER_BLOCK_PREVIOUS_CONTRACT not in [name for name, _ in blocks]


def test_truncation_lists_drop_contract_second():
    from core.llm import (
        _ANALYZER_BLOCK_PREVIOUS_CONTRACT,
        _ANALYZER_TRUNCATION_DATA,
        _ANALYZER_TRUNCATION_KNOWLEDGE,
    )

    for profile in (_ANALYZER_TRUNCATION_DATA, _ANALYZER_TRUNCATION_KNOWLEDGE):
        assert profile[0] == "UNTRUSTED_CONVERSATION_HISTORY"
        assert profile[1] == _ANALYZER_BLOCK_PREVIOUS_CONTRACT


def test_clarify_profile_preserves_contract_and_history_longest():
    from core.llm import (
        _ANALYZER_BLOCK_PREVIOUS_CONTRACT,
        _ANALYZER_TRUNCATION_DATA,
        _move_history_to_end,
    )

    moved = _move_history_to_end(_ANALYZER_TRUNCATION_DATA, "clarify")
    assert moved[-1] == "UNTRUSTED_CONVERSATION_HISTORY"
    assert moved[-2] == _ANALYZER_BLOCK_PREVIOUS_CONTRACT
    # Non-clarify profiles keep the base order.
    assert _move_history_to_end(_ANALYZER_TRUNCATION_DATA, "default") == list(
        _ANALYZER_TRUNCATION_DATA
    )


# --- cache key + planner threading -----------------------------------------

class _RecordingCache:
    def __init__(self, canned_response: str):
        self.keys: list[str] = []
        self._canned = canned_response

    def get(self, key: str):
        self.keys.append(key)
        return self._canned

    def set(self, key: str, value: str):  # pragma: no cover - not hit (cache hits)
        pass

    def mark_in_flight(self, key: str):  # pragma: no cover
        pass


def test_previous_contract_participates_in_cache_key(monkeypatch):
    from core import llm as llm_core

    canned = QuestionAnalysis(**_qa_payload()).model_dump_json()
    recorder = _RecordingCache(canned)
    monkeypatch.setattr(llm_core, "llm_cache", recorder)

    llm_core.llm_analyze_question("same in USD", previous_contract='{"a":1}')
    llm_core.llm_analyze_question("same in USD", previous_contract='{"a":2}')

    assert len(recorder.keys) == 2
    assert recorder.keys[0] != recorder.keys[1]


def test_planner_threads_snapshot_only_when_flag_on(monkeypatch):
    from agent import planner

    captured: list[str] = []

    def _fake_llm_analyze(user_query, conversation_history=None, previous_contract="",
                          evidence_anomaly_note=""):
        captured.append(previous_contract)
        return QuestionAnalysis(**_qa_payload())

    monkeypatch.setattr(planner, "llm_analyze_question", _fake_llm_analyze)

    ctx = QueryContext(query="monthly balancing prices for 2023")
    ctx.previous_contract_snapshot = '{"top_tool":"get_prices"}'

    monkeypatch.setattr(planner, "ENABLE_CONTRACT_CONTINUITY", False)
    planner.analyze_question(ctx, source="llm_active")
    monkeypatch.setattr(planner, "ENABLE_CONTRACT_CONTINUITY", True)
    ctx2 = QueryContext(query="monthly balancing prices for 2023")
    ctx2.previous_contract_snapshot = '{"top_tool":"get_prices"}'
    planner.analyze_question(ctx2, source="llm_active")

    assert captured == ["", '{"top_tool":"get_prices"}']
