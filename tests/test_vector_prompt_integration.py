import json
import os

import sqlalchemy

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


class _DummyResult:
    def fetchall(self):
        return []

    def keys(self):
        return []


class _DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        return _DummyResult()


class _DummyEngine:
    def connect(self):
        return _DummyConnection()


sqlalchemy.create_engine = lambda *args, **kwargs: _DummyEngine()  # type: ignore[assignment]

from agent import planner, summarizer  # noqa: E402
from contracts.question_analysis import QuestionAnalysis  # noqa: E402
from contracts.vector_knowledge import (  # noqa: E402
    RetrievalStrategy,
    VectorChunkRecord,
    VectorKnowledgeBundle,
    VectorKnowledgeMode,
)
import core.llm as llm_core  # noqa: E402
from core.llm import SummaryEnvelope  # noqa: E402
from models import QueryContext  # noqa: E402


def _vector_bundle():
    return VectorKnowledgeBundle(
        query="What is GENEX?",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.hybrid,
        top_k=4,
        chunk_count=1,
        chunks=[],
    )


def _analysis_payload():
    return {
        "version": "question_analysis_v1",
        "raw_query": "what is genex?",
        "canonical_query_en": "What is GENEX?",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "conceptual_definition",
            "analysis_mode": "light",
            "intent": "market_participant_definition",
            "needs_clarification": False,
            "confidence": 0.95,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
        },
        "knowledge": {"candidate_topics": [{"name": "market_structure", "score": 0.98}]},
        "tooling": {"candidate_tools": []},
        "sql_hints": {
            "metric": None,
            "entities": [],
            "aggregation": None,
            "dimensions": [],
            "period": None,
        },
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
            "preferred_chart_family": None,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_trend_context": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
    }


def test_planner_passes_active_vector_prompt(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        planner,
        "_generate_plan_and_sql_with_retry",
        lambda **kwargs: captured.update(kwargs) or ({"intent": "general"}, "select 1"),
    )
    ctx = QueryContext(query="Need plan", mode="analyst", lang_instruction="Respond in English.")
    ctx.question_analysis = QuestionAnalysis.model_validate(_analysis_payload())
    ctx.question_analysis_source = "llm_active"
    ctx.vector_knowledge = _vector_bundle()
    ctx.vector_knowledge_source = "vector_active"
    ctx.vector_knowledge_prompt = "EXTERNAL_SOURCE_PASSAGES:\n[1] Market Rules"

    planner.generate_plan(ctx)

    assert "EXTERNAL_SOURCE_PASSAGES" in captured["vector_knowledge"]


def test_conceptual_summary_passes_active_vector_prompt(monkeypatch):
    captured = {}
    monkeypatch.setattr(summarizer, "get_relevant_domain_knowledge", lambda *args, **kwargs: '{"market_structure":"GENEX text"}')
    monkeypatch.setattr(
        summarizer,
        "llm_summarize_structured",
        lambda *args, **kwargs: captured.update(kwargs) or SummaryEnvelope(
            answer="GENEX answer",
            claims=[],
            citations=["domain_knowledge"],
            confidence=0.9,
        ),
    )
    ctx = QueryContext(query="what is genex?", lang_instruction="Respond in English.")
    ctx.question_analysis = QuestionAnalysis.model_validate(_analysis_payload())
    ctx.question_analysis_source = "llm_active"
    ctx.vector_knowledge = _vector_bundle()
    ctx.vector_knowledge_source = "vector_active"
    ctx.vector_knowledge_prompt = "EXTERNAL_SOURCE_PASSAGES:\n[1] Market Rules"

    out = summarizer.answer_conceptual(ctx)

    assert out.summary == "GENEX answer"
    assert "EXTERNAL_SOURCE_PASSAGES" in captured["vector_knowledge"]


def test_conceptual_summary_uses_vector_as_primary_evidence(monkeypatch):
    captured = {}

    def _fake_domain_knowledge(_query, use_cache=True, preferred_topics=None):
        captured["preferred_topics"] = preferred_topics
        return json.dumps(
            {
                "eligible_participants": "Eligible participants are listed in the market rules.",
                "exchange_participation": "Registration is required before participation.",
                "general_definitions": "General background definition.",
            }
        )

    monkeypatch.setattr(summarizer, "get_relevant_domain_knowledge", _fake_domain_knowledge)
    monkeypatch.setattr(
        summarizer,
        "llm_summarize_structured",
        lambda *args, **kwargs: captured.update(kwargs) or SummaryEnvelope(
            answer="Export answer",
            claims=[],
            citations=["external_source_passages"],
            confidence=0.9,
        ),
    )
    ctx = QueryContext(query="How can electricity be exported?", lang_instruction="Respond in English.")
    ctx.vector_knowledge = VectorKnowledgeBundle(
        query="How can electricity be exported?",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.hybrid,
        top_k=4,
        chunk_count=1,
        chunks=[
            VectorChunkRecord(
                id="chunk-1",
                document_id="doc-1",
                document_title="Electricity (Capacity) Market Rules",
                source_key="capacity_rules",
                section_title="Export conditions",
                text_content="Export is allowed subject to the listed procedure.",
            )
        ],
        filters={"preferred_topics": ["eligible_participants", "exchange_participation"]},
    )
    ctx.vector_knowledge_source = "vector_active"
    ctx.vector_knowledge_prompt = (
        "EXTERNAL_SOURCE_PASSAGES:\n"
        "[1] Electricity (Capacity) Market Rules | section: Export conditions"
    )

    out = summarizer.answer_conceptual(ctx)

    assert out.summary == "Export answer"
    assert "PRIMARY EVIDENCE RULES" in captured["stats_hint"]
    assert captured["preferred_topics"] == ["eligible_participants", "exchange_participation"]
    assert json.loads(captured["domain_knowledge"]) == {
        "eligible_participants": "Eligible participants are listed in the market rules.",
        "exchange_participation": "Registration is required before participation.",
        "general_definitions": "General background definition.",
    }
    assert "EXTERNAL_SOURCE_PASSAGES" in captured["vector_knowledge"]


def test_structured_summary_prompt_prioritizes_external_source_passages(monkeypatch):
    captured = {}

    class _DummyCache:
        def get(self, _key):
            return None

        def set(self, _key, _value):
            return None

    class _DummyMessage:
        content = '{"answer":"ok","claims":[],"citations":["external_source_passages"],"confidence":0.9}'
        response_metadata = {}

    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_args, **_kwargs: None)

    def _capture_invoke(_llm, messages, _model_name):
        captured["system"] = messages[0][1]
        captured["prompt"] = messages[1][1]
        return _DummyMessage()

    monkeypatch.setattr(llm_core, "_invoke_with_resilience", _capture_invoke)

    llm_core.llm_summarize_structured(
        user_query="How can electricity be exported?",
        data_preview="",
        stats_hint="conceptual",
        lang_instruction="Respond in English.",
        domain_knowledge='{"general_definitions":"background"}',
        vector_knowledge="EXTERNAL_SOURCE_PASSAGES:\n[1] Capacity rules | section: Export conditions",
    )

    assert "primary evidence" in captured["system"].lower()
    assert 'prefer citing "external_source_passages"' in captured["prompt"].lower()
