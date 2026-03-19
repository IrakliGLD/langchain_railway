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
from contracts.vector_knowledge import RetrievalStrategy, VectorKnowledgeBundle, VectorKnowledgeMode  # noqa: E402
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
