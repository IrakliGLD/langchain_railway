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

from agent import pipeline  # noqa: E402
from contracts.vector_knowledge import RetrievalStrategy, VectorChunkRecord, VectorKnowledgeBundle, VectorKnowledgeMode  # noqa: E402


def test_pipeline_collects_vector_knowledge_in_shadow_mode(monkeypatch):
    bundle = VectorKnowledgeBundle(
        query="What is GENEX?",
        retrieval_mode=VectorKnowledgeMode.shadow,
        strategy=RetrievalStrategy.hybrid,
        top_k=4,
        chunk_count=1,
        chunks=[
            VectorChunkRecord(
                id="chunk-1",
                document_id="doc-1",
                document_title="Market Rules",
                source_key="rules",
                text_content="GENEX operates the exchange.",
            )
        ],
    )
    monkeypatch.setattr(pipeline, "ENABLE_VECTOR_KNOWLEDGE_SHADOW", True)
    monkeypatch.setattr(pipeline, "ENABLE_VECTOR_KNOWLEDGE_HINTS", False)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_SHADOW", False)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_HINTS", False)
    monkeypatch.setattr(pipeline.planner, "prepare_context", lambda ctx: setattr(ctx, "is_conceptual", True) or ctx)
    monkeypatch.setattr(pipeline, "retrieve_vector_knowledge", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(pipeline, "format_vector_knowledge_for_prompt", lambda _bundle: "EXTERNAL_SOURCE_PASSAGES:\n[1] Market Rules")
    monkeypatch.setattr(
        pipeline.summarizer,
        "answer_conceptual",
        lambda ctx: setattr(ctx, "summary", "Conceptual answer") or ctx,
    )

    out = pipeline.process_query("what is genex?", trace_id="trace-vk", session_id="session-vk")

    assert out.summary == "Conceptual answer"
    assert out.vector_knowledge is not None
    assert out.vector_knowledge.chunk_count == 1
    assert out.vector_knowledge_source == "vector_shadow"
    assert out.stage_timings_ms["stage_0_3_vector_knowledge"] >= 0.0


def test_pipeline_logs_top_section_titles_for_vector_knowledge(monkeypatch):
    captured = {}
    bundle = VectorKnowledgeBundle(
        query="How can electricity be exported?",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.hybrid,
        top_k=4,
        chunk_count=2,
        chunks=[
            VectorChunkRecord(
                id="chunk-1",
                document_id="doc-1",
                document_title="Electricity (Capacity) Market Rules",
                source_key="capacity_rules",
                section_title="Export conditions",
                text_content="Export rules text.",
            ),
            VectorChunkRecord(
                id="chunk-2",
                document_id="doc-1",
                document_title="Electricity (Capacity) Market Rules",
                source_key="capacity_rules",
                section_path="Part II > Registration",
                text_content="Registration rules text.",
            ),
        ],
    )
    monkeypatch.setattr(pipeline, "ENABLE_VECTOR_KNOWLEDGE_SHADOW", False)
    monkeypatch.setattr(pipeline, "ENABLE_VECTOR_KNOWLEDGE_HINTS", True)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_SHADOW", False)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_HINTS", False)
    monkeypatch.setattr(pipeline.planner, "prepare_context", lambda ctx: setattr(ctx, "is_conceptual", True) or ctx)
    monkeypatch.setattr(pipeline, "retrieve_vector_knowledge", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        pipeline,
        "format_vector_knowledge_for_prompt",
        lambda _bundle: "EXTERNAL_SOURCE_PASSAGES:\n[1] Capacity rules",
    )
    monkeypatch.setattr(pipeline, "trace_detail", lambda *_args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        pipeline.summarizer,
        "answer_conceptual",
        lambda ctx: setattr(ctx, "summary", "Conceptual answer") or ctx,
    )

    pipeline.process_query("how can electricity be exported?", trace_id="trace-sections", session_id="session-sections")

    assert captured["top_sources"] == [
        "Electricity (Capacity) Market Rules",
        "Electricity (Capacity) Market Rules",
    ]
    assert captured["top_sections"] == [
        "Electricity (Capacity) Market Rules | Export conditions",
        "Electricity (Capacity) Market Rules | Part II > Registration",
    ]
