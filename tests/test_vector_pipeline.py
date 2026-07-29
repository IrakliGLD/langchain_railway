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
from contracts.vector_knowledge import (  # noqa: E402
    HybridRetrievalDiagnostics,
    HybridRetrievalMode,
    RetrievalStrategy,
    VectorChunkRecord,
    VectorKnowledgeBundle,
    VectorKnowledgeMode,
    VectorRetrievalFailure,
    VectorRetrievalFailureStage,
    VectorRetrievalOutcome,
    VectorRetrievalTier,
)


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
    assert captured["packed_chunk_count"] == 2
    assert captured["packed_sections"] == [
        "[1] Electricity (Capacity) Market Rules | section: Export conditions",
        "[2] Electricity (Capacity) Market Rules | section: Part II > Registration",
    ]
    assert captured["packed_truncated"] is False


def test_pipeline_traces_typed_retrieval_failure_without_raw_error(monkeypatch):
    events = []
    bundle = VectorKnowledgeBundle(
        query="What is GENEX?",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.dense_with_deterministic_rerank,
        top_k=4,
        outcome=VectorRetrievalOutcome.unavailable,
        failure=VectorRetrievalFailure(
            stage=VectorRetrievalFailureStage.query_embedding,
            reason="401/API_KEY_INVALID",
        ),
        error="query_embedding:401/API_KEY_INVALID",
    )
    monkeypatch.setattr(pipeline, "ENABLE_VECTOR_KNOWLEDGE_SHADOW", False)
    monkeypatch.setattr(pipeline, "ENABLE_VECTOR_KNOWLEDGE_HINTS", True)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_SHADOW", False)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_HINTS", False)
    monkeypatch.setattr(
        pipeline.planner,
        "prepare_context",
        lambda ctx: setattr(ctx, "is_conceptual", True) or ctx,
    )
    monkeypatch.setattr(
        pipeline,
        "retrieve_vector_knowledge",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        pipeline,
        "trace_detail",
        lambda *args, **kwargs: events.append((args, kwargs)),
    )
    monkeypatch.setattr(
        pipeline.summarizer,
        "answer_conceptual",
        lambda ctx: setattr(ctx, "summary", "Fallback answer") or ctx,
    )

    out = pipeline.process_query(
        "what is genex?",
        trace_id="trace-vk-failure",
        session_id="session-vk-failure",
    )

    vector_event = next(
        kwargs
        for args, kwargs in events
        if "stage_0_3_vector_knowledge" in args
    )
    assert vector_event["outcome"] == "unavailable"
    assert vector_event["failure_stage"] == "query_embedding"
    assert vector_event["failure_reason"] == "401/API_KEY_INVALID"
    assert "error" not in vector_event
    assert out.vector_knowledge_error == "query_embedding:401/API_KEY_INVALID"


def test_pipeline_traces_hybrid_shadow_disagreement(monkeypatch):
    events = []
    dense = VectorChunkRecord(
        id="chunk-dense",
        document_id="doc-dense",
        document_title="Dense Rules",
        source_key="dense-rules",
        section_title="Dense section",
        text_content="Dense evidence.",
        similarity_score=0.80,
    )
    fused = VectorChunkRecord(
        id="chunk-lexical",
        document_id="doc-lexical",
        document_title="Lexical Rules",
        source_key="lexical-rules",
        section_title="Exact phrase",
        text_content="Lexical evidence.",
    )
    bundle = VectorKnowledgeBundle(
        query="What is the exact phrase?",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.dense_with_deterministic_rerank,
        top_k=1,
        chunk_count=1,
        chunks=[dense],
        outcome=VectorRetrievalOutcome.matches,
        hybrid_diagnostics=HybridRetrievalDiagnostics(
            mode=HybridRetrievalMode.shadow,
            dense_chunk_ids=[dense.id],
            lexical_chunk_ids=[fused.id],
            fused_chunks=[fused],
        ),
    )
    monkeypatch.setattr(pipeline, "ENABLE_VECTOR_KNOWLEDGE_SHADOW", False)
    monkeypatch.setattr(pipeline, "ENABLE_VECTOR_KNOWLEDGE_HINTS", True)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_SHADOW", False)
    monkeypatch.setattr(pipeline, "ENABLE_QUESTION_ANALYZER_HINTS", False)
    monkeypatch.setattr(
        pipeline.planner,
        "prepare_context",
        lambda ctx: setattr(ctx, "is_conceptual", True) or ctx,
    )
    monkeypatch.setattr(
        pipeline,
        "retrieve_vector_knowledge",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        pipeline,
        "trace_detail",
        lambda *args, **kwargs: events.append((args, kwargs)),
    )
    monkeypatch.setattr(
        pipeline.summarizer,
        "answer_conceptual",
        lambda ctx: setattr(ctx, "summary", "Conceptual answer") or ctx,
    )

    pipeline.process_query(
        "what is the exact phrase?",
        trace_id="trace-hybrid-shadow",
        session_id="session-hybrid-shadow",
    )

    hybrid_event = next(
        kwargs
        for args, kwargs in events
        if "stage_0_3_vector_knowledge_hybrid" in args
    )
    assert hybrid_event["hybrid_mode"] == "shadow"
    assert hybrid_event["cutover_applied"] is False
    assert hybrid_event["dense_chunk_ids"] == ["chunk-dense"]
    assert hybrid_event["lexical_chunk_ids"] == ["chunk-lexical"]
    assert hybrid_event["fused_chunk_ids"] == ["chunk-lexical"]
    assert hybrid_event["dense_lexical_overlap_count"] == 0
    assert hybrid_event["dense_fused_overlap_count"] == 0
    assert hybrid_event["fused_sections"] == [
        "Lexical Rules | Exact phrase",
    ]
    assert hybrid_event["lexical_failure_stage"] == ""
    assert hybrid_event["lexical_failure_reason"] == ""


def test_pipeline_marks_disabled_or_skipped_retrieval_as_not_run(monkeypatch):
    monkeypatch.setattr(pipeline, "ENABLE_VECTOR_KNOWLEDGE_SHADOW", False)
    monkeypatch.setattr(pipeline, "ENABLE_VECTOR_KNOWLEDGE_HINTS", False)
    monkeypatch.setattr(
        pipeline,
        "retrieve_vector_knowledge",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled retrieval must not be called")
        ),
    )
    ctx = pipeline.QueryContext(query="Deterministic data query")

    pipeline._run_vector_knowledge_stage(
        ctx,
        VectorRetrievalTier.SKIP,
        "Deterministic data query",
    )

    assert (
        ctx.vector_knowledge_outcome
        is VectorRetrievalOutcome.not_run
    )
    assert ctx.vector_knowledge_error == ""
