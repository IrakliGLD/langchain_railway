"""Runtime contracts for structured LLM outputs."""

# Re-export question-analysis contracts used across planner, pipeline, and summarizer.
from contracts.question_analysis import QuestionAnalysis
from contracts.question_analysis_catalogs import (
    QUESTION_ANALYSIS_CHART_POLICY,
    QUESTION_ANALYSIS_DERIVED_METRIC_CATALOG,
    QUESTION_ANALYSIS_QUERY_TYPE_GUIDE,
    QUESTION_ANALYSIS_TOOL_CATALOG,
    QUESTION_ANALYSIS_TOPIC_CATALOG,
)
from contracts.vector_knowledge import (
    ChunkIngestRecord,
    DocumentRegistration,
    HybridRetrievalDiagnostics,
    HybridRetrievalMode,
    IngestionResult,
    RetrievalStrategy,
    VectorChunkRecord,
    VectorDocumentRecord,
    VectorKnowledgeBundle,
    VectorKnowledgeMode,
    VectorRetrievalFailure,
    VectorRetrievalFailureStage,
    VectorRetrievalFilters,
    VectorRetrievalOutcome,
)

# Keep the public contract surface explicit for downstream imports.
__all__ = [
    "QuestionAnalysis",
    "QUESTION_ANALYSIS_QUERY_TYPE_GUIDE",
    "QUESTION_ANALYSIS_TOPIC_CATALOG",
    "QUESTION_ANALYSIS_TOOL_CATALOG",
    "QUESTION_ANALYSIS_CHART_POLICY",
    "QUESTION_ANALYSIS_DERIVED_METRIC_CATALOG",
    "VectorKnowledgeMode",
    "HybridRetrievalMode",
    "HybridRetrievalDiagnostics",
    "RetrievalStrategy",
    "VectorDocumentRecord",
    "VectorChunkRecord",
    "VectorRetrievalFilters",
    "VectorKnowledgeBundle",
    "VectorRetrievalOutcome",
    "VectorRetrievalFailureStage",
    "VectorRetrievalFailure",
    "DocumentRegistration",
    "ChunkIngestRecord",
    "IngestionResult",
]
