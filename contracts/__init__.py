"""Runtime contracts for structured LLM outputs."""

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
    IngestionResult,
    RetrievalStrategy,
    VectorChunkRecord,
    VectorDocumentRecord,
    VectorKnowledgeBundle,
    VectorKnowledgeMode,
    VectorRetrievalFilters,
)

__all__ = [
    "QuestionAnalysis",
    "QUESTION_ANALYSIS_QUERY_TYPE_GUIDE",
    "QUESTION_ANALYSIS_TOPIC_CATALOG",
    "QUESTION_ANALYSIS_TOOL_CATALOG",
    "QUESTION_ANALYSIS_CHART_POLICY",
    "QUESTION_ANALYSIS_DERIVED_METRIC_CATALOG",
    "VectorKnowledgeMode",
    "RetrievalStrategy",
    "VectorDocumentRecord",
    "VectorChunkRecord",
    "VectorRetrievalFilters",
    "VectorKnowledgeBundle",
    "DocumentRegistration",
    "ChunkIngestRecord",
    "IngestionResult",
]
