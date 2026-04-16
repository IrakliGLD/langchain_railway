"""Runtime contracts for vector-backed knowledge retrieval."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Retrieval-mode enums describe how vector knowledge should be assembled and used.
class VectorKnowledgeMode(str, Enum):
    """How vector retrieval output is being used in the pipeline."""

    shadow = "shadow"
    active = "active"


class RetrievalStrategy(str, Enum):
    """High-level retrieval mode for knowledge assembly."""

    curated_only = "curated_only"
    vector_only = "vector_only"
    hybrid = "hybrid"


class VectorRetrievalTier(str, Enum):
    """Retrieval effort tier — how aggressively to search vector knowledge.

    Chosen upstream from ``answer_kind`` + ``render_style`` so the retrieval
    cost tracks what the final answer will actually use:

    * ``FULL`` — default top-K + full candidate pool. For knowledge and
      explanation answers that consume the retrieved passages directly.
    * ``LIGHT`` — ``top_k=2``, reduced candidate pool, no re-rank.  Enough
      context for narrative data answers that only sprinkle in background
      definitions.
    * ``SKIP`` — do not call the vector store at all.  Deterministic data
      paths bypass the LLM summarizer, and clarify paths have no data to
      ground.
    """

    FULL = "full"
    LIGHT = "light"
    SKIP = "skip"


# Stored document and chunk records mirror the Supabase persistence model.
class VectorDocumentRecord(BaseModel):
    """Metadata for a source document stored in Supabase."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_key: str
    title: str
    document_type: str = ""
    issuer: str = ""
    language: str = ""
    source_url: Optional[str] = None
    storage_path: Optional[str] = None
    logical_key: str = ""
    effective_date: Optional[str] = None
    effective_end_date: Optional[str] = None
    published_date: Optional[str] = None
    version_label: Optional[str] = None
    is_latest: bool = True
    is_active: bool = True
    abolished: bool = False
    supersedes_document_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_lifecycle(self) -> "VectorDocumentRecord":
        if not self.logical_key.strip():
            self.logical_key = self.source_key
        if self.abolished and self.is_active:
            raise ValueError("abolished documents cannot be active")
        return self


class VectorChunkRecord(BaseModel):
    """A vector-searchable text chunk plus source metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    document_title: str = ""
    document_type: str = ""
    document_issuer: str = ""
    source_key: str = ""
    chunk_index: int = 0
    section_title: str = ""
    section_path: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text_content: str
    token_count: int = 0
    language: str = ""
    topics: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity_score: Optional[float] = None

    @field_validator("text_content")
    @classmethod
    def _validate_text_content(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("text_content cannot be empty")
        return text


class VectorRetrievalFilters(BaseModel):
    """Optional filters for vector-backed document retrieval."""

    model_config = ConfigDict(extra="forbid")

    document_types: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    issuers: List[str] = Field(default_factory=list)
    preferred_topics: List[str] = Field(default_factory=list)
    boost_terms: List[str] = Field(default_factory=list)


# Retrieval bundles are carried through the runtime after search completes.
class VectorKnowledgeBundle(BaseModel):
    """Structured retrieval output carried through the pipeline."""

    model_config = ConfigDict(extra="forbid")

    query: str
    retrieval_mode: VectorKnowledgeMode
    strategy: RetrievalStrategy
    top_k: int = 0
    chunk_count: int = 0
    chunks: List[VectorChunkRecord] = Field(default_factory=list)
    filters: VectorRetrievalFilters = Field(default_factory=VectorRetrievalFilters)
    error: str = ""


# Registration and ingestion payloads define the write-side contract for new sources.
class DocumentRegistration(BaseModel):
    """Input payload for registering a source document before chunk ingestion."""

    model_config = ConfigDict(extra="forbid")

    source_key: str
    title: str
    document_type: str = ""
    issuer: str = ""
    language: str = "en"
    source_url: Optional[str] = None
    storage_path: Optional[str] = None
    logical_key: str = ""
    effective_date: Optional[str] = None
    effective_end_date: Optional[str] = None
    published_date: Optional[str] = None
    version_label: Optional[str] = None
    is_latest: bool = True
    is_active: bool = True
    abolished: bool = False
    supersedes_document_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_lifecycle(self) -> "DocumentRegistration":
        if not self.logical_key.strip():
            self.logical_key = self.source_key
        if self.abolished and self.is_active:
            raise ValueError("abolished documents cannot be active")
        return self


class ChunkIngestRecord(BaseModel):
    """A chunk prepared for embedding and persistence."""

    model_config = ConfigDict(extra="forbid")

    chunk_index: int
    text_content: str
    token_count: int = 0
    section_title: str = ""
    section_path: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    language: str = "en"
    topics: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text_content")
    @classmethod
    def _validate_chunk_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("chunk text cannot be empty")
        return text


class IngestionResult(BaseModel):
    """Result summary for a document ingestion run."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_count: int
    embedding_dimension: int
    source_key: str
