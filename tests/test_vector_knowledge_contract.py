import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts.vector_knowledge import (
    ChunkIngestRecord,
    DocumentRegistration,
    RetrievalStrategy,
    VectorChunkRecord,
    VectorDocumentRecord,
    VectorKnowledgeBundle,
    VectorKnowledgeMode,
)
from knowledge.vector_embeddings import _validate_embedding_dimensions


def test_vector_chunk_record_rejects_empty_text():
    with pytest.raises(ValidationError):
        VectorChunkRecord(
            id="chunk-1",
            document_id="doc-1",
            text_content="",
        )


def test_vector_knowledge_bundle_defaults_are_strict():
    bundle = VectorKnowledgeBundle(
        query="What is GENEX?",
        retrieval_mode=VectorKnowledgeMode.shadow,
        strategy=RetrievalStrategy.hybrid,
    )
    assert bundle.chunk_count == 0
    assert bundle.chunks == []
    assert bundle.filters.preferred_topics == []


def test_chunk_ingest_record_requires_text():
    with pytest.raises(ValidationError):
        ChunkIngestRecord(chunk_index=0, text_content="   ")


def test_schema_sql_declares_expected_embedding_dimension():
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "knowledge_vector.sql"
    sql = schema_path.read_text(encoding="utf-8")
    assert "embedding vector(1536)" in sql
    assert "create schema if not exists knowledge;" in sql.lower()
    assert "create extension if not exists pgcrypto;" in sql.lower()
    assert "logical_key text not null default ''" in sql.lower()
    assert "effective_end_date date null" in sql.lower()
    assert "is_latest boolean not null default true" in sql.lower()
    assert "abolished boolean not null default false" in sql.lower()
    assert "supersedes_document_id uuid null references knowledge.documents(id) on delete set null" in sql.lower()


def test_structured_summary_prompt_allows_external_source_citations():
    llm_path = Path(__file__).resolve().parents[1] / "core" / "llm.py"
    source = llm_path.read_text(encoding="utf-8")
    assert "external_source_passages" in source


def test_embedding_dimension_validator_rejects_wrong_lengths():
    with pytest.raises(RuntimeError):
        _validate_embedding_dimensions(
            [[0.1, 0.2, 0.3]],
            expected_dimension=1536,
            label="document_embedding",
        )


def test_document_registration_lifecycle_defaults():
    document = DocumentRegistration(
        source_key="doc-v1",
        title="Test Document",
    )
    assert document.logical_key == "doc-v1"
    assert document.is_latest is True
    assert document.is_active is True
    assert document.abolished is False
    assert document.supersedes_document_id is None


def test_vector_document_record_accepts_lifecycle_fields():
    record = VectorDocumentRecord(
        id="doc-1",
        source_key="doc-v2",
        title="Test Document v2",
        logical_key="doc",
        effective_end_date="2025-12-31",
        is_latest=False,
        is_active=False,
        abolished=True,
        supersedes_document_id="doc-0",
    )
    assert record.logical_key == "doc"
    assert record.effective_end_date == "2025-12-31"
    assert record.is_latest is False
    assert record.is_active is False
    assert record.abolished is True
    assert record.supersedes_document_id == "doc-0"


def test_document_registration_rejects_abolished_active_conflict():
    with pytest.raises(ValidationError):
        DocumentRegistration(
            source_key="doc-v3",
            title="Abolished but active",
            abolished=True,
            is_active=True,
        )
