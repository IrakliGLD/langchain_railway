"""Persistence helpers for Supabase-backed document chunks."""

from __future__ import annotations

import json
from typing import List

from sqlalchemy import bindparam, text

from config import (
    VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION,
    VECTOR_KNOWLEDGE_MIN_SIMILARITY,
    VECTOR_KNOWLEDGE_SCHEMA,
)
from contracts.vector_knowledge import (
    ChunkIngestRecord,
    DocumentRegistration,
    IngestionResult,
    VectorChunkRecord,
    VectorRetrievalFilters,
)
from core.query_executor import ENGINE


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def _validate_embedding_length(values: List[float], *, label: str) -> None:
    actual_dimension = len(values)
    if actual_dimension != VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION:
        raise ValueError(
            f"{label} dimension mismatch: expected {VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION}, got {actual_dimension}"
        )


class KnowledgeVectorStore:
    """SQL helpers for storing vectorized document chunks in Supabase Postgres."""

    def __init__(self, schema: str = VECTOR_KNOWLEDGE_SCHEMA) -> None:
        self.schema = schema

    def upsert_document(self, document: DocumentRegistration) -> str:
        sql = text(
            f"""
            insert into {self.schema}.documents (
                source_key, title, document_type, issuer, language, source_url,
                storage_path, effective_date, published_date, version_label, metadata
            ) values (
                :source_key, :title, :document_type, :issuer, :language, :source_url,
                :storage_path, :effective_date, :published_date, :version_label, cast(:metadata as jsonb)
            )
            on conflict (source_key) do update set
                title = excluded.title,
                document_type = excluded.document_type,
                issuer = excluded.issuer,
                language = excluded.language,
                source_url = excluded.source_url,
                storage_path = excluded.storage_path,
                effective_date = excluded.effective_date,
                published_date = excluded.published_date,
                version_label = excluded.version_label,
                metadata = excluded.metadata,
                updated_at = now()
            returning id::text
            """
        )
        params = document.model_dump(mode="json")
        params["metadata"] = json.dumps(params["metadata"])
        with ENGINE.begin() as conn:
            result = conn.execute(sql, params).scalar_one()
        return str(result)

    def replace_document_chunks(
        self,
        *,
        document_id: str,
        source_key: str,
        chunks: List[ChunkIngestRecord],
        embeddings: List[List[float]],
    ) -> IngestionResult:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        for idx, embedding in enumerate(embeddings):
            _validate_embedding_length(embedding, label=f"chunk_embedding[{idx}]")
        with ENGINE.begin() as conn:
            conn.execute(
                text(f"delete from {self.schema}.document_chunks where document_id = :document_id"),
                {"document_id": document_id},
            )
            insert_sql = text(
                f"""
                insert into {self.schema}.document_chunks (
                    document_id, chunk_index, section_title, section_path, page_start, page_end,
                    text_content, token_count, language, topics, metadata, embedding
                ) values (
                    :document_id, :chunk_index, :section_title, :section_path, :page_start, :page_end,
                    :text_content, :token_count, :language, cast(:topics as jsonb),
                    cast(:metadata as jsonb), cast(:embedding as vector)
                )
                """
            )
            for chunk, embedding in zip(chunks, embeddings):
                params = chunk.model_dump(mode="json")
                params.update(
                    {
                        "document_id": document_id,
                        "topics": json.dumps(params["topics"]),
                        "metadata": json.dumps(params["metadata"]),
                        "embedding": _vector_literal(embedding),
                    }
                )
                conn.execute(insert_sql, params)
        return IngestionResult(
            document_id=document_id,
            chunk_count=len(chunks),
            embedding_dimension=(len(embeddings[0]) if embeddings else 0),
            source_key=source_key,
        )

    def search_chunks(
        self,
        *,
        query_embedding: List[float],
        filters: VectorRetrievalFilters | None = None,
        top_k: int = 4,
        candidate_k: int = 12,
        min_similarity: float = VECTOR_KNOWLEDGE_MIN_SIMILARITY,
    ) -> List[VectorChunkRecord]:
        filters = filters or VectorRetrievalFilters()
        _validate_embedding_length(query_embedding, label="query_embedding")
        embedding_literal = _vector_literal(query_embedding)
        clauses = ["d.is_active = true"]
        params = {"embedding": embedding_literal, "candidate_k": max(candidate_k, top_k)}
        bind_params = []

        preferred_topics = [topic for topic in filters.preferred_topics if str(topic or "").strip()]
        languages = [language for language in filters.languages if str(language or "").strip()]
        document_types = [doc_type for doc_type in filters.document_types if str(doc_type or "").strip()]
        issuers = [issuer for issuer in filters.issuers if str(issuer or "").strip()]

        if preferred_topics:
            clauses.append(
                """
                exists (
                    select 1
                    from jsonb_array_elements_text(c.topics) as topic(value)
                    where topic.value in :preferred_topics
                )
                """
            )
            params["preferred_topics"] = preferred_topics
            bind_params.append(bindparam("preferred_topics", expanding=True))
        if languages:
            clauses.append("c.language in :languages")
            params["languages"] = languages
            bind_params.append(bindparam("languages", expanding=True))
        if document_types:
            clauses.append("d.document_type in :document_types")
            params["document_types"] = document_types
            bind_params.append(bindparam("document_types", expanding=True))
        if issuers:
            clauses.append("d.issuer in :issuers")
            params["issuers"] = issuers
            bind_params.append(bindparam("issuers", expanding=True))

        sql = text(
            f"""
            select
                c.id::text as id,
                c.document_id::text as document_id,
                d.title as document_title,
                d.document_type as document_type,
                d.issuer as document_issuer,
                d.source_key as source_key,
                c.chunk_index,
                c.section_title,
                c.section_path,
                c.page_start,
                c.page_end,
                c.text_content,
                c.token_count,
                c.language,
                c.topics,
                c.metadata,
                (1 - (c.embedding <=> cast(:embedding as vector))) as similarity_score
            from {self.schema}.document_chunks c
            join {self.schema}.documents d on d.id = c.document_id
            where {" and ".join(clauses)}
            order by c.embedding <=> cast(:embedding as vector)
            limit :candidate_k
            """
        )
        if bind_params:
            sql = sql.bindparams(*bind_params)
        with ENGINE.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        results: List[VectorChunkRecord] = []
        for row in rows:
            score = float(row.get("similarity_score") or 0.0)
            if score < float(min_similarity):
                continue
            topics = row.get("topics") or []
            metadata = row.get("metadata") or {}
            results.append(
                VectorChunkRecord(
                    id=str(row["id"]),
                    document_id=str(row["document_id"]),
                    document_title=str(row.get("document_title") or ""),
                    document_type=str(row.get("document_type") or ""),
                    document_issuer=str(row.get("document_issuer") or ""),
                    source_key=str(row.get("source_key") or ""),
                    chunk_index=int(row.get("chunk_index") or 0),
                    section_title=str(row.get("section_title") or ""),
                    section_path=str(row.get("section_path") or ""),
                    page_start=row.get("page_start"),
                    page_end=row.get("page_end"),
                    text_content=str(row.get("text_content") or ""),
                    token_count=int(row.get("token_count") or 0),
                    language=str(row.get("language") or ""),
                    topics=list(topics),
                    metadata=dict(metadata),
                    similarity_score=score,
                )
            )
            if len(results) >= top_k:
                break
        return results
