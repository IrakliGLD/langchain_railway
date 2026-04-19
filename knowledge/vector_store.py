"""Persistence helpers for Supabase-backed document chunks."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import List

from sqlalchemy import bindparam, text
from contracts.vector_knowledge import (
    ChunkIngestRecord,
    DocumentRegistration,
    IngestionResult,
    VectorChunkRecord,
    VectorRetrievalFilters,
)


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION = _env_int("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", 1536)
VECTOR_KNOWLEDGE_MIN_SIMILARITY = _env_float("VECTOR_KNOWLEDGE_MIN_SIMILARITY", 0.2)
VECTOR_KNOWLEDGE_SCHEMA = os.getenv("VECTOR_KNOWLEDGE_SCHEMA", "knowledge").strip() or "knowledge"
VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT = _env_int("VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT", 3)
VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_SECTION = _env_int("VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_SECTION", 1)
VECTOR_KNOWLEDGE_DIVERSITY_SCORE_TOLERANCE = _env_float(
    "VECTOR_KNOWLEDGE_DIVERSITY_SCORE_TOLERANCE",
    0.08,
)



@lru_cache(maxsize=1)
def _get_engine():
    # Reuse the main query_executor ENGINE to avoid dual-pool competition
    # against Supabase PgBouncer's limited connection slots.
    # The main engine now carries prepare_threshold=None for PgBouncer
    # compatibility, making a separate engine unnecessary.
    #
    # Previously this created an independent pool (pool_size=5, max_overflow=2)
    # which, combined with the main engine's pool, could exceed PgBouncer limits
    # and cause cascading ConnectionTimeout errors across all pipeline stages.
    from core.query_executor import ENGINE as _main_engine
    return _main_engine


ENGINE = None


def _resolve_engine():
    return ENGINE or _get_engine()


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def _validate_embedding_length(values: List[float], *, label: str) -> None:
    actual_dimension = len(values)
    if actual_dimension != VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION:
        raise ValueError(
            f"{label} dimension mismatch: expected {VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION}, got {actual_dimension}"
        )


def _normalize_match_text(value: object) -> str:
    normalized = str(value or "").strip().lower()
    normalized = re.sub(r"[_/\-]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized)


def _topic_overlap_boost(
    candidate: VectorChunkRecord,
    preferred_topics: list[str],
) -> float:
    """Score boost for chunks whose topics overlap with preferred_topics."""
    if not preferred_topics:
        return 0.0
    chunk_topics = set(candidate.topics or [])
    matched = chunk_topics & set(preferred_topics)
    if not matched:
        return 0.0
    # Scale: 1 match → 0.15, 2 → 0.20, 3+ → 0.25 (capped)
    return min(0.25, 0.10 + 0.05 * len(matched))


def _market_concept_reference_boost(
    *,
    filters: VectorRetrievalFilters,
    title_text: str,
    source_text: str,
    section_text: str,
    metadata_text: str,
    body_text: str,
) -> float:
    """Boost concept-oriented chunks for market-design queries."""

    market_design_topics = {
        "market_design",
        "market_transition",
        "electricity_market_transitory_model",
        "electricity_market_target_model",
    }
    if not market_design_topics.intersection(set(filters.preferred_topics or [])):
        return 0.0

    concept_reference_terms = (
        "market model concept",
        "market concept",
        "market design concept",
        "ბაზრის მოდელის კონცეფცი",
        "კონცეფციით",
    )

    reference_boost = 0.0
    if any(term in title_text for term in concept_reference_terms):
        reference_boost += 0.18
    if any(term in source_text for term in concept_reference_terms):
        reference_boost += 0.14
    if any(term in section_text for term in concept_reference_terms):
        reference_boost += 0.10
    if any(term in metadata_text for term in concept_reference_terms):
        reference_boost += 0.06
    if any(term in body_text for term in concept_reference_terms):
        reference_boost += 0.12
    return min(reference_boost, 0.32)


def _candidate_retrieval_score(
    candidate: VectorChunkRecord,
    *,
    filters: VectorRetrievalFilters,
) -> float:
    base_score = float(candidate.similarity_score or 0.0)
    topic_boost = _topic_overlap_boost(candidate, filters.preferred_topics)

    if not filters.boost_terms:
        return base_score + topic_boost

    title_text = _normalize_match_text(candidate.document_title)
    source_text = _normalize_match_text(candidate.source_key)
    section_text = _normalize_match_text(f"{candidate.section_title} {candidate.section_path}")
    topic_text = _normalize_match_text(" ".join(candidate.topics))
    metadata_text = _normalize_match_text(json.dumps(candidate.metadata, ensure_ascii=False, sort_keys=True))
    body_text = _normalize_match_text(candidate.text_content[:1200])

    boost = _market_concept_reference_boost(
        filters=filters,
        title_text=title_text,
        source_text=source_text,
        section_text=section_text,
        metadata_text=metadata_text,
        body_text=body_text,
    )
    for term in filters.boost_terms:
        normalized_term = _normalize_match_text(term)
        if not normalized_term:
            continue
        if normalized_term in title_text:
            boost += 0.24
        if normalized_term in source_text:
            boost += 0.18
        if normalized_term in section_text:
            boost += 0.20
        if normalized_term in topic_text:
            boost += 0.16
        if normalized_term in metadata_text:
            boost += 0.08
        if normalized_term in body_text:
            boost += 0.04
    return base_score + topic_boost + min(boost, 0.85)


def _section_key(candidate: VectorChunkRecord) -> str:
    """Stable section identity scoped to a single document."""
    if candidate.section_path.strip():
        return f"{candidate.document_id}:{candidate.section_path.strip()}"
    if candidate.section_title.strip():
        return f"{candidate.document_id}:{candidate.section_title.strip()}"
    return f"{candidate.document_id}:chunk_{candidate.chunk_index}"


def _apply_document_diversity(
    candidates: List[VectorChunkRecord],
    *,
    top_k: int,
    filters: VectorRetrievalFilters,
) -> List[VectorChunkRecord]:
    if len(candidates) < top_k or top_k <= 1:
        return candidates[:top_k]

    max_per_document = max(1, VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT)
    max_per_section = max(1, VECTOR_KNOWLEDGE_MAX_CHUNKS_PER_SECTION)
    candidate_scores = {
        candidate.id: _candidate_retrieval_score(candidate, filters=filters)
        for candidate in candidates
    }
    best_score = max(candidate_scores.values())
    diversity_floor = max(
        VECTOR_KNOWLEDGE_MIN_SIMILARITY,
        best_score - max(0.0, VECTOR_KNOWLEDGE_DIVERSITY_SCORE_TOLERANCE),
    )

    competitive: List[VectorChunkRecord] = [
        candidate
        for candidate in candidates
        if candidate_scores[candidate.id] >= diversity_floor
    ]

    competitive_docs = {candidate.document_id for candidate in competitive}
    selection_pool = competitive
    selection_max_per_document = max_per_document

    if len(competitive_docs) <= 1:
        dominant_doc_id = competitive[0].document_id if competitive else candidates[0].document_id
        selection_pool = [
            candidate for candidate in candidates if candidate.document_id == dominant_doc_id
        ]
        selection_max_per_document = top_k

    selected: List[VectorChunkRecord] = []
    selected_ids: set[str] = set()
    per_document_counts: dict[str, int] = {}
    per_section_counts: dict[str, int] = {}

    for candidate in selection_pool:
        doc_id = candidate.document_id
        section_id = _section_key(candidate)
        if per_document_counts.get(doc_id, 0) >= selection_max_per_document:
            continue
        if per_section_counts.get(section_id, 0) >= max_per_section:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.id)
        per_document_counts[doc_id] = per_document_counts.get(doc_id, 0) + 1
        per_section_counts[section_id] = per_section_counts.get(section_id, 0) + 1
        if len(selected) >= top_k:
            return selected

    backfill_pool = selection_pool if len(competitive_docs) <= 1 else candidates
    seen_sections = {_section_key(candidate) for candidate in selected}

    for candidate in backfill_pool:
        if candidate.id in selected_ids:
            continue
        section_id = _section_key(candidate)
        if section_id in seen_sections:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.id)
        seen_sections.add(section_id)
        if len(selected) >= top_k:
            return selected[:top_k]

    for candidate in backfill_pool:
        if candidate.id in selected_ids:
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


class KnowledgeVectorStore:
    """SQL helpers for storing vectorized document chunks in Supabase Postgres."""

    def __init__(self, schema: str = VECTOR_KNOWLEDGE_SCHEMA) -> None:
        self.schema = schema

    def count_active_documents(self) -> int:
        sql = text(f"select count(*) from {self.schema}.documents where is_active = true")
        with _resolve_engine().begin() as conn:
            return int(conn.execute(sql).scalar_one())

    def upsert_document(self, document: DocumentRegistration) -> str:
        sql = text(
            f"""
            insert into {self.schema}.documents (
                source_key, title, document_type, issuer, language, source_url,
                storage_path, logical_key, effective_date, effective_end_date,
                published_date, version_label, is_latest, is_active, abolished,
                supersedes_document_id, metadata
            ) values (
                :source_key, :title, :document_type, :issuer, :language, :source_url,
                :storage_path, :logical_key, :effective_date, :effective_end_date,
                :published_date, :version_label, :is_latest, :is_active, :abolished,
                :supersedes_document_id, cast(:metadata as jsonb)
            )
            on conflict (source_key) do update set
                title = excluded.title,
                document_type = excluded.document_type,
                issuer = excluded.issuer,
                language = excluded.language,
                source_url = excluded.source_url,
                storage_path = excluded.storage_path,
                logical_key = excluded.logical_key,
                effective_date = excluded.effective_date,
                effective_end_date = excluded.effective_end_date,
                published_date = excluded.published_date,
                version_label = excluded.version_label,
                is_latest = excluded.is_latest,
                is_active = excluded.is_active,
                abolished = excluded.abolished,
                supersedes_document_id = excluded.supersedes_document_id,
                metadata = excluded.metadata,
                updated_at = now()
            returning id::text
            """
        )
        params = document.model_dump(mode="json")
        params["metadata"] = json.dumps(params["metadata"])
        with _resolve_engine().begin() as conn:
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
        with _resolve_engine().begin() as conn:
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

        # preferred_topics are used as a soft scoring boost in
        # _candidate_retrieval_score(), NOT as a hard SQL filter.
        # This allows semantically similar chunks from all documents
        # to enter the candidate pool, with topic-matching chunks
        # ranked higher during Python-side reranking.
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
        with _resolve_engine().begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        # Use a relaxed floor for the candidate pool so that chunks with
        # low raw similarity but high topic/keyword boost can still enter
        # the reranking stage.  The real min_similarity gate is applied
        # *after* _candidate_retrieval_score() below.
        _floor_delta = _env_float("VECTOR_KNOWLEDGE_CANDIDATE_FLOOR_DELTA", 0.10)
        candidate_floor = max(0.0, float(min_similarity) - _floor_delta)
        candidates: List[VectorChunkRecord] = []
        for row in rows:
            score = float(row.get("similarity_score") or 0.0)
            if score < candidate_floor:
                continue
            topics = row.get("topics") or []
            metadata = row.get("metadata") or {}
            candidates.append(
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
        candidates.sort(
            key=lambda candidate: (
                _candidate_retrieval_score(candidate, filters=filters),
                float(candidate.similarity_score or 0.0),
                -int(candidate.chunk_index or 0),
            ),
            reverse=True,
        )
        # Apply the real min_similarity threshold on the *boosted* score so
        # that topic/keyword boosts can rescue weak-but-relevant matches.
        candidates = [
            c for c in candidates
            if _candidate_retrieval_score(c, filters=filters) >= float(min_similarity)
        ]
        return _apply_document_diversity(candidates, top_k=top_k, filters=filters)
