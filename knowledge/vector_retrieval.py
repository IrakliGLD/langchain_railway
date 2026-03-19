"""Hybrid retrieval helpers for curated + vector-backed knowledge."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from contracts.question_analysis import QuestionAnalysis
from contracts.vector_knowledge import (
    RetrievalStrategy,
    VectorKnowledgeBundle,
    VectorKnowledgeMode,
    VectorRetrievalFilters,
)

if TYPE_CHECKING:
    from knowledge.vector_embeddings import EmbeddingProvider
    from knowledge.vector_store import KnowledgeVectorStore


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def build_vector_filters(question_analysis: Optional[QuestionAnalysis]) -> VectorRetrievalFilters:
    """Convert question-analyzer output into retrieval filters."""

    if question_analysis is None:
        return VectorRetrievalFilters()
    topics = [
        candidate.name.value
        for candidate in sorted(
            question_analysis.knowledge.candidate_topics,
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        if candidate.score >= 0.2
    ][:4]
    languages: list[str] = []

    def _add_language(value: object) -> None:
        code = getattr(value, "value", value)
        normalized = str(code or "").strip().lower()
        if not normalized or normalized == "other" or normalized in languages:
            return
        languages.append(normalized)

    _add_language(question_analysis.language.answer_language)
    _add_language(question_analysis.language.input_language)
    if question_analysis.canonical_query_en.strip():
        _add_language("en")
    return VectorRetrievalFilters(
        preferred_topics=topics,
        languages=languages,
    )


def retrieve_vector_knowledge(
    query_text: str,
    *,
    retrieval_mode: VectorKnowledgeMode,
    question_analysis: Optional[QuestionAnalysis] = None,
    store: "KnowledgeVectorStore" | None = None,
    embedding_provider: "EmbeddingProvider" | None = None,
) -> VectorKnowledgeBundle:
    """Fetch top vector chunks for a user query."""

    filters = build_vector_filters(question_analysis)
    top_k = _int_env("VECTOR_KNOWLEDGE_TOP_K", 4)
    candidate_k = top_k * _int_env("VECTOR_KNOWLEDGE_SEARCH_MULTIPLIER", 3)
    min_similarity = _float_env("VECTOR_KNOWLEDGE_MIN_SIMILARITY", 0.2)
    if store is None:
        from knowledge.vector_store import KnowledgeVectorStore

        store = KnowledgeVectorStore()
    if embedding_provider is None:
        from knowledge.vector_embeddings import get_embedding_provider

        embedding_provider = get_embedding_provider()

    try:
        query_embedding = embedding_provider.embed_query(query_text)
        chunks = store.search_chunks(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            candidate_k=candidate_k,
            min_similarity=min_similarity,
        )
        return VectorKnowledgeBundle(
            query=query_text,
            retrieval_mode=retrieval_mode,
            strategy=RetrievalStrategy.hybrid,
            top_k=top_k,
            chunk_count=len(chunks),
            chunks=chunks,
            filters=filters,
        )
    except Exception as exc:
        return VectorKnowledgeBundle(
            query=query_text,
            retrieval_mode=retrieval_mode,
            strategy=RetrievalStrategy.hybrid,
            top_k=top_k,
            chunk_count=0,
            chunks=[],
            filters=filters,
            error=str(exc),
        )


def format_vector_knowledge_for_prompt(bundle: Optional[VectorKnowledgeBundle], *, max_chars: int | None = None) -> str:
    """Format retrieved chunks as prompt-safe context."""

    if bundle is None or not bundle.chunks:
        return ""
    max_chars = max_chars or _int_env("VECTOR_KNOWLEDGE_MAX_CHARS", 3500)
    parts = ["EXTERNAL_SOURCE_PASSAGES:"]
    total_chars = len(parts[0])
    for idx, chunk in enumerate(bundle.chunks, start=1):
        header = f"[{idx}] {chunk.document_title or chunk.source_key}"
        if chunk.section_title:
            header += f" | section: {chunk.section_title}"
        if chunk.page_start is not None:
            header += f" | page: {chunk.page_start}"
        body = chunk.text_content.strip()
        entry = f"{header}\n{body}"
        if total_chars + len(entry) + 2 > max_chars:
            break
        parts.append(entry)
        total_chars += len(entry) + 2
    return "\n\n".join(parts)
