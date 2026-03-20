"""Hybrid retrieval helpers for curated + vector-backed knowledge."""

from __future__ import annotations

import os
import re
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


_GENERIC_BOOST_STOPWORDS = {
    "a", "an", "and", "are", "electricity", "explain", "formed", "formation",
    "how", "in", "is", "market", "of", "on", "the", "what", "who",
    "when", "where", "why", "რა", "არის", "როგორ", "ვინ", "როდის", "ბაზარი",
    "ელექტროენერგიის", "ელექტროენერგია", "explanation", "process", "required",
}


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _extract_boost_terms(
    query_text: str,
    question_analysis: Optional[QuestionAnalysis],
) -> list[str]:
    combined = " ".join(
        part
        for part in [
            str(query_text or "").strip(),
            (question_analysis.canonical_query_en.strip() if question_analysis else ""),
        ]
        if part
    )
    normalized = _normalized_text(combined)
    boost_terms: list[str] = []

    def _add(term: str) -> None:
        normalized_term = _normalized_text(term)
        if (
            not normalized_term
            or normalized_term in _GENERIC_BOOST_STOPWORDS
            or normalized_term in boost_terms
        ):
            return
        boost_terms.append(normalized_term)

    phrase_rules = [
        (("capacity market", "capacity", "guaranteed capacity", "სიმძლავრის", "სიმძლავრე"), ["capacity market", "capacity", "guaranteed capacity"]),
        (("export", "exports", "ექსპორტ", "экспорт", "cross-border", "interconnection", "interconnector"), ["export", "cross-border", "interconnection"]),
        (("balancing", "balancing electricity", "საბალანსო", "баланс"), ["balancing", "balancing electricity"]),
        (("buyer", "buyers", "purchaser", "consumer", "customers", "მყიდველ", "მომხმარებ"), ["buyer", "buyers", "consumer", "customer"]),
        (("participant", "participants", "registration", "register", "candidate", "მონაწილ", "რეგისტრ"), ["participant", "participants", "registration", "register"]),
        (("day-ahead", "day ahead", "intraday", "დღით ადრე", "დღიური"), ["day-ahead", "intraday"]),
        (("transitory", "transition", "transitional", "გარდამავალი", "transition model"), ["transitory", "transition", "transitional"]),
    ]
    for triggers, terms in phrase_rules:
        if any(trigger in normalized for trigger in triggers):
            for term in terms:
                _add(term)

    for token in re.findall(r"[\w\u10A0-\u10FF-]+", normalized):
        if len(token) < 4:
            continue
        _add(token)

    return boost_terms[:10]


def build_vector_filters(
    question_analysis: Optional[QuestionAnalysis],
    *,
    query_text: str = "",
) -> VectorRetrievalFilters:
    """Convert question-analyzer output into retrieval filters."""

    if question_analysis is None:
        return VectorRetrievalFilters(boost_terms=_extract_boost_terms(query_text, None))
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
        boost_terms=_extract_boost_terms(query_text, question_analysis),
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

    filters = build_vector_filters(question_analysis, query_text=query_text)
    top_k = _int_env("VECTOR_KNOWLEDGE_TOP_K", 4)
    candidate_k = top_k * _int_env("VECTOR_KNOWLEDGE_SEARCH_MULTIPLIER", 3)
    if filters.boost_terms:
        candidate_k = max(candidate_k, top_k * 6)
    min_similarity = _float_env("VECTOR_KNOWLEDGE_MIN_SIMILARITY", 0.2)
    try:
        if store is None:
            from knowledge.vector_store import KnowledgeVectorStore

            store = KnowledgeVectorStore()
        if embedding_provider is None:
            from knowledge.vector_embeddings import get_embedding_provider

            embedding_provider = get_embedding_provider()
        query_embedding = embedding_provider.embed_query(query_text)
        chunks = store.search_chunks(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            candidate_k=candidate_k,
            min_similarity=min_similarity,
        )
        if not chunks and filters.languages:
            relaxed_filters = filters.model_copy(update={"languages": []})
            chunks = store.search_chunks(
                query_embedding=query_embedding,
                filters=relaxed_filters,
                top_k=top_k,
                candidate_k=candidate_k,
                min_similarity=min_similarity,
            )
            filters = relaxed_filters
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
