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
    "a",
    "an",
    "and",
    "are",
    "electricity",
    "explain",
    "explanation",
    "formation",
    "formed",
    "how",
    "in",
    "is",
    "market",
    "of",
    "on",
    "process",
    "required",
    "the",
    "what",
    "when",
    "where",
    "who",
    "why",
}


def _normalized_text(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = re.sub(r"[_/\-]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized)


def _combined_query_text(
    query_text: str,
    question_analysis: Optional[QuestionAnalysis],
) -> str:
    return " ".join(
        part
        for part in [
            str(query_text or "").strip(),
            (question_analysis.canonical_query_en.strip() if question_analysis else ""),
        ]
        if part
    )


def _extract_bridge_topics(
    query_text: str,
    question_analysis: Optional[QuestionAnalysis],
) -> list[str]:
    normalized = _normalized_text(_combined_query_text(query_text, question_analysis))
    bridge_topics: list[str] = []

    def _add(topic: str) -> None:
        topic = str(topic or "").strip()
        if topic and topic not in bridge_topics:
            bridge_topics.append(topic)

    def _has_any(*phrases: str) -> bool:
        return any(_normalized_text(phrase) in normalized for phrase in phrases)

    if _has_any("export", "exports", "electricity export", "cross border", "cross-border", "interconnection", "interconnector"):
        _add("electricity_export")
        _add("cross_border_trade")
        _add("electricity_import")

    if _has_any("import", "imports", "electricity import"):
        _add("electricity_import")
        _add("cross_border_trade")

    if _has_any("transitory", "transitional", "transition model", "transition"):
        _add("electricity_market_transitory_model")
        _add("electricity_balancing_transitory_model")
        _add("market_transition")

    if _has_any("capacity market", "guaranteed capacity", "capacity"):
        _add("capacity_market")

    if _has_any("buyer", "buyers", "purchaser", "consumer", "customer"):
        _add("balancing_energy_buyers")
        _add("balancing_electricity_buyers")

    if _has_any("seller", "sellers", "supplier", "suppliers"):
        _add("balancing_electricity_sellers")

    if _has_any(
        "eligible",
        "eligibility",
        "who can trade",
        "who may trade",
        "who is eligible to trade",
        "participant",
        "participants",
        "registration",
        "register",
        "trade on the exchange",
    ):
        _add("eligible_participants")
        _add("exchange_participation")
        _add("wholesale_market_participants")
        # Preserve compatibility with previously ingested typoed tags.
        _add("whoesale_market_participants")

    if _has_any("exchange", "electricity exchange"):
        _add("exchange_rules")
        _add("exchange_participation")

    if _has_any("day ahead", "day-ahead"):
        _add("day_ahead_market")

    if _has_any("intraday", "intra day", "intra-day"):
        _add("intraday_market")

    return bridge_topics[:12]


def _extract_boost_terms(
    query_text: str,
    question_analysis: Optional[QuestionAnalysis],
) -> list[str]:
    normalized = _normalized_text(_combined_query_text(query_text, question_analysis))
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
        (
            ("capacity market", "capacity", "guaranteed capacity"),
            ["capacity market", "capacity", "guaranteed capacity"],
        ),
        (
            ("export", "exports", "import", "imports", "cross-border", "cross border", "interconnection", "interconnector"),
            ["export", "import", "cross border", "interconnection", "cross_border_trade", "electricity_export", "electricity_import"],
        ),
        (
            ("balancing", "balancing electricity"),
            ["balancing", "balancing electricity"],
        ),
        (
            ("buyer", "buyers", "purchaser", "consumer", "customers"),
            ["buyer", "buyers", "consumer", "customer"],
        ),
        (
            ("participant", "participants", "registration", "register", "candidate", "eligible", "eligibility"),
            ["participant", "participants", "registration", "register", "eligible", "eligibility"],
        ),
        (
            ("day-ahead", "day ahead", "intraday"),
            ["day-ahead", "intraday"],
        ),
        (
            ("transitory", "transition", "transitional", "transition model"),
            ["transitory", "transition", "transitional", "market transition", "electricity market transitory model"],
        ),
    ]
    for triggers, terms in phrase_rules:
        if any(_normalized_text(trigger) in normalized for trigger in triggers):
            for term in terms:
                _add(term)

    for token in re.findall(r"[\w\u10A0-\u10FF-]+", normalized):
        if len(token) < 4:
            continue
        _add(token)

    return boost_terms[:12]


def build_vector_filters(
    question_analysis: Optional[QuestionAnalysis],
    *,
    query_text: str = "",
) -> VectorRetrievalFilters:
    """Convert question-analyzer output into retrieval filters."""

    if question_analysis is None:
        return VectorRetrievalFilters(
            preferred_topics=_extract_bridge_topics(query_text, None),
            boost_terms=_extract_boost_terms(query_text, None),
        )

    analyzer_topics = [
        candidate.name.value
        for candidate in sorted(
            question_analysis.knowledge.candidate_topics,
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        if candidate.score >= 0.2
    ][:4]
    bridge_topics = _extract_bridge_topics(query_text, question_analysis)
    topics: list[str] = []
    languages: list[str] = []

    def _add_topic(value: str) -> None:
        topic = str(value or "").strip()
        if topic and topic not in topics:
            topics.append(topic)

    def _add_language(value: object) -> None:
        code = getattr(value, "value", value)
        normalized = str(code or "").strip().lower()
        if not normalized or normalized == "other" or normalized in languages:
            return
        languages.append(normalized)

    for topic in bridge_topics:
        _add_topic(topic)
    for topic in analyzer_topics:
        if topic != "general_definitions":
            _add_topic(topic)
    for topic in analyzer_topics:
        if topic == "general_definitions":
            _add_topic(topic)

    _add_language(question_analysis.language.answer_language)
    _add_language(question_analysis.language.input_language)
    if question_analysis.canonical_query_en.strip():
        _add_language("en")

    return VectorRetrievalFilters(
        preferred_topics=topics,
        languages=languages,
        boost_terms=_extract_boost_terms(query_text, question_analysis),
    )


def _sparse_corpus_relaxed_similarity(
    store: "KnowledgeVectorStore",
    *,
    current_min_similarity: float,
) -> float | None:
    count_documents = getattr(store, "count_active_documents", None)
    if not callable(count_documents):
        return None
    try:
        active_documents = int(count_documents())
    except Exception:
        return None
    sparse_doc_limit = _int_env("VECTOR_KNOWLEDGE_SPARSE_CORPUS_MAX_DOCS", 2)
    if active_documents > sparse_doc_limit:
        return None
    relaxed = _float_env("VECTOR_KNOWLEDGE_SPARSE_MIN_SIMILARITY", 0.12)
    if relaxed >= current_min_similarity:
        return None
    return max(0.0, relaxed)


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
        if not chunks:
            relaxed_similarity = _sparse_corpus_relaxed_similarity(
                store,
                current_min_similarity=min_similarity,
            )
            if relaxed_similarity is not None:
                chunks = store.search_chunks(
                    query_embedding=query_embedding,
                    filters=filters,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    min_similarity=relaxed_similarity,
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
