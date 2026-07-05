"""Persistence helpers for Supabase-backed document chunks."""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import List, Optional, Tuple

from sqlalchemy import bindparam, text

from contracts.vector_knowledge import (
    ChunkIngestRecord,
    ChunkReference,
    DocumentRegistration,
    IngestionResult,
    VectorChunkRecord,
    VectorRetrievalFilters,
)


log = logging.getLogger("Enai")

# Schema names are interpolated directly into SQL identifiers (identifiers
# cannot be bound parameters), so constrain them to a bare SQL-identifier shape
# as defence in depth even though the value is env-derived today.
_SCHEMA_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
# Fix #3 (2026-05-16): soft penalty applied to chunks whose topic-tag
# set has zero overlap with the analyzer-derived ``preferred_topics``.
# Q4 production trace 4cda8fdd — query about wholesale guaranteed-source
# generators pulled chunks from a net-metering RETAIL-market document
# because embedding similarity matched superficial words ("generator",
# "surplus", "forecast") and the existing ``_topic_overlap_boost`` only
# adds positive score when there IS overlap — silence on zero overlap.
# A small symmetric penalty breaks ties in favor of on-topic chunks
# without overpowering legitimate cross-topic semantic matches.
# Magnitude is in the same scale as the existing boost (0.10-0.25).
#
# Fix B (2026-05-16): bumped default from 0.05 → 0.10 after the
# 2026-05-16 follow-up retest. Q4 still retrieved retail-net-metering
# chunks even with Fix #3 active because (a) the embedding-similarity
# gap was larger than 0.05, and (b) net-metering chunks have
# "tariffs" in their topics — overlapping with the analyzer's
# ``preferred_topics=["market_structure","cfd_ppa","tariffs"]`` —
# so the zero-overlap penalty never fired. Bumping the magnitude
# helps the (a) case; the (b) case is addressed by the new
# ``_document_domain_mismatch_penalty`` below.
VECTOR_TOPIC_AFFINITY_MISMATCH_PENALTY = _env_float(
    "VECTOR_TOPIC_AFFINITY_MISMATCH_PENALTY",
    0.10,
)
# Fix B (2026-05-16): penalty applied when a chunk's source document
# is clearly from a different market segment than the query's
# preferred_topics suggest — e.g. a Retail Market / Net Metering
# document chunk retrieved for a wholesale Balancing-market query.
# This penalty fires even when there IS topic overlap (because the
# Georgian topic taxonomy has ambiguous tags like ``tariffs`` that
# appear on both retail and wholesale chunks). Document-level domain
# mismatch is a stronger signal than chunk-tag overlap.
# Magnitude (0.20 by default) is sized to overcome the smallest
# boost level (+0.15 for 1-topic overlap) and break the tie.
VECTOR_DOCUMENT_DOMAIN_PENALTY = _env_float(
    "VECTOR_DOCUMENT_DOMAIN_PENALTY",
    0.20,
)

# Document-title / source-key markers that strongly indicate a chunk
# is from the RETAIL electricity-market segment (consumers,
# micro-generators, net metering, retail tariff schemes) rather than
# the WHOLESALE segment (balancing market, market operator, exchange,
# CfDs, guaranteed capacity). Includes Georgian originals for chunks
# whose document_title is in Georgian.
#
# Georgian entries use STEMS (without grammatical case endings) so the
# substring match works across nominative, genitive, dative, etc.
# E.g. ``ნეტო აღრიცხვ`` matches both ``ნეტო აღრიცხვა`` (nom.) and
# ``ნეტო აღრიცხვის`` (gen.).
_RETAIL_DOMAIN_DOCUMENT_MARKERS = (
    "net metering",
    "net billing",
    "retail market",
    "retail electricity",
    "micro generator",
    "micro-generator",
    "consumer protection",
    "household consumer",
    "ნეტო აღრიცხვ",
    "ნეტო ანგარიშსწორებ",
    "მიკროსიმძლავრ",
    "მცირე სიმძლავრ",
    "საცალო ბაზრ",
    "მომხმარებლის უფლებ",
)

# Analyzer-derived preferred_topics that strongly indicate the query
# is about the WHOLESALE market segment (where retail-only documents
# are off-topic by construction).
_WHOLESALE_INTENT_TOPICS = frozenset({
    "balancing_price",
    "balancing_market",
    "balancing_electricity",
    "balancing_electricity_sellers",
    "balancing_electricity_buyers",
    "market_structure",
    "cfd_ppa",
    "electricity_market_transitory_model",
    "electricity_balancing_transitory_model",
    "market_transition",
    "market_design",
    "electricity_market_target_model",
    "capacity_market",
    "wholesale_market_participants",
    "exchange_participation",
    "exchange_rules",
    "day_ahead_market",
    "intraday_market",
    "generation_mix",
    "pso_trading",
    "cross_border_trade",
    "electricity_export",
    "electricity_import",
})



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
    from core.db import ENGINE as _main_engine
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


# Phase B.1 columns. When the schema migration has NOT been applied (or
# the deployed code arrived ahead of the migration), the SELECT statement
# crashes with ``column c.article_number does not exist`` and every
# vector retrieval call returns an empty error bundle. The flag below
# caches a column-existence probe so the store can fall back to the
# pre-Phase-B SELECT shape until the migration lands.
_PHASE_B1_COLUMNS = (
    "article_number",
    "chapter_number",
    "parent_chapter",
    "section_kind",
    "outgoing_refs",
)
_phase_b1_columns_present: Optional[bool] = None


def _check_phase_b1_columns_present() -> bool:
    """Return True iff every Phase B.1 column exists on
    ``knowledge.document_chunks``.

    Only a positive result is cached. A negative or failed probe re-checks on
    the next call, so a schema migration that lands after process start is
    picked up without a restart (the old behaviour cached the first ``False``
    forever, silently dropping B.1 columns for the process lifetime).
    """
    global _phase_b1_columns_present
    if _phase_b1_columns_present:
        return True
    try:
        with _resolve_engine().begin() as conn:
            rows = conn.execute(
                text(
                    """
                    select column_name
                    from information_schema.columns
                    where table_schema = :schema
                      and table_name = 'document_chunks'
                      and column_name = any(:names)
                    """
                ),
                {"schema": VECTOR_KNOWLEDGE_SCHEMA, "names": list(_PHASE_B1_COLUMNS)},
            ).mappings().all()
        found = {row["column_name"] for row in rows}
        present = all(col in found for col in _PHASE_B1_COLUMNS)
    except Exception:
        # Probe failed (transient DB error, unexpected row shape, or schema not
        # reachable yet): do not cache — retry on the next call so a migration
        # that lands after startup is picked up without a restart.
        return False
    if present:
        _phase_b1_columns_present = True
    return present


def _parse_outgoing_refs(raw: object, chunk_id: str = "") -> List[ChunkReference]:
    """Deserialise the ``outgoing_refs`` JSONB column into structured refs.

    Phase B.1: the column may be ``None`` (legacy rows pre-migration),
    an empty list, or a list of dicts (post-Phase-B.2 ingest). Malformed
    entries are dropped rather than failing the whole row, but each drop is now
    logged (with the owning ``chunk_id`` when known) so silent cross-reference
    loss is visible; operator can re-run the backfill to repair them.
    """
    if not raw:
        return []
    ref_id = chunk_id or "<unknown>"
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            log.warning("outgoing_refs JSON parse failed for chunk %s; dropping the column value.", ref_id)
            return []
    if not isinstance(raw, list):
        log.warning("outgoing_refs for chunk %s is %s, expected list; dropping.", ref_id, type(raw).__name__)
        return []
    refs: List[ChunkReference] = []
    for entry in raw:
        if not isinstance(entry, dict):
            log.warning("Skipping non-dict outgoing_refs entry for chunk %s: %r", ref_id, entry)
            continue
        try:
            refs.append(ChunkReference(**entry))
        except (TypeError, ValueError) as exc:
            log.warning("Dropping malformed outgoing_refs entry for chunk %s (%s): %r", ref_id, exc, entry)
            continue
    return refs


def _row_to_chunk_record(
    row,
    *,
    similarity_score: Optional[float],
) -> VectorChunkRecord:
    """Materialise a SELECT row into a :class:`VectorChunkRecord`.

    Centralised so ``search_chunks`` and ``fetch_chunks_by_index`` stay in
    sync as Phase B columns land. Adding a column means one site to edit.
    """
    topics = row.get("topics") or []
    metadata = row.get("metadata") or {}
    return VectorChunkRecord(
        id=str(row["id"]),
        document_id=str(row["document_id"]),
        document_title=str(row.get("document_title") or ""),
        document_type=str(row.get("document_type") or ""),
        document_issuer=str(row.get("document_issuer") or ""),
        source_key=str(row.get("source_key") or ""),
        chunk_index=int(row.get("chunk_index") or 0),
        section_title=str(row.get("section_title") or ""),
        section_path=str(row.get("section_path") or ""),
        article_number=str(row.get("article_number") or ""),
        chapter_number=str(row.get("chapter_number") or ""),
        parent_chapter=str(row.get("parent_chapter") or ""),
        section_kind=str(row.get("section_kind") or ""),
        outgoing_refs=_parse_outgoing_refs(row.get("outgoing_refs"), chunk_id=str(row.get("id") or "")),
        page_start=row.get("page_start"),
        page_end=row.get("page_end"),
        text_content=str(row.get("text_content") or ""),
        token_count=int(row.get("token_count") or 0),
        language=str(row.get("language") or ""),
        topics=list(topics),
        metadata=dict(metadata),
        similarity_score=similarity_score,
    )


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


def _topic_mismatch_penalty(
    candidate: VectorChunkRecord,
    preferred_topics: list[str],
) -> float:
    """Symmetric soft penalty for chunks with zero topic overlap.

    Returns a NEGATIVE float (or 0.0). Applied alongside the positive
    ``_topic_overlap_boost`` in ``_candidate_retrieval_score``.

    Triggers only when:
      - ``preferred_topics`` is non-empty (analyzer expressed a topic intent),
      - chunk ``topics`` is non-empty (the corpus tagged this chunk), and
      - intersection is empty (the chunk's topic tags do not match any
        analyzer intent).

    When either side is empty we cannot tell — return 0.0 (no penalty).
    Magnitude is governed by ``VECTOR_TOPIC_AFFINITY_MISMATCH_PENALTY``
    (default 0.10 after Fix B), comparable to the smallest boost level
    so it breaks ties without overpowering legitimate semantic wins.

    See Q4 production trace 4cda8fdd (2026-05-16).
    """
    if VECTOR_TOPIC_AFFINITY_MISMATCH_PENALTY <= 0.0:
        return 0.0
    if not preferred_topics:
        return 0.0
    chunk_topics = set(candidate.topics or [])
    if not chunk_topics:
        return 0.0
    if chunk_topics & set(preferred_topics):
        return 0.0
    return -float(VECTOR_TOPIC_AFFINITY_MISMATCH_PENALTY)


def _document_domain_mismatch_penalty(
    candidate: VectorChunkRecord,
    preferred_topics: list[str],
) -> float:
    """Penalty when a chunk's source document is from a different market
    segment than the query's ``preferred_topics`` suggest.

    Fix B (2026-05-16) — Q4 production retest after Fix #3 still
    retrieved Electricity Retail Market Rules / Net Metering chunks
    for a wholesale guaranteed-source query because:

      1. The embedding similarity gap exceeded the 0.05 zero-overlap
         penalty magnitude (addressed by bumping it to 0.10), AND
      2. Net-metering chunks happen to have ``"tariffs"`` in their
         topics tag set — overlapping with the analyzer's
         ``preferred_topics=["market_structure","cfd_ppa","tariffs"]``
         — so the zero-overlap penalty never fired and the chunk
         instead received a +0.15 ``_topic_overlap_boost``.

    The ``tariffs`` tag is ambiguous: it covers both retail tariff
    schemes (net metering, micro-generators) and wholesale tariffs
    (regulated HPP, regulated TPP, guaranteed capacity). Topic-level
    overlap is therefore not a sufficient signal — we also need
    document-level domain awareness.

    This helper fires when:
      - ``preferred_topics`` contains at least one wholesale-domain
        topic (the query is clearly about wholesale market),
      - AND the chunk's ``document_title`` or ``source_key`` contains
        a retail-domain marker (net metering / micro-generator /
        retail / consumer / Georgian originals).

    Returns a NEGATIVE float sized (default 0.20) to overcome the
    smallest possible boost (+0.15 for 1-topic overlap) and break the
    tie. Returns 0.0 when either side fails the trigger conditions or
    the env kill-switch sets the magnitude to 0.
    """
    if VECTOR_DOCUMENT_DOMAIN_PENALTY <= 0.0:
        return 0.0
    if not preferred_topics:
        return 0.0
    if not (set(preferred_topics) & _WHOLESALE_INTENT_TOPICS):
        return 0.0
    title_lower = (candidate.document_title or "").lower()
    source_lower = (candidate.source_key or "").lower()
    haystack = f"{title_lower} {source_lower}"
    if not haystack.strip():
        return 0.0
    if any(marker in haystack for marker in _RETAIL_DOMAIN_DOCUMENT_MARKERS):
        return -float(VECTOR_DOCUMENT_DOMAIN_PENALTY)
    return 0.0


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
    topic_penalty = _topic_mismatch_penalty(candidate, filters.preferred_topics)
    domain_penalty = _document_domain_mismatch_penalty(
        candidate, filters.preferred_topics
    )

    if not filters.boost_terms:
        return base_score + topic_boost + topic_penalty + domain_penalty

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
    return base_score + topic_boost + topic_penalty + domain_penalty + min(boost, 0.85)


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
        if not _SCHEMA_IDENTIFIER_RE.match(schema):
            raise ValueError(f"Invalid vector-knowledge schema identifier: {schema!r}")
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
            # Phase B.1 gate: when the new columns don't exist, write the
            # pre-Phase-B shape so re-ingestion against a not-yet-migrated
            # DB still succeeds.  The Phase B.2 structural fields will not
            # be persisted in that case — but the chunk is still searchable
            # via the legacy SELECT path, which is the correct degraded
            # behaviour during a partial rollout.
            if _check_phase_b1_columns_present():
                insert_sql = text(
                    f"""
                    insert into {self.schema}.document_chunks (
                        document_id, chunk_index, section_title, section_path,
                        article_number, chapter_number, parent_chapter, section_kind,
                        outgoing_refs,
                        page_start, page_end,
                        text_content, token_count, language, topics, metadata, embedding
                    ) values (
                        :document_id, :chunk_index, :section_title, :section_path,
                        :article_number, :chapter_number, :parent_chapter, :section_kind,
                        cast(:outgoing_refs as jsonb),
                        :page_start, :page_end,
                        :text_content, :token_count, :language, cast(:topics as jsonb),
                        cast(:metadata as jsonb), cast(:embedding as vector)
                    )
                    """
                )
            else:
                insert_sql = text(
                    f"""
                    insert into {self.schema}.document_chunks (
                        document_id, chunk_index, section_title, section_path,
                        page_start, page_end,
                        text_content, token_count, language, topics, metadata, embedding
                    ) values (
                        :document_id, :chunk_index, :section_title, :section_path,
                        :page_start, :page_end,
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
                        "outgoing_refs": json.dumps(params.get("outgoing_refs") or []),
                        "embedding": _vector_literal(embedding),
                    }
                )
                # Defensive defaults for callers that pre-date Phase B and may
                # not yet emit the structural fields. The ingest contract has
                # defaults so missing keys are unusual; this is belt-and-braces.
                params.setdefault("article_number", "")
                params.setdefault("chapter_number", "")
                params.setdefault("parent_chapter", "")
                params.setdefault("section_kind", "")
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

        phase_b1_cols = (
            ",\n                ".join(f"c.{col}" for col in _PHASE_B1_COLUMNS)
            if _check_phase_b1_columns_present()
            else ""
        )
        phase_b1_select = (f"\n                {phase_b1_cols}," if phase_b1_cols else "")
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
                c.section_path,{phase_b1_select}
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
            candidates.append(_row_to_chunk_record(row, similarity_score=score))
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

    def fetch_chunks_by_index(
        self,
        pairs: List[Tuple[str, int]],
    ) -> List[VectorChunkRecord]:
        """Fetch chunks by (document_id, chunk_index) — used for adjacency
        expansion at retrieval (Phase A.1 of the cross-reference plan).

        Embedding/similarity are NOT computed for adjacency hits; the caller
        decides how to score them.  ``similarity_score`` is therefore left
        at ``None`` on the returned records.
        """
        # Deduplicate, drop invalid pairs, and short-circuit empty input.
        unique_pairs = []
        seen: set[Tuple[str, int]] = set()
        for doc_id, chunk_idx in pairs:
            if doc_id is None or chunk_idx is None or chunk_idx < 0:
                continue
            key = (str(doc_id), int(chunk_idx))
            if key in seen:
                continue
            seen.add(key)
            unique_pairs.append(key)
        if not unique_pairs:
            return []

        # Postgres ``in (values, ...)`` over composite keys via a VALUES join.
        values_clause = ",".join(
            f"(:doc_id_{i}, :chunk_idx_{i})" for i in range(len(unique_pairs))
        )
        params: dict = {}
        for i, (doc_id, chunk_idx) in enumerate(unique_pairs):
            params[f"doc_id_{i}"] = doc_id
            params[f"chunk_idx_{i}"] = chunk_idx

        phase_b1_cols = (
            ",\n                ".join(f"c.{col}" for col in _PHASE_B1_COLUMNS)
            if _check_phase_b1_columns_present()
            else ""
        )
        phase_b1_select = (f"\n                {phase_b1_cols}," if phase_b1_cols else "")
        sql = text(
            f"""
            with wanted (document_id, chunk_index) as (
                values {values_clause}
            )
            select
                c.id::text as id,
                c.document_id::text as document_id,
                d.title as document_title,
                d.document_type as document_type,
                d.issuer as document_issuer,
                d.source_key as source_key,
                c.chunk_index,
                c.section_title,
                c.section_path,{phase_b1_select}
                c.page_start,
                c.page_end,
                c.text_content,
                c.token_count,
                c.language,
                c.topics,
                c.metadata
            from {self.schema}.document_chunks c
            join {self.schema}.documents d on d.id = c.document_id
            join wanted w
              on cast(w.document_id as uuid) = c.document_id
             and w.chunk_index = c.chunk_index
            where d.is_active = true
            """
        )
        with _resolve_engine().begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        return [_row_to_chunk_record(row, similarity_score=None) for row in rows]

    def fetch_chunks_by_article(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[VectorChunkRecord]:
        """Fetch chunks by ``(document_id, article_number)`` — used for
        one-hop reference expansion (Phase B.3 of the cross-reference
        plan).

        Backed by ``idx_knowledge_chunks_article`` (partial index over
        non-empty ``article_number``) so the lookup is O(log n) regardless
        of corpus size.  Embeddings are not computed for resolved chunks;
        ``similarity_score`` stays ``None`` so packing can score them
        against their parent.
        """
        # Phase B.3 cannot run before the Phase B.1 schema migration —
        # the ``article_number`` column it queries on does not exist.
        # Return an empty result so callers (the resolver, the pack
        # function) degrade gracefully to pre-Phase-B behaviour rather
        # than SQL-crashing the entire retrieval call.
        if not _check_phase_b1_columns_present():
            return []
        # Deduplicate and drop invalid pairs.
        unique_pairs: list[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for doc_id, article_number in pairs:
            if not doc_id or not article_number:
                continue
            article = str(article_number).strip()
            if not article:
                continue
            key = (str(doc_id), article)
            if key in seen:
                continue
            seen.add(key)
            unique_pairs.append(key)
        if not unique_pairs:
            return []

        values_clause = ",".join(
            f"(:doc_id_{i}, :article_{i})" for i in range(len(unique_pairs))
        )
        params: dict = {}
        for i, (doc_id, article_number) in enumerate(unique_pairs):
            params[f"doc_id_{i}"] = doc_id
            params[f"article_{i}"] = article_number

        sql = text(
            f"""
            with wanted (document_id, article_number) as (
                values {values_clause}
            )
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
                c.article_number,
                c.chapter_number,
                c.parent_chapter,
                c.section_kind,
                c.outgoing_refs,
                c.page_start,
                c.page_end,
                c.text_content,
                c.token_count,
                c.language,
                c.topics,
                c.metadata
            from {self.schema}.document_chunks c
            join {self.schema}.documents d on d.id = c.document_id
            join wanted w
              on cast(w.document_id as uuid) = c.document_id
             and w.article_number = c.article_number
            where d.is_active = true
              and c.article_number <> ''
            """
        )
        with _resolve_engine().begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        return [_row_to_chunk_record(row, similarity_score=None) for row in rows]
