"""Hybrid retrieval helpers for curated + vector-backed knowledge."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
from typing import TYPE_CHECKING, Optional

log = logging.getLogger("Enai")

from config import ANALYZER_TOPIC_MIN_SCORE
from contracts.question_analysis import QuestionAnalysis
from contracts.vector_knowledge import (
    RetrievalStrategy,
    VectorKnowledgeBundle,
    VectorKnowledgeMode,
    VectorRetrievalFilters,
    VectorRetrievalTier,
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


_ADJACENCY_MODE_VALUES = frozenset({"off", "shadow", "on"})


def get_adjacency_mode() -> str:
    """Return the adjacency-expansion mode: 'off' (default), 'shadow', 'on'.

    Defaults to ``off`` so Phase A.2 is dark in production until the operator
    opts in. Unknown values fall back to ``off`` to avoid silent rollouts on
    typos. Exposed as public so trace emitters (pipeline.py) can include the
    mode in observability events without re-reading the env directly.
    """
    raw = str(os.getenv("VECTOR_ADJACENCY_MODE", "off")).strip().lower()
    return raw if raw in _ADJACENCY_MODE_VALUES else "off"


# Internal alias retained for the few call sites inside this module — keeping
# them consistent with the public name avoids drift if the policy changes.
_adjacency_mode = get_adjacency_mode


_REFERENCE_EXPANSION_MODE_VALUES = frozenset({"off", "shadow", "on"})


def get_reference_expansion_mode() -> str:
    """Return the reference-expansion mode: 'off' (default), 'shadow', 'on'.

    Defaults to ``off`` so Phase B.3 is dark in production until the
    operator opts in.  Unknown values fall back to ``off`` to prevent
    silent rollouts on typos.  Independent of ``VECTOR_ADJACENCY_MODE``
    — the two expansion strategies can be flipped separately.
    """
    raw = str(os.getenv("VECTOR_REFERENCE_EXPANSION_MODE", "off")).strip().lower()
    return raw if raw in _REFERENCE_EXPANSION_MODE_VALUES else "off"


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
    combined: list[str] = []
    for part in [
        str(query_text or "").strip(),
        (question_analysis.raw_query.strip() if question_analysis else ""),
        (question_analysis.canonical_query_en.strip() if question_analysis else ""),
    ]:
        if part and part not in combined:
            combined.append(part)
    return " ".join(combined)


@dataclass(frozen=True)
class PackedVectorPrompt:
    prompt: str
    headers: list[str]
    truncated: bool


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
        # Always add concept-level topic
        _add("electricity_market_transitory_model")
        _add("market_transition")

        # Rules-specific topic only when query signals rules/operational intent
        if _has_any("rule", "rules", "settlement", "dispatch", "scheduling",
                    "penalty", "penalties", "metering", "billing",
                    "market operator", "transmission system operator"):
            _add("electricity_balancing_transitory_model")

        # Reinforce concept/design topics for concept-oriented queries
        if _has_any("who", "what is", "concept", "design", "eligible",
                    "model", "structure", "architecture", "period"):
            _add("market_design")

    if _has_any("capacity market", "guaranteed capacity", "capacity"):
        _add("capacity_market")

    if _has_any("buyer", "buyers", "purchaser", "consumer", "customer"):
        _add("balancing_energy_buyers")
        _add("balancing_electricity_buyers")

    if _has_any("seller", "sellers", "supplier", "suppliers"):
        _add("balancing_electricity_sellers")

    # Compound: "exchange" + "registration" → specific exchange registration topics.
    # Must fire before the individual "registration" and "exchange" checks so that
    # exchange-specific topics appear first in the priority list.
    if _has_any("exchange") and _has_any("registration", "register"):
        _add("participant_registration")
        _add("exchange_registration")
        _add("exchange_participation")
        _add("exchange_rules")

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

    if _has_any("deregulation", "deregulate", "liberalization", "liberalisation"):
        _add("deregulation_plan")
        _add("market_transition")
        _add("market_design")

    if _has_any("market model", "market concept", "market design", "market architecture"):
        _add("market_design")
        _add("electricity_market_transitory_model")
        _add("electricity_market_target_model")
        _add("market_transition")

    if _has_any("target model", "target market"):
        _add("electricity_market_target_model")
        _add("market_design")

    if _has_any("balancing market", "balancing energy"):
        _add("balancing_market")
        _add("balancing_electricity")

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

    # Compound: "exchange" + "registration" → specific boost terms that match
    # the Day-Ahead/Intraday exchange document's topic_text and content.
    if (
        any(_normalized_text(t) in normalized for t in ("exchange",))
        and any(_normalized_text(t) in normalized for t in ("registration", "register"))
    ):
        for term in ["exchange registration", "participant registration", "exchange participation"]:
            _add(term)

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
        (
            ("deregulation", "deregulate", "liberalization", "liberalisation"),
            ["deregulation", "deregulation plan", "market transition"],
        ),
        (
            ("market model", "market concept", "market design", "market architecture"),
            ["market model", "market concept", "market design", "transitory model", "target model"],
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
        if candidate.score >= ANALYZER_TOPIC_MIN_SCORE
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


_LIGHT_TIER_TOP_K = 2


def retrieve_vector_knowledge(
    query_text: str,
    *,
    retrieval_mode: VectorKnowledgeMode,
    question_analysis: Optional[QuestionAnalysis] = None,
    store: "KnowledgeVectorStore" | None = None,
    embedding_provider: "EmbeddingProvider" | None = None,
    tier: VectorRetrievalTier = VectorRetrievalTier.FULL,
) -> VectorKnowledgeBundle:
    """Fetch top vector chunks for a user query.

    ``tier`` (Phase D) lets callers shrink or skip the search for query
    shapes that won't consume the retrieved passages:

    * ``FULL`` — default top-K + over-fetch multiplier (~top_k * 3).
    * ``LIGHT`` — ``top_k=2`` and a tighter candidate pool (boost_terms
      no longer expand the candidate_k to ``top_k * 6``).  Skips the
      sparse-corpus similarity relaxation too — narrative data answers
      only sprinkle in context, so pulling noisy low-similarity chunks
      costs more prompt budget than it buys.
    * ``SKIP`` — returns an empty bundle immediately without touching
      the store or the embedder.
    """

    filters = build_vector_filters(question_analysis, query_text=query_text)

    if tier == VectorRetrievalTier.SKIP:
        return VectorKnowledgeBundle(
            query=query_text,
            retrieval_mode=retrieval_mode,
            strategy=RetrievalStrategy.hybrid,
            top_k=0,
            chunk_count=0,
            chunks=[],
            filters=filters,
        )

    if tier == VectorRetrievalTier.LIGHT:
        top_k = _LIGHT_TIER_TOP_K
        candidate_k = top_k * _int_env("VECTOR_KNOWLEDGE_SEARCH_MULTIPLIER", 3)
        # Intentionally do NOT expand candidate_k on boost_terms at LIGHT —
        # narrative data answers need a small, on-topic slice, not a noisy
        # deep pull.
    else:
        top_k = _int_env("VECTOR_KNOWLEDGE_TOP_K", 6)
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
        if not chunks and filters.preferred_topics:
            log.info(
                "vector_retrieval: topic relaxation – clearing preferred_topics=%s",
                filters.preferred_topics,
            )
            relaxed_filters = filters.model_copy(update={"preferred_topics": []})
            chunks = store.search_chunks(
                query_embedding=query_embedding,
                filters=relaxed_filters,
                top_k=top_k,
                candidate_k=candidate_k,
                min_similarity=min_similarity,
            )
            filters = relaxed_filters
        # Sparse-corpus similarity relaxation: only at FULL tier. LIGHT skips
        # the fallback — a narrative data answer would rather have no extra
        # context than low-similarity noise in its summarizer prompt.
        if not chunks and tier == VectorRetrievalTier.FULL:
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
        bundle = VectorKnowledgeBundle(
            query=query_text,
            retrieval_mode=retrieval_mode,
            strategy=RetrievalStrategy.hybrid,
            top_k=top_k,
            chunk_count=len(chunks),
            chunks=chunks,
            filters=filters,
        )
        # Phase A.2: adjacency expansion. When mode != "off", fetch preceding
        # and following section chunks and stash them on ``bundle.adjacent_chunks``.
        # The pack function ignores this field until A.3 cutover, so behaviour
        # in shadow mode is byte-identical to today's prompt content.
        if _adjacency_mode() != "off" and bundle.chunks:
            bundle.adjacent_chunks = resolve_adjacent_chunks(bundle, store=store)
        # Phase B.3: reference expansion.  Same shadow/on/off pattern as
        # adjacency, independent env flag so the two expansion strategies
        # can be rolled out separately.  Pack function ignores
        # ``reference_chunks`` until B.4 cutover.
        if get_reference_expansion_mode() != "off" and bundle.chunks:
            bundle.reference_chunks = resolve_reference_chunks(bundle, store=store)
        return bundle
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


def resolve_adjacent_chunks(
    bundle: Optional[VectorKnowledgeBundle],
    *,
    store: "KnowledgeVectorStore" | None = None,
) -> list:
    """Return chunks immediately before and after each chunk in ``bundle``.

    Phase A.1 of the cross-reference plan: a pure adjacency lookup that does
    not yet feed packing. ``resolve_adjacent_chunks`` is called separately
    from the main retrieve path so the existing top-K result remains
    byte-identical until the env flag flips in A.2/A.3.

    Behaviour contract:
    - For each chunk in ``bundle``, fetch ``(document_id, chunk_index - 1)``
      and ``(document_id, chunk_index + 1)``.
    - ``chunk_index <= 0`` skips the preceding neighbour.
    - Chunks already in ``bundle`` are NOT returned (no self-shadow).
    - Duplicates across the input set are collapsed.
    - Returns an empty list if ``bundle`` is None / empty / errored.
    - Cross-document leakage is impossible — adjacency is keyed on
      ``(document_id, chunk_index)`` and the store query is exact-match.
    """

    if bundle is None or not bundle.chunks:
        return []
    if bundle.error:
        # When the underlying retrieval errored we already returned an empty
        # bundle; adjacency is meaningless. Stay silent.
        return []

    seen_chunks: set[tuple[str, int]] = set()
    pairs: list[tuple[str, int]] = []
    for chunk in bundle.chunks:
        doc_id = str(chunk.document_id or "").strip()
        if not doc_id:
            continue
        idx = int(chunk.chunk_index or 0)
        seen_chunks.add((doc_id, idx))

    for chunk in bundle.chunks:
        doc_id = str(chunk.document_id or "").strip()
        if not doc_id:
            continue
        idx = int(chunk.chunk_index or 0)
        for neighbour in (idx - 1, idx + 1):
            if neighbour < 0:
                continue
            key = (doc_id, neighbour)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            pairs.append(key)

    if not pairs:
        return []

    if store is None:
        from knowledge.vector_store import KnowledgeVectorStore

        store = KnowledgeVectorStore()
    try:
        return store.fetch_chunks_by_index(pairs)
    except Exception as exc:
        log.warning(
            "Adjacency expansion failed; continuing without it: %s", exc
        )
        return []


# Budgets for reference expansion.  Conservative defaults — sensible
# regulatory chunks rarely have more than a few outbound refs; chunks
# with dozens (long enumeration sections) would otherwise create
# expansion avalanches that bust the pack budget.  Tune in Phase B.5
# once production data is in.
REFERENCE_EXPANSION_PER_CHUNK_BUDGET = 3
REFERENCE_EXPANSION_TOTAL_BUDGET = 10


def resolve_reference_chunks(
    bundle: Optional[VectorKnowledgeBundle],
    *,
    store: "KnowledgeVectorStore" | None = None,
) -> list:
    """Resolve each top-K chunk's ``outgoing_refs`` to actual chunks.

    Phase B.3 of the cross-reference plan: one-hop, same-document
    expansion via the canonical ``(document_id, article_number)``
    lookup.  Self-article refs are skipped (they resolve to the citing
    chunk itself).  Chapter refs are deferred to B.5 — the store
    method for ``(document_id, chapter_number)`` lookup isn't built
    yet, and chapter cross-references account for only 13 of the 506
    refs across the live corpus.

    Behaviour contract:
    - Iterate ``bundle.chunks`` and collect article-kind outgoing
      references, applying a per-chunk budget so a single chunk with
      many refs can't blow the pack alone.
    - Apply a total request budget across all top-K chunks combined.
    - Drop pairs whose resolved chunk is already present in
      ``bundle.chunks`` (no self-shadow).
    - Deduplicate ``(document_id, article_number)`` pairs across the
      input set so multiple primaries that cite the same article only
      trigger one DB hit.
    - Swallow store errors with a warning so the main retrieval path
      stays usable even if the expander breaks.
    """

    if bundle is None or not bundle.chunks or bundle.error:
        return []

    # Build the set of already-known chunks so we never echo them back.
    seen_chunks: set[tuple[str, int]] = set()
    for chunk in bundle.chunks:
        doc_id = str(chunk.document_id or "").strip()
        if not doc_id:
            continue
        seen_chunks.add((doc_id, int(chunk.chunk_index or 0)))

    pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    # Walk chunks in their retrieval order so per-chunk budgets apply
    # to the highest-scoring chunks first.
    for chunk in bundle.chunks:
        if len(pairs) >= REFERENCE_EXPANSION_TOTAL_BUDGET:
            break
        doc_id = str(chunk.document_id or "").strip()
        if not doc_id:
            continue
        per_chunk_count = 0
        for ref in chunk.outgoing_refs or []:
            if per_chunk_count >= REFERENCE_EXPANSION_PER_CHUNK_BUDGET:
                break
            if len(pairs) >= REFERENCE_EXPANSION_TOTAL_BUDGET:
                break
            # Skip self-article — would loop the chunk back to itself.
            # Skip chapter for now — handled separately in Phase B.5
            # once the chapter-resolver SQL lands.
            kind_value = getattr(ref.kind, "value", str(ref.kind or ""))
            if kind_value != "article":
                continue
            number = str(ref.number or "").strip()
            if not number:
                continue
            key = (doc_id, number)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append(key)
            per_chunk_count += 1

    if not pairs:
        return []

    if store is None:
        from knowledge.vector_store import KnowledgeVectorStore

        store = KnowledgeVectorStore()
    try:
        resolved = store.fetch_chunks_by_article(pairs)
    except Exception as exc:
        log.warning(
            "Reference expansion failed; continuing without it: %s", exc
        )
        return []

    # Drop chunks that are already in the primary bundle — adjacency
    # and reference expansion can legitimately collide with the top-K
    # set when the same article was directly matched and also cited.
    filtered: list = []
    for chunk in resolved:
        doc_id = str(chunk.document_id or "").strip()
        key = (doc_id, int(chunk.chunk_index or 0))
        if key in seen_chunks:
            continue
        filtered.append(chunk)
    return filtered


def _format_chunk_header(
    chunk,
    *,
    index: int,
    tag: str = "",
) -> str:
    """Build the ``[N] doc | type: … | section: …`` header used by the
    packer.  ``tag`` (when non-empty) is appended as ``| {tag}`` so
    expansion chunks can flag context-vs-match for the LLM
    (``adjacent`` for Phase A.3, ``referenced`` for Phase B.4).
    """
    header = f"[{index}] {chunk.document_title or chunk.source_key}"
    if chunk.document_type:
        header += f" | type: {chunk.document_type}"
    section_label = chunk.section_title or chunk.section_path
    if section_label:
        header += f" | section: {section_label}"
    if (
        chunk.section_path
        and _normalized_text(chunk.section_path) != _normalized_text(section_label)
    ):
        header += f" | locator: {chunk.section_path}"
    if chunk.page_start is not None:
        header += f" | page: {chunk.page_start}"
    if tag:
        header += f" | {tag}"
    return header


def _sort_adjacent_by_parent_score(
    primary_chunks: list,
    adjacent_chunks: list,
) -> list:
    """Order adjacency chunks by the similarity score of the primary chunk
    they neighbour, descending. When budget is tight the highest-priority
    adjacency packs first.

    A chunk in ``adjacent_chunks`` is considered the neighbour of any
    primary chunk in the same document with ``chunk_index`` differing by 1.
    A chunk that neighbours multiple primaries takes the highest score.
    """
    if not adjacent_chunks:
        return []
    scored = []
    for adj in adjacent_chunks:
        best_score = 0.0
        for primary in primary_chunks:
            if primary.document_id != adj.document_id:
                continue
            if abs(int(primary.chunk_index or 0) - int(adj.chunk_index or 0)) != 1:
                continue
            score = float(primary.similarity_score or 0.0)
            if score > best_score:
                best_score = score
        scored.append((best_score, adj))
    # Stable sort: ties keep original order, which matches DB return order
    # — irrelevant for correctness, helpful for trace reproducibility.
    scored.sort(key=lambda item: item[0], reverse=True)
    return [adj for _, adj in scored]


def _sort_references_by_parent_score(
    primary_chunks: list,
    reference_chunks: list,
) -> list:
    """Order reference chunks by the similarity score of the primary chunk
    that cited them, descending.  Same rationale as adjacency: when the
    pack budget is tight the highest-priority reference packs first.

    A reference chunk is considered the target of any primary chunk in
    the same document whose ``outgoing_refs`` includes an article
    reference with matching ``number``.  A reference cited by multiple
    primaries takes the highest score.
    """
    if not reference_chunks:
        return []
    scored = []
    for ref_chunk in reference_chunks:
        best_score = 0.0
        ref_article = str(ref_chunk.article_number or "").strip()
        for primary in primary_chunks:
            if primary.document_id != ref_chunk.document_id:
                continue
            score = float(primary.similarity_score or 0.0)
            # Match by article number on any of the primary's outgoing
            # article-kind refs.
            for outgoing in primary.outgoing_refs or []:
                kind_value = getattr(outgoing.kind, "value", str(outgoing.kind or ""))
                if kind_value != "article":
                    continue
                if str(outgoing.number or "").strip() != ref_article:
                    continue
                if score > best_score:
                    best_score = score
                break
        scored.append((best_score, ref_chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [ref for _, ref in scored]


def pack_vector_knowledge_for_prompt(
    bundle: Optional[VectorKnowledgeBundle],
    *,
    max_chars: int | None = None,
) -> PackedVectorPrompt:
    """Pack retrieved chunks as prompt-safe context and expose included headers.

    Pack order (Phase B.4):

    1. **Primary** (``bundle.chunks``) — direct semantic matches. Always
       packs first; never displaced by expansion chunks. Truncating a
       primary sets ``truncated=True`` AND aborts the expansion passes
       (no point fitting a low-signal expansion when a higher-signal
       primary already didn't fit).
    2. **References** (``bundle.reference_chunks``) — chunks the
       primary cited explicitly via ``outgoing_refs``. Packed under the
       remaining budget when ``VECTOR_REFERENCE_EXPANSION_MODE == "on"``.
       Higher signal than adjacency because the citing chunk
       *deliberately* referred to them. Tagged ``| referenced``.
    3. **Adjacency** (``bundle.adjacent_chunks``) — siblings by
       ``chunk_index``. Packed last because they're "context bleed"
       rather than an intentional cross-reference. Packed under the
       remaining budget when ``VECTOR_ADJACENCY_MODE == "on"``. Tagged
       ``| adjacent``.

    Either expansion list is independently env-gated; either can be
    "off" while the other is "on". Both gated off → byte-identical to
    the pre-A.3/B.4 packer.
    """

    if bundle is None or not bundle.chunks:
        return PackedVectorPrompt(prompt="", headers=[], truncated=False)
    max_chars = max_chars or _int_env("VECTOR_KNOWLEDGE_MAX_CHARS", 9000)
    parts = ["EXTERNAL_SOURCE_PASSAGES:"]
    total_chars = len(parts[0])
    headers: list[str] = []
    truncated = False
    next_index = 1
    for chunk in bundle.chunks:
        header = _format_chunk_header(chunk, index=next_index)
        body = chunk.text_content.strip()
        entry = f"{header}\n{body}"
        if total_chars + len(entry) + 2 > max_chars:
            truncated = True
            break
        parts.append(entry)
        headers.append(header)
        total_chars += len(entry) + 2
        next_index += 1

    def _try_pack_expansion(chunks_to_pack, *, tag: str) -> None:
        """Pack each expansion chunk under remaining budget. Individual
        entries that don't fit are skipped (``continue``, not ``break``)
        so smaller later entries still get a chance to fit. Mutates the
        outer ``parts``, ``headers``, ``total_chars``, ``next_index``,
        ``truncated`` via nonlocal."""
        nonlocal total_chars, next_index, truncated
        for chunk in chunks_to_pack:
            header = _format_chunk_header(chunk, index=next_index, tag=tag)
            body = chunk.text_content.strip()
            entry = f"{header}\n{body}"
            if total_chars + len(entry) + 2 > max_chars:
                truncated = True
                continue
            parts.append(entry)
            headers.append(header)
            total_chars += len(entry) + 2
            next_index += 1

    # Phase B.4 cutover: pack reference chunks before adjacency. References
    # carry higher signal — the citing chunk explicitly cited them — so
    # they earn the remaining budget first when both expansion lists are on.
    if not truncated and get_reference_expansion_mode() == "on" and bundle.reference_chunks:
        ordered_refs = _sort_references_by_parent_score(
            list(bundle.chunks), list(bundle.reference_chunks)
        )
        _try_pack_expansion(ordered_refs, tag="referenced")

    # Phase A.3 cutover: pack adjacency chunks last under remaining budget.
    if not truncated and get_adjacency_mode() == "on" and bundle.adjacent_chunks:
        ordered_adj = _sort_adjacent_by_parent_score(
            list(bundle.chunks), list(bundle.adjacent_chunks)
        )
        _try_pack_expansion(ordered_adj, tag="adjacent")

    return PackedVectorPrompt(
        prompt="\n\n".join(parts),
        headers=headers,
        truncated=truncated,
    )


def format_vector_knowledge_for_prompt(bundle: Optional[VectorKnowledgeBundle], *, max_chars: int | None = None) -> str:
    """Format retrieved chunks as prompt-safe context."""

    return pack_vector_knowledge_for_prompt(bundle, max_chars=max_chars).prompt
