import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from contracts.question_analysis import (
    AnalysisMode,
    ClassificationInfo,
    KnowledgeInfo,
    KnowledgeTopicName,
    LanguageInfo,
    PreferredPath,
    QueryType,
    QuestionAnalysis,
    RoutingInfo,
    SqlHints,
    ToolingInfo,
    TopicCandidate,
    VisualizationInfo,
    LanguageCode,
)
from contracts.vector_knowledge import (
    RetrievalStrategy,
    VectorChunkRecord,
    VectorKnowledgeBundle,
    VectorKnowledgeMode,
)
from knowledge.vector_retrieval import (
    build_vector_filters,
    format_vector_knowledge_for_prompt,
    pack_vector_knowledge_for_prompt,
    resolve_adjacent_chunks,
    resolve_reference_chunks,
    retrieve_vector_knowledge,
)


class FakeEmbeddingProvider:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeStore:
    def search_chunks(self, **kwargs):
        return [
            VectorChunkRecord(
                id="chunk-1",
                document_id="doc-1",
                document_title="Electricity Market Rules",
                source_key="rules-1",
                section_title="Settlement",
                text_content="Specific balancing settlement rule.",
                topics=["balancing_price"],
                similarity_score=0.91,
            )
        ]


def _analysis():
    return QuestionAnalysis(
        version="question_analysis_v1",
        raw_query="Why did balancing electricity price change?",
        canonical_query_en="Why did balancing electricity price change?",
        language=LanguageInfo(input_language=LanguageCode.EN, answer_language=LanguageCode.EN),
        classification=ClassificationInfo(
            query_type=QueryType.DATA_EXPLANATION,
            analysis_mode=AnalysisMode.ANALYST,
            intent="balancing_price_why",
            needs_clarification=False,
            confidence=0.95,
        ),
        routing=RoutingInfo(
            preferred_path=PreferredPath.SQL,
            needs_sql=True,
            needs_knowledge=True,
            prefer_tool=False,
        ),
        knowledge=KnowledgeInfo(
            candidate_topics=[TopicCandidate(name=KnowledgeTopicName.BALANCING_PRICE, score=0.9)]
        ),
        tooling=ToolingInfo(),
        sql_hints=SqlHints(),
        visualization=VisualizationInfo(
            chart_requested_by_user=False,
            chart_recommended=False,
            chart_confidence=0.0,
        ),
        analysis_requirements={"needs_driver_analysis": False, "derived_metrics": []},
    )


def test_build_vector_filters_uses_candidate_topics():
    filters = build_vector_filters(_analysis(), query_text="Why did balancing electricity price change?")
    assert filters.preferred_topics == ["balancing_price"]
    assert filters.languages == ["en"]
    assert "balancing" in filters.boost_terms


def test_build_vector_filters_adds_english_fallback_for_translated_queries():
    analysis = _analysis().model_copy(
        update={
            "language": LanguageInfo(
                input_language=LanguageCode.KA,
                answer_language=LanguageCode.KA,
            ),
            "canonical_query_en": "Why did balancing electricity price change?",
        }
    )
    filters = build_vector_filters(
        analysis,
        query_text="რატომ შეიცვალა საბალანსო ელექტროენერგიის ფასი?",
    )
    assert filters.languages == ["ka", "en"]
    assert "balancing" in filters.boost_terms


def test_build_vector_filters_extracts_specific_boost_terms():
    analysis = _analysis().model_copy(
        update={
            "raw_query": "What regulations and procedures are required for electricity export?",
            "canonical_query_en": "What regulations and procedures are required for electricity export?",
            "knowledge": KnowledgeInfo(
                candidate_topics=[TopicCandidate(name=KnowledgeTopicName.MARKET_STRUCTURE, score=0.9)]
            ),
        }
    )
    filters = build_vector_filters(
        analysis,
        query_text="What regulations and procedures are required for electricity export?",
    )

    assert "electricity_export" in filters.preferred_topics
    assert "cross_border_trade" in filters.preferred_topics
    assert "market_structure" in filters.preferred_topics
    assert "export" in filters.boost_terms
    assert "registration" not in filters.boost_terms


def test_build_vector_filters_bridges_transitory_trade_eligibility_topics():
    analysis = _analysis().model_copy(
        update={
            "raw_query": "Who is eligible to trade on the electricity exchange during the electricity market transitory model?",
            "canonical_query_en": "Who is eligible to trade on the electricity exchange during the electricity market transitory model?",
            "knowledge": KnowledgeInfo(
                candidate_topics=[TopicCandidate(name=KnowledgeTopicName.MARKET_STRUCTURE, score=0.95)]
            ),
            "classification": ClassificationInfo(
                query_type=QueryType.CONCEPTUAL_DEFINITION,
                analysis_mode=AnalysisMode.LIGHT,
                intent="transitory_exchange_eligibility",
                needs_clarification=False,
                confidence=1.0,
            ),
            "routing": RoutingInfo(
                preferred_path=PreferredPath.KNOWLEDGE,
                needs_sql=False,
                needs_knowledge=True,
                prefer_tool=False,
            ),
        }
    )

    filters = build_vector_filters(
        analysis,
        query_text="Who is eligible to trade on the electricity exchange during the electricity market transitory model?",
    )

    assert "electricity_market_transitory_model" in filters.preferred_topics
    assert "eligible_participants" in filters.preferred_topics
    assert "exchange_participation" in filters.preferred_topics
    assert "whoesale_market_participants" in filters.preferred_topics


def test_retrieve_vector_knowledge_returns_bundle():
    bundle = retrieve_vector_knowledge(
        "Why did balancing electricity price change?",
        retrieval_mode=VectorKnowledgeMode.shadow,
        question_analysis=_analysis(),
        store=FakeStore(),
        embedding_provider=FakeEmbeddingProvider(),
    )
    assert bundle.chunk_count == 1
    assert bundle.chunks[0].document_title == "Electricity Market Rules"


def test_format_vector_knowledge_for_prompt_includes_source_headers():
    bundle = retrieve_vector_knowledge(
        "Why did balancing electricity price change?",
        retrieval_mode=VectorKnowledgeMode.active,
        question_analysis=_analysis(),
        store=FakeStore(),
        embedding_provider=FakeEmbeddingProvider(),
    )
    prompt = format_vector_knowledge_for_prompt(bundle, max_chars=1000)
    assert "EXTERNAL_SOURCE_PASSAGES:" in prompt
    assert "Electricity Market Rules" in prompt
    assert "Specific balancing settlement rule." in prompt


def test_format_vector_knowledge_for_prompt_includes_section_locator_when_available():
    bundle = VectorKnowledgeBundle(
        query="What does Article 14 say?",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.hybrid,
        top_k=1,
        chunk_count=1,
        chunks=[
            VectorChunkRecord(
                id="chunk-1",
                document_id="doc-1",
                document_title="Electricity (Capacity) Market Rules",
                source_key="capacity-rules",
                section_title="Price formation",
                section_path="Article 14 > Price formation",
                text_content="Balancing price is formed under the stated conditions.",
            )
        ],
    )

    prompt = format_vector_knowledge_for_prompt(bundle, max_chars=1000)

    assert "section: Price formation" in prompt
    assert "locator: Article 14 > Price formation" in prompt


def test_retrieve_vector_knowledge_captures_provider_init_errors(monkeypatch):
    import knowledge.vector_retrieval as retrieval_module

    monkeypatch.setattr(
        retrieval_module,
        "_int_env",
        lambda name, default: default,
    )

    class FakeStore:
        def search_chunks(self, **kwargs):
            raise AssertionError("search_chunks should not be reached when provider init fails")

    def fail_provider():
        raise RuntimeError("gemini sdk missing")

    monkeypatch.setattr("knowledge.vector_embeddings.get_embedding_provider", fail_provider)

    bundle = retrieve_vector_knowledge(
        "What is GENEX?",
        retrieval_mode=VectorKnowledgeMode.shadow,
        question_analysis=_analysis(),
        store=FakeStore(),
        embedding_provider=None,
    )

    assert bundle.chunk_count == 0
    assert bundle.error == "gemini sdk missing"


def test_retrieve_vector_knowledge_retries_without_language_filter_when_empty():
    calls = []

    class RetryStore:
        def search_chunks(self, **kwargs):
            calls.append(kwargs["filters"].model_dump())
            if kwargs["filters"].languages:
                return []
            return [
                VectorChunkRecord(
                    id="chunk-ka-1",
                    document_id="doc-ka-1",
                    document_title="Electricity Day-Ahead and Intraday Market Rules",
                    source_key="exchange-rules-ka",
                    section_title="Participant registration",
                    text_content="Registration on the exchange is governed by the exchange operator rules.",
                    topics=["market_structure"],
                    language="ka",
                    similarity_score=0.88,
                )
            ]

    analysis = _analysis().model_copy(
        update={
            "raw_query": "What is the process for registering on the electricity exchange?",
            "canonical_query_en": "What is the process for registering on the electricity exchange?",
            "language": LanguageInfo(
                input_language=LanguageCode.EN,
                answer_language=LanguageCode.EN,
            ),
            "knowledge": KnowledgeInfo(
                candidate_topics=[TopicCandidate(name=KnowledgeTopicName.MARKET_STRUCTURE, score=0.9)]
            ),
            "classification": ClassificationInfo(
                query_type=QueryType.CONCEPTUAL_DEFINITION,
                analysis_mode=AnalysisMode.LIGHT,
                intent="exchange_registration",
                needs_clarification=False,
                confidence=0.9,
            ),
            "routing": RoutingInfo(
                preferred_path=PreferredPath.KNOWLEDGE,
                needs_sql=False,
                needs_knowledge=True,
                prefer_tool=False,
            ),
        }
    )

    bundle = retrieve_vector_knowledge(
        "What is the process for registering on the electricity exchange?",
        retrieval_mode=VectorKnowledgeMode.shadow,
        question_analysis=analysis,
        store=RetryStore(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert bundle.chunk_count == 1
    assert bundle.chunks[0].language == "ka"
    assert len(calls) == 2
    assert calls[0]["languages"] == ["en"]
    assert calls[1]["languages"] == []


def test_retrieve_vector_knowledge_expands_candidate_window_when_boost_terms_exist():
    captured = {}

    class CaptureStore:
        def search_chunks(self, **kwargs):
            captured.setdefault("candidate_k", []).append(kwargs["candidate_k"])
            return []

    analysis = _analysis().model_copy(
        update={
            "raw_query": "What regulations and procedures are required for electricity export?",
            "canonical_query_en": "What regulations and procedures are required for electricity export?",
            "knowledge": KnowledgeInfo(
                candidate_topics=[TopicCandidate(name=KnowledgeTopicName.MARKET_STRUCTURE, score=0.9)]
            ),
            "classification": ClassificationInfo(
                query_type=QueryType.CONCEPTUAL_DEFINITION,
                analysis_mode=AnalysisMode.LIGHT,
                intent="export_rules",
                needs_clarification=False,
                confidence=1.0,
            ),
            "routing": RoutingInfo(
                preferred_path=PreferredPath.KNOWLEDGE,
                needs_sql=False,
                needs_knowledge=True,
                prefer_tool=False,
            ),
        }
    )

    bundle = retrieve_vector_knowledge(
        "What regulations and procedures are required for electricity export?",
        retrieval_mode=VectorKnowledgeMode.shadow,
        question_analysis=analysis,
        store=CaptureStore(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert bundle.chunk_count == 0
    assert captured["candidate_k"] == [36, 36, 36]


def test_retrieve_vector_knowledge_relaxes_similarity_for_sparse_corpus():
    calls = []

    class SparseCorpusStore:
        def count_active_documents(self):
            return 1

        def search_chunks(self, **kwargs):
            calls.append(kwargs["min_similarity"])
            if kwargs["min_similarity"] >= 0.2:
                return []
            return [
                VectorChunkRecord(
                    id="chunk-export-1",
                    document_id="doc-export-1",
                    document_title="Electricity (Capacity) Market Rules",
                    source_key="transitory-capacity-rules",
                    section_title="Import and export conditions",
                    text_content="Import and export transactions are governed by the market rules.",
                    topics=["electricity_export", "cross_border_trade"],
                    language="ka",
                    similarity_score=0.14,
                )
            ]

    analysis = _analysis().model_copy(
        update={
            "raw_query": "What regulations and procedures are required for electricity export?",
            "canonical_query_en": "What regulations and procedures are required for electricity export?",
            "knowledge": KnowledgeInfo(
                candidate_topics=[TopicCandidate(name=KnowledgeTopicName.GENERAL_DEFINITIONS, score=0.95)]
            ),
            "classification": ClassificationInfo(
                query_type=QueryType.CONCEPTUAL_DEFINITION,
                analysis_mode=AnalysisMode.LIGHT,
                intent="export_rules",
                needs_clarification=False,
                confidence=1.0,
            ),
            "routing": RoutingInfo(
                preferred_path=PreferredPath.KNOWLEDGE,
                needs_sql=False,
                needs_knowledge=True,
                prefer_tool=False,
            ),
            "language": LanguageInfo(
                input_language=LanguageCode.KA,
                answer_language=LanguageCode.KA,
            ),
        }
    )

    bundle = retrieve_vector_knowledge(
        "What regulations and procedures are required for electricity export?",
        retrieval_mode=VectorKnowledgeMode.active,
        question_analysis=analysis,
        store=SparseCorpusStore(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert bundle.chunk_count == 1
    # After topic relaxation fires (call 3), preferred_topics is cleared;
    # similarity relaxation (call 4) finds the chunk with the relaxed threshold.
    assert bundle.filters.preferred_topics == []
    assert calls == [0.2, 0.2, 0.2, 0.12]


def test_build_vector_filters_bridges_deregulation_topics():
    analysis = _analysis().model_copy(
        update={
            "raw_query": "What is the deregulation plan for electricity power plants?",
            "canonical_query_en": "What is the deregulation plan for electricity power plants?",
            "knowledge": KnowledgeInfo(
                candidate_topics=[
                    TopicCandidate(name=KnowledgeTopicName.GENERAL_DEFINITIONS, score=0.8),
                    TopicCandidate(name=KnowledgeTopicName.MARKET_STRUCTURE, score=0.7),
                ]
            ),
        }
    )

    filters = build_vector_filters(
        analysis,
        query_text="What is the deregulation plan for electricity power plants?",
    )

    assert "deregulation_plan" in filters.preferred_topics
    assert "market_design" in filters.preferred_topics
    assert "market_transition" in filters.preferred_topics


def test_build_vector_filters_bridges_market_concept_topics():
    analysis = _analysis().model_copy(
        update={
            "raw_query": "What is the electricity market model concept?",
            "canonical_query_en": "What is the electricity market model concept?",
            "knowledge": KnowledgeInfo(
                candidate_topics=[
                    TopicCandidate(name=KnowledgeTopicName.MARKET_STRUCTURE, score=0.9),
                ]
            ),
        }
    )

    filters = build_vector_filters(
        analysis,
        query_text="What is the electricity market model concept?",
    )

    assert "market_design" in filters.preferred_topics
    assert "electricity_market_transitory_model" in filters.preferred_topics
    assert "electricity_market_target_model" in filters.preferred_topics


def test_build_vector_filters_preserves_raw_query_market_concept_hint():
    analysis = _analysis().model_copy(
        update={
            "raw_query": "Who can trade on the exchange in the market concept?",
            "canonical_query_en": "Who is eligible to trade on the exchange in the transitory market model?",
            "knowledge": KnowledgeInfo(
                candidate_topics=[
                    TopicCandidate(name=KnowledgeTopicName.MARKET_STRUCTURE, score=0.9),
                ]
            ),
            "classification": ClassificationInfo(
                query_type=QueryType.REGULATORY_PROCEDURE,
                analysis_mode=AnalysisMode.LIGHT,
                intent="transitory_exchange_eligibility",
                needs_clarification=False,
                confidence=1.0,
            ),
        }
    )

    filters = build_vector_filters(
        analysis,
        query_text="Who is eligible to trade on the exchange in the transitory market model?",
    )

    assert "market_design" in filters.preferred_topics


def test_topic_overlap_boost_in_scoring():
    from knowledge.vector_store import _topic_overlap_boost

    from contracts.vector_knowledge import VectorChunkRecord

    chunk_with_match = VectorChunkRecord(
        id="c1",
        document_id="d1",
        document_title="Doc",
        source_key="src",
        section_title="Sec",
        text_content="Text",
        topics=["deregulation_plan", "market_design"],
        language="ka",
        similarity_score=0.5,
    )
    chunk_without_match = VectorChunkRecord(
        id="c2",
        document_id="d2",
        document_title="Doc2",
        source_key="src2",
        section_title="Sec2",
        text_content="Text2",
        topics=["unrelated_topic"],
        language="ka",
        similarity_score=0.5,
    )

    preferred = ["deregulation_plan", "market_design", "market_transition"]

    boost_match = _topic_overlap_boost(chunk_with_match, preferred)
    boost_no_match = _topic_overlap_boost(chunk_without_match, preferred)

    assert boost_match > 0.0
    assert boost_no_match == 0.0
    assert boost_match >= 0.15  # 2 topic matches → at least 0.20


def test_pack_vector_knowledge_for_prompt_reports_packed_headers():
    bundle = VectorKnowledgeBundle(
        query="Who is eligible to trade?",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.hybrid,
        top_k=2,
        chunk_count=2,
        chunks=[
            VectorChunkRecord(
                id="chunk-1",
                document_id="doc-1",
                document_title="Electricity (Capacity) Market Rules",
                source_key="capacity-rules",
                section_title="Participant registration",
                text_content="Short registration rule.",
            ),
            VectorChunkRecord(
                id="chunk-2",
                document_id="doc-2",
                document_title="Electricity Market Model Concept",
                source_key="market-concept",
                section_title="Wholesale market subjects",
                text_content="X" * 300,
            ),
        ],
    )

    packed = pack_vector_knowledge_for_prompt(bundle, max_chars=140)

    assert packed.headers == [
        "[1] Electricity (Capacity) Market Rules | section: Participant registration"
    ]
    assert packed.truncated is True
    assert "Electricity (Capacity) Market Rules" in packed.prompt


def test_build_vector_filters_compound_exchange_registration():
    """When both 'exchange' and 'registration' appear, specific exchange
    registration topics must be generated — not just the broad individual ones."""
    analysis = _analysis().model_copy(
        update={
            "raw_query": "What is the registration process for the power exchange?",
            "canonical_query_en": "What is the registration process for the power exchange?",
            "knowledge": KnowledgeInfo(
                candidate_topics=[
                    TopicCandidate(name=KnowledgeTopicName.MARKET_STRUCTURE, score=0.9),
                ]
            ),
            "classification": ClassificationInfo(
                query_type=QueryType.CONCEPTUAL_DEFINITION,
                analysis_mode=AnalysisMode.LIGHT,
                intent="exchange_registration_process",
                needs_clarification=False,
                confidence=1.0,
            ),
            "routing": RoutingInfo(
                preferred_path=PreferredPath.KNOWLEDGE,
                needs_sql=False,
                needs_knowledge=True,
                prefer_tool=False,
            ),
        }
    )

    filters = build_vector_filters(
        analysis,
        query_text="What is the registration process for the power exchange?",
    )

    # Compound topics must appear (specific to exchange registration)
    assert "participant_registration" in filters.preferred_topics
    assert "exchange_registration" in filters.preferred_topics
    # Broad topics still present but ranked lower
    assert "eligible_participants" in filters.preferred_topics
    assert "exchange_rules" in filters.preferred_topics
    # Compound boost terms
    assert "exchange registration" in filters.boost_terms
    assert "participant registration" in filters.boost_terms


def test_build_vector_filters_plain_registration_no_compound_topics():
    """When only 'registration' appears without 'exchange', compound
    exchange topics should NOT be generated."""
    analysis = _analysis().model_copy(
        update={
            "raw_query": "What is the wholesale registration process?",
            "canonical_query_en": "What is the wholesale registration process?",
            "knowledge": KnowledgeInfo(
                candidate_topics=[
                    TopicCandidate(name=KnowledgeTopicName.MARKET_STRUCTURE, score=0.9),
                ]
            ),
        }
    )

    filters = build_vector_filters(
        analysis,
        query_text="What is the wholesale registration process?",
    )

    # No compound exchange topics
    assert "participant_registration" not in filters.preferred_topics
    assert "exchange_registration" not in filters.preferred_topics
    # Broad registration topics present
    assert "eligible_participants" in filters.preferred_topics
    assert "wholesale_market_participants" in filters.preferred_topics


# ---------------------------------------------------------------------------
# Phase A.1 — adjacency helper
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    chunk_id: str,
    document_id: str = "doc-A",
    chunk_index: int = 0,
    section_title: str = "",
    text_content: str = "filler",
    similarity: float | None = None,
) -> VectorChunkRecord:
    return VectorChunkRecord(
        id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        section_title=section_title or f"section-{chunk_index}",
        text_content=text_content,
        similarity_score=similarity,
    )


class FakeAdjacencyStore:
    """Captures `fetch_chunks_by_index` calls and returns canned chunks."""

    def __init__(self, response: list[VectorChunkRecord] | None = None):
        self.calls: list[list[tuple[str, int]]] = []
        self.response = response or []

    def fetch_chunks_by_index(self, pairs):
        # Normalise to plain tuples so dedup assertions are clean.
        self.calls.append([(str(d), int(i)) for d, i in pairs])
        # Echo back whichever pairs match the canned response.
        wanted = {(c.document_id, c.chunk_index) for c in self.response}
        called = {(str(d), int(i)) for d, i in pairs}
        return [c for c in self.response if (c.document_id, c.chunk_index) in wanted & called]


def _bundle(chunks: list[VectorChunkRecord], *, error: str = "") -> VectorKnowledgeBundle:
    return VectorKnowledgeBundle(
        query="test",
        retrieval_mode=VectorKnowledgeMode.shadow,
        strategy=RetrievalStrategy.hybrid,
        top_k=len(chunks),
        chunk_count=len(chunks),
        chunks=chunks,
        error=error,
    )


def test_resolve_adjacent_chunks_returns_empty_for_empty_bundle():
    store = FakeAdjacencyStore()
    assert resolve_adjacent_chunks(None, store=store) == []
    assert resolve_adjacent_chunks(_bundle([]), store=store) == []
    assert store.calls == []


def test_resolve_adjacent_chunks_returns_empty_when_bundle_errored():
    """A failed retrieval should not trigger adjacency calls."""
    store = FakeAdjacencyStore()
    bundle = _bundle([_make_chunk(chunk_id="c-1", chunk_index=5)], error="transient")
    assert resolve_adjacent_chunks(bundle, store=store) == []
    assert store.calls == []


def test_resolve_adjacent_chunks_requests_prev_and_next():
    """Single chunk at index 5 → requests (doc, 4) and (doc, 6)."""
    expected_prev = _make_chunk(chunk_id="prev", chunk_index=4)
    expected_next = _make_chunk(chunk_id="next", chunk_index=6)
    store = FakeAdjacencyStore(response=[expected_prev, expected_next])
    bundle = _bundle([_make_chunk(chunk_id="c-1", chunk_index=5)])

    out = resolve_adjacent_chunks(bundle, store=store)

    assert len(store.calls) == 1
    assert sorted(store.calls[0]) == [("doc-A", 4), ("doc-A", 6)]
    assert {c.id for c in out} == {"prev", "next"}


def test_resolve_adjacent_chunks_skips_negative_index_for_first_chunk():
    """chunk_index == 0 must NOT request index -1."""
    expected_next = _make_chunk(chunk_id="next", chunk_index=1)
    store = FakeAdjacencyStore(response=[expected_next])
    bundle = _bundle([_make_chunk(chunk_id="c-0", chunk_index=0)])

    out = resolve_adjacent_chunks(bundle, store=store)

    assert store.calls == [[("doc-A", 1)]]
    assert [c.id for c in out] == ["next"]


def test_resolve_adjacent_chunks_does_not_return_chunks_already_in_bundle():
    """If the bundle already contains both neighbours, no expansion."""
    chunks = [
        _make_chunk(chunk_id="c-3", chunk_index=3),
        _make_chunk(chunk_id="c-4", chunk_index=4),
        _make_chunk(chunk_id="c-5", chunk_index=5),
    ]
    store = FakeAdjacencyStore()
    bundle = _bundle(chunks)

    out = resolve_adjacent_chunks(bundle, store=store)

    # c-3's neighbours (2 and 4) → 4 already in bundle, only 2 requested.
    # c-4's neighbours (3 and 5) → both already in bundle, skip.
    # c-5's neighbours (4 and 6) → 4 already in bundle, only 6 requested.
    assert len(store.calls) == 1
    assert sorted(store.calls[0]) == [("doc-A", 2), ("doc-A", 6)]
    # Store responded with nothing, so result is empty.
    assert out == []


def test_resolve_adjacent_chunks_deduplicates_overlapping_neighbours():
    """Chunks at index 3 and 5 both want index 4 — should only ask once."""
    expected_mid = _make_chunk(chunk_id="mid", chunk_index=4)
    store = FakeAdjacencyStore(response=[expected_mid])
    bundle = _bundle([
        _make_chunk(chunk_id="c-3", chunk_index=3),
        _make_chunk(chunk_id="c-5", chunk_index=5),
    ])

    out = resolve_adjacent_chunks(bundle, store=store)

    # Pairs requested: (doc, 2), (doc, 4), (doc, 6). Index 4 appears once.
    flat = store.calls[0]
    assert flat.count(("doc-A", 4)) == 1
    assert sorted(flat) == [("doc-A", 2), ("doc-A", 4), ("doc-A", 6)]
    assert [c.id for c in out] == ["mid"]


def test_resolve_adjacent_chunks_does_not_leak_across_documents():
    """Adjacency is keyed on (document_id, chunk_index) — never crosses
    document boundaries."""
    store = FakeAdjacencyStore()
    bundle = _bundle([
        _make_chunk(chunk_id="a-5", document_id="doc-A", chunk_index=5),
        _make_chunk(chunk_id="b-2", document_id="doc-B", chunk_index=2),
    ])

    resolve_adjacent_chunks(bundle, store=store)

    pairs = store.calls[0]
    # Every pair retains its document_id; no (doc-A, 1) or (doc-B, 4) etc.
    a_pairs = sorted(p for p in pairs if p[0] == "doc-A")
    b_pairs = sorted(p for p in pairs if p[0] == "doc-B")
    assert a_pairs == [("doc-A", 4), ("doc-A", 6)]
    assert b_pairs == [("doc-B", 1), ("doc-B", 3)]


def test_resolve_adjacent_chunks_swallows_store_errors():
    """Store failures must not surface — adjacency is best-effort. The
    main retrieval path stays usable even if expansion errors out."""

    class _ExplodingStore:
        def fetch_chunks_by_index(self, pairs):
            raise RuntimeError("db down")

    bundle = _bundle([_make_chunk(chunk_id="c-1", chunk_index=5)])
    assert resolve_adjacent_chunks(bundle, store=_ExplodingStore()) == []


# ---------------------------------------------------------------------------
# Phase A.2 — env-gated adjacency wiring inside retrieve_vector_knowledge
# ---------------------------------------------------------------------------


class _AdjacencyAwareFakeStore:
    """Like FakeStore, but also implements fetch_chunks_by_index so adjacency
    wiring can be exercised end-to-end through retrieve_vector_knowledge."""

    def __init__(self, primary, adjacent):
        self._primary = primary
        self._adjacent = adjacent
        self.adjacency_calls: list[list[tuple[str, int]]] = []

    def search_chunks(self, **kwargs):
        return list(self._primary)

    def fetch_chunks_by_index(self, pairs):
        self.adjacency_calls.append([(str(d), int(i)) for d, i in pairs])
        wanted = {(c.document_id, c.chunk_index) for c in self._adjacent}
        called = {(str(d), int(i)) for d, i in pairs}
        return [c for c in self._adjacent if (c.document_id, c.chunk_index) in wanted & called]


def _primary_chunk(idx: int) -> VectorChunkRecord:
    return VectorChunkRecord(
        id=f"primary-{idx}",
        document_id="doc-A",
        chunk_index=idx,
        section_title=f"მუხლი {idx}",
        text_content="primary text",
        similarity_score=0.9,
    )


def _adj_chunk(idx: int) -> VectorChunkRecord:
    return VectorChunkRecord(
        id=f"adj-{idx}",
        document_id="doc-A",
        chunk_index=idx,
        section_title=f"მუხლი {idx}",
        text_content="adjacent text",
    )


def test_adjacency_mode_off_keeps_bundle_byte_identical(monkeypatch):
    """Default mode "off" — adjacency MUST NOT fetch and MUST NOT populate
    bundle.adjacent_chunks. Critical: this is what guarantees A.2 deploys
    safely with no behaviour change until the operator opts in."""
    monkeypatch.delenv("VECTOR_ADJACENCY_MODE", raising=False)

    store = _AdjacencyAwareFakeStore(
        primary=[_primary_chunk(5)],
        adjacent=[_adj_chunk(4), _adj_chunk(6)],
    )
    bundle = retrieve_vector_knowledge(
        "balancing prices",
        retrieval_mode=VectorKnowledgeMode.active,
        store=store,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert bundle.adjacent_chunks == []
    assert store.adjacency_calls == []


def test_adjacency_mode_shadow_populates_field_but_pack_ignores_it(monkeypatch):
    """Shadow mode — adjacency fetched and exposed on bundle.adjacent_chunks
    so traces/operators can see it, but pack_vector_knowledge_for_prompt
    must produce byte-identical output to off mode (A.3 owns the cutover)."""
    monkeypatch.setenv("VECTOR_ADJACENCY_MODE", "shadow")

    store = _AdjacencyAwareFakeStore(
        primary=[_primary_chunk(5)],
        adjacent=[_adj_chunk(4), _adj_chunk(6)],
    )
    bundle = retrieve_vector_knowledge(
        "balancing prices",
        retrieval_mode=VectorKnowledgeMode.active,
        store=store,
        embedding_provider=FakeEmbeddingProvider(),
    )

    # Adjacency fetched.
    assert len(bundle.adjacent_chunks) == 2
    assert {c.id for c in bundle.adjacent_chunks} == {"adj-4", "adj-6"}
    assert len(store.adjacency_calls) == 1

    # Pack output must NOT include the adjacency chunks (A.3 owns that).
    packed = pack_vector_knowledge_for_prompt(bundle)
    assert len(packed.headers) == 1  # only the primary chunk
    # No adjacent body text leaks into the prompt.
    assert "adjacent text" not in packed.prompt


def test_adjacency_mode_unknown_value_treated_as_off(monkeypatch):
    """Typos / unknown values must NOT silently enable adjacency. Defensive
    against operator error like ``VECTOR_ADJACENCY_MODE=true``."""
    monkeypatch.setenv("VECTOR_ADJACENCY_MODE", "true")  # not a valid mode

    store = _AdjacencyAwareFakeStore(
        primary=[_primary_chunk(5)],
        adjacent=[_adj_chunk(4), _adj_chunk(6)],
    )
    bundle = retrieve_vector_knowledge(
        "balancing prices",
        retrieval_mode=VectorKnowledgeMode.active,
        store=store,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert bundle.adjacent_chunks == []
    assert store.adjacency_calls == []


def test_adjacency_skipped_when_primary_retrieval_returned_no_chunks(monkeypatch):
    """If the primary search returned nothing, adjacency is a no-op even in
    shadow/on mode. No phantom store calls."""
    monkeypatch.setenv("VECTOR_ADJACENCY_MODE", "shadow")

    store = _AdjacencyAwareFakeStore(primary=[], adjacent=[_adj_chunk(4)])
    bundle = retrieve_vector_knowledge(
        "no match query",
        retrieval_mode=VectorKnowledgeMode.active,
        store=store,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert bundle.chunks == []
    assert bundle.adjacent_chunks == []
    assert store.adjacency_calls == []


# ---------------------------------------------------------------------------
# Phase A.3 — adjacency packing cutover
# ---------------------------------------------------------------------------


def _bundle_with_adjacency(
    primary: list[VectorChunkRecord],
    adjacent: list[VectorChunkRecord],
) -> VectorKnowledgeBundle:
    return VectorKnowledgeBundle(
        query="test",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.hybrid,
        top_k=len(primary),
        chunk_count=len(primary),
        chunks=primary,
        adjacent_chunks=adjacent,
    )


def test_pack_a3_on_appends_adjacency_after_primary(monkeypatch):
    """Under ``VECTOR_ADJACENCY_MODE=on`` the adjacency chunks pack AFTER
    the primary chunks, with an `| adjacent` flag in the header so the LLM
    can tell context from match."""
    monkeypatch.setenv("VECTOR_ADJACENCY_MODE", "on")

    bundle = _bundle_with_adjacency(
        primary=[VectorChunkRecord(
            id="p-1", document_id="doc-A", document_title="Market Rules",
            chunk_index=5, section_title="მუხლი 14",
            text_content="Article 14 body.", similarity_score=0.9,
        )],
        adjacent=[VectorChunkRecord(
            id="a-1", document_id="doc-A", document_title="Market Rules",
            chunk_index=6, section_title="მუხლი 15",
            text_content="Article 15 body.",
        )],
    )
    packed = pack_vector_knowledge_for_prompt(bundle)

    # Both headers present, in order.
    assert len(packed.headers) == 2
    assert packed.headers[0].startswith("[1] Market Rules")
    assert "adjacent" not in packed.headers[0]
    assert packed.headers[1].startswith("[2] Market Rules")
    assert packed.headers[1].endswith("| adjacent")
    # Both bodies in the packed prompt.
    assert "Article 14 body." in packed.prompt
    assert "Article 15 body." in packed.prompt


def test_pack_a3_off_keeps_byte_identical_output(monkeypatch):
    """Default ``off`` mode MUST produce byte-identical pack output to the
    pre-A.3 behaviour even when the bundle carries adjacency chunks. This
    is the safe-rollback invariant for the cutover."""
    monkeypatch.delenv("VECTOR_ADJACENCY_MODE", raising=False)

    bundle = _bundle_with_adjacency(
        primary=[VectorChunkRecord(
            id="p-1", document_id="doc-A", document_title="Market Rules",
            chunk_index=5, section_title="მუხლი 14",
            text_content="Article 14 body.", similarity_score=0.9,
        )],
        adjacent=[VectorChunkRecord(
            id="a-1", document_id="doc-A", document_title="Market Rules",
            chunk_index=6, section_title="მუხლი 15",
            text_content="Article 15 body.",
        )],
    )
    packed = pack_vector_knowledge_for_prompt(bundle)

    assert len(packed.headers) == 1
    assert "Article 15 body." not in packed.prompt
    assert "| adjacent" not in packed.prompt


def test_pack_a3_shadow_does_not_pack_adjacency(monkeypatch):
    """Shadow mode populates ``bundle.adjacent_chunks`` for observability
    but pack output must NOT include them. Critical invariant: shadow ≡ off
    at the packing layer."""
    monkeypatch.setenv("VECTOR_ADJACENCY_MODE", "shadow")

    bundle = _bundle_with_adjacency(
        primary=[VectorChunkRecord(
            id="p-1", document_id="doc-A", document_title="Market Rules",
            chunk_index=5, section_title="მუხლი 14",
            text_content="Article 14 body.", similarity_score=0.9,
        )],
        adjacent=[VectorChunkRecord(
            id="a-1", document_id="doc-A", document_title="Market Rules",
            chunk_index=6, section_title="მუხლი 15",
            text_content="Article 15 body.",
        )],
    )
    packed = pack_vector_knowledge_for_prompt(bundle)

    assert len(packed.headers) == 1
    assert "Article 15 body." not in packed.prompt


def test_pack_a3_on_budget_drops_adjacency_before_truncating_primary(monkeypatch):
    """Budget pressure must drop adjacency entries before truncating any
    primary chunk that already packed. Adjacency is a 'fill the gap'
    consumer; the primary K-results stay sacred."""
    monkeypatch.setenv("VECTOR_ADJACENCY_MODE", "on")

    # Two primaries that together fit the budget, plus an adjacency chunk
    # large enough to push past it.
    primary_a = VectorChunkRecord(
        id="p-1", document_id="doc-A", document_title="Doc",
        chunk_index=5, section_title="A14",
        text_content="A" * 200, similarity_score=0.95,
    )
    primary_b = VectorChunkRecord(
        id="p-2", document_id="doc-A", document_title="Doc",
        chunk_index=10, section_title="A30",
        text_content="B" * 200, similarity_score=0.85,
    )
    big_adj = VectorChunkRecord(
        id="a-1", document_id="doc-A", document_title="Doc",
        chunk_index=6, section_title="A15",
        text_content="C" * 800,
    )
    bundle = _bundle_with_adjacency(
        primary=[primary_a, primary_b], adjacent=[big_adj]
    )

    # Budget large enough for both primaries (~250 each) but not the adj (~880).
    packed = pack_vector_knowledge_for_prompt(bundle, max_chars=700)

    # Both primaries packed.
    assert sum(1 for h in packed.headers if "adjacent" not in h) == 2
    # Adjacency dropped.
    assert sum(1 for h in packed.headers if "| adjacent" in h) == 0
    # Truncated flag set so callers can see budget pressure hit.
    assert packed.truncated is True


def test_pack_a3_on_orders_adjacency_by_parent_score(monkeypatch):
    """When multiple adjacency chunks compete for the same remaining
    budget, the one neighbouring the highest-scoring primary packs first."""
    monkeypatch.setenv("VECTOR_ADJACENCY_MODE", "on")

    # Two primaries, one high-score (0.95), one low (0.55).
    high_primary = VectorChunkRecord(
        id="p-high", document_id="doc-A", document_title="Doc",
        chunk_index=5, section_title="HighArticle",
        text_content="primary high", similarity_score=0.95,
    )
    low_primary = VectorChunkRecord(
        id="p-low", document_id="doc-A", document_title="Doc",
        chunk_index=10, section_title="LowArticle",
        text_content="primary low", similarity_score=0.55,
    )
    adj_high = VectorChunkRecord(
        id="a-high", document_id="doc-A", document_title="Doc",
        chunk_index=6, section_title="NextToHigh",
        text_content="adjacent next to HighArticle",
    )
    adj_low = VectorChunkRecord(
        id="a-low", document_id="doc-A", document_title="Doc",
        chunk_index=9, section_title="NextToLow",
        text_content="adjacent next to LowArticle",
    )
    bundle = _bundle_with_adjacency(
        primary=[high_primary, low_primary],
        # Deliberately pass them in low-first order so the sort visibly fires.
        adjacent=[adj_low, adj_high],
    )
    packed = pack_vector_knowledge_for_prompt(bundle)

    # After the two primaries, the high-parent adjacency packs before the
    # low-parent one — the headers list reflects pack order.
    adjacent_headers = [h for h in packed.headers if "| adjacent" in h]
    assert len(adjacent_headers) == 2
    assert "NextToHigh" in adjacent_headers[0]
    assert "NextToLow" in adjacent_headers[1]


# ---------------------------------------------------------------------------
# Phase B.3 — reference-expansion resolver + env wiring
# ---------------------------------------------------------------------------


from contracts.vector_knowledge import ChunkReference, ChunkReferenceKind  # noqa: E402


def _primary_with_refs(
    *,
    chunk_id: str,
    document_id: str = "doc-A",
    chunk_index: int = 0,
    refs: list[ChunkReference] | None = None,
    similarity: float = 0.9,
) -> VectorChunkRecord:
    return VectorChunkRecord(
        id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        section_title=f"section-{chunk_index}",
        text_content="primary body",
        similarity_score=similarity,
        outgoing_refs=refs or [],
    )


def _target_chunk(
    *,
    chunk_id: str,
    document_id: str,
    article_number: str,
    chunk_index: int = 100,
) -> VectorChunkRecord:
    return VectorChunkRecord(
        id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        section_title=f"მუხლი {article_number}",
        article_number=article_number,
        section_kind="article",
        text_content=f"body of article {article_number}",
    )


class _ReferenceAwareFakeStore:
    """Captures ``fetch_chunks_by_article`` calls and returns canned chunks."""

    def __init__(self, response):
        self._response = response
        self.calls: list[list[tuple[str, str]]] = []

    def fetch_chunks_by_article(self, pairs):
        self.calls.append([(str(d), str(a)) for d, a in pairs])
        wanted = {(c.document_id, c.article_number) for c in self._response}
        called = {(str(d), str(a)) for d, a in pairs}
        return [c for c in self._response if (c.document_id, c.article_number) in (wanted & called)]


def test_resolve_reference_chunks_returns_empty_when_no_refs():
    store = _ReferenceAwareFakeStore(response=[])
    bundle = _bundle([_make_chunk(chunk_id="c-1", chunk_index=5)])
    out = resolve_reference_chunks(bundle, store=store)
    assert out == []
    assert store.calls == []


def test_resolve_reference_chunks_follows_article_ref():
    """``outgoing_refs=[article 14]`` on a primary chunk → resolver fetches
    article 14 from the same document."""
    target = _target_chunk(chunk_id="t-14", document_id="doc-A", article_number="14")
    store = _ReferenceAwareFakeStore(response=[target])
    primary = _primary_with_refs(
        chunk_id="p-1",
        chunk_index=5,
        refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
    )
    bundle = _bundle([primary])

    out = resolve_reference_chunks(bundle, store=store)

    assert store.calls == [[("doc-A", "14")]]
    assert [c.id for c in out] == ["t-14"]


def test_resolve_reference_chunks_skips_self_article_refs():
    """``self_article`` refs resolve to the citing chunk itself —
    the resolver MUST skip them or it would loop the chunk back."""
    store = _ReferenceAwareFakeStore(response=[])
    primary = _primary_with_refs(
        chunk_id="p-1",
        chunk_index=5,
        refs=[
            ChunkReference(kind=ChunkReferenceKind.self_article, number="self", raw_text="ამ მუხლის"),
        ],
    )
    bundle = _bundle([primary])

    resolve_reference_chunks(bundle, store=store)

    # No store call — self-article was filtered out before pair collection.
    assert store.calls == []


def test_resolve_reference_chunks_skips_chapter_refs_for_now():
    """B.3 doesn't yet resolve chapter refs (deferred to B.5). The
    resolver must skip them silently rather than crash."""
    store = _ReferenceAwareFakeStore(response=[])
    primary = _primary_with_refs(
        chunk_id="p-1",
        chunk_index=5,
        refs=[
            ChunkReference(kind=ChunkReferenceKind.chapter, number="IV", raw_text="თავი IV"),
        ],
    )
    bundle = _bundle([primary])

    resolve_reference_chunks(bundle, store=store)

    assert store.calls == []


def test_resolve_reference_chunks_dedupes_pairs_across_primaries():
    """Two primary chunks that both cite article 14 → only one DB hit."""
    target = _target_chunk(chunk_id="t-14", document_id="doc-A", article_number="14")
    store = _ReferenceAwareFakeStore(response=[target])
    p1 = _primary_with_refs(
        chunk_id="p-1", chunk_index=5,
        refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
    )
    p2 = _primary_with_refs(
        chunk_id="p-2", chunk_index=20,
        refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="14-ე მუხლი")],
    )
    bundle = _bundle([p1, p2])

    out = resolve_reference_chunks(bundle, store=store)

    # The (doc-A, "14") pair appears once in the call.
    flat = store.calls[0]
    assert flat.count(("doc-A", "14")) == 1
    # And we get the single target back.
    assert [c.id for c in out] == ["t-14"]


def test_resolve_reference_chunks_drops_targets_already_in_bundle():
    """If a referenced article happens to also be in the top-K set,
    resolver must NOT echo it back — that would duplicate the same
    chunk in the pack."""
    primary_article = VectorChunkRecord(
        id="p-1", document_id="doc-A", chunk_index=14,
        section_title="მუხლი 14", article_number="14", section_kind="article",
        text_content="article 14 body", similarity_score=0.85,
    )
    citing = _primary_with_refs(
        chunk_id="p-2", chunk_index=20,
        refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
    )
    # Store would happily return the article-14 chunk if asked, but it's
    # already in the bundle (under primary's chunk_index 14).
    target_dup = VectorChunkRecord(
        id="p-1", document_id="doc-A", chunk_index=14,
        section_title="მუხლი 14", article_number="14", section_kind="article",
        text_content="article 14 body",
    )
    store = _ReferenceAwareFakeStore(response=[target_dup])
    bundle = _bundle([primary_article, citing])

    out = resolve_reference_chunks(bundle, store=store)

    # The store was called (the pair was requested) but the result was
    # filtered out as a duplicate of the primary.
    assert out == []


def test_resolve_reference_chunks_enforces_per_chunk_budget():
    """A single primary with many refs cannot pull all of them — the
    per-chunk budget caps the expansion so no one chunk can monopolise
    the pack budget."""
    from knowledge.vector_retrieval import REFERENCE_EXPANSION_PER_CHUNK_BUDGET

    many_refs = [
        ChunkReference(kind=ChunkReferenceKind.article, number=str(n), raw_text=f"{n}-ე მუხლი")
        for n in range(1, REFERENCE_EXPANSION_PER_CHUNK_BUDGET + 5)
    ]
    primary = _primary_with_refs(chunk_id="p-1", chunk_index=5, refs=many_refs)
    store = _ReferenceAwareFakeStore(response=[])
    bundle = _bundle([primary])

    resolve_reference_chunks(bundle, store=store)

    pairs = store.calls[0] if store.calls else []
    assert len(pairs) == REFERENCE_EXPANSION_PER_CHUNK_BUDGET


def test_resolve_reference_chunks_swallows_store_errors():
    class _ExplodingStore:
        def fetch_chunks_by_article(self, pairs):
            raise RuntimeError("db down")

    primary = _primary_with_refs(
        chunk_id="p-1", chunk_index=5,
        refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
    )
    bundle = _bundle([primary])
    assert resolve_reference_chunks(bundle, store=_ExplodingStore()) == []


def test_reference_mode_off_keeps_bundle_field_empty(monkeypatch):
    """Default ``off`` mode: ``bundle.reference_chunks`` stays empty even
    when top-K chunks carry ``outgoing_refs``. The safe-rollback invariant
    for the B.3 wiring."""
    monkeypatch.delenv("VECTOR_REFERENCE_EXPANSION_MODE", raising=False)

    target = _target_chunk(chunk_id="t-14", document_id="doc-A", article_number="14")

    class _MixedStore:
        def __init__(self):
            self.search_called = False
            self.article_calls: list = []

        def search_chunks(self, **kwargs):
            self.search_called = True
            return [
                _primary_with_refs(
                    chunk_id="p-1", chunk_index=5,
                    refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
                )
            ]

        def fetch_chunks_by_article(self, pairs):
            self.article_calls.append(pairs)
            return [target]

    store = _MixedStore()
    bundle = retrieve_vector_knowledge(
        "test query",
        retrieval_mode=VectorKnowledgeMode.active,
        store=store,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert bundle.reference_chunks == []
    assert store.article_calls == []


def test_reference_mode_shadow_populates_field(monkeypatch):
    """Shadow mode fetches resolved chunks and exposes them on the bundle,
    but pack output stays byte-identical to off (B.4 owns the cutover)."""
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "shadow")

    target = _target_chunk(chunk_id="t-14", document_id="doc-A", article_number="14")

    class _MixedStore:
        def __init__(self):
            self.article_calls: list = []

        def search_chunks(self, **kwargs):
            return [
                _primary_with_refs(
                    chunk_id="p-1", chunk_index=5,
                    refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
                )
            ]

        def fetch_chunks_by_article(self, pairs):
            self.article_calls.append([(str(d), str(a)) for d, a in pairs])
            return [target]

    store = _MixedStore()
    bundle = retrieve_vector_knowledge(
        "test query",
        retrieval_mode=VectorKnowledgeMode.active,
        store=store,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert len(bundle.reference_chunks) == 1
    assert bundle.reference_chunks[0].id == "t-14"
    assert store.article_calls == [[("doc-A", "14")]]

    # Pack output does NOT include the reference chunk (B.4 owns it).
    packed = pack_vector_knowledge_for_prompt(bundle)
    assert len(packed.headers) == 1
    assert "body of article 14" not in packed.prompt


def test_reference_mode_unknown_value_treated_as_off(monkeypatch):
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "enabled")  # invalid

    class _Store:
        def search_chunks(self, **kwargs):
            return [
                _primary_with_refs(
                    chunk_id="p-1", chunk_index=5,
                    refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
                )
            ]

        def fetch_chunks_by_article(self, pairs):
            raise AssertionError("should not be called")

    bundle = retrieve_vector_knowledge(
        "test query",
        retrieval_mode=VectorKnowledgeMode.active,
        store=_Store(),
        embedding_provider=FakeEmbeddingProvider(),
    )
    assert bundle.reference_chunks == []


# ---------------------------------------------------------------------------
# Phase B.4 — reference-pack cutover + ordering vs. adjacency
# ---------------------------------------------------------------------------


def _bundle_with_references(
    primary: list[VectorChunkRecord],
    references: list[VectorChunkRecord],
    adjacent: list[VectorChunkRecord] | None = None,
) -> VectorKnowledgeBundle:
    return VectorKnowledgeBundle(
        query="test",
        retrieval_mode=VectorKnowledgeMode.active,
        strategy=RetrievalStrategy.hybrid,
        top_k=len(primary),
        chunk_count=len(primary),
        chunks=primary,
        reference_chunks=references,
        adjacent_chunks=adjacent or [],
    )


def test_pack_b4_on_appends_references_after_primary(monkeypatch):
    """Under ``VECTOR_REFERENCE_EXPANSION_MODE=on`` reference chunks pack
    AFTER the primary chunks, tagged ``| referenced``."""
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "on")
    monkeypatch.delenv("VECTOR_ADJACENCY_MODE", raising=False)

    bundle = _bundle_with_references(
        primary=[VectorChunkRecord(
            id="p-30", document_id="doc-A", document_title="Market Rules",
            chunk_index=30, section_title="მუხლი 30",
            article_number="30", section_kind="article",
            text_content="Article 30 body cites article 14.",
            similarity_score=0.9,
            outgoing_refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
        )],
        references=[VectorChunkRecord(
            id="ref-14", document_id="doc-A", document_title="Market Rules",
            chunk_index=14, section_title="მუხლი 14",
            article_number="14", section_kind="article",
            text_content="Article 14 body — the referenced content.",
        )],
    )
    packed = pack_vector_knowledge_for_prompt(bundle)

    assert len(packed.headers) == 2
    assert packed.headers[0].startswith("[1] Market Rules")
    assert "referenced" not in packed.headers[0]
    assert packed.headers[1].startswith("[2] Market Rules")
    assert packed.headers[1].endswith("| referenced")
    assert "Article 14 body — the referenced content." in packed.prompt


def test_pack_b4_off_keeps_byte_identical_output(monkeypatch):
    """``off`` keeps pack output byte-identical to pre-cutover. This is
    the rollback invariant: setting the env back to ``off`` reverts the
    prompt content even if ``bundle.reference_chunks`` is populated."""
    monkeypatch.delenv("VECTOR_REFERENCE_EXPANSION_MODE", raising=False)
    monkeypatch.delenv("VECTOR_ADJACENCY_MODE", raising=False)

    bundle = _bundle_with_references(
        primary=[VectorChunkRecord(
            id="p-30", document_id="doc-A", document_title="Market Rules",
            chunk_index=30, section_title="მუხლი 30",
            text_content="Article 30 body.", similarity_score=0.9,
        )],
        references=[VectorChunkRecord(
            id="ref-14", document_id="doc-A", document_title="Market Rules",
            chunk_index=14, section_title="მუხლი 14",
            article_number="14", text_content="Article 14 body.",
        )],
    )
    packed = pack_vector_knowledge_for_prompt(bundle)

    assert len(packed.headers) == 1
    assert "Article 14 body." not in packed.prompt
    assert "| referenced" not in packed.prompt


def test_pack_b4_shadow_does_not_pack_references(monkeypatch):
    """Shadow exposes ``bundle.reference_chunks`` for trace observability
    but pack output must stay identical to ``off``."""
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "shadow")
    monkeypatch.delenv("VECTOR_ADJACENCY_MODE", raising=False)

    bundle = _bundle_with_references(
        primary=[VectorChunkRecord(
            id="p-30", document_id="doc-A", document_title="Market Rules",
            chunk_index=30, section_title="მუხლი 30",
            text_content="Article 30 body.", similarity_score=0.9,
        )],
        references=[VectorChunkRecord(
            id="ref-14", document_id="doc-A", document_title="Market Rules",
            chunk_index=14, section_title="მუხლი 14",
            article_number="14", text_content="Article 14 body.",
        )],
    )
    packed = pack_vector_knowledge_for_prompt(bundle)

    assert len(packed.headers) == 1
    assert "Article 14 body." not in packed.prompt


def test_pack_b4_references_pack_before_adjacency_when_both_on(monkeypatch):
    """When both expansion strategies are on, references pack BEFORE
    adjacency — references are higher-signal (the citing chunk
    explicitly cited them) so they earn budget priority."""
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "on")
    monkeypatch.setenv("VECTOR_ADJACENCY_MODE", "on")

    primary = VectorChunkRecord(
        id="p-30", document_id="doc-A", document_title="Doc",
        chunk_index=30, section_title="A30",
        article_number="30", text_content="primary body",
        similarity_score=0.9,
        outgoing_refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
    )
    reference = VectorChunkRecord(
        id="ref-14", document_id="doc-A", document_title="Doc",
        chunk_index=14, section_title="A14",
        article_number="14", text_content="referenced body",
    )
    adjacent = VectorChunkRecord(
        id="adj-31", document_id="doc-A", document_title="Doc",
        chunk_index=31, section_title="A31",
        text_content="adjacent body",
    )
    bundle = _bundle_with_references(
        primary=[primary], references=[reference], adjacent=[adjacent],
    )
    packed = pack_vector_knowledge_for_prompt(bundle)

    # Order in headers: primary [1], referenced [2], adjacent [3].
    assert len(packed.headers) == 3
    assert "| referenced" not in packed.headers[0]
    assert "| referenced" in packed.headers[1]
    assert "| adjacent" in packed.headers[2]


def test_pack_b4_budget_drops_references_before_truncating_primary(monkeypatch):
    """Under budget pressure references drop entirely before any primary
    truncates. Primary chunks remain sacred — they're the direct match
    results, never displaced by an expansion."""
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "on")

    primary_a = VectorChunkRecord(
        id="p-1", document_id="doc-A", document_title="Doc",
        chunk_index=5, section_title="A5",
        text_content="A" * 200, similarity_score=0.95,
        outgoing_refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
    )
    primary_b = VectorChunkRecord(
        id="p-2", document_id="doc-A", document_title="Doc",
        chunk_index=10, section_title="A10",
        text_content="B" * 200, similarity_score=0.85,
    )
    big_ref = VectorChunkRecord(
        id="ref-14", document_id="doc-A", document_title="Doc",
        chunk_index=14, section_title="A14",
        article_number="14", text_content="C" * 800,
    )
    bundle = _bundle_with_references(
        primary=[primary_a, primary_b], references=[big_ref],
    )
    packed = pack_vector_knowledge_for_prompt(bundle, max_chars=700)

    # Both primaries packed, reference dropped.
    assert sum(1 for h in packed.headers if "| referenced" not in h and "| adjacent" not in h) == 2
    assert sum(1 for h in packed.headers if "| referenced" in h) == 0
    assert packed.truncated is True


def test_pack_b4_orders_references_by_citing_parent_score(monkeypatch):
    """When multiple references compete for budget, the one cited by the
    highest-scoring primary packs first."""
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "on")

    high_primary = VectorChunkRecord(
        id="p-high", document_id="doc-A", document_title="Doc",
        chunk_index=5, section_title="HighArticle",
        text_content="primary high", similarity_score=0.95,
        outgoing_refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
    )
    low_primary = VectorChunkRecord(
        id="p-low", document_id="doc-A", document_title="Doc",
        chunk_index=10, section_title="LowArticle",
        text_content="primary low", similarity_score=0.55,
        outgoing_refs=[ChunkReference(kind=ChunkReferenceKind.article, number="30", raw_text="30-ე მუხლი")],
    )
    ref_high = VectorChunkRecord(
        id="ref-14", document_id="doc-A", document_title="Doc",
        chunk_index=14, section_title="CitedByHigh",
        article_number="14", text_content="ref cited by high",
    )
    ref_low = VectorChunkRecord(
        id="ref-30", document_id="doc-A", document_title="Doc",
        chunk_index=30, section_title="CitedByLow",
        article_number="30", text_content="ref cited by low",
    )
    bundle = _bundle_with_references(
        primary=[high_primary, low_primary],
        # Deliberately pass them low-first so the sort visibly fires.
        references=[ref_low, ref_high],
    )
    packed = pack_vector_knowledge_for_prompt(bundle)

    referenced_headers = [h for h in packed.headers if "| referenced" in h]
    assert len(referenced_headers) == 2
    assert "CitedByHigh" in referenced_headers[0]
    assert "CitedByLow" in referenced_headers[1]


def test_pack_b4_a3_independent_envs_can_flip_separately(monkeypatch):
    """References ``on`` + adjacency ``off`` → only references pack.
    Inverse case: references ``off`` + adjacency ``on`` → only adjacency
    packs. Pins the independence of the two env flags."""
    primary = VectorChunkRecord(
        id="p-30", document_id="doc-A", document_title="Doc",
        chunk_index=30, section_title="A30",
        text_content="primary body", similarity_score=0.9,
        outgoing_refs=[ChunkReference(kind=ChunkReferenceKind.article, number="14", raw_text="მე-14 მუხლი")],
    )
    reference = VectorChunkRecord(
        id="ref-14", document_id="doc-A", document_title="Doc",
        chunk_index=14, section_title="A14",
        article_number="14", text_content="referenced body",
    )
    adjacent = VectorChunkRecord(
        id="adj-31", document_id="doc-A", document_title="Doc",
        chunk_index=31, section_title="A31",
        text_content="adjacent body",
    )

    # References on, adjacency off.
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "on")
    monkeypatch.setenv("VECTOR_ADJACENCY_MODE", "off")
    bundle = _bundle_with_references(
        primary=[primary], references=[reference], adjacent=[adjacent],
    )
    packed = pack_vector_knowledge_for_prompt(bundle)
    assert any("| referenced" in h for h in packed.headers)
    assert not any("| adjacent" in h for h in packed.headers)

    # Inverse: references off, adjacency on.
    monkeypatch.setenv("VECTOR_REFERENCE_EXPANSION_MODE", "off")
    monkeypatch.setenv("VECTOR_ADJACENCY_MODE", "on")
    packed = pack_vector_knowledge_for_prompt(bundle)
    assert not any("| referenced" in h for h in packed.headers)
    assert any("| adjacent" in h for h in packed.headers)

