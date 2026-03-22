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
from contracts.vector_knowledge import VectorChunkRecord, VectorKnowledgeMode
from knowledge.vector_retrieval import (
    build_vector_filters,
    format_vector_knowledge_for_prompt,
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
    assert captured["candidate_k"] == [24, 24, 24]


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
