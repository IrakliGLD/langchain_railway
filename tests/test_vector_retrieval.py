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
    filters = build_vector_filters(_analysis())
    assert filters.preferred_topics == ["balancing_price"]
    assert filters.languages == ["en"]


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
    filters = build_vector_filters(analysis)
    assert filters.languages == ["ka", "en"]


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
