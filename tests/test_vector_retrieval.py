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
