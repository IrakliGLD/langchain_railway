"""Phase D: three-tier vector retrieval + answer-kind-aware summarizer.

Covers:
- ``_resolve_vector_retrieval_tier`` policy mapping
- ``retrieve_vector_knowledge(tier=SKIP|LIGHT|FULL)`` runtime behavior
- ``_select_summarizer_truncation_priority`` per-answer-kind dispatch
- ``llm_summarize_structured`` clearing domain knowledge on DETERMINISTIC
"""

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from types import SimpleNamespace

from contracts.question_analysis import AnswerKind, RenderStyle
from contracts.vector_knowledge import (
    VectorChunkRecord,
    VectorKnowledgeMode,
    VectorRetrievalTier,
)
from knowledge.vector_retrieval import retrieve_vector_knowledge


# ---------------------------------------------------------------------------
# _resolve_vector_retrieval_tier — policy unit tests
# ---------------------------------------------------------------------------


def _resolve(answer_kind, render_style, *, is_conceptual=False):
    from agent.pipeline import _resolve_vector_retrieval_tier

    return _resolve_vector_retrieval_tier(
        answer_kind=answer_kind,
        render_style=render_style,
        is_conceptual=is_conceptual,
    )


def test_tier_knowledge_is_full():
    assert _resolve(AnswerKind.KNOWLEDGE, RenderStyle.NARRATIVE) == VectorRetrievalTier.FULL
    assert _resolve(AnswerKind.KNOWLEDGE, RenderStyle.DETERMINISTIC) == VectorRetrievalTier.FULL


def test_tier_explanation_is_full():
    assert _resolve(AnswerKind.EXPLANATION, RenderStyle.NARRATIVE) == VectorRetrievalTier.FULL
    assert _resolve(AnswerKind.EXPLANATION, RenderStyle.DETERMINISTIC) == VectorRetrievalTier.FULL


def test_tier_clarify_is_skip():
    # CLARIFY wins over render_style — no data to ground regardless.
    assert _resolve(AnswerKind.CLARIFY, RenderStyle.NARRATIVE) == VectorRetrievalTier.SKIP
    assert _resolve(AnswerKind.CLARIFY, RenderStyle.DETERMINISTIC) == VectorRetrievalTier.SKIP


def test_tier_deterministic_data_shapes_skip():
    for ak in (
        AnswerKind.SCALAR,
        AnswerKind.LIST,
        AnswerKind.TIMESERIES,
        AnswerKind.COMPARISON,
        AnswerKind.FORECAST,
        AnswerKind.SCENARIO,
    ):
        assert _resolve(ak, RenderStyle.DETERMINISTIC) == VectorRetrievalTier.SKIP, ak


def test_tier_narrative_data_shapes_light():
    for ak in (
        AnswerKind.SCALAR,
        AnswerKind.LIST,
        AnswerKind.TIMESERIES,
        AnswerKind.COMPARISON,
        AnswerKind.FORECAST,
        AnswerKind.SCENARIO,
    ):
        assert _resolve(ak, RenderStyle.NARRATIVE) == VectorRetrievalTier.LIGHT, ak


def test_tier_fallback_conceptual_without_qa_is_full():
    assert _resolve(None, None, is_conceptual=True) == VectorRetrievalTier.FULL


def test_tier_fallback_non_conceptual_without_qa_is_light():
    # Preserves retrieval on narrative queries with a failed/shadow analyzer,
    # but at reduced cost (top_k=2).
    assert _resolve(None, None, is_conceptual=False) == VectorRetrievalTier.LIGHT


# ---------------------------------------------------------------------------
# retrieve_vector_knowledge(tier=...) — runtime behavior
# ---------------------------------------------------------------------------


class _Embed:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class _Store:
    def __init__(self):
        self.calls: list[dict] = []

    def search_chunks(self, **kwargs):
        self.calls.append(kwargs)
        return [
            VectorChunkRecord(
                id="c1",
                document_id="d1",
                document_title="Rules",
                source_key="rules-1",
                section_title="Settlement",
                text_content="Some passage.",
                topics=["balancing"],
                similarity_score=0.9,
            )
        ]


def test_retrieve_skip_returns_empty_bundle_without_calling_store():
    store = _Store()
    bundle = retrieve_vector_knowledge(
        "any query",
        retrieval_mode=VectorKnowledgeMode.shadow,
        store=store,
        embedding_provider=_Embed(),
        tier=VectorRetrievalTier.SKIP,
    )
    assert bundle.chunk_count == 0
    assert bundle.chunks == []
    assert bundle.top_k == 0
    assert bundle.error == ""
    assert store.calls == []  # store never touched


def test_retrieve_light_uses_top_k_two():
    store = _Store()
    bundle = retrieve_vector_knowledge(
        "why did balancing price rise?",
        retrieval_mode=VectorKnowledgeMode.active,
        store=store,
        embedding_provider=_Embed(),
        tier=VectorRetrievalTier.LIGHT,
    )
    assert bundle.top_k == 2
    assert store.calls, "search_chunks should still be called on LIGHT"
    assert store.calls[0]["top_k"] == 2


def test_retrieve_full_uses_default_top_k():
    # Default env VECTOR_KNOWLEDGE_TOP_K is 6.
    store = _Store()
    bundle = retrieve_vector_knowledge(
        "why did balancing price rise?",
        retrieval_mode=VectorKnowledgeMode.active,
        store=store,
        embedding_provider=_Embed(),
        tier=VectorRetrievalTier.FULL,
    )
    assert bundle.top_k >= 6
    assert store.calls[0]["top_k"] >= 6


def test_retrieve_default_tier_is_full():
    # Backward compatibility: callers that don't pass ``tier`` get FULL.
    store = _Store()
    bundle = retrieve_vector_knowledge(
        "some query",
        retrieval_mode=VectorKnowledgeMode.shadow,
        store=store,
        embedding_provider=_Embed(),
    )
    assert bundle.top_k >= 6


# ---------------------------------------------------------------------------
# _select_summarizer_truncation_priority — per-answer-kind dispatch
# ---------------------------------------------------------------------------


def _qa(answer_kind):
    # Minimal stand-in: the selector only reads ``answer_kind``.
    return SimpleNamespace(answer_kind=answer_kind)


def test_summarizer_profile_forecast_preserves_data():
    from core.llm import (
        _TRUNCATION_PRIORITY_FORECAST_SCENARIO,
        _select_summarizer_truncation_priority,
    )

    assert (
        _select_summarizer_truncation_priority(question_analysis=_qa(AnswerKind.FORECAST))
        is _TRUNCATION_PRIORITY_FORECAST_SCENARIO
    )
    # Stats + data preview must be the LAST two to drop.
    assert _TRUNCATION_PRIORITY_FORECAST_SCENARIO[-1] == "UNTRUSTED_STATISTICS"
    assert _TRUNCATION_PRIORITY_FORECAST_SCENARIO[-2] == "UNTRUSTED_DATA_PREVIEW"


def test_summarizer_profile_scenario_matches_forecast():
    from core.llm import (
        _TRUNCATION_PRIORITY_FORECAST_SCENARIO,
        _select_summarizer_truncation_priority,
    )

    assert (
        _select_summarizer_truncation_priority(question_analysis=_qa(AnswerKind.SCENARIO))
        is _TRUNCATION_PRIORITY_FORECAST_SCENARIO
    )


def test_summarizer_profile_explanation_sheds_stats_first():
    from core.llm import (
        _TRUNCATION_PRIORITY_EXPLANATION,
        _select_summarizer_truncation_priority,
    )

    profile = _select_summarizer_truncation_priority(
        question_analysis=_qa(AnswerKind.EXPLANATION)
    )
    assert profile is _TRUNCATION_PRIORITY_EXPLANATION
    # Stats are shed before domain / external passages.
    assert profile.index("UNTRUSTED_STATISTICS") < profile.index("UNTRUSTED_DOMAIN_KNOWLEDGE")
    assert profile.index("UNTRUSTED_STATISTICS") < profile.index(
        "UNTRUSTED_EXTERNAL_SOURCE_PASSAGES"
    )


def test_summarizer_profile_knowledge_matches_legacy_knowledge_profile():
    from core.llm import (
        _TRUNCATION_PRIORITY_KNOWLEDGE,
        _select_summarizer_truncation_priority,
    )

    assert (
        _select_summarizer_truncation_priority(question_analysis=_qa(AnswerKind.KNOWLEDGE))
        is _TRUNCATION_PRIORITY_KNOWLEDGE
    )


def test_summarizer_profile_clarify_matches_knowledge_profile():
    from core.llm import (
        _TRUNCATION_PRIORITY_KNOWLEDGE,
        _select_summarizer_truncation_priority,
    )

    assert (
        _select_summarizer_truncation_priority(question_analysis=_qa(AnswerKind.CLARIFY))
        is _TRUNCATION_PRIORITY_KNOWLEDGE
    )


def test_summarizer_profile_data_shapes_match_data_profile():
    from core.llm import (
        _TRUNCATION_PRIORITY_DATA,
        _select_summarizer_truncation_priority,
    )

    for ak in (AnswerKind.SCALAR, AnswerKind.LIST, AnswerKind.TIMESERIES, AnswerKind.COMPARISON):
        assert (
            _select_summarizer_truncation_priority(question_analysis=_qa(ak))
            is _TRUNCATION_PRIORITY_DATA
        ), ak


def test_summarizer_profile_falls_back_to_response_mode_without_qa():
    from core.llm import (
        _TRUNCATION_PRIORITY_DATA,
        _TRUNCATION_PRIORITY_KNOWLEDGE,
        _select_summarizer_truncation_priority,
    )

    assert (
        _select_summarizer_truncation_priority(response_mode="knowledge_primary")
        is _TRUNCATION_PRIORITY_KNOWLEDGE
    )
    assert (
        _select_summarizer_truncation_priority(resolution_policy="clarify")
        is _TRUNCATION_PRIORITY_KNOWLEDGE
    )
    assert (
        _select_summarizer_truncation_priority(response_mode="data_primary")
        is _TRUNCATION_PRIORITY_DATA
    )


def test_summarizer_profile_uses_effective_answer_kind_when_qa_absent():
    from core.llm import (
        _TRUNCATION_PRIORITY_FORECAST_SCENARIO,
        _select_summarizer_truncation_priority,
    )

    assert (
        _select_summarizer_truncation_priority(effective_answer_kind=AnswerKind.FORECAST)
        is _TRUNCATION_PRIORITY_FORECAST_SCENARIO
    )


def test_summarizer_profile_prefers_qa_over_effective_answer_kind():
    from core.llm import (
        _TRUNCATION_PRIORITY_EXPLANATION,
        _select_summarizer_truncation_priority,
    )

    assert (
        _select_summarizer_truncation_priority(
            question_analysis=_qa(AnswerKind.EXPLANATION),
            effective_answer_kind=AnswerKind.FORECAST,
        )
        is _TRUNCATION_PRIORITY_EXPLANATION
    )
