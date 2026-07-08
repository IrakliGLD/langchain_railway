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

from contracts.question_analysis import AnswerKind, KnowledgeTopicName, RenderStyle
from contracts.vector_knowledge import (
    VectorChunkRecord,
    VectorKnowledgeMode,
    VectorRetrievalTier,
)
from knowledge.vector_retrieval import retrieve_vector_knowledge

# ---------------------------------------------------------------------------
# _resolve_vector_retrieval_tier — policy unit tests
# ---------------------------------------------------------------------------


def _resolve(answer_kind, render_style, *, is_conceptual=False, topics=None):
    from agent.pipeline import _resolve_vector_retrieval_tier

    return _resolve_vector_retrieval_tier(
        answer_kind=answer_kind,
        render_style=render_style,
        is_conceptual=is_conceptual,
        topics=topics,
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


def test_tier_deterministic_data_shape_with_conceptual_disagreement_is_light():
    """Disagreement-rescue: heuristic says conceptual, analyzer says
    deterministic data shape.  Keep retrieval warm at LIGHT so the
    summarizer has regulatory context if the question is ambiguous.

    Regression for 2026-05-14 incident on
        "what is a price of electricity esco paying to sellers of
         balancing electricity?"
    where SCALAR+DETERMINISTIC bypassed the regulation corpus entirely.
    """
    for ak in (
        AnswerKind.SCALAR,
        AnswerKind.LIST,
        AnswerKind.TIMESERIES,
        AnswerKind.COMPARISON,
        AnswerKind.FORECAST,
        AnswerKind.SCENARIO,
    ):
        assert (
            _resolve(ak, RenderStyle.DETERMINISTIC, is_conceptual=True)
            == VectorRetrievalTier.LIGHT
        ), ak


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


# ---------------------------------------------------------------------------
# Market-structure rescue (2026-05-18, Q2 production trace c995f0c7)
#
# When the analyzer flags a query as touching balancing-price / tariffs /
# market-structure / direct-contract / cross-border / exchange-transition /
# CFD-PPA / PSO-trading topics, vector retrieval must
# stay warm at LIGHT even for ``render_style=DETERMINISTIC`` data shapes.
# Otherwise the LLM runs Stage 4 with ``domain_knowledge_in_prompt=0 chars``
# and applies wrong column mappings (e.g. "small hydro" → ``price_regulated_hpp_*``,
# ignoring the actual deregulated/regulated/PPA settlement-path semantics
# documented in balancing_price.md and tariffs.md).
# ---------------------------------------------------------------------------


def test_tier_market_structure_topic_rescues_deterministic_data_to_light():
    """Balancing-price topic + comparison + deterministic must yield LIGHT,
    not SKIP. This is the exact Q2 trace c995f0c7 shape (analyzer:
    answer_kind=comparison, render_style=deterministic,
    candidate_topics=[balancing_price, generation_mix, market_structure])."""
    for ak in (
        AnswerKind.SCALAR,
        AnswerKind.LIST,
        AnswerKind.TIMESERIES,
        AnswerKind.COMPARISON,
        AnswerKind.FORECAST,
        AnswerKind.SCENARIO,
    ):
        assert (
            _resolve(
                ak,
                RenderStyle.DETERMINISTIC,
                topics=[KnowledgeTopicName.BALANCING_PRICE],
            )
            == VectorRetrievalTier.LIGHT
        ), ak


def test_tier_market_structure_topics_each_individually_rescue():
    """Each market-structure topic must independently trigger
    the LIGHT rescue when paired with deterministic data shape."""
    for topic in (
        KnowledgeTopicName.BALANCING_PRICE,
        KnowledgeTopicName.TARIFFS,
        KnowledgeTopicName.MARKET_STRUCTURE,
        KnowledgeTopicName.DIRECT_CONTRACTS,
        KnowledgeTopicName.CROSS_BORDER_TRADE,
        KnowledgeTopicName.EXCHANGE_TRANSITION,
        KnowledgeTopicName.CFD_PPA,
        KnowledgeTopicName.PSO_TRADING,
    ):
        assert (
            _resolve(
                AnswerKind.COMPARISON,
                RenderStyle.DETERMINISTIC,
                topics=[topic],
            )
            == VectorRetrievalTier.LIGHT
        ), topic


def test_tier_non_market_structure_topics_keep_deterministic_skip():
    """Topics outside the market-structure set must NOT trigger the rescue.
    Pure-data topics (generation_mix, seasonal_patterns, currency_influence,
    sql_examples, general_definitions) keep the SKIP behavior — the
    deterministic renderer handles them without needing knowledge files."""
    for topic in (
        KnowledgeTopicName.GENERATION_MIX,
        KnowledgeTopicName.SEASONAL_PATTERNS,
        KnowledgeTopicName.CURRENCY_INFLUENCE,
        KnowledgeTopicName.SQL_EXAMPLES,
        KnowledgeTopicName.GENERAL_DEFINITIONS,
    ):
        assert (
            _resolve(
                AnswerKind.COMPARISON,
                RenderStyle.DETERMINISTIC,
                topics=[topic],
            )
            == VectorRetrievalTier.SKIP
        ), topic


def test_tier_market_structure_topic_mixed_with_others_still_rescues():
    """A market-structure topic appearing alongside non-rescue topics
    (the realistic case — analyzer emits up to 5 candidate topics) must
    still trigger the LIGHT rescue."""
    assert (
        _resolve(
            AnswerKind.COMPARISON,
            RenderStyle.DETERMINISTIC,
            topics=[
                KnowledgeTopicName.BALANCING_PRICE,
                KnowledgeTopicName.GENERATION_MIX,
                KnowledgeTopicName.MARKET_STRUCTURE,
            ],
        )
        == VectorRetrievalTier.LIGHT
    )


def test_tier_topics_none_or_empty_preserves_existing_skip():
    """When topics is None (analyzer absent) or empty (analyzer succeeded
    but emitted no topics), the deterministic-data SKIP behavior must
    remain unchanged — the rescue only fires on positive topic signal."""
    assert (
        _resolve(AnswerKind.COMPARISON, RenderStyle.DETERMINISTIC, topics=None)
        == VectorRetrievalTier.SKIP
    )
    assert (
        _resolve(AnswerKind.COMPARISON, RenderStyle.DETERMINISTIC, topics=[])
        == VectorRetrievalTier.SKIP
    )


def test_tier_market_structure_does_not_override_clarify_or_knowledge():
    """The rescue only applies inside the deterministic-data-shape branch.
    CLARIFY and KNOWLEDGE/EXPLANATION decisions must be unaffected by
    topic signal."""
    assert (
        _resolve(
            AnswerKind.CLARIFY,
            RenderStyle.DETERMINISTIC,
            topics=[KnowledgeTopicName.BALANCING_PRICE],
        )
        == VectorRetrievalTier.SKIP
    )
    assert (
        _resolve(
            AnswerKind.KNOWLEDGE,
            RenderStyle.NARRATIVE,
            topics=[KnowledgeTopicName.BALANCING_PRICE],
        )
        == VectorRetrievalTier.FULL
    )


def test_tier_fallback_conceptual_without_qa_is_full():
    assert _resolve(None, None, is_conceptual=True) == VectorRetrievalTier.FULL


def test_tier_fallback_non_conceptual_without_qa_is_light():
    # Preserves retrieval on narrative queries with a failed/shadow analyzer,
    # but at reduced cost (top_k=2).
    assert _resolve(None, None, is_conceptual=False) == VectorRetrievalTier.LIGHT


def test_fast_mode_forces_skip_regardless_of_answer_kind(monkeypatch):
    from agent import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "PIPELINE_MODE", "fast")

    assert _resolve(AnswerKind.KNOWLEDGE, RenderStyle.NARRATIVE) == VectorRetrievalTier.SKIP
    assert _resolve(AnswerKind.EXPLANATION, RenderStyle.NARRATIVE) == VectorRetrievalTier.SKIP
    assert _resolve(AnswerKind.SCALAR, RenderStyle.NARRATIVE) == VectorRetrievalTier.SKIP


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


def test_summarizer_profile_explanation_preserves_stats():
    from core.llm import (
        _TRUNCATION_PRIORITY_EXPLANATION,
        _select_summarizer_truncation_priority,
    )

    profile = _select_summarizer_truncation_priority(
        question_analysis=_qa(AnswerKind.EXPLANATION)
    )
    assert profile is _TRUNCATION_PRIORITY_EXPLANATION
    # Stats are heavily protected to ensure math grounding doesn't fail.
    assert profile.index("UNTRUSTED_STATISTICS") > profile.index("UNTRUSTED_DOMAIN_KNOWLEDGE")
    assert profile.index("UNTRUSTED_STATISTICS") > profile.index(
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


# ---------------------------------------------------------------------------
# Phase 2.c — Truncation priority invariants (audit, 2026-05-13)
#
# Production logs (2026-05-13) showed 3 of 12 summarizer prompts hitting the
# section-aware truncation, dropping UNTRUSTED_CONVERSATION_HISTORY,
# UNTRUSTED_DOMAIN_KNOWLEDGE, and UNTRUSTED_EXTERNAL_SOURCE_PASSAGES.  The
# truncation order is governed by per-profile priority lists in core/llm.py.
# An audit confirmed the lists already follow the design invariants below;
# these tests pin them so a future edit cannot silently regress them.
#
# Design invariants:
#   1. UNTRUSTED_CONVERSATION_HISTORY is the FIRST section dropped in every
#      summarizer profile.  After the analyzer canonicalises the question,
#      history is mostly noise — its evidentiary value is lower than data,
#      domain knowledge, or external passages.
#
#   2. For KNOWLEDGE-grounded answers (KNOWLEDGE / CLARIFY / REGULATORY_PROCEDURE)
#      UNTRUSTED_EXTERNAL_SOURCE_PASSAGES is the LAST section dropped.  These
#      passages are the primary evidence for procedural and regulatory content.
#
#   3. For DATA-grounded answers (SCALAR / LIST / TIMESERIES / COMPARISON /
#      EXPLANATION / FORECAST / SCENARIO) UNTRUSTED_STATISTICS is the LAST
#      section dropped.  Derived metrics + pre-computed stats are what the
#      summarizer grounds its arithmetic claims against.
# ---------------------------------------------------------------------------


def _all_summarizer_profiles():
    """Return (name, list) for every summarizer-side truncation profile."""
    from core import llm as llm_core
    return [
        ("_TRUNCATION_PRIORITY", llm_core._TRUNCATION_PRIORITY),
        ("_TRUNCATION_PRIORITY_DATA", llm_core._TRUNCATION_PRIORITY_DATA),
        ("_TRUNCATION_PRIORITY_KNOWLEDGE", llm_core._TRUNCATION_PRIORITY_KNOWLEDGE),
        ("_TRUNCATION_PRIORITY_EXPLANATION", llm_core._TRUNCATION_PRIORITY_EXPLANATION),
        ("_TRUNCATION_PRIORITY_FORECAST_SCENARIO", llm_core._TRUNCATION_PRIORITY_FORECAST_SCENARIO),
    ]


def test_invariant_history_dropped_first_in_every_summarizer_profile():
    """Invariant 1: UNTRUSTED_CONVERSATION_HISTORY is the FIRST section
    dropped in every summarizer-side truncation profile.
    """
    for name, profile in _all_summarizer_profiles():
        assert profile[0] == "UNTRUSTED_CONVERSATION_HISTORY", (
            f"{name}[0] = {profile[0]!r} (expected 'UNTRUSTED_CONVERSATION_HISTORY'). "
            "Phase 2.c invariant: history is the first section dropped in every "
            "summarizer profile — see comment block in tests/test_vector_retrieval_tier.py."
        )


def test_invariant_knowledge_profile_preserves_external_passages_last():
    """Invariant 2: for KNOWLEDGE answers the EXTERNAL_SOURCE_PASSAGES section
    is the LAST section dropped — it carries the primary regulatory evidence.
    """
    from core.llm import _TRUNCATION_PRIORITY_KNOWLEDGE
    assert _TRUNCATION_PRIORITY_KNOWLEDGE[-1] == "UNTRUSTED_EXTERNAL_SOURCE_PASSAGES", (
        "_TRUNCATION_PRIORITY_KNOWLEDGE[-1] must be 'UNTRUSTED_EXTERNAL_SOURCE_PASSAGES'. "
        "Phase 2.c invariant: regulatory answers ground in external passages."
    )
    # And domain knowledge is the second-most protected (after passages).
    assert _TRUNCATION_PRIORITY_KNOWLEDGE[-2] == "UNTRUSTED_DOMAIN_KNOWLEDGE", (
        "_TRUNCATION_PRIORITY_KNOWLEDGE[-2] must be 'UNTRUSTED_DOMAIN_KNOWLEDGE'."
    )


def test_invariant_data_grounded_profiles_preserve_statistics_last():
    """Invariant 3: for DATA-grounded answer kinds (SCALAR/LIST/TIMESERIES/
    COMPARISON/EXPLANATION/FORECAST/SCENARIO) the STATISTICS section is the
    LAST section dropped.
    """
    from core import llm as llm_core
    data_grounded = [
        ("_TRUNCATION_PRIORITY_DATA", llm_core._TRUNCATION_PRIORITY_DATA),
        ("_TRUNCATION_PRIORITY_EXPLANATION", llm_core._TRUNCATION_PRIORITY_EXPLANATION),
        ("_TRUNCATION_PRIORITY_FORECAST_SCENARIO", llm_core._TRUNCATION_PRIORITY_FORECAST_SCENARIO),
    ]
    for name, profile in data_grounded:
        assert profile[-1] == "UNTRUSTED_STATISTICS", (
            f"{name}[-1] = {profile[-1]!r} (expected 'UNTRUSTED_STATISTICS'). "
            "Phase 2.c invariant: data-grounded answers ground in statistics."
        )


def test_invariant_no_profile_drops_data_preview_or_statistics_before_history():
    """Defensive invariant: data preview and statistics must never come
    before history in the drop order.  Catches a likely regression mode where
    someone moves history later (e.g. clarify overlay leaking) without also
    updating the summarizer-side profile.
    """
    for name, profile in _all_summarizer_profiles():
        history_idx = profile.index("UNTRUSTED_CONVERSATION_HISTORY")
        for protected in ("UNTRUSTED_DATA_PREVIEW", "UNTRUSTED_STATISTICS"):
            if protected in profile:
                assert profile.index(protected) > history_idx, (
                    f"{name}: {protected} (idx {profile.index(protected)}) "
                    f"is at or before HISTORY (idx {history_idx}). "
                    "Phase 2.c defensive invariant."
                )
