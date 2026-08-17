"""The guidance focus must not depend on what retrieval happened to return.

2026-08-17: the same question, differing only in the company name, produced
answers of 2,265 and 3,689 characters. The cause was not the wording and not the
company -- it was a cascade:

    analyzer non-determinism -> different candidate_topics
      -> different preferred_topics -> different retrieval ranking
      -> different retrieved document_type set
      -> focus general vs regulation -> guidance 20,430 vs 23,450 chars

``get_query_focus`` is deterministic on the query text and returns "general" for
both (its tariff branch needs ტარიფი; the question says ფასი), so the whole
difference came from the retrieval-driven fallback.

Two fixes, both here: a retail frame resolves its focus from the FRAME, and the
fallback -- when it is still used -- picks by a stated priority instead of by set
iteration order.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from core.llm import _resolve_summarizer_focus

# The Georgian make-or-buy question, canonicalised. Contains ფასი (price), not
# ტარიფი -- which is why get_query_focus returns "general" for it.
QUESTION = (
    "For Telmico commercial customers connected at 6-10 kV, is the retail price "
    "cheaper than buying on the wholesale market?"
)

# A retail frame is identified from stats_hint, the same signal the retail rules
# already load off.
RETAIL_STATS = (
    "--- Column Aggregates (66 rows) ---\n"
    "final_price_net_gel_kwh: mean=0.1983 min=0.1900 max=0.2100 n=66\n"
    "--- ANNUAL MAKE-OR-BUY COMPARISON ---\n"
    "  2022 FULL    12/12: supply=0.1450 benchmark=0.1470\n"
)
NON_RETAIL_STATS = "--- Column Aggregates (24 rows) ---\np_bal_gel: mean=180.0 n=24\n"


def _bundle(*document_types):
    return SimpleNamespace(
        chunks=[SimpleNamespace(document_type=dt) for dt in document_types]
    )


class TestARetailFrameResolvesItsFocusFromTheFrame:
    def test_regulation_chunks_do_not_change_a_retail_frame_focus(self):
        """The exact 2026-08-17 divergence: one run retrieved a law chunk.

        A retail comparison is a TARIFF question whatever the retriever returned.
        """
        assert _resolve_summarizer_focus(QUESTION, _bundle("regulation"), RETAIL_STATS) == "tariff"

    def test_the_focus_is_identical_with_and_without_those_chunks(self):
        """Two runs of the same question must brief the model identically."""
        with_law = _resolve_summarizer_focus(QUESTION, _bundle("law", "knowledge"), RETAIL_STATS)
        without = _resolve_summarizer_focus(QUESTION, _bundle("knowledge"), RETAIL_STATS)

        assert with_law == without == "tariff"

    def test_it_holds_when_no_chunks_were_retrieved_at_all(self):
        assert _resolve_summarizer_focus(QUESTION, None, RETAIL_STATS) == "tariff"


class TestTheFallbackIsOrderIndependent:
    def test_the_same_document_types_give_the_same_focus_either_order(self):
        """``for dt in doc_types: ... break`` iterated a SET, so the winner
        depended on set ordering rather than on any stated priority."""
        one_way = _resolve_summarizer_focus(QUESTION, _bundle("law", "knowledge"), NON_RETAIL_STATS)
        other_way = _resolve_summarizer_focus(QUESTION, _bundle("knowledge", "law"), NON_RETAIL_STATS)

        assert one_way == other_way

    def test_a_regulation_document_still_reaches_the_regulation_focus(self):
        """The fallback keeps working where it is genuinely the right answer."""
        assert _resolve_summarizer_focus(
            QUESTION, _bundle("law"), NON_RETAIL_STATS
        ) == "regulation"

    def test_unmapped_document_types_leave_the_focus_alone(self):
        assert _resolve_summarizer_focus(
            QUESTION, _bundle("knowledge", "dataset"), NON_RETAIL_STATS
        ) == "general"


class TestTheQueryTextStaysAuthoritative:
    def test_a_query_that_names_its_own_focus_wins_over_the_fallback(self):
        """get_query_focus is deterministic; only "general" defers to chunks."""
        assert _resolve_summarizer_focus(
            "რა არის მიწოდების ტარიფი?", _bundle("law"), NON_RETAIL_STATS
        ) == "tariff"

    def test_a_balancing_question_is_untouched_by_a_retail_frame_check(self):
        assert _resolve_summarizer_focus(
            "what drove the balancing price in 2024?", None, NON_RETAIL_STATS
        ) == "balancing"
