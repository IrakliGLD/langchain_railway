from __future__ import annotations

from knowledge import get_knowledge_for_topics, infer_topic_matches, load_knowledge

LIBERALIZATION_STATUS_QUERY = (
    "What is the situation with power plant liberalization? "
    "Are many plants still regulated?"
)


def test_liberalization_status_query_selects_market_structure_and_tariffs():
    assert {"market_structure", "tariffs"} <= infer_topic_matches(
        LIBERALIZATION_STATUS_QUERY
    )


def test_liberalization_knowledge_qualifies_planned_dates_as_unverified():
    load_knowledge()

    knowledge = get_knowledge_for_topics(
        ["tariffs"],
        fallback_query=LIBERALIZATION_STATUS_QUERY,
    )

    assert "Current-status and freshness rule" in knowledge
    assert "must not be treated as confirmation that deregulation occurred" in knowledge
    assert "must not be counted as currently regulated" in knowledge
    assert "scheduled from May 2026; completion unverified" in knowledge
    assert "expected from May 2026" not in knowledge
