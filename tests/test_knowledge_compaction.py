import json
from pathlib import Path

import knowledge


def test_compact_knowledge_json_selects_relevant_markdown_sections_within_budget():
    payload = json.dumps(
        {
            "balancing_price": """# Balancing Price

## Definition
Balancing price is the settlement price for imbalances.

## Why Price Does Not Fall During Surplus
During surplus, mandatory purchase rules and seller composition can keep the balancing price elevated.
This mechanism explains why extra supply does not automatically create a lower offer price.

## Unrelated Historical Appendix
This appendix contains a long chronology of unrelated administrative events. """
            + ("Unrelated archive material. " * 80)
        }
    )

    result = knowledge.compact_knowledge_json(
        payload,
        query="Why did the balancing price not fall during surplus?",
        max_chars=650,
    )

    assert len(result) <= 650
    selected = json.loads(result)["balancing_price"]
    assert "Why Price Does Not Fall During Surplus" in selected
    assert "mandatory purchase rules" in selected
    assert "Unrelated Historical Appendix" not in selected


def test_compact_knowledge_json_removes_exact_vector_overlap_but_keeps_complementary_context():
    duplicate = (
        "Official settlement rules require registered participants to submit schedules "
        "before the gate closes for the relevant trading interval."
    )
    complement = (
        "Curated operational context explains that late schedule changes can increase "
        "imbalance exposure even when registration is already complete."
    )
    payload = json.dumps(
        {
            "market_structure": (
                "# Market Structure\n\n## Scheduling and Registration\n"
                f"{duplicate}\n\n{complement}"
            )
        }
    )

    result = knowledge.compact_knowledge_json(
        payload,
        query="Explain participant scheduling and registration",
        max_chars=1000,
        exclude_text=f"[1] Market Rules\n{duplicate}",
    )

    selected = json.loads(result)["market_structure"]
    assert duplicate not in selected
    assert complement in selected


def test_compact_knowledge_json_preserves_unfiltered_contract_without_a_budget():
    payload = json.dumps({"topic": "# Topic\nComplete content."})

    assert knowledge.compact_knowledge_json(payload, query="topic") == payload


def test_knowledge_selection_golden_corpus_preserves_required_evidence_anchors():
    corpus_path = Path(__file__).parents[1] / "evaluation" / "knowledge_selection_golden.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    knowledge.load_knowledge()

    for case in corpus["cases"]:
        full = knowledge.get_knowledge_json_with_topics(
            case["topics"],
            fallback_query=case["query"],
            use_cache=False,
        )
        selected = knowledge.compact_knowledge_json(
            full,
            query=case["query"],
            max_chars=case["max_chars"],
        )

        assert len(selected) <= case["max_chars"], case["id"]
        for anchor in case["must_include"]:
            assert anchor in selected, (case["id"], anchor)
