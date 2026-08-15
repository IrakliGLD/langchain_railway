"""Content contract for the retail-tariff knowledge topic.

These assert load-bearing FACTS, not prose. Each maps to a way an answer would
be wrong: reporting a net figure as a consumer price, naming the wrong company,
claiming the wrong territory, or mixing two categories into one final price.
"""
import pathlib

KNOWLEDGE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "knowledge" / "network_supply_tariffs.md"
)
KNOWLEDGE = KNOWLEDGE_PATH.read_text(encoding="utf-8")


def test_vat_basis_is_settled_not_listed_as_unknown():
    """The dashboard settles this: the view stores tariffs NET of VAT.

    networkSupplyChart.js computes vat = net * 0.18 and total = net + vat on top
    of the published final_price. An earlier draft of this file recorded the
    treatment as undeterminable, which would have the model hedge on a question
    it can answer.
    """
    lowered = KNOWLEDGE.lower()

    assert "net of vat" in lowered
    assert "18%" in KNOWLEDGE
    assert "not determinable from the" not in lowered, (
        "the VAT open item is settled and must no longer be listed as unknown"
    )


def test_reporting_rules_are_stated():
    """Report net by default; quote final_price rather than a computed sum."""
    lowered = KNOWLEDGE.lower()

    assert "gel/kwh" in lowered
    assert "final_price" in KNOWLEDGE
