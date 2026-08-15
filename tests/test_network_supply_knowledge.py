"""Content contract for the retail-tariff knowledge topic.

These assert load-bearing FACTS, not prose. Each maps to a way an answer would
be wrong: reporting a net figure as a consumer price, naming the wrong company,
claiming the wrong territory, or mixing two categories into one final price.
"""
import os
import pathlib

# The tool-matrix drift test imports agent.tools.*, which imports config, which
# validates its settings at import time. Same preamble as the other modules.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

KNOWLEDGE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "knowledge" / "network_supply_tariffs.md"
)
KNOWLEDGE = KNOWLEDGE_PATH.read_text(encoding="utf-8")

# Markdown wraps prose, so a phrase can straddle a newline. Collapse whitespace
# before matching multi-word phrases -- otherwise these tests fail on reflow
# rather than on missing content.
import re  # noqa: E402

KNOWLEDGE_FLAT = re.sub(r"\s+", " ", KNOWLEDGE)


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


def test_every_company_code_has_its_full_legal_name():
    """An answer naming 'epg' rather than Energo-Pro Georgia is unhelpful; one
    naming the wrong company is wrong."""
    for code, name in [
        ("gse", "Georgian State Electrosystem"),
        ("telmico", "Tbilisi Electricity Supply Company"),
        ("eps", "EP Georgia Supply"),
        ("epg", "Energo-Pro Georgia"),
    ]:
        assert name in KNOWLEDGE, f"{code} is missing its full name ({name})"


def test_service_territories_including_the_suburb_nuance():
    """EPG/EPS also serve some Tbilisi suburbs.

    Without this the model states that Telasi/Telmico serve Tbilisi
    exclusively, so 'a household in Tbilisi' resolves to one supplier when it
    may be either.
    """
    lowered = KNOWLEDGE.lower()

    assert "tbilisi" in lowered
    assert "suburb" in lowered


def test_gse_is_identified_as_the_transmission_system_operator():
    assert "TSO" in KNOWLEDGE


def test_knowledge_forbids_mixing_categories():
    """Components from different categories cannot be combined into one price."""
    lowered = KNOWLEDGE.lower()

    assert "never mix" in lowered or "do not mix" in lowered


def test_knowledge_states_the_wholesale_comparison_basis():
    """Benchmark = (p_bal_gel + p_gcap_gel) / 1000, in GEL/kWh.

    The charge goes on the wholesale side rather than being taken off the
    tariff, so the regulated figure stays equal to what is actually charged.
    """
    flat = KNOWLEDGE_FLAT.lower()

    assert "p_gcap_gel" in KNOWLEDGE
    assert "guaranteed capacity" in flat
    assert "added to the wholesale" in flat
    assert "(p_bal_gel + p_gcap_gel) / 1000" in KNOWLEDGE_FLAT


def test_knowledge_file_lists_every_tool_category_voltage():
    """The tool's matrix is the source of truth. If the knowledge file drifts
    from it, the model reads one set of categories and the tool serves another.
    """
    from agent.tools.end_user_price_tools import END_USER_CATEGORIES

    for category in END_USER_CATEGORIES:
        assert category.volate in KNOWLEDGE, f"{category.volate} missing from the topic file"
