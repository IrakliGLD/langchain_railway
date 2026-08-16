"""A capability must be registered everywhere its behaviour is governed.

Adding a tool or topic means touching several registries. The ones that make
it *visible* (enum, catalog) are obvious; the ones that make it *work* are
not, and every miss so far has failed silently -- a warning at most.

``get_end_user_prices`` hit this three times in one session:

* absent from ``validate_tool_relevance`` -- every routed call was rejected
  before executing, so the tool never ran;
* absent from ``_TOOL_ADAPTERS`` -- evidence finalization skipped on every
  request, logged only as "No frame adapter for tool ...";
* absent from the analyzer's routing examples -- 0 mentions against 2-6 for
  every other tool, so classification was reliable only when the question's
  wording happened to overlap the catalog concepts.

This gate turns each of those from a production symptom into a failing test.
It proves PRESENCE, not quality: a weak example still passes. That is the
point -- the failure mode was silent absence, and absence is what a registry
can check.

Exemptions must be explicit, named and reasoned. An empty exemption set is
the healthy state.
"""
import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest  # noqa: E402

from contracts.question_analysis import KnowledgeTopicName, ToolName  # noqa: E402
from contracts.question_analysis_catalogs import (  # noqa: E402
    QUESTION_ANALYSIS_TOOL_CATALOG,
    QUESTION_ANALYSIS_TOPIC_CATALOG,
)

#: Tools that legitimately have no frame adapter, with the reason. Adding a
#: name here is a deliberate statement that evidence finalization does not
#: apply, not a way to silence the gate.
_FRAME_ADAPTER_EXEMPT: dict = {}

#: Topics with no data tool behind them are knowledge-only by design.
_TOOL_COVERAGE_EXEMPT: dict = {}

_TOOL_NAMES = sorted(tool.value for tool in ToolName)
_TOPIC_NAMES = sorted(topic.value for topic in KnowledgeTopicName)


@pytest.mark.parametrize("tool", _TOOL_NAMES)
def test_every_tool_is_in_the_analyzer_tool_catalog(tool):
    """Without a catalog entry the analyzer cannot nominate the tool at all."""
    catalogued = {entry["name"] for entry in QUESTION_ANALYSIS_TOOL_CATALOG}
    assert tool in catalogued


@pytest.mark.parametrize("topic", _TOPIC_NAMES)
def test_every_topic_is_in_the_analyzer_topic_catalog(topic):
    catalogued = {entry["name"] for entry in QUESTION_ANALYSIS_TOPIC_CATALOG}
    assert topic in catalogued


@pytest.mark.parametrize("tool", _TOOL_NAMES)
def test_every_tool_has_a_relevance_mapping(tool):
    """A tool missing here is rejected after routing, so it never executes."""
    from utils.query_validation import validate_tool_relevance

    # Signature is (query, tool_name) -- passing them the other way round makes
    # the query string look like an unknown tool and every case "fails".
    _, reason = validate_tool_relevance("tariff price generation demand trade", tool)
    assert "Unknown tool relevance mapping" not in str(reason), reason


@pytest.mark.parametrize("tool", _TOOL_NAMES)
def test_every_tool_has_a_frame_adapter(tool):
    """Without one, evidence finalization is skipped for that tool -- which
    matters most for multi-tool answers, where it assembles the evidence."""
    from agent.frame_adapters import _TOOL_ADAPTERS

    if tool in _FRAME_ADAPTER_EXEMPT:
        pytest.skip(f"exempt: {_FRAME_ADAPTER_EXEMPT[tool]}")
    assert tool in _TOOL_ADAPTERS, (
        f"{tool} has no frame adapter; evidence finalization will be skipped "
        "on every request that uses it"
    )


@pytest.mark.parametrize("tool", _TOOL_NAMES)
def test_every_tool_carries_routing_examples(tool):
    """Examples live in the catalog as data, not hand-written in the prompt.

    Hand-written prose is why coverage drifted: nothing required a new tool to
    appear there, so ``get_end_user_prices`` shipped with 0 mentions while
    every other tool had 2-6, and its classification was reliable only when
    the question's wording happened to match the catalog concepts.
    """
    entry = next(e for e in QUESTION_ANALYSIS_TOOL_CATALOG if e["name"] == tool)
    examples = entry.get("example_questions") or []

    assert examples, f"{tool} has no example_questions in the tool catalog"
    assert all(isinstance(q, str) and q.strip() for q in examples)


@pytest.mark.parametrize("topic", _TOPIC_NAMES)
def test_every_topic_carries_routing_examples(topic):
    entry = next(e for e in QUESTION_ANALYSIS_TOPIC_CATALOG if e["name"] == topic)
    examples = entry.get("example_questions") or []

    assert examples, f"{topic} has no example_questions in the topic catalog"
    assert all(isinstance(q, str) and q.strip() for q in examples)


def test_the_examples_reach_the_analyzer_prompt():
    """The catalogs are injected as compact JSON, so a new field flows through
    with no plumbing -- but only if that is actually still true."""
    from core.llm import _TOOL_CATALOG_JSON, _TOPIC_CATALOG_JSON

    tool_entry = next(
        e for e in QUESTION_ANALYSIS_TOOL_CATALOG if e.get("example_questions")
    )
    topic_entry = next(
        e for e in QUESTION_ANALYSIS_TOPIC_CATALOG if e.get("example_questions")
    )

    assert tool_entry["example_questions"][0] in _TOOL_CATALOG_JSON
    assert topic_entry["example_questions"][0] in _TOPIC_CATALOG_JSON


def test_exemptions_are_reasoned():
    """An exemption without a reason is a silenced gate."""
    for name, reason in {**_FRAME_ADAPTER_EXEMPT, **_TOOL_COVERAGE_EXEMPT}.items():
        assert isinstance(reason, str) and len(reason) > 20, (
            f"exemption for {name} needs a real reason, got {reason!r}"
        )


def test_the_retail_adapter_canonicalises_to_the_wholesale_unit():
    """The whole point of registering it: a retail price and a balancing price
    must land in the same canonical unit, or the comparison the tool exists to
    serve is a factor of 1000 out."""
    import pandas as pd

    from agent.frame_adapters import METRIC_UNITS, adapt_end_user_prices

    df = pd.DataFrame([
        {
            "date": "2026-06-01",
            "supplier": "telmico",
            "category": "220/380|hh|cat2",
            "series_label": "Telmico — Household 101-301 kWh",
            "final_price_net_gel_kwh": 0.1983,
        }
    ])
    frame = adapt_end_user_prices(df)

    assert frame.rows, "adapter produced no observations"
    row = frame.rows[0]
    assert row["unit"] == "tetri/kWh"
    assert round(row["value"], 3) == 19.830
    # Same canonical value the wholesale side reaches from GEL/MWh.
    wholesale = METRIC_UNITS.get("price.gel").storage_to_canonical(198.3)
    assert round(row["value"], 6) == round(wholesale, 6)
    assert METRIC_UNITS.are_compatible("retail_price.gel", "price.gel")
    # The series must stay identifiable: a retail price without its company
    # and category is not a price anyone pays.
    assert row["entity_id"] == "220/380|hh|cat2"
    assert "Telmico" in row["entity_label"]


def test_plant_fleet_questions_reach_the_data():
    """2026-08-16: two fleet questions came back ambiguous/knowledge with
    candidate_tools=[] (confidence 0.88 and 0.32) and were answered as prose
    from generation_mix knowledge, while by_capacity, by_commissioning,
    capacity_factor and ownership_concentration sat one query away."""
    from agent.retail_routing import looks_like_plant_fleet_question

    for question in (
        "What is the installed capacity by band?",
        "How has the capacity factor changed for hydro?",
        "When was most of the fleet commissioned?",
        "What is the ownership concentration of generation?",
        "How many power plants are there?",
    ):
        assert looks_like_plant_fleet_question(question), question


def test_capacity_wording_that_means_something_else_is_not_claimed():
    """"capacity" collides with the guaranteed capacity FEE and with
    cross-border interconnection capacity -- different subjects, different
    data. Claiming them would route a tariff question to the fleet views."""
    from agent.retail_routing import looks_like_plant_fleet_question

    for question in (
        "What is the guaranteed capacity fee?",
        "What is the interconnection capacity with Turkey?",
        "How is cross-border capacity allocated?",
    ):
        assert not looks_like_plant_fleet_question(question), question


def test_plant_fleet_wording_routes_to_the_right_knowledge_topics():
    """Before these entries, "installed capacity" matched only the guaranteed
    capacity fee and cross-border interconnection entries, so a fleet question
    was answered from the wrong knowledge entirely."""
    from knowledge import infer_topic_matches

    assert "generation_mix" in infer_topic_matches("installed capacity by band")
    assert "generation_mix" in infer_topic_matches("capacity factor for hydro")
    assert "generation_mix" in infer_topic_matches("when were plants commissioned")
    concentration = infer_topic_matches("ownership concentration of generation")
    assert {"generation_mix", "market_structure"} & concentration

    # And the collisions still resolve to their own topics.
    assert "cross_border_capacity" in infer_topic_matches("interconnection capacity")


def test_the_plant_fleet_rules_are_registered_and_state_the_unit_trap():
    from skills.loader import _EXPECTED_FILES, load_reference

    assert ("energy-analyst", "plant-fleet-rules.md") in _EXPECTED_FILES
    rules = load_reference("energy-analyst", "plant-fleet-rules.md").lower()
    assert rules, "plant-fleet-rules.md is missing"
    # generation_mwh is plain MWh while neighbouring quantity columns are
    # thousand MWh -- a factor of 1000 waiting to happen.
    assert "thousand mwh" in rules
    assert "generation_mwh" in rules
    assert "never pool" in rules or "do not pool" in rules
