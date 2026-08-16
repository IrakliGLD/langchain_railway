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


class TestAmbiguousDataBackedQuestionsReachTheData:
    """The general form of the retail and plant-fleet routing rules.

    2026-08-16 15:34: a fleet question came back ambiguous / knowledge /
    candidate_tools=[] with candidate_topics=[market_structure,
    generation_mix] and was answered as prose -- AFTER wording markers had
    been added for exactly that domain. The phrasings are unbounded; the
    datasets are not, so the rule keys on the analyzer's topic nomination.
    """

    @staticmethod
    def _ctx(topics, query_type="ambiguous"):
        from contracts.question_analysis import QuestionAnalysis
        from models import QueryContext

        payload = {
            "version": "question_analysis_v1",
            "raw_query": "q",
            "canonical_query_en": "q",
            "language": {"input_language": "en", "answer_language": "en"},
            "classification": {
                "query_type": query_type, "analysis_mode": "light",
                "intent": "i", "needs_clarification": False, "confidence": 0.7,
            },
            "routing": {
                "preferred_path": "knowledge", "needs_sql": False,
                "needs_knowledge": True, "prefer_tool": False,
                "needs_multi_tool": False, "evidence_roles": [],
            },
            "knowledge": {
                "candidate_topics": [{"name": t, "score": 0.8} for t in topics]
            },
            "tooling": {"candidate_tools": []},
            "sql_hints": {},
            "visualization": {
                "chart_requested_by_user": False, "chart_recommended": False,
                "chart_confidence": 0.0,
            },
            "analysis_requirements": {
                "needs_driver_analysis": False,
                "needs_correlation_context": False, "derived_metrics": [],
            },
        }
        ctx = QueryContext(query="q")
        ctx.question_analysis = QuestionAnalysis(**payload)
        ctx.question_analysis_source = "llm_active"
        ctx.resolved_query = "q"
        return ctx

    def test_the_exact_production_topic_set_now_routes_to_data(self):
        from agent.pipeline import _derive_response_mode

        ctx = self._ctx(["market_structure", "generation_mix"])
        assert _derive_response_mode(ctx) == "data_primary"

    def test_a_knowledge_only_topic_set_stays_on_knowledge(self):
        from agent.pipeline import _derive_response_mode

        ctx = self._ctx(["market_structure", "exchange_transition"])
        assert _derive_response_mode(ctx) == "knowledge_primary"

    def test_a_definition_question_stays_on_knowledge(self):
        """Conceptual questions are classified conceptual_definition, not
        ambiguous -- that is what keeps 'what is a capacity factor' as prose."""
        from agent.pipeline import _derive_response_mode

        ctx = self._ctx(["generation_mix"], query_type="conceptual_definition")
        assert _derive_response_mode(ctx) == "knowledge_primary"


class TestPlantViewsMatchTheDatabase:
    """Verified against screenshots of the live views (2026-08-16).

    Three scales sit in one ownership_concentration row and two of them were
    undocumented, which is the same class of error that produced the GEL/kWh
    vs GEL/MWh failures on the retail side.
    """

    def test_the_documented_columns_match_the_views(self):
        from context import DB_SCHEMA_DOC

        assert (
            "by_capacity(date, entity, segment, quantity, facility_count)" in DB_SCHEMA_DOC
        )
        # by_commissioning genuinely has no facility_count.
        assert "by_commissioning(date, entity, segment, quantity)" in DB_SCHEMA_DOC
        assert (
            "ownership_concentration(date, segment, total_generation, owner_count, "
            "hhi, top1_share, top3_share, top5_share)" in DB_SCHEMA_DOC
        )

    def test_the_scales_are_documented_where_each_is_needed(self):
        """DB_SCHEMA_DOC feeds SQL GENERATION and has a 9,000-char tripwire, so
        it carries only what a query writer needs. How to READ the numbers
        belongs in guidance, which no truncation profile sheds."""
        from context import DB_SCHEMA_DOC
        from skills.loader import load_reference

        # Terse in the schema doc: enough not to write a wrong WHERE clause.
        assert "0-10000" in DB_SCHEMA_DOC
        assert "ratios 0..1" in DB_SCHEMA_DOC
        assert "partition the same monthly total" in DB_SCHEMA_DOC

        # Full interpretation in guidance.
        rules = load_reference("energy-analyst", "plant-fleet-rules.md")
        assert "highly concentrated" in rules
        assert "1000.976" in rules, "the verified reconciliation must be quotable"

    def test_by_capacity_is_not_described_as_installed_capacity(self):
        from skills.loader import load_reference

        rules = load_reference("energy-analyst", "plant-fleet-rules.md")
        assert "does not contain installed capacity" in rules
        assert "0–10000" in rules or "0-10000" in rules


def test_the_capacity_factor_formula_is_documented_and_correct():
    """Verified against the live view: 38266 / (223.070 x 744) = 0.2305679639
    against a published 0.2305679638."""
    from skills.loader import load_reference

    published = 0.2305679638386302
    computed = 38266.0 / (223.070 * 744)
    assert abs(computed - published) < 1e-9

    rules = load_reference("energy-analyst", "plant-fleet-rules.md")
    assert "installed_capacity_mw × hours_in_month" in rules
    assert "744" in rules, "hours_in_month is calendar hours; say so"


def test_the_thousandfold_unit_gap_is_documented_with_evidence():
    """capacity_factor.generation_mwh (MWh) vs by_capacity.quantity (thousand
    MWh). Asserted from the earlier session, now confirmed on every 2020-01
    band -- 46,179 vs 46.179 and so on."""
    from context import DB_SCHEMA_DOC
    from skills.loader import load_reference

    assert "THOUSAND" in DB_SCHEMA_DOC and "1000x" in DB_SCHEMA_DOC
    rules = load_reference("energy-analyst", "plant-fleet-rules.md")
    assert "46,179" in rules and "46.179" in rules


def test_trade_by_ownership_has_no_segment_column():
    """It was wrongly grouped with the four views that carry segment='total'."""
    from context import DB_SCHEMA_DOC

    assert "trade_by_ownership(date, ownership, quantity)" in DB_SCHEMA_DOC
    assert "these four views" in DB_SCHEMA_DOC
    assert "trade_by_ownership has no\n  segment column" in DB_SCHEMA_DOC


def test_trade_is_not_presented_as_a_share_of_generation():
    from context import DB_SCHEMA_DOC

    assert "TRADE, not generation" in DB_SCHEMA_DOC


class TestCapacityBandsAreCompleteAndOrdered:
    """There are EIGHT bands and they do not sort alphabetically.

    ORDER BY capacity_category yields 101-200, 11-20, 201-500, 21-50, 51-100,
    6-10, <=5, more than 500 -- which makes any "largest band" or "smallest
    band" claim wrong. capacity_category_order exists for this; by_capacity
    has no order column at all.
    """

    BANDS = ["<=5", "6-10", "11-20", "21-50", "51-100", "101-200", "201-500", "more than 500"]

    def test_all_eight_bands_are_in_the_schema_doc(self):
        from context import DB_SCHEMA_DOC

        for band in self.BANDS:
            assert f"'{band}'" in DB_SCHEMA_DOC, f"band {band!r} missing"
        assert "8-band" in DB_SCHEMA_DOC

    def test_all_eight_bands_are_enumerated_in_guidance(self):
        """The summarizer reads guidance, not DB_SCHEMA_DOC. An answer that
        covers "all bands" has to know there are eight."""
        from skills.loader import load_reference

        rules = load_reference("energy-analyst", "plant-fleet-rules.md")
        for band in self.BANDS:
            assert f"`{band}`" in rules, f"band {band!r} not enumerated in guidance"

    def test_the_lexical_sort_trap_is_documented_in_both_places(self):
        from context import DB_SCHEMA_DOC
        from skills.loader import load_reference

        assert "capacity_category_order" in DB_SCHEMA_DOC
        assert "101-200 before 11-20" in DB_SCHEMA_DOC

        rules = load_reference("energy-analyst", "plant-fleet-rules.md")
        assert "Never sort these alphabetically" in rules
        assert "no** order column" in rules or "no order column" in rules

    def test_the_lexical_sort_really_is_wrong(self):
        """Guard the premise, so this documentation cannot become folklore."""
        assert sorted(self.BANDS) != self.BANDS
        assert sorted(self.BANDS)[0] == "101-200"

    def test_the_five_commissioning_cohorts_are_not_confused_with_the_bands(self):
        from skills.loader import load_reference

        rules = load_reference("energy-analyst", "plant-fleet-rules.md")
        for cohort in ("<=1990", "1991-2000", "2001-2010", "2011-2020", "after 2020"):
            assert cohort in rules, f"cohort {cohort!r} missing"
        assert "vintages, not sizes" in rules
