"""Content rules the annual make-or-buy block needs in order to be read correctly.

The block makes year-to-year sign flips visible. Two facts decide what a reader
should conclude from them, and without either the natural reading is wrong:

* the supply tariff is fixed for a regulatory period against an EXPECTED
  wholesale price, and the shortfall is settled in the NEXT period -- so a year
  where regulated sits above wholesale may simply be recovering an earlier
  year, and one year's sign is not a verdict on that year's market;
* household regulated prices sit below the benchmark structurally, so the live
  question is the commercial categories.

Assertions here are on load-bearing facts and on what actually reaches the
prompt, not on the presence of any particular sentence.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import re

from knowledge import get_knowledge_for_topics, load_knowledge
from skills.loader import load_reference

RETAIL_QUERY = "is it cheaper to buy on the wholesale market than the regulated tariff"


def _flat(text: str) -> str:
    """Lowercased with runs of whitespace collapsed.

    These documents are hard-wrapped prose, so a phrase can land either side of
    a line break depending on where the paragraph happens to reflow. Asserting a
    FACT is the point; asserting a particular wrapping is not, and would break
    on the next edit that shifts a margin.
    """
    return re.sub(r"\s+", " ", (text or "")).lower()


def _retail_knowledge() -> str:
    load_knowledge()
    return _flat(
        get_knowledge_for_topics(
            ["network_supply_tariffs"], fallback_query=RETAIL_QUERY
        )
    )


def _retail_rules() -> str:
    return _flat(load_reference("energy-analyst", "retail-tariff-rules.md"))


def test_knowledge_explains_why_the_retail_price_lags_the_wholesale_one():
    """Fact 3, the mechanism.

    The file already said a balancing-price movement does not reach the end-user
    price "directly or immediately" and then stopped. The missing half is WHY:
    the tariff is set for a regulatory period against an expected wholesale
    cost, and the difference is settled in the next period.
    """
    knowledge = _retail_knowledge()

    assert "regulatory period" in knowledge
    assert "next regulatory period" in knowledge
    assert "expected" in knowledge or "forecast" in knowledge


def test_knowledge_scopes_the_household_claim_instead_of_predicting_it():
    """Fact 2, phrased as a standing scoped statement.

    "expected to stay this way" is a prediction; "under current arrangements"
    is a claim about now with its scope attached, which is both more accurate
    and consistent with the file's existing "only under the current
    transitional model".
    """
    knowledge = _retail_knowledge()

    assert "under current arrangements" in knowledge
    assert "universal supplier" in knowledge
    assert "expected to stay" not in knowledge


def test_knowledge_points_at_currency_influence_rather_than_restating_it():
    """Fact 1b: the consequence belongs here, the mechanism stays where it is."""
    knowledge = _retail_knowledge()

    assert "exchange-rate risk" in knowledge or "exchange rate risk" in knowledge
    assert "currency_influence" in knowledge


def test_rules_allow_a_per_year_record_but_still_forbid_a_switching_strategy():
    """The reconciliation.

    "Compare over a sustained period, not month by month" read literally argues
    against showing per-year detail at all, and a model obeying it would
    suppress the block. A switching STRATEGY stays forbidden -- the choice is
    irreversible -- while a per-year RECORD OF OUTCOMES is the volatility
    evidence that irreversibility makes central.
    """
    rules = _retail_rules()

    assert "per year" in rules or "per-year" in rules
    # The irreversibility rule must survive the reconciliation intact.
    assert "irreversible" in rules
    assert "cannot return to regulated supply" in rules


def test_rules_say_one_years_sign_is_not_a_verdict_on_that_year():
    """Fact 3, the reading rule -- the load-bearing half.

    This lives in guidance rather than only in knowledge because guidance is in
    no truncation profile: on the 2026-08-16 trace the knowledge path delivered
    0 chars while guidance delivered 14,149.
    """
    rules = _retail_rules()

    assert "regulatory period" in rules
    assert "recovering" in rules or "recover" in rules


def test_rules_tell_the_answer_to_quote_the_annual_block():
    rules = _retail_rules()

    assert "annual make-or-buy comparison" in rules


def test_rules_name_the_household_categories_as_a_settled_case():
    rules = _retail_rules()

    assert "household" in rules
    assert "commercial" in rules


def test_coverage_and_depth_are_reconciled_not_left_contradicting():
    """Audit finding, Phase 3.

    "Give the general picture from the data -- every category, never averaged"
    and "give the household categories one line, spend the analysis on the
    commercial ones" are both in this document and neither says which wins. A
    model can satisfy one by violating the other.

    They are reconcilable: every category is still COVERED, what differs is the
    DEPTH each gets. The document has to say so rather than leaving the reader
    to guess.
    """
    rules = _retail_rules()

    assert "every category" in rules
    assert "depth" in rules
    # Coverage is not the thing being traded away.
    assert "still covered" in rules or "every category is still" in rules


def test_knowledge_says_the_consumers_own_load_shape_moves_the_wholesale_cost():
    """Domain owner, 2026-08-17. Absent from every file before this.

    The regulated stack is flat, so its mean is its per-kWh price. The wholesale
    side is seasonal, so an unweighted mean is what a consumer pays only if
    consumption is flat across the year. The asymmetry is the point: the caveat
    applies to ONE side of the comparison.
    """
    knowledge = _retail_knowledge()

    assert "unweighted" in knowledge or "consumption-weighted" in knowledge
    assert "summer" in knowledge and "winter" in knowledge
    # The direction has to be stated, not left for the reader to infer.
    assert "summer-heavy" in knowledge or "more consumption in summer" in knowledge


def test_rules_forbid_quoting_the_unweighted_mean_as_what_a_consumer_pays():
    """And require the seasonal split, which seasonal_patterns.md already demands.

    "ALWAYS mention summer and winter averages separately when comparing prices
    -- never use annual averages only." The annual block reported annual means
    only, so the rule and the evidence disagreed.
    """
    rules = _retail_rules()

    assert "unweighted" in rules
    assert "benchmark by season" in rules
    # The flat side must stay unsplit.
    assert "never split" in rules or "not split by season" in rules


def test_retail_rules_load_whenever_the_annual_block_is_present():
    """Wiring: the reading rule must reach the prompt whenever the data does.

    The rules were keyed only on ``final_price_net_gel_kwh``. The block is keyed
    on the supply tariff and the benchmark. Those coincide today because the
    tool returns all of them together, but a frame carrying the block without
    the final-price column would ship per-year verdicts with no rule for
    reading them -- which is exactly the failure fact 3 exists to prevent.
    """
    from core.llm import _retail_rules_apply

    assert _retail_rules_apply("... final_price_net_gel_kwh: mean=0.1983 ...") is True
    assert _retail_rules_apply("--- ANNUAL MAKE-OR-BUY COMPARISON ---\n2022 ...") is True
    assert _retail_rules_apply("--- Column Aggregates (12 rows) ---") is False
    assert _retail_rules_apply("") is False
    assert _retail_rules_apply(None) is False


def test_both_modules_agree_on_the_annual_section_name():
    """Drift guard: two modules name this section, and a rename in one is silent.

    ``agent/analyzer.py`` emits the header; ``core/llm.py`` both prioritises the
    section and triggers the retail rules off it. If they disagree the block is
    shed first and the rules stop loading -- with no error either way.
    """
    from agent.analyzer import _ANNUAL_COMPARISON_HEADER
    from core.llm import ANNUAL_COMPARISON_SECTION

    assert _ANNUAL_COMPARISON_HEADER == f"--- {ANNUAL_COMPARISON_SECTION} ---"
