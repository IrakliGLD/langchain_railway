"""Routing contract tests for the negligible-import PPA/CfD approximation."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent import planner
from models import QueryContext
from utils.residual_price import is_implied_ppa_cfd_price_query

QUERY = (
    "Prices of regulated and deregulated plants are known. Find months where the share of import "
    "in the balancing basket is less than 0.2% and calculate the weighted average PPA/CfD price."
)


def test_active_routing_uses_deterministic_contract_without_llm(monkeypatch):
    monkeypatch.setattr(
        planner,
        "llm_analyze_question",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    out = planner.analyze_question_active(QueryContext(query=QUERY))

    assert out.question_analysis_source == "llm_active"
    assert out.question_analysis.classification.intent == "implied_ppa_cfd_price_approximation"
    assert out.question_analysis.render_style.value == "deterministic"
    assert out.question_analysis.tooling.candidate_tools[0].name.value == "get_prices"
    assert out.question_analysis.analysis_requirements.derived_metrics == []


def test_signal_rejects_non_negligible_or_underdefined_requests():
    assert is_implied_ppa_cfd_price_query(QUERY)
    assert not is_implied_ppa_cfd_price_query(QUERY.replace("0.2%", "20%"))
    assert not is_implied_ppa_cfd_price_query(QUERY.replace("PPA/CfD", "PPA"))
    assert not is_implied_ppa_cfd_price_query(QUERY.replace("regulated and deregulated", "deregulated"))
    assert not is_implied_ppa_cfd_price_query("What is the average PPA/CfD price?")


# ---------------------------------------------------------------------------
# Threshold framing (items 1+2). The same analytical constraint — "restrict to
# months where the unknown-price residual is negligible" — can be stated from
# either side:
#   * uncovered/complement side: "import share is less than 0.5%"
#   * covered/coverage side:     "ppa+cfd+regulated+deregulated > 99.5%"
# Both must normalize to ONE quantity (max_uncovered_share) that routing AND
# the downstream filter consult, so they cannot disagree. Prod trace
# span-28451264 (2026-07-25): the coverage-side phrasing missed the routing
# gate entirely (query_type=ambiguous, confidence=0.30), and would ALSO have
# filtered share_import > 0.995 (zero months) had it routed.
# ---------------------------------------------------------------------------

COVERAGE_QUERY = (
    "Balancing electricity price is a weighted average price of regulated thermal and hydro, "
    "deregulated renewables, import, ppa and cfd. I want to estimate the average price of cfd "
    "and ppa combined. Find months where the share of ppa, cfd, regulated and deregulated "
    "plants combined are more than 99.5%."
)


def test_coverage_side_phrasing_routes_like_complement_side():
    """The user's real phrasing (coverage-side) must route deterministically."""
    assert is_implied_ppa_cfd_price_query(COVERAGE_QUERY)


def test_both_framings_normalize_to_same_uncovered_share():
    from utils.residual_price import extract_residual_coverage_threshold

    coverage = extract_residual_coverage_threshold(COVERAGE_QUERY)
    complement = extract_residual_coverage_threshold(QUERY.replace("0.2%", "0.5%"))

    assert coverage is not None and complement is not None
    # "more than 99.5% covered" == "less than 0.5% uncovered"
    assert coverage.max_uncovered_share == pytest.approx(0.005)
    assert complement.max_uncovered_share == pytest.approx(0.005)
    assert coverage.framing == "covered"
    assert complement.framing == "uncovered"


def test_explicit_import_threshold_wins_over_unrelated_high_percentage():
    from utils.residual_price import extract_residual_coverage_threshold

    query = (
        "The balancing price changed by more than 99.9% in one month. "
        "Prices of regulated and deregulated plants are known. Find months "
        "where the share of import in the balancing basket is less than 0.2% "
        "and calculate the weighted average PPA/CfD price."
    )

    threshold = extract_residual_coverage_threshold(query)

    assert threshold is not None
    assert threshold.framing == "uncovered"
    assert threshold.max_uncovered_share == pytest.approx(0.002)


def test_unrelated_percentage_is_not_a_residual_coverage_threshold():
    from utils.residual_price import extract_residual_coverage_threshold

    assert (
        extract_residual_coverage_threshold(
            "The balancing price changed by more than 99.9% year over year."
        )
        is None
    )


def test_threshold_authority_rejects_non_negligible_framings():
    from utils.residual_price import extract_residual_coverage_threshold

    # "more than 5% covered" => 95% uncovered: not a negligible-residual request.
    loose = extract_residual_coverage_threshold(COVERAGE_QUERY.replace("99.5%", "5%"))
    assert loose is None or loose.max_uncovered_share > 0.01
    # "less than 20% uncovered" likewise exceeds the negligible bound.
    wide = extract_residual_coverage_threshold(QUERY.replace("0.2%", "20%"))
    assert wide is None or wide.max_uncovered_share > 0.01


def test_import_share_filter_is_uncovered_side_for_coverage_phrasing():
    """The downstream filter must restrict import share DOWNWARD (<= 0.005),
    never `share_import > 0.995`, which would match zero months."""
    from utils.residual_price import resolve_import_share_filter

    operator, threshold, _phrase = resolve_import_share_filter(COVERAGE_QUERY)

    assert operator in {"lt", "le"}
    assert threshold == pytest.approx(0.005)


def test_import_share_filter_preserves_complement_side_semantics():
    from utils.residual_price import resolve_import_share_filter

    operator, threshold, _phrase = resolve_import_share_filter(QUERY)

    assert operator in {"lt", "le"}
    assert threshold == pytest.approx(0.002)


@pytest.mark.parametrize(
    "price_phrase",
    ["weighted average", "average price", "weighted avg", "avg price", "mean price"],
)
def test_price_intent_vocabulary_is_not_single_phrase(price_phrase):
    """Item 2: the price-intent trigger must not hinge on one exact phrase."""
    query = COVERAGE_QUERY.replace("weighted average price", f"{price_phrase} price of")
    assert is_implied_ppa_cfd_price_query(query)


# ---------------------------------------------------------------------------
# Item 3 — teach the CONTRACT, not just the regex.
#
# The three intents in summarizer._RESIDUAL_DIRECT_INTENTS were reachable only
# via the keyword gate: nothing in the analyzer prompt or catalogs mentioned
# them, so the LLM would have had to guess the exact string. Worse, the
# "unusual numeric calculation" rule told it to prefer ambiguous/clarify when
# the residual bucket looked underdefined — which is what produced
# query_type=ambiguous confidence=0.30 on a fully-specified question (prod
# traces span-fef3a3f9 / span-28451264). The prompt must name this capability
# and reserve clarify for genuinely underspecified requests.
# ---------------------------------------------------------------------------


def test_analyzer_prompt_teaches_implied_ppa_cfd_intent():
    import core.llm as llm_core

    rules = llm_core._ANALYZER_CORE_RULES
    assert "implied_ppa_cfd_price_approximation" in rules
    # It must pin the full deterministic contract, not just name the intent.
    assert "render_style=deterministic" in rules
    assert "get_prices" in rules


def test_analyzer_prompt_documents_both_threshold_framings():
    """The prompt must teach that the constraint can be phrased from either
    side, so the LLM recognizes the coverage-side wording too."""
    import core.llm as llm_core

    rules = llm_core._ANALYZER_CORE_RULES.lower()
    assert "99.5" in rules  # coverage-side example
    assert "more than" in rules and "less than" in rules


def test_analyzer_prompt_reserves_clarify_for_underspecified_residual():
    """Clarify must remain the answer when NO threshold is given, but must not
    swallow a request that states the threshold and which prices are known."""
    import core.llm as llm_core

    rules = llm_core._ANALYZER_CORE_RULES
    marker = "implied_ppa_cfd_price_approximation"
    idx = rules.find(marker)
    assert idx != -1
    # The clarify carve-out must be qualified by the absence of a threshold.
    window = rules[max(0, idx - 1200): idx + 1200].lower()
    assert "clarify" in window
    assert "no threshold" in window or "without a threshold" in window


def test_driver_enrichment_honours_analyzer_emitted_residual_intent():
    """Integration half of item 3: teaching the prompt is useless if the
    columns the deterministic answer needs are never fetched.

    The enrichment gate previously keyed only off the query regex plus
    comparison/driver flags. An analyzer-emitted residual intent on a
    data_retrieval question therefore skipped enrichment, leaving
    `residual_contribution_*` / `known_price_coverage_ok` absent so
    _build_residual_weighted_price_direct_answer returned None."""
    from types import SimpleNamespace

    import pandas as pd

    from agent import pipeline
    from contracts.question_analysis import AnswerKind, RenderStyle

    ctx = QueryContext(query="estimate the combined ppa and cfd price")
    ctx.df = pd.DataFrame({"date": ["2026-05-01"], "p_bal_gel": [137.0]})
    ctx.question_analysis = SimpleNamespace(
        answer_kind=AnswerKind.TIMESERIES,
        render_style=RenderStyle.DETERMINISTIC,
        classification=SimpleNamespace(
            intent="implied_ppa_cfd_price_approximation",
            query_type=SimpleNamespace(value="data_retrieval"),
        ),
        analysis_requirements=SimpleNamespace(
            needs_driver_analysis=False,
            needs_correlation_context=False,
        ),
    )
    ctx.question_analysis_source = "llm_active"
    invocation = SimpleNamespace(name="get_prices", params={"metric": "balancing"})

    assert pipeline._should_enrich_balancing_driver_context(
        ctx, invocation, is_explanation=False,
    ) is True


def test_driver_enrichment_ignores_shadow_analyzer_residual_intent():
    from types import SimpleNamespace

    import pandas as pd

    from agent import pipeline
    from contracts.question_analysis import AnswerKind, RenderStyle

    ctx = QueryContext(query="estimate the combined ppa and cfd price")
    ctx.df = pd.DataFrame({"date": ["2026-05-01"], "p_bal_gel": [137.0]})
    ctx.question_analysis = SimpleNamespace(
        answer_kind=AnswerKind.TIMESERIES,
        render_style=RenderStyle.DETERMINISTIC,
        classification=SimpleNamespace(
            intent="implied_ppa_cfd_price_approximation",
            query_type=SimpleNamespace(value="data_retrieval"),
        ),
        analysis_requirements=SimpleNamespace(
            needs_driver_analysis=False,
            needs_correlation_context=False,
        ),
    )
    ctx.question_analysis_source = "llm_shadow"
    invocation = SimpleNamespace(name="get_prices", params={"metric": "balancing"})

    assert pipeline._should_enrich_balancing_driver_context(
        ctx, invocation, is_explanation=False,
    ) is False
