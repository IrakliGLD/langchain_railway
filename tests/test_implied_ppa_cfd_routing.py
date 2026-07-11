"""Routing contract tests for the negligible-import PPA/CfD approximation."""

from __future__ import annotations

import os

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
