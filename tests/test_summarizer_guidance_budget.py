"""Guidance has to obey a budget, like every other part of the prompt.

Incident 2026-08-17. Between the request that answered and the two that timed
out, only one prompt section grew: guidance, 15,723 -> 33,385 chars, when the
make-or-buy retail path pinned render_style to narrative and pulled in
``retail-tariff-rules.md`` (15.5 KB) alongside the energy-analyst blocks. The
resulting 88k prompt could not finish inside the request budget.

Guidance sits in the prompt *suffix*, and ``_section_aware_truncate`` only ever
drops ``UNTRUSTED_*`` sections, so the one part of the prompt that grew was the
one part that could not be trimmed. The budget could therefore only ever cut
evidence -- backwards. These tests pin a guidance budget with an explicit
shedding order: reference material goes before the rules that govern the answer.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import config  # noqa: E402
import core.llm as llm_core  # noqa: E402


def test_a_guidance_budget_is_configured():
    """Unbounded assembly is what let guidance double between two requests."""
    assert config.SUMMARIZER_GUIDANCE_MAX_CHARS > 0
    assert config.SUMMARIZER_GUIDANCE_MAX_CHARS <= 30000


def test_guidance_under_budget_is_untouched():
    parts = ["FOCUS RULES:\nkeep this", "RETAIL TARIFF RULES:\nand this"]

    packed = llm_core._pack_guidance(parts, budget=10_000)

    assert packed == "FOCUS RULES:\nkeep this\n\nRETAIL TARIFF RULES:\nand this"


def test_generic_background_is_shed_first():
    """Seasonal rules and the entity taxonomy are background on every query."""
    focus = "FOCUS RULES:\n" + ("f" * 3000)
    retail = "RETAIL TARIFF RULES:\n" + ("r" * 4000)
    seasonal = "SEASONAL DOMAIN RULES:\n" + ("s" * 1800)
    taxonomy = "ENTITY TAXONOMY:\n" + ("t" * 1800)

    packed = llm_core._pack_guidance(
        [focus, retail, seasonal, taxonomy], budget=8_000
    )

    assert "SEASONAL DOMAIN RULES:" not in packed
    assert "ENTITY TAXONOMY:" not in packed
    assert "FOCUS RULES:" in packed
    assert "RETAIL TARIFF RULES:" in packed
    assert len(packed) <= 8_000


def test_frame_specific_rules_are_trimmed_rather_than_dropped():
    """Retail rules are loaded only when a retail frame is present.

    Dropping them whole on a retail question trades a timeout for a wrong
    answer about the tariff stack -- which is the thing they exist to prevent.
    A truncated block still carries its headline rules; an absent one carries
    nothing.
    """
    focus = "FOCUS RULES:\n" + ("f" * 1800)
    balancing = "# Balancing Analysis Template\n" + ("b" * 8100)
    retail = "RETAIL TARIFF RULES:\n" + ("r" * 15400)
    taxonomy = "ENTITY TAXONOMY:\n" + ("t" * 1800)

    packed = llm_core._pack_guidance(
        [focus, balancing, retail, taxonomy], budget=20_000
    )

    assert "RETAIL TARIFF RULES:" in packed
    assert "FOCUS RULES:" in packed
    assert len(packed) <= 20_000


def test_packing_preserves_assembly_order_for_what_survives():
    """Shedding must not reshuffle the rules the model still reads."""
    focus = "FOCUS RULES:\n" + ("f" * 1000)
    structure = "ANSWER STRUCTURE FOR THIS QUERY:\n" + ("s" * 1000)
    retail = "RETAIL TARIFF RULES:\n" + ("r" * 15000)

    packed = llm_core._pack_guidance([focus, structure, retail], budget=3_000)

    assert packed.index("FOCUS RULES:") < packed.index("ANSWER STRUCTURE FOR THIS QUERY:")


def test_a_single_oversized_block_is_truncated_not_dropped_silently():
    """One block bigger than the whole budget still has to say something."""
    packed = llm_core._pack_guidance(
        ["FOCUS RULES:\n" + ("f" * 40000)], budget=5_000
    )

    assert "FOCUS RULES:" in packed
    assert len(packed) <= 5_000


class _DummyCache:
    def get(self, _key):
        return None

    def set(self, _key, _value):
        return None


class _DummyMessage:
    content = '{"answer":"ok","claims":[],"citations":[],"confidence":0.9}'
    response_metadata: dict = {}


def test_assembled_guidance_never_exceeds_the_budget(monkeypatch):
    """End to end: the census figure that blew up must now be bounded."""
    captured: dict = {}

    monkeypatch.setattr(llm_core, "llm_cache", _DummyCache())
    monkeypatch.setattr(llm_core, "get_llm_for_stage", lambda *_a, **_k: object())
    monkeypatch.setattr(llm_core, "_log_usage_for_message", lambda *_a, **_k: None)
    monkeypatch.setattr(llm_core.metrics, "log_llm_call", lambda *_a, **_k: None)
    monkeypatch.setattr(llm_core, "SUMMARIZER_GUIDANCE_MAX_CHARS", 6_000)

    def _capture(_llm, messages, *_args, **_kwargs):
        captured["prompt"] = messages[1][1]
        return _DummyMessage()

    monkeypatch.setattr(llm_core, "_invoke_at_stage", _capture)

    # A retail frame plus a balancing focus: the exact combination that pulled
    # retail-tariff-rules.md, seasonal-rules.md and entity-taxonomy.md together.
    llm_core.llm_summarize_structured(
        user_query="Is retail cheaper than wholesale for a 35-100 kV customer?",
        data_preview="date,final_price_net_gel_kwh\n2026-07-01,0.2013",
        stats_hint=(
            "--- ANNUAL MAKE-OR-BUY COMPARISON ---\n"
            "final_price_net_gel_kwh vs p_bal_gel\n"
            "balancing price context for a retail comparison"
        ),
    )

    census = llm_core._summarizer_prompt_census(captured["prompt"])
    assert census["guidance_chars"] <= 6_000
