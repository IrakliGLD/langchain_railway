"""Offline disagreement harness for the grounding shadow comparison (audit P1-7).

These fixtures are the "review disagreement cases, not just pass cases" artifact
the phased-audit skill requires before any cutover of the ratio->percent (x100)
grounding-token change. They assert that:

  * the seam is opt-in (default still emits x100; the flag suppresses it),
  * the candidate policy FIXES the target false-PASS (a raw [0,1] value in a
    non-ratio column no longer mints a bare percent token an answer can match),
  * the candidate policy does NOT regress the legitimate cases the x100 rule was
    built for (genuine share_* columns; ratios serialized into stats_hint).

Nothing here touches the live gate — `_is_summary_grounded` is unchanged.
"""

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import logging  # noqa: E402

from agent import summarizer  # noqa: E402
from agent.summary_grounding import build_grounding_tokens  # noqa: E402
from core.llm import SummaryEnvelope  # noqa: E402
from models import GroundingPolicy, QueryContext  # noqa: E402


def test_tokenize_seam_is_opt_in():
    # Default preserves the current behaviour (x100 for a [0,1] cell)...
    assert "85" in summarizer._tokenize_cell_value(0.85)
    # ...and the harness flag suppresses only the percent expansion.
    assert "85" not in summarizer._tokenize_cell_value(0.85, emit_ratio_percent=False)


def test_public_grounding_boundary_matches_summarizer_compatibility_export():
    ctx = QueryContext(query="price", cols=["value"], rows=[(12.5,)])

    assert build_grounding_tokens(ctx) == summarizer._build_grounding_tokens(ctx)


def test_is_ratio_column_recognises_share_and_ratio_names():
    assert summarizer._is_ratio_column("share_import")
    assert summarizer._is_ratio_column("import_dependency_ratio")
    assert summarizer._is_ratio_column("mom_percent_change")
    assert not summarizer._is_ratio_column("p_bal_gel")
    assert not summarizer._is_ratio_column("corr_coef")
    assert not summarizer._is_ratio_column("")


def test_shadow_flags_false_pass_on_non_ratio_cell():
    # A raw correlation-coefficient of 0.85 in a NON-ratio column currently mints
    # the percent token "85", so an ungrounded answer "85" false-PASSES today.
    ctx = QueryContext(
        query="What is the correlation?",
        cols=["month", "corr_coef"],
        rows=[("2024-01", 0.85)],
    )
    ctx.grounding_policy = GroundingPolicy.STRICT_NUMERIC
    envelope = SummaryEnvelope(
        answer="The headline figure is 85.",
        claims=[],
        citations=["data_preview"],
        confidence=0.9,
    )
    cmp = summarizer.compare_grounding_policies(envelope, ctx)
    assert cmp.current_passed is True  # x100 everywhere -> "85" matches
    assert cmp.candidate_passed is False  # corr_coef not a ratio col -> no "85"
    assert cmp.disagree is True
    assert cmp.divergent_tokens  # the removed percent token(s)


def test_shadow_preserves_legitimate_share_percentage():
    # share_import IS a ratio column, so "42%" from 0.42 stays grounded under both.
    ctx = QueryContext(
        query="What is import share?",
        cols=["month", "share_import"],
        rows=[("2024-01", 0.42)],
    )
    ctx.grounding_policy = GroundingPolicy.STRICT_NUMERIC
    envelope = SummaryEnvelope(
        answer="Import share reached 42%.",
        claims=[],
        citations=["data_preview"],
        confidence=0.9,
    )
    cmp = summarizer.compare_grounding_policies(envelope, ctx)
    assert cmp.current_passed is True
    assert cmp.candidate_passed is True
    assert cmp.disagree is False


def test_shadow_hook_logs_disagreement_without_changing_gate(monkeypatch, caplog):
    # With the shadow flag on, the live gate still returns its unchanged decision
    # (True here) AND logs the disagreement for later review.
    monkeypatch.setattr(summarizer, "_GROUNDING_SHADOW_LOG", True)
    ctx = QueryContext(
        query="What is the correlation?",
        cols=["month", "corr_coef"],
        rows=[("2024-01", 0.85)],
    )
    ctx.grounding_policy = GroundingPolicy.STRICT_NUMERIC
    envelope = SummaryEnvelope(
        answer="The headline figure is 85.",
        claims=[],
        citations=["data_preview"],
        confidence=0.9,
    )
    with caplog.at_level(logging.WARNING, logger="Enai"):
        assert summarizer._is_summary_grounded(envelope, ctx) is True  # gate UNCHANGED
    assert any("grounding shadow DISAGREE" in r.getMessage() for r in caplog.records)


def test_shadow_hook_silent_when_disabled(monkeypatch, caplog):
    monkeypatch.setattr(summarizer, "_GROUNDING_SHADOW_LOG", False)
    ctx = QueryContext(
        query="What is the correlation?",
        cols=["month", "corr_coef"],
        rows=[("2024-01", 0.85)],
    )
    ctx.grounding_policy = GroundingPolicy.STRICT_NUMERIC
    envelope = SummaryEnvelope(
        answer="The headline figure is 85.", claims=[], citations=["data_preview"], confidence=0.9
    )
    with caplog.at_level(logging.WARNING, logger="Enai"):
        assert summarizer._is_summary_grounded(envelope, ctx) is True
    assert not any("grounding shadow" in r.getMessage() for r in caplog.records)


def test_shadow_preserves_stats_hint_ratio_percentage():
    # A ratio serialized into stats_hint (0.0666 -> "6.66%") must stay grounded
    # under both policies: the candidate keeps x100 on the stats_hint text path.
    ctx = QueryContext(
        query="What is the growth rate?",
        stats_hint="CAGR ratio: 0.0666",
    )
    ctx.grounding_policy = GroundingPolicy.EVIDENCE_AWARE
    envelope = SummaryEnvelope(
        answer="The CAGR is about 6.66%.",
        claims=[],
        citations=["stats_hint"],
        confidence=0.9,
    )
    cmp = summarizer.compare_grounding_policies(envelope, ctx)
    assert cmp.current_passed is True
    assert cmp.candidate_passed is True
    assert cmp.disagree is False
