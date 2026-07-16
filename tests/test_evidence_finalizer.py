"""Tests for the P4.1 evidence-finalization routine (finding H1).

Covers the off/shadow/enforce modes, stale-frame invalidation, the typed
FinalizationResult/event log, and the skip paths (empty df, no tool, no
adapter). The keystone assertion: in ``enforce`` mode a *normal* tool result
attaches ``ctx.evidence_frame`` (which pre-P4.1 only happened on the recovery
path), while ``off``/``shadow`` leave normal-path attachment untouched.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from agent import evidence_finalizer  # noqa: E402
from agent.evidence_validator import EvidenceGap  # noqa: E402
from agent.provenance import stamp_provenance, tool_invocation_hash  # noqa: E402
from contracts.evidence_frames import ObservationFrame  # noqa: E402
from contracts.question_analysis import AnswerKind, QuestionAnalysis, RenderStyle  # noqa: E402
from models import QueryContext  # noqa: E402


def _qa_payload(answer_kind: str = "timeseries") -> dict:
    return {
        "version": "question_analysis_v1",
        "raw_query": "monthly balancing prices for 2024",
        "canonical_query_en": "monthly balancing prices for 2024",
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "data_retrieval",
            "analysis_mode": "light",
            "intent": "price series",
            "needs_clarification": False,
            "confidence": 0.9,
        },
        "routing": {
            "preferred_path": "tool",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": True,
            "needs_multi_tool": False,
            "evidence_roles": [],
        },
        "knowledge": {},
        "tooling": {"candidate_tools": [{"name": "get_prices", "score": 0.9, "reason": "price data"}]},
        "sql_hints": {},
        "visualization": {
            "chart_requested_by_user": False,
            "chart_recommended": False,
            "chart_confidence": 0.0,
        },
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
        "answer_kind": answer_kind,
    }


def _price_ctx(answer_kind: str = "timeseries", *, authoritative: bool = True) -> QueryContext:
    ctx = QueryContext(query="monthly balancing prices for 2024")
    df = pd.DataFrame(
        {"date": ["2024-01-01", "2024-02-01", "2024-03-01"], "p_bal_gel": [150.0, 160.0, 155.0]}
    )
    ctx.df = df
    ctx.cols = list(df.columns)
    ctx.rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    ctx.used_tool = True
    ctx.tool_name = "get_prices"
    ctx.tool_params = {"currency": "gel"}
    stamp_provenance(
        ctx, ctx.cols, ctx.rows, source="tool",
        query_hash=tool_invocation_hash("get_prices", {"currency": "gel"}),
    )
    if authoritative:
        ctx.question_analysis = QuestionAnalysis(**_qa_payload(answer_kind))
        ctx.question_analysis_source = "llm_active"
    return ctx


@pytest.fixture(autouse=True)
def _restore_mode():
    original = evidence_finalizer.EVIDENCE_FINALIZATION_MODE
    yield
    evidence_finalizer.EVIDENCE_FINALIZATION_MODE = original


def _set_mode(mode: str) -> None:
    evidence_finalizer.EVIDENCE_FINALIZATION_MODE = mode


# --- off mode ----------------------------------------------------------------

def test_off_mode_normal_call_is_noop():
    _set_mode("off")
    ctx = _price_ctx()
    result = evidence_finalizer.finalize_evidence(ctx, stage="stage_0_8_settled")
    assert result.action == evidence_finalizer.ACTION_SKIPPED_OFF
    assert ctx.evidence_frame is None
    assert ctx.evidence_frame_shadow is None
    # The event is still recorded for observability.
    assert ctx.evidence_finalization_events[-1]["action"] == "skipped_off"


def test_off_mode_legacy_attach_still_attaches():
    _set_mode("off")
    ctx = _price_ctx()
    result = evidence_finalizer.finalize_evidence(
        ctx, stage="recovery_apply_tool_result", legacy_attach=True,
    )
    assert result.action == evidence_finalizer.ACTION_ATTACHED
    assert isinstance(ctx.evidence_frame, ObservationFrame)


# --- shadow mode (default) ---------------------------------------------------

def test_shadow_mode_builds_shadow_frame_without_attaching():
    _set_mode("shadow")
    ctx = _price_ctx()
    result = evidence_finalizer.finalize_evidence(ctx, stage="stage_0_8_settled")
    assert result.action == evidence_finalizer.ACTION_SHADOW_BUILT
    assert result.frame_type == "ObservationFrame"
    # Behavior-neutral: the rendered frame is untouched.
    assert ctx.evidence_frame is None
    assert isinstance(ctx.evidence_frame_shadow, ObservationFrame)


def test_shadow_mode_does_not_degrade_render_style():
    _set_mode("shadow")
    ctx = _price_ctx()
    ctx.question_analysis.render_style = RenderStyle.DETERMINISTIC
    # Even if the (monkeypatched) validator reports a hard gap, shadow must not
    # mutate render_style.
    evidence_finalizer.validate_evidence = lambda *a, **k: EvidenceGap(
        AnswerKind.SCALAR, "bad shape", correctable=False,
    )
    try:
        evidence_finalizer.finalize_evidence(ctx, stage="stage_0_8_settled")
    finally:
        # Restore the real validator for other tests in the module.
        from agent.evidence_validator import validate_evidence as _real
        evidence_finalizer.validate_evidence = _real
    assert ctx.question_analysis.render_style == RenderStyle.DETERMINISTIC


# --- enforce mode ------------------------------------------------------------

def test_enforce_mode_attaches_frame_on_normal_path():
    _set_mode("enforce")
    ctx = _price_ctx()
    result = evidence_finalizer.finalize_evidence(ctx, stage="stage_0_8_settled")
    assert result.action == evidence_finalizer.ACTION_ATTACHED
    assert isinstance(ctx.evidence_frame, ObservationFrame)
    # Provenance flows from ctx.provenance_refs into the frame.
    assert ctx.evidence_frame.provenance_refs


def test_enforce_mode_holdback_builds_shadow_without_attaching():
    _set_mode("enforce")
    ctx = _price_ctx()
    ctx.p4_rollout_decisions["evidence_finalization"] = False
    result = evidence_finalizer.finalize_evidence(ctx, stage="stage_0_8_settled")
    assert result.mode == "shadow"
    assert result.action == evidence_finalizer.ACTION_SHADOW_BUILT
    assert ctx.evidence_frame is None
    assert isinstance(ctx.evidence_frame_shadow, ObservationFrame)


def test_enforce_mode_degrades_render_style_on_noncorrectable_gap(monkeypatch):
    _set_mode("enforce")
    ctx = _price_ctx()
    ctx.question_analysis.render_style = RenderStyle.DETERMINISTIC
    monkeypatch.setattr(
        evidence_finalizer, "validate_evidence",
        lambda *a, **k: EvidenceGap(AnswerKind.SCALAR, "wrong frame type", correctable=False),
    )
    evidence_finalizer.finalize_evidence(ctx, stage="stage_0_8_settled")
    assert ctx.question_analysis.render_style == RenderStyle.NARRATIVE
    assert ctx.evidence_gap is not None
    assert ctx.evidence_gap.correctable is False


def test_enforce_mode_keeps_render_style_on_correctable_gap(monkeypatch):
    _set_mode("enforce")
    ctx = _price_ctx()
    ctx.question_analysis.render_style = RenderStyle.DETERMINISTIC
    monkeypatch.setattr(
        evidence_finalizer, "validate_evidence",
        lambda *a, **k: EvidenceGap(AnswerKind.TIMESERIES, "not enough periods", correctable=True),
    )
    evidence_finalizer.finalize_evidence(ctx, stage="stage_0_8_settled")
    # Correctable gaps do not force narrative; downstream may re-plan/degrade.
    assert ctx.question_analysis.render_style == RenderStyle.DETERMINISTIC
    assert ctx.evidence_gap is not None
    assert ctx.evidence_gap.correctable is True


# --- skip paths --------------------------------------------------------------

def test_empty_df_is_skipped_and_invalidates_stale_frame():
    _set_mode("enforce")
    ctx = _price_ctx()
    # Pretend a stale frame is already attached.
    ctx.evidence_frame = ObservationFrame(rows=[{"period": "x", "value": 1.0}], provenance_refs=[])
    ctx.df = ctx.df.iloc[0:0]
    result = evidence_finalizer.finalize_evidence(ctx, stage="stage_0_8_settled")
    assert result.action == evidence_finalizer.ACTION_SKIPPED_EMPTY
    # Stale frame cleared.
    assert ctx.evidence_frame is None


def test_no_tool_identity_is_skipped():
    _set_mode("enforce")
    ctx = _price_ctx()
    ctx.tool_name = None
    result = evidence_finalizer.finalize_evidence(ctx, stage="stage_1_sql_fallback")
    assert result.action == evidence_finalizer.ACTION_SKIPPED_NO_TOOL
    assert ctx.evidence_frame is None


def test_unknown_tool_has_no_adapter():
    _set_mode("enforce")
    ctx = _price_ctx()
    ctx.tool_name = "get_moon_phase"
    result = evidence_finalizer.finalize_evidence(ctx, stage="stage_0_8_settled", tool_name="get_moon_phase")
    assert result.action == evidence_finalizer.ACTION_SKIPPED_NO_ADAPTER
    assert ctx.evidence_frame is None


# --- invalidation ------------------------------------------------------------

def test_invalidate_frame_clears_attached_and_shadow():
    _set_mode("shadow")
    ctx = _price_ctx()
    ctx.evidence_frame = ObservationFrame(rows=[{"period": "x", "value": 1.0}], provenance_refs=[])
    ctx.evidence_frame_shadow = ObservationFrame(rows=[{"period": "y", "value": 2.0}], provenance_refs=[])
    ctx.evidence_gap = EvidenceGap(AnswerKind.SCALAR, "x")
    result = evidence_finalizer.invalidate_frame(ctx, stage="stage_0_9_reanalysis", reason="reset")
    assert result is not None
    assert result.action == evidence_finalizer.ACTION_INVALIDATED
    assert ctx.evidence_frame is None
    assert ctx.evidence_frame_shadow is None
    assert ctx.evidence_gap is None


def test_invalidate_frame_noop_when_nothing_attached():
    ctx = _price_ctx()
    assert ctx.evidence_frame is None
    assert evidence_finalizer.invalidate_frame(ctx, stage="s", reason="r") is None


# --- typed result + event log ------------------------------------------------

def test_result_is_typed_and_event_logged():
    _set_mode("shadow")
    ctx = _price_ctx()
    result = evidence_finalizer.finalize_evidence(ctx, stage="stage_0_8_settled")
    assert result.has_frame is True
    event = result.as_event()
    assert event["stage"] == "stage_0_8_settled"
    assert event["mode"] == "shadow"
    assert event["frame_type"] == "ObservationFrame"
    # The event is appended to the per-request log.
    assert ctx.evidence_finalization_events[-1] == event


# --- never-raise wrappers ----------------------------------------------------

def test_safe_finalize_swallows_exceptions(monkeypatch):
    _set_mode("shadow")
    ctx = _price_ctx()
    monkeypatch.setattr(
        evidence_finalizer, "adapt_tool_result",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    # Must not propagate — observability cannot break the request.
    assert evidence_finalizer.safe_finalize(ctx, stage="stage_0_8_settled") is None
    assert ctx.evidence_frame is None


def test_safe_invalidate_swallows_exceptions(monkeypatch):
    ctx = _price_ctx()

    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(evidence_finalizer, "invalidate_frame", _boom)
    assert evidence_finalizer.safe_invalidate(ctx, stage="s", reason="r") is None
