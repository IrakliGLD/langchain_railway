"""One evidence finalization routine (P4.1, finding H1).

Before this module, canonical evidence frames were attached only on the
analyzer *recovery* path (``_apply_tool_result``): normal successful tool
execution never built a frame, never ran ``validate_evidence``, and therefore
could not reach the generic deterministic renderer. This module is the single
routine called after every step that produces or mutates tabular evidence:

- normal primary tool execution (``_execute_evidence_step`` with
  ``is_primary=True``),
- analyzer tool recovery (``_apply_tool_result``),
- the Stage 0.8 secondary-evidence merge (``merge_evidence_into_context``),
- balancing driver-context enrichment, and
- Stage 3 analyzer enrichment.

Rollout is governed by ``ENAI_EVIDENCE_FINALIZATION_MODE``:

- ``off``     — new call sites do nothing; only the legacy recovery-path
                attachment (``legacy_attach=True``) still attaches frames.
                Byte-identical to pre-P4.1 behavior.
- ``shadow``  — every call site builds the frame and runs validation, but the
                result is stored on ``ctx.evidence_frame_shadow`` and emitted
                as telemetry only. ``ctx.evidence_frame`` / ``ctx.evidence_gap``
                / ``render_style`` are untouched except on the legacy recovery
                path. Behavior-neutral; produces the comparison telemetry the
                P4 exit gate requires. This is the default.
- ``enforce`` — every call site attaches ``ctx.evidence_frame`` (superseding
                any earlier frame), binds provenance, stores the typed gap,
                and applies the not-correctable → NARRATIVE degrade. The
                generic renderer becomes reachable on the normal path (H1).

Stale-frame invalidation (``invalidate_frame``) is active in **all** modes:
whenever the pipeline clears or replaces its evidence (relevance block, tool
failure, re-analysis reset), any previously attached frame is removed so a
later render can never use evidence the pipeline already discarded. That is a
correctness fix, not a rollout-gated behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from agent.evidence_validator import EvidenceGap, validate_evidence
from agent.frame_adapters import adapt_tool_result
from agent.p4_rollout import GATE_EVIDENCE_FINALIZATION, gate_is_active
from config import EVIDENCE_FINALIZATION_MODE as _CONFIG_MODE
from contracts.question_analysis import RenderStyle
from utils.metrics import metrics
from utils.trace_logging import trace_detail

if TYPE_CHECKING:  # pragma: no cover - typing only
    from contracts.evidence_frames import CanonicalFrame
    from models import QueryContext

log = logging.getLogger("Enai")

# Module-level so tests monkeypatch ``agent.evidence_finalizer.EVIDENCE_FINALIZATION_MODE``
# (same convention as agent.pipeline.ENABLE_EVIDENCE_PLANNER).
EVIDENCE_FINALIZATION_MODE = _CONFIG_MODE

# Finalization actions (typed vocabulary for events/metrics).
ACTION_ATTACHED = "attached"                # frame bound to ctx.evidence_frame
ACTION_SHADOW_BUILT = "shadow_built"        # frame built for telemetry only
ACTION_SKIPPED_OFF = "skipped_off"          # mode=off and not a legacy site
ACTION_SKIPPED_EMPTY = "skipped_empty"      # no rows to frame
ACTION_SKIPPED_NO_TOOL = "skipped_no_tool"  # evidence has no tool identity (SQL fallback)
ACTION_SKIPPED_NO_ADAPTER = "skipped_no_adapter"  # tool has no frame adapter
ACTION_INVALIDATED = "invalidated"          # stale frame cleared


def _effective_mode(ctx: "QueryContext") -> str:
    """Resolve enforce -> enforce/shadow for this request's stable cohort."""
    if EVIDENCE_FINALIZATION_MODE != "enforce":
        return EVIDENCE_FINALIZATION_MODE
    return (
        "enforce"
        if gate_is_active(ctx, GATE_EVIDENCE_FINALIZATION, default=True)
        else "shadow"
    )


@dataclass
class FinalizationResult:
    """Typed outcome of one finalization call (P4.1 acceptance artifact)."""

    stage: str
    mode: str
    action: str
    frame_type: str = ""
    frame_rows: int = 0
    gap_reason: str = ""
    gap_correctable: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_frame(self) -> bool:
        return self.action in (ACTION_ATTACHED, ACTION_SHADOW_BUILT)

    def as_event(self) -> Dict[str, Any]:
        event = {
            "stage": self.stage,
            "mode": self.mode,
            "action": self.action,
        }
        if self.frame_type:
            event["frame_type"] = self.frame_type
            event["frame_rows"] = self.frame_rows
        if self.gap_reason:
            event["gap_reason"] = self.gap_reason
            event["gap_correctable"] = self.gap_correctable
        if self.detail:
            event.update(self.detail)
        return event


def _record(ctx: "QueryContext", result: FinalizationResult) -> FinalizationResult:
    """Append the event to ctx, bump the counter, and emit a trace detail."""
    try:
        ctx.evidence_finalization_events.append(result.as_event())
    except AttributeError:
        # Foreign/minimal ctx objects in unit tests may lack the field.
        pass
    metrics.log_evidence_finalization(result.stage, result.action)
    trace_detail(
        log, ctx, "evidence_finalization", result.action,
        checkpoint=result.stage,
        mode=result.mode,
        frame_type=result.frame_type,
        frame_rows=result.frame_rows,
        gap_reason=result.gap_reason,
    )
    return result


def _resolve_contract(ctx: "QueryContext", tool_name: str):
    """Extract answer_kind + the matching tool candidate's filter condition."""
    answer_kind = None
    filter_cond = None
    if ctx.has_authoritative_question_analysis:
        qa = ctx.question_analysis
        answer_kind = qa.answer_kind
        for tc in qa.tooling.candidate_tools:
            if tc.name.value == tool_name and tc.params_hint is not None:
                filter_cond = tc.params_hint.filter
                break
    return answer_kind, filter_cond


def invalidate_frame(ctx: "QueryContext", *, stage: str, reason: str) -> Optional[FinalizationResult]:
    """Clear any attached/shadow frame because the evidence it framed is gone.

    Active in every mode: a cleared or replaced evidence set must never leave
    a stale canonical frame behind for Stage 4 to render.
    Returns a FinalizationResult when something was actually cleared, else None.
    """
    had_frame = getattr(ctx, "evidence_frame", None) is not None
    had_shadow = getattr(ctx, "evidence_frame_shadow", None) is not None
    if not had_frame and not had_shadow:
        return None
    ctx.evidence_frame = None
    ctx.evidence_gap = None
    ctx.evidence_frame_shadow = None
    ctx.evidence_frame_stage = ""
    result = FinalizationResult(
        stage=stage,
        mode=_effective_mode(ctx),
        action=ACTION_INVALIDATED,
        detail={"reason": reason, "had_attached": had_frame, "had_shadow": had_shadow},
    )
    return _record(ctx, result)


def finalize_evidence(
    ctx: "QueryContext",
    *,
    stage: str,
    tool_name: Optional[str] = None,
    tool_params: Optional[Dict[str, Any]] = None,
    legacy_attach: bool = False,
) -> FinalizationResult:
    """Build/refresh the canonical evidence frame from the current ctx state.

    Args:
        ctx: The query context whose ``df`` holds the evidence to frame.
        stage: Checkpoint label ("stage_0_5_primary", "recovery_apply_tool_result",
            "stage_0_8_merge", "stage_0_8_enrichment", "stage_3_enrich").
        tool_name/tool_params: Tool identity of the evidence. Default to
            ``ctx.tool_name`` / ``ctx.tool_params`` (post-merge checkpoints
            re-frame whatever tool currently owns ctx.df).
        legacy_attach: True only for the pre-P4.1 recovery call site, which
            attached frames in production before this module existed. It keeps
            attaching in every mode so ``off``/``shadow`` remain behavior-
            identical to the deployed pipeline.

    The returned FinalizationResult is also appended (as a dict event) to
    ``ctx.evidence_finalization_events`` and counted in metrics.
    """
    mode = _effective_mode(ctx)
    resolved_tool = (tool_name or ctx.tool_name or "").strip()

    if mode == "off" and not legacy_attach:
        return _record(ctx, FinalizationResult(stage=stage, mode=mode, action=ACTION_SKIPPED_OFF))

    if ctx.df is None or ctx.df.empty:
        # No evidence rows: any lingering frame is stale by definition.
        invalidate_frame(ctx, stage=stage, reason="empty_evidence")
        return _record(ctx, FinalizationResult(stage=stage, mode=mode, action=ACTION_SKIPPED_EMPTY))

    if not resolved_tool:
        # SQL-fallback / untyped evidence has no adapter vocabulary yet.
        # Recorded (not silent) so shadow telemetry sizes the uncovered share.
        return _record(ctx, FinalizationResult(stage=stage, mode=mode, action=ACTION_SKIPPED_NO_TOOL))

    answer_kind, filter_cond = _resolve_contract(ctx, resolved_tool)
    prov_refs = list(ctx.provenance_refs)

    frame: Optional["CanonicalFrame"] = adapt_tool_result(
        tool_name=resolved_tool,
        df=ctx.df,
        provenance_refs=prov_refs,
        filter_cond=filter_cond,
        answer_kind=answer_kind,
    )
    if frame is None:
        return _record(
            ctx,
            FinalizationResult(
                stage=stage, mode=mode, action=ACTION_SKIPPED_NO_ADAPTER,
                detail={"tool": resolved_tool},
            ),
        )

    gap: Optional[EvidenceGap] = validate_evidence(frame, answer_kind)

    attach = legacy_attach or mode == "enforce"
    result = FinalizationResult(
        stage=stage,
        mode=mode,
        action=ACTION_ATTACHED if attach else ACTION_SHADOW_BUILT,
        frame_type=type(frame).__name__,
        frame_rows=len(frame.rows),
        gap_reason=(gap.reason if gap is not None else ""),
        gap_correctable=(bool(gap.correctable) if gap is not None else False),
        detail={"tool": resolved_tool},
    )

    if attach:
        ctx.evidence_frame = frame
        ctx.evidence_gap = gap
        ctx.evidence_frame_stage = stage
        log.info(
            "Finalized canonical evidence frame: type=%s rows=%d (tool=%s stage=%s mode=%s)",
            type(frame).__name__, len(frame.rows), resolved_tool, stage, mode,
        )
        if gap is not None and not gap.correctable:
            # Same degrade the recovery path applied pre-P4.1: an evidence
            # shape that cannot satisfy the contract routes to LLM narrative.
            log.warning(
                "Evidence gap (not correctable): %s — degrading render_style to narrative", gap,
            )
            if ctx.has_authoritative_question_analysis:
                ctx.question_analysis.render_style = RenderStyle.NARRATIVE
        elif gap is not None:
            log.warning("Evidence gap (correctable): %s — downstream may re-plan or degrade", gap)
    else:
        ctx.evidence_frame_shadow = frame
        ctx.evidence_frame_stage = stage

    return _record(ctx, result)


def safe_finalize(ctx: "QueryContext", *, stage: str, **kwargs) -> Optional[FinalizationResult]:
    """finalize_evidence wrapper that can never break the request.

    In the default ``shadow`` mode finalization runs on every request purely for
    telemetry, so a defect in frame-building or validation must degrade to "no
    frame", never propagate. In ``enforce`` mode a swallowed error simply leaves
    ``ctx.evidence_frame`` unset, which routes Stage 4 to the LLM narrative path
    — the safe fallback. Mirrors the never-raise contract of the render-fitness
    and fixture-candidate observability hooks.
    """
    try:
        return finalize_evidence(ctx, stage=stage, **kwargs)
    except Exception:  # noqa: BLE001 - observability must not break the pipeline
        log.debug("evidence finalization failed (stage=%s)", stage, exc_info=True)
        return None


def safe_invalidate(ctx: "QueryContext", *, stage: str, reason: str) -> Optional[FinalizationResult]:
    """invalidate_frame wrapper that can never break the request."""
    try:
        return invalidate_frame(ctx, stage=stage, reason=reason)
    except Exception:  # noqa: BLE001 - observability must not break the pipeline
        log.debug("evidence frame invalidation failed (stage=%s)", stage, exc_info=True)
        return None
