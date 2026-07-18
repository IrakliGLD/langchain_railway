"""Characterization tests for deadline-aware pipeline stage sequencing."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.stage_orchestrator import PipelineStageOrchestrator


@dataclass
class _Context:
    value: int = 0


@dataclass
class _TerminalResult:
    ctx: _Context
    terminal: bool


def test_stage_budget_is_checked_before_exactly_one_context_transition():
    calls = []

    def require_budget(ctx, stage):
        calls.append(("budget", stage, ctx.value))

    def stage(ctx):
        calls.append(("stage", ctx.value))
        return _Context(ctx.value + 1)

    orchestrator = PipelineStageOrchestrator(_Context(), require_budget=require_budget)

    result = orchestrator.run("stage_1", stage)

    assert result == _Context(1)
    assert orchestrator.context == _Context(1)
    assert calls == [("budget", "stage_1", 0), ("stage", 0)]


def test_budget_failure_prevents_stage_side_effects_and_context_adoption():
    calls = []

    def require_budget(_ctx, _stage):
        raise TimeoutError("expired")

    orchestrator = PipelineStageOrchestrator(
        _Context(),
        require_budget=require_budget,
    )

    with pytest.raises(TimeoutError, match="expired"):
        orchestrator.run("stage_1", lambda ctx: calls.append(ctx) or _Context(1))

    assert calls == []
    assert orchestrator.context == _Context(0)


@pytest.mark.parametrize("terminal", [False, True])
def test_terminal_stage_adopts_context_and_returns_only_terminal_decision(terminal):
    calls = []
    orchestrator = PipelineStageOrchestrator(
        _Context(),
        require_budget=lambda ctx, stage: calls.append((stage, ctx.value)),
    )

    should_stop = orchestrator.run_terminal(
        "stage_terminal",
        lambda ctx: _TerminalResult(_Context(ctx.value + 2), terminal),
    )

    assert should_stop is terminal
    assert orchestrator.context == _Context(2)
    assert calls == [("stage_terminal", 0)]


def test_effect_stage_preserves_context_identity_and_call_order():
    context = _Context()
    calls = []
    orchestrator = PipelineStageOrchestrator(
        context,
        require_budget=lambda ctx, stage: calls.append(("budget", stage, ctx is context)),
    )

    orchestrator.run_effect(
        "stage_effect",
        lambda ctx: calls.append(("effect", ctx is context)),
    )

    assert orchestrator.context is context
    assert calls == [
        ("budget", "stage_effect", True),
        ("effect", True),
    ]


def test_adopt_tracks_context_from_nested_pipeline_without_duplicate_budget_check():
    calls = []
    orchestrator = PipelineStageOrchestrator(
        _Context(),
        require_budget=lambda ctx, stage: calls.append((stage, ctx.value)),
    )

    result = orchestrator.adopt(_Context(7))

    assert result == _Context(7)
    assert orchestrator.context == _Context(7)
    assert calls == []
