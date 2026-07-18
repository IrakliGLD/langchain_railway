"""Deadline-first context transitions for query-pipeline stage sequencing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

ContextT = TypeVar("ContextT")


class TerminalStageResult(Protocol[ContextT]):
    ctx: ContextT
    terminal: bool


class PipelineStageOrchestrator:
    """Own the invariant shared by every deadline-aware pipeline stage."""

    def __init__(
        self,
        context: ContextT,
        *,
        require_budget: Callable[[ContextT, str], None],
    ) -> None:
        self._context = context
        self._require_budget = require_budget

    @property
    def context(self) -> ContextT:
        return self._context

    def adopt(self, context: ContextT) -> ContextT:
        """Adopt context returned by a nested sub-pipeline with its own budget checks."""
        self._context = context
        return self._context

    def run(
        self,
        stage_name: str,
        operation: Callable[[ContextT], ContextT],
    ) -> ContextT:
        self._require_budget(self._context, stage_name)
        self._context = operation(self._context)
        return self._context

    def run_effect(
        self,
        stage_name: str,
        operation: Callable[[ContextT], Any],
    ) -> None:
        self._require_budget(self._context, stage_name)
        operation(self._context)

    def run_terminal(
        self,
        stage_name: str,
        operation: Callable[[ContextT], TerminalStageResult[ContextT]],
    ) -> bool:
        self._require_budget(self._context, stage_name)
        result = operation(self._context)
        self._context = result.ctx
        return bool(result.terminal)
