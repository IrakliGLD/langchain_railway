"""Evidence validation against answer_kind requirements.

Called during evidence collection to catch gaps early rather than
discovering them as Stage 4 failures.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from agent.shape_requirements import get_requirement
from contracts.evidence_frames import (
    CanonicalFrame,
    ComparisonFrame,
    EntitySetFrame,
    ObservationFrame,
)
from contracts.question_analysis import AnswerKind

log = logging.getLogger("Enai")


class EvidenceGap:
    """Describes a mismatch between evidence and answer_kind requirements."""

    def __init__(self, answer_kind: AnswerKind, reason: str, *, correctable: bool = False):
        self.answer_kind = answer_kind
        self.reason = reason
        self.correctable = correctable

    def __repr__(self) -> str:
        return f"EvidenceGap({self.answer_kind.value}: {self.reason}, correctable={self.correctable})"


def validate_evidence(
    frame: Optional[CanonicalFrame],
    answer_kind: Optional[AnswerKind],
) -> EvidenceGap | None:
    """Check whether the evidence frame satisfies the answer_kind requirements.

    Returns None if validation passes, or an EvidenceGap describing the problem.
    """
    if answer_kind is None:
        return None  # No contract to validate against.

    if frame is None or frame.is_empty():
        return EvidenceGap(
            answer_kind,
            "No evidence collected",
            correctable=True,  # Planner can try a different tool.
        )

    if answer_kind == AnswerKind.SCALAR:
        return _validate_scalar(frame)
    if answer_kind == AnswerKind.LIST:
        return _validate_list(frame)
    if answer_kind == AnswerKind.TIMESERIES:
        return _validate_timeseries(frame)
    if answer_kind == AnswerKind.COMPARISON:
        return _validate_comparison(frame)
    if answer_kind == AnswerKind.FORECAST:
        return _validate_forecast(frame)

    # EXPLANATION, SCENARIO, KNOWLEDGE, CLARIFY — no structural requirements.
    return None


def validate_analysis_requirements(ctx) -> list[str]:
    """Validate analyzer-requested analytical context after Stage 3 enrichment.

    This is deliberately diagnostic-only. It gives operators a structured
    disagreement signal before any future enforcement decision changes answers.
    """

    if not getattr(ctx, "has_authoritative_question_analysis", False):
        return []
    analysis = ctx.question_analysis
    if analysis is None:
        return []
    requirements = analysis.analysis_requirements
    gaps: list[str] = []
    frame = getattr(ctx, "df", None)
    columns = list(getattr(frame, "columns", []))

    if requirements.needs_trend_context:
        time_col = next(
            (
                column
                for column in columns
                if any(token in str(column).lower() for token in ("date", "month", "year", "period"))
            ),
            None,
        )
        period_count = 0
        if time_col is not None and frame is not None and not frame.empty:
            periods = pd.to_datetime(frame[time_col], errors="coerce").dropna()
            period_count = int(periods.nunique())
        if period_count < 2:
            gaps.append("trend_context_missing")

    if requirements.needs_driver_analysis:
        has_driver_columns = any(
            str(column).lower().startswith(("share_", "contribution_", "tariff_"))
            or str(column).lower() == "xrate"
            for column in columns
        )
        has_driver_artifact = bool(getattr(ctx, "analysis_evidence", None)) or (
            "CAUSAL CONTEXT" in str(getattr(ctx, "stats_hint", "") or "")
        )
        if not has_driver_columns and not has_driver_artifact:
            gaps.append("driver_context_missing")

    if requirements.needs_correlation_context:
        has_correlations = bool(getattr(ctx, "correlation_results", None)) or (
            "CORRELATION MATRIX" in str(getattr(ctx, "stats_hint", "") or "")
        )
        if not has_correlations:
            gaps.append("correlation_context_missing")

    return gaps


def _validate_scalar(frame: CanonicalFrame) -> EvidenceGap | None:
    if not isinstance(frame, ObservationFrame):
        return EvidenceGap(AnswerKind.SCALAR, f"Expected ObservationFrame, got {type(frame).__name__}")
    # Scalar needs at least one value.
    has_value = any(r.get("value") is not None for r in frame.rows)
    if not has_value:
        return EvidenceGap(AnswerKind.SCALAR, "No non-null values in frame", correctable=True)
    return None


def _validate_list(frame: CanonicalFrame) -> EvidenceGap | None:
    if isinstance(frame, EntitySetFrame):
        if len(frame.rows) == 0:
            return EvidenceGap(AnswerKind.LIST, "EntitySetFrame is empty", correctable=True)
        return None
    if isinstance(frame, ObservationFrame):
        if len(frame.entities) == 0:
            return EvidenceGap(AnswerKind.LIST, "ObservationFrame has no entities for list", correctable=True)
        return None
    return EvidenceGap(AnswerKind.LIST, f"Unexpected frame type {type(frame).__name__}")


def _validate_timeseries(frame: CanonicalFrame) -> EvidenceGap | None:
    if not isinstance(frame, ObservationFrame):
        return EvidenceGap(AnswerKind.TIMESERIES, f"Expected ObservationFrame, got {type(frame).__name__}")
    min_periods = get_requirement(AnswerKind.TIMESERIES).min_periods
    if len(frame.periods) < min_periods:
        return EvidenceGap(
            AnswerKind.TIMESERIES,
            f"Only {len(frame.periods)} period(s) — need at least {min_periods} for timeseries",
            correctable=True,
        )
    return None


def _validate_comparison(frame: CanonicalFrame) -> EvidenceGap | None:
    if isinstance(frame, ComparisonFrame):
        if len(frame.rows) == 0:
            return EvidenceGap(AnswerKind.COMPARISON, "ComparisonFrame is empty", correctable=True)
        return None
    if isinstance(frame, ObservationFrame):
        # COMPARISON shape: ≥2 periods OR ≥2 entities (see ShapeRequirement
        # ``accepts_multi_period_or_entity``).
        if len(frame.periods) < 2 and len(frame.entities) < 2:
            return EvidenceGap(
                AnswerKind.COMPARISON,
                "Need at least 2 periods or 2 entities for comparison",
                correctable=True,
            )
        return None
    return EvidenceGap(AnswerKind.COMPARISON, f"Unexpected frame type {type(frame).__name__}")


def _validate_forecast(frame: CanonicalFrame) -> EvidenceGap | None:
    if not isinstance(frame, ObservationFrame):
        return EvidenceGap(AnswerKind.FORECAST, f"Expected ObservationFrame, got {type(frame).__name__}")
    # Forecast needs enough history for trend calculation (see ShapeRequirement).
    min_periods = get_requirement(AnswerKind.FORECAST).min_periods
    if len(frame.periods) < min_periods:
        return EvidenceGap(
            AnswerKind.FORECAST,
            f"Only {len(frame.periods)} period(s) — need at least {min_periods} for forecast trend",
            correctable=True,
        )
    return None
