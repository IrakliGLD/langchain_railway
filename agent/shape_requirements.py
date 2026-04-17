"""Shared answer-shape requirements used by both validators.

Two modules enforce ``answer_kind`` contracts at different stages of the
pipeline:

* ``agent/evidence_planner.py`` (pre-execution) — checks the *plan* (tool
  params) can in principle produce the required shape.
* ``agent/evidence_validator.py`` (post-execution) — checks the collected
  *frame* actually has the required shape.

Keeping the raw thresholds in a single module prevents the two validators
from drifting as requirements evolve: raising a FORECAST's minimum from 3
periods to 4 now needs only a one-line edit here, and both sides pick it
up automatically.  Nothing here is behavioural; callers own the policy for
*how* to react to a missing requirement (warn vs reject vs repair).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from contracts.question_analysis import AnswerKind


@dataclass(frozen=True)
class ShapeRequirement:
    """Minimum evidence shape required to answer a question of a given kind.

    Fields are split by which stage consumes them:

    Post-execution (evidence_validator, checked against the canonical frame):
      * ``min_periods`` — minimum distinct time periods on an ObservationFrame.
      * ``min_entities`` — minimum distinct entities on the evidence frame.

    Pre-execution (evidence_planner, checked against the primary tool's
    resolved params):
      * ``requires_date_range`` — ``start_date`` and ``end_date`` must both
        be set on the primary step's params.
      * ``requires_multi_period_range`` — ``start_date != end_date`` (a
        single-day range is a SCALAR, not a TIMESERIES).
      * ``requires_scenario_metric`` — ``analysis_requirements`` must
        include at least one scenario-family ``DerivedMetricRequest``.

    Semantic:
      * ``accepts_multi_period_or_entity`` — the COMPARISON rule: either
        ≥2 periods OR ≥2 entities satisfies the shape.  Both validators
        need to implement the disjunction so it is surfaced as a single
        flag rather than as two thresholds that imply AND.
    """

    min_periods: int = 0
    min_entities: int = 0
    requires_date_range: bool = False
    requires_multi_period_range: bool = False
    requires_scenario_metric: bool = False
    accepts_multi_period_or_entity: bool = False


SHAPE_REQUIREMENTS: Mapping[AnswerKind, ShapeRequirement] = {
    AnswerKind.SCALAR: ShapeRequirement(),
    AnswerKind.LIST: ShapeRequirement(min_entities=1),
    AnswerKind.TIMESERIES: ShapeRequirement(
        min_periods=2,
        requires_date_range=True,
        requires_multi_period_range=True,
    ),
    AnswerKind.COMPARISON: ShapeRequirement(
        accepts_multi_period_or_entity=True,
    ),
    AnswerKind.FORECAST: ShapeRequirement(
        min_periods=3,
        requires_date_range=True,
    ),
    AnswerKind.SCENARIO: ShapeRequirement(
        requires_scenario_metric=True,
    ),
    # EXPLANATION, KNOWLEDGE, CLARIFY — no structural requirements.
}


_DEFAULT_REQUIREMENT = ShapeRequirement()


def get_requirement(answer_kind: AnswerKind | None) -> ShapeRequirement:
    """Return the shape requirement for an answer_kind (permissive default)."""
    if answer_kind is None:
        return _DEFAULT_REQUIREMENT
    return SHAPE_REQUIREMENTS.get(answer_kind, _DEFAULT_REQUIREMENT)
