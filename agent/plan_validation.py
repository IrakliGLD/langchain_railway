"""Typed evidence-plan validation against the finalized answer contract."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent.shape_requirements import get_requirement
from contracts.question_analysis import (
    _SCENARIO_METRIC_NAMES,
    AnswerKind,
    EvidenceRole,
    QuestionAnalysis,
    RenderStyle,
    ToolName,
)

log = logging.getLogger("Enai")

# ---------------------------------------------------------------------------
# Plan validation against answer_kind (P4.2, finding M3)
# ---------------------------------------------------------------------------

# Severity vocabulary for plan-validation issues.
SEVERITY_WARN = "warn"      # advisory: execution can still legitimately satisfy the contract
SEVERITY_REJECT = "reject"  # the plan provably cannot satisfy the answer contract


@dataclass
class PlanValidationIssue:
    """One typed finding from plan validation."""

    rule: str
    severity: str
    message: str


@dataclass
class PlanValidationResult:
    """Typed outcome of validating an evidence plan against the answer contract.

    ``rejects`` are issues where the planned evidence provably cannot satisfy
    the contract shape; ``warnings`` are advisory signals — execution may
    still succeed via tool defaults, post-execution evidence validation, or
    the generic renderer's own opt-out. Deterministic repairs (currently the
    LIST-tariff entity expansion in ``_repair_list_tariff_entities``) run
    during plan expansion, BEFORE validation — so a repaired plan is what
    gets validated, and it is validated exactly once.
    """

    issues: List[PlanValidationIssue] = field(default_factory=list)

    def add(self, rule: str, severity: str, message: str) -> None:
        self.issues.append(PlanValidationIssue(rule=rule, severity=severity, message=message))

    @property
    def rejects(self) -> List[PlanValidationIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_REJECT]

    @property
    def warnings(self) -> List[PlanValidationIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_WARN]

    def as_event(self) -> Dict[str, Any]:
        return {
            "issues": [
                {"rule": issue.rule, "severity": issue.severity, "message": issue.message}
                for issue in self.issues
            ],
        }


def validate_plan_against_answer_kind(
    steps: List[Dict[str, Any]],
    qa: QuestionAnalysis,
    raw_query: str,
) -> PlanValidationResult:
    """Check that planned evidence steps can produce the expected answer shape.

    Returns a typed :class:`PlanValidationResult` and keeps emitting the same
    warning log lines. This function never mutates or rejects steps itself —
    enforcement of reject-severity issues is the pipeline's job, gated by
    ``ENAI_PLAN_VALIDATION_MODE`` (default ``warn`` = observe only, matching
    the previously documented warning-only behavior).

    Severity policy: a condition is ``reject`` only when the planned evidence
    provably cannot satisfy the contract (a COMPARISON that explicitly pins a
    single period for a single comparand; a SCENARIO with nothing to
    compute). Missing-date-range and enumeration conditions stay ``warn``
    because tools legitimately default to recent-history windows and
    post-execution evidence validation covers the shape (a plan-time reject
    there would break "recent prices"-style queries that succeed today).

    Rules checked:
    - COMPARISON → multiple evidence sources, multi-entity primary params,
      an inherently multi-entity tool, OR a primary step whose date range
      spans more than one period (e.g. "Jan vs Feb" fetched as a single
      get_prices call with start_date=2025-01-01, end_date=2025-02-28).
    - TIMESERIES → primary step has a non-degenerate date range
      (start_date + end_date, AND start_date != end_date so the series
      contains more than a single point).
    - FORECAST → primary step has a date range (historical basis for
      trendline extrapolation).
    - SCENARIO → analysis_requirements contains at least one scenario-family
      derived metric.  (The request-level contract validator in
      ``contracts/question_analysis.py`` already enforces that any scenario
      metric request has a populated ``scenario_factor``, so a planner-level
      param check would be redundant.)
    - LIST → primary step's entities param is non-empty or tool naturally
      returns entity-enumerated rows (get_balancing_composition,
      get_generation_mix).
    - render_style → if DETERMINISTIC, narrative-augmentation steps
      (CORRELATION_DRIVER / COMPOSITION_CONTEXT / TARIFF_CONTEXT) are
      suspicious — a deterministic data lookup should usually not need
      narrative context.  If NARRATIVE with an inherently narrative
      answer_kind (EXPLANATION / FORECAST / SCENARIO), a single-step
      plan is likewise suspicious — narrative answers typically need
      supporting context beyond the primary dataset.
    """
    result = PlanValidationResult()
    answer_kind = qa.answer_kind
    if answer_kind is None or not steps:
        return result
    render_style = qa.render_style

    primary = steps[0]
    primary_params = primary.get("params") or {}
    primary_tool = primary.get("tool_name", "")

    if answer_kind == AnswerKind.COMPARISON:
        has_multi_source = len(steps) >= 2
        entities = primary_params.get("entities") or []
        has_multi_entity = len(entities) >= 2
        # Tools that inherently return multi-entity rows count as comparison-capable.
        inherently_multi = primary_tool in {
            ToolName.GET_BALANCING_COMPOSITION.value,
            ToolName.GET_GENERATION_MIX.value,
        }
        # A single-source single-entity plan is still comparison-capable if
        # the date range spans more than one period — the tool produces a
        # multi-row frame that ``compute_mom`` / generic renderer can
        # compare across periods.
        start_date = primary_params.get("start_date")
        end_date = primary_params.get("end_date")
        has_multi_period = bool(start_date and end_date and start_date != end_date)
        if (
            not has_multi_source
            and not has_multi_entity
            and not inherently_multi
            and not has_multi_period
        ):
            degenerate_range = bool(start_date and end_date and start_date == end_date)
            if degenerate_range:
                # Reject: the plan explicitly pins ONE period for ONE
                # comparand from ONE source — no execution outcome can
                # produce a faithful comparison.
                result.add(
                    "comparison_single_point_range",
                    SEVERITY_REJECT,
                    "COMPARISON plan pins a single explicit period for a "
                    "single comparand",
                )
            else:
                # Warn only: with no explicit range the tool's default
                # recent-history window still yields multi-period rows that
                # compute_mom / the generic renderer can compare, and the
                # post-execution evidence validator covers the residual.
                result.add(
                    "comparison_single_source",
                    SEVERITY_WARN,
                    "COMPARISON plan has a single source with single/no "
                    "entity and no explicit multi-period range",
                )
            log.warning(
                "Plan validation: answer_kind=COMPARISON but plan has single "
                "source with single/no entity and no multi-period range. "
                "query=%.80s",
                raw_query,
            )

    elif answer_kind == AnswerKind.TIMESERIES:
        requirement = get_requirement(AnswerKind.TIMESERIES)
        start_date = primary_params.get("start_date")
        end_date = primary_params.get("end_date")
        has_date_range = bool(start_date and end_date)
        if requirement.requires_date_range and not has_date_range:
            # Warn only: tools default to a recent-history window (DESC +
            # LIMIT), which legitimately yields a renderable series.
            result.add(
                "timeseries_missing_range",
                SEVERITY_WARN,
                "TIMESERIES primary step lacks an explicit date range",
            )
            log.warning(
                "Plan validation: answer_kind=TIMESERIES but primary step "
                "lacks date range. query=%.80s",
                raw_query,
            )
        elif requirement.requires_multi_period_range and start_date == end_date:
            # A single-point "range" is a SCALAR, not a TIMESERIES — flag it
            # so downstream callers know the answer kind may be mis-shaped.
            result.add(
                "timeseries_single_point_range",
                SEVERITY_WARN,
                "TIMESERIES primary step range is a single point",
            )
            log.warning(
                "Plan validation: answer_kind=TIMESERIES but primary step "
                "range is a single point (start_date == end_date). "
                "query=%.80s",
                raw_query,
            )

    elif answer_kind == AnswerKind.FORECAST:
        requirement = get_requirement(AnswerKind.FORECAST)
        has_date_range = bool(primary_params.get("start_date") and primary_params.get("end_date"))
        if requirement.requires_date_range and not has_date_range:
            # Warn only: the forecast fit-window expansion supplies history
            # downstream even when the plan carries no explicit range.
            result.add(
                "forecast_missing_range",
                SEVERITY_WARN,
                "FORECAST primary step lacks an explicit date range",
            )
            log.warning(
                "Plan validation: answer_kind=FORECAST but primary step "
                "lacks date range. query=%.80s",
                raw_query,
            )

    elif answer_kind == AnswerKind.SCENARIO:
        # `DerivedMetricRequest` at contracts/question_analysis.py:397-408
        # already enforces `scenario_factor is not None` on any scenario-family
        # metric request, so we only need to verify at least one such request
        # is present.  Missing `scenario_factor` cannot reach this point.
        derived = qa.analysis_requirements.derived_metrics or []
        has_scenario = any(m.metric_name in _SCENARIO_METRIC_NAMES for m in derived)
        if not has_scenario:
            # Reject: the deterministic scenario engine has nothing to
            # compute, so any numeric scenario answer would be ungrounded.
            result.add(
                "scenario_missing_metric",
                SEVERITY_REJECT,
                "SCENARIO contract has no scenario-family derived metric",
            )
            log.warning(
                "Plan validation: answer_kind=SCENARIO but no scenario-family "
                "derived metric found. query=%.80s",
                raw_query,
            )

    elif answer_kind == AnswerKind.LIST:
        entities = primary_params.get("entities") or []
        # ``get_tariffs`` is only "inherently enumerated" when called with an
        # explicit entity set; with an empty list it falls back to a narrow
        # default that silently omits some regulated plants.  Flag that case
        # so the planner's LIST-repair (or missing entity_scope) is visible.
        if primary_tool == ToolName.GET_TARIFFS.value and not entities:
            # Warn only, and note the repair already ran: _repair_list_tariff_entities
            # executes during expansion, so reaching this point means no
            # entity_scope alias was available to expand.
            result.add(
                "list_tariffs_default_subset",
                SEVERITY_WARN,
                "LIST get_tariffs has no entities after repair; default subset applies",
            )
            log.warning(
                "Plan validation: answer_kind=LIST, tool=get_tariffs, no entities "
                "and no entity_scope alias to expand. Result will use default "
                "subset. query=%.80s",
                raw_query,
            )
        # These tools naturally produce entity-enumerated rows.
        inherently_enumerated = primary_tool in {
            ToolName.GET_BALANCING_COMPOSITION.value,
            ToolName.GET_GENERATION_MIX.value,
            ToolName.GET_TARIFFS.value,
        }
        if not entities and not inherently_enumerated:
            result.add(
                "list_not_enumerated",
                SEVERITY_WARN,
                "LIST primary step has no entities and tool is not "
                "inherently entity-enumerated",
            )
            log.warning(
                "Plan validation: answer_kind=LIST but primary step has no "
                "entities and tool is not inherently entity-enumerated. "
                "query=%.80s",
                raw_query,
            )

    # --- render_style cross-check ------------------------------------------
    # Independent of answer_kind, surface plans whose evidence shape does not
    # match the stated rendering mode.  Still warning-only (see top-level
    # docstring) — final enforcement is the job of the summarizer / generic
    # renderer, which can opt-out of narrative context when it sees
    # render_style=DETERMINISTIC.
    if render_style == RenderStyle.DETERMINISTIC and len(steps) > 1:
        _NARRATIVE_AUGMENTATION_ROLES = {
            EvidenceRole.CORRELATION_DRIVER.value,
            EvidenceRole.COMPOSITION_CONTEXT.value,
            EvidenceRole.TARIFF_CONTEXT.value,
        }
        narrative_steps = [
            step.get("role")
            for step in steps[1:]
            if step.get("role") in _NARRATIVE_AUGMENTATION_ROLES
        ]
        if narrative_steps:
            result.add(
                "deterministic_with_narrative_steps",
                SEVERITY_WARN,
                "render_style=DETERMINISTIC plan carries narrative-augmentation steps",
            )
            log.warning(
                "Plan validation: render_style=DETERMINISTIC but plan has "
                "%d narrative-augmentation step(s) (%s). query=%.80s",
                len(narrative_steps),
                sorted(set(narrative_steps)),
                raw_query,
            )
    elif render_style == RenderStyle.NARRATIVE:
        _NARRATIVE_ANSWER_KINDS = {
            AnswerKind.EXPLANATION,
            AnswerKind.FORECAST,
            AnswerKind.SCENARIO,
        }
        if answer_kind in _NARRATIVE_ANSWER_KINDS and len(steps) == 1:
            result.add(
                "narrative_single_step",
                SEVERITY_WARN,
                "render_style=NARRATIVE with an inherently narrative "
                "answer_kind but only primary data planned",
            )
            log.warning(
                "Plan validation: render_style=NARRATIVE and "
                "answer_kind=%s but plan has only primary data "
                "(no context/driver). query=%.80s",
                answer_kind.value,
                raw_query,
            )

    # Phase 16 (§16.4.7): visualization cross-check.  Runs after all
    # answer_kind checks so it always fires when a visualization contract
    # is present. Stays log-only by documented design (enforcement of
    # presentation downgrades is a separate future policy flag).
    _cross_check_visualization(steps, qa, raw_query)

    return result


def _cross_check_visualization(
    steps: List[Dict[str, Any]],
    qa: QuestionAnalysis,
    raw_query: str,
) -> None:
    """Warn when the visualization contract cannot be satisfied by the evidence plan.

    Phase 16 (§16.4.7) — warn-only for first release.  Enforcement (downgrade
    ``primary_presentation`` to ``"table"``) is behind a future policy flag.

    Three checks:
    1. ``primary_presentation ∈ {chart, chart_plus_table}`` but no evidence
       step carries date params → the result will be a static table, not a
       time-series chart.  Renderer will silently fall back.
    2. ``visual_goal ∈ {trend, relationship}`` but the plan has no date range
       → trend/relationship charts require a time axis; single-point evidence
       cannot produce one.
    3. ``time_grain == "season"`` but the evidence date range is shorter than
       2 years → summer and winter buckets will be identical or missing;
       a seasonal chart will be misleading.
    """
    visualization = getattr(qa, "visualization", None)
    if visualization is None:
        return

    primary_presentation = getattr(
        getattr(visualization, "primary_presentation", None), "value", None
    )
    visual_goal = getattr(
        getattr(visualization, "visual_goal", None), "value", None
    )
    time_grain = getattr(
        getattr(visualization, "time_grain", None), "value", None
    )

    # Determine whether any step has a date range — a proxy for "will return
    # time-series rows".  At planning time we only have params; the actual
    # column types are unknown until execution.
    def _has_date_params(step: Dict[str, Any]) -> bool:
        params = step.get("params") or {}
        return bool(params.get("start_date") or params.get("end_date"))

    any_date_params = any(_has_date_params(step) for step in steps)

    # --- Check 1: chart requested but no time-series evidence ---
    _CHART_PRESENTATIONS = {"chart", "chart_plus_table"}
    if primary_presentation in _CHART_PRESENTATIONS and not any_date_params:
        log.warning(
            "Visualization cross-check: primary_presentation=%s but no evidence "
            "step has date params — result will be tabular, not time-series. "
            "Chart rendering may fall back to table. query=%.80s",
            primary_presentation,
            raw_query,
        )

    # --- Check 2: trend/relationship visual_goal with no time axis ---
    _TEMPORAL_GOALS = {"trend", "relationship"}
    if visual_goal in _TEMPORAL_GOALS and not any_date_params:
        log.warning(
            "Visualization cross-check: visual_goal=%s requires a time axis "
            "but no evidence step has date params. Single-point data cannot "
            "produce a meaningful trend. query=%.80s",
            visual_goal,
            raw_query,
        )

    # --- Check 3: season grain needs ≥ 2 years of history ---
    if time_grain == "season" and steps:
        primary_params = steps[0].get("params") or {}
        start_date = primary_params.get("start_date")
        end_date = primary_params.get("end_date")
        if start_date and end_date:
            try:
                from datetime import date as _date
                _start = _date.fromisoformat(str(start_date)[:10])
                _end = _date.fromisoformat(str(end_date)[:10])
                span_years = (_end - _start).days / 365.25
                if span_years < 2.0:
                    log.warning(
                        "Visualization cross-check: time_grain=season but "
                        "evidence span is %.1f years (< 2). Summer and winter "
                        "buckets may be missing or identical. query=%.80s",
                        span_years,
                        raw_query,
                    )
            except Exception:
                pass  # malformed dates — skip the check silently



# Compatibility for callers that used the former evidence_planner private helper.
_validate_plan_against_answer_kind = validate_plan_against_answer_kind
