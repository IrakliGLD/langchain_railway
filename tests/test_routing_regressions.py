"""Routing regression tests for the 2026-05-09 fix series.

Per the architecture doc §5.3 (Analyzer Misclassification standing
quality area), these tests pin the routing fixes from the 2026-05-09
fix series so that future analyzer-prompt edits, cross-check policy
changes, or schema migrations cannot silently regress them.

What this file covers:

- **Q1 cross-check exception (commit 59e3b61)**: legal LIST shapes
  with high confidence survive the cross-check for regulatory /
  conceptual queries instead of being clobbered to KNOWLEDGE.

- **Q2 schema defaults (commit dd50db6)**: an empty
  ``visualization: {}`` object validates against the QuestionAnalysis
  contract; the three chart fields default to safe values
  (``False``, ``False``, ``0.0``). Pre-Phase-7 this raised a Pydantic
  validation error that cascaded to heuristic fallback.

- **Q3 render_style pinning (commit 989ed43)**: the analyzer prompt's
  supply-structure few-shot example pins ``render_style=deterministic``
  so that "trend and structure of power supply" queries skip vector
  retrieval and ground in the data preview.

- **Phase 1 vocabulary clarity (commit e8874f5)**: the analyzer
  prompt explicitly prohibits emitting ``query_type`` values like
  ``data_explanation`` into the ``answer_kind`` field.

These are pure unit tests with no real LLM calls. They exercise the
post-LLM logic (cross-check, schema validation, prompt content) so
the regression check runs in milliseconds and can sit in CI.

If any of these fails after a prompt or contract change, the failure
indicates a likely production regression — investigate before
shipping. The relevant patterns are documented in
``skills/pipeline-failure-diagnostics/references/failure-taxonomy.md``.
"""

import os

# Ensure config validation passes before importing pipeline modules.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import sqlalchemy  # noqa: E402


class _DummyResult:
    def fetchall(self):
        return []

    def keys(self):
        return []


class _DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        return _DummyResult()


class _DummyEngine:
    def connect(self):
        return _DummyConnection()


sqlalchemy.create_engine = lambda *args, **kwargs: _DummyEngine()  # type: ignore[assignment]


import pytest  # noqa: E402

from agent.pipeline import _cross_check_answer_kind  # noqa: E402
from contracts.question_analysis import (  # noqa: E402
    AnswerKind,
    QuestionAnalysis,
    VisualizationInfo,
)
from models import QueryContext  # noqa: E402


# ---------------------------------------------------------------------------
# QuestionAnalysis payload helper — minimal fixture builder.
# ---------------------------------------------------------------------------


def _make_qa(
    *,
    raw_query: str,
    query_type: str,
    answer_kind: str,
    confidence: float,
    visualization: dict | None = None,
    canonical_query_en: str | None = None,
) -> QuestionAnalysis:
    """Construct a QuestionAnalysis with just enough fields to exercise the
    cross-check + downstream contract checks. Defaults are deliberately
    minimal — tests fill in what they need.
    """
    payload = {
        "version": "question_analysis_v1",
        "raw_query": raw_query,
        "canonical_query_en": canonical_query_en or raw_query,
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": query_type,
            "analysis_mode": "light",
            "intent": "routing_regression_test_fixture",
            "needs_clarification": False,
            "confidence": confidence,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "knowledge"
            if query_type in {"regulatory_procedure", "conceptual_definition"}
            else "tool",
            "needs_sql": False,
            "needs_knowledge": True,
            "prefer_tool": False,
        },
        "knowledge": {"candidate_topics": []},
        "tooling": {"candidate_tools": []},
        "sql_hints": {
            "metric": None,
            "entities": [],
            "aggregation": None,
            "dimensions": [],
            "period": None,
        },
        "visualization": visualization if visualization is not None else {},
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_trend_context": False,
            "needs_correlation_context": False,
            "derived_metrics": [],
        },
        "answer_kind": answer_kind,
        "render_style": "narrative",
        "grouping": "none",
    }
    return QuestionAnalysis.model_validate(payload)


def _make_ctx(qa: QuestionAnalysis, query: str | None = None) -> QueryContext:
    return QueryContext(
        query=query or qa.raw_query,
        mode="light",
        question_analysis=qa,
        question_analysis_source="llm_active",
    )


# ---------------------------------------------------------------------------
# Q1 cross-check exception — legal LIST survives for regulatory queries
# ---------------------------------------------------------------------------


class TestLegalListCrossCheckException:
    """Pins commit 59e3b61: high-confidence LIST shapes for regulatory or
    conceptual queries are not clobbered to KNOWLEDGE by the cross-check.
    """

    def test_regulatory_procedure_list_survives_cross_check(self):
        """The original Q1 bug: "who can trade on the exchange during the
        transitory market model?" was classified as regulatory_procedure
        with answer_kind=list, but the cross-check overrode it to KNOWLEDGE
        because LIST is not in _SAFE_ANSWER_KINDS. The fix added an
        exception for legal/conceptual queries with confidence >= 0.85.
        """
        qa = _make_qa(
            raw_query="who can trade on the exchange during the transitory market model?",
            query_type="regulatory_procedure",
            answer_kind="list",
            confidence=0.9,
        )
        ctx = _make_ctx(qa)

        _cross_check_answer_kind(ctx)

        assert qa.answer_kind == AnswerKind.LIST, (
            "Legal-list exception should preserve LIST shape for "
            "regulatory_procedure queries with high confidence — "
            "regression of commit 59e3b61"
        )

    def test_conceptual_definition_list_survives_cross_check(self):
        """Same exception applies to conceptual_definition queries — the
        analyzer can mis-route an eligibility question as conceptual but
        still emit answer_kind=list, and the cross-check should trust it.
        """
        qa = _make_qa(
            raw_query="who can trade on the exchange during the transitory market model?",
            query_type="conceptual_definition",
            answer_kind="list",
            confidence=0.9,
        )
        ctx = _make_ctx(qa)

        _cross_check_answer_kind(ctx)

        assert qa.answer_kind == AnswerKind.LIST

    def test_exception_does_not_fire_for_data_queries(self):
        """The exception is narrowly scoped to regulatory / conceptual.
        A LIST emitted for a data_retrieval query goes through the normal
        cross-check — derived TIMESERIES from data_retrieval is not in
        the deterministic mapping, so LIST stays.
        """
        qa = _make_qa(
            raw_query="list monthly balancing prices",
            query_type="data_retrieval",
            answer_kind="list",
            confidence=0.9,
        )
        ctx = _make_ctx(qa)

        _cross_check_answer_kind(ctx)

        # data_retrieval is intentionally omitted from
        # _QUERY_TYPE_TO_ANSWER_KIND ("too coarse"); derived is None;
        # cross-check returns early; LIST preserved.
        assert qa.answer_kind == AnswerKind.LIST

    def test_low_confidence_legal_list_does_not_use_exception(self):
        """The exception requires confidence >= 0.85. At lower confidence
        the cross-check falls through to the standard safer-option logic.
        For regulatory_procedure the derived kind is KNOWLEDGE (safe);
        LIST is not in the safe set; so the override clobbers to KNOWLEDGE.
        """
        qa = _make_qa(
            raw_query="who can participate?",
            query_type="regulatory_procedure",
            answer_kind="list",
            confidence=0.5,
        )
        ctx = _make_ctx(qa)

        _cross_check_answer_kind(ctx)

        assert qa.answer_kind == AnswerKind.KNOWLEDGE, (
            "Below-threshold confidence should not trigger the legal-list "
            "exception; the cross-check should clobber LIST to KNOWLEDGE"
        )


# ---------------------------------------------------------------------------
# Q2 schema defaults — empty visualization object validates
# ---------------------------------------------------------------------------


class TestVisualizationSchemaDefaults:
    """Pins commit dd50db6: VisualizationInfo.chart_requested_by_user,
    chart_recommended, chart_confidence all have defaults so an empty
    ``visualization: {}`` payload doesn't crash Pydantic validation.

    Pre-fix, this caused a cascade where an analyzer emitting an empty
    visualization object would fail schema validation, set
    ``analyzer_available=False``, and force heuristic-fallback routing.
    """

    def test_empty_visualization_object_validates(self):
        v = VisualizationInfo.model_validate({})
        assert v.chart_requested_by_user is False
        assert v.chart_recommended is False
        assert v.chart_confidence == 0.0

    def test_explicit_values_override_defaults(self):
        v = VisualizationInfo.model_validate(
            {
                "chart_requested_by_user": True,
                "chart_recommended": True,
                "chart_confidence": 0.92,
            }
        )
        assert v.chart_requested_by_user is True
        assert v.chart_recommended is True
        assert v.chart_confidence == 0.92

    def test_question_analysis_with_empty_visualization_validates(self):
        """End-to-end: a QuestionAnalysis with visualization={} should
        validate without error and downstream code should see the defaults.
        """
        qa = _make_qa(
            raw_query="why did balancing electricity price change in November 2021?",
            query_type="data_explanation",
            answer_kind="explanation",
            confidence=0.9,
            visualization={},
        )
        assert qa.visualization.chart_requested_by_user is False
        assert qa.visualization.chart_recommended is False
        assert qa.visualization.chart_confidence == 0.0


# ---------------------------------------------------------------------------
# Q3 + Phase 1 — analyzer prompt content
# ---------------------------------------------------------------------------


class TestAnalyzerPromptContent:
    """Pins commits e8874f5 (vocabulary clarity), 989ed43 (supply-structure
    few-shot), and the regulatory-eligibility few-shot in the analyzer
    prompt. These are string-level checks on the assembled
    _ANALYZER_CORE_RULES — if any few-shot example is removed or the
    vocabulary clarification is dropped, the test catches it.
    """

    @pytest.fixture
    def core_rules(self) -> str:
        from core.llm import _ANALYZER_CORE_RULES
        return _ANALYZER_CORE_RULES

    def test_vocabulary_clarification_present(self, core_rules: str):
        """Q2 root cause: the LLM was emitting ``answer_kind=data_explanation``
        (a query_type value) and crashing Pydantic. The fix added an
        explicit prohibition in _ANALYZER_CORE_RULES.
        """
        assert "answer_kind` and `query_type` are DIFFERENT" in core_rules, (
            "Vocabulary-distinction rule missing from analyzer prompt — "
            "regression of commit e8874f5"
        )
        assert "data_explanation" in core_rules, (
            "Explicit example of a query_type value (data_explanation) "
            "missing from the prohibition list"
        )

    def test_query_type_to_answer_kind_mapping_present(self, core_rules: str):
        """Q1/Q2 fix introduced a mapping table from query_type to
        typical answer_kind defaults. Asserting the mapping for the
        key cases survives prompt edits.
        """
        assert "regulatory_procedure" in core_rules
        assert "conceptual_definition" in core_rules
        # The mapping should mention LIST for regulatory_procedure when
        # the source enumerates items
        regulatory_section = core_rules[
            core_rules.find("regulatory_procedure"):
            core_rules.find("regulatory_procedure") + 400
        ]
        assert "list" in regulatory_section.lower(), (
            "regulatory_procedure -> list mapping (for enumerated sources) "
            "missing from prompt; affects Q1 routing accuracy"
        )

    def test_visualization_required_fields_rule_present(self, core_rules: str):
        """Q2 fix: even though the schema now defaults the three required
        chart fields, the analyzer prompt still has an explicit rule that
        these fields should be emitted (with safe defaults for non-chart
        questions). Belt-and-suspenders.
        """
        assert "chart_requested_by_user" in core_rules, (
            "Visualization rule (chart_requested_by_user emission) "
            "missing — regression of commit dd50db6 prompt change"
        )
        assert "visualization" in core_rules

    def test_eligibility_few_shot_present(self, core_rules: str):
        """Q1 fix added a few-shot block for eligibility / participation
        questions. The example should route them to regulatory_procedure
        + answer_kind=list.
        """
        # Check for one of the canonical eligibility example phrasings
        assert (
            "who can trade on the exchange" in core_rules.lower()
            or "what documents are required to register" in core_rules.lower()
            or "what conditions must" in core_rules.lower()
        ), (
            "Eligibility few-shot example missing — regression of commit "
            "989ed43; affects Q1 routing"
        )

    def test_supply_structure_few_shot_pins_render_style(self, core_rules: str):
        """Q3 fix: the supply-structure few-shot example pins
        ``render_style=deterministic`` so that "trend and structure of
        power supply" doesn't slip into narrative mode (which would trigger
        vector retrieval and grounding fallback).
        """
        # The supply-structure block must be present
        lower = core_rules.lower()
        assert "trend and structure" in lower, (
            "Supply-structure few-shot missing — regression of commit "
            "989ed43"
        )
        # And it must pin render_style=deterministic
        supply_section_start = lower.find("trend and structure")
        supply_section = core_rules[supply_section_start: supply_section_start + 1500]
        assert "render_style=deterministic" in supply_section.lower(), (
            "Supply-structure few-shot is no longer pinning "
            "render_style=deterministic — Q3 grounding-fallback bug will "
            "regress on typo'd phrasings"
        )

    def test_multi_clause_data_intent_rule_present(self, core_rules: str):
        """Fix E (2026-05-17) — Q7 production trace 5ba12f3c.

        The analyzer classified "Define 'guaranteed source', list the
        three most recent guaranteed-source generators by name, and show
        their average sale price to ESCO in the last quarter." as a pure
        ``conceptual_definition`` because of the ``Define`` prefix, then
        dropped the ``list`` and ``show`` clauses entirely. The fix adds
        a multi-clause rule that promotes data-verb clauses over the
        definition prefix.

        Regression guards:
        - the rule block exists,
        - at least one of the canonical Q7-shape few-shot examples
          ("Define 'guaranteed source', list...", "Define CfD and list...",
          "What is balancing electricity and show...") survives,
        - the rule mentions both ``conceptual_definition`` (what we must
          NOT classify as) and ``data_retrieval`` (the target shape).
        """
        lower = core_rules.lower()
        assert "multi-clause" in lower, (
            "Multi-clause data-intent rule missing — Q7 misrouting will regress"
        )
        # The rule must explicitly forbid pure conceptual_definition for
        # multi-clause queries with data verbs.
        multi_idx = lower.find("multi-clause")
        rule_section = lower[multi_idx: multi_idx + 2500]
        assert "conceptual_definition" in rule_section, (
            "Multi-clause rule must reference conceptual_definition (the "
            "wrong classification) to anchor the override"
        )
        assert "data_retrieval" in rule_section, (
            "Multi-clause rule must reference data_retrieval (the target "
            "classification when listing/showing entities)"
        )
        # At least one of the Q7-shape examples must be present.
        assert (
            "define 'guaranteed source'" in rule_section
            or "define cfd and list" in rule_section
            or "what is balancing electricity and show" in rule_section
        ), (
            "Multi-clause few-shot example missing — Q7-class regression risk"
        )


# ---------------------------------------------------------------------------
# Phase 1.a — SCENARIO override quantitative-anchor gate (2026-05-13 logs)
# ---------------------------------------------------------------------------


def _make_qa_with_scenario_metric(
    *, raw_query: str, scenario_factor: float = 1.34
) -> QuestionAnalysis:
    """Build a QA where the analyzer emits a scenario derived metric on top
    of an EXPLANATION answer_kind — the configuration that triggered the
    failing override in production (request 5, trace 968cf082).
    """
    payload = {
        "version": "question_analysis_v1",
        "raw_query": raw_query,
        "canonical_query_en": raw_query,
        "language": {"input_language": "en", "answer_language": "en"},
        "classification": {
            "query_type": "data_explanation",
            "analysis_mode": "analyst",
            "intent": "scenario_anchor_gate_fixture",
            "needs_clarification": False,
            "confidence": 0.8,
            "ambiguities": [],
        },
        "routing": {
            "preferred_path": "tool",
            "needs_sql": False,
            "needs_knowledge": False,
            "prefer_tool": True,
        },
        "knowledge": {"candidate_topics": []},
        "tooling": {"candidate_tools": []},
        "sql_hints": {
            "metric": None,
            "entities": [],
            "aggregation": None,
            "dimensions": [],
            "period": None,
        },
        "visualization": {},
        "analysis_requirements": {
            "needs_driver_analysis": False,
            "needs_trend_context": False,
            "needs_correlation_context": False,
            "derived_metrics": [
                {
                    "metric": "balancing",
                    "metric_name": "scenario_scale",
                    "target_metric": "balancing",
                    "rank_limit": None,
                    "scenario_factor": scenario_factor,
                    "scenario_volume": 1.0,
                    "scenario_aggregation": None,
                    "season": None,
                }
            ],
        },
        "answer_kind": "explanation",
        "render_style": "narrative",
        "grouping": "none",
    }
    return QuestionAnalysis.model_validate(payload)


def _simulate_scenario_override_block(ctx: QueryContext) -> AnswerKind | None:
    """Run the exact override logic from process_query around line 1893
    against the supplied context, then return the resulting answer_kind.

    The function in pipeline.py mutates ``ctx.question_analysis.answer_kind``
    in place; we mirror that here so the test pins observable behaviour.
    """
    from agent import pipeline

    qa = ctx.question_analysis
    assert qa is not None, "fixture must supply question_analysis"
    if qa.answer_kind in pipeline._SCENARIO_OVERRIDE_ELIGIBLE:
        derived = qa.analysis_requirements.derived_metrics or []
        has_scenario_metric = any(
            m.metric_name in pipeline._SCENARIO_DERIVED_METRICS for m in derived
        )
        if has_scenario_metric and pipeline._query_has_quantitative_anchor(ctx.query):
            qa.answer_kind = AnswerKind.SCENARIO
    return qa.answer_kind


class TestScenarioOverrideAnchorGate:
    """Pin the 2026-05-13 fix that prevents SCENARIO override from firing
    on queries lacking a quantitative anchor.

    Production trace 968cf082 saw analyzer emit scenario_factor=1.34 from
    "if more ppa will be signed, how this will affect the price?" — a
    fabricated number with no source in the query — and the generic
    renderer produced a meaningless "Result: 24511.48 (× 1.34)". The
    gate suppresses the override and lets the request fall back to the
    narrative LLM path.
    """

    def test_anchorless_ppa_query_does_not_override_to_scenario(self):
        """The exact production failure: no number, no multiplicative word."""
        qa = _make_qa_with_scenario_metric(
            raw_query="if more ppa will be signed, how this will affect the price?",
            scenario_factor=1.34,
        )
        ctx = _make_ctx(qa)
        result = _simulate_scenario_override_block(ctx)
        assert result == AnswerKind.EXPLANATION, (
            "Anchorless query routed to SCENARIO — regression of Phase 1.a "
            "production fix (2026-05-13)"
        )

    def test_explicit_percent_anchor_does_override_to_scenario(self):
        """Real scenario queries with a percent must still route to SCENARIO."""
        qa = _make_qa_with_scenario_metric(
            raw_query="if PPA share rises by 20%, how will the balancing price change?",
            scenario_factor=1.20,
        )
        ctx = _make_ctx(qa)
        result = _simulate_scenario_override_block(ctx)
        assert result == AnswerKind.SCENARIO

    def test_double_keyword_anchors_override(self):
        """Multiplicative words (double / halve / triple) count as anchors."""
        qa = _make_qa_with_scenario_metric(
            raw_query="what if we double imports?",
            scenario_factor=2.0,
        )
        ctx = _make_ctx(qa)
        result = _simulate_scenario_override_block(ctx)
        assert result == AnswerKind.SCENARIO

    def test_increase_by_anchors_override(self):
        """Directional verbs followed by 'by' count as anchors."""
        qa = _make_qa_with_scenario_metric(
            raw_query="if thermal generation increases by 30 GEL/MWh, what happens?",
            scenario_factor=1.3,
        )
        ctx = _make_ctx(qa)
        result = _simulate_scenario_override_block(ctx)
        assert result == AnswerKind.SCENARIO

    def test_no_scenario_metric_means_no_override_regardless_of_anchor(self):
        """If the analyzer didn't emit a scenario metric, the override path
        is skipped entirely — even when the query contains numbers."""
        from agent import pipeline

        # Re-use the fixture but strip the scenario metric.
        qa = _make_qa_with_scenario_metric(
            raw_query="the price was 150 GEL/MWh in May",
            scenario_factor=1.0,
        )
        qa.analysis_requirements.derived_metrics = []
        ctx = _make_ctx(qa)
        result = _simulate_scenario_override_block(ctx)
        assert result == AnswerKind.EXPLANATION
        # Sanity: the anchor helper is independent of this path.
        assert pipeline._query_has_quantitative_anchor(ctx.query)

    def test_quantitative_anchor_helper_unit_cases(self):
        """Direct unit tests on the helper for regex-edge confidence."""
        from agent.pipeline import _query_has_quantitative_anchor as has_anchor

        # Negative cases — the failure modes from production
        assert not has_anchor("if more ppa will be signed, how will it affect price?")
        assert not has_anchor("what if rules change in the future?")
        assert not has_anchor("")
        assert not has_anchor(None)  # type: ignore[arg-type]

        # Positive cases — real scenario queries
        assert has_anchor("if price rises by 30%")
        assert has_anchor("what if we double thermal output")
        assert has_anchor("halve the import share and recompute")
        assert has_anchor("price increases by 20 GEL")
        assert has_anchor("100 MW added scenario")
        assert has_anchor("scenario where imports triple")
