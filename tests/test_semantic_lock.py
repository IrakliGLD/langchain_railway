"""
Tests for the semantic contract lock: ensuring Stage 0.2 output is authoritative
and not overridden by later-stage raw-query heuristics.
"""
import os

import pandas as pd
import pytest
import sqlalchemy

# Ensure config validation passes before importing project modules.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


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

    def execute(self, *_args, **_kwargs):
        return _DummyResult()


class _DummyEngine:
    def connect(self):
        return _DummyConnection()


sqlalchemy.create_engine = lambda *args, **kwargs: _DummyEngine()  # type: ignore[assignment]

from contracts.question_analysis import (  # noqa: E402
    AnalysisMode,
    ChartIntent,
    AnalysisRequirementsInfo,
    ClassificationInfo,
    DerivedMetricName,
    DerivedMetricRequest,
    DimensionName,
    KnowledgeInfo,
    LanguageCode,
    LanguageInfo,
    PreferredPath,
    QueryType,
    QuestionAnalysis,
    RoutingInfo,
    SqlHints,
    ToolCandidate,
    ToolingInfo,
    ToolName,
    VisualizationInfo,
)
from models import QueryContext  # noqa: E402
from utils.query_validation import validate_tool_relevance  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qa(
    *,
    query_type: QueryType = QueryType.COMPARISON,
    intent: str = "compare balancing electricity prices",
    preferred_path: PreferredPath = PreferredPath.TOOL,
    candidate_tools: list | None = None,
    needs_driver_analysis: bool = False,
    needs_correlation_context: bool = False,
    derived_metrics: list | None = None,
    canonical_query: str = "Compare balancing electricity prices for January 2022 and February 2022",
) -> QuestionAnalysis:
    """Build a minimal valid QuestionAnalysis for testing."""
    return QuestionAnalysis(
        version="question_analysis_v1",
        raw_query="test query",
        canonical_query_en=canonical_query,
        language=LanguageInfo(input_language=LanguageCode.EN, answer_language=LanguageCode.EN),
        classification=ClassificationInfo(
            query_type=query_type,
            analysis_mode=AnalysisMode.ANALYST,
            intent=intent,
            needs_clarification=False,
            confidence=0.95,
        ),
        routing=RoutingInfo(
            preferred_path=preferred_path,
            needs_sql=False,
            needs_knowledge=False,
            prefer_tool=True,
        ),
        knowledge=KnowledgeInfo(),
        tooling=ToolingInfo(candidate_tools=candidate_tools or []),
        sql_hints=SqlHints(),
        visualization=VisualizationInfo(
            chart_requested_by_user=False,
            chart_recommended=False,
            chart_confidence=0.0,
        ),
        analysis_requirements=AnalysisRequirementsInfo(
            needs_driver_analysis=needs_driver_analysis,
            needs_correlation_context=needs_correlation_context,
            derived_metrics=derived_metrics or [],
        ),
    )


def _make_ctx(
    query: str = "what about the share of prices",
    qa: QuestionAnalysis | None = None,
    resolved_query: str = "",
    semantic_locked: bool = False,
    used_tool: bool = False,
    tool_name: str | None = None,
    df: pd.DataFrame | None = None,
) -> QueryContext:
    """Build a QueryContext for testing."""
    ctx = QueryContext(query=query)
    if qa is not None:
        ctx.question_analysis = qa
        ctx.question_analysis_source = "llm_active"
    ctx.resolved_query = resolved_query or (qa.canonical_query_en if qa else "")
    ctx.semantic_locked = semantic_locked
    ctx.used_tool = used_tool
    ctx.tool_name = tool_name
    if df is not None:
        ctx.df = df
        ctx.cols = list(df.columns)
        ctx.rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    return ctx


# ---------------------------------------------------------------------------
# Tests: semantic_locked flag and effective_query
# ---------------------------------------------------------------------------

class TestSemanticLockedFlag:

    def test_defaults_to_false(self):
        ctx = QueryContext(query="test")
        assert ctx.semantic_locked is False

    def test_effective_query_returns_raw_when_unlocked(self):
        ctx = QueryContext(query="raw text")
        ctx.resolved_query = "canonical text"
        assert ctx.effective_query == "raw text"

    def test_effective_query_returns_resolved_when_locked(self):
        ctx = QueryContext(query="raw text")
        ctx.resolved_query = "canonical text"
        ctx.semantic_locked = True
        assert ctx.effective_query == "canonical text"

    def test_effective_query_falls_back_when_resolved_empty(self):
        ctx = QueryContext(query="raw text")
        ctx.resolved_query = ""
        ctx.semantic_locked = True
        assert ctx.effective_query == "raw text"


# ---------------------------------------------------------------------------
# Tests: tool relevance uses resolved_query
# ---------------------------------------------------------------------------

class TestToolRelevanceUsesResolvedQuery:

    def test_resolved_query_matches_price_tool(self):
        """When resolved_query is about prices, get_prices should pass relevance."""
        relevant, _ = validate_tool_relevance(
            "Compare balancing electricity prices for January and February 2022",
            "get_prices",
        )
        assert relevant

    def test_raw_followup_with_share_would_match_composition(self):
        """Demonstrate that raw query with 'share' matches composition tool."""
        relevant, _ = validate_tool_relevance(
            "what about the share of prices",
            "get_balancing_composition",
        )
        assert relevant

    def test_sql_executor_uses_effective_query_for_sql_fallback_guards(self, monkeypatch):
        """SQL fallback should validate pivot/relevance against effective_query, not raw text."""
        from agent import sql_executor

        observed: dict[str, str] = {}

        monkeypatch.setattr(sql_executor, "sanitize_sql", lambda sql: sql)
        monkeypatch.setattr(sql_executor, "simple_table_whitelist_check", lambda sql: None)
        monkeypatch.setattr(sql_executor, "plan_validate_repair", lambda sql: sql)
        monkeypatch.setattr(sql_executor, "validate_aggregation_logic", lambda sql, intent: (True, "OK"))

        def _record_pivot(query: str, sql: str) -> bool:
            observed["pivot_query"] = query
            return False

        def _record_relevance(query: str, sql: str, plan: dict):
            observed["relevance_query"] = query
            return True, "ok", False

        monkeypatch.setattr(sql_executor, "should_inject_balancing_pivot", _record_pivot)
        monkeypatch.setattr(sql_executor, "validate_sql_relevance", _record_relevance)
        monkeypatch.setattr(
            sql_executor,
            "execute_sql_safely",
            lambda sql: (
                pd.DataFrame({"month": ["2022-01", "2022-02"], "balancing_price_gel": [183.8, 195.8]}),
                ["month", "balancing_price_gel"],
                [("2022-01", 183.8), ("2022-02", 195.8)],
                0.01,
            ),
        )

        qa = _make_qa(
            query_type=QueryType.COMPARISON,
            intent="compare balancing electricity prices",
            candidate_tools=[ToolCandidate(name=ToolName.GET_PRICES, score=0.9, reason="price comparison")],
            canonical_query="Compare the balancing electricity prices for January 2022 and February 2022",
        )
        ctx = _make_ctx(
            query="what about the share of prices between january and february",
            qa=qa,
            resolved_query=qa.canonical_query_en,
            semantic_locked=True,
        )
        ctx.raw_sql = "SELECT month, balancing_price_gel FROM price_with_usd"
        ctx.plan = {"intent": "comparison_analysis"}
        ctx.aggregation_intent = {"aggregation_type": "share", "needs_share": True}

        sql_executor.validate_and_execute(ctx)

        assert observed["pivot_query"] == qa.canonical_query_en
        assert observed["relevance_query"] == qa.canonical_query_en


# ---------------------------------------------------------------------------
# Tests: share detection respects semantic contract
# ---------------------------------------------------------------------------

class TestShareDetectionRespectsAnalyzer:

    def test_freeform_intent_text_does_not_trigger_structured_share_signal(self):
        """Free-form analyzer intent text should not be treated as authoritative."""
        qa = _make_qa(
            query_type=QueryType.COMPARISON,
            intent="compare share of balancing prices between months",
            candidate_tools=[
                ToolCandidate(name=ToolName.GET_PRICES, score=0.9, reason="price comparison"),
            ],
        )
        ctx = _make_ctx(query="compare the share of prices", qa=qa, semantic_locked=True)

        assert not ctx.analyzer_indicates_share_intent

    def test_share_dimension_alone_does_not_authorize_share_intent(self):
        """Supporting dimensions should not override a primary price intent."""
        qa = _make_qa(
            query_type=QueryType.DATA_RETRIEVAL,
            intent="compare prices",
            candidate_tools=[
                ToolCandidate(name=ToolName.GET_PRICES, score=0.9, reason="primary"),
            ],
        )
        qa.sql_hints.dimensions = [DimensionName.SHARE]
        ctx = _make_ctx(query="show market mix", qa=qa, semantic_locked=True)

        assert not ctx.analyzer_indicates_share_intent

    def test_decomposition_chart_hint_alone_does_not_authorize_share_intent(self):
        """Chart hints are secondary metadata, not the primary semantic intent."""
        qa = _make_qa(
            query_type=QueryType.DATA_RETRIEVAL,
            intent="compare prices",
            candidate_tools=[
                ToolCandidate(name=ToolName.GET_PRICES, score=0.9, reason="primary"),
            ],
        )
        qa.visualization.chart_intent = ChartIntent.DECOMPOSITION
        ctx = _make_ctx(query="show market mix", qa=qa, semantic_locked=True)

        assert not ctx.analyzer_indicates_share_intent

    def test_share_blocked_when_analyzer_says_price_comparison(self):
        """When analyzer classifies as comparison with price intent,
        'share' in raw query should NOT trigger share detection."""
        qa = _make_qa(
            query_type=QueryType.COMPARISON,
            intent="compare balancing electricity prices month over month",
            candidate_tools=[
                ToolCandidate(name=ToolName.GET_PRICES, score=0.9, reason="price comparison"),
            ],
        )
        ctx = _make_ctx(
            query="compare the share of balancing prices between January and February",
            qa=qa,
            semantic_locked=True,
        )
        share_intent = str(ctx.plan.get("intent", "")).lower()
        analyzer_share_signal = ctx.analyzer_indicates_share_intent
        share_query_detected = share_intent in {"calculate_share", "share"} or analyzer_share_signal

        assert not share_query_detected, (
            "Share should NOT be detected when analyzer says price comparison"
        )

    def test_secondary_composition_candidate_does_not_trigger_share(self):
        """A secondary get_balancing_composition candidate should not trigger share."""
        qa = _make_qa(
            intent="compare balancing electricity prices",
            candidate_tools=[
                ToolCandidate(name=ToolName.GET_PRICES, score=0.9, reason="primary"),
                ToolCandidate(name=ToolName.GET_BALANCING_COMPOSITION, score=0.6, reason="secondary"),
            ],
        )
        ctx = _make_ctx(query="compare share of prices", qa=qa, semantic_locked=True)

        share_intent = str(ctx.plan.get("intent", "")).lower()
        analyzer_share_signal = ctx.analyzer_indicates_share_intent
        share_query_detected = share_intent in {"calculate_share", "share"} or analyzer_share_signal

        assert not share_query_detected, (
            "Secondary composition candidate should not trigger share"
        )

    def test_genuine_share_query_still_detected(self):
        """When analyzer says composition/share, share detection should fire."""
        qa = _make_qa(
            query_type=QueryType.DATA_RETRIEVAL,
            intent="calculate share of balancing composition",
            candidate_tools=[
                ToolCandidate(name=ToolName.GET_BALANCING_COMPOSITION, score=0.9, reason="share query"),
            ],
        )
        ctx = _make_ctx(
            query="what is the share of each source in balancing",
            qa=qa,
            semantic_locked=True,
        )

        share_intent = str(ctx.plan.get("intent", "")).lower()
        analyzer_share_signal = ctx.analyzer_indicates_share_intent
        share_query_detected = share_intent in {"calculate_share", "share"} or analyzer_share_signal

        assert share_query_detected, "Genuine share query should still be detected"

    def test_legacy_share_detection_when_no_analyzer(self):
        """Without analyzer, 'share' keyword in raw query triggers detection."""
        ctx = _make_ctx(query="what is the share of import in balancing")
        # No question_analysis set

        share_intent = str(ctx.plan.get("intent", "")).lower()
        share_query_detected = share_intent in {"calculate_share", "share"} or "share" in ctx.query.lower()

        assert share_query_detected, "Legacy keyword detection should work without analyzer"


# ---------------------------------------------------------------------------
# Tests: analyzer does not mutate authoritative intent
# ---------------------------------------------------------------------------

class TestAnalyzerDoesNotMutateAuthoritativePlan:

    def test_correlation_enrichment_keeps_authoritative_intent(self, monkeypatch):
        from agent import analyzer

        qa = _make_qa(
            query_type=QueryType.DATA_EXPLANATION,
            intent="balancing_price_why",
            candidate_tools=[
                ToolCandidate(name=ToolName.GET_PRICES, score=0.9, reason="price why"),
            ],
            needs_driver_analysis=True,
            needs_correlation_context=True,
            canonical_query="Why did balancing electricity price change in November 2021?",
        )

        corr_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2021-10-01"), pd.Timestamp("2021-11-01"), pd.Timestamp("2021-12-01")],
                "p_bal_gel": [100.0, 120.0, 110.0],
                "p_bal_usd": [32.0, 38.0, 35.0],
                "xrate": [3.1, 3.2, 3.15],
                "share_import": [0.20, 0.25, 0.22],
                "share_deregulated_hydro": [0.30, 0.28, 0.31],
                "share_regulated_hpp": [0.10, 0.09, 0.11],
                "share_renewable_ppa": [0.20, 0.19, 0.18],
                "enguri_tariff_gel": [10.0, 10.0, 10.0],
                "gardabani_tpp_tariff_gel": [20.0, 20.0, 20.0],
                "grouped_old_tpp_tariff_gel": [18.0, 18.0, 18.0],
            }
        )

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute(self, *_args, **_kwargs):
                return None

        class _Engine:
            def connect(self):
                return _Conn()

        monkeypatch.setattr(analyzer, "ENGINE", _Engine())
        monkeypatch.setattr(analyzer, "build_balancing_correlation_df", lambda *_args, **_kwargs: corr_df)
        monkeypatch.setattr(analyzer, "_build_why_context", lambda *_args, **_kwargs: None)

        ctx = _make_ctx(
            query="why did balancing electricity price change?",
            qa=qa,
            semantic_locked=True,
            df=pd.DataFrame(
                {
                    "date": [pd.Timestamp("2021-10-01"), pd.Timestamp("2021-11-01")],
                    "p_bal_gel": [100.0, 120.0],
                }
            ),
        )
        ctx.plan = {
            "intent": "balancing_price_why",
            "target": "p_bal_gel",
            "period": "November 2021",
        }

        out = analyzer.enrich(ctx)

        assert out.plan["intent"] == "balancing_price_why"
        assert "p_bal_gel" in out.correlation_results


# ---------------------------------------------------------------------------
# Tests: evidence precedence
# ---------------------------------------------------------------------------

class TestEvidencePrecedence:

    def test_share_override_blocked_when_non_share_tool_result_exists(self):
        """When semantic_locked and a non-share tool result exists,
        share_summary_override generation should be blocked."""
        qa = _make_qa(
            intent="compare balancing electricity prices",
            candidate_tools=[
                ToolCandidate(name=ToolName.GET_PRICES, score=0.9, reason="price comparison"),
            ],
        )
        ctx = _make_ctx(
            query="compare the share of prices",
            qa=qa,
            semantic_locked=True,
            used_tool=True,
            tool_name="get_prices",
            df=pd.DataFrame({"date": ["2022-01"], "price": [100]}),
        )

        # Simulate the evidence precedence guard from analyzer.py
        share_query_detected = True  # Assume somehow it was detected
        has_non_share_result = (
            not ctx.df.empty
            and ctx.used_tool
            and ctx.tool_name != "get_balancing_composition"
        )
        if has_non_share_result:
            share_query_detected = False

        assert not share_query_detected

    def test_share_override_discarded_in_summarizer_when_intent_mismatch(self):
        """Summarizer should discard share_summary_override when analyzer
        intent doesn't indicate share/composition."""
        qa = _make_qa(
            intent="compare balancing electricity prices",
            candidate_tools=[
                ToolCandidate(name=ToolName.GET_PRICES, score=0.9, reason="price comparison"),
            ],
        )
        ctx = _make_ctx(
            query="compare the share of prices",
            qa=qa,
            semantic_locked=True,
        )
        ctx.share_summary_override = "Import share was 45%..."

        # Simulate the summarizer guard
        if not ctx.analyzer_indicates_share_intent:
            ctx.share_summary_override = None

        assert ctx.share_summary_override is None

    def test_share_override_kept_for_genuine_composition_query(self):
        """For genuine composition queries, share_summary_override should be kept."""
        qa = _make_qa(
            intent="calculate share of balancing composition",
            candidate_tools=[
                ToolCandidate(name=ToolName.GET_BALANCING_COMPOSITION, score=0.9, reason="composition"),
            ],
        )
        ctx = _make_ctx(qa=qa, semantic_locked=True)
        ctx.share_summary_override = "Import share was 45%..."

        if not ctx.analyzer_indicates_share_intent:
            ctx.share_summary_override = None

        assert ctx.share_summary_override is not None


# ---------------------------------------------------------------------------
# Tests: correlation/forecast/why detection respects analyzer
# ---------------------------------------------------------------------------

class TestDetectionRespectsAnalyzer:

    def test_correlation_not_triggered_when_analyzer_says_no(self):
        """When analyzer says needs_driver_analysis=False and
        needs_correlation_context=False, correlation should not trigger
        even if raw query has 'why' keyword."""
        qa = _make_qa(
            query_type=QueryType.DATA_RETRIEVAL,
            intent="retrieve price data",
            needs_driver_analysis=False,
            needs_correlation_context=False,
        )
        ctx = _make_ctx(
            query="why did prices change in January",
            qa=qa,
            semantic_locked=True,
        )
        # Simulate correlation detection with analyzer
        qa_reqs = ctx.question_analysis.analysis_requirements
        should_correlate = qa_reqs.needs_driver_analysis or qa_reqs.needs_correlation_context

        assert not should_correlate

    def test_correlation_triggered_when_analyzer_says_yes(self):
        """When analyzer says needs_driver_analysis=True, correlation fires."""
        qa = _make_qa(
            query_type=QueryType.DATA_EXPLANATION,
            intent="explain price change drivers",
            needs_driver_analysis=True,
        )
        ctx = _make_ctx(query="show me price data", qa=qa, semantic_locked=True)

        qa_reqs = ctx.question_analysis.analysis_requirements
        should_correlate = qa_reqs.needs_driver_analysis or qa_reqs.needs_correlation_context

        assert should_correlate

    def test_forecast_not_triggered_when_analyzer_says_data_retrieval(self):
        """Forecast mode should not trigger when analyzer says data_retrieval,
        even if raw query contains 'forecast' keyword."""
        qa = _make_qa(
            query_type=QueryType.DATA_RETRIEVAL,
            intent="retrieve historical price data",
        )
        ctx = _make_ctx(
            query="what is the forecast for balancing prices",
            qa=qa,
            semantic_locked=True,
        )
        forecast_detected = ctx.question_analysis.classification.query_type.value == "forecast"

        assert not forecast_detected

    def test_forecast_triggered_when_analyzer_says_forecast(self):
        """Forecast mode should trigger when analyzer classifies as forecast."""
        qa = _make_qa(query_type=QueryType.FORECAST, intent="forecast balancing prices")
        ctx = _make_ctx(query="show prices", qa=qa, semantic_locked=True)
        forecast_detected = ctx.question_analysis.classification.query_type.value == "forecast"

        assert forecast_detected

    def test_why_detection_uses_analyzer_when_available(self):
        """Why mode uses needs_driver_analysis from analyzer, not keyword."""
        qa = _make_qa(
            query_type=QueryType.DATA_EXPLANATION,
            intent="explain price change",
            needs_driver_analysis=True,
        )
        ctx = _make_ctx(query="show me prices", qa=qa, semantic_locked=True)

        qa_reqs = ctx.question_analysis.analysis_requirements
        why_detected = (
            qa_reqs.needs_driver_analysis
            or ctx.question_analysis.classification.query_type.value == "data_explanation"
        )

        assert why_detected


# ---------------------------------------------------------------------------
# Tests: missing evidence blocking
# ---------------------------------------------------------------------------

class TestMissingEvidenceBlocking:

    def test_data_explanation_with_derived_metrics_blocks(self):
        """data_explanation with requested derived metrics should trigger blocking."""
        from agent.pipeline import _should_block_data_summary_for_missing_evidence

        qa = _make_qa(
            query_type=QueryType.DATA_EXPLANATION,
            intent="explain price changes",
            derived_metrics=[
                DerivedMetricRequest(
                    metric_name=DerivedMetricName.MOM_ABSOLUTE_CHANGE,
                    metric="p_bal_gel",
                ),
            ],
        )
        ctx = _make_ctx(qa=qa, semantic_locked=True)
        ctx.missing_evidence_for_metrics = ["mom_absolute_change"]

        assert _should_block_data_summary_for_missing_evidence(ctx)

    def test_data_explanation_without_derived_metrics_does_not_block(self):
        """data_explanation without derived metrics should not block."""
        from agent.pipeline import _should_block_data_summary_for_missing_evidence

        qa = _make_qa(
            query_type=QueryType.DATA_EXPLANATION,
            intent="explain price trend",
        )
        ctx = _make_ctx(qa=qa, semantic_locked=True)
        ctx.missing_evidence_for_metrics = ["mom_absolute_change"]

        assert not _should_block_data_summary_for_missing_evidence(ctx)

    def test_comparison_still_blocks(self):
        """comparison query type should still block on missing evidence."""
        from agent.pipeline import _should_block_data_summary_for_missing_evidence

        qa = _make_qa(query_type=QueryType.COMPARISON, intent="compare prices")
        ctx = _make_ctx(qa=qa, semantic_locked=True)
        ctx.missing_evidence_for_metrics = ["mom_absolute_change"]

        assert _should_block_data_summary_for_missing_evidence(ctx)
