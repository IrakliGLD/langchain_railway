"""The pipeline accepts a caller-supplied question analysis."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest

from agent import pipeline as pipeline_module
from agent.pipeline import process_query
from contracts.question_analysis import PreferredPath, QueryType
from tests.test_semantic_lock import _make_qa


def test_an_injected_analysis_skips_the_analyzer_call(monkeypatch):
    """The planner already decided; re-deriving it costs a call per track.

    A report track serializes the planner's structured intent into a question
    and pays a second model call to parse it back. Passing the analysis in
    removes the round-trip without touching any stage that follows.
    """

    monkeypatch.setattr(
        pipeline_module.planner,
        "analyze_question_active",
        lambda _ctx: pytest.fail(
            "an injected analysis must not be re-derived"
        ),
    )
    analysis = _make_qa(
        query_type=QueryType.DATA_EXPLANATION,
        preferred_path=PreferredPath.TOOL,
    )
    captured: dict = {}

    def capture(ctx, *_args, **_kwargs):
        captured["analysis"] = ctx.question_analysis
        captured["source"] = ctx.question_analysis_source
        raise RuntimeError("stop")

    # Stop the run at the stage after analysis: everything beyond it reaches
    # the database and the vector index, and this suite is about which analysis
    # the pipeline adopts, not what it does with it. Patched without
    # raising=False so a renamed stage fails loudly instead of letting the run
    # continue to a real provider call.
    monkeypatch.setattr(pipeline_module, "_run_vector_knowledge_stage", capture)

    with pytest.raises(RuntimeError, match="stop"):
        process_query(
            "What was the May 2026 balancing price?",
            answer_mode="report",
            question_analysis=analysis,
        )

    assert captured["analysis"] is analysis
    assert captured["source"] == "injected"


def test_without_an_injected_analysis_the_analyzer_still_runs(monkeypatch):
    """The default path is untouched: no caller supplies one today."""

    analyzed: list = []

    def analyze(ctx):
        analyzed.append(ctx.query)
        raise RuntimeError("stop")

    monkeypatch.setattr(
        pipeline_module.planner,
        "analyze_question_active",
        analyze,
    )

    with pytest.raises(RuntimeError, match="stop"):
        process_query(
            "What was the May 2026 balancing price?",
            answer_mode="report",
        )

    assert analyzed == ["What was the May 2026 balancing price?"]
