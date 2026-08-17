"""A make-or-buy answer is an assessment, so it is written, not rendered.

2026-08-17: the same question came back once as ``render_style=deterministic``
and once as ``narrative``. The deterministic run produced the worse answer, and
the system had already noticed the mismatch -- that run logged

    Plan validation: render_style=DETERMINISTIC but plan has 1
    narrative-augmentation step(s) (['tariff_context'])

and carried on anyway. A make-or-buy answer carries irreversibility, seasonality,
load-shape and regulatory-cycle caveats; a deterministic renderer cannot express
any of them.

The override idiom is not new: agent/evidence_finalizer.py:257-264 already sets
``render_style = NARRATIVE`` when an evidence gap cannot be corrected.
"""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.analyzer import _pin_make_or_buy_render_style
from contracts.question_analysis import RenderStyle
from models import QueryContext
from tests.test_annual_comparison_block import _rows


def _ctx(records, *, render_style=RenderStyle.DETERMINISTIC, authoritative=True):
    df = pd.DataFrame(records)
    ctx = QueryContext(query="regulated vs wholesale", df=df, cols=list(df.columns))
    if authoritative:
        from tests.test_end_user_scope_clarification import _ctx as _analysis_ctx

        analysis = _analysis_ctx("q", topics=("network_supply_tariffs",)).question_analysis
        analysis.render_style = render_style
        ctx.question_analysis = analysis
        ctx.question_analysis_source = "llm_active"
    return ctx


def _no_benchmark(records):
    out = []
    for record in records:
        row = dict(record)
        row.pop("wholesale_benchmark_gel_kwh", None)
        out.append(row)
    return out


def test_a_deterministic_make_or_buy_frame_is_pinned_to_narrative():
    ctx = _ctx(_rows(2022, range(1, 13), 0.145, 0.147))

    _pin_make_or_buy_render_style(ctx)

    assert ctx.question_analysis.render_style == RenderStyle.NARRATIVE


def test_a_narrative_frame_is_left_alone():
    ctx = _ctx(_rows(2022, range(1, 13), 0.145, 0.147), render_style=RenderStyle.NARRATIVE)

    _pin_make_or_buy_render_style(ctx)

    assert ctx.question_analysis.render_style == RenderStyle.NARRATIVE


def test_a_frame_without_a_benchmark_keeps_its_deterministic_render():
    """The pin is for make-or-buy specifically, not for every retail frame.

    A plain tariff time series renders deterministically just fine.
    """
    ctx = _ctx(_no_benchmark(_rows(2022, range(1, 13), 0.145, 0.147)))

    _pin_make_or_buy_render_style(ctx)

    assert ctx.question_analysis.render_style == RenderStyle.DETERMINISTIC


def test_a_non_authoritative_analysis_is_not_touched():
    """Same guard the evidence-finalizer degrade uses."""
    ctx = _ctx(_rows(2022, range(1, 13), 0.145, 0.147), authoritative=False)

    _pin_make_or_buy_render_style(ctx)  # must not raise

    assert ctx.question_analysis is None


def test_an_empty_frame_is_not_touched():
    ctx = _ctx([])
    ctx.df = pd.DataFrame()

    _pin_make_or_buy_render_style(ctx)

    assert ctx.question_analysis.render_style == RenderStyle.DETERMINISTIC


def test_stage_3_enrichment_applies_the_pin():
    """Wiring: a pin nothing calls changes no answer."""
    from agent.analyzer import enrich

    ctx = _ctx(_rows(2022, range(1, 13), 0.145, 0.147))
    ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False, name=None)]

    enrich(ctx)

    assert ctx.question_analysis.render_style == RenderStyle.NARRATIVE
