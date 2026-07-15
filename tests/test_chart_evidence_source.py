"""Tests for P4.3 canonical chart sourcing (finding M1).

Charts must not silently disagree with the canonical answer. When a finalized
ObservationFrame narrowed the evidence, the chart follows it; otherwise the raw
ctx.df is charted and that fallback is recorded and surfaced on chart metadata,
never applied silently.
"""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd

from agent.chart_pipeline import (
    _attach_evidence_identity,
    _normalize_period_keys,
    _select_chart_source,
    build_chart,
)
from contracts.evidence_frames import EntitySetFrame, ObservationFrame
from models import QueryContext
from utils.metrics import metrics


def _price_df():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "p_bal_gel": [10.0, 175.0, 9.0, 220.0],
    })


def _obs_frame(periods):
    return ObservationFrame(
        rows=[
            {"period": p, "entity_id": "balancing_price_gel", "entity_label": "x",
             "metric": "balancing_price", "value": 1.0, "unit": "tetri/kWh"}
            for p in periods
        ],
    )


# ---------------------------------------------------------------------------
# _normalize_period_keys
# ---------------------------------------------------------------------------


class TestNormalizePeriodKeys:
    def test_iso_dates(self):
        assert _normalize_period_keys(["2024-01-01", "2024-02-01"]) == ["2024-01-01", "2024-02-01"]

    def test_year_only_int_not_treated_as_epoch(self):
        # The nanosecond-epoch trap: 2024 must stay "2024", not 1970-...
        assert _normalize_period_keys([2024, 2025]) == ["2024", "2025"]

    def test_year_only_string(self):
        assert _normalize_period_keys(["2024", "2025.0"]) == ["2024", "2025"]

    def test_unparseable_returns_none(self):
        assert _normalize_period_keys(["not-a-date"]) is None

    def test_none_value_returns_none(self):
        assert _normalize_period_keys([None]) is None


# ---------------------------------------------------------------------------
# _select_chart_source
# ---------------------------------------------------------------------------


class TestSelectChartSource:
    def test_no_frame_uses_raw_df_unchanged(self):
        ctx = QueryContext(query="prices")
        ctx.df = _price_df()
        ctx.cols = list(ctx.df.columns)
        before = dict(metrics.chart_source_events)

        out = _select_chart_source(ctx)

        assert len(out) == 4
        assert ctx.chart_evidence_identity["source"] == "raw_ctx_df"
        assert ctx.chart_evidence_identity["filterApplied"] is False
        assert metrics.chart_source_events.get("raw_ctx_df", 0) == before.get("raw_ctx_df", 0) + 1

    def test_frame_filtered_restricts_chart_to_frame_periods(self):
        ctx = QueryContext(query="prices above 15")
        ctx.df = _price_df()
        ctx.cols = list(ctx.df.columns)
        # A threshold filter kept only Feb and Apr.
        ctx.evidence_frame = _obs_frame(["2024-02-01", "2024-04-01"])

        out = _select_chart_source(ctx)

        assert ctx.chart_evidence_identity["source"] == "canonical_frame_filtered"
        assert ctx.chart_evidence_identity["filterApplied"] is True
        assert sorted(out["date"].tolist()) == ["2024-02-01", "2024-04-01"]
        assert ctx.chart_evidence_identity["unit"] == "tetri/kWh"

    def test_frame_aligned_keeps_full_df(self):
        ctx = QueryContext(query="prices")
        ctx.df = _price_df()
        ctx.cols = list(ctx.df.columns)
        ctx.evidence_frame = _obs_frame(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])

        out = _select_chart_source(ctx)

        assert ctx.chart_evidence_identity["source"] == "canonical_frame_aligned"
        assert ctx.chart_evidence_identity["filterApplied"] is False
        assert len(out) == 4

    def test_frame_unmatched_periods_keeps_full_df_and_flags(self):
        ctx = QueryContext(query="prices")
        ctx.df = _price_df()
        ctx.cols = list(ctx.df.columns)
        # Frame periods that do not exist in the df → never silently empty.
        ctx.evidence_frame = _obs_frame(["2099-01-01"])

        out = _select_chart_source(ctx)

        assert ctx.chart_evidence_identity["source"] == "raw_ctx_df_frame_unmatched"
        assert len(out) == 4

    def test_non_observation_frame_falls_back_to_raw(self):
        ctx = QueryContext(query="which plants")
        ctx.df = _price_df()
        ctx.cols = list(ctx.df.columns)
        ctx.evidence_frame = EntitySetFrame(rows=[{"entity_id": "enguri_hpp"}])

        _select_chart_source(ctx)

        assert ctx.chart_evidence_identity["source"] == "raw_ctx_df"

    def test_restriction_never_empties_chart(self):
        # Even a degenerate subset keeps at least the matched rows.
        ctx = QueryContext(query="prices")
        ctx.df = _price_df()
        ctx.cols = list(ctx.df.columns)
        ctx.evidence_frame = _obs_frame(["2024-03-01"])

        out = _select_chart_source(ctx)

        assert not out.empty
        assert out["date"].tolist() == ["2024-03-01"]


# ---------------------------------------------------------------------------
# _attach_evidence_identity
# ---------------------------------------------------------------------------


class TestAttachEvidenceIdentity:
    def test_stamps_fields(self):
        meta: dict = {}
        _attach_evidence_identity(meta, {
            "source": "canonical_frame_filtered", "filterApplied": True, "unit": "tetri/kWh",
        })
        assert meta["evidenceSource"] == "canonical_frame_filtered"
        assert meta["evidenceFilterApplied"] is True
        assert meta["evidenceUnit"] == "tetri/kWh"

    def test_empty_identity_is_noop(self):
        meta: dict = {}
        _attach_evidence_identity(meta, {})
        assert meta == {}


# ---------------------------------------------------------------------------
# build_chart integration
# ---------------------------------------------------------------------------


class TestBuildChartIntegration:
    def test_chart_meta_carries_evidence_source_default_path(self):
        # "plot" is an explicit chart keyword; with no analyzer payload the
        # no-analysis heuristic draws a chart for >=2 rows.
        ctx = QueryContext(query="plot balancing prices over time")
        ctx.df = _price_df()
        ctx.cols = list(ctx.df.columns)
        ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False, name=None)]

        ctx = build_chart(ctx)

        charts = ctx.charts or ([{"metadata": ctx.chart_meta}] if ctx.chart_meta else [])
        assert charts, "expected a chart to be generated"
        assert charts[0]["metadata"]["evidenceSource"] == "raw_ctx_df"

    def test_filtered_frame_narrows_charted_rows(self):
        ctx = QueryContext(query="plot months where balancing price exceeded 15 tetri")
        ctx.df = _price_df()
        ctx.cols = list(ctx.df.columns)
        ctx.rows = [tuple(r) for r in ctx.df.itertuples(index=False, name=None)]
        ctx.evidence_frame = _obs_frame(["2024-02-01", "2024-04-01"])

        ctx = build_chart(ctx)

        charts = ctx.charts or ([{"metadata": ctx.chart_meta, "data": ctx.chart_data}]
                                if ctx.chart_meta else [])
        assert charts, "expected a chart to be generated"
        meta = charts[0]["metadata"]
        assert meta["evidenceSource"] == "canonical_frame_filtered"
        assert meta["evidenceFilterApplied"] is True
        # Only the two surviving periods are charted.
        assert len(charts[0]["data"]) == 2


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
