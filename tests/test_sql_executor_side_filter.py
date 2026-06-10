"""Tests for the type_tech market-side filter on the legacy SQL path (audit L1).

Total/ambiguous queries must keep all rows; only explicit side intent narrows.
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

from agent.sql_executor import (
    DEMAND_TECH_TYPES,
    SUPPLY_TECH_TYPES,
    TRANSIT_TECH_TYPES,
    apply_type_tech_side_filter,
)


def _mixed_df() -> pd.DataFrame:
    """One row per market side, using a real value from each tech-type list."""
    return pd.DataFrame(
        {
            "type_tech": [SUPPLY_TECH_TYPES[0], DEMAND_TECH_TYPES[0], TRANSIT_TECH_TYPES[0]],
            "quantity_tech": [100.0, 40.0, 5.0],
        }
    )


class TestTypeTechSideFilter:
    def test_total_query_keeps_all_rows(self):
        """An ambiguous 'total' query must NOT be silently reduced to supply."""
        df = _mixed_df()
        out, side = apply_type_tech_side_filter(df, "total electricity in 2024")
        assert side is None
        assert len(out) == 3
        assert set(out["type_tech"]) == set(df["type_tech"])

    def test_generation_query_narrows_to_supply(self):
        out, side = apply_type_tech_side_filter(_mixed_df(), "show electricity generation in 2024")
        assert side == "supply"
        assert set(out["type_tech"]) <= set(SUPPLY_TECH_TYPES)
        assert len(out) == 1

    def test_consumption_query_narrows_to_demand(self):
        out, side = apply_type_tech_side_filter(_mixed_df(), "electricity consumption in 2024")
        assert side == "demand"
        assert set(out["type_tech"]) <= set(DEMAND_TECH_TYPES)

    def test_transit_query_narrows_to_transit(self):
        out, side = apply_type_tech_side_filter(_mixed_df(), "transit volume in 2024")
        assert side == "transit"
        assert set(out["type_tech"]) <= set(TRANSIT_TECH_TYPES)

    def test_demand_intent_wins_over_supply_word(self):
        """When both a demand and a supply word appear, demand takes priority."""
        out, side = apply_type_tech_side_filter(_mixed_df(), "export of generated electricity")
        assert side == "demand"

    def test_no_type_tech_column_is_noop(self):
        df = pd.DataFrame({"date": ["2024-01-01"], "p_bal_gel": [10.0]})
        out, side = apply_type_tech_side_filter(df, "generation in 2024")
        assert side is None
        assert out is df

    def test_empty_side_filter_keeps_original(self):
        """If the matched side has no rows, keep the original frame rather than emptying it."""
        df = pd.DataFrame({"type_tech": [DEMAND_TECH_TYPES[0]], "quantity_tech": [40.0]})
        out, side = apply_type_tech_side_filter(df, "show generation in 2024")
        assert side is None
        assert len(out) == 1

    def test_multilingual_generation_intent(self):
        """Georgian/Russian generation terms also signal supply intent."""
        for q in ["გენერაცია 2024", "выработка электроэнергии 2024"]:
            _, side = apply_type_tech_side_filter(_mixed_df(), q)
            assert side == "supply", q
