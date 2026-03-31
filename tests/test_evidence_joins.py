"""Tests for analysis.evidence_joins."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd
import pytest

from analysis.evidence_joins import join_evidence, join_evidence_with_provenance


class TestJoinEvidence:
    def test_prices_composition_join(self):
        prices = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
            "p_bal_gel": [50.0, 55.0, 60.0],
            "xrate": [2.7, 2.8, 2.9],
        })
        composition = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
            "share_import": [0.3, 0.25, 0.2],
            "share_thermal_ppa": [0.4, 0.45, 0.5],
        })

        result = join_evidence(prices, composition, "get_prices", "get_balancing_composition")

        assert "share_import" in result.columns
        assert "share_thermal_ppa" in result.columns
        assert "p_bal_gel" in result.columns
        assert len(result) == 3

    def test_composition_prices_join(self):
        composition = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "share_import": [0.3, 0.25],
        })
        prices = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "p_bal_gel": [50.0, 55.0],
            "p_bal_usd": [18.5, 19.6],
            "xrate": [2.7, 2.8],
        })

        result = join_evidence(composition, prices, "get_balancing_composition", "get_prices")

        assert "p_bal_gel" in result.columns
        assert "xrate" in result.columns
        assert "share_import" in result.columns

    def test_tariffs_prices_join(self):
        prices = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "p_bal_gel": [50.0, 55.0],
        })
        tariffs = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "tariff_gel": [30.0, 31.0],
        })

        result = join_evidence(prices, tariffs, "get_prices", "get_tariffs")

        assert "tariff_gel" in result.columns
        assert "p_bal_gel" in result.columns

    def test_empty_secondary_returns_primary_unchanged(self):
        prices = pd.DataFrame({
            "date": ["2023-01-01"],
            "p_bal_gel": [50.0],
        })
        empty = pd.DataFrame()

        result = join_evidence(prices, empty, "get_prices", "get_balancing_composition")

        assert result.equals(prices)

    def test_empty_primary_returns_empty(self):
        empty = pd.DataFrame()
        comp = pd.DataFrame({"date": ["2023-01-01"], "share_import": [0.3]})

        result = join_evidence(empty, comp, "get_prices", "get_balancing_composition")

        assert result.empty

    def test_no_overlapping_dates_produces_nans(self):
        prices = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "p_bal_gel": [50.0, 55.0],
        })
        comp = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01"],
            "share_import": [0.3, 0.25],
        })

        result = join_evidence(prices, comp, "get_prices", "get_balancing_composition")

        assert "share_import" in result.columns
        assert result["share_import"].isna().all()

    def test_mismatched_date_formats(self):
        prices = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "p_bal_gel": [50.0, 55.0],
        })
        comp = pd.DataFrame({
            "date": ["01/01/2023", "02/01/2023"],
            "share_import": [0.3, 0.25],
        })

        result = join_evidence(prices, comp, "get_prices", "get_balancing_composition")

        # Should handle different date formats gracefully
        assert "share_import" in result.columns

    def test_generation_prices_join(self):
        gen = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "hydro": [100, 120],
            "thermal": [50, 45],
        })
        prices = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "p_bal_gel": [50.0, 55.0],
            "xrate": [2.7, 2.8],
        })

        result = join_evidence(gen, prices, "get_generation_mix", "get_prices")

        assert "p_bal_gel" in result.columns
        assert "hydro" in result.columns


class TestJoinEvidenceWithProvenance:
    def test_provenance_returned_on_successful_join(self):
        prices = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "p_bal_gel": [50.0, 55.0],
        })
        comp = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"],
            "share_import": [0.3, 0.25],
        })

        result, prov = join_evidence_with_provenance(
            prices, comp, "get_prices", "get_balancing_composition",
        )

        assert "share_import" in result.columns
        assert prov is not None
        assert prov["primary_tool"] == "get_prices"
        assert prov["secondary_tool"] == "get_balancing_composition"
        assert prov["join_key"] == "date"
        assert prov["join_type"] == "left"
        assert "share_import" in prov["columns_added"]
        assert prov["primary_rows"] == 2
        assert prov["merged_rows"] == 2

    def test_provenance_none_for_empty_secondary(self):
        prices = pd.DataFrame({
            "date": ["2023-01-01"],
            "p_bal_gel": [50.0],
        })
        empty = pd.DataFrame()

        result, prov = join_evidence_with_provenance(
            prices, empty, "get_prices", "get_balancing_composition",
        )

        assert result.equals(prices)
        assert prov is None

    def test_provenance_none_for_empty_primary(self):
        empty = pd.DataFrame()
        comp = pd.DataFrame({"date": ["2023-01-01"], "share_import": [0.3]})

        result, prov = join_evidence_with_provenance(
            empty, comp, "get_prices", "get_balancing_composition",
        )

        assert result.empty
        assert prov is None
