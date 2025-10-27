"""
Basic tests for main.py functionality.

To run tests: pytest tests/
"""
import os
from typing import Any

import pytest
import pandas as pd
import numpy as np
import sqlalchemy


class DummyResult:
    """Minimal result object mimicking SQLAlchemy behaviour used at import time."""

    def fetchall(self):
        return []

    def keys(self):
        return []


class DummyConnection:
    """Context manager returning predictable results for execute calls."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args: Any, **kwargs: Any):
        return DummyResult()


class DummyEngine:
    """Engine stub that avoids real database access during tests."""

    def connect(self):
        return DummyConnection()


# Provide required environment variables and stub the engine *before* importing main.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("APP_SECRET_KEY", "test-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
sqlalchemy.create_engine = lambda *args, **kwargs: DummyEngine()  # type: ignore[assignment]


from main import (
    quick_stats,
    rows_to_preview,
    build_trade_share_cte,
    generate_share_summary,
)  # noqa: E402


class TestQuickStats:
    """Test cases for quick_stats function."""

    def test_empty_rows(self):
        """Test that empty rows return appropriate message."""
        result = quick_stats([], [])
        assert result == "0 rows."

    def test_basic_stats(self):
        """Test basic statistics generation."""
        rows = [(1, 100), (2, 200), (3, 300)]
        cols = ["id", "value"]
        result = quick_stats(rows, cols)
        assert "Rows: 3" in result

    def test_year_month_counting(self):
        """Test that year-month counting logic works correctly."""
        # Create test data with monthly records
        rows = [
            ("2023-01-01", 100),
            ("2023-02-01", 110),
            ("2023-03-01", 120),
        ]
        cols = ["date", "value"]
        result = quick_stats(rows, cols)
        # Should complete without errors
        assert "Rows: 3" in result


class TestRowsToPreview:
    """Test cases for rows_to_preview function."""

    def test_basic_preview(self):
        """Test that rows can be previewed correctly."""
        rows = [(1, "test"), (2, "data")]
        cols = ["id", "name"]
        result = rows_to_preview(rows, cols, max_rows=10)
        assert "test" in result
        assert "data" in result

    def test_numeric_rounding(self):
        """Test that numeric values are rounded correctly."""
        rows = [(1, 3.14159), (2, 2.71828)]
        cols = ["id", "value"]
        result = rows_to_preview(rows, cols)
        # Should round to 3 decimal places
        assert "3.142" in result or "3.141" in result


class TestCAGRCalculation:
    """Test CAGR calculation edge cases."""

    def test_cagr_with_zero_first_value(self):
        """Test that CAGR handles zero first value gracefully."""
        # This would be tested by calling _generate_cagr_forecast
        # with a DataFrame containing zero values
        # For now, this is a placeholder for future implementation
        pass

    def test_cagr_with_negative_values(self):
        """Test that CAGR handles negative values gracefully."""
        # Placeholder for future implementation
        pass


class TestSQLValidation:
    """Test SQL validation functionality."""

    def test_cte_alias_handling(self):
        """Test that CTEs without aliases don't crash."""
        # Placeholder - would test simple_table_whitelist_check
        pass


class TestTradeSharePivot:
    """Tests for the auto-pivot SQL helper used when share columns are hallucinated."""

    def test_basic_wrapping(self):
        original = (
            "SELECT date, share_renewable_ppa FROM trade_derived_entities "
            "WHERE segment = 'balancing_electricity'"
        )
        rewritten = build_trade_share_cte(original)
        assert rewritten.strip().startswith("WITH tde AS"), rewritten
        assert "FROM tde" in rewritten
        assert "WHERE segment = 'balancing_electricity'" in rewritten

    def test_preserves_existing_cte(self):
        original = (
            "WITH latest AS (SELECT * FROM trade_derived_entities) "
            "SELECT * FROM latest"
        )
        rewritten = build_trade_share_cte(original)
        assert rewritten.strip().startswith("WITH tde AS"), rewritten
        assert "latest AS" in rewritten

    def test_alias_survives_replacement(self):
        original = "SELECT t.date FROM trade_derived_entities t"
        rewritten = build_trade_share_cte(original)
        assert "FROM tde t" in rewritten

    def test_includes_aggregated_shares(self):
        original = "SELECT * FROM trade_derived_entities"
        rewritten = build_trade_share_cte(original)
        assert "share_all_ppa" in rewritten
        assert "share_all_renewables" in rewritten


class TestShareSummaryOverride:
    """Validate deterministic share summaries for direct share questions."""

    def test_all_ppa_share_with_breakdown(self):
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-06-01")],
                "share_all_ppa": [0.32],
                "share_renewable_ppa": [0.2],
                "share_thermal_ppa": [0.12],
            }
        )
        plan = {
            "intent": "calculate_share",
            "target": "share of PPA in balancing electricity",
            "period": "2024-06",
        }
        summary = generate_share_summary(
            df,
            plan,
            "What was the share of PPA in balancing electricity in June 2024?",
        )
        assert summary is not None
        assert "June 2024" in summary
        assert "32.0%" in summary
        assert "renewable PPAs 20.0%" in summary
        assert "thermal PPAs 12.0%" in summary

    def test_specific_share_selection(self):
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-06-01")],
                "share_import": [0.41],
                "share_all_ppa": [0.33],
            }
        )
        plan = {
            "intent": "calculate_share",
            "target": "share of import in balancing electricity",
            "period": "2024-06",
        }
        summary = generate_share_summary(
            df,
            plan,
            "What share did import have in balancing electricity during June 2024?",
        )
        assert summary is not None
        assert "41.0%" in summary
        assert "Import" in summary or "Imports" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
