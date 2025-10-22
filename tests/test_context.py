"""
Tests for context.py configuration.
"""
import pytest


class TestTechTypeGroups:
    """Test tech type group configurations."""

    def test_fallback_consistency(self):
        """Test that fallback constants match context.py values."""
        from context import SUPPLY_TECH_TYPES, DEMAND_TECH_TYPES, TRANSIT_TECH_TYPES

        # Verify expected values are present
        assert "hydro" in SUPPLY_TECH_TYPES
        assert "thermal" in SUPPLY_TECH_TYPES
        assert "solar" in SUPPLY_TECH_TYPES

        assert "supply-distribution" in DEMAND_TECH_TYPES
        assert "direct customers" in DEMAND_TECH_TYPES
        assert "abkhazeti" in DEMAND_TECH_TYPES

        assert "transit" in TRANSIT_TECH_TYPES

    def test_no_duplicates(self):
        """Test that tech type lists don't have duplicates."""
        from context import SUPPLY_TECH_TYPES, DEMAND_TECH_TYPES

        assert len(SUPPLY_TECH_TYPES) == len(set(SUPPLY_TECH_TYPES))
        assert len(DEMAND_TECH_TYPES) == len(set(DEMAND_TECH_TYPES))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
