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


class TestSchemaConsistency:
    """Verify context.py stays aligned with config.py STATIC_ALLOWED_TABLES.

    Note: config.py requires env vars at import time, so we parse
    STATIC_ALLOWED_TABLES directly from the source file to avoid
    needing a live database URL in CI/test environments.
    """

    @staticmethod
    def _parse_static_allowed_tables() -> set:
        """Parse STATIC_ALLOWED_TABLES from config.py source without importing."""
        import ast
        import pathlib

        config_path = pathlib.Path(__file__).parent.parent / "config.py"
        source = config_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "STATIC_ALLOWED_TABLES"
            ):
                return set(ast.literal_eval(node.value))

        raise RuntimeError("STATIC_ALLOWED_TABLES not found in config.py")

    def test_all_allowed_tables_in_schema_dict(self):
        """Every view in STATIC_ALLOWED_TABLES must have an entry in DB_SCHEMA_DICT."""
        from context import DB_SCHEMA_DICT

        allowed = self._parse_static_allowed_tables()
        for table in allowed:
            assert table in DB_SCHEMA_DICT["views"], (
                f"{table} is in STATIC_ALLOWED_TABLES but missing from DB_SCHEMA_DICT"
            )

    def test_all_allowed_tables_in_view_labels(self):
        """Every view in STATIC_ALLOWED_TABLES must have a human-readable label."""
        from context import VIEW_LABELS

        allowed = self._parse_static_allowed_tables()
        for table in allowed:
            assert table in VIEW_LABELS, (
                f"{table} is in STATIC_ALLOWED_TABLES but missing from VIEW_LABELS"
            )

    def test_schema_dict_columns_have_labels(self):
        """Every column in DB_SCHEMA_DICT must have an entry in COLUMN_LABELS."""
        from context import DB_SCHEMA_DICT, COLUMN_LABELS

        missing = []
        for view_name, view_info in DB_SCHEMA_DICT["views"].items():
            for col in view_info["columns"]:
                if col not in COLUMN_LABELS:
                    missing.append(f"{view_name}.{col}")

        assert not missing, (
            f"Columns missing from COLUMN_LABELS: {missing}"
        )

    def test_no_phantom_views_in_schema_doc(self):
        """No view in DB_SCHEMA_DOC should be absent from STATIC_ALLOWED_TABLES."""
        import re
        from context import DB_SCHEMA_DOC

        allowed = self._parse_static_allowed_tables()

        # Extract view names from the "Available Views:" section
        # Pattern matches lines like "- view_name(col1, col2, ...)"
        view_pattern = re.compile(r"^- (\w+)\(", re.MULTILINE)
        doc_views = set(view_pattern.findall(DB_SCHEMA_DOC))

        phantom = doc_views - allowed
        assert not phantom, (
            f"Views in DB_SCHEMA_DOC but not in STATIC_ALLOWED_TABLES (phantom): {phantom}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
