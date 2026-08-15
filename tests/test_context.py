"""
Tests for context.py configuration.
"""
import pytest


class TestTechTypeGroups:
    """Test tech type group configurations."""

    def test_fallback_consistency(self):
        """Test that fallback constants match context.py values."""
        from context import DEMAND_TECH_TYPES, SUPPLY_TECH_TYPES, TRANSIT_TECH_TYPES

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
        from context import DEMAND_TECH_TYPES, SUPPLY_TECH_TYPES

        assert len(SUPPLY_TECH_TYPES) == len(set(SUPPLY_TECH_TYPES))
        assert len(DEMAND_TECH_TYPES) == len(set(DEMAND_TECH_TYPES))


def parse_static_allowed_tables() -> set:
    """Parse STATIC_ALLOWED_TABLES from config.py source without importing.

    ``config.py`` raises ``RuntimeError: Missing SUPABASE_DB_URL`` at import
    time, so importing it here would make these tests pass only when a module
    that sets the env (e.g. ``test_config.py``) happened to run first in the
    same session.  Reading the source keeps them order-independent.
    """
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


class TestSchemaConsistency:
    """Verify context.py stays aligned with config.py STATIC_ALLOWED_TABLES.

    Note: config.py requires env vars at import time, so we parse
    STATIC_ALLOWED_TABLES directly from the source file to avoid
    needing a live database URL in CI/test environments.
    """

    _parse_static_allowed_tables = staticmethod(parse_static_allowed_tables)

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
        from context import COLUMN_LABELS, DB_SCHEMA_DICT

        missing = []
        for view_name, view_info in DB_SCHEMA_DICT["views"].items():
            for col in view_info["columns"]:
                if col not in COLUMN_LABELS:
                    missing.append(f"{view_name}.{col}")

        assert not missing, (
            f"Columns missing from COLUMN_LABELS: {missing}"
        )

    def test_energy_balance_contract_matches_deployed_materialized_view(self):
        """Readiness and SQL planning must use the deployed energy-balance columns."""
        from context import DB_SCHEMA_DICT, DB_SCHEMA_DOC

        assert DB_SCHEMA_DICT["views"]["energy_balance_long_mv"]["columns"] == [
            "year",
            "sector",
            "energy_source",
            "volume_tj",
        ]
        assert (
            "- energy_balance_long_mv(year, sector, energy_source, volume_tj)"
            in DB_SCHEMA_DOC
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


class TestScrubSchemaMentions:
    """Pin the narrative-safety contract of ``scrub_schema_mentions``.

    The scrubber's job is to mask SQL/schema tokens (column names, view
    names, derived-share columns) leaking into LLM narrative output.  It
    must NOT replace ordinary English words that happen to overlap with
    a schema concept (e.g. the participle "balancing").  Regression
    motivated by production output containing:

        "the Balancing Electricity electricity price increased"
        "sources used for Balancing Electricity the grid"
        "for Balancing Electricity purposes"

    caused by ``VALUE_LABELS["balancing"] = "Balancing Electricity"`` doing
    a case-insensitive ``\\b{word}\\b`` replace on every natural-language
    occurrence of "balancing".  That entry has been removed; the schema
    has no bare ``balancing`` column (SQL uses ``p_bal_*``, ``bal_*``,
    ``share_*``) so removal is loss-free for scrubbing purposes.
    """

    def test_narrative_word_balancing_unchanged(self):
        """The participle 'balancing' must survive scrubbing intact."""
        from context import scrub_schema_mentions

        before = (
            "The balancing price increased by 7.3% in December 2025. "
            "Generation sources used for balancing the grid shifted, "
            "and electricity purchased for balancing purposes rose."
        )
        after = scrub_schema_mentions(before)
        assert "Balancing Electricity electricity" not in after
        assert "Balancing Electricity the grid" not in after
        assert "Balancing Electricity purposes" not in after
        # The natural-language word itself should remain.
        assert "balancing price" in after
        assert "balancing the grid" in after
        assert "balancing purposes" in after

    def test_common_english_columns_survive_scrubbing(self):
        """Ordinary English words that are also column names must not be replaced.

        ``demand_tariff_mv`` and ``capacity_factor`` introduced the columns
        ``value``, ``activity``, ``company`` and ``technology``.  They need
        COLUMN_LABELS entries (readiness/display + the coverage test require
        it) but replacing them in prose mangles ordinary sentences -- the same
        failure VALUE_LABELS documents for bare words.
        """
        from context import scrub_schema_mentions

        before = (
            "The value of imports rose while wind technology matured. "
            "Each company reported higher trading activity."
        )
        after = scrub_schema_mentions(before)
        assert "The value of imports rose" in after
        assert "wind technology matured" in after
        assert "Each company reported higher trading activity." in after

    def test_scrub_exemptions_are_real_columns(self):
        """An exemption for a column that has no label is dead configuration."""
        from context import COLUMN_LABELS, SCRUB_EXEMPT_COLUMNS

        orphans = sorted(SCRUB_EXEMPT_COLUMNS - set(COLUMN_LABELS))
        assert not orphans, f"SCRUB_EXEMPT_COLUMNS entries with no COLUMN_LABELS entry: {orphans}"


class TestDashboardSharedViews:
    """Pin the six materialized views shared with the dashboard's
    Network & Supply and Plant Analytics tabs.

    Column lists were verified against the live database on 2026-08-15.
    They feed REQUIRED_SCHEMA_COLUMNS (main.py), so drift here shows up as a
    permanent schema-readiness gap rather than a test failure in production.
    """

    EXPECTED_COLUMNS = {
        "trade_by_ownership": ["date", "ownership", "quantity"],
        "ownership_concentration": [
            "date", "segment", "total_generation", "owner_count",
            "hhi", "top1_share", "top3_share", "top5_share",
        ],
        "by_capacity": ["date", "entity", "segment", "quantity", "facility_count"],
        "by_commissioning": ["date", "entity", "segment", "quantity"],
        "capacity_factor": [
            "date", "technology", "capacity_category", "capacity_category_order",
            "segment", "facility_count", "generation_mwh", "installed_capacity_mw",
            "hours_in_month", "capacity_factor", "capacity_factor_percent",
        ],
        "demand_tariff_mv": [
            "date", "company", "activity", "volate",
            "level_1_cat", "level_2_cat", "value",
        ],
    }

    def test_columns_match_deployed_materialized_views(self):
        from context import DB_SCHEMA_DICT

        for view_name, columns in self.EXPECTED_COLUMNS.items():
            assert DB_SCHEMA_DICT["views"][view_name]["columns"] == columns, (
                f"{view_name} columns drifted from the deployed view"
            )

    def test_demand_tariff_id_is_deliberately_omitted(self):
        """The view has a demand_tariff_id column; the contract must not claim it.

        It is NULL on every calculated row, so exposing it invites a join on a
        null key.  REQUIRED_SCHEMA_COLUMNS checks required-subset-of-reflected,
        so omitting a real column is safe.
        """
        from context import DB_SCHEMA_DICT

        assert "demand_tariff_id" not in DB_SCHEMA_DICT["views"]["demand_tariff_mv"]["columns"]

    def test_views_are_allowlisted_and_joinable(self):
        from context import DB_JOINS

        allowed = parse_static_allowed_tables()
        for view_name in self.EXPECTED_COLUMNS:
            assert view_name in allowed, f"{view_name} missing from the SQL allowlist"
            assert view_name in DB_JOINS, f"{view_name} missing from DB_JOINS"
            assert DB_JOINS[view_name]["join_on"] == "date"

    def test_schema_doc_declares_every_shared_view(self):
        """A view the planner may query but cannot see in the schema text is unusable.

        ``test_no_phantom_views_in_schema_doc`` guards the other direction
        (documented but not allowlisted); this guards allowlisted but not
        documented, for the six views this change introduced.
        """
        import re

        from context import DB_SCHEMA_DOC

        documented = set(re.findall(r"^- (\w+)\(", DB_SCHEMA_DOC, re.MULTILINE))
        missing = sorted(set(self.EXPECTED_COLUMNS) - documented)
        assert not missing, f"Shared views absent from DB_SCHEMA_DOC 'Available Views': {missing}"

    def test_schema_doc_carries_the_load_bearing_rules(self):
        """Each rule here maps to a way the model would otherwise be silently wrong."""
        from context import DB_SCHEMA_DOC

        # GEL/kWh vs GEL/MWh vs MWh -- three coexisting scales.
        assert "GEL/kWh" in DB_SCHEMA_DOC
        assert "THREE scales now coexist" in DB_SCHEMA_DOC
        # demand_tariff_mv runs to 2030 but complete prices stop at the last final_price month.
        assert "USABLE RANGE" in DB_SCHEMA_DOC
        assert "activity = 'final_price'" in DB_SCHEMA_DOC
        # Blank dimensions are '' and not NULL.
        assert "EMPTY STRINGS" in DB_SCHEMA_DOC
        # The plant-fleet segment is 'total' only -- the balancing filter matches nothing.
        assert "'total'" in DB_SCHEMA_DOC
        # capacity_factor vs capacity_factor_percent.
        assert "Never multiply capacity_factor_percent by 100" in DB_SCHEMA_DOC

    def test_schema_doc_columns_match_the_schema_dict(self):
        """The prompt text and the runtime contract must not drift apart.

        DB_SCHEMA_DICT drives readiness reflection; DB_SCHEMA_DOC is what the
        planner LLM actually reads.  A column in one and not the other means
        either the model invents a column that does not exist, or it never
        learns about one that does.

        Only views declared in the doc's "Available Views" list are compared --
        ``dates_mv`` and ``monthly_cpi_mv`` are allowlisted but undocumented,
        a pre-existing gap this test deliberately does not assert away.
        """
        import re

        from context import DB_SCHEMA_DICT, DB_SCHEMA_DOC

        documented = {
            match.group(1): [col.strip() for col in match.group(2).split(",")]
            for match in re.finditer(r"^- (\w+)\((.*?)\)$", DB_SCHEMA_DOC, re.MULTILINE)
        }

        mismatches = {
            view: {"doc": columns, "dict": DB_SCHEMA_DICT["views"][view]["columns"]}
            for view, columns in documented.items()
            if view in DB_SCHEMA_DICT["views"]
            and columns != DB_SCHEMA_DICT["views"][view]["columns"]
        }
        assert not mismatches, f"DB_SCHEMA_DOC / DB_SCHEMA_DICT column drift: {mismatches}"

    def test_schema_doc_forbids_converting_end_user_tariffs_to_per_mwh(self):
        """Production trace: 0.15289 GEL/kWh was reported as 152.89.

        The model converted to GEL/MWh to match the convention every other view
        uses. The converted figure is in no row, so the grounding gate stripped
        it -- 10 of 15 numeric tokens unmatched, answer cut to 273 of 1,587
        characters.
        """
        from context import DB_SCHEMA_DOC

        assert "Report demand_tariff_mv values in GEL/kWh as stored" in DB_SCHEMA_DOC

    def test_schema_doc_states_the_vat_basis(self):
        """value and final_price are net; 18% is levied on top."""
        from context import DB_SCHEMA_DOC

        assert "NET of VAT" in DB_SCHEMA_DOC
        assert "18%" in DB_SCHEMA_DOC

    def test_schema_doc_states_the_wholesale_comparison_basis(self):
        """Without this the model compares a bare balancing price against a
        tariff that already bundles the guaranteed capacity charge."""
        from context import DB_SCHEMA_DOC

        assert "(p_bal_gel + p_gcap_gel) / 1000" in DB_SCHEMA_DOC

    def test_schema_doc_stays_within_a_sane_prompt_budget(self):
        """DB_SCHEMA_DOC is injected verbatim into every SQL-planning prompt.

        It is absent from every list in ``core/prompt_budget.py``, so it is
        never truncated -- every character it grows is taken from
        UNTRUSTED_DOMAIN_KNOWLEDGE, which *is* shed first.  This cap is a
        tripwire, not a target: if a future change needs more room, move
        detail into the selectively-loaded SQL examples instead.
        """
        from context import DB_SCHEMA_DOC

        assert len(DB_SCHEMA_DOC) < 9000, (
            f"DB_SCHEMA_DOC is {len(DB_SCHEMA_DOC)} chars; it crowds out domain knowledge "
            "in the planner prompt. Move detail into knowledge/sql_example_selector.py."
        )

    def test_value_labels_does_not_have_bare_balancing_key(self):
        """Guard the underlying VALUE_LABELS dict to prevent re-introduction."""
        from context import VALUE_LABELS

        assert "balancing" not in VALUE_LABELS, (
            "VALUE_LABELS['balancing'] re-introduces the over-eager scrubber; "
            "see the comment in context.py at the VALUE_LABELS definition."
        )

    def test_schema_columns_still_scrubbed(self):
        """The fix must not regress scrubbing of actual schema tokens."""
        from context import scrub_schema_mentions

        # share_* derived columns must still be replaced with their labels.
        out = scrub_schema_mentions("share_renewable_ppa rose 5%")
        assert "share_renewable_ppa" not in out
        # The label-replaced version must appear.
        assert "Share of Renewable PPA" in out or "Renewable PPA" in out

    def test_generation_tech_words_not_doubled(self):
        """Bare generation-tech words must survive scrubbing intact — no
        'Hydro Generation generation' doubling (2026-07-08 production report)."""
        from context import scrub_schema_mentions

        before = (
            "Increased hydro generation lowered prices, while thermal "
            "generation and wind generation rose. Transit flows and "
            "losses were stable; import and export shifted."
        )
        after = scrub_schema_mentions(before)
        assert "Hydro Generation generation" not in after
        assert "Thermal Generation generation" not in after
        assert "Wind Generation generation" not in after
        assert "Transit Flows flows" not in after
        # The natural-language words themselves remain.
        assert "hydro generation" in after
        assert "thermal generation" in after
        assert "wind generation" in after

    def test_value_labels_excludes_bare_common_words(self):
        """Guard: ordinary English words must never re-enter the scrubber map."""
        from context import VALUE_LABELS

        banned = {
            "hydro", "thermal", "wind", "solar", "import", "export",
            "transit", "losses", "direct customers", "balancing",
        }
        leaked = banned & set(VALUE_LABELS)
        assert not leaked, (
            f"VALUE_LABELS re-introduces over-eager bare-word scrubbing for "
            f"{sorted(leaked)}; see the INVARIANT comment at VALUE_LABELS."
        )


class TestStripInlineCitationMarkers:
    """Inline citation tags leaked by the LLM must be removed from the answer."""

    def test_strips_asterisk_bracket_marker(self):
        from context import strip_inline_citation_markers

        out = strip_inline_citation_markers(
            "cheaper electricity on the exchange. *[domain_knowledge]*"
        )
        assert out == "cheaper electricity on the exchange."

    def test_strips_cjk_bracket_marker(self):
        from context import strip_inline_citation_markers

        out = strip_inline_citation_markers("CAGR of 2.5% per year【statistics】.")
        assert out == "CAGR of 2.5% per year."

    def test_strips_multi_anchor_marker_keeping_one_space(self):
        from context import strip_inline_citation_markers

        out = strip_inline_citation_markers("see [statistics, domain_knowledge] here")
        assert out == "see here"

    def test_leaves_real_bracketed_content_intact(self):
        from context import strip_inline_citation_markers

        # A year in brackets is not a citation anchor and must survive.
        out = strip_inline_citation_markers("the year [2024] was notable")
        assert out == "the year [2024] was notable"

    def test_preserves_line_structure(self):
        from context import strip_inline_citation_markers

        out = strip_inline_citation_markers("first point. *[domain_knowledge]*\n2. second")
        assert out == "first point.\n2. second"

    def test_strips_evidence_block_anchor_not_in_base_list(self):
        from context import strip_inline_citation_markers

        # why-context evidence-block anchors (regulated_plant_sales,
        # component_pressure, …) are NOT in the base list but are snake_case
        # identifiers, so the shape-based matcher must still strip them.
        out = strip_inline_citation_markers(
            "tariff of 40.21 GEL/MWh (14.73 USD/MWh)【regulated_plant_sales】."
        )
        assert out == "tariff of 40.21 GEL/MWh (14.73 USD/MWh)."
        assert "regulated_plant_sales" not in out
        out2 = strip_inline_citation_markers("pressure rose [component_pressure]")
        assert out2 == "pressure rose"

    def test_leaves_single_word_bracket_prose_intact(self):
        from context import strip_inline_citation_markers

        # A bracketed common word (no underscore, not a base anchor) is prose,
        # not a citation anchor, and must survive.
        assert strip_inline_citation_markers("see the [note] above") == "see the [note] above"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
