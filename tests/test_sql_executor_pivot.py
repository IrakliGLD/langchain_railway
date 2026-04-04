"""SQL-shape tests for agent.sql_executor balancing pivot definitions."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.sql_executor import BALANCING_SHARE_PIVOT_SQL, build_trade_share_cte


# ---------------------------------------------------------------------------
# BALANCING_SHARE_PIVOT_SQL
# ---------------------------------------------------------------------------

class TestBalancingSharePivotSQL:
    """Shape assertions for the canonical runtime balancing pivot."""

    def test_cfd_scheme_raw_share_present(self):
        assert "CfD_scheme" in BALANCING_SHARE_PIVOT_SQL
        assert "share_cfd_scheme" in BALANCING_SHARE_PIVOT_SQL

    def test_all_eight_entity_shares_present(self):
        for col in [
            "share_import",
            "share_deregulated_hydro",
            "share_regulated_hpp",
            "share_regulated_new_tpp",
            "share_regulated_old_tpp",
            "share_renewable_ppa",
            "share_thermal_ppa",
            "share_cfd_scheme",
        ]:
            assert col in BALANCING_SHARE_PIVOT_SQL, f"Missing column: {col}"

    def test_share_all_ppa_excludes_cfd(self):
        """share_all_ppa = renewable_ppa + thermal_ppa only (literal PPA)."""
        sql = BALANCING_SHARE_PIVOT_SQL
        # Formula is the text ending with "AS share_all_ppa"
        alias_end = sql.find("AS share_all_ppa")
        assert alias_end != -1, "AS share_all_ppa not found"
        # Walk back to the start of this expression (previous newline)
        formula_start = sql.rfind("\n", 0, alias_end)
        ppa_formula = sql[formula_start:alias_end]
        assert "CfD_scheme" not in ppa_formula, (
            "share_all_ppa must not include CfD_scheme (literal PPA only)"
        )

    def test_share_all_renewables_includes_cfd(self):
        """share_all_renewables must include CfD_scheme (support-scheme renewable-like)."""
        sql = BALANCING_SHARE_PIVOT_SQL
        alias_end = sql.find("AS share_all_renewables")
        assert alias_end != -1, "AS share_all_renewables not found"
        formula_start = sql.rfind("\n", 0, alias_end - 100)  # go back far enough
        formula = sql[formula_start:alias_end]
        assert "CfD_scheme" in formula, "share_all_renewables must include CfD_scheme"

    def test_share_all_renewables_is_sum_not_single_max(self):
        """share_all_renewables must sum individual MAX calls, not use IN(...) with a single MAX."""
        sql = BALANCING_SHARE_PIVOT_SQL
        alias_end = sql.find("AS share_all_renewables")
        formula_start = sql.rfind("\n", 0, alias_end - 100)
        formula = sql[formula_start:alias_end]
        assert "entity IN" not in formula, (
            "share_all_renewables must not use MAX(CASE WHEN entity IN (...)) — use sum of individual MAXes"
        )


# ---------------------------------------------------------------------------
# build_trade_share_cte
# ---------------------------------------------------------------------------

class TestBuildTradeSharCte:
    """Shape assertions for the CTE injection path."""

    def _get_cte(self) -> str:
        base_sql = "SELECT * FROM trade_derived_entities WHERE segment='balancing'"
        return build_trade_share_cte(base_sql)

    def test_cfd_scheme_raw_share_present(self):
        cte = self._get_cte()
        assert "CfD_scheme" in cte
        assert "share_cfd_scheme" in cte

    def test_all_eight_entity_shares_present(self):
        cte = self._get_cte()
        for col in [
            "share_import",
            "share_deregulated_hydro",
            "share_regulated_hpp",
            "share_regulated_new_tpp",
            "share_regulated_old_tpp",
            "share_renewable_ppa",
            "share_thermal_ppa",
            "share_cfd_scheme",
        ]:
            assert col in cte, f"Missing column in CTE: {col}"

    def test_share_all_ppa_excludes_cfd(self):
        cte = self._get_cte()
        alias_end = cte.find("AS share_all_ppa")
        assert alias_end != -1
        formula_start = cte.rfind("\n", 0, alias_end)
        ppa_formula = cte[formula_start:alias_end]
        assert "CfD_scheme" not in ppa_formula

    def test_share_all_renewables_includes_cfd(self):
        cte = self._get_cte()
        assert "share_all_renewables" in cte
        alias_end = cte.find("AS share_all_renewables")
        formula_start = cte.rfind("\n", 0, alias_end - 100)
        formula = cte[formula_start:alias_end]
        assert "CfD_scheme" in formula
