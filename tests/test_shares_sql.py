"""SQL-shape tests for analysis.shares."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import analysis.shares as shares


class _FakeResult:
    def fetchall(self):
        return []

    def keys(self):
        return [
            "date",
            "price_deregulated_hydro_gel",
            "price_deregulated_hydro_usd",
            "price_regulated_hpp_gel",
            "price_regulated_hpp_usd",
        ]


class _FakeConn:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params or {}))
        return _FakeResult()


def test_compute_entity_price_contributions_includes_usd_columns_and_date_filters(monkeypatch):
    monkeypatch.setattr(shares, "check_dataframe_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(shares, "text", lambda sql: sql)

    conn = _FakeConn()
    shares.compute_entity_price_contributions(
        conn,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    assert len(conn.calls) == 1
    sql, params = conn.calls[0]

    assert "p.p_dereg_usd AS price_deregulated_hydro_usd" in sql
    assert "AS price_regulated_hpp_usd" in sql
    assert "AS price_regulated_new_tpp_usd" in sql
    assert "AS price_regulated_old_tpp_usd" in sql
    assert "AS contribution_deregulated_hydro_usd" in sql
    assert "AS total_known_contributions_usd" in sql
    assert "AS residual_contribution_ppa_import_usd" in sql
    assert "t.date >= :start_date" in sql
    assert "t.date <= :end_date" in sql
    assert "p.date >= :start_date" in sql
    assert "p.date <= :end_date" in sql
    assert params["start_date"] == "2023-01-01"
    assert params["end_date"] == "2023-12-31"
    # Tariffs now come from the quantity-weighted balancing view
    assert "mv_balancing_trade_with_tariff" in sql
    assert "tariff_with_usd" not in sql
    # Data-driven grouping via entities_mv.type, not hardcoded plant lists
    assert "entities_mv em" in sql or "entities_mv" in sql
    assert "em.type" in sql
    # Weighted average pattern with NULL safety
    assert "balancing_quantity" in sql
    assert "NULLIF" in sql
    # CfD_scheme must appear in shares CTE and produce its own share column
    assert "CfD_scheme" in sql
    assert "qty_cfd_scheme" in sql
    assert "share_cfd_scheme" in sql
    # share_ppa_import_total residual must include CfD volume
    assert "qty_cfd_scheme" in sql.split("share_ppa_import_total")[0].split("qty_cfd_scheme")[-1] or \
           sql.count("qty_cfd_scheme") >= 2  # appears in shares CTE + residual formula
    # share_all_ppa must NOT absorb CfD (literal PPA only)
    ppa_total_formula = sql[sql.find("share_all_ppa"):sql.find("share_all_ppa") + 300] if "share_all_ppa" in sql else ""
    assert "cfd" not in ppa_total_formula.lower()


def test_build_balancing_correlation_df_uses_weighted_tariffs(monkeypatch):
    """build_balancing_correlation_df sources tariffs from mv_balancing_trade_with_tariff."""
    monkeypatch.setattr(shares, "check_dataframe_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(shares, "text", lambda sql: sql)

    conn = _FakeConn()
    shares.build_balancing_correlation_df(conn)

    assert len(conn.calls) == 1
    sql, _ = conn.calls[0]

    # Output columns preserved
    assert "enguri_tariff_gel" in sql
    assert "gardabani_tpp_tariff_gel" in sql
    assert "grouped_old_tpp_tariff_gel" in sql
    # Source is the new weighted view
    assert "mv_balancing_trade_with_tariff" in sql
    assert "tariff_with_usd" not in sql
    # All series use weighted average, data-driven grouping, NULL safety
    assert "balancing_quantity" in sql
    assert "entities_mv" in sql
    assert "em.type" in sql
    assert "NULLIF" in sql
    # CfD_scheme share present in correlation function too
    assert "CfD_scheme" in sql
    assert "qty_cfd_scheme" in sql
    assert "share_cfd_scheme" in sql
    assert "(s.qty_dereg_hydro + s.qty_reg_hpp + s.qty_ren_ppa + s.qty_cfd_scheme)" in sql
