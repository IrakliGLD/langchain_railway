from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pandas as pd

from agent.tools import generation_tools


def test_share_sql_uses_market_side_denominator_before_type_filter(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_statement(statement, params):
        captured["sql"] = str(statement)
        captured["params"] = params
        return pd.DataFrame(), [], []

    monkeypatch.setattr(generation_tools, "run_statement", fake_run_statement)

    generation_tools.get_generation_mix(types=["hydro"], mode="share")

    sql = str(captured["sql"])
    assert "PARTITION BY period, tech_side" in sql
    assert "WHEN type_tech IN ('hydro'" in sql
    assert "WHEN type_tech IN ('abkhazeti'" in sql

    base_start = sql.index("FROM tech_quantity_view")
    base_end = sql.index("GROUP BY period, type_tech")
    final_filter_start = sql.index("FROM with_shares")
    assert "type_tech IN" not in sql[base_start:base_end]
    assert "type_tech IN" in sql[final_filter_start:]
    assert captured["params"]["types"] == ["hydro"]


def test_quantity_sql_keeps_type_filter_in_grouped_query(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_statement(statement, params):
        captured["sql"] = str(statement)
        captured["params"] = params
        return pd.DataFrame(), [], []

    monkeypatch.setattr(generation_tools, "run_statement", fake_run_statement)

    generation_tools.get_generation_mix(types=["hydro"], mode="quantity")

    sql = str(captured["sql"])
    assert "tech_side" not in sql
    assert "FROM tech_quantity_view\nWHERE type_tech IN" in sql
    assert captured["params"]["types"] == ["hydro"]


# ---------------------------------------------------------------------------
# share_basis="generation" — generation-mix shares must be shares of TOTAL
# GENERATION (hydro+thermal+wind+solar sum to 1), not of total supply
# (2026-07-09 generation-mix chart report).
# ---------------------------------------------------------------------------

def _capture(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_statement(statement, params):
        captured["sql"] = str(statement)
        captured["params"] = params
        return pd.DataFrame(), [], []

    monkeypatch.setattr(generation_tools, "run_statement", fake_run_statement)
    return captured


def test_generation_basis_share_sql_restricts_base_to_generation_techs(monkeypatch):
    captured = _capture(monkeypatch)

    generation_tools.get_generation_mix(mode="share", share_basis="generation")

    sql = str(captured["sql"])
    # Denominator scope: the base aggregate only sees generation techs, so
    # shares are computed over total generation.
    base_start = sql.index("FROM tech_quantity_view")
    base_end = sql.index("GROUP BY period, type_tech")
    base = sql[base_start:base_end]
    assert "'hydro'" in base and "'thermal'" in base
    assert "'wind'" in base and "'solar'" in base
    assert "'import'" not in base
    assert "'self-cons'" not in base
    # Single-scope denominator: no per-side partition needed.
    assert "PARTITION BY period, tech_side" not in sql
    assert "PARTITION BY period" in sql


def test_generation_basis_share_sql_still_applies_type_filter_after_shares(monkeypatch):
    captured = _capture(monkeypatch)

    generation_tools.get_generation_mix(
        types=["hydro"], mode="share", share_basis="generation"
    )

    sql = str(captured["sql"])
    final_filter_start = sql.index("FROM with_shares")
    assert "type_tech IN" in sql[final_filter_start:]
    assert captured["params"]["types"] == ["hydro"]


def test_default_share_basis_keeps_market_side_denominator(monkeypatch):
    captured = _capture(monkeypatch)

    generation_tools.get_generation_mix(mode="share")

    sql = str(captured["sql"])
    assert "PARTITION BY period, tech_side" in sql


def test_invalid_share_basis_rejected(monkeypatch):
    _capture(monkeypatch)

    try:
        generation_tools.get_generation_mix(mode="share", share_basis="bogus")
    except ValueError as exc:
        assert "share_basis" in str(exc)
    else:
        raise AssertionError("expected ValueError for bogus share_basis")
