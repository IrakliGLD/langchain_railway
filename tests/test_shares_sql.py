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
