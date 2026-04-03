"""Focused tests for tariff tool category aliases."""

from __future__ import annotations

import os

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from agent.tools import tariff_tools


def test_get_tariffs_supports_category_level_aliases(monkeypatch):
    captured = {}

    def _fake_run_text_query(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return None, [], []

    monkeypatch.setattr(tariff_tools, "run_text_query", _fake_run_text_query)

    tariff_tools.get_tariffs(
        start_date="2024-01-01",
        end_date="2024-02-01",
        entities=["regulated_hpp", "regulated_new_tpp", "regulated_old_tpp"],
        currency="usd",
    )

    assert "regulated_hpp_tariff_usd" in captured["sql"]
    assert "regulated_new_tpp_tariff_usd" in captured["sql"]
    assert "regulated_old_tpp_tariff_usd" in captured["sql"]
    assert captured["params"]["start_date"] == "2024-01-01"
    assert captured["params"]["end_date"] == "2024-02-01"


def test_get_tariffs_treats_none_currency_as_default_gel(monkeypatch):
    captured = {}

    def _fake_run_text_query(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return None, [], []

    monkeypatch.setattr(tariff_tools, "run_text_query", _fake_run_text_query)

    tariff_tools.get_tariffs(entities=["regulated_new_tpp"], currency=None)

    assert "regulated_new_tpp_tariff_gel" in captured["sql"]
