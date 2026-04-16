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


def test_get_tariffs_supports_both_currencies(monkeypatch):
    captured = {}

    def _fake_run_text_query(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return None, [], []

    monkeypatch.setattr(tariff_tools, "run_text_query", _fake_run_text_query)

    tariff_tools.get_tariffs(
        start_date="2024-01-01",
        end_date="2024-02-01",
        entities=["regulated_new_tpp", "regulated_old_tpp"],
        currency="both",
    )

    assert "regulated_new_tpp_tariff_gel" in captured["sql"]
    assert "regulated_new_tpp_tariff_usd" in captured["sql"]
    assert "regulated_old_tpp_tariff_gel" in captured["sql"]
    assert "regulated_old_tpp_tariff_usd" in captured["sql"]


def test_get_tariffs_defaults_to_exact_plant_aliases(monkeypatch):
    captured = {}

    def _fake_run_text_query(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return None, [], []

    monkeypatch.setattr(tariff_tools, "run_text_query", _fake_run_text_query)

    tariff_tools.get_tariffs(currency="gel")

    assert "enguri_hpp_tariff_gel" in captured["sql"]
    assert "vardnili_hpp_tariff_gel" in captured["sql"]
    assert "dzevrula_hpp_tariff_gel" in captured["sql"]
    assert "lajanuri_hpp_tariff_gel" in captured["sql"]
    assert "zhinvali_hpp_tariff_gel" in captured["sql"]
    assert "vartsikhe_hpp_tariff_gel" in captured["sql"]
    assert "khramhesi_i_tariff_gel" in captured["sql"]
    assert "khramhesi_ii_tariff_gel" in captured["sql"]
    assert "gardabani_tpp_tariff_gel" in captured["sql"]
    assert "mktvari_tpp_tariff_gel" in captured["sql"]
    assert "tbilsresi_tpp_tariff_gel" in captured["sql"]
    assert "gpower_tpp_tariff_gel" in captured["sql"]
    assert "grouped_old_tpp_tariff_gel" not in captured["sql"]


def test_get_tariffs_uses_limited_date_spine_and_filtered_tariff_cte(monkeypatch):
    captured = {}

    def _fake_run_text_query(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return None, [], []

    monkeypatch.setattr(tariff_tools, "run_text_query", _fake_run_text_query)

    tariff_tools.get_tariffs(currency="both")

    assert "WITH dates AS (" in captured["sql"]
    assert "filtered_tariffs AS (" in captured["sql"]
    assert "JOIN dates d ON d.date = src.date" in captured["sql"]
    assert "LEFT JOIN filtered_tariffs t ON t.date = d.date" in captured["sql"]
    assert "GROUP BY d.date" in captured["sql"]
    assert "WHERE t.date = d.date AND t.entity =" not in captured["sql"]
    assert "LIMIT :limit" in captured["sql"]
    assert any(key.startswith("selected_entity_") for key in captured["params"])


def test_get_tariffs_uses_exact_entity_names_for_regulated_hpp(monkeypatch):
    captured = {}

    def _fake_run_text_query(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return None, [], []

    monkeypatch.setattr(tariff_tools, "run_text_query", _fake_run_text_query)

    tariff_tools.get_tariffs(entities=["regulated_hpp"], currency="usd")

    assert "regulated_hpp_tariff_usd" in captured["sql"]
    assert "ILIKE '%energo-pro%'" not in captured["sql"]
    assert captured["params"]["regulated_hpp_entity_1"] == "enguri hpp"
    assert captured["params"]["regulated_hpp_entity_2"] == "vardnili hpp"
    assert captured["params"]["regulated_hpp_entity_3"] == "dzevrula hpp"
    assert captured["params"]["regulated_hpp_entity_4"] == "gumati hpp"
    assert captured["params"]["regulated_hpp_entity_5"] == "shaori hpp"
    assert captured["params"]["regulated_hpp_entity_6"] == "rioni hpp"
    assert captured["params"]["regulated_hpp_entity_7"] == "lajanuri hpp"
    assert captured["params"]["regulated_hpp_entity_8"] == "zhinvali hpp"
    assert captured["params"]["regulated_hpp_entity_9"] == "vartsikhe hpp"
    assert captured["params"]["regulated_hpp_entity_10"] == "khramhesi I"
    assert captured["params"]["regulated_hpp_entity_11"] == "khramhesi II"
    assert captured["params"]["regulated_hpp_entity_3_dereg_start"] == "2026-05-01"
    assert captured["params"]["regulated_hpp_entity_4_dereg_start"] == "2024-05-01"
    assert captured["params"]["regulated_hpp_entity_5_dereg_start"] == "2021-01-01"
    assert captured["params"]["regulated_hpp_entity_6_dereg_start"] == "2022-05-01"
    assert captured["params"]["regulated_hpp_entity_7_dereg_start"] == "2027-01-01"
    assert "t.entity = :regulated_hpp_entity_1" in captured["sql"]
    assert "t.entity = :regulated_hpp_entity_2" in captured["sql"]
    assert "(t.entity = :regulated_hpp_entity_3 AND t.date < :regulated_hpp_entity_3_dereg_start)" in captured["sql"]


def test_resolve_tariff_alias_entities_filters_by_deregulation_date():
    assert tariff_tools.resolve_tariff_alias_entities("regulated_hpp", as_of="2024-02-01") == [
        "enguri hpp",
        "vardnili hpp",
        "dzevrula hpp",
        "gumati hpp",
        "lajanuri hpp",
        "zhinvali hpp",
        "vartsikhe hpp",
        "khramhesi I",
        "khramhesi II",
    ]
    assert tariff_tools.resolve_tariff_alias_entities("regulated_hpp", as_of="2026-06-01") == [
        "enguri hpp",
        "vardnili hpp",
        "lajanuri hpp",
        "zhinvali hpp",
        "vartsikhe hpp",
        "khramhesi I",
        "khramhesi II",
    ]
