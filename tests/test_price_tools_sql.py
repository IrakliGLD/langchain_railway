"""
Regression: get_prices() must not raise AmbiguousParameter when date filters
are passed. Tests the SQL generation path, not the DB connection.
"""
import pytest
import sys
from unittest.mock import patch, MagicMock
import pandas as pd

# Mock required environment variables before importing any agent modules
import os
os.environ["SUPABASE_DB_URL"] = "postgresql://user:pass@host:5432/db"
os.environ["ENAI_GATEWAY_SECRET"] = "dummy"
os.environ["ENAI_SESSION_SIGNING_SECRET"] = "dummy"
os.environ["ENAI_EVALUATE_SECRET"] = "dummy"
os.environ["GOOGLE_API_KEY"] = "dummy"

# Mock DB engine creation before importing anything from agent
import sys
from unittest.mock import patch, MagicMock
sys.modules['psycopg'] = MagicMock()
sys.modules['sqlalchemy'] = MagicMock()
sys.modules['core.query_executor'] = MagicMock()

import agent.tools.price_tools

MOCK_RESULT = (pd.DataFrame(), [], [])


def _call_get_prices(**kwargs):
    """Call get_prices with mocked run_text_query, return (sql, params)."""
    captured = {}
    def fake_run(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return MOCK_RESULT
    
    with patch("agent.tools.price_tools.run_text_query", side_effect=fake_run):
        from agent.tools import price_tools
        price_tools.get_prices(**kwargs)
        
    return captured["sql"], captured["params"]


def test_both_dates_no_null_check():
    sql, params = _call_get_prices(
        start_date="2022-11-01", end_date="2022-11-30",
        metric="balancing", currency="gel",
    )
    # Old pattern would put :start_date in both IS NULL and >= positions
    assert "IS NULL" not in sql, "AmbiguousParameter pattern must not appear in SQL"
    assert "start_date" in params
    assert "end_date" in params
    assert params["start_date"] == "2022-11-01"
    assert params["end_date"] == "2022-11-30"


def test_no_dates_no_where_clause():
    sql, params = _call_get_prices(metric="balancing", currency="gel")
    assert "start_date" not in params
    assert "end_date" not in params
    assert "IS NULL" not in sql


def test_only_start_date():
    sql, params = _call_get_prices(start_date="2022-11-01", metric="balancing", currency="gel")
    assert "start_date" in params
    assert "end_date" not in params
    assert "IS NULL" not in sql


def test_yearly_granularity_no_ambiguous_params():
    sql, params = _call_get_prices(
        start_date="2022-01-01", end_date="2022-12-31",
        metric="balancing", currency="gel", granularity="yearly",
    )
    assert "IS NULL" not in sql
    assert params.get("start_date") == "2022-01-01"
    assert params.get("end_date") == "2022-12-31"
