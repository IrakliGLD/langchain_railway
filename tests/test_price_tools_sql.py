"""
Regression: get_prices() must not raise AmbiguousParameter when date filters
are passed. Tests the SQL generation path, not the DB connection.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import pandas as pd

# Mock required environment variables before importing any agent modules
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@host:5432/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "dummy")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "dummy")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")

# Mock DB-dependent modules, saving originals so we can restore them after
# import and avoid poisoning other tests that share the same process.
_mocked_keys = ['psycopg', 'sqlalchemy', 'core.query_executor']
_saved: dict = {k: sys.modules.get(k) for k in _mocked_keys}
for _k in _mocked_keys:
    sys.modules[_k] = MagicMock()

import agent.tools.price_tools  # noqa: E402

# Restore original modules so other tests are not affected.
for _k, _v in _saved.items():
    if _v is None:
        sys.modules.pop(_k, None)
    else:
        sys.modules[_k] = _v

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
