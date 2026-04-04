"""
Basic tests for main.py functionality.

To run tests: pytest tests/
"""
import os
import time
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
import pandas as pd
import numpy as np
import sqlalchemy
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from sqlalchemy import text


class DummyResult:
    """Minimal result object mimicking SQLAlchemy behaviour used at import time."""

    def fetchall(self):
        return []

    def keys(self):
        return []


class DummyConnection:
    """Context manager returning predictable results for execute calls."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args: Any, **kwargs: Any):
        return DummyResult()


class DummyEngine:
    """Engine stub that avoids real database access during tests."""

    def connect(self):
        return DummyConnection()


# Provide required environment variables and stub the engine *before* importing main.
os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
sqlalchemy.create_engine = lambda *args, **kwargs: DummyEngine()  # type: ignore[assignment]


from main import (
    BALANCING_SEGMENT_NORMALIZER,
    BALANCING_SHARE_PIVOT_SQL,
    build_trade_share_cte,
    generate_share_summary,
    fetch_balancing_share_panel,
    build_share_shift_notes,
    ensure_share_dataframe,
)  # noqa: E402
import main as main_module  # noqa: E402
from analysis.stats import quick_stats, rows_to_preview
from agent.tools import composition_tools
from agent.tools import common as tool_common
from core.sql_generator import sanitize_sql
from utils import auth as auth_module
from utils.session_memory import issue_session_token


TEST_SUPABASE_JWT_SECRET = "test-supabase-jwt-secret"
TEST_BEARER_SUBJECT = "00000000-0000-0000-0000-000000000123"


def _fake_query_context():
    return SimpleNamespace(
        summary="ok",
        chart_data=None,
        chart_type=None,
        chart_meta={},
        stage_timings_ms={},
        summary_claims=[],
        summary_citations=[],
        summary_confidence=1.0,
        summary_provenance_coverage=1.0,
        summary_claim_provenance=[],
        summary_provenance_gate_passed=True,
        summary_provenance_gate_reason="ok",
        provenance_query_hash="",
        provenance_source="tool",
    )


def _install_successful_ask_mocks(monkeypatch):
    monkeypatch.setattr(main_module, "process_query", lambda **_kwargs: _fake_query_context())
    monkeypatch.setattr(main_module, "append_exchange", lambda *_args, **_kwargs: None)


def _make_bearer_token(
    *,
    secret: str = TEST_SUPABASE_JWT_SECRET,
    sub: str = TEST_BEARER_SUBJECT,
    exp_offset_seconds: int = 3600,
) -> str:
    payload = {
        "sub": sub,
        "aud": "authenticated",
        "exp": int(time.time()) + exp_offset_seconds,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def _make_request(headers: dict[str, str], client_host: str = "127.0.0.1") -> Request:
    raw_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/ask",
        "raw_path": b"/ask",
        "query_string": b"",
        "headers": raw_headers,
        "client": (client_host, 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


def _set_public_bearer_auth(monkeypatch, enabled: bool) -> None:
    monkeypatch.setattr(auth_module, "ENABLE_PUBLIC_BEARER_AUTH", enabled)
    monkeypatch.setattr(
        auth_module,
        "ENAI_AUTH_MODE",
        "gateway_and_bearer" if enabled else "gateway_only",
    )
    monkeypatch.setattr(main_module, "ENABLE_PUBLIC_BEARER_AUTH", enabled)


def _clear_rate_limit_buckets() -> None:
    main_module._preauth_rate_buckets.clear()
    main_module._gateway_rate_buckets.clear()
    main_module._user_rate_buckets.clear()


class TestQuickStats:
    """Test cases for quick_stats function."""

    def test_empty_rows(self):
        """Test that empty rows return appropriate message."""
        result = quick_stats([], [])
        assert result == "0 rows."

    def test_basic_stats(self):
        """Test basic statistics generation."""
        rows = [(1, 100), (2, 200), (3, 300)]
        cols = ["id", "value"]
        result = quick_stats(rows, cols)
        assert "Rows: 3" in result

    def test_year_month_counting(self):
        """Test that year-month counting logic works correctly."""
        # Create test data with monthly records
        rows = [
            ("2023-01-01", 100),
            ("2023-02-01", 110),
            ("2023-03-01", 120),
        ]
        cols = ["date", "value"]
        result = quick_stats(rows, cols)
        # Should complete without errors
        assert "Rows: 3" in result


class TestRowsToPreview:
    """Test cases for rows_to_preview function."""

    def test_basic_preview(self):
        """Test that rows can be previewed correctly."""
        rows = [(1, "test"), (2, "data")]
        cols = ["id", "name"]
        result = rows_to_preview(rows, cols, max_rows=10)
        assert "test" in result
        assert "data" in result

    def test_numeric_rounding(self):
        """Test that numeric values are rounded correctly."""
        rows = [(1, 3.14159), (2, 2.71828)]
        cols = ["id", "value"]
        result = rows_to_preview(rows, cols)
        # Should round to 3 decimal places
        assert "3.142" in result or "3.141" in result

    def test_csv_format(self):
        """Output should be CSV (comma-separated), not padded table."""
        rows = [(1, "alpha", 10.5), (2, "beta", 20.3)]
        cols = ["id", "name", "value"]
        result = rows_to_preview(rows, cols)
        lines = [line.strip() for line in result.strip().splitlines()]
        # Header must be CSV
        assert lines[0] == "id,name,value"
        # Data rows comma-separated
        assert "alpha" in lines[1]
        assert "," in lines[1]

    def test_large_preview_triggers_truncation(self):
        """When preview exceeds max_preview_chars, rows are dropped from the middle."""
        # 200 rows × 10 numeric columns → would exceed a small cap
        import random
        random.seed(42)
        rows = [("row_" + str(i), *(random.random() for _ in range(9))) for i in range(200)]
        cols = ["label"] + [f"col_{j}" for j in range(9)]
        result = rows_to_preview(rows, cols, max_preview_chars=5000)
        assert len(result) <= 5500  # some tolerance for final iteration
        # First and last rows preserved
        lines = [line.strip() for line in result.strip().splitlines()]
        assert lines[1].startswith("row_0,")     # first data row
        assert lines[-1].startswith("row_199,")  # last data row


class TestCAGRCalculation:
    """Test CAGR calculation edge cases."""

    def test_cagr_with_zero_first_value(self):
        """Test that CAGR handles zero first value gracefully."""
        # This would be tested by calling _generate_cagr_forecast
        # with a DataFrame containing zero values
        # For now, this is a placeholder for future implementation
        pass

    def test_cagr_with_negative_values(self):
        """Test that CAGR handles negative values gracefully."""
        # Placeholder for future implementation
        pass


class TestSQLValidation:
    """Test SQL validation functionality."""

    def test_cte_alias_handling(self):
        """Test that CTEs without aliases don't crash."""
        sql = "WITH x AS (SELECT 1 AS a) SELECT * FROM x"
        out = sanitize_sql(sql)
        assert out.strip().upper().startswith("WITH")

    def test_rejects_data_modifying_cte(self):
        """CTEs with DML must be rejected by sanitizer."""
        sql = "WITH x AS (DELETE FROM price_with_usd RETURNING *) SELECT * FROM x"
        with pytest.raises(HTTPException) as exc:
            sanitize_sql(sql)
        assert "read-only SELECT" in str(exc.value.detail)


class TestTypedToolRowContract:
    """Ensure typed tools return rows as plain tuples (same contract as SQL fallback path)."""

    class _RowLike:
        def __init__(self, *values):
            self._values = values

        def __iter__(self):
            return iter(self._values)

        def __len__(self):
            return len(self._values)

        def __getitem__(self, index):
            return self._values[index]

    class _FakeResult:
        def __init__(self, rows, cols):
            self._rows = rows
            self._cols = cols

        def fetchall(self):
            return self._rows

        def keys(self):
            return self._cols

    class _FakeConnection:
        def __init__(self, rows, cols):
            self._result = TestTypedToolRowContract._FakeResult(rows, cols)
            self._calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *_args, **_kwargs):
            self._calls += 1
            if self._calls == 1:
                return DummyResult()  # SET TRANSACTION READ ONLY
            return self._result

    class _FakeEngine:
        def __init__(self, rows, cols):
            self._rows = rows
            self._cols = cols

        def connect(self):
            return TestTypedToolRowContract._FakeConnection(self._rows, self._cols)

    def test_run_text_query_normalizes_rows_to_tuples(self, monkeypatch):
        rows = [
            self._RowLike(1, "alpha"),
            self._RowLike(2, "beta"),
        ]
        cols = ["id", "name"]

        monkeypatch.setattr(tool_common, "ENGINE", self._FakeEngine(rows, cols))
        monkeypatch.setattr(tool_common, "check_dataframe_memory", lambda _df: None)

        df, out_cols, out_rows = tool_common.run_text_query("SELECT id, name FROM t")

        assert out_cols == cols
        assert out_rows == [(1, "alpha"), (2, "beta")]
        assert all(isinstance(r, tuple) for r in out_rows)
        assert list(df.columns) == cols

    def test_run_statement_normalizes_rows_to_tuples(self, monkeypatch):
        rows = [
            self._RowLike("2024-01-01", 10.5),
            self._RowLike("2024-02-01", 11.5),
        ]
        cols = ["date", "p_bal_usd"]

        monkeypatch.setattr(tool_common, "ENGINE", self._FakeEngine(rows, cols))
        monkeypatch.setattr(tool_common, "check_dataframe_memory", lambda _df: None)

        df, out_cols, out_rows = tool_common.run_statement(text("SELECT date, p_bal_usd FROM t"))

        assert out_cols == cols
        assert out_rows == [("2024-01-01", 10.5), ("2024-02-01", 11.5)]
        assert all(isinstance(r, tuple) for r in out_rows)
        assert list(df.columns) == cols


def test_ask_uses_gateway_secret_and_not_evaluate_secret(monkeypatch):
    _install_successful_ask_mocks(monkeypatch)
    # Override the already-captured config constants so the test is deterministic
    # regardless of shell env or module import ordering.
    monkeypatch.setattr(main_module, "GATEWAY_SHARED_SECRET", "test-gateway-key")
    monkeypatch.setattr(auth_module, "GATEWAY_SHARED_SECRET", "test-gateway-key")
    monkeypatch.setattr(main_module, "EVALUATE_ADMIN_SECRET", "test-evaluate-key")

    client = TestClient(main_module.app)

    ok = client.post(
        "/ask",
        json={"query": "Show balancing price trend in 2024."},
        headers={"X-App-Key": "test-gateway-key"},
    )
    assert ok.status_code == 200
    assert ok.json()["answer"] == "ok"
    assert ok.headers.get("X-Session-Token")

    unauthorized = client.post(
        "/ask",
        json={"query": "Show balancing price trend in 2024."},
        headers={"X-App-Key": "test-evaluate-key"},
    )
    assert unauthorized.status_code == 401


def test_ask_accepts_valid_public_bearer(monkeypatch):
    _install_successful_ask_mocks(monkeypatch)
    _set_public_bearer_auth(monkeypatch, True)
    monkeypatch.setattr(auth_module, "SUPABASE_JWT_SECRET", TEST_SUPABASE_JWT_SECRET)
    monkeypatch.setattr(main_module, "ASK_RATE_LIMIT_PREAUTH_PER_MINUTE", 100)
    _clear_rate_limit_buckets()

    token = _make_bearer_token()
    client = TestClient(main_module.app)
    response = client.post(
        "/ask",
        json={"query": "Show balancing price trend in 2024."},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "ok"
    assert response.headers.get("X-Session-Token")
    assert response.headers.get("X-LLM-Total-Tokens") is None
    assert response.headers.get("X-LLM-Estimated-Cost-USD") is None
    _clear_rate_limit_buckets()


def test_public_bearer_rate_limit_returns_429(monkeypatch):
    _install_successful_ask_mocks(monkeypatch)
    _set_public_bearer_auth(monkeypatch, True)
    monkeypatch.setattr(auth_module, "SUPABASE_JWT_SECRET", TEST_SUPABASE_JWT_SECRET)
    monkeypatch.setattr(main_module, "ASK_RATE_LIMIT_PUBLIC_PER_MINUTE", 1)
    monkeypatch.setattr(main_module, "ASK_RATE_LIMIT_PREAUTH_PER_MINUTE", 100)
    _clear_rate_limit_buckets()

    token = _make_bearer_token()
    client = TestClient(main_module.app)
    headers = {"Authorization": f"Bearer {token}"}

    first = client.post(
        "/ask",
        json={"query": "Show balancing price trend in 2024."},
        headers=headers,
    )
    second = client.post(
        "/ask",
        json={"query": "Show balancing price trend in 2024."},
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "Rate limit exceeded"
    _clear_rate_limit_buckets()


def test_gateway_rate_limit_key_ignores_forged_session_token(monkeypatch):
    monkeypatch.setattr(main_module, "GATEWAY_SHARED_SECRET", "test-gateway-key")
    monkeypatch.setattr(main_module, "SESSION_SIGNING_SECRET", "test-session-key")
    request = _make_request(
        {
            "X-App-Key": "test-gateway-key",
            "X-Session-Token": "forged-token",
            "X-Forwarded-For": "198.51.100.10",
        }
    )

    key = main_module._rate_limit_key(request)

    assert key == "gateway_ip:198.51.100.10"


def test_gateway_rate_limit_key_uses_verified_session_token(monkeypatch):
    monkeypatch.setattr(main_module, "GATEWAY_SHARED_SECRET", "test-gateway-key")
    monkeypatch.setattr(main_module, "SESSION_SIGNING_SECRET", "test-session-key")
    _, session_token = issue_session_token("test-session-key")
    request = _make_request(
        {
            "X-App-Key": "test-gateway-key",
            "X-Session-Token": session_token,
            "X-Forwarded-For": "198.51.100.10",
        }
    )

    key = main_module._rate_limit_key(request)

    assert key.startswith("gateway_session:")


def test_gateway_rate_limit_is_per_verified_session(monkeypatch):
    monkeypatch.setattr(main_module, "GATEWAY_SHARED_SECRET", "test-gateway-key")
    monkeypatch.setattr(main_module, "SESSION_SIGNING_SECRET", "test-session-key")
    monkeypatch.setattr(main_module, "ASK_RATE_LIMIT_GATEWAY_PER_MINUTE", 1)
    _clear_rate_limit_buckets()

    _, session_token_one = issue_session_token("test-session-key")
    _, session_token_two = issue_session_token("test-session-key")
    request_one = _make_request(
        {
            "X-App-Key": "test-gateway-key",
            "X-Session-Token": session_token_one,
            "X-Forwarded-For": "198.51.100.10",
        }
    )
    request_two = _make_request(
        {
            "X-App-Key": "test-gateway-key",
            "X-Session-Token": session_token_two,
            "X-Forwarded-For": "198.51.100.10",
        }
    )
    request_one_again = _make_request(
        {
            "X-App-Key": "test-gateway-key",
            "X-Session-Token": session_token_one,
            "X-Forwarded-For": "198.51.100.10",
        }
    )

    assert main_module._check_gateway_rate_limit(request_one) is True
    assert main_module._check_gateway_rate_limit(request_two) is True
    assert main_module._check_gateway_rate_limit(request_one_again) is False
    _clear_rate_limit_buckets()


def test_gateway_rate_limit_invalid_session_token_falls_back_to_ip(monkeypatch):
    monkeypatch.setattr(main_module, "GATEWAY_SHARED_SECRET", "test-gateway-key")
    monkeypatch.setattr(main_module, "SESSION_SIGNING_SECRET", "test-session-key")
    monkeypatch.setattr(main_module, "ASK_RATE_LIMIT_GATEWAY_PER_MINUTE", 1)
    _clear_rate_limit_buckets()

    first = _make_request(
        {
            "X-App-Key": "test-gateway-key",
            "X-Session-Token": "forged-token-one",
            "X-Forwarded-For": "198.51.100.10",
        }
    )
    second = _make_request(
        {
            "X-App-Key": "test-gateway-key",
            "X-Session-Token": "forged-token-two",
            "X-Forwarded-For": "198.51.100.10",
        }
    )

    assert main_module._check_gateway_rate_limit(first) is True
    assert main_module._check_gateway_rate_limit(second) is False
    _clear_rate_limit_buckets()


def test_public_bearer_rate_limit_is_per_user_not_per_ip(monkeypatch):
    _install_successful_ask_mocks(monkeypatch)
    _set_public_bearer_auth(monkeypatch, True)
    monkeypatch.setattr(auth_module, "SUPABASE_JWT_SECRET", TEST_SUPABASE_JWT_SECRET)
    monkeypatch.setattr(main_module, "ASK_RATE_LIMIT_PUBLIC_PER_MINUTE", 1)
    monkeypatch.setattr(main_module, "ASK_RATE_LIMIT_PREAUTH_PER_MINUTE", 100)
    _clear_rate_limit_buckets()

    client = TestClient(main_module.app)
    token_one = _make_bearer_token(sub="00000000-0000-0000-0000-000000000001")
    token_two = _make_bearer_token(sub="00000000-0000-0000-0000-000000000002")

    first_user = client.post(
        "/ask",
        json={"query": "Show balancing price trend in 2024."},
        headers={"Authorization": f"Bearer {token_one}"},
    )
    second_user = client.post(
        "/ask",
        json={"query": "Show balancing price trend in 2024."},
        headers={"Authorization": f"Bearer {token_two}"},
    )
    first_user_again = client.post(
        "/ask",
        json={"query": "Show balancing price trend in 2024."},
        headers={"Authorization": f"Bearer {token_one}"},
    )

    assert first_user.status_code == 200
    assert second_user.status_code == 200
    assert first_user_again.status_code == 429
    _clear_rate_limit_buckets()


def test_bearer_auth_rejected_when_gateway_only_mode(monkeypatch):
    _install_successful_ask_mocks(monkeypatch)
    _set_public_bearer_auth(monkeypatch, False)
    monkeypatch.setattr(auth_module, "SUPABASE_JWT_SECRET", TEST_SUPABASE_JWT_SECRET)

    token = _make_bearer_token()
    client = TestClient(main_module.app)
    response = client.post(
        "/ask",
        json={"query": "Show balancing price trend in 2024."},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_evaluate_rejects_gateway_secret(monkeypatch):
    monkeypatch.setattr(main_module, "GATEWAY_SHARED_SECRET", "test-gateway-key")
    monkeypatch.setattr(main_module, "EVALUATE_ADMIN_SECRET", "test-evaluate-key")
    monkeypatch.setattr(main_module, "ENABLE_EVALUATE_ENDPOINT", True)
    client = TestClient(main_module.app)
    response = client.get("/evaluate", headers={"X-App-Key": "test-gateway-key"})
    assert response.status_code == 401


def test_metrics_disabled_and_auth_behavior(monkeypatch):
    monkeypatch.setattr(main_module, "ENABLE_METRICS_ENDPOINT", False)
    client = TestClient(main_module.app)

    disabled = client.get("/metrics")
    assert disabled.status_code == 404

    monkeypatch.setattr(main_module, "ENABLE_METRICS_ENDPOINT", True)
    monkeypatch.setattr(main_module, "EVALUATE_ADMIN_SECRET", "test-evaluate-key")

    unauthorized = client.get("/metrics")
    assert unauthorized.status_code == 401

    wrong_secret = client.get("/metrics", headers={"X-App-Key": "wrong"})
    assert wrong_secret.status_code == 401

    authorized = client.get("/metrics", headers={"X-App-Key": "test-evaluate-key"})
    assert authorized.status_code == 200
    assert authorized.json()["status"] == "healthy"


class TestTradeSharePivot:
    """Tests for the auto-pivot SQL helper used when share columns are hallucinated."""

    def test_basic_wrapping(self):
        original = (
            "SELECT date, share_renewable_ppa FROM trade_derived_entities "
            "WHERE segment = 'balancing'"
        )
        rewritten = build_trade_share_cte(original)
        assert rewritten.strip().startswith("WITH tde AS"), rewritten
        assert "FROM tde" in rewritten
        assert "WHERE segment = 'balancing'" in rewritten

    def test_preserves_existing_cte(self):
        original = (
            "WITH latest AS (SELECT * FROM trade_derived_entities) "
            "SELECT * FROM latest"
        )
        rewritten = build_trade_share_cte(original)
        assert rewritten.strip().startswith("WITH tde AS"), rewritten
        assert "latest AS" in rewritten

    def test_alias_survives_replacement(self):
        original = "SELECT t.date FROM trade_derived_entities t"
        rewritten = build_trade_share_cte(original)
        assert "FROM tde t" in rewritten

    def test_includes_aggregated_shares(self):
        original = "SELECT * FROM trade_derived_entities"
        rewritten = build_trade_share_cte(original)
        assert "share_all_ppa" in rewritten
        assert "share_all_renewables" in rewritten


class TestBalancingSharePanel:
    """Ensure the deterministic balancing share panel helper returns expected structure."""

    def test_fetch_balancing_share_panel_structure(self):
        rows = [
            (
                pd.Timestamp("2024-06-01"),
                "balancing",
                0.1,
                0.2,
                0.3,
                0.4,
                0.05,
                0.15,
                0.25,
                0.08,
                0.6,
                0.73,
                0.65,
            )
        ]
        cols = [
            "date",
            "segment",
            "share_import",
            "share_deregulated_hydro",
            "share_regulated_hpp",
            "share_regulated_new_tpp",
            "share_regulated_old_tpp",
            "share_renewable_ppa",
            "share_thermal_ppa",
            "share_cfd_scheme",
            "share_all_ppa",
            "share_all_renewables",
            "share_total_hpp",
        ]

        class FakeResult:
            def __init__(self, data, headers):
                self._data = data
                self._headers = headers

            def fetchall(self):
                return self._data

            def keys(self):
                return self._headers

        class FakeConn:
            def __init__(self, data, headers):
                self._data = data
                self._headers = headers
                self.last_sql = None

            def execute(self, clause):
                self.last_sql = str(clause).strip()
                return FakeResult(self._data, self._headers)

        conn = FakeConn(rows, cols)
        df = fetch_balancing_share_panel(conn)

        assert not df.empty
        assert list(df.columns) == cols
        assert conn.last_sql == BALANCING_SHARE_PIVOT_SQL
        assert pytest.approx(df.iloc[0]["share_renewable_ppa"], rel=1e-6) == 0.15
        assert pytest.approx(df.iloc[0]["share_cfd_scheme"], rel=1e-6) == 0.08
        assert BALANCING_SEGMENT_NORMALIZER in BALANCING_SHARE_PIVOT_SQL


class TestEnsureShareDataFrame:
    """Exercise the share dataframe resolver for both fast-path and fallback."""

    def test_returns_existing_dataframe_without_query(self):
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-06-01")],
                "share_import": [0.41],
            }
        )

        class NoOpConn:
            def execute(self, *_args, **_kwargs):
                raise AssertionError("Fallback should not execute when share columns exist")

        resolved, used_fallback = ensure_share_dataframe(df, NoOpConn())
        assert resolved is df
        assert used_fallback is False

    def test_fetches_fallback_when_missing_shares(self):
        rows = [
            (
                pd.Timestamp("2024-06-01"),
                "balancing",
                0.11,
                0.22,
            )
        ]
        cols = ["date", "segment", "share_import", "share_renewable_ppa"]

        class FakeResult:
            def __init__(self, data, headers):
                self._data = data
                self._headers = headers

            def fetchall(self):
                return self._data

            def keys(self):
                return self._headers

        class FakeConn:
            def __init__(self):
                self.calls = 0

            def execute(self, clause):
                self.calls += 1
                assert str(clause).strip() == BALANCING_SHARE_PIVOT_SQL
                return FakeResult(rows, cols)

        empty_df = pd.DataFrame()
        conn = FakeConn()
        resolved, used_fallback = ensure_share_dataframe(empty_df, conn)

        assert used_fallback is True
        assert conn.calls == 1
        assert list(resolved.columns) == cols
        assert pytest.approx(resolved.iloc[0]["share_renewable_ppa"], rel=1e-6) == 0.22


class TestShareSummaryOverride:
    """Validate deterministic share summaries for direct share questions."""

    def test_all_ppa_share_with_breakdown(self):
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-06-01")],
                "share_all_ppa": [0.32],
                "share_renewable_ppa": [0.2],
                "share_thermal_ppa": [0.12],
            }
        )
        plan = {
            "intent": "calculate_share",
            "target": "share of PPA in balancing electricity",
            "period": "2024-06",
        }
        summary = generate_share_summary(
            df,
            plan,
            "What was the share of PPA in balancing electricity in June 2024?",
        )
        assert summary is not None
        assert "June 2024" in summary
        assert "32.0%" in summary


class TestShareShiftNotes:
    """Ensure share shift helper produces explanatory notes for 'why' analysis."""

    def test_highlight_changes_and_usd_sources(self):
        cur = {
            "share_regulated_hpp": 0.30,
            "share_deregulated_hydro": 0.10,
            "share_renewable_ppa": 0.25,
            "share_thermal_ppa": 0.20,
            "share_import": 0.05,
        }
        prev = {
            "share_regulated_hpp": 0.45,
            "share_deregulated_hydro": 0.18,
            "share_renewable_ppa": 0.15,
            "share_thermal_ppa": 0.12,
            "share_import": 0.05,
        }

        notes = build_share_shift_notes(cur, prev)
        assert any("Share shifts month-over-month" in n for n in notes)
        assert any("Cheaper balancing supply contracted" in n for n in notes)
        assert any("Expensive balancing supply expanded" in n for n in notes)
        assert any("USD-denominated sellers" in n for n in notes)

    def test_specific_share_selection(self):
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-06-01")],
                "share_import": [0.41],
                "share_all_ppa": [0.33],
            }
        )
        plan = {
            "intent": "calculate_share",
            "target": "share of import in balancing electricity",
            "period": "2024-06",
        }
        summary = generate_share_summary(
            df,
            plan,
            "What share did import have in balancing electricity during June 2024?",
        )
        assert summary is not None
        assert "41.0%" in summary
        assert "Import" in summary or "Imports" in summary


class TestBalancingCompositionTool:
    def test_uses_canonical_balancing_segment(self, monkeypatch):
        captured = {}

        def _fake_run(sql, params):
            captured["sql"] = sql
            captured["params"] = params
            return pd.DataFrame(), [], []

        monkeypatch.setattr(composition_tools, "run_text_query", _fake_run)
        composition_tools.get_balancing_composition(
            start_date="2024-01-01",
            end_date="2024-12-31",
            entities=["import"],
        )

        assert "LOWER(REPLACE(segment, ' ', '_')) = 'balancing'" in captured["sql"]
        assert "'balancing'::text AS segment" in captured["sql"]
        assert captured["params"]["start_date"] == "2024-01-01"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
