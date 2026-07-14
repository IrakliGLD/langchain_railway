from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError, ProgrammingError

from core import db_gateway


class _Breaker:
    def __init__(self, *, allowed: bool = True, reason: str = "closed") -> None:
        self.allowed = allowed
        self.reason = reason
        self.successes = 0
        self.failures = 0

    def allow_request(self):
        return self.allowed, self.reason

    def record_success(self):
        self.successes += 1

    def record_failure(self):
        self.failures += 1


class _ConnectionContext:
    def __init__(self, connection) -> None:
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, traceback):
        return False


class _Engine:
    def __init__(self, connection=None) -> None:
        self.connection = object() if connection is None else connection
        self.connect_calls = 0
        self.begin_calls = 0

    def connect(self):
        self.connect_calls += 1
        return _ConnectionContext(self.connection)

    def begin(self):
        self.begin_calls += 1
        return _ConnectionContext(self.connection)


class _FailingEngine(_Engine):
    def connect(self):
        self.connect_calls += 1
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


def test_open_database_circuit_never_touches_engine(monkeypatch):
    breaker = _Breaker(allowed=False, reason="open")
    engine = _Engine()
    monkeypatch.setattr(db_gateway, "db_circuit_breaker", breaker)

    with pytest.raises(HTTPException) as exc:
        with db_gateway.database_connection(engine, operation="typed_tool"):
            pytest.fail("open breaker must not yield a connection")

    assert exc.value.status_code == 503
    assert engine.connect_calls == 0
    assert breaker.successes == 0
    assert breaker.failures == 0


def test_invalid_sql_does_not_count_as_infrastructure_failure(monkeypatch):
    breaker = _Breaker()
    engine = _Engine()
    monkeypatch.setattr(db_gateway, "db_circuit_breaker", breaker)

    with pytest.raises(ProgrammingError):
        with db_gateway.database_connection(engine, operation="fallback_sql"):
            raise ProgrammingError("SELECT broken", {}, Exception("syntax error"))

    assert breaker.failures == 0
    assert breaker.successes == 1


def test_connection_failure_counts_toward_database_breaker(monkeypatch):
    breaker = _Breaker()
    engine = _Engine()
    monkeypatch.setattr(db_gateway, "db_circuit_breaker", breaker)

    with pytest.raises(OperationalError):
        with db_gateway.database_connection(engine, operation="vector_search"):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    assert breaker.failures == 1
    assert breaker.successes == 0


def test_connection_acquisition_failure_releases_half_open_probe(monkeypatch):
    breaker = _Breaker()
    engine = _FailingEngine()
    monkeypatch.setattr(db_gateway, "db_circuit_breaker", breaker)

    with pytest.raises(OperationalError):
        with db_gateway.database_connection(engine, operation="readiness_probe"):
            pytest.fail("failed acquisition must not yield a connection")

    assert engine.connect_calls == 1
    assert breaker.failures == 1
    assert breaker.successes == 0


def test_successful_transaction_records_success(monkeypatch):
    breaker = _Breaker()
    engine = _Engine()
    monkeypatch.setattr(db_gateway, "db_circuit_breaker", breaker)

    with db_gateway.database_connection(engine, operation="vector_ingest", begin=True):
        pass

    assert engine.begin_calls == 1
    assert engine.connect_calls == 0
    assert breaker.successes == 1
    assert breaker.failures == 0


def test_runtime_database_paths_do_not_bypass_gateway():
    repository_root = Path(__file__).resolve().parents[1]
    runtime_paths = [
        repository_root / "main.py",
        *(repository_root / "agent").rglob("*.py"),
        *(repository_root / "knowledge").rglob("*.py"),
        *(repository_root / "core").rglob("*.py"),
    ]
    bypass_pattern = re.compile(
        r"(?:\b[A-Za-z_]*ENGINE\b|\bengine\b|_resolve_engine\(\))"
        r"\.(?:connect|begin)\(\)"
    )
    violations = []

    for path in runtime_paths:
        if path.name == "db_gateway.py":
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if bypass_pattern.search(line):
                violations.append(f"{path.relative_to(repository_root)}:{line_number}")

    assert violations == []


@pytest.mark.parametrize(
    ("sqlstate", "expected"),
    [
        ("08006", True),
        ("40001", True),
        ("40P01", True),
        ("57014", True),
        ("57P03", True),
        ("42601", False),
        ("42P01", False),
        ("23505", False),
    ],
)
def test_sqlstate_classification(sqlstate, expected):
    class _DriverError(Exception):
        pass

    original = _DriverError("driver error")
    original.sqlstate = sqlstate
    error = ProgrammingError("statement", {}, original)

    assert db_gateway.is_transient_database_error(error) is expected
