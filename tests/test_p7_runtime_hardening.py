"""P7.A database identity, packaging, and deployment invariants."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from config import STATIC_ALLOWED_TABLES
from core import query_executor
from core.query_executor import DatabaseRuntimeIdentity

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_database_identity_requires_expected_role_and_read_only_default():
    identity = DatabaseRuntimeIdentity(
        current_user="enai_api_readonly",
        expected_user="enai_api_readonly",
        default_transaction_read_only=True,
    )
    assert identity.ready is True

    assert identity.__class__(
        current_user="postgres",
        expected_user="enai_api_readonly",
        default_transaction_read_only=True,
    ).ready is False
    assert identity.__class__(
        current_user="enai_api_readonly",
        expected_user="enai_api_readonly",
        default_transaction_read_only=False,
    ).ready is False
    assert identity.__class__(
        current_user="development_role",
        expected_user="",
        default_transaction_read_only=False,
    ).ready is True


def test_runtime_database_identity_reads_the_connected_role(monkeypatch):
    class _Result:
        def mappings(self):
            return self

        def one(self):
            return {
                "current_user": "enai_api_readonly",
                "default_read_only": "on",
            }

    class _Connection:
        def execute(self, _statement):
            return _Result()

    @contextmanager
    def _connection(*_args, **_kwargs):
        yield _Connection()

    monkeypatch.setattr(query_executor, "DATABASE_RUNTIME_ROLE", "enai_api_readonly")
    monkeypatch.setattr(query_executor, "database_connection", _connection)

    identity = query_executor.get_database_runtime_identity()

    assert identity.ready is True
    assert identity.protected_metadata() == {
        "current_user": "enai_api_readonly",
        "expected_user": "enai_api_readonly",
        "role_matches": True,
        "default_transaction_read_only": True,
        "ready": True,
    }


def test_runtime_database_identity_failure_is_not_reported_ready(monkeypatch):
    @contextmanager
    def _connection(*_args, **_kwargs):
        raise RuntimeError("private connection failure")
        yield

    monkeypatch.setattr(query_executor, "DATABASE_RUNTIME_ROLE", "enai_api_readonly")
    monkeypatch.setattr(query_executor, "database_connection", _connection)

    identity = query_executor.get_database_runtime_identity()

    assert identity.current_user == ""
    assert identity.ready is False
    assert query_executor.is_database_available() is False


def test_query_executor_enforces_dataframe_memory_limit():
    frame = pd.DataFrame({"value": ["x" * 4096]})

    with pytest.raises(HTTPException) as exc:
        query_executor.check_dataframe_memory(frame, max_mb=0)

    assert exc.value.status_code == 413
    query_executor.check_dataframe_memory(frame, max_mb=1)


def test_query_executor_fetches_in_batches_and_caps_rows(monkeypatch):
    class _Result:
        def __init__(self):
            self._batches = [[(1,), (2,)], [(3,), (4,)], []]

        def keys(self):
            return ["value"]

        def fetchmany(self, _batch_size):
            return self._batches.pop(0)

    class _Connection:
        def __init__(self):
            self.result = _Result()
            self.statements = []

        def execute(self, statement):
            self.statements.append(str(statement))
            if len(self.statements) == 1:
                return None
            return self.result

    connection = _Connection()

    @contextmanager
    def _connection(*_args, **_kwargs):
        yield connection

    clock = iter([10.0, 10.25])
    monkeypatch.setattr(query_executor, "database_connection", _connection)
    monkeypatch.setattr(query_executor, "MAX_ROWS", 3)
    monkeypatch.setattr(query_executor.time, "time", lambda: next(clock))

    frame, columns, rows, elapsed = query_executor.execute_sql_safely("select value")

    assert connection.statements[0] == "SET TRANSACTION READ ONLY"
    assert columns == ["value"]
    assert rows == [(1,), (2,), (3,)]
    assert frame.to_dict(orient="records") == [
        {"value": 1},
        {"value": 2},
        {"value": 3},
    ]
    assert elapsed == 0.25


def test_runtime_role_script_is_read_only_and_covers_denial_controls():
    sql = (ROOT / "scripts" / "least_privilege_api_role.sql").read_text(encoding="utf-8").lower()

    assert "alter role enai_api_readonly set default_transaction_read_only = on" in sql
    assert "alter role enai_api_readonly noinherit" in sql
    assert "rolsuper or rolreplication or rolbypassrls" in sql
    assert "alter role enai_api_readonly noinherit nosuperuser" not in sql
    assert "grant select on" in sql
    assert "grant insert" not in sql
    assert "grant update" not in sql
    assert "grant delete" not in sql
    assert "change_me_strong_secret" not in sql
    assert (ROOT / "scripts" / "verify_runtime_database_role.py").is_file()
    assert {f"public.{table}" for table in STATIC_ALLOWED_TABLES} == {
        token.rstrip(",")
        for token in sql.split()
        if token.startswith("public.") and token.rstrip(",") != "public.function"
    }


def test_backend_container_is_pinned_non_root_and_uses_runtime_dependencies_only():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11.15-slim-bookworm@sha256:" in dockerfile
    assert "COPY requirements.txt" in dockerfile
    assert "requirements-dev.txt" not in dockerfile
    assert "USER enai" in dockerfile
    assert "EXPOSE 3000" in dockerfile
    assert "COPY . ." not in dockerfile
    assert "prompts ./prompts" not in dockerfile


def test_docker_context_and_railway_config_are_fail_closed():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for excluded in [".git", ".env*", "tests/", "docs/", "privacy_exports/", "*.log"]:
        assert excluded in dockerignore

    railway = json.loads((ROOT / "railway.json").read_text(encoding="utf-8"))
    assert railway["build"]["builder"] == "DOCKERFILE"
    assert railway["build"]["dockerfilePath"] == "Dockerfile"
    assert railway["deploy"]["startCommand"] == "python main.py"
    assert railway["deploy"]["healthcheckPath"] == "/readyz"
    assert railway["deploy"]["overlapSeconds"] == 0
    assert railway["deploy"]["drainingSeconds"] == 30
    assert "buildCommand" not in railway["build"]


def test_release_evidence_workflow_builds_exact_sha_and_emits_scan_artifacts():
    workflow = (ROOT / ".github" / "workflows" / "backend-release-evidence.yml").read_text(encoding="utf-8")

    assert "git_ref" in workflow
    assert "docker build" in workflow
    assert "pip-audit" in workflow
    assert "cyclonedx-py" in workflow
    assert "backend-sbom.cdx.json" in workflow
    assert "backend-release-manifest.json" in workflow
    assert "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5" in workflow
    assert workflow.index("mkdir -p artifacts") < workflow.index("cyclonedx-py")
    assert "if docker image inspect" in workflow


def test_runtime_role_verifier_rolls_back_unexpectedly_allowed_probes():
    verifier = (ROOT / "scripts" / "verify_runtime_database_role.py").read_text(
        encoding="utf-8"
    )

    assert 'conn.execute("begin")' in verifier
    assert "finally:" in verifier
    assert "conn.rollback()" in verifier
