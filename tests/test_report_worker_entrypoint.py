"""Separate-process report worker startup contract."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

import report_worker


def test_disabled_report_worker_is_a_noop(monkeypatch):
    monkeypatch.setattr(report_worker, "REPORT_WORKER_ENABLED", False)
    monkeypatch.setattr(
        report_worker,
        "build_report_worker_runtime",
        lambda: pytest.fail("disabled worker must not build runtime state"),
    )

    assert report_worker.main() == 0


def test_worker_requires_its_own_write_capable_database_url(monkeypatch):
    monkeypatch.setattr(report_worker, "REPORT_WORKER_ENABLED", True)
    monkeypatch.setattr(report_worker, "REPORT_WORKER_DB_URL", None)

    with pytest.raises(RuntimeError, match="ENAI_REPORT_WORKER_DB_URL"):
        report_worker.build_report_worker_runtime()


def test_worker_runtime_wires_repository_processor_and_bounded_pool(monkeypatch):
    captured = {}
    engine = SimpleNamespace(dispose=lambda: None)
    repository = object()
    processor = object()
    worker = object()

    monkeypatch.setattr(report_worker, "REPORT_WORKER_ENABLED", True)
    monkeypatch.setattr(
        report_worker,
        "REPORT_WORKER_DB_URL",
        "postgresql://writer:secret@localhost/enai",
    )
    monkeypatch.setattr(
        report_worker,
        "create_engine",
        lambda url, **kwargs: (
            captured.update(url=url, engine_kwargs=kwargs) or engine
        ),
    )
    monkeypatch.setattr(
        report_worker,
        "PostgresReportJobRepository",
        lambda **kwargs: (
            captured.update(repository_kwargs=kwargs) or repository
        ),
    )
    monkeypatch.setattr(
        report_worker,
        "ReportJobProcessor",
        lambda **kwargs: (
            captured.update(processor_kwargs=kwargs) or processor
        ),
    )
    monkeypatch.setattr(
        report_worker,
        "ReportJobWorker",
        lambda **kwargs: (
            captured.update(worker_kwargs=kwargs) or worker
        ),
    )

    runtime = report_worker.build_report_worker_runtime()

    assert runtime == (worker, processor, engine)
    assert captured["url"].startswith("postgresql+psycopg://")
    assert captured["engine_kwargs"]["pool_size"] == 2
    assert captured["engine_kwargs"]["max_overflow"] == 0
    assert captured["repository_kwargs"]["engine"] is engine
    assert captured["worker_kwargs"]["repository"] is repository
    assert captured["processor_kwargs"]["max_section_workers"] == (
        report_worker.REPORT_SECTION_MAX_WORKERS
    )
