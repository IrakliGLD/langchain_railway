"""Separate-process report worker startup contract."""

from __future__ import annotations

import os
import signal
import threading
from types import SimpleNamespace

import pytest

import knowledge as knowledge_module

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


def test_worker_rejects_a_job_deadline_that_can_outlive_its_lease(monkeypatch):
    monkeypatch.setattr(report_worker, "REPORT_WORKER_ENABLED", True)
    monkeypatch.setattr(
        report_worker,
        "REPORT_WORKER_DB_URL",
        "postgresql://writer:secret@localhost/enai",
    )
    monkeypatch.setattr(report_worker, "REPORT_WORKER_LEASE_SECONDS", 600)
    monkeypatch.setattr(report_worker, "REPORT_JOB_TIMEOUT_SECONDS", 600)

    with pytest.raises(
        RuntimeError,
        match="ENAI_REPORT_WORKER_LEASE_SECONDS",
    ):
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
    monkeypatch.setattr(report_worker, "REPORT_WORKER_LEASE_SECONDS", 630)
    monkeypatch.setattr(report_worker, "REPORT_JOB_TIMEOUT_SECONDS", 600)
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
    assert captured["processor_kwargs"]["job_timeout_seconds"] == (
        report_worker.REPORT_JOB_TIMEOUT_SECONDS
    )
    assert captured["worker_kwargs"]["lease_seconds"] == 630


def test_enabled_worker_initializes_process_knowledge_before_polling(monkeypatch):
    events = []
    engine = SimpleNamespace(dispose=lambda: events.append("disposed"))
    worker = SimpleNamespace(
        run_until_stopped=lambda _processor, *, stop_event: events.append(
            "polled"
        ),
        handoff_active_job_for_shutdown=lambda: events.append("handed_off"),
    )

    monkeypatch.setattr(report_worker, "REPORT_WORKER_ENABLED", True)
    monkeypatch.setattr(
        knowledge_module,
        "load_knowledge",
        lambda: events.append("knowledge_loaded"),
    )
    monkeypatch.setattr(
        report_worker,
        "build_report_worker_runtime",
        lambda: (worker, object(), engine),
    )
    monkeypatch.setattr(report_worker.signal, "signal", lambda *_args: None)

    assert report_worker.main() == 0
    assert events == [
        "knowledge_loaded",
        "polled",
        "handed_off",
        "disposed",
    ]


def test_sigterm_triggers_active_job_handoff_before_worker_exit(monkeypatch):
    events = []
    handlers = {}
    handoff_finished = threading.Event()
    engine = SimpleNamespace(dispose=lambda: events.append("disposed"))

    class Worker:
        def run_until_stopped(self, _processor, *, stop_event):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            assert stop_event.is_set()
            assert handoff_finished.wait(timeout=2)
            events.append("polled")

        def handoff_active_job_for_shutdown(self):
            events.append("handed_off")
            handoff_finished.set()
            return True

    monkeypatch.setattr(report_worker, "REPORT_WORKER_ENABLED", True)
    monkeypatch.setattr(knowledge_module, "load_knowledge", lambda: None)
    monkeypatch.setattr(
        report_worker,
        "build_report_worker_runtime",
        lambda: (Worker(), object(), engine),
    )
    monkeypatch.setattr(
        report_worker.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )

    assert report_worker.main() == 0
    assert events == ["handed_off", "polled", "disposed"]
