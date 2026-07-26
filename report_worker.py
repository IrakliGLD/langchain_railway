"""Opt-in, separate-process worker for durable analytical report jobs."""

from __future__ import annotations

import logging
import os
import re
import signal
import socket
import threading
import urllib.parse
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from config import (
    DB_CONNECT_TIMEOUT_SECONDS,
    DB_POOL_TIMEOUT_SECONDS,
    DB_STATEMENT_TIMEOUT_MS,
    REPORT_SECTION_MAX_WORKERS,
    REPORT_WORKER_DB_URL,
    REPORT_WORKER_ENABLED,
    REPORT_WORKER_LEASE_SECONDS,
    REPORT_WORKER_POLL_INTERVAL_MS,
    REPORT_WORKER_RETRY_DELAY_SECONDS,
)
from core.db_gateway import database_connection
from core.report_job_processor import ReportJobProcessor
from core.report_job_repository import PostgresReportJobRepository
from core.report_job_worker import ReportJobWorker

log = logging.getLogger("Enai.ReportWorker")


def _coerce_worker_db_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("postgres", "postgresql"):
        return url.replace(parsed.scheme, "postgresql+psycopg", 1)
    if not parsed.scheme.startswith("postgresql+"):
        return "postgresql+psycopg://" + url.split("://", 1)[-1]
    return url


def _worker_id() -> str:
    raw = f"{socket.gethostname()}:{os.getpid()}"
    normalized = re.sub(r"[^A-Za-z0-9._:-]", "-", raw)[:128]
    return normalized if normalized[0].isalnum() else f"worker-{os.getpid()}"


def build_report_worker_runtime() -> tuple[
    ReportJobWorker,
    ReportJobProcessor,
    Any,
]:
    if not REPORT_WORKER_ENABLED:
        raise RuntimeError(
            "Report worker runtime cannot be built while "
            "ENAI_REPORT_WORKER_ENABLED is false."
        )
    if not REPORT_WORKER_DB_URL:
        raise RuntimeError(
            "ENAI_REPORT_WORKER_DB_URL is required for the write-capable "
            "report worker service."
        )

    engine = create_engine(
        _coerce_worker_db_url(REPORT_WORKER_DB_URL),
        poolclass=QueuePool,
        pool_size=2,
        max_overflow=0,
        pool_timeout=DB_POOL_TIMEOUT_SECONDS,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={
            "connect_timeout": DB_CONNECT_TIMEOUT_SECONDS,
            "options": f"-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}",
            "prepare_threshold": None,
        },
    )
    repository = PostgresReportJobRepository(
        engine=engine,
        database_connection=database_connection,
    )
    processor = ReportJobProcessor(
        max_section_workers=REPORT_SECTION_MAX_WORKERS,
    )
    worker = ReportJobWorker(
        repository=repository,
        worker_id=_worker_id(),
        lease_seconds=REPORT_WORKER_LEASE_SECONDS,
        retry_delay_seconds=REPORT_WORKER_RETRY_DELAY_SECONDS,
        poll_interval_seconds=REPORT_WORKER_POLL_INTERVAL_MS / 1000,
        logger=log,
    )
    return worker, processor, engine


def main() -> int:
    if not REPORT_WORKER_ENABLED:
        log.info(
            "Report worker is disabled; set ENAI_REPORT_WORKER_ENABLED=true "
            "only on the dedicated worker service."
        )
        return 0

    worker, processor, engine = build_report_worker_runtime()
    stop_event = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    log.info("Report worker started.")
    try:
        worker.run_until_stopped(processor, stop_event=stop_event)
    finally:
        engine.dispose()
        log.info("Report worker stopped.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    raise SystemExit(main())
