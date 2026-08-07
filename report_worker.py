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

import knowledge as knowledge_module
from config import (
    DB_CONNECT_TIMEOUT_SECONDS,
    DB_POOL_TIMEOUT_SECONDS,
    DB_STATEMENT_TIMEOUT_MS,
    ENABLE_REPORT_PARTIAL_TRACK_EVIDENCE,
    REPORT_JOB_TIMEOUT_SECONDS,
    REPORT_MAX_GENERATIVE_CALLS,
    REPORT_PIPELINE_V2_MODE,
    REPORT_RESEARCH_MAX_TRACKS,
    REPORT_RESEARCH_MAX_WORKERS,
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
from knowledge.embedding_service import require_embedding_capability

log = logging.getLogger("Enai.ReportWorker")
_REPORT_WORKER_LEASE_SAFETY_MARGIN_SECONDS = 30
_REPORT_WORKER_SHUTDOWN_HANDOFF_WAIT_SECONDS = 10


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
    if (
        REPORT_WORKER_LEASE_SECONDS
        < REPORT_JOB_TIMEOUT_SECONDS
        + _REPORT_WORKER_LEASE_SAFETY_MARGIN_SECONDS
    ):
        raise RuntimeError(
            "ENAI_REPORT_WORKER_LEASE_SECONDS must be at least "
            "ENAI_REPORT_JOB_TIMEOUT_SECONDS plus 30 seconds."
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
        job_timeout_seconds=REPORT_JOB_TIMEOUT_SECONDS,
        pipeline_v2_mode=REPORT_PIPELINE_V2_MODE,
        max_generative_calls=REPORT_MAX_GENERATIVE_CALLS,
        max_research_tracks=REPORT_RESEARCH_MAX_TRACKS,
        max_research_workers=REPORT_RESEARCH_MAX_WORKERS,
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

    knowledge_module.load_knowledge()
    require_embedding_capability()
    worker, processor, engine = build_report_worker_runtime()
    stop_event = threading.Event()
    handoff_finished = threading.Event()

    def handoff_active_job() -> None:
        stop_event.wait()
        try:
            worker.handoff_active_job_for_shutdown()
        finally:
            handoff_finished.set()

    handoff_thread = threading.Thread(
        target=handoff_active_job,
        name="report-worker-shutdown-handoff",
        daemon=True,
    )
    handoff_thread.start()

    def request_stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    # Every flag that changes report behaviour belongs on this line. An
    # operator who sets one and cannot see it took effect is left reading
    # outcomes that look identical to the flag never having been read at all.
    log.info(
        "Report worker started. section_workers=%s job_timeout_seconds=%s "
        "pipeline_v2_mode=%s generative_call_budget=%s "
        "research_max_tracks=%s research_max_workers=%s "
        "partial_track_evidence=%s",
        REPORT_SECTION_MAX_WORKERS,
        REPORT_JOB_TIMEOUT_SECONDS,
        REPORT_PIPELINE_V2_MODE,
        REPORT_MAX_GENERATIVE_CALLS,
        REPORT_RESEARCH_MAX_TRACKS,
        REPORT_RESEARCH_MAX_WORKERS,
        "enabled" if ENABLE_REPORT_PARTIAL_TRACK_EVIDENCE else "disabled",
    )
    try:
        worker.run_until_stopped(processor, stop_event=stop_event)
    finally:
        stop_event.set()
        if not handoff_finished.wait(
            timeout=_REPORT_WORKER_SHUTDOWN_HANDOFF_WAIT_SECONDS
        ):
            log.error("Report worker shutdown handoff did not finish before timeout.")
        engine.dispose()
        log.info("Report worker stopped.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    raise SystemExit(main())
