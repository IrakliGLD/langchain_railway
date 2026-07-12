# main.py v20.0 — Pipeline Architecture
# Phase 6: Pipeline extraction complete
#
# REFACTORING STATUS:
# ✅ Phase 1-5 Complete: ~2,850 lines extracted to 12 modules
# ✅ Phase 6: ask_post decomposed into 5-stage pipeline under agent/
#    planner → sql_executor → analyzer → summarizer → chart_pipeline
#    Orchestrated by agent.pipeline.process_query()
#
# Extracted Modules:
# - config.py: All configuration, constants, regex patterns
# - models.py: Pydantic models (Question, APIResponse, MetricsResponse)
# - utils/metrics.py: Metrics tracking class
# - utils/language.py: Language detection (Georgian/Russian/English)
# - core/query_executor.py: ENGINE, execute_sql_safely
# - core/sql_generator.py: SQL validation, sanitization, repair
# - core/llm.py: LLM instances, caching, SQL generation, summarization (983 lines!)
# - analysis/stats.py: Statistical analysis, trend calculation
# - analysis/seasonal.py: Seasonal analysis (summer/winter)
# - analysis/shares.py: Entity shares, price decomposition
# - visualization/chart_selector.py: Chart type selection logic
# - visualization/chart_builder.py: Chart data preparation
#
# All function calls in ask_post() and other endpoints now use imported modules.

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Phase 1D Security: Rate limiting
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError, OperationalError, SQLAlchemyError
from starlette.middleware.base import BaseHTTPMiddleware

import knowledge as knowledge_module

# Phase 6: Pipeline
from agent.answer_provenance import build_answer_provenance
from agent.contract_continuity import continuity_snapshot_json
from agent.pipeline import process_query
from analysis.seasonal import compute_seasonal_average
from analysis.seasonal_stats import calculate_seasonal_stats, detect_monthly_timeseries, format_seasonal_stats
from analysis.shares import (
    build_balancing_correlation_df,
    compute_entity_price_contributions,
    compute_weighted_balancing_price,
)

# Phase 3: Analysis modules
from analysis.stats import quick_stats, rows_to_preview

# ============================================================================
# REFACTORED MODULES (Phases 1-4)
# ============================================================================
# Phase 1: Configuration constants (explicit imports — Q5, 2026-06-10; the
# former star import hid this dependency surface behind a ruff ignore)
from config import (
    ALLOWED_TABLES,
    ASK_RATE_LIMIT_GATEWAY_PER_MINUTE,
    ASK_RATE_LIMIT_PREAUTH_PER_MINUTE,
    ASK_RATE_LIMIT_PUBLIC_PER_MINUTE,
    ENABLE_EVALUATE_ENDPOINT,
    ENABLE_METRICS_ENDPOINT,
    ENABLE_PUBLIC_BEARER_AUTH,
    ENAI_AUTH_MODE,
    EVALUATE_ADMIN_SECRET,
    GATEWAY_SHARED_SECRET,
    GEMINI_MODEL,
    MAX_REQUEST_BODY_BYTES,
    MODEL_TYPE,
    NVIDIA_MODEL,
    OPENAI_MODEL,
    SESSION_SIGNING_SECRET,
    STATIC_ALLOWED_TABLES,
    SUPABASE_JWT_SECRET,
    TRUST_PROXY_CLIENT_IP,
)

# Schema & helpers
from context import (
    COLUMN_LABELS,
    DB_SCHEMA_DICT,
    DB_SCHEMA_DOC,
    DERIVED_LABELS,
    scrub_schema_mentions,
)
from core.llm import (
    classify_query_type,
    get_primary_model_name,
    get_query_focus,
    llm_cache,
    llm_generate_plan_and_sql,
    llm_summarize,
    make_gemini,
    make_openai,
)
from core.query_executor import ENGINE, execute_sql_safely, is_database_available
from core.sql_generator import plan_validate_repair, sanitize_sql, simple_table_whitelist_check
from guardrails.firewall import build_safe_refusal_message, inspect_query
from models import APIResponse, MetricsResponse, Question
from utils.auth import CallerContext, authenticate_request
from utils.language import detect_language, get_language_instruction

# Phase 2: Core modules
from utils.metrics import metrics
from utils.query_validation import is_conceptual_question, should_skip_sql_execution, validate_sql_relevance
from utils.resilience import get_resilience_snapshot, request_backpressure_gate
from utils.session_memory import (
    append_exchange,
    get_history,
    get_last_contract,
    get_or_issue_session,
    resolve_session_token,
    seed_history,
    set_last_contract,
)
from visualization.chart_builder import prepare_chart_data

# Phase 4: Visualization modules
from visualization.chart_selector import detect_column_types, infer_dimension, select_chart_type, should_generate_chart

# ============================================================================

# -----------------------------
# Boot + Config
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("Enai")


class _DropUvicornAccess404(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return int(record.args[4]) != 404  # type: ignore[index]
        except (TypeError, ValueError, IndexError):
            return True


logging.getLogger("uvicorn.access").addFilter(_DropUvicornAccess404())

# Load knowledge files from knowledge/ directory at startup
knowledge_module.load_knowledge()

resolved_auth_mode = "gateway_and_bearer" if ENABLE_PUBLIC_BEARER_AUTH else "gateway_only"
log.info("Auth mode enabled: %s", resolved_auth_mode)

if ENABLE_METRICS_ENDPOINT:
    log.warning("/metrics endpoint enabled; keep it admin-only and network-restricted")
if ENABLE_EVALUATE_ENDPOINT:
    log.warning("/evaluate endpoint enabled; keep it disabled in production or isolate it to an admin worker")

if not ENABLE_PUBLIC_BEARER_AUTH and SUPABASE_JWT_SECRET and ENAI_AUTH_MODE == "gateway_only":
    log.info("SUPABASE_JWT_SECRET loaded but bearer auth is disabled by ENAI_AUTH_MODE")

# Request ID tracking for observability
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# CORS Configuration: Parse allowed origins from environment
# Default to localhost for development if not specified
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]  # Clean up whitespace
log.info(f"🔒 CORS: Allowed origins: {ALLOWED_ORIGINS}")

# Note: Metrics, config variables, and other extracted code now imported from modules
# See imports section above for: config.py, models.py, utils.metrics, core.*, analysis.*, visualization.*


# ---------------------------------------------------------------------------
# Re-exports: functions moved to agent/ but kept importable from main
# for backward compatibility (tests, evaluate endpoint, etc.)
# ---------------------------------------------------------------------------
from agent.analyzer import (
    BALANCING_SHARE_METADATA,
    MONTH_NAME_TO_NUMBER,
    build_share_shift_notes,
    generate_share_summary,
)
from agent.planner import (  # noqa: F401 — backward compat re-exports
    ANALYTICAL_KEYWORDS,
    detect_analysis_mode,
)
from agent.sql_executor import (
    BALANCING_SEGMENT_NORMALIZER,
    BALANCING_SHARE_PIVOT_SQL,
    build_trade_share_cte,
    ensure_share_dataframe,
    fetch_balancing_share_panel,
    should_inject_balancing_pivot,
)

# Intentionally unused import to keep backward compat:
_ = should_inject_balancing_pivot  # noqa: F811

# ---------------------------------------------------------------------------
# Original function definitions (lines 137-737 of v19.0) have been moved:
#   should_inject_balancing_pivot, build_trade_share_cte,
#   fetch_balancing_share_panel, ensure_share_dataframe        → agent/sql_executor.py
#   generate_share_summary, build_share_shift_notes,
#   _parse_period_hint, _select_share_column                   → agent/analyzer.py
#   detect_analysis_mode, ANALYTICAL_KEYWORDS                  → agent/planner.py
#   BALANCING_SHARE_METADATA, MONTH_NAME_TO_NUMBER,
#   BALANCING_SHARE_PIVOT_SQL, BALANCING_SEGMENT_NORMALIZER    → agent/ constants
# ---------------------------------------------------------------------------

# -----------------------------
# Database Schema Reflection
# -----------------------------
SCHEMA_MAP: Dict[str, set] = {}
REQUIRED_SCHEMA_COLUMNS = {
    table_name: set(DB_SCHEMA_DICT["views"][table_name]["columns"])
    for table_name in STATIC_ALLOWED_TABLES
}


def required_schema_is_ready(schema_map: Dict[str, set]) -> bool:
    """Return whether every allow-listed relation exposes its required columns."""
    return all(
        required_columns.issubset(set(schema_map.get(table_name) or set()))
        for table_name, required_columns in REQUIRED_SCHEMA_COLUMNS.items()
    )


def refresh_schema_map() -> bool:
    """Best-effort schema reflection; never raises to caller."""
    global SCHEMA_MAP
    try:
        with ENGINE.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT m.matviewname AS view_name, a.attname AS column_name
                    FROM pg_matviews m
                    JOIN pg_attribute a ON m.matviewname::regclass = a.attrelid
                    WHERE a.attnum > 0 AND NOT a.attisdropped
                    AND m.schemaname = 'public';
                    """
                )
            )
            rows = result.fetchall()

        schema_map: Dict[str, set] = {}
        for view_name, column_name in rows:
            schema_map.setdefault(str(view_name).lower(), set()).add(str(column_name).lower())

        SCHEMA_MAP = schema_map
        # Reflection describes availability; it must never expand the explicit
        # SQL security allow-list when an unrelated materialized view appears.
        # The allow-list is initialized from STATIC_ALLOWED_TABLES and is not
        # mutated during refresh, avoiding a transient empty set under
        # concurrent readiness probes.
        log.info("Schema reflection complete: %s views", len(SCHEMA_MAP))
        return True
    except Exception as exc:
        SCHEMA_MAP = {}
        log.warning("Schema reflection unavailable: %s", exc)
        return False


# -----------------------------
# Analysis Functions: Imported from analysis/* modules (lines 83-88)
# -----------------------------
# build_balancing_correlation_df, compute_weighted_balancing_price,
# compute_entity_price_contributions, compute_seasonal_average,
# compute_share_changes - all imported from analysis modules

# -----------------------------
# App
# -----------------------------
# Single source of truth for the app version (Q5, 2026-06-10 — the file-header
# comment and FastAPI(version=...) previously disagreed: "v20.0" vs "18.6").
__version__ = "20.0"
CHAT_GATEWAY_CONTRACT_VERSION = "chat-gateway-v1"
_SAFE_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def resolve_request_id(candidate: Optional[str]) -> str:
    """Preserve a safe caller correlation ID or create a new UUID.

    Request IDs are observability metadata, never authentication material. The
    restricted alphabet prevents control-character/log-injection problems while
    retaining UUIDs and the browser's ``req-...`` fallback format.
    """
    if candidate and _SAFE_REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return str(uuid.uuid4())


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Startup hook: prime schema cache and validate skills.

    Replaces the deprecated ``@app.on_event("startup")`` (Q5, 2026-06-10).
    """
    refresh_schema_map()
    from skills.loader import validate_skills, warmup_cache
    validate_skills()
    warmup_cache()
    yield


app = FastAPI(title="Enai Analyst (Gemini)", version=__version__, lifespan=_lifespan)
security_log = logging.getLogger("EnaiSecurity")


def log_security_event(event_type: str, request: Optional[Request] = None, **details: Any) -> None:
    """Emit structured security logs for audit pipelines."""
    payload: Dict[str, Any] = {
        "event_type": event_type,
        "request_id": request_id_var.get() or "",
        "path": request.url.path if request else "",
        "method": request.method if request else "",
        "client_ip": (request.client.host if request and request.client else ""),
        "details": details,
    }
    metrics.log_security_event(event_type)
    security_log.warning(json.dumps(payload, ensure_ascii=True, sort_keys=True))


def filter_caller_history(
    raw_history: Optional[List[Any]],
    *,
    is_bearer: bool,
    max_item_chars: int = 2000,
    max_items: int = 3,
) -> Tuple[List[Dict[str, str]], int]:
    """Normalize caller-provided conversation history into seedable turns.

    Returns (history_items, blocked_count).

    Gateway-mode history is server-loaded by the edge function and trusted, so it
    is only length-capped (preserving prior behavior). In public-bearer mode the
    history is client-controlled and untrusted (audit S6): each turn's question
    and answer are run through the firewall — turns that trip a block rule are
    dropped, and surviving text is replaced with its sanitized form.
    """
    items: List[Dict[str, str]] = []
    blocked = 0
    for turn in (raw_history or [])[:max_items]:
        if hasattr(turn, "model_dump"):
            turn = turn.model_dump()
        if not (isinstance(turn, dict) and turn.get("question")):
            continue
        question = str(turn.get("question", ""))[:max_item_chars]
        answer = str(turn.get("answer", ""))[:max_item_chars]
        if is_bearer:
            q_decision = inspect_query(question)
            a_decision = inspect_query(answer)
            if q_decision.action == "block" or a_decision.action == "block":
                blocked += 1
                continue
            question = q_decision.sanitized_query
            answer = a_decision.sanitized_query
        items.append({"question": question, "answer": answer})
    return items, blocked

# Caller-aware rate limiting: key by authenticated identity when available,
# fall back to IP for unauthenticated/rejected traffic.
def _rate_limit_key(request: Request) -> str:
    """Derive rate-limit key from auth headers (pre-authentication peek).

    Only the gateway secret is cheap enough to verify here (constant-time
    string compare).  Bearer tokens are NOT verified — using a token hash
    as the key would let attackers forge unlimited buckets with random
    strings.  All non-gateway traffic is keyed by IP address.
    """
    app_key = request.headers.get("x-app-key") or ""
    if app_key and GATEWAY_SHARED_SECRET and hmac.compare_digest(app_key, GATEWAY_SHARED_SECRET):
        session_token = (request.headers.get("x-session-token") or "").strip()
        if session_token:
            session_id = resolve_session_token(session_token, SESSION_SIGNING_SECRET)
            if session_id:
                session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
                return f"gateway_session:{session_hash}"

        forwarded_for = (request.headers.get("x-forwarded-for") or "").strip()
        if forwarded_for:
            client_ip = forwarded_for.split(",", 1)[0].strip()
            if client_ip:
                return f"gateway_ip:{client_ip}"

        return f"gateway_ip:{get_remote_address(request)}"
    return f"ip:{get_remote_address(request)}"


# Post-auth and pre-auth rate limiters. Simple sliding-window counters in
# process memory.
_preauth_rate_buckets: Dict[str, List[float]] = {}
_preauth_rate_lock = threading.Lock()
_gateway_rate_buckets: Dict[str, List[float]] = {}
_gateway_rate_lock = threading.Lock()
_user_rate_buckets: Dict[str, List[float]] = {}
_user_rate_lock = threading.Lock()


# Amortized bucket eviction: without it, subjects that stop sending keep an
# entry forever and each bucket map grows with every distinct IP/session the
# process ever sees. Swept at most once per interval, per bucket map.
_BUCKET_SWEEP_INTERVAL_SECONDS = 300.0
_bucket_last_sweep: Dict[int, float] = {}


def _check_sliding_window_rate_limit(
    *,
    subject_id: str,
    buckets: Dict[str, List[float]],
    bucket_lock: threading.Lock,
    max_requests: int,
    window_seconds: float = 60.0,
) -> bool:
    """Return True when the subject stays within the sliding-window budget."""
    now = time.time()
    with bucket_lock:
        if now - _bucket_last_sweep.get(id(buckets), 0.0) > _BUCKET_SWEEP_INTERVAL_SECONDS:
            _bucket_last_sweep[id(buckets)] = now
            stale = [
                key for key, stamps in buckets.items()
                if not stamps or now - stamps[-1] >= window_seconds
            ]
            for key in stale:
                del buckets[key]
        timestamps = [t for t in buckets.get(subject_id, []) if now - t < window_seconds]
        if len(timestamps) >= max_requests:
            buckets[subject_id] = timestamps
            return False
        timestamps.append(now)
        buckets[subject_id] = timestamps
        return True


def _preauth_client_ip(request: Request) -> str:
    """Client IP for the pre-auth limiter.

    Behind the platform proxy every request shares one socket peer, so keying
    on it collapses all callers into a single bucket — one abusive client
    exhausts the pre-auth budget for everyone. Trust exactly one proxy hop:
    take the LAST X-Forwarded-For entry (appended by the trusted edge), so
    client-supplied spoof entries earlier in the list are ignored. This
    deliberately differs from the verified-gateway key in _rate_limit_key,
    which may take the FIRST entry because the secret-checked gateway
    supplies clean forwarding. TRUST_PROXY_CLIENT_IP=false restores the
    socket peer for direct-exposure deployments.
    """
    if TRUST_PROXY_CLIENT_IP:
        forwarded_for = (request.headers.get("x-forwarded-for") or "").strip()
        if forwarded_for:
            last_hop = forwarded_for.rsplit(",", 1)[-1].strip()
            if last_hop:
                return last_hop
    return get_remote_address(request)


def _check_preauth_rate_limit(request: Request) -> bool:
    """Apply a coarse per-client guard before auth to dampen abusive bursts."""
    return _check_sliding_window_rate_limit(
        subject_id=f"ip:{_preauth_client_ip(request)}",
        buckets=_preauth_rate_buckets,
        bucket_lock=_preauth_rate_lock,
        max_requests=ASK_RATE_LIMIT_PREAUTH_PER_MINUTE,
    )


def _check_gateway_rate_limit(request: Request) -> bool:
    """Apply the gateway limiter using a verified session-aware key."""
    return _check_sliding_window_rate_limit(
        subject_id=_rate_limit_key(request),
        buckets=_gateway_rate_buckets,
        bucket_lock=_gateway_rate_lock,
        max_requests=ASK_RATE_LIMIT_GATEWAY_PER_MINUTE,
    )


def _check_user_rate_limit(subject_id: str) -> bool:
    """Return True if the user is within their per-minute rate limit."""
    return _check_sliding_window_rate_limit(
        subject_id=subject_id,
        buckets=_user_rate_buckets,
        bucket_lock=_user_rate_lock,
        max_requests=ASK_RATE_LIMIT_PUBLIC_PER_MINUTE,
    )

class RequestBodyLimitMiddleware:
    """Reject oversized /ask bodies before FastAPI reads or validates them."""

    def __init__(self, app: Any, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/ask"
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError):
                response = JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
                await response(scope, receive, send)
                return
            if declared_size < 0:
                response = JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
                await response(scope, receive, send)
                return
            if declared_size > self.max_body_bytes:
                response = JSONResponse({"detail": "Request body too large"}, status_code=413)
                await response(scope, receive, send)
                return

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.disconnect":
                return
            if message_type != "http.request":
                continue
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_body_bytes:
                response = JSONResponse({"detail": "Request body too large"}, status_code=413)
                await response(scope, receive, send)
                return
            body.extend(chunk)
            more_body = bool(message.get("more_body", False))

        replayed = False

        async def replay_receive() -> Dict[str, Any]:
            nonlocal replayed
            if replayed:
                return {"type": "http.request", "body": b"", "more_body": False}
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_receive, send)


# Request ID middleware for observability and debugging
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Preserve a safe correlation ID and publish the gateway contract version."""

    async def dispatch(self, request: Request, call_next):
        request_id = resolve_request_id(request.headers.get("x-request-id"))
        context_token = request_id_var.set(request_id)

        try:
            response = await call_next(request)
        except Exception as e:
            log.error(f"[{request_id}] {request.method} {request.url.path} error: {e}")
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Enai-Contract-Version"] = CHAT_GATEWAY_CONTRACT_VERSION
            matched = request.scope.get("route") is not None
            level = logging.INFO if matched else logging.DEBUG
            log.log(level, f"[{request_id}] {request.method} {request.url.path} → {response.status_code}")
            return response
        finally:
            request_id_var.reset(context_token)

app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=MAX_REQUEST_BODY_BYTES)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Specific origins from environment (SECURITY FIX)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Only needed methods
    allow_headers=[
        "Content-Type",
        "x-app-key",
        "Authorization",
        "x-session-token",
        "x-request-id",
        "x-enai-contract-version",
    ],
    expose_headers=[
        "X-Request-ID",
        "X-Trace-ID",
        "X-Session-Token",
        "X-Enai-Contract-Version",
        "X-LLM-Total-Tokens",
        "X-LLM-Estimated-Cost-USD",
    ],
)

# -----------------------------
# Models: Imported from models.py (line 64)
# -----------------------------
# Question, APIResponse, MetricsResponse are imported at top of file

# -----------------------------
# LLM + Planning helpers
# -----------------------------

# -----------------------------
# LLM Functions: Imported from core/llm.py (lines 72-80)
# -----------------------------
# make_gemini, make_openai, llm_cache, classify_query_type, get_query_focus,
# llm_generate_plan_and_sql, llm_summarize - all imported from core.llm

# ------------------------------------------------------------------
# REMOVED: llm_plan_analysis - Combined into llm_generate_plan_and_sql
# ------------------------------------------------------------------


# REMOVED: FEW_SHOT_SQL — canonical copy lives in core/llm.py

# -----------------------------
# LLM SQL Generation: Imported from core.llm (line 76)
# -----------------------------
# llm_generate_plan_and_sql() is imported from core.llm

# -----------------------------
# Data helpers: Imported from analysis.stats (line 83)
# -----------------------------
# rows_to_preview() and quick_stats() are imported from analysis.stats
# (startup hook moved to the _lifespan context manager above — Q5, 2026-06-10)


@app.get("/ask")
def ask_get():
    auth_hint = (
        "Authorization: Bearer <token> or X-App-Key"
        if ENABLE_PUBLIC_BEARER_AUTH
        else "X-App-Key"
    )
    return {
        "message": f"POST /ask with JSON: {{'query': '...'}} and a valid {auth_hint} header."
    }

@app.get("/healthz")
def healthz():
    """Liveness probe: process is up."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness probe: live DB connectivity plus current required schema."""
    db_ready = is_database_available()
    schema_refreshed = refresh_schema_map() if db_ready else False
    schema_ready = bool(
        db_ready
        and schema_refreshed
        and required_schema_is_ready(SCHEMA_MAP)
    )
    ready = db_ready and schema_ready
    payload = {
        "status": "ready" if ready else "degraded",
        "database_ready": db_ready,
        "schema_ready": schema_ready,
    }
    if ready:
        return payload
    return JSONResponse(status_code=503, content=payload)


@app.get("/metrics")
def get_metrics(x_app_key: Optional[str] = Header(None, alias="X-App-Key")):
    """Return application metrics for observability. Disabled by default in production."""
    if not ENABLE_METRICS_ENDPOINT:
        raise HTTPException(status_code=404, detail="Not found")
    if not x_app_key or not EVALUATE_ADMIN_SECRET or not hmac.compare_digest(x_app_key, EVALUATE_ADMIN_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")
    pool = getattr(ENGINE, "pool", None)
    pool_size = None
    checked_out = None
    if pool is not None:
        size_fn = getattr(pool, "size", None)
        checked_out_fn = getattr(pool, "checkedout", None)
        if callable(size_fn):
            pool_size = size_fn()
        if callable(checked_out_fn):
            checked_out = checked_out_fn()

    return {
        "status": "healthy",
        "metrics": metrics.get_stats(),
        "cache": llm_cache.stats(),  # Phase 1 optimization: cache metrics
        "model": {
            "type": MODEL_TYPE,
            "active_model": get_primary_model_name(),
            "gemini_model": GEMINI_MODEL if MODEL_TYPE == "gemini" else None,
            "openai_model": OPENAI_MODEL if MODEL_TYPE == "openai" else None,
            "nvidia_model": NVIDIA_MODEL if MODEL_TYPE == "nvidia" else None,
        },
        "database": {
            "pool_size": pool_size,
            "checked_out": checked_out,
        },
        "resilience": get_resilience_snapshot(),
    }


@app.get("/evaluate")
def evaluate(
    x_app_key: str = Header(..., alias="X-App-Key"),
    mode: str = Query("quick", description="Test mode: quick (10 queries) or full (75 queries)"),
    type: Optional[str] = Query(None, description="Filter by query type: single_value, list, comparison, trend, analyst"),
    query_id: Optional[str] = Query(None, description="Run specific query by ID (e.g., sv_001)"),
    format: str = Query("html", description="Output format: html or json")
):
    """
    Run evaluation tests against the query engine.

    Purpose: Validate query generation and answer quality to ensure optimizations
             don't degrade quality.

    Examples:
        GET /evaluate?mode=quick&format=html  (Quick test, browser view)
        GET /evaluate?mode=full&format=json   (Full test, JSON results)
        GET /evaluate?type=analyst            (Test only analyst queries)
        GET /evaluate?query_id=sv_001         (Test specific query)

    Authentication: Requires X-App-Key header using the evaluate admin secret
    """
    if not ENABLE_EVALUATE_ENDPOINT:
        raise HTTPException(status_code=404, detail="Not found")
    if not x_app_key or not EVALUATE_ADMIN_SECRET or not hmac.compare_digest(x_app_key, EVALUATE_ADMIN_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Import evaluation engine
        from evaluation_engine import filter_queries, generate_summary, load_evaluation_dataset, run_single_evaluation

        # Load dataset
        dataset = load_evaluation_dataset()
        queries = dataset["queries"]

        # Filter queries
        filtered_queries = filter_queries(queries, mode=mode, query_type=type, query_id=query_id)

        if not filtered_queries:
            raise HTTPException(status_code=404, detail="No queries found matching filters")

        # Define API function that calls pipeline internally
        def call_api_internal(query_text: str):
            """Call the pipeline internally without HTTP overhead."""
            try:
                start = time.time()
                # Call the pipeline directly
                ctx = process_query(query=query_text)
                elapsed_ms = (time.time() - start) * 1000

                # Convert response model to dict
                response_dict = {
                    "sql": ctx.safe_sql,
                    "answer": ctx.summary,
                    "data": ctx.chart_data
                }
                return response_dict, elapsed_ms, ""

            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000 if 'start' in locals() else 0
                return {}, elapsed_ms, str(e)

        # Run evaluations
        results = []
        for query_data in filtered_queries:
            result = run_single_evaluation(query_data, call_api_internal)
            results.append(result)
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)

        # Generate summary
        summary = generate_summary(results, dataset)

        # Return results
        if format == "json":
            return {
                "summary": summary,
                "results": results
            }
        else:
            # Return HTML for browser viewing
            html = generate_html_report(summary, results, filtered_queries)
            return HTMLResponse(content=html)

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Evaluation dataset not found. Ensure evaluation_dataset.json is deployed."
        )
    except Exception as e:
        log.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


def generate_html_report(summary: Dict[str, Any], results: List[Dict[str, Any]], queries: List[Dict[str, Any]]) -> str:
    """Generate HTML report for browser viewing."""

    pass_rate = summary["pass_rate"] * 100
    pass_color = "green" if pass_rate >= 90 else "orange" if pass_rate >= 70 else "red"

    # Build results table
    results_html = ""
    for result in results:
        status_icon = "✓" if result["status"] == "pass" else "✗" if result["status"] == "fail" else "⚠"
        status_color = "green" if result["status"] == "pass" else "red" if result["status"] == "fail" else "orange"

        details = []
        if not result.get("sql_valid", True):
            details.append(f"SQL issues: {', '.join(result.get('sql_missing', []))}")
        if not result.get("quality_valid", True):
            details.append(f"Quality issues: {', '.join(result.get('quality_failed', []))}")
        if not result.get("performance_valid", True):
            details.append(result.get("performance_msg", "Performance issue"))

        details_html = "<br>".join(details) if details else "All checks passed"

        results_html += f"""
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="padding: 8px;"><span style="color: {status_color}; font-weight: bold;">{status_icon}</span></td>
            <td style="padding: 8px;"><code>{result['id']}</code></td>
            <td style="padding: 8px;">{result['type']}</td>
            <td style="padding: 8px;">{result['query'][:80]}...</td>
            <td style="padding: 8px;">{result['elapsed_ms']:.0f}ms</td>
            <td style="padding: 8px; font-size: 0.9em;">{details_html}</td>
        </tr>
        """

    # Build type breakdown
    type_rows = ""
    for qtype, stats in sorted(summary["by_type"].items()):
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        type_rows += f"""
        <tr>
            <td style="padding: 8px;">{qtype}</td>
            <td style="padding: 8px;">{stats['passed']}/{stats['total']}</td>
            <td style="padding: 8px;">{rate:.1f}%</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .summary {{ background: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
            .metric-label {{ font-weight: bold; color: #666; }}
            .metric-value {{ font-size: 1.3em; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 8px; }}
            tr:hover {{ background: #f5f5f5; }}
            .pass-rate {{ font-size: 2em; font-weight: bold; color: {pass_color}; }}
            code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 Evaluation Report</h1>

            <div class="summary">
                <div class="metric">
                    <span class="metric-label">Pass Rate:</span>
                    <span class="pass-rate">{pass_rate:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Queries:</span>
                    <span class="metric-value">{summary['total_queries']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Passed:</span>
                    <span class="metric-value" style="color: green;">{summary['passed']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Failed:</span>
                    <span class="metric-value" style="color: red;">{summary['failed']}</span>
                </div>
                {f'<div class="metric"><span class="metric-label">Errors:</span><span class="metric-value" style="color: orange;">{summary["errors"]}</span></div>' if summary['errors'] > 0 else ''}
            </div>

            <h2>📊 Performance Metrics</h2>
            <div class="summary">
                <div class="metric">
                    <span class="metric-label">Avg Response Time:</span>
                    <span class="metric-value">{summary['performance']['avg_time_ms']:.0f}ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Simple Queries:</span>
                    <span class="metric-value">{summary['performance']['avg_simple_ms']:.0f}ms</span>
                    <span style="color: #666;">(target: &lt;8s)</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Complex Queries:</span>
                    <span class="metric-value">{summary['performance']['avg_complex_ms']:.0f}ms</span>
                    <span style="color: #666;">(target: &lt;45s)</span>
                </div>
            </div>

            <h2>📋 Results by Type</h2>
            <table>
                <tr>
                    <th>Query Type</th>
                    <th>Passed/Total</th>
                    <th>Pass Rate</th>
                </tr>
                {type_rows}
            </table>

            <h2>🔍 Issue Breakdown</h2>
            <div class="summary">
                <div class="metric">
                    <span class="metric-label">SQL Pattern Issues:</span>
                    <span class="metric-value">{summary['issues']['sql_pattern_issues']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Quality Issues:</span>
                    <span class="metric-value">{summary['issues']['quality_issues']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Performance Issues:</span>
                    <span class="metric-value">{summary['issues']['performance_issues']}</span>
                </div>
            </div>

            <h2>📝 Detailed Results</h2>
            <table>
                <tr>
                    <th style="width: 40px;">Status</th>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Query</th>
                    <th>Time</th>
                    <th>Details</th>
                </tr>
                {results_html}
            </table>

            <div style="margin-top: 30px; padding: 15px; background: #e3f2fd; border-left: 4px solid #2196F3; border-radius: 4px;">
                <strong>💡 Tip:</strong> Add <code>?format=json</code> to get results as JSON for programmatic access.
                <br>
                <strong>🔧 Filters:</strong> Use <code>?mode=full</code>, <code>?type=analyst</code>, or <code>?query_id=sv_001</code> to customize tests.
            </div>

            <div style="margin-top: 20px; text-align: center; color: #999; font-size: 0.9em;">
                Generated: {summary['timestamp']}<br>
                Dataset Version: {summary['dataset_version']}
            </div>
        </div>
    </body>
    </html>
    """

    return html



@app.post('/ask', response_model=APIResponse)
def ask_post(
    request: Request,
    response: Response,
    q: Question,
    x_app_key: Optional[str] = Header(None, alias='X-App-Key'),
    x_session_token: Optional[str] = Header(None, alias='X-Session-Token'),
    x_enai_contract_version: Optional[str] = Header(None, alias='X-Enai-Contract-Version'),
    authorization: Optional[str] = Header(None, alias='Authorization'),
):
    t0 = time.time()
    trace_id = request_id_var.get() or str(uuid.uuid4())
    response.headers["X-Trace-ID"] = trace_id
    metrics.start_request_telemetry(trace_id)
    request_llm_telemetry = None

    def _finalize_request_telemetry():
        nonlocal request_llm_telemetry
        if request_llm_telemetry is None:
            request_llm_telemetry = metrics.finalize_request_telemetry()
        total = request_llm_telemetry.get("total_tokens", 0)
        if total > 0:
            _caller = getattr(request.state, "caller", None)
            log.info(
                "LLM usage: calls=%d prompt_tokens=%d completion_tokens=%d total_tokens=%d cost_usd=%.6f models=%s trace_id=%s auth_mode=%s subject_id=%s",
                request_llm_telemetry.get("llm_calls", 0),
                request_llm_telemetry.get("prompt_tokens", 0),
                request_llm_telemetry.get("completion_tokens", 0),
                total,
                request_llm_telemetry.get("estimated_cost_usd", 0.0),
                ",".join(request_llm_telemetry.get("models", {}).keys()),
                request_llm_telemetry.get("trace_id", ""),
                _caller.auth_mode if _caller else "unknown",
                _caller.subject_id if _caller else "unknown",
            )
        # Only expose LLM usage headers to gateway callers (internal).
        _caller = getattr(request.state, "caller", None)
        if _caller is None or _caller.auth_mode == "gateway":
            response.headers["X-LLM-Total-Tokens"] = str(total)
            response.headers["X-LLM-Estimated-Cost-USD"] = f"{float(request_llm_telemetry.get('estimated_cost_usd', 0.0)):.8f}"
        return request_llm_telemetry

    if not _check_preauth_rate_limit(request):
        _finalize_request_telemetry()
        log_security_event("preauth_rate_limit_exceeded", request=request)
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        caller: CallerContext = authenticate_request(
            x_app_key=x_app_key,
            authorization=authorization,
        )
    except HTTPException:
        _finalize_request_telemetry()
        log_security_event(
            "unauthorized_request",
            request=request,
            provided_app_key=bool(x_app_key),
            provided_bearer=bool(authorization),
        )
        raise
    request.state.caller = caller

    # Keep the declaration optional while the backend and edge deploy
    # independently. Once a caller declares a version, fail closed before any
    # model or database work if it is not supported.
    if x_enai_contract_version is not None and not hmac.compare_digest(
        x_enai_contract_version,
        CHAT_GATEWAY_CONTRACT_VERSION,
    ):
        _finalize_request_telemetry()
        log_security_event(
            "unsupported_gateway_contract",
            request=request,
            provided_version=x_enai_contract_version,
            supported_version=CHAT_GATEWAY_CONTRACT_VERSION,
        )
        raise HTTPException(status_code=409, detail="Unsupported chat gateway contract version")

    if caller.auth_mode == "gateway":
        if not _check_gateway_rate_limit(request):
            _finalize_request_telemetry()
            log_security_event(
                "gateway_rate_limit_exceeded",
                request=request,
                rate_limit_key=_rate_limit_key(request),
            )
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    else:
        if not _check_user_rate_limit(caller.subject_id):
            _finalize_request_telemetry()
            log_security_event(
                "public_bearer_rate_limit_exceeded",
                request=request,
                subject_id=caller.subject_id,
            )
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Referer validation
    referer = request.headers.get('referer') or request.headers.get('Referer')
    if referer:
        try:
            parsed = urlparse(referer)
            referer_origin = f'{parsed.scheme}://{parsed.netloc}'
        except Exception:
            _finalize_request_telemetry()
            log_security_event(
                "invalid_referer_header",
                request=request,
                referer=referer,
            )
            raise HTTPException(status_code=403, detail='Forbidden')
        if not any(referer_origin == origin.rstrip('/') for origin in ALLOWED_ORIGINS):
            _finalize_request_telemetry()
            log.warning(f'Blocked request with disallowed Referer: {referer_origin}')
            log_security_event(
                "disallowed_referer",
                request=request,
                referer=referer_origin,
            )
            raise HTTPException(status_code=403, detail='Forbidden')

    slot_acquired = request_backpressure_gate.try_acquire()
    if not slot_acquired:
        metrics.log_load_shed()
        _finalize_request_telemetry()
        log_security_event(
            "load_shed_backpressure",
            request=request,
            max_concurrent=get_resilience_snapshot()["request_backpressure"]["max_concurrent"],
        )
        raise HTTPException(status_code=503, detail="Service busy. Please retry shortly.")

    try:
        session_id, session_token, reused_existing_session = get_or_issue_session(
            x_session_token,
            SESSION_SIGNING_SECRET,
        )
        response.headers["X-Session-Token"] = session_token
        if x_session_token and not reused_existing_session:
            log_security_event(
                "invalid_session_token",
                request=request,
                provided=True,
            )
        bound_history = get_history(session_id)

        # If the in-process session has no history yet (e.g. first request after
        # deploy or session expiry) but the edge function provided server-loaded
        # history from the chat_history table, use it as a seed.  This bridges
        # the gap where the edge function cannot forward session tokens but *can*
        # load persisted turns from the database on behalf of the authenticated user.
        _MAX_HISTORY_ITEM_CHARS = 2000  # ~500 tokens per item, 6 items max
        if not bound_history and q.conversation_history:
            bound_history, blocked_history_turns = filter_caller_history(
                q.conversation_history,
                is_bearer=(caller.auth_mode == "public_bearer"),
                max_item_chars=_MAX_HISTORY_ITEM_CHARS,
            )
            if blocked_history_turns:
                log_security_event(
                    "bearer_history_turn_blocked",
                    request=request,
                    subject_id=caller.subject_id,
                    blocked_turns=blocked_history_turns,
                )
            if bound_history:
                seed_history(session_id, bound_history)
                log.info(
                    "Seeded session history from edge-function-provided turns. "
                    "session_id=%s turns=%d",
                    session_id, len(bound_history),
                )
        elif q.conversation_history and bound_history:
            # Session already has history — ignore the client-provided duplicate
            log.debug(
                "Ignoring edge-function history; session already has %d turns.",
                len(bound_history),
            )

        metrics.log_session_history_context(len(bound_history))

        firewall_decision = inspect_query(q.query)
        metrics.log_firewall_decision(firewall_decision.action)
        if firewall_decision.action == "block":
            log_security_event(
                "firewall_blocked_query",
                request=request,
                rules=firewall_decision.matched_rules,
                risk_score=firewall_decision.risk_score,
            )
            exec_time = time.time() - t0
            metrics.log_request(exec_time)
            req_usage = _finalize_request_telemetry()
            return APIResponse(
                answer=build_safe_refusal_message(firewall_decision),
                charts=None,
                chart_data=None,
                chart_type=None,
                chart_metadata={
                    "guardrail_action": firewall_decision.action,
                    "guardrail_reason": firewall_decision.reason,
                    "guardrail_risk_score": firewall_decision.risk_score,
                    "trace_id": trace_id,
                    "llm_telemetry": req_usage,
                },
                execution_time=exec_time,
            )

        query_text = firewall_decision.sanitized_query or q.query
        if firewall_decision.action == "warn":
            log_security_event(
                "firewall_warned_query",
                request=request,
                rules=firewall_decision.matched_rules,
                risk_score=firewall_decision.risk_score,
            )
        if query_text != q.query:
            log_security_event(
                "firewall_sanitized_query",
                request=request,
                original_len=len(q.query),
                sanitized_len=len(query_text),
            )

        try:
            ctx = process_query(
                query=query_text,
                conversation_history=bound_history,
                trace_id=trace_id,
                session_id=session_id,
                previous_contract_snapshot=get_last_contract(session_id),
            )
        except HTTPException:
            _finalize_request_telemetry()
            raise
        except Exception as e:
            _finalize_request_telemetry()
            log.exception('Pipeline failed')
            log_security_event(
                "pipeline_execution_failure",
                request=request,
                error=str(e),
            )
            # Generic detail for every auth mode: the exception is already in
            # the server log (log.exception above) and the security event.
            raise HTTPException(status_code=500, detail="Query processing failed")

        append_exchange(session_id, query_text, ctx.summary)
        _contract_snapshot = continuity_snapshot_json(ctx)
        if _contract_snapshot:
            set_last_contract(session_id, _contract_snapshot)
        req_usage = _finalize_request_telemetry()

        exec_time = time.time() - t0
        metrics.log_request(exec_time)
        log.info(f'Finished request in {exec_time:.2f}s')

        response_chart_meta = dict(ctx.chart_meta or {})
        response_chart_meta.update(
            {
                "trace_id": trace_id,
                "stage_timings_ms": dict(ctx.stage_timings_ms),
                "llm_telemetry": req_usage,
                "session_bound_history_turns": len(bound_history),
                "summary_claims": list(ctx.summary_claims or []),
                "summary_citations": list(ctx.summary_citations or []),
                "summary_confidence": float(ctx.summary_confidence),
                "summary_provenance_coverage": float(ctx.summary_provenance_coverage),
                "summary_claim_provenance": list(ctx.summary_claim_provenance or []),
                "summary_provenance_gate_passed": bool(ctx.summary_provenance_gate_passed),
                "summary_provenance_gate_reason": str(ctx.summary_provenance_gate_reason or ""),
                "provenance_query_hash": str(ctx.provenance_query_hash or ""),
                "provenance_source": str(ctx.provenance_source or ""),
                "provenance_refs": list(getattr(ctx, "provenance_refs", []) or []),
                "answer_provenance": build_answer_provenance(ctx),
            }
        )

        return APIResponse(
            answer=ctx.summary,
            charts=(getattr(ctx, "charts", None) or None),
            chart_data=ctx.chart_data,
            chart_type=ctx.chart_type,
            chart_metadata=response_chart_meta,
            execution_time=exec_time,
        )
    finally:
        request_backpressure_gate.release()


# ... [server startup block identical] ...



# -----------------------------
# Server Startup (CRITICAL FIX)
# -----------------------------
# This block runs the application when the script is executed directly (e.g., by a Docker ENTRYPOINT)
if __name__ == "__main__":
    try:
        import uvicorn
        port = int(os.getenv("PORT", 8000))

        # CRITICAL: host '0.0.0.0' is required for container accessibility
        log.info(f"🚀 Starting Uvicorn server on 0.0.0.0:{port}")
        uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
    except ImportError:
        log.error("Uvicorn is not installed. Please install it with 'pip install uvicorn'.")
        raise
    except Exception:
        log.exception("FATAL: Uvicorn server failed to start")
        raise
