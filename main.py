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
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
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
from agent.metric_units import METRIC_UNITS
from agent.pipeline import process_query
from agent.public_metadata import build_public_response_metadata, project_public_charts
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
    ASK_DEFAULT_REQUEST_BUDGET_MS,
    ASK_MAX_REQUEST_BUDGET_MS,
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
    HTTP_SERVER_PORT,
    HTTP_SERVER_WORKERS,
    MAX_REQUEST_BODY_BYTES,
    MODEL_TYPE,
    NVIDIA_MODEL,
    OPENAI_MODEL,
    SCHEMA_READINESS_CACHE_TTL_SECONDS,
    SCHEMA_READINESS_RETRY_INTERVAL_SECONDS,
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
from contracts.chat_gateway_v2 import (
    build_chat_gateway_v2_response,
    serialize_chat_gateway_v2_response,
)
from core.db_gateway import database_connection
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
from core.query_executor import (
    ENGINE,
    execute_sql_safely,
    get_database_runtime_identity,
    is_database_available,
)
from core.sql_generator import plan_validate_repair, sanitize_sql, simple_table_whitelist_check
from guardrails.firewall import build_safe_refusal_message, inspect_query
from models import (
    CHAT_GATEWAY_CONTRACT_VERSION,
    CHAT_GATEWAY_V2_CONTRACT_VERSION,
    SUPPORTED_CHAT_GATEWAY_CONTRACT_VERSIONS,
    APIErrorResponse,
    APIResponse,
    MetricsResponse,
    Question,
    TerminalOutcome,
)
from utils.auth import CallerContext, authenticate_request
from utils.language import detect_language, get_language_instruction

# Phase 2: Core modules
from utils.metrics import metrics
from utils.privacy_logging import (
    PrivacyLogFilter,
    hash_private_identifier,
    sanitize_security_details,
)
from utils.query_validation import is_conceptual_question, should_skip_sql_execution, validate_sql_relevance
from utils.rate_limits import InMemoryRateLimitRepository
from utils.request_deadline import (
    MINIMUM_START_BUDGET_MS,
    InvalidRequestBudget,
    RequestDeadlineExceeded,
    build_request_deadline,
)
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
log.addFilter(PrivacyLogFilter())


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

# Request and internal span tracking for observability
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
span_id_var: ContextVar[str] = ContextVar("span_id", default="")

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
_SCHEMA_MAP_REFRESHED_AT: Optional[float] = None
_SCHEMA_MAP_REFRESH_FAILED_AT: Optional[float] = None
_SCHEMA_REFRESH_LOCK = threading.Lock()
REQUIRED_SCHEMA_COLUMNS = {
    table_name: set(DB_SCHEMA_DICT["views"][table_name]["columns"])
    for table_name in STATIC_ALLOWED_TABLES
}


def schema_readiness_gaps(schema_map: Dict[str, set]) -> Dict[str, List[str]]:
    """Return allow-listed relations and required columns absent from reflection."""
    gaps: Dict[str, List[str]] = {}
    for table_name, required_columns in REQUIRED_SCHEMA_COLUMNS.items():
        actual_columns = set(schema_map.get(table_name) or set())
        missing_columns = sorted(required_columns - actual_columns)
        if missing_columns:
            gaps[table_name] = missing_columns
    return gaps


def required_schema_is_ready(schema_map: Dict[str, set]) -> bool:
    """Return whether every allow-listed relation exposes its required columns."""
    return not schema_readiness_gaps(schema_map)


def _schema_map_cache_is_fresh(now_monotonic: Optional[float] = None) -> bool:
    """Return whether the last successful reflection remains inside its TTL."""
    if _SCHEMA_MAP_REFRESHED_AT is None:
        return False
    now = time.monotonic() if now_monotonic is None else now_monotonic
    age_seconds = now - _SCHEMA_MAP_REFRESHED_AT
    return 0 <= age_seconds <= SCHEMA_READINESS_CACHE_TTL_SECONDS


def _schema_refresh_retry_is_due(now_monotonic: Optional[float] = None) -> bool:
    """Return whether a failed reflection may be retried without probe thrashing."""
    if _SCHEMA_MAP_REFRESH_FAILED_AT is None:
        return True
    now = time.monotonic() if now_monotonic is None else now_monotonic
    age_seconds = now - _SCHEMA_MAP_REFRESH_FAILED_AT
    return age_seconds < 0 or age_seconds >= SCHEMA_READINESS_RETRY_INTERVAL_SECONDS


def _ensure_schema_map_current() -> bool:
    """Refresh stale schema state once across concurrent readiness probes."""
    if _schema_map_cache_is_fresh():
        return True
    if not _schema_refresh_retry_is_due():
        return False
    with _SCHEMA_REFRESH_LOCK:
        if _schema_map_cache_is_fresh():
            return True
        if not _schema_refresh_retry_is_due():
            return False
        return refresh_schema_map()


def refresh_schema_map() -> bool:
    """Best-effort schema reflection; never raises to caller."""
    global SCHEMA_MAP, _SCHEMA_MAP_REFRESHED_AT, _SCHEMA_MAP_REFRESH_FAILED_AT
    try:
        with database_connection(ENGINE, operation="schema_reflection") as conn:
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
        _SCHEMA_MAP_REFRESHED_AT = time.monotonic()
        _SCHEMA_MAP_REFRESH_FAILED_AT = None
        # Reflection describes availability; it must never expand the explicit
        # SQL security allow-list when an unrelated materialized view appears.
        # The allow-list is initialized from STATIC_ALLOWED_TABLES and is not
        # mutated during refresh, avoiding a transient empty set under
        # concurrent readiness probes.
        log.info("Schema reflection complete: %s views", len(SCHEMA_MAP))
        gaps = schema_readiness_gaps(SCHEMA_MAP)
        if gaps:
            gap_summary = "; ".join(
                f"{table_name}({','.join(columns)})"
                for table_name, columns in sorted(gaps.items())
            )
            log.warning("Schema readiness missing required columns: %s", gap_summary)
        return True
    except Exception as exc:
        # Retain the last known map for diagnostics, but leave its timestamp
        # stale so readiness cannot pass from an outdated snapshot.
        _SCHEMA_MAP_REFRESH_FAILED_AT = time.monotonic()
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
# CHAT_GATEWAY_CONTRACT_VERSION now lives with the contract models in models.py
# (P6.A) so the published schema artifact and its drift test can read it without
# importing the FastAPI app. Re-exported here for the existing call sites.
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


def create_internal_span_id() -> str:
    """Allocate a backend-local span distinct from the end-to-end request ID."""
    return f"span-{uuid.uuid4().hex}"


def resolve_parent_span_id(candidate: Optional[str]) -> str:
    """Accept a safe upstream span only as non-authoritative trace metadata."""
    return candidate if candidate and _SAFE_REQUEST_ID_RE.fullmatch(candidate) else ""


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


class AskAPIError(HTTPException):
    """An intentional /ask failure with a stable public error contract."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.retryable = retryable


_ASK_ERROR_DEFAULTS: Dict[int, Tuple[str, str, bool]] = {
    400: ("INVALID_REQUEST", "Invalid request", False),
    401: ("AUTHENTICATION_REQUIRED", "Authentication required", False),
    403: ("FORBIDDEN", "Request forbidden", False),
    404: ("NOT_FOUND", "Resource not found", False),
    408: ("REQUEST_TIMEOUT", "Request timed out", True),
    409: ("REQUEST_CONFLICT", "Request conflict", False),
    413: ("REQUEST_TOO_LARGE", "Request body too large", False),
    422: ("VALIDATION_FAILED", "Request validation failed", False),
    429: ("RATE_LIMITED", "Rate limit exceeded", True),
    500: ("INTERNAL_ERROR", "Internal server error", False),
    502: ("UPSTREAM_FAILURE", "Upstream service failed", True),
    503: ("SERVICE_UNAVAILABLE", "Service unavailable", True),
    504: ("UPSTREAM_TIMEOUT", "Upstream service timed out", True),
}
ASK_ERROR_RESPONSES = {
    status_code: {
        "model": APIErrorResponse,
        "description": message,
    }
    for status_code, (_code, message, _retryable) in _ASK_ERROR_DEFAULTS.items()
    if status_code != 404
}


def _ask_error_payload(
    *,
    status_code: int,
    request_id: str,
    code: Optional[str] = None,
    message: Optional[str] = None,
    retryable: Optional[bool] = None,
) -> Dict[str, Any]:
    default_code, default_message, default_retryable = _ASK_ERROR_DEFAULTS.get(
        status_code,
        ("REQUEST_FAILED", "Request failed", False),
    )
    envelope = APIErrorResponse.model_validate(
        {
            "error": {
                "code": code or default_code,
                "message": message or default_message,
                "retryable": default_retryable if retryable is None else retryable,
                "request_id": request_id,
            }
        }
    )
    return envelope.model_dump(mode="json")


@app.exception_handler(HTTPException)
async def _safe_ask_http_error(request: Request, exc: HTTPException):
    if request.url.path != "/ask":
        return await http_exception_handler(request, exc)
    code = exc.code if isinstance(exc, AskAPIError) else None
    retryable = exc.retryable if isinstance(exc, AskAPIError) else None
    message = str(exc.detail) if isinstance(exc, AskAPIError) else None
    return JSONResponse(
        status_code=exc.status_code,
        content=_ask_error_payload(
            status_code=exc.status_code,
            request_id=request_id_var.get() or resolve_request_id(None),
            code=code,
            message=message,
            retryable=retryable,
        ),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def _safe_ask_validation_error(request: Request, exc: RequestValidationError):
    if request.url.path != "/ask":
        return await request_validation_exception_handler(request, exc)
    return JSONResponse(
        status_code=422,
        content=_ask_error_payload(
            status_code=422,
            request_id=request_id_var.get() or resolve_request_id(None),
        ),
    )


def log_security_event(event_type: str, request: Optional[Request] = None, **details: Any) -> None:
    """Emit structured security logs for audit pipelines."""
    payload: Dict[str, Any] = {
        "event_type": event_type,
        "request_id": request_id_var.get() or "",
        "span_id": span_id_var.get() or "",
        "path": request.url.path if request else "",
        "method": request.method if request else "",
        "client_ip": hash_private_identifier(
            request.client.host if request and request.client else "",
            namespace="client_ip",
        ),
        "details": sanitize_security_details(details),
    }
    metrics.log_security_event(event_type)
    security_log.warning(json.dumps(payload, ensure_ascii=True, sort_keys=True))


def filter_caller_history(
    raw_history: Optional[List[Any]],
    *,
    is_bearer: Optional[bool] = None,
    max_item_chars: int = 2000,
    max_items: int = 3,
) -> Tuple[List[Dict[str, str]], int]:
    """Normalize caller-provided conversation history into seedable turns.

    Returns (history_items, blocked_count).

    Every transport is untrusted, including edge-loaded database history and
    in-process session history. ``is_bearer`` remains as an ignored compatibility
    argument for callers compiled against the P1 helper signature.
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
def _rate_limit_key(
    request: Request,
    caller: Optional[CallerContext] = None,
) -> str:
    """Derive rate-limit key from auth headers (pre-authentication peek).

    Only the gateway secret is cheap enough to verify here (constant-time
    string compare).  Bearer tokens are NOT verified — using a token hash
    as the key would let attackers forge unlimited buckets with random
    strings.  All non-gateway traffic is keyed by IP address.
    """
    if caller and caller.actor_assertion_verified and caller.actor_id and caller.session_id:
        actor_session_hash = hashlib.sha256(
            f"{caller.actor_id}\n{caller.session_id}".encode("utf-8")
        ).hexdigest()[:16]
        return f"gateway_actor_session:{actor_session_hash}"

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


# One process-local repository owns post-auth and pre-auth sliding-window
# state. Request authentication/key derivation stays in this API boundary;
# storage, locking, and bounded eviction are hidden behind the P8.A interface.
_rate_limit_repository = InMemoryRateLimitRepository()


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
    return _rate_limit_repository.consume(
        "preauth",
        f"ip:{_preauth_client_ip(request)}",
        max_requests=ASK_RATE_LIMIT_PREAUTH_PER_MINUTE,
    )


def _check_gateway_rate_limit(
    request: Request,
    caller: Optional[CallerContext] = None,
) -> bool:
    """Apply the gateway limiter using a verified session-aware key."""
    return _rate_limit_repository.consume(
        "gateway",
        _rate_limit_key(request, caller),
        max_requests=ASK_RATE_LIMIT_GATEWAY_PER_MINUTE,
    )


def _check_user_rate_limit(subject_id: str) -> bool:
    """Return True if the user is within their per-minute rate limit."""
    return _rate_limit_repository.consume(
        "user",
        subject_id,
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
                response = JSONResponse(
                    _ask_error_payload(
                        status_code=400,
                        request_id=request_id_var.get() or resolve_request_id(None),
                        code="INVALID_CONTENT_LENGTH",
                        message="Invalid Content-Length",
                    ),
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if declared_size < 0:
                response = JSONResponse(
                    _ask_error_payload(
                        status_code=400,
                        request_id=request_id_var.get() or resolve_request_id(None),
                        code="INVALID_CONTENT_LENGTH",
                        message="Invalid Content-Length",
                    ),
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if declared_size > self.max_body_bytes:
                response = JSONResponse(
                    _ask_error_payload(
                        status_code=413,
                        request_id=request_id_var.get() or resolve_request_id(None),
                    ),
                    status_code=413,
                )
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
                response = JSONResponse(
                    _ask_error_payload(
                        status_code=413,
                        request_id=request_id_var.get() or resolve_request_id(None),
                    ),
                    status_code=413,
                )
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
    """Preserve request identity and allocate a separate backend span."""

    async def dispatch(self, request: Request, call_next):
        request_id = resolve_request_id(request.headers.get("x-request-id"))
        span_id = create_internal_span_id()
        parent_span_id = resolve_parent_span_id(request.headers.get("x-enai-span-id"))
        request.state.parent_span_id = parent_span_id
        request_token = request_id_var.set(request_id)
        span_token = span_id_var.set(span_id)

        try:
            response = await call_next(request)
        except Exception as exc:
            log.error(
                "[%s/%s] %s %s error_class=%s",
                request_id,
                span_id,
                request.method,
                request.url.path,
                type(exc).__name__,
            )
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Trace-ID"] = span_id
            response.headers["X-Enai-Span-ID"] = span_id
            declared_contract = request.headers.get("x-enai-contract-version")
            response.headers["X-Enai-Contract-Version"] = (
                declared_contract
                if declared_contract in SUPPORTED_CHAT_GATEWAY_CONTRACT_VERSIONS
                else CHAT_GATEWAY_CONTRACT_VERSION
            )
            matched = request.scope.get("route") is not None
            level = logging.INFO if matched else logging.DEBUG
            log.log(
                level,
                "[%s/%s] %s %s → %s",
                request_id,
                span_id,
                request.method,
                request.url.path,
                response.status_code,
            )
            return response
        finally:
            span_id_var.reset(span_token)
            request_id_var.reset(request_token)

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
        "x-enai-request-budget-ms",
        "x-enai-actor-id",
        "x-enai-session-id",
        "x-enai-actor-issued-at",
        "x-enai-actor-signature",
        "x-enai-span-id",
    ],
    expose_headers=[
        "X-Request-ID",
        "X-Trace-ID",
        "X-Enai-Span-ID",
        "X-Session-Token",
        "X-Enai-Contract-Version",
        "X-Enai-Request-Budget-Ms",
        "X-Enai-Deadline-Remaining-Ms",
        "X-Enai-Retry-Owner",
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
    schema_refreshed = _ensure_schema_map_current() if db_ready else False
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

    runtime_identity = get_database_runtime_identity()
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
            "runtime_identity": runtime_identity.protected_metadata(),
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



@app.post('/ask', response_model=APIResponse, responses=ASK_ERROR_RESPONSES)
def ask_post(
    request: Request,
    response: Response,
    q: Question,
    x_app_key: Optional[str] = Header(None, alias='X-App-Key'),
    x_session_token: Optional[str] = Header(None, alias='X-Session-Token'),
    x_enai_contract_version: Optional[str] = Header(None, alias='X-Enai-Contract-Version'),
    x_enai_request_budget_ms: Optional[str] = Header(None, alias='X-Enai-Request-Budget-Ms'),
    x_enai_actor_id: Optional[str] = Header(None, alias='X-Enai-Actor-Id'),
    x_enai_session_id: Optional[str] = Header(None, alias='X-Enai-Session-Id'),
    x_enai_actor_issued_at: Optional[str] = Header(None, alias='X-Enai-Actor-Issued-At'),
    x_enai_actor_signature: Optional[str] = Header(None, alias='X-Enai-Actor-Signature'),
    authorization: Optional[str] = Header(None, alias='Authorization'),
):
    t0 = time.time()
    request_started_monotonic = time.monotonic()
    request_id = request_id_var.get() or resolve_request_id(None)
    trace_id = span_id_var.get() or create_internal_span_id()
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Enai-Span-ID"] = trace_id
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
                hash_private_identifier(
                    _caller.subject_id if _caller else "unknown",
                    namespace="subject",
                ),
            )
        # Only expose LLM usage headers to gateway callers (internal).
        _caller = getattr(request.state, "caller", None)
        if _caller is None or _caller.auth_mode == "gateway":
            response.headers["X-LLM-Total-Tokens"] = str(total)
            response.headers["X-LLM-Estimated-Cost-USD"] = f"{float(request_llm_telemetry.get('estimated_cost_usd', 0.0)):.8f}"
        return request_llm_telemetry

    def _v2_response(payload) -> JSONResponse:
        result = JSONResponse(
            content=serialize_chat_gateway_v2_response(payload)
        )
        for header_name, header_value in response.headers.items():
            if header_name.lower().startswith("x-"):
                result.headers[header_name] = header_value
        return result

    if not _check_preauth_rate_limit(request):
        _finalize_request_telemetry()
        log_security_event("preauth_rate_limit_exceeded", request=request)
        raise AskAPIError(429, "RATE_LIMITED", "Rate limit exceeded", retryable=True)

    selected_contract_version = (
        x_enai_contract_version or CHAT_GATEWAY_CONTRACT_VERSION
    )
    if selected_contract_version not in SUPPORTED_CHAT_GATEWAY_CONTRACT_VERSIONS:
        _finalize_request_telemetry()
        log_security_event(
            "unsupported_gateway_contract",
            request=request,
            provided_version=selected_contract_version,
            supported_versions=sorted(SUPPORTED_CHAT_GATEWAY_CONTRACT_VERSIONS),
        )
        raise AskAPIError(
            409,
            "UNSUPPORTED_CONTRACT_VERSION",
            "Unsupported chat gateway contract version",
        )
    request.state.chat_gateway_contract_version = selected_contract_version

    try:
        caller: CallerContext = authenticate_request(
            x_app_key=x_app_key,
            authorization=authorization,
            request_id=request_id,
            contract_version=selected_contract_version,
            x_actor_id=x_enai_actor_id,
            x_actor_session_id=x_enai_session_id,
            x_actor_issued_at=x_enai_actor_issued_at,
            x_actor_signature=x_enai_actor_signature,
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
    if q.user_id is not None:
        log_security_event(
            "non_authoritative_user_id_ignored",
            request=request,
            supplied=True,
        )
    if caller.auth_mode == "gateway":
        log.info(
            "Gateway actor assertion: verified=%s request_id=%s span_id=%s parent_span_id=%s",
            caller.actor_assertion_verified,
            request_id,
            trace_id,
            getattr(request.state, "parent_span_id", ""),
        )

    try:
        request_deadline = build_request_deadline(
            x_enai_request_budget_ms,
            default_budget_ms=ASK_DEFAULT_REQUEST_BUDGET_MS,
            maximum_budget_ms=ASK_MAX_REQUEST_BUDGET_MS,
            now_monotonic=request_started_monotonic,
        )
    except InvalidRequestBudget:
        _finalize_request_telemetry()
        log_security_event(
            "invalid_request_budget",
            request=request,
            supplied=bool(x_enai_request_budget_ms),
        )
        raise AskAPIError(400, "INVALID_REQUEST_BUDGET", "Invalid request budget")

    try:
        request_deadline.ensure_remaining(
            "pipeline_start",
            minimum_ms=MINIMUM_START_BUDGET_MS,
        )
    except RequestDeadlineExceeded:
        _finalize_request_telemetry()
        log_security_event("request_deadline_exhausted", request=request, stage="pipeline_start")
        raise AskAPIError(
            408,
            "REQUEST_DEADLINE_EXCEEDED",
            "Request deadline exceeded",
            retryable=True,
        )

    response.headers["X-Enai-Request-Budget-Ms"] = str(request_deadline.budget_ms)
    response.headers["X-Enai-Deadline-Remaining-Ms"] = str(request_deadline.remaining_ms())
    response.headers["X-Enai-Retry-Owner"] = request_deadline.retry_owner

    if caller.auth_mode == "gateway":
        if not _check_gateway_rate_limit(request, caller):
            _finalize_request_telemetry()
            log_security_event(
                "gateway_rate_limit_exceeded",
                request=request,
                rate_limit_key=_rate_limit_key(request, caller),
            )
            raise AskAPIError(429, "RATE_LIMITED", "Rate limit exceeded", retryable=True)
    else:
        if not _check_user_rate_limit(caller.subject_id):
            _finalize_request_telemetry()
            log_security_event(
                "public_bearer_rate_limit_exceeded",
                request=request,
                subject_id=caller.subject_id,
            )
            raise AskAPIError(429, "RATE_LIMITED", "Rate limit exceeded", retryable=True)

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
            raise AskAPIError(403, "FORBIDDEN_ORIGIN", "Request forbidden")
        if not any(referer_origin == origin.rstrip('/') for origin in ALLOWED_ORIGINS):
            _finalize_request_telemetry()
            log.warning(f'Blocked request with disallowed Referer: {referer_origin}')
            log_security_event(
                "disallowed_referer",
                request=request,
                referer=referer_origin,
            )
            raise AskAPIError(403, "FORBIDDEN_ORIGIN", "Request forbidden")

    slot_acquired = request_backpressure_gate.try_acquire()
    if not slot_acquired:
        metrics.log_load_shed()
        _finalize_request_telemetry()
        log_security_event(
            "load_shed_backpressure",
            request=request,
            max_concurrent=get_resilience_snapshot()["request_backpressure"]["max_concurrent"],
        )
        raise AskAPIError(
            503,
            "CAPACITY_EXHAUSTED",
            "Service busy. Please retry shortly.",
            retryable=True,
        )

    try:
        session_id, session_token, reused_existing_session = get_or_issue_session(
            x_session_token,
            SESSION_SIGNING_SECRET,
            actor_id=caller.actor_id,
            authoritative_session_id=(
                caller.session_id if caller.actor_assertion_verified else None
            ),
        )
        response.headers["X-Session-Token"] = session_token
        if x_session_token and not reused_existing_session:
            log_security_event(
                "invalid_session_token",
                request=request,
                provided=True,
            )
        stored_history = get_history(session_id, actor_id=caller.actor_id)

        # All history sources are hostile prompt data, including server-loaded
        # database turns and this process's own cached copy. Inspect both the
        # persisted/session path and the current transport before choosing one.
        bound_history, blocked_stored_turns = filter_caller_history(stored_history)
        caller_history, blocked_caller_turns = filter_caller_history(
            q.conversation_history,
            max_item_chars=2000,
        )
        blocked_history_turns = blocked_stored_turns + blocked_caller_turns
        if blocked_history_turns:
            log_security_event(
                "untrusted_history_turn_blocked",
                request=request,
                subject_id=caller.subject_id,
                stored_turns=blocked_stored_turns,
                caller_turns=blocked_caller_turns,
            )

        # If the in-process session has no history yet (e.g. first request after
        # deploy or session expiry) but the edge function provided server-loaded
        # history from the chat_history table, use it as a seed.  This bridges
        # the gap where the edge function cannot forward session tokens but *can*
        # load persisted turns from the database on behalf of the authenticated user.
        if not bound_history and caller_history:
            bound_history = caller_history
            if bound_history:
                seed_history(session_id, bound_history, actor_id=caller.actor_id)
                log.info(
                    "Seeded actor-bound session history. session_hash=%s turns=%d",
                    hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12],
                    len(bound_history),
                )
        elif caller_history and bound_history:
            # Session already has history — ignore the client-provided duplicate
            log.debug(
                "Ignoring duplicate caller history; session already has %d turns.",
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
            public_guardrail_metadata = build_public_response_metadata(
                None,
                request_id=request_id,
                trace_id=trace_id,
                metric_unit_registry_version=METRIC_UNITS.version,
                request_deadline=request_deadline.public_metadata(),
                answer_provenance=build_answer_provenance(None),
                protected_telemetry={"llm_telemetry": req_usage},
                guardrail={
                    "action": firewall_decision.action,
                    "reason": firewall_decision.reason,
                    "risk_score": firewall_decision.risk_score,
                },
            )
            safe_answer = build_safe_refusal_message(firewall_decision)
            if selected_contract_version == CHAT_GATEWAY_V2_CONTRACT_VERSION:
                return _v2_response(
                    build_chat_gateway_v2_response(
                        None,
                        answer=safe_answer,
                        request_id=request_id,
                        execution_time=exec_time,
                        answer_provenance=build_answer_provenance(None),
                        session_continuity_available=bool(
                            reused_existing_session or bound_history
                        ),
                        terminal_outcome=TerminalOutcome.POLICY_BLOCKED,
                    )
                )
            return APIResponse(
                answer=safe_answer,
                charts=None,
                chart_data=None,
                chart_type=None,
                chart_metadata=public_guardrail_metadata,
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
                previous_contract_snapshot=get_last_contract(
                    session_id,
                    actor_id=caller.actor_id,
                ),
                request_deadline=request_deadline,
            )
        except RequestDeadlineExceeded as exc:
            _finalize_request_telemetry()
            log_security_event(
                "request_deadline_exhausted",
                request=request,
                stage=exc.stage,
            )
            raise AskAPIError(
                408,
                "REQUEST_DEADLINE_EXCEEDED",
                "Request deadline exceeded",
                retryable=True,
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
                error_class=type(e).__name__,
            )
            # Generic detail for every auth mode: the exception is already in
            # the server log (log.exception above) and the security event.
            raise AskAPIError(
                500,
                "QUERY_PROCESSING_FAILED",
                "Query processing failed",
            )

        append_exchange(session_id, query_text, ctx.summary, actor_id=caller.actor_id)
        _contract_snapshot = continuity_snapshot_json(ctx)
        if _contract_snapshot:
            set_last_contract(session_id, _contract_snapshot, actor_id=caller.actor_id)
        req_usage = _finalize_request_telemetry()

        exec_time = time.time() - t0
        metrics.log_request(exec_time)
        log.info(f'Finished request in {exec_time:.2f}s')

        response_chart_meta = build_public_response_metadata(
            ctx,
            request_id=request_id,
            trace_id=trace_id,
            metric_unit_registry_version=METRIC_UNITS.version,
            request_deadline=request_deadline.public_metadata(),
            answer_provenance=build_answer_provenance(ctx),
            protected_telemetry={
                "stage_timings_ms": dict(ctx.stage_timings_ms),
                "llm_telemetry": req_usage,
                "session_bound_history_turns": len(bound_history),
                "summary_claims": list(ctx.summary_claims or []),
                "summary_claim_provenance": list(ctx.summary_claim_provenance or []),
            },
        )

        response.headers["X-Enai-Deadline-Remaining-Ms"] = str(request_deadline.remaining_ms())
        if selected_contract_version == CHAT_GATEWAY_V2_CONTRACT_VERSION:
            return _v2_response(
                build_chat_gateway_v2_response(
                    ctx,
                    answer=ctx.summary,
                    request_id=request_id,
                    execution_time=exec_time,
                    answer_provenance=build_answer_provenance(ctx),
                    session_continuity_available=bool(
                        reused_existing_session or bound_history
                    ),
                )
            )

        return APIResponse(
            answer=ctx.summary,
            charts=(project_public_charts(getattr(ctx, "charts", None)) or None),
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

        # CRITICAL: host '0.0.0.0' is required for container accessibility
        log.info(f"🚀 Starting Uvicorn server on 0.0.0.0:{HTTP_SERVER_PORT}")
        # Pass the app object directly so executing main.py does not import and
        # initialize the module a second time inside Uvicorn.
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=HTTP_SERVER_PORT,
            workers=HTTP_SERVER_WORKERS,
            log_level="info",
        )
    except ImportError:
        log.error("Uvicorn is not installed. Please install it with 'pip install uvicorn'.")
        raise
    except Exception:
        log.exception("FATAL: Uvicorn server failed to start")
        raise
