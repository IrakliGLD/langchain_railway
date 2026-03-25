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
import os
import re
import time
import logging
import uuid
import json
from urllib.parse import urlparse
from contextvars import ContextVar
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Header, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Phase 1D Security: Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from sqlalchemy import text
from sqlalchemy.exc import OperationalError, DatabaseError, SQLAlchemyError

import pandas as pd
import numpy as np

from dotenv import load_dotenv

# Schema & helpers
from context import DB_SCHEMA_DOC, scrub_schema_mentions, COLUMN_LABELS, DERIVED_LABELS
import knowledge as knowledge_module

# ============================================================================
# REFACTORED MODULES (Phases 1-4)
# ============================================================================
# Phase 1: Configuration and Models
from config import *  # All configuration constants
from models import Question, APIResponse, MetricsResponse

# Phase 6: Pipeline
from agent.pipeline import process_query

# Phase 2: Core modules
from utils.metrics import metrics
from utils.session_memory import get_or_issue_session, get_history, append_exchange, seed_history
from utils.auth import authenticate_request, CallerContext
from utils.resilience import request_backpressure_gate, get_resilience_snapshot
from utils.language import detect_language, get_language_instruction
from utils.query_validation import (
    is_conceptual_question,
    validate_sql_relevance,
    should_skip_sql_execution
)
from core.query_executor import ENGINE, execute_sql_safely, is_database_available
from core.sql_generator import simple_table_whitelist_check, sanitize_sql, plan_validate_repair
from core.llm import (
    llm_cache,
    make_gemini,
    make_openai,
    llm_generate_plan_and_sql,
    llm_summarize,
    classify_query_type,
    get_query_focus
)

# Phase 3: Analysis modules
from analysis.stats import quick_stats, rows_to_preview
from analysis.seasonal import compute_seasonal_average
from analysis.seasonal_stats import (
    detect_monthly_timeseries,
    calculate_seasonal_stats,
    format_seasonal_stats
)
from analysis.shares import (
    build_balancing_correlation_df,
    compute_weighted_balancing_price,
    compute_entity_price_contributions
)
from guardrails.firewall import inspect_query, build_safe_refusal_message

# Phase 4: Visualization modules
from visualization.chart_selector import (
    should_generate_chart,
    infer_dimension,
    detect_column_types,
    select_chart_type
)
from visualization.chart_builder import prepare_chart_data
# ============================================================================

# -----------------------------
# Boot + Config
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("Enai")

# Load knowledge files from knowledge/ directory at startup
knowledge_module.load_knowledge()

if not SUPABASE_JWT_SECRET:
    log.warning("SUPABASE_JWT_SECRET not set — public bearer auth is disabled")

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
from agent.sql_executor import (
    should_inject_balancing_pivot,
    build_trade_share_cte,
    fetch_balancing_share_panel,
    ensure_share_dataframe,
    BALANCING_SHARE_PIVOT_SQL,
    BALANCING_SEGMENT_NORMALIZER,
)
from agent.analyzer import (
    generate_share_summary,
    build_share_shift_notes,
    BALANCING_SHARE_METADATA,
    MONTH_NAME_TO_NUMBER,
)
from agent.planner import (  # noqa: F401 — backward compat re-exports
    detect_analysis_mode,
    ANALYTICAL_KEYWORDS,
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
        if schema_map:
            ALLOWED_TABLES.clear()
            ALLOWED_TABLES.update(schema_map.keys())
        else:
            ALLOWED_TABLES.clear()
            ALLOWED_TABLES.update(STATIC_ALLOWED_TABLES)
        log.info("Schema reflection complete: %s views", len(SCHEMA_MAP))
        return True
    except Exception as exc:
        SCHEMA_MAP = {}
        ALLOWED_TABLES.clear()
        ALLOWED_TABLES.update(STATIC_ALLOWED_TABLES)
        log.warning("Schema reflection unavailable at startup: %s", exc)
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
app = FastAPI(title="Enai Analyst (Gemini)", version="18.6") # Version bump
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
        return "gateway:internal"
    return f"ip:{get_remote_address(request)}"


# Post-auth per-user rate limiter for bearer callers.
# Simple sliding-window counter in process memory.
_user_rate_buckets: Dict[str, List[float]] = {}
_user_rate_lock = __import__("threading").Lock()


def _check_user_rate_limit(subject_id: str) -> bool:
    """Return True if the user is within their per-minute rate limit."""
    now = time.time()
    window = 60.0
    max_requests = ASK_RATE_LIMIT_PUBLIC_PER_MINUTE
    with _user_rate_lock:
        timestamps = _user_rate_buckets.get(subject_id, [])
        # Prune expired entries
        timestamps = [t for t in timestamps if now - t < window]
        if not timestamps:
            _user_rate_buckets.pop(subject_id, None)
            return True
        if len(timestamps) >= max_requests:
            _user_rate_buckets[subject_id] = timestamps
            return False
        timestamps.append(now)
        _user_rate_buckets[subject_id] = timestamps
        return True


limiter = Limiter(key_func=_rate_limit_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request ID middleware for observability and debugging
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracing and debugging."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)

        # Log request start
        log.info(f"[{request_id}] {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            log.info(f"[{request_id}] Response: {response.status_code}")
            return response
        except Exception as e:
            log.error(f"[{request_id}] Error: {e}")
            raise

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Specific origins from environment (SECURITY FIX)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Only needed methods
    allow_headers=["Content-Type", "x-app-key", "Authorization", "x-session-token"],  # Only needed headers
    expose_headers=["X-Request-ID", "X-Trace-ID", "X-Session-Token", "X-LLM-Total-Tokens", "X-LLM-Estimated-Cost-USD"],
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
@app.on_event("startup")
def on_startup() -> None:
    """Non-blocking startup hook to prime schema cache and validate skills."""
    refresh_schema_map()
    from skills.loader import validate_skills, warmup_cache
    validate_skills()
    warmup_cache()


@app.get("/ask")
def ask_get():
    return {
        "message": "POST /ask with JSON: {'query': '...'} and a valid Authorization header."
    }

@app.get("/healthz")
def healthz():
    """Liveness probe: process is up."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness probe: DB connectivity + schema reflection health."""
    db_ready = is_database_available()
    schema_ready = bool(SCHEMA_MAP)
    payload = {
        "status": "ready" if db_ready else "degraded",
        "database_ready": db_ready,
        "schema_ready": schema_ready,
    }
    if db_ready:
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
            "gemini_model": GEMINI_MODEL if MODEL_TYPE == "gemini" else None,
            "openai_model": OPENAI_MODEL if MODEL_TYPE == "openai" else None,
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
        from evaluation_engine import (
            load_evaluation_dataset,
            filter_queries,
            run_single_evaluation,
            generate_summary
        )

        # Load dataset
        dataset = load_evaluation_dataset()
        queries = dataset["queries"]

        # Filter queries
        filtered_queries = filter_queries(queries, mode=mode, query_type=type, query_id=query_id)

        if not filtered_queries:
            raise HTTPException(status_code=404, detail=f"No queries found matching filters")

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

# main.py v18.7 — Gemini Analyst (chart rules + period aggregation)
# (Only added targeted comments/logic for: 1) chart axis restriction; 2) user-defined period aggregation)

# ... [all your imports and setup remain IDENTICAL above this point] ...


@app.post('/ask', response_model=APIResponse)
@limiter.limit(f"{ASK_RATE_LIMIT_GATEWAY_PER_MINUTE}/minute")
def ask_post(
    request: Request,
    response: Response,
    q: Question,
    x_app_key: Optional[str] = Header(None, alias='X-App-Key'),
    x_session_token: Optional[str] = Header(None, alias='X-Session-Token'),
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

    # Per-user rate limit for public bearer callers (post-auth).
    # The decorator-level limiter uses IP keys and can't differentiate
    # by authenticated identity.  This enforces per-user limits after
    # the token has been verified.
    if caller.auth_mode == "public_bearer":
        if not _check_user_rate_limit(caller.subject_id):
            _finalize_request_telemetry()
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
            bound_history = [
                {
                    "question": str(t.get("question", ""))[:_MAX_HISTORY_ITEM_CHARS],
                    "answer": str(t.get("answer", ""))[:_MAX_HISTORY_ITEM_CHARS],
                }
                for t in q.conversation_history[:3]
                if isinstance(t, dict) and t.get("question")
            ]
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
            _caller = getattr(request.state, "caller", None)
            if _caller and _caller.auth_mode == "public_bearer":
                raise HTTPException(status_code=500, detail="Query processing failed")
            raise HTTPException(status_code=500, detail=f'Query processing failed: {e}')

        append_exchange(session_id, query_text, ctx.summary)
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
            }
        )

        return APIResponse(
            answer=ctx.summary,
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
    except Exception as e:
        log.error(f"FATAL: Uvicorn server failed to start: {e}")
