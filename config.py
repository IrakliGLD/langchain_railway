"""
Application configuration and constants.

Extracts all environment variables, constants, and configuration
from the monolithic main.py for better organization.
"""
import os
import re
from textwrap import dedent

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===================================================================
# Environment Variables
# ===================================================================

# LLM API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# NVIDIA (build.nvidia.com) is OpenAI-API-compatible; reached via ChatOpenAI
# with a custom base_url. Key supplied via env, like the providers above.
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Database
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def _read_secret_env(*names: str):
    """Read a secret env var, preferring the first name and tolerating wrapped quotes."""
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value or None
    return None


def _read_bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read an integer environment setting and fail closed outside safe bounds."""
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _read_single_worker_count(*names: str) -> int:
    """Reject worker settings that would split process-local runtime state."""
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None or not raw_value.strip():
            continue
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be the integer 1") from exc
        if value != 1:
            raise RuntimeError(
                f"{name} must be 1 while session, replay, and rate-limit state are process-local"
            )
    return 1


# API Security
# Prefer the new ENAI_* names; fall back to the earlier split-secret names during rollout.
GATEWAY_SHARED_SECRET = _read_secret_env("ENAI_GATEWAY_SECRET", "GATEWAY_SHARED_SECRET")
SESSION_SIGNING_SECRET = _read_secret_env("ENAI_SESSION_SIGNING_SECRET", "SESSION_SIGNING_SECRET")
EVALUATE_ADMIN_SECRET = _read_secret_env("ENAI_EVALUATE_SECRET", "EVALUATE_ADMIN_SECRET")
ENAI_AUTH_MODE = (os.getenv("ENAI_AUTH_MODE", "gateway_only").strip().lower() or "gateway_only")
ENAI_DEPLOYMENT_ENV = (os.getenv("ENAI_DEPLOYMENT_ENV", "development").strip().lower() or "development")
# The Railway service uses a fixed target port.  Keep the application bind
# independent of Railway's injected dynamic PORT value so the process and
# healthcheck target cannot silently diverge.
HTTP_SERVER_PORT = _read_bounded_int_env("ENAI_HTTP_PORT", 3000, 1024, 65535)
HTTP_SERVER_WORKERS = _read_single_worker_count(
    "ENAI_HTTP_WORKERS",
    "WEB_CONCURRENCY",
    "UVICORN_WORKERS",
)
SCHEMA_READINESS_CACHE_TTL_SECONDS = _read_bounded_int_env(
    "ENAI_SCHEMA_READINESS_CACHE_TTL_SECONDS", 60, 5, 3600
)
SCHEMA_READINESS_RETRY_INTERVAL_SECONDS = _read_bounded_int_env(
    "ENAI_SCHEMA_READINESS_RETRY_INTERVAL_SECONDS", 10, 1, 300
)
# P7.A database identity gate. Staging/production must connect as the
# dedicated read-only role; development/test can leave the value empty.
DATABASE_RUNTIME_ROLE = os.getenv(
    "ENAI_DB_RUNTIME_ROLE",
    "enai_api_readonly" if ENAI_DEPLOYMENT_ENV in {"staging", "production"} else "",
).strip()
# Raw routing-fixture capture is a local/test-only diagnostic. Production
# observability remains content-free even when other debug traces are enabled.
FIXTURE_CAPTURE_MODE = os.getenv("ENAI_FIXTURE_CAPTURE_MODE", "off").strip().lower() or "off"
try:
    FIXTURE_CAPTURE_SAMPLE_RATE = float(os.getenv("ENAI_FIXTURE_CAPTURE_SAMPLE_RATE", "0"))
except ValueError as exc:
    raise RuntimeError("ENAI_FIXTURE_CAPTURE_SAMPLE_RATE must be a number") from exc
if FIXTURE_CAPTURE_MODE not in {"off", "raw"}:
    raise RuntimeError("ENAI_FIXTURE_CAPTURE_MODE must be one of: off, raw")
if not 0.0 <= FIXTURE_CAPTURE_SAMPLE_RATE <= 1.0:
    raise RuntimeError("ENAI_FIXTURE_CAPTURE_SAMPLE_RATE must be between 0 and 1")
if FIXTURE_CAPTURE_MODE == "raw" and ENAI_DEPLOYMENT_ENV not in {"development", "test"}:
    raise RuntimeError("Raw fixture capture is restricted to development and test")
# P3.A rollout gate for the edge-signed actor/session/request assertion.
# ``optional`` accepts independently deployed P1 gateways while verifying any
# assertion that is present. Switch to ``required`` only after the P3.B edge
# artifact is deployed and its signed requests are visible in backend logs.
GATEWAY_ACTOR_ASSERTION_MODE = (
    os.getenv("ENAI_GATEWAY_ACTOR_ASSERTION_MODE", "optional").strip().lower()
    or "optional"
)
GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS = _read_bounded_int_env(
    "ENAI_GATEWAY_ACTOR_ASSERTION_MAX_AGE_SECONDS",
    120,
    minimum=30,
    maximum=900,
)
# Phase 13 rollout flag. When "1"/"true", chart_pipeline attaches a long-form
# ChartFrame payload under chart_meta["longFrame"] for frontend consumers.
# Default off so wire format stays wide until the frontend renderer
# adopts the new shape.
ENAI_CHART_LONGFORM = os.getenv("ENAI_CHART_LONGFORM", "0").strip().lower() in ("1", "true", "yes", "on")
# Supabase JWT secret for local bearer-token verification.
# Direct bearer authentication is an explicit security-boundary decision.
# Merely configuring SUPABASE_JWT_SECRET must never enable it implicitly.
SUPABASE_JWT_SECRET = _read_secret_env("SUPABASE_JWT_SECRET")
ALLOW_EVALUATE_ENDPOINT = os.getenv("ALLOW_EVALUATE_ENDPOINT", "false").lower() in ("1", "true", "yes", "on")
ENABLE_PUBLIC_BEARER_AUTH = ENAI_AUTH_MODE == "gateway_and_bearer"

# LLM Configuration
# MODEL_TYPE selects the active provider: "gemini" (default), "openai", or "nvidia".
MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# NVIDIA (build.nvidia.com) — OpenAI-API-compatible endpoint driven via ChatOpenAI.
# Any NIM model id works env-only, e.g. NVIDIA_MODEL=z-ai/glm-5.2 (with
# MODEL_TYPE=nvidia); namespaced vendor/model ids classify to this provider
# for cost attribution automatically. Restart required (env read at import).
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-120b")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Output-token cap and sampling temperature for the NVIDIA client. max_tokens
# matters for reasoning models (gpt-oss-120b, glm-5.2) whose hidden reasoning
# counts against the output budget — raise NVIDIA_MAX_TOKENS (e.g. 16384) if
# answers get truncated.
NVIDIA_MAX_TOKENS = max(1, int(os.getenv("NVIDIA_MAX_TOKENS", "4096")))
NVIDIA_TEMPERATURE = float(os.getenv("NVIDIA_TEMPERATURE", "0"))
# Wall-clock bound per NVIDIA call, in seconds. Defaults to 90; explicitly set
# to 0 only when an unbounded call is intentionally required. A slow shared-endpoint model (the
# z-ai/glm-5.2 incident: 113s analyzer + 192s summarizer calls at ~12 tok/s)
# times out and falls back to OpenAI instead of holding the request for
# minutes; max_retries drops to 1 because retrying a timeout on a slow model
# only multiplies the wait (same rationale as the Gemini summarizer client).
_raw_nvidia_timeout = os.getenv("NVIDIA_TIMEOUT_SECONDS", "90").strip()
NVIDIA_TIMEOUT_SECONDS: float | None = float(_raw_nvidia_timeout) if _raw_nvidia_timeout and float(_raw_nvidia_timeout) > 0 else None

# P5.1 (finding H13): OpenAI had no explicit per-call timeout, so a stalled
# OpenAI call — whether primary or the universal fallback — could hold a request
# open indefinitely, defeating the end-to-end deadline. Bound it like Gemini
# (120s default) and NVIDIA; a timeout drops retries to 1 so a slow call fails
# over once rather than multiplying the wait. Set 0 to intentionally unbound.
_raw_openai_timeout = os.getenv("OPENAI_TIMEOUT_SECONDS", "120").strip()
OPENAI_TIMEOUT_SECONDS: float | None = (
    float(_raw_openai_timeout) if _raw_openai_timeout and float(_raw_openai_timeout) > 0 else None
)

# Per-stage model overrides.  When set, the named pipeline stage uses this
# model instead of the global GEMINI_MODEL / OPENAI_MODEL.  Leave unset (or
# empty) to inherit the global default.  Only Gemini model names are supported
# for overrides; the global MODEL_TYPE still governs the provider choice.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "").strip() or None
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "").strip() or None
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "").strip() or None

# Thinking-budget cap for the router/question-analyzer stage.
# Limits thinking tokens on Gemini 2.5 models to prevent latency spirals.
# Default 1024 (lowered from 2048, 2026-07-07): classification rarely needs
# more, and the budget is paid in latency+cost on every request; raise via
# env if routing accuracy regresses. Set to 0 to disable thinking.
# Non-thinking models silently ignore this parameter.
_raw_tb = os.getenv("ROUTER_THINKING_BUDGET", "1024").strip()
ROUTER_THINKING_BUDGET: int | None = int(_raw_tb) if _raw_tb else None

# Pipeline effort mode (Phase E / 14.9 steps 9-10).
# ``deep`` preserves the current high-quality defaults.
# ``fast`` keeps the same contract shape but reduces prompt/retrieval effort.
PIPELINE_MODE = (os.getenv("PIPELINE_MODE", "deep").strip().lower() or "deep")
if PIPELINE_MODE not in {"deep", "fast"}:
    raise RuntimeError("Invalid PIPELINE_MODE. Expected one of: deep, fast")

# Query Limits
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))
# 256 KiB fits a maximally valid Question payload even when every non-BMP
# character is represented as a JSON surrogate-pair escape. The 1 MiB ceiling
# prevents configuration drift from silently disabling pre-parse containment.
MAX_REQUEST_BODY_BYTES = _read_bounded_int_env(
    "MAX_REQUEST_BODY_BYTES",
    262144,
    minimum=262144,
    maximum=1048576,
)
ASK_DEFAULT_REQUEST_BUDGET_MS = _read_bounded_int_env(
    "ASK_DEFAULT_REQUEST_BUDGET_MS",
    110000,
    minimum=1000,
    maximum=300000,
)
ASK_MAX_REQUEST_BUDGET_MS = _read_bounded_int_env(
    "ASK_MAX_REQUEST_BUDGET_MS",
    115000,
    minimum=ASK_DEFAULT_REQUEST_BUDGET_MS,
    maximum=300000,
)
ENABLE_TYPED_TOOLS = os.getenv("ENABLE_TYPED_TOOLS", "true").lower() in ("1", "true", "yes", "on")
ENABLE_EVIDENCE_PLANNER = os.getenv("ENABLE_EVIDENCE_PLANNER", "true").lower() in ("1", "true", "yes", "on")
# Stage 0.8: prefetch secondary evidence tool calls concurrently (2 workers,
# sized to the core/db.py pool budget of 5). Results are applied strictly in
# plan order, so storage/log/trace order is identical to the serial path;
# this is a kill switch, not a rollout gate.
EVIDENCE_PARALLEL_SECONDARY = os.getenv("EVIDENCE_PARALLEL_SECONDARY", "true").lower() in ("1", "true", "yes", "on")
# Stage 0.2 contract continuity (slice 1): inject the previous turn's routed
# contract as a TRUSTED analyzer-prompt block so follow-ups ("and for 2023")
# are interpreted as deltas. Default OFF — cutover criteria in
# docs/active/query_pipeline_architecture.md §5.
ENABLE_CONTRACT_CONTINUITY = os.getenv("ENABLE_CONTRACT_CONTINUITY", "false").lower() in ("1", "true", "yes", "on")
# Evidence-triggered re-analysis (design item 1): on surprising primary
# evidence (empty frame / period gap), re-run Stage 0.2 ONCE with the anomaly
# as trusted context and re-execute the plan. Detection + counters are always
# on (shadow); the retry is gated here. Default OFF — enablement criteria in
# docs/active/query_pipeline_architecture.md §5.
ENABLE_EVIDENCE_REANALYSIS = os.getenv("ENABLE_EVIDENCE_REANALYSIS", "false").lower() in ("1", "true", "yes", "on")
# F3: percentage gates are evaluated per trusted actor/session/request. A
# value of 100 preserves the historical meaning of explicitly enabling the
# master flag; operators can set 5/25/100 for a deterministic canary.
EVIDENCE_REANALYSIS_PERCENT = _read_bounded_int_env(
    "ENAI_EVIDENCE_REANALYSIS_PERCENT", 100, minimum=0, maximum=100,
)
# P4.1 (finding H1): one evidence-finalization routine on every evidence path.
#   off     — legacy behavior: frames attach only on the analyzer recovery path.
#   shadow  — frames are built + validated everywhere but stored as telemetry
#             only (ctx.evidence_frame_shadow); rendering behavior unchanged.
#   enforce — frames attach on every path; the generic deterministic renderer
#             becomes reachable for normal tool execution.
# Default "shadow": behavior-neutral, and produces the comparison telemetry the
# P4 exit gate requires (house pattern: detection always on, cutover gated).
EVIDENCE_FINALIZATION_MODE = (
    os.getenv("ENAI_EVIDENCE_FINALIZATION_MODE", "shadow").strip().lower() or "shadow"
)
EVIDENCE_FINALIZATION_ENFORCE_PERCENT = _read_bounded_int_env(
    "ENAI_EVIDENCE_FINALIZATION_ENFORCE_PERCENT", 100, minimum=0, maximum=100,
)
# P4.2 (finding M3): plan validation against the answer contract.
#   warn    — typed issues are computed, logged, and counted; behavior unchanged.
#   enforce — reject-severity issues terminal-clarify BEFORE any tool/DB call.
# Default "warn" keeps the documented warning-only behavior until the operator
# reviews shadow counters and cuts over.
PLAN_VALIDATION_MODE = (
    os.getenv("ENAI_PLAN_VALIDATION_MODE", "warn").strip().lower() or "warn"
)
PLAN_VALIDATION_ENFORCE_PERCENT = _read_bounded_int_env(
    "ENAI_PLAN_VALIDATION_ENFORCE_PERCENT", 100, minimum=0, maximum=100,
)
# P4.4 (finding H12): honest terminal outcomes. When enabled, a data-primary
# request whose generated SQL fails validation/relevance returns a transparent
# evidence-unavailable answer (no numeric claims) instead of a conceptual
# narrative that masks the data failure. The terminal-outcome taxonomy and its
# shadow telemetry are always on regardless of this flag; only the user-facing
# routing change is gated. Default OFF until the operator reviews the
# evidence_unavailable_shadow counter and cuts over.
ENABLE_HONEST_TERMINAL_OUTCOMES = os.getenv(
    "ENAI_ENABLE_HONEST_TERMINAL_OUTCOMES", "false"
).strip().lower() in ("1", "true", "yes", "on")
HONEST_TERMINAL_OUTCOMES_PERCENT = _read_bounded_int_env(
    "ENAI_HONEST_TERMINAL_OUTCOMES_PERCENT", 100, minimum=0, maximum=100,
)
# Pre-auth rate limiting: trust the platform proxy's X-Forwarded-For (last
# hop) for the client IP. Default on — production always sits behind the
# Railway edge, where the socket peer is the proxy and would collapse every
# caller into one shared bucket. Set to false for direct-exposure deployments.
TRUST_PROXY_CLIENT_IP = os.getenv("TRUST_PROXY_CLIENT_IP", "true").lower() in ("1", "true", "yes", "on")
ENABLE_AGENT_LOOP = os.getenv("ENABLE_AGENT_LOOP", "true").lower() in ("1", "true", "yes", "on")
ENABLE_QUESTION_ANALYZER_SHADOW = os.getenv("ENABLE_QUESTION_ANALYZER_SHADOW", "false").lower() in ("1", "true", "yes", "on")
ENABLE_QUESTION_ANALYZER_HINTS = os.getenv("ENABLE_QUESTION_ANALYZER_HINTS", "true").lower() in ("1", "true", "yes", "on")
ENABLE_TRACE_DEBUG_ARTIFACTS = os.getenv("ENABLE_TRACE_DEBUG_ARTIFACTS", "false").lower() in ("1", "true", "yes", "on")
ENABLE_SKILL_PROMPTS_SUMMARIZER = os.getenv("ENABLE_SKILL_PROMPTS_SUMMARIZER", "true").lower() in ("1", "true", "yes", "on")
ENABLE_SKILL_PROMPTS_PLANNER = os.getenv("ENABLE_SKILL_PROMPTS_PLANNER", "true").lower() in ("1", "true", "yes", "on")
ENABLE_VECTOR_KNOWLEDGE_SHADOW = os.getenv("ENABLE_VECTOR_KNOWLEDGE_SHADOW", "false").lower() in ("1", "true", "yes", "on")
ENABLE_METRICS_ENDPOINT = os.getenv("ENABLE_METRICS_ENDPOINT", "false").lower() in ("1", "true", "yes", "on")
ENABLE_EVALUATE_ENDPOINT = os.getenv("ENABLE_EVALUATE_ENDPOINT", "false").lower() in ("1", "true", "yes", "on")
ENABLE_VECTOR_KNOWLEDGE_HINTS = os.getenv("ENABLE_VECTOR_KNOWLEDGE_HINTS", "true").lower() in ("1", "true", "yes", "on")
TRACE_TEXT_MAX_CHARS = max(120, int(os.getenv("TRACE_TEXT_MAX_CHARS", "800")))
TRACE_MAX_LIST_ITEMS = max(1, int(os.getenv("TRACE_MAX_LIST_ITEMS", "8")))
PROMPT_BUDGET_MAX_CHARS = max(1500, int(os.getenv("PROMPT_BUDGET_MAX_CHARS", "45000")))
# Per-stage prompt budgets (Phase 2.b, 2026-05-13).  Both default to the
# legacy single-knob PROMPT_BUDGET_MAX_CHARS so existing deployments keep
# their current behaviour; operators raise SUMMARIZER_PROMPT_BUDGET_MAX_CHARS
# independently when the summarizer prompt routinely exceeds the analyzer
# budget (typical in deep mode where DOMAIN_KNOWLEDGE + EXTERNAL_SOURCE_PASSAGES
# expand the prompt past 90k chars before truncation kicks in).
ANALYZER_PROMPT_BUDGET_MAX_CHARS = max(
    1500,
    int(os.getenv("ANALYZER_PROMPT_BUDGET_MAX_CHARS", str(PROMPT_BUDGET_MAX_CHARS))),
)
SUMMARIZER_PROMPT_BUDGET_MAX_CHARS = max(
    1500,
    int(os.getenv("SUMMARIZER_PROMPT_BUDGET_MAX_CHARS", str(PROMPT_BUDGET_MAX_CHARS))),
)
FAST_MODE_ANALYZER_BUDGET = max(1500, int(os.getenv("FAST_MODE_ANALYZER_BUDGET", "20000")))
FAST_MODE_SUMMARIZER_BUDGET = max(1500, int(os.getenv("FAST_MODE_SUMMARIZER_BUDGET", "15000")))
ROUTER_ENABLE_SEMANTIC_FALLBACK = os.getenv("ROUTER_ENABLE_SEMANTIC_FALLBACK", "true").lower() in ("1", "true", "yes", "on")
ROUTER_SEMANTIC_MIN_SCORE = min(
    1.0,
    max(0.1, float(os.getenv("ROUTER_SEMANTIC_MIN_SCORE", "0.55"))),
)
SESSION_HISTORY_MAX_TURNS = max(1, int(os.getenv("SESSION_HISTORY_MAX_TURNS", "3")))
SESSION_IDLE_TTL_SECONDS = max(60, int(os.getenv("SESSION_IDLE_TTL_SECONDS", "3600")))
ASK_MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("ASK_MAX_CONCURRENT_REQUESTS", "8")))
ASK_BACKPRESSURE_TIMEOUT_SECONDS = max(0.0, float(os.getenv("ASK_BACKPRESSURE_TIMEOUT_SECONDS", "0.0")))
ASK_RATE_LIMIT_PUBLIC_PER_MINUTE = max(1, int(os.getenv("ASK_RATE_LIMIT_PUBLIC_PER_MINUTE", "10")))
ASK_RATE_LIMIT_GATEWAY_PER_MINUTE = max(1, int(os.getenv("ASK_RATE_LIMIT_GATEWAY_PER_MINUTE", "30")))
ASK_RATE_LIMIT_PREAUTH_PER_MINUTE = max(1, int(os.getenv("ASK_RATE_LIMIT_PREAUTH_PER_MINUTE", "300")))
VECTOR_KNOWLEDGE_TOP_K = max(1, int(os.getenv("VECTOR_KNOWLEDGE_TOP_K", "6")))
VECTOR_KNOWLEDGE_SEARCH_MULTIPLIER = max(1, int(os.getenv("VECTOR_KNOWLEDGE_SEARCH_MULTIPLIER", "3")))
VECTOR_KNOWLEDGE_MAX_CHARS = max(500, int(os.getenv("VECTOR_KNOWLEDGE_MAX_CHARS", "9000")))
VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER = os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_PROVIDER", "openai").strip().lower() or "openai"
VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION = max(1, int(os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_DIMENSION", "1536")))
VECTOR_KNOWLEDGE_EMBEDDING_MODEL = os.getenv("VECTOR_KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-small").strip()
VECTOR_KNOWLEDGE_SCHEMA = os.getenv("VECTOR_KNOWLEDGE_SCHEMA", "knowledge").strip() or "knowledge"
VECTOR_KNOWLEDGE_STORAGE_BUCKET = os.getenv("VECTOR_KNOWLEDGE_STORAGE_BUCKET", "knowledge-documents").strip() or "knowledge-documents"
VECTOR_KNOWLEDGE_MIN_SIMILARITY = min(1.0, max(0.0, float(os.getenv("VECTOR_KNOWLEDGE_MIN_SIMILARITY", "0.2"))))
VECTOR_KNOWLEDGE_CHUNK_TARGET_TOKENS = max(100, int(os.getenv("VECTOR_KNOWLEDGE_CHUNK_TARGET_TOKENS", "650")))
VECTOR_KNOWLEDGE_CHUNK_OVERLAP_TOKENS = max(0, int(os.getenv("VECTOR_KNOWLEDGE_CHUNK_OVERLAP_TOKENS", "100")))
PROVENANCE_MIN_COVERAGE = min(
    1.0,
    max(0.0, float(os.getenv("PROVENANCE_MIN_COVERAGE", "0.8"))),
)

# Pipeline routing thresholds (env-overridable)
ANALYZER_CONFIDENCE_OVERRIDE_THRESHOLD = float(os.getenv("ANALYZER_CONFIDENCE_OVERRIDE_THRESHOLD", "0.8"))
ANALYZER_TOOL_MIN_SCORE = float(os.getenv("ANALYZER_TOOL_MIN_SCORE", "0.55"))
ROUTER_SEMANTIC_GAP_THRESHOLD = float(os.getenv("ROUTER_SEMANTIC_GAP_THRESHOLD", "0.08"))
TOOL_RELEVANCE_OVERLAP_THRESHOLD = float(os.getenv("TOOL_RELEVANCE_OVERLAP_THRESHOLD", "0.3"))
ANALYZER_TOPIC_MIN_SCORE = float(os.getenv("ANALYZER_TOPIC_MIN_SCORE", "0.2"))

# Reliability / circuit breaker settings
LLM_CB_FAILURE_THRESHOLD = max(1, int(os.getenv("LLM_CB_FAILURE_THRESHOLD", "5")))
LLM_CB_RESET_TIMEOUT_SECONDS = max(5, int(os.getenv("LLM_CB_RESET_TIMEOUT_SECONDS", "30")))
DB_CB_FAILURE_THRESHOLD = max(1, int(os.getenv("DB_CB_FAILURE_THRESHOLD", "5")))
DB_CB_RESET_TIMEOUT_SECONDS = max(5, int(os.getenv("DB_CB_RESET_TIMEOUT_SECONDS", "30")))

# Optional LLM cost telemetry rates (USD per 1K tokens).
# Defaults to 0 so deployments can explicitly configure their own pricing.
OPENAI_INPUT_COST_PER_1K_USD = float(os.getenv("OPENAI_INPUT_COST_PER_1K_USD", "0"))
OPENAI_OUTPUT_COST_PER_1K_USD = float(os.getenv("OPENAI_OUTPUT_COST_PER_1K_USD", "0"))
GEMINI_INPUT_COST_PER_1K_USD = float(os.getenv("GEMINI_INPUT_COST_PER_1K_USD", "0"))
GEMINI_OUTPUT_COST_PER_1K_USD = float(os.getenv("GEMINI_OUTPUT_COST_PER_1K_USD", "0"))
NVIDIA_INPUT_COST_PER_1K_USD = float(os.getenv("NVIDIA_INPUT_COST_PER_1K_USD", "0"))
NVIDIA_OUTPUT_COST_PER_1K_USD = float(os.getenv("NVIDIA_OUTPUT_COST_PER_1K_USD", "0"))

# Memory Limits (PRODUCTION SAFETY: Prevents OOM errors)
MAX_RESULT_SIZE_MB = int(os.getenv("MAX_RESULT_SIZE_MB", "100"))

# ===================================================================
# Validation
# ===================================================================

def validate_runtime_settings(
    *,
    supabase_db_url: str | None,
    gateway_shared_secret: str | None,
    session_signing_secret: str | None,
    evaluate_admin_secret: str | None,
    auth_mode: str,
    deployment_env: str,
    supabase_jwt_secret: str | None,
    enable_evaluate_endpoint: bool,
    allow_evaluate_endpoint: bool,
    model_type: str,
    google_api_key: str | None,
    nvidia_api_key: str | None = None,
    gateway_actor_assertion_mode: str = "optional",
    evidence_finalization_mode: str = "shadow",
    plan_validation_mode: str = "warn",
    openai_api_key: str | None = None,
) -> None:
    valid_auth_modes = {"gateway_only", "gateway_and_bearer"}
    valid_deployment_envs = {"development", "staging", "production", "test"}

    if auth_mode not in valid_auth_modes:
        raise RuntimeError(
            "Invalid ENAI_AUTH_MODE. Expected one of: gateway_only, gateway_and_bearer"
        )
    if evidence_finalization_mode not in {"off", "shadow", "enforce"}:
        raise RuntimeError(
            "Invalid ENAI_EVIDENCE_FINALIZATION_MODE. Expected one of: off, shadow, enforce"
        )
    if plan_validation_mode not in {"warn", "enforce"}:
        raise RuntimeError(
            "Invalid ENAI_PLAN_VALIDATION_MODE. Expected one of: warn, enforce"
        )
    if deployment_env not in valid_deployment_envs:
        raise RuntimeError(
            "Invalid ENAI_DEPLOYMENT_ENV. Expected one of: development, staging, production, test"
        )
    if gateway_actor_assertion_mode not in {"optional", "required"}:
        raise RuntimeError(
            "Invalid ENAI_GATEWAY_ACTOR_ASSERTION_MODE. Expected one of: optional, required"
        )
    if not supabase_db_url:
        raise RuntimeError("Missing SUPABASE_DB_URL")
    if not gateway_shared_secret:
        raise RuntimeError("Missing ENAI_GATEWAY_SECRET (or legacy GATEWAY_SHARED_SECRET)")
    if not session_signing_secret:
        raise RuntimeError("Missing ENAI_SESSION_SIGNING_SECRET (or legacy SESSION_SIGNING_SECRET)")
    if not evaluate_admin_secret:
        raise RuntimeError("Missing ENAI_EVALUATE_SECRET (or legacy EVALUATE_ADMIN_SECRET)")
    if auth_mode == "gateway_and_bearer" and not supabase_jwt_secret:
        raise RuntimeError("ENAI_AUTH_MODE=gateway_and_bearer requires SUPABASE_JWT_SECRET")
    if auth_mode == "gateway_and_bearer" and deployment_env != "test":
        raise RuntimeError(
            "ENAI_AUTH_MODE=gateway_and_bearer is temporarily restricted to test environments "
            "until direct callers use the server-owned entitlement path"
        )
    if enable_evaluate_endpoint:
        if deployment_env not in {"development", "test"}:
            raise RuntimeError(
                "ENABLE_EVALUATE_ENDPOINT is only allowed when ENAI_DEPLOYMENT_ENV is development or test"
            )
        if not allow_evaluate_endpoint:
            raise RuntimeError(
                "ENABLE_EVALUATE_ENDPOINT=true requires ALLOW_EVALUATE_ENDPOINT=true"
            )
    valid_model_types = {"gemini", "openai", "nvidia"}
    if model_type not in valid_model_types:
        raise RuntimeError(
            "Invalid MODEL_TYPE. Expected one of: gemini, openai, nvidia"
        )
    if model_type == "gemini" and not google_api_key:
        raise RuntimeError("MODEL_TYPE=gemini but GOOGLE_API_KEY is missing")
    if model_type == "nvidia" and not nvidia_api_key:
        raise RuntimeError("MODEL_TYPE=nvidia but NVIDIA_API_KEY is missing")
    # P5.4 (finding M11): OpenAI-primary deployments previously started without
    # a key and failed at first use; every selected provider now validates its
    # credential at startup like the other two.
    if model_type == "openai" and not openai_api_key:
        raise RuntimeError("MODEL_TYPE=openai but OPENAI_API_KEY is missing")


validate_runtime_settings(
    supabase_db_url=SUPABASE_DB_URL,
    gateway_shared_secret=GATEWAY_SHARED_SECRET,
    session_signing_secret=SESSION_SIGNING_SECRET,
    evaluate_admin_secret=EVALUATE_ADMIN_SECRET,
    auth_mode=ENAI_AUTH_MODE,
    deployment_env=ENAI_DEPLOYMENT_ENV,
    supabase_jwt_secret=SUPABASE_JWT_SECRET,
    enable_evaluate_endpoint=ENABLE_EVALUATE_ENDPOINT,
    allow_evaluate_endpoint=ALLOW_EVALUATE_ENDPOINT,
    model_type=MODEL_TYPE,
    google_api_key=GOOGLE_API_KEY,
    nvidia_api_key=NVIDIA_API_KEY,
    gateway_actor_assertion_mode=GATEWAY_ACTOR_ASSERTION_MODE,
    evidence_finalization_mode=EVIDENCE_FINALIZATION_MODE,
    plan_validation_mode=PLAN_VALIDATION_MODE,
    openai_api_key=OPENAI_API_KEY,
)

# ===================================================================
# Database Configuration
# ===================================================================

# Allowed tables for SQL validation (whitelist)
STATIC_ALLOWED_TABLES = {
    "dates_mv",
    "entities_mv",
    "price_with_usd",
    "tariff_with_usd",
    "tech_quantity_view",
    "trade_derived_entities",
    "monthly_cpi_mv",
    "energy_balance_long_mv",
    "mv_balancing_trade_with_tariff",
}

ALLOWED_TABLES = set(STATIC_ALLOWED_TABLES)

# Allowed PostgreSQL-specific functions (Anonymous in sqlglot).
# Standard SQL functions (SUM, AVG, ROUND, CAST, etc.) are recognized by
# sqlglot as named classes and allowed implicitly.  Only functions that
# sqlglot cannot classify end up as Anonymous nodes — these must be on
# this allowlist or they are rejected by simple_table_whitelist_check().
ALLOWED_PG_FUNCTIONS = {
    # PostgreSQL-specific functions that sqlglot classifies as Anonymous.
    # Standard SQL functions (SUM, ROUND, CAST, COALESCE, window functions,
    # etc.) are recognized by sqlglot as named Func subclasses and are
    # allowed implicitly — they do NOT need to be listed here.
    #
    # NEVER add to this list: pg_sleep, pg_read_file, pg_terminate_backend,
    # pg_cancel_backend, set_config, dblink, lo_import, lo_export, pg_ls_dir,
    # pg_stat_file, pg_advisory_lock, pg_reload_conf, query_to_xml,
    # inet_server_addr, inet_client_addr, current_setting.
    "make_date", "age",
    "replace",
    "clock_timestamp", "statement_timestamp",
    "regexp_matches", "regexp_split_to_table",
    "json_build_object", "jsonb_build_object",
    "row_to_json", "jsonb_agg",
}

# Named sqlglot Func subclasses that leak server info or pose risk.
# These bypass the Anonymous check, so we deny them explicitly.
DENIED_SQL_FUNC_CLASSES = {
    "currentdatabase", "currentversion", "currentuser",
    "sessionuser", "currentschema",
}

# Table synonyms for auto-correction
TABLE_SYNONYMS = {
    "prices": "price",
    "tariffs": "tariff_gen",
    "price_usd": "price_with_usd",
    "tariff_usd": "tariff_with_usd",
    "price_with_usd": "price_with_usd",
}

# Column synonyms for auto-correction
COLUMN_SYNONYMS = {
    "tech_type": "type_tech",
    "quantity_mwh": "quantity_tech",
}

# Pre-compiled regex patterns for SQL synonym replacement (performance)
SYNONYM_PATTERNS = [
    (re.compile(r"\bprices\b", re.IGNORECASE), "price_with_usd"),
    (re.compile(r"\btariffs\b", re.IGNORECASE), "tariff_with_usd"),
    (re.compile(r"\btech_quantity\b", re.IGNORECASE), "tech_quantity_view"),
    (re.compile(r"\btrade\b", re.IGNORECASE), "trade_derived_entities"),
    (re.compile(r"\bentities\b", re.IGNORECASE), "entities_mv"),
    (re.compile(r"\bmonthly_cpi\b", re.IGNORECASE), "monthly_cpi_mv"),
    (re.compile(r"\benergy_balance_long\b", re.IGNORECASE), "energy_balance_long_mv"),
]

# Pre-compiled regex for LIMIT detection
LIMIT_PATTERN = re.compile(r"\bLIMIT\s*\d+\b", re.IGNORECASE)

# ===================================================================
# Analysis Configuration
# ===================================================================

# Seasonal months (canonical source of truth; tuples for immutability)
SUMMER_MONTHS = (4, 5, 6, 7)
WINTER_MONTHS = (1, 2, 3, 8, 9, 10, 11, 12)

# Balancing segment normalizer
BALANCING_SEGMENT_NORMALIZER = "LOWER(REPLACE(segment, ' ', '_'))"

# Balancing share pivot SQL — canonical definition lives in agent/sql_executor.py.
# Do not define a separate copy here to avoid drift.


