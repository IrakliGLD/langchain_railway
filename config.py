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


# API Security
# Prefer the new ENAI_* names; fall back to the earlier split-secret names during rollout.
GATEWAY_SHARED_SECRET = _read_secret_env("ENAI_GATEWAY_SECRET", "GATEWAY_SHARED_SECRET")
SESSION_SIGNING_SECRET = _read_secret_env("ENAI_SESSION_SIGNING_SECRET", "SESSION_SIGNING_SECRET")
EVALUATE_ADMIN_SECRET = _read_secret_env("ENAI_EVALUATE_SECRET", "EVALUATE_ADMIN_SECRET")

# LLM Configuration
MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Per-stage model overrides.  When set, the named pipeline stage uses this
# model instead of the global GEMINI_MODEL / OPENAI_MODEL.  Leave unset (or
# empty) to inherit the global default.  Only Gemini model names are supported
# for overrides; the global MODEL_TYPE still governs the provider choice.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "").strip() or None
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "").strip() or None
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "").strip() or None

# Thinking-budget cap for the router/question-analyzer stage.
# Limits thinking tokens on Gemini 2.5 models to prevent latency spirals.
# Default 2048 is enough for classification; set to 0 to disable thinking.
# Non-thinking models silently ignore this parameter.
_raw_tb = os.getenv("ROUTER_THINKING_BUDGET", "2048").strip()
ROUTER_THINKING_BUDGET: int | None = int(_raw_tb) if _raw_tb else None

# Query Limits
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))
ENABLE_TYPED_TOOLS = os.getenv("ENABLE_TYPED_TOOLS", "true").lower() in ("1", "true", "yes", "on")
ENABLE_AGENT_LOOP = os.getenv("ENABLE_AGENT_LOOP", "true").lower() in ("1", "true", "yes", "on")
ENABLE_QUESTION_ANALYZER_SHADOW = os.getenv("ENABLE_QUESTION_ANALYZER_SHADOW", "false").lower() in ("1", "true", "yes", "on")
ENABLE_QUESTION_ANALYZER_HINTS = os.getenv("ENABLE_QUESTION_ANALYZER_HINTS", "true").lower() in ("1", "true", "yes", "on")
ENABLE_TRACE_DEBUG_ARTIFACTS = os.getenv("ENABLE_TRACE_DEBUG_ARTIFACTS", "false").lower() in ("1", "true", "yes", "on")
ENABLE_SKILL_PROMPTS_SUMMARIZER = os.getenv("ENABLE_SKILL_PROMPTS_SUMMARIZER", "true").lower() in ("1", "true", "yes", "on")
ENABLE_SKILL_PROMPTS_PLANNER = os.getenv("ENABLE_SKILL_PROMPTS_PLANNER", "true").lower() in ("1", "true", "yes", "on")
ENABLE_VECTOR_KNOWLEDGE_SHADOW = os.getenv("ENABLE_VECTOR_KNOWLEDGE_SHADOW", "false").lower() in ("1", "true", "yes", "on")
ENABLE_VECTOR_KNOWLEDGE_HINTS = os.getenv("ENABLE_VECTOR_KNOWLEDGE_HINTS", "true").lower() in ("1", "true", "yes", "on")
TRACE_TEXT_MAX_CHARS = max(120, int(os.getenv("TRACE_TEXT_MAX_CHARS", "800")))
TRACE_MAX_LIST_ITEMS = max(1, int(os.getenv("TRACE_MAX_LIST_ITEMS", "8")))
AGENT_MAX_ROUNDS = max(1, int(os.getenv("AGENT_MAX_ROUNDS", "3")))
AGENT_TOOL_PREVIEW_ROWS = max(1, int(os.getenv("AGENT_TOOL_PREVIEW_ROWS", "10")))
AGENT_TOOL_PREVIEW_MAX_CHARS = max(200, int(os.getenv("AGENT_TOOL_PREVIEW_MAX_CHARS", "3000")))
AGENT_TOOL_TIMEOUT_SECONDS = max(1, int(os.getenv("AGENT_TOOL_TIMEOUT_SECONDS", "15")))
AGENT_TOOL_RETRY_ATTEMPTS = max(1, int(os.getenv("AGENT_TOOL_RETRY_ATTEMPTS", "2")))
PROMPT_BUDGET_MAX_CHARS = max(1500, int(os.getenv("PROMPT_BUDGET_MAX_CHARS", "45000")))
ROUTER_ENABLE_SEMANTIC_FALLBACK = os.getenv("ROUTER_ENABLE_SEMANTIC_FALLBACK", "true").lower() in ("1", "true", "yes", "on")
ROUTER_SEMANTIC_MIN_SCORE = min(
    1.0,
    max(0.1, float(os.getenv("ROUTER_SEMANTIC_MIN_SCORE", "0.55"))),
)
SESSION_HISTORY_MAX_TURNS = max(1, int(os.getenv("SESSION_HISTORY_MAX_TURNS", "3")))
SESSION_IDLE_TTL_SECONDS = max(60, int(os.getenv("SESSION_IDLE_TTL_SECONDS", "3600")))
ASK_MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("ASK_MAX_CONCURRENT_REQUESTS", "8")))
ASK_BACKPRESSURE_TIMEOUT_SECONDS = max(0.0, float(os.getenv("ASK_BACKPRESSURE_TIMEOUT_SECONDS", "0.0")))
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

# Memory Limits (PRODUCTION SAFETY: Prevents OOM errors)
MAX_RESULT_SIZE_MB = int(os.getenv("MAX_RESULT_SIZE_MB", "100"))

# ===================================================================
# Validation
# ===================================================================

if not SUPABASE_DB_URL:
    raise RuntimeError("Missing SUPABASE_DB_URL")
if not GATEWAY_SHARED_SECRET:
    raise RuntimeError("Missing ENAI_GATEWAY_SECRET (or legacy GATEWAY_SHARED_SECRET)")
if not SESSION_SIGNING_SECRET:
    raise RuntimeError("Missing ENAI_SESSION_SIGNING_SECRET (or legacy SESSION_SIGNING_SECRET)")
if not EVALUATE_ADMIN_SECRET:
    raise RuntimeError("Missing ENAI_EVALUATE_SECRET (or legacy EVALUATE_ADMIN_SECRET)")
if MODEL_TYPE == "gemini" and not GOOGLE_API_KEY:
    raise RuntimeError("MODEL_TYPE=gemini but GOOGLE_API_KEY is missing")

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
    "energy_balance_long_mv"
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

# Seasonal months
SUMMER_MONTHS = [4, 5, 6, 7]
WINTER_MONTHS = [1, 2, 3, 8, 9, 10, 11, 12]

# Balancing segment normalizer
BALANCING_SEGMENT_NORMALIZER = "LOWER(REPLACE(segment, ' ', '_'))"

# Balancing share pivot SQL template
BALANCING_SHARE_PIVOT_SQL = dedent(
    f"""
    SELECT
        t.date,
        'balancing'::text AS segment,
        SUM(t.quantity) AS total_quantity_debug,
        SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_import,
        SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_deregulated_hydro,
        SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_regulated_hpp,
        SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_regulated_new_tpp,
        SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_regulated_old_tpp,
        SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_renewable_ppa,
        SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_thermal_ppa,
        SUM(CASE WHEN t.entity IN ('renewable_ppa','thermal_ppa') THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_all_ppa,
        SUM(CASE WHEN t.entity IN ('deregulated_hydro','regulated_hpp','renewable_ppa') THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_all_renewables,
        SUM(CASE WHEN t.entity IN ('deregulated_hydro','regulated_hpp') THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity), 0) AS share_total_hpp
    FROM trade_derived_entities t
    WHERE {BALANCING_SEGMENT_NORMALIZER} = 'balancing'
      AND t.entity IN (
        'import', 'deregulated_hydro', 'regulated_hpp',
        'regulated_new_tpp', 'regulated_old_tpp',
        'renewable_ppa', 'thermal_ppa'
      )
    GROUP BY t.date
    ORDER BY t.date
    """
).strip()


