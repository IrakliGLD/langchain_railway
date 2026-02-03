# main.py v19.0 — Refactored with modular architecture
# Phase 5: Integrated with extracted modules
#
# REFACTORING STATUS:
# ✅ Phase 1-4 Complete: ~2,850 lines extracted to 12 modules
# ✅ All imports updated to use new modules
# 🔄 Note: Some duplicate function definitions remain in this file for safety
#          These are superseded by the imported modules and will be removed in Phase 5.2
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

import os
import re
import time
import logging
import uuid
import json
from contextvars import ContextVar
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
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
from domain_knowledge import DOMAIN_KNOWLEDGE
from sql_helpers import (
    detect_aggregation_intent,
    validate_aggregation_logic,
    enhance_sql_examples_for_aggregation,
    get_aggregation_guidance
)

# ============================================================================
# REFACTORED MODULES (Phases 1-4)
# ============================================================================
# Phase 1: Configuration and Models
from config import *  # All configuration constants
from models import Question, APIResponse, MetricsResponse

# Phase 2: Core modules
from utils.metrics import metrics
from utils.language import detect_language, get_language_instruction
from utils.query_validation import (
    is_conceptual_question,
    validate_sql_relevance,
    should_skip_sql_execution
)
from core.query_executor import ENGINE, execute_sql_safely
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

# Request ID tracking for observability
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# CORS Configuration: Parse allowed origins from environment
# Default to localhost for development if not specified
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]  # Clean up whitespace
log.info(f"🔒 CORS: Allowed origins: {ALLOWED_ORIGINS}")

# Note: Metrics, config variables, and other extracted code now imported from modules
# See imports section above for: config.py, models.py, utils.metrics, core.*, analysis.*, visualization.*


def should_inject_balancing_pivot(user_query: str, sql: str) -> bool:
    """
    Detect if query is asking for balancing share but SQL doesn't include share calculations.

    Returns True if:
    - User query mentions balancing-related concepts
    - User query mentions entity types
    - SQL uses trade_derived_entities directly (not through pivot)
    - SQL lacks share columns (no manual share calculation attempted)

    This forces pivot injection for queries like:
    - "what was the share of renewable PPA in balancing electricity?"
    - "show me the composition of balancing market in june 2024"

    Strategy: If SQL already has share calculations, let it run - the fixed
    Example 5 few-shot should teach correct entity filtering. Only inject pivot
    when LLM completely misses the share calculation.
    """
    query_lower = user_query.lower()
    sql_lower = sql.lower()

    balancing_keywords = ["balancing", "share", "composition", "mix", "weight", "proportion"]
    entity_keywords = ["ppa", "renewable", "thermal", "import", "hydro", "tpp", "hpp", "entity", "entities"]

    has_balancing = any(k in query_lower for k in balancing_keywords)
    has_entity = any(k in query_lower for k in entity_keywords)
    has_trade = "trade_derived_entities" in sql_lower
    has_share_col = any(f"share_{e}" in sql_lower for e in ["import", "renewable", "ppa", "hydro", "tpp", "hpp"])

    # Only inject pivot if query is about balancing shares but SQL doesn't have share columns
    # If share columns exist, trust that Example 5 taught correct entity filtering
    return has_balancing and has_entity and has_trade and not has_share_col


def build_trade_share_cte(original_sql: str) -> str:
    """Inject a balancing electricity share pivot as a CTE and alias original SQL to it.

    Uses a unique CTE name to avoid conflicts if original SQL already uses 'tde' as alias.
    """
    # Generate unique CTE name to avoid collisions (uuid already imported at top)
    cte_name = f"tde_{uuid.uuid4().hex[:6]}"

    table_pattern = re.compile(r"\btrade_derived_entities\b", re.IGNORECASE)
    pivot_alias_sql = table_pattern.sub(cte_name, original_sql)

    with_pattern = re.compile(r"^\s*WITH\b", re.IGNORECASE)
    if with_pattern.match(pivot_alias_sql):
        return with_pattern.sub(
            f"WITH {cte_name} AS ({BALANCING_SHARE_PIVOT_SQL}), ",
            pivot_alias_sql,
            count=1,
        )

    return f"WITH {cte_name} AS ({BALANCING_SHARE_PIVOT_SQL})\n{pivot_alias_sql}"


def fetch_balancing_share_panel(conn) -> pd.DataFrame:
    """Return a DataFrame with monthly balancing share ratios for each entity group.

    LIMIT added to prevent huge result sets when used as fallback.
    """
    # Add LIMIT to prevent returning all months (could be 100+ rows)
    limited_sql = f"{BALANCING_SHARE_PIVOT_SQL}\nLIMIT {MAX_ROWS}"

    res = conn.execute(text(limited_sql))
    rows = res.fetchall()
    cols = list(res.keys())
    df = pd.DataFrame(rows, columns=[str(c) for c in cols])
    return df


def ensure_share_dataframe(
    df: Optional[pd.DataFrame], conn
) -> tuple[pd.DataFrame, bool]:
    """Ensure we have a dataframe containing share_* columns for summarisation.

    Returns the dataframe to use plus a flag indicating whether the deterministic
    pivot fallback was executed.
    """

    if df is None:
        df = pd.DataFrame()

    has_share_cols = any(isinstance(c, str) and c.startswith("share_") for c in df.columns)
    if not df.empty and has_share_cols:
        return df, False

    fallback_df = fetch_balancing_share_panel(conn)
    if fallback_df.empty:
        log.warning(
            "Deterministic balancing share pivot returned 0 rows; check segment naming in trade_derived_entities."
        )

        # DIAGNOSTIC: Check what segment values actually exist
        try:
            diag_sql = """
            SELECT DISTINCT segment,
                   COUNT(*) as row_count,
                   MIN(date) as earliest_date,
                   MAX(date) as latest_date
            FROM trade_derived_entities
            GROUP BY segment
            ORDER BY segment
            """
            diag_res = conn.execute(text(diag_sql))
            diag_rows = diag_res.fetchall()
            log.warning(f"🔍 DIAGNOSTIC: Found {len(diag_rows)} distinct segment values in trade_derived_entities:")
            for row in diag_rows:
                log.warning(f"   - '{row[0]}': {row[1]} rows (from {row[2]} to {row[3]})")
        except Exception as e:
            log.error(f"Diagnostic query failed: {e}")

        return df, False

    return fallback_df, True


BALANCING_SHARE_METADATA: dict[str, dict[str, Any]] = {
    "share_regulated_hpp": {"label": "regulated HPP", "cost": "cheap", "usd_linked": False},
    "share_deregulated_hydro": {"label": "deregulated hydro", "cost": "cheap", "usd_linked": False},
    "share_total_hpp": {"label": "total HPP", "cost": "cheap", "usd_linked": False},
    "share_import": {"label": "imports", "cost": "expensive", "usd_linked": True},
    "share_regulated_new_tpp": {"label": "regulated new TPP", "cost": "expensive", "usd_linked": True},
    "share_regulated_old_tpp": {"label": "regulated old TPP", "cost": "expensive", "usd_linked": True},
    "share_renewable_ppa": {"label": "renewable PPAs", "cost": "expensive", "usd_linked": True},
    "share_thermal_ppa": {"label": "thermal PPAs", "cost": "expensive", "usd_linked": True},
    "share_all_ppa": {"label": "all PPAs", "cost": "expensive", "usd_linked": True},
    "share_all_renewables": {"label": "all renewables", "cost": "mixed", "usd_linked": True},
}


def build_share_shift_notes(
    cur_shares: dict[str, float],
    prev_shares: dict[str, float],
) -> list[str]:
    """Generate textual notes describing month-over-month share changes."""

    if not cur_shares:
        return []

    highlights: list[str] = []
    cheap_losses: list[str] = []
    expensive_gains: list[str] = []
    usd_gains: list[str] = []

    for key, meta in BALANCING_SHARE_METADATA.items():
        if key not in cur_shares or key not in prev_shares:
            continue
        cur_val = cur_shares[key]
        prev_val = prev_shares.get(key)
        if prev_val is None:
            continue
        delta = cur_val - prev_val
        if abs(delta) < 0.001:
            continue

        direction = "rose" if delta > 0 else "fell"
        highlights.append(
            f"{meta['label']} {direction} by {abs(delta) * 100:.1f} pp to {cur_val * 100:.1f}%"
        )

        if meta.get("cost") == "cheap" and delta < 0:
            cheap_losses.append(f"{meta['label']} ↓{abs(delta) * 100:.1f} pp")
        if meta.get("cost") == "expensive" and delta > 0:
            expensive_gains.append(f"{meta['label']} ↑{delta * 100:.1f} pp")
        if meta.get("usd_linked") and delta > 0:
            usd_gains.append(meta["label"])

    notes: list[str] = []
    if highlights:
        notes.append("Share shifts month-over-month: " + "; ".join(highlights) + ".")
    if cheap_losses:
        notes.append(
            "Cheaper balancing supply contracted: " + ", ".join(cheap_losses) + "."
        )
    if expensive_gains:
        notes.append(
            "Higher-cost groups expanded their weight: "
            + ", ".join(expensive_gains)
            + "."
        )
    if usd_gains:
        uniq_sources = sorted(set(usd_gains))
        notes.append(
            "USD-denominated sellers gained share ("
            + ", ".join(uniq_sources)
            + "), amplifying GEL price pressure."
        )

    return notes


MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def _parse_period_hint(period_hint: str, user_query: str) -> tuple[Optional[pd.Period], Optional[str]]:
    """Derive a pandas Period (monthly or yearly) from the LLM plan or the raw query."""

    period_hint = (period_hint or "").strip()
    if period_hint:
        normalized = period_hint.replace("/", "-").lower()
        if re.match(r"^(?:19|20)\d{2}-\d{2}$", normalized):
            try:
                per = pd.Period(normalized, freq="M")
                return per, per.to_timestamp().strftime("%B %Y")
            except Exception:
                pass
        if re.match(r"^(?:19|20)\d{2}$", normalized):
            try:
                per = pd.Period(normalized, freq="Y")
                return per, str(per.year)
            except Exception:
                pass

    text = user_query.lower()
    month_match = re.search(r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})",
                            text)
    if month_match:
        month_token, year_token = month_match.groups()
        # Dictionary already contains all variations (jan, january, sep, sept, september, etc.)
        month = MONTH_NAME_TO_NUMBER.get(month_token)
        if month:
            year = int(year_token)
            try:
                per = pd.Period(f"{year}-{month:02d}", freq="M")
                return per, per.to_timestamp().strftime("%B %Y")
            except Exception:
                pass

    year_match = re.search(r"(20\d{2})", text)
    if year_match:
        try:
            per = pd.Period(year_match.group(1), freq="Y")
            return per, str(per.year)
        except Exception:
            pass

    return None, None


def _select_share_column(share_cols: list[str], target_text: str) -> Optional[str]:
    """Choose the most relevant share column based on the user's target description."""

    target_text = target_text.lower()
    priority_map: list[tuple[str, tuple[str, ...]]] = [
        ("share_renewable_ppa", ("renewable", "ppa")),
        ("share_thermal_ppa", ("thermal", "ppa")),
        ("share_all_ppa", ("ppa",)),
        ("share_all_renewables", ("renewable",)),
        ("share_import", ("import",)),
        ("share_deregulated_hydro", ("deregulated", "hydro")),
        ("share_regulated_new_tpp", ("new", "tpp")),
        ("share_regulated_old_tpp", ("old", "tpp")),
        ("share_regulated_hpp", ("regulated", "hpp")),
        ("share_total_hpp", ("total", "hpp")),
    ]

    for col, keywords in priority_map:
        if col in share_cols and all(k in target_text for k in keywords):
            return col

    if share_cols:
        if len(share_cols) == 1:
            return share_cols[0]
        # Prefer the aggregate PPA share if no better match is found.
        if "share_all_ppa" in share_cols:
            return "share_all_ppa"
        return share_cols[0]

    return None


def generate_share_summary(df: pd.DataFrame, plan: Dict[str, Any], user_query: str) -> Optional[str]:
    """Produce a deterministic textual answer for share queries to avoid LLM hallucinations."""

    if df is None or df.empty:
        return None

    share_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("share_")]
    if not share_cols:
        return None

    working_df = df.copy()
    date_col = next((c for c in working_df.columns if isinstance(c, str) and "date" in c.lower()), None)

    target_period, period_label = _parse_period_hint(plan.get("period", ""), user_query)

    selected_row = None
    if date_col:
        working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
        working_df = working_df.dropna(subset=[date_col])
        if not working_df.empty:
            working_df = working_df.sort_values(date_col)
            if target_period is not None:
                if target_period.freqstr.startswith("M"):
                    match = working_df[working_df[date_col].dt.to_period("M") == target_period.asfreq("M")]
                else:
                    match = working_df[working_df[date_col].dt.year == target_period.year]
                if not match.empty:
                    selected_row = match.iloc[-1]
            if selected_row is None:
                selected_row = working_df.iloc[-1]
                if period_label is None:
                    ts = selected_row[date_col]
                    if isinstance(ts, pd.Timestamp):
                        period_label = ts.strftime("%B %Y")
                    elif isinstance(ts, pd.Period):
                        period_label = ts.to_timestamp().strftime("%B %Y")
                    else:
                        period_label = str(ts)
        else:
            return None
    else:
        if plan.get("period"):
            period_label = plan["period"]
        selected_row = working_df.iloc[-1]

    if selected_row is None:
        return None

    target_text = f"{plan.get('target', '')} {user_query}"
    share_col = _select_share_column(share_cols, target_text)
    if not share_col:
        return None

    raw_value = selected_row.get(share_col)
    try:
        share_value = float(raw_value)
    except (TypeError, ValueError):
        return None

    if pd.isna(share_value):
        return None

    # 🔍 VALIDATION: Check if shares sum to ~1.0 (indicates correct denominator)
    # Define base entities that constitute the whole (non-overlapping)
    BASE_SHARE_COLS = [
        "share_import",
        "share_deregulated_hydro",
        "share_regulated_hpp",
        "share_regulated_new_tpp",
        "share_regulated_old_tpp",
        "share_renewable_ppa",
        "share_thermal_ppa"
    ]

    # Only sum base entities to avoid double-counting aggregates like share_all_ppa
    total_shares = sum(
        float(selected_row.get(c, 0))
        for c in BASE_SHARE_COLS
        if c in selected_row and not pd.isna(selected_row.get(c))
    )
    if abs(total_shares - 1.0) > 0.05:
        log.warning(
            f"⚠️ Base share columns sum to {total_shares:.3f} instead of 1.0 — possible denominator bug. "
            f"Date: {selected_row.get(date_col) if date_col else 'N/A'}"
        )
        # Continue anyway, but warn that shares might be incorrect

    label = DERIVED_LABELS.get(share_col, share_col.replace("_", " ").title())

    if period_label is None:
        period_label = "the selected period"

    lines = [f"In {period_label}, {label} was {share_value * 100:.1f}% of balancing electricity."]

    if share_col == "share_all_ppa":
        breakdown_parts = []
        for extra_col, extra_label in (
            ("share_renewable_ppa", "renewable PPAs"),
            ("share_thermal_ppa", "thermal PPAs"),
        ):
            if extra_col in selected_row.index:
                extra_val = selected_row.get(extra_col)
                try:
                    extra_val_f = float(extra_val)
                except (TypeError, ValueError):
                    continue
                if pd.notna(extra_val_f):
                    breakdown_parts.append(f"{extra_label} {extra_val_f * 100:.1f}%")
        if breakdown_parts:
            lines.append("Breakdown: " + ", ".join(breakdown_parts) + ".")

    return " ".join(lines) if lines else None

# -----------------------------
# Database Schema Reflection
# -----------------------------
# Note: ENGINE is imported from core.query_executor
with ENGINE.connect() as conn:
    conn.execute(text("SELECT 1"))
    log.info("✅ Database connectivity verified")

    try:
        # Reflect only materialized views (exclude base tables)
        result = conn.execute(
            text("""
                SELECT m.matviewname AS view_name, a.attname AS column_name
                FROM pg_matviews m
                JOIN pg_attribute a ON m.matviewname::regclass = a.attrelid
                WHERE a.attnum > 0 AND NOT a.attisdropped
                AND m.schemaname = 'public';
            """)
        )
        rows = result.fetchall()

        # Build schema map for column-level validation
        SCHEMA_MAP = {}
        for v, c in rows:
            SCHEMA_MAP.setdefault(v.lower(), set()).add(c.lower())

        # Materialized views only
        ALLOWED_TABLES = set(SCHEMA_MAP.keys())

        log.info(f"🧩 Found materialized views: {sorted(ALLOWED_TABLES)}")
        log.info(f"📜 Final ALLOWED_TABLES (views only): {sorted(ALLOWED_TABLES)}")

        # Optional: show schema details for each view
        for view, cols in SCHEMA_MAP.items():
            log.info(f"📘 {view}: {sorted(cols)}")

    except Exception as e:
        log.warning(f"⚠️ Could not reflect materialized views: {e}")
        SCHEMA_MAP = {}
        # Fall back to the static allow-list so transient reflection issues
        # don't incorrectly block legitimate queries.
        ALLOWED_TABLES = set(STATIC_ALLOWED_TABLES)


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

# Phase 1D Security: Configure rate limiter (10 requests/minute per IP)
limiter = Limiter(key_func=get_remote_address)
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
    allow_headers=["Content-Type", "x-app-key", "Authorization"],  # Only needed headers
)

# -----------------------------
# Models: Imported from models.py (line 64)
# -----------------------------
# Question, APIResponse, MetricsResponse are imported at top of file

# -----------------------------
# LLM + Planning helpers
# -----------------------------

# ------------------------------------------------------------------
# STEP 1: Domain Knowledge Reasoning (NEW - ADD HERE)
# ------------------------------------------------------------------
def llm_analyze_with_domain_knowledge(user_query: str, lang_instruction: str) -> str:
    """
    First LLM call: Pure reasoning using domain knowledge.
    Forces the model to think like an energy analyst BEFORE writing SQL.
    """
    # Phase 1B Optimization: Check cache first
    cache_input = f"domain_reasoning|{user_query}|{lang_instruction}"
    cached_response = llm_cache.get(cache_input)
    if cached_response:
        log.info("Domain Reasoning: (cached)")
        return cached_response

    system = (
        "You are a senior energy market analyst for Georgia. "
        "Interpret the user's question using ONLY the domain knowledge. "
        "Do not write SQL. Do not mention tables. "
        "Answer in 3 parts: "
        "1. Intent: What is the user asking? (e.g., price trend, share, driver) "
        "2. Key Concepts: List domain concepts involved "
        "3. Reasoning: Explain using domain knowledge "
        f"{lang_instruction}"
    )
    domain_json = get_relevant_domain_knowledge(user_query, use_cache=False)
    prompt = f"""
User question: {user_query}

Domain Knowledge:
{domain_json}

Respond in structured text only.
"""
    try:
        llm = make_gemini()
        response = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        log.info(f"Domain Reasoning:\n{response}")

        # Phase 1B Optimization: Cache the response
        llm_cache.set(cache_input, response)

        return response
    except Exception as e:
        log.warning(f"Domain reasoning failed: {e}. Using fallback.")
        fallback = "Intent: general\nKey Concepts: balancing price\nReasoning: Use xrate and entity shares from trade_derived_entities."
        llm_cache.set(cache_input, fallback)  # Cache fallback too
        return fallback

# -----------------------------
# LLM Functions: Imported from core/llm.py (lines 72-80)
# -----------------------------
# make_gemini, make_openai, llm_cache, classify_query_type, get_query_focus,
# llm_generate_plan_and_sql, llm_summarize - all imported from core.llm

# Optimized: Use set for O(1) lookup instead of list
ANALYTICAL_KEYWORDS = {
    "trend", "change", "growth", "increase", "decrease", "compare", "impact",
    "volatility", "pattern", "season", "relationship", "correlation", "evolution",
    "driver", "cause", "effect", "factor", "reason", "influence", "depend", "why", "behind"
}

def detect_analysis_mode(user_query: str) -> str:
    """Detect if query requires analytical mode based on keywords.

    Optimized: Uses set for O(1) lookup, converts to lowercase once.
    """
    query_lower = user_query.lower()

    # Simple fact queries → light mode (higher priority)
    simple_patterns = [
        "what is", "what was", "list", "show", "give me",
        "რა არის", "რამდენი", "покажи", "что такое"
    ]
    if any(p in query_lower for p in simple_patterns):
        return "light"

    # Deep analysis keywords → analyst mode
    analyst_keywords = [
        "trend over time", "correlation", "driver", "impact on",
        "relationship between", "explain the dynamics", "analyze",
        "what drives", "what causes", "why does"
    ]
    if any(k in query_lower for k in analyst_keywords):
        return "analyst"

    # Fallback to old logic for other analytical keywords
    if any(kw in query_lower for kw in ANALYTICAL_KEYWORDS):
        return "analyst"

    return "light"


def get_relevant_domain_knowledge(user_query: str, use_cache: bool = True) -> str:
    """Return domain knowledge JSON, filtered by query focus to reduce token usage.

    Args:
        user_query: The user's query text
        use_cache: If True, use full cached JSON. If False, select relevant sections only.
    Returns:
        JSON string of domain knowledge (full or filtered)

    This function can reduce token usage by 50-70% when use_cache=False by including
    only sections relevant to the query focus area.
    """
    if use_cache:
        return _DOMAIN_KNOWLEDGE_JSON

    query_lower = user_query.lower()
    relevant = {}

    # ------------------------------------------------------------------
    # Keyword triggers – each key must exist in DOMAIN_KNOWLEDGE
    # ------------------------------------------------------------------
    triggers = {
        "BalancingPriceDrivers": [
            "balancing", "price", "p_bal", "cost", "driver", "why", "increase", "decrease",
            "xrate", "share", "composition", "hydro", "thermal", "import", "ppa"
        ],
        "BalancingPriceFormation": [
            "weighted", "average", "weighting", "entity", "segment", "balancing_electricity"
        ],
        "BalancingMarketStructure": [
            "balancing market", "imbalance", "settlement", "esco", "brp", "monthly"
        ],
        "BalancingPriceDecomposition": [
            "decomposition", "contribution", "share_change", "entity contribution"
        ],
        "BalancingMarketLogic": [
            "deviation", "forecast", "actual", "residual", "mix"
        ],
        "TariffStructure": [
            "tariff", "regulated", "enguri", "vardnili", "gardabani", "tpp", "cost-plus",
            "gnerc", "capacity fee"
        ],
        "TariffDependencies": [
            "enguri", "gardabani", "old tpp", "mtkvar", "g-power"
        ],
        "tariff_entities": [
            "engurhesi", "energo-pro", "dzevruli", "gumati", "shaori", "rioni",
            "lajanuri", "zhinvali", "khrami", "mtkvari energy", "tbilisi tpp"
        ],
        "price_with_usd": [
            "p_bal_gel", "p_bal_usd", "p_dereg_gel", "p_dereg_usd", "usd"
        ],
        "CurrencyInfluence": [
            "gel", "usd", "exchange rate", "depreciation", "xrate"
        ],
        "SeasonalityPatterns": [
            "summer", "winter", "april", "july", "august", "march", "season"
        ],
        "SeasonalTrends": [
            "seasonal", "cagr", "trend", "hydro dominant", "thermal dominant"
        ],
        "CfD_Contracts": [
            "cfd", "contract for difference", "strike price", "renewable ppa", "central dispatch"
        ],
        "RenewableIntegration": [
            "renewable", "ppa", "solar", "wind", "integration"
        ],
        "ImportDependence": [
            "import", "export", "dependence", "turkey", "azeri"
        ],
        "TransmissionNetworkDevelopment": [
            "transmission", "tyndp", "gse", "west-east", "congestion", "substation"
        ],
        "GenerationAdequacyAndForecast": [
            "adequacy", "forecast", "capacity", "plexos", "2034"
        ],
        "MarketParticipantsAndDataSources": [
            "gnerc", "esco", "gse", "genex", "geostat", "participant"
        ],
        "DataEvidenceIntegration": [
            "evidence", "view", "materialized", "chart", "cpi"
        ]
    }

    for section, keywords in triggers.items():
        if any(k in query_lower for k in keywords):
            if section in DOMAIN_KNOWLEDGE:
                relevant[section] = DOMAIN_KNOWLEDGE[section]

    # ------------------------------------------------------------------
    # FALLBACK: always include BalancingPriceDrivers (core for price questions)
    # ------------------------------------------------------------------
    if not relevant:
        relevant = {
            "note": "No specific domain knowledge matched the query.",
            "BalancingPriceDrivers": DOMAIN_KNOWLEDGE.get("BalancingPriceDrivers", {})
        }

    return json.dumps(relevant, indent=2)


    
    if use_cache:
        # Use full pre-cached JSON (fastest, but more tokens)
        return _DOMAIN_KNOWLEDGE_JSON

    # Selective approach: filter by query focus
    query_lower = user_query.lower()
    query_focus = get_query_focus(user_query)

    relevant = {}

    # Include domain knowledge based on query focus
    if query_focus == "balancing":
        # Balancing price queries get full balancing context
        relevant["BalancingPriceDrivers"] = DOMAIN_KNOWLEDGE.get("BalancingPriceDrivers", {})
        relevant["BalancingPriceFormation"] = DOMAIN_KNOWLEDGE.get("BalancingPriceFormation", {})
        relevant["CurrencyInfluence"] = DOMAIN_KNOWLEDGE.get("CurrencyInfluence", {})
        # May also need tariffs for correlation
        if any(word in query_lower for word in ["tariff", "correlation", "driver", "factor"]):
            relevant["TariffStructure"] = DOMAIN_KNOWLEDGE.get("TariffStructure", {})

    elif query_focus == "cpi":
        # CPI queries get CPI context only
        relevant["CPI"] = DOMAIN_KNOWLEDGE.get("CPI", {})
        # May mention prices peripherally
        if any(word in query_lower for word in ["price", "electricity", "ფასი"]):
            relevant["CurrencyInfluence"] = DOMAIN_KNOWLEDGE.get("CurrencyInfluence", {})

    elif query_focus == "tariff":
        # Tariff queries get tariff context
        relevant["TariffStructure"] = DOMAIN_KNOWLEDGE.get("TariffStructure", {})
        relevant["tariff_entities"] = DOMAIN_KNOWLEDGE.get("tariff_entities", {})
        # May ask about tariff impact on prices
        if any(word in query_lower for word in ["impact", "affect", "influence", "price"]):
            relevant["CurrencyInfluence"] = DOMAIN_KNOWLEDGE.get("CurrencyInfluence", {})

    elif query_focus == "generation":
        # Generation queries get generation/trade context
        relevant["trade"] = DOMAIN_KNOWLEDGE.get("trade", {})
        if any(word in query_lower for word in ["seasonal", "summer", "winter"]):
            relevant["SeasonalPattern"] = DOMAIN_KNOWLEDGE.get("SeasonalPattern", {})

    elif query_focus == "trade":
        # Trade/import/export queries
        relevant["TradePattern"] = DOMAIN_KNOWLEDGE.get("TradePattern", {})
        relevant["trade"] = DOMAIN_KNOWLEDGE.get("trade", {})

    else:
        # General queries or unclear focus - include minimal context
        # Check for specific keywords to add relevant sections
        if any(word in query_lower for word in ["balancing", "баланс", "საბალანსო"]):
            relevant["BalancingPriceDrivers"] = DOMAIN_KNOWLEDGE.get("BalancingPriceDrivers", {})

        if any(word in query_lower for word in ["tariff", "ტარიფი", "тариф"]):
            relevant["TariffStructure"] = DOMAIN_KNOWLEDGE.get("TariffStructure", {})

        if any(word in query_lower for word in ["season", "summer", "winter"]):
            relevant["SeasonalPattern"] = DOMAIN_KNOWLEDGE.get("SeasonalPattern", {})

        if any(word in query_lower for word in ["cpi", "inflation"]):
            relevant["CPI"] = DOMAIN_KNOWLEDGE.get("CPI", {})

    # If no relevant sections found, provide minimal context
    if not relevant:
        relevant["price_with_usd"] = DOMAIN_KNOWLEDGE.get("price_with_usd", {})

    log.info(f"📚 Domain knowledge filtered: focus={query_focus}, sections={list(relevant.keys())}")
    return json.dumps(relevant, indent=2)


# ------------------------------------------------------------------
# REMOVED: llm_plan_analysis - Combined into llm_generate_plan_and_sql
# ------------------------------------------------------------------

FEW_SHOT_SQL = """
-- Example 1: Monthly average balancing price (USD)
SELECT
  EXTRACT(YEAR FROM date) AS year,
  EXTRACT(MONTH FROM date) AS month,
  AVG(p_bal_usd) AS avg_balancing_usd
FROM price_with_usd
GROUP BY 1,2
ORDER BY 1,2
LIMIT 3750;

-- Example 2: Single-month balancing price (USD)
SELECT p_bal_usd
FROM price_with_usd
WHERE date = '2024-05-01'
LIMIT 3750;

-- Example 3: Generation (thousand MWh) by technology per month
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  type_tech,
  SUM(quantity_tech) AS qty_thousand_mwh
FROM tech_quantity_view
GROUP BY 1,2
ORDER BY 1,2
LIMIT 3750;

-- Example 4: CPI monthly values for electricity fuels
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  cpi
FROM monthly_cpi_mv
WHERE cpi_type = 'electricity_gas_and_other_fuels'
ORDER BY date
LIMIT 3750;

-- Example 5: Balancing price GEL vs shares (no raw quantities)
-- IMPORTANT: Use ILIKE or lowercase comparison for segment to handle different casings
-- Database may contain 'Balancing Electricity', 'balancing', or other variants
-- CRITICAL: Filter entities in denominator to only include relevant balancing entities
WITH shares AS (
  SELECT
    t.date,
    SUM(t.quantity) AS total_qty,
    SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) AS qty_import,
    SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) AS qty_dereg_hydro,
    SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) AS qty_reg_hpp
  FROM trade_derived_entities t
  WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing'
    AND t.entity IN ('import', 'deregulated_hydro', 'regulated_hpp',
                     'regulated_new_tpp', 'regulated_old_tpp',
                     'renewable_ppa', 'thermal_ppa')
  GROUP BY t.date
)
SELECT
  TO_CHAR(p.date, 'YYYY-MM') AS month,
  p.p_bal_gel,
  (s.qty_import / NULLIF(s.total_qty,0))      AS share_import,
  (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) AS share_deregulated_hydro,
  (s.qty_reg_hpp / NULLIF(s.total_qty,0))     AS share_regulated_hpp
FROM price_with_usd p
LEFT JOIN shares s ON s.date = p.date
ORDER BY p.date
LIMIT 3750;

-- Example 6: Balancing price (GEL, USD) + tariffs (Enguri, Gardabani, old TPPs) + xrate
WITH tariffs AS (
  SELECT
    d.date,
    (SELECT t1.tariff_gel FROM tariff_with_usd t1 WHERE t1.date = d.date AND t1.entity = 'ltd "engurhesi"1' LIMIT 1) AS enguri_tariff_gel,
    (SELECT t2.tariff_gel FROM tariff_with_usd t2 WHERE t2.date = d.date AND t2.entity = 'ltd "gardabni thermal power plant"' LIMIT 1) AS gardabani_tpp_tariff_gel,
    (SELECT AVG(t3.tariff_gel) FROM tariff_with_usd t3 WHERE t3.date = d.date AND t3.entity IN ('ltd "mtkvari energy"', 'ltd "iec" (tbilresi)', 'ltd "g power" (capital turbines)')) AS grouped_old_tpp_tariff_gel
  FROM price_with_usd d
)
SELECT
  p.date,
  p.p_bal_gel,
  p.p_bal_usd,
  p.xrate,
  tr.enguri_tariff_gel,
  tr.gardabani_tpp_tariff_gel,
  tr.grouped_old_tpp_tariff_gel
FROM price_with_usd p
LEFT JOIN tariffs tr ON tr.date = p.date
ORDER BY p.date
LIMIT 3750;

-- Example 7: Summer vs Winter averages
WITH seasons AS (
  SELECT date,
         CASE WHEN EXTRACT(MONTH FROM date) IN (4,5,6,7) THEN 'summer' ELSE 'winter' END AS season
  FROM price_with_usd
)
SELECT
  s.season,
  AVG(p.p_bal_gel) AS avg_bal_price_gel,
  AVG(tr.enguri_tariff_gel) AS avg_enguri_tariff_gel
FROM seasons s
JOIN price_with_usd p ON p.date = s.date
JOIN (
  SELECT date, tariff_gel AS enguri_tariff_gel
  FROM tariff_with_usd WHERE entity = 'ltd "engurhesi"1'
) tr ON tr.date = s.date
GROUP BY s.season
ORDER BY s.season
LIMIT 3750;

-- Example 8: Renewable PPA share in balancing electricity for specific month
-- CRITICAL: Always use LOWER(REPLACE(segment, ' ', '_')) for segment filtering
-- This example shows how to calculate share of a specific entity
WITH shares AS (
  SELECT
    date,
    SUM(quantity) AS total_qty,
    SUM(CASE WHEN entity = 'renewable_ppa' THEN quantity ELSE 0 END) AS qty_renewable_ppa,
    SUM(CASE WHEN entity = 'thermal_ppa' THEN quantity ELSE 0 END) AS qty_thermal_ppa,
    SUM(CASE WHEN entity = 'import' THEN quantity ELSE 0 END) AS qty_import
  FROM trade_derived_entities
  WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing'
  GROUP BY date
)
SELECT
  date,
  (qty_renewable_ppa / NULLIF(total_qty, 0)) AS share_renewable_ppa,
  (qty_thermal_ppa / NULLIF(total_qty, 0)) AS share_thermal_ppa,
  (qty_import / NULLIF(total_qty, 0)) AS share_import
FROM shares
WHERE date = '2024-06-01'
ORDER BY date
LIMIT 3750;

-- Example 9: Simple entity list (NO price context needed)
SELECT DISTINCT entity
FROM trade_derived_entities
ORDER BY entity
LIMIT 3750;

-- Example 10: Single tariff value query (NO balancing context needed)
SELECT tariff_gel
FROM tariff_with_usd
WHERE entity = 'ltd "engurhesi"1'
  AND date = '2024-06-01'
LIMIT 1;

-- Example 11: Generation by technology (NO price context needed)
SELECT
  type_tech,
  SUM(quantity_tech) AS total_generation_thousand_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
GROUP BY type_tech
ORDER BY total_generation_thousand_mwh DESC
LIMIT 3750;

-- Example 12: CPI trend (NO electricity price context needed)
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  cpi AS electricity_fuels_cpi
FROM monthly_cpi_mv
WHERE cpi_type = 'electricity_gas_and_other_fuels'
  AND date >= '2023-01-01'
ORDER BY date
LIMIT 3750;

-- Example 13: Tariff comparison (NO balancing price context needed)
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  entity,
  tariff_gel
FROM tariff_with_usd
WHERE entity IN ('ltd "engurhesi"1', 'ltd "gardabni thermal power plant"')
  AND date >= '2024-01-01'
ORDER BY date, entity
LIMIT 3750;

-- ============================================================================
-- AGGREGATION EXAMPLES (CRITICAL for Total vs Breakdown disambiguation)
-- ============================================================================

-- Example A1: TOTAL generation (single number, all technologies)
-- User: "What was total generation in 2023?"
-- Intent: Single total across ALL technologies
SELECT
  SUM(quantity_tech) * 1000 AS total_generation_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
  AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
LIMIT 3750;
-- IMPORTANT: NO GROUP BY - returns single row

-- Example A2: TOTAL generation BY TECHNOLOGY (breakdown)
-- User: "What was total generation by technology in 2023?"
-- Intent: Total for EACH technology
SELECT
  type_tech,
  SUM(quantity_tech) * 1000 AS total_generation_mwh
FROM tech_quantity_view
WHERE EXTRACT(YEAR FROM date) = 2023
  AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
GROUP BY type_tech
ORDER BY total_generation_mwh DESC
LIMIT 3750;
-- IMPORTANT: Has GROUP BY - returns multiple rows (one per technology)

-- Example A3: AVERAGE balancing price (single number)
-- User: "What was average balancing price in 2023?"
-- Intent: Single average across entire year
SELECT
  AVG(p_bal_gel) AS average_balancing_price_gel
FROM price_with_usd
WHERE EXTRACT(YEAR FROM date) = 2023
LIMIT 3750;
-- IMPORTANT: NO GROUP BY - returns single average

-- Example A4: SHARE calculation (percentage breakdown)
-- User: "What is share of each technology in total generation for 2023?"
-- Intent: Percentage contribution of each technology
WITH totals AS (
  SELECT
    type_tech,
    SUM(quantity_tech) AS tech_total
  FROM tech_quantity_view
  WHERE EXTRACT(YEAR FROM date) = 2023
    AND type_tech IN ('hydro', 'thermal', 'wind', 'solar')
  GROUP BY type_tech
),
grand_total AS (
  SELECT SUM(tech_total) AS overall_total FROM totals
)
SELECT
  t.type_tech,
  t.tech_total * 1000 AS generation_mwh,
  gt.overall_total * 1000 AS total_generation_mwh,
  ROUND((t.tech_total / gt.overall_total) * 100, 2) AS share_percent
FROM totals t, grand_total gt
ORDER BY share_percent DESC
LIMIT 3750;
-- IMPORTANT: Uses CTE to calculate shares properly
"""

# -----------------------------
# LLM SQL Generation: Imported from core.llm (line 76)
# -----------------------------
# llm_generate_plan_and_sql() is imported from core.llm

# -----------------------------
# Data helpers: Imported from analysis.stats (line 83)
# -----------------------------
# rows_to_preview() and quick_stats() are imported from analysis.stats
@app.get("/ask")
def ask_get():
    return {
        "message": "✅ /ask is active. Send POST with JSON: {'query': 'What was the average balancing price in 2023?'} and header X-App-Key."
    }

@app.get("/metrics")
def get_metrics():
    """Return application metrics for observability."""
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
            "pool_size": ENGINE.pool.size(),
            "checked_out": ENGINE.pool.checkedout(),
        }
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

    Authentication: Requires X-App-Key header
    """
    if x_app_key != APP_SECRET_KEY:
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

        # Define API function that calls /ask internally
        def call_api_internal(query_text: str):
            """Call the ask endpoint internally without HTTP overhead."""
            try:
                start = time.time()
                # Create Question object
                q = Question(query=query_text)
                # Call ask_post directly (bypassing HTTP)
                # Create a mock request object for rate limiting
                from starlette.datastructures import Headers
                mock_request = type('Request', (), {
                    'client': type('Client', (), {'host': '127.0.0.1'})(),
                    'headers': Headers({'x-app-key': APP_SECRET_KEY})
                })()

                response = ask_post(mock_request, q, APP_SECRET_KEY)
                elapsed_ms = (time.time() - start) * 1000

                # Convert response model to dict
                response_dict = {
                    "sql": response.sql,
                    "answer": response.answer,
                    "data": response.data
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
            from fastapi.responses import HTMLResponse
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


@app.post("/ask", response_model=APIResponse)
@limiter.limit("10/minute")  # Phase 1D Security: Rate limiting (10 requests/minute per IP)
def ask_post(request: Request, q: Question, x_app_key: str = Header(..., alias="X-App-Key")):
    t0 = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    mode = detect_analysis_mode(q.query)
    log.info(f"🧭 Selected mode: {mode}")

    # Detect query language for multilingual response
    lang_code = detect_language(q.query)
    lang_instruction = get_language_instruction(lang_code)
    log.info(f"🌍 Detected language: {lang_code}")

    # Phase 1C: Domain reasoning is now internal to SQL generation (merged into single LLM call)
    # This saves ~14s by eliminating one network round-trip

    plan = {}

    # ========================================================================
    # Option 4: Pre-check for Conceptual Questions (SAVE LLM CALL)
    # For purely conceptual questions, skip plan+SQL generation entirely
    # ========================================================================
    is_conceptual = is_conceptual_question(q.query)
    if is_conceptual:
        log.info(f"🎓 Conceptual question detected - skipping plan+SQL generation (saves LLM call)")
        log.info(f"📝 Query: {q.query}")

        # Answer directly using domain knowledge (no SQL needed)
        try:
            # Classify the type of conceptual question for better guidance
            query_lower = q.query.lower()

            # General energy terms that have definitions in GeneralDefinitions
            general_terms = [
                "renewable energy", "განახლებადი ენერგია", "возобновляемая энергия",
                "electricity market", "ელექტროენერგიის ბაზარი", "рынок электроэнергии",
                "balancing market", "საბალანსო ბაზარი", "балансирующий рынок",
                "tariff", "ტარიფი", "тариф",
                "ppa", "power purchase agreement",
                "cfd", "contract for difference",
                "hydropower", "ჰიდროენერგია", "гидроэнергия",
                "thermal power", "თერმული ენერგია", "тепловая энергия",
                "import", "export", "იმპორტი", "ექსპორტი",
                "demand", "მოთხოვნა", "generation mix", "გენერაციის სტრუქტურა",
                "capacity", "სიმძლავრე", "regulated", "deregulated",
                "exchange rate", "გაცვლითი კურსი", "обменный курс"
            ]

            # Domain-specific terms (Georgia market specifics)
            domain_terms = [
                "enguri", "vardnili", "gardabani", "gnerc", "esco", "gse",
                "ენგური", "ვარდნილი", "გარდაბანი",
                "საქართველო", "georgia"
            ]

            is_general_question = any(term in query_lower for term in general_terms)
            is_domain_specific = any(term in query_lower for term in domain_terms)

            # Determine the appropriate hint based on question type
            if is_general_question and not is_domain_specific:
                # Pure general question - provide definition + Georgia context
                conceptual_hint = (
                    "NOTE: This is a GENERAL conceptual/definitional question about energy terminology. "
                    "No database query was executed. "
                    "\n\n"
                    "RESPONSE FORMAT (MANDATORY):\n"
                    "1. **General Definition**: Start with a clear, universal definition of the concept "
                    "(2-3 sentences explaining what it is, how it works generally).\n"
                    "2. **Georgia Context**: Then provide Georgia-specific context showing how this concept "
                    "applies in the Georgian electricity market (2-3 sentences).\n"
                    "\n"
                    "Use the GeneralDefinitions section from domain knowledge if the term is defined there. "
                    "Structure your answer with these two clear sections."
                )
                log.info("📖 General conceptual question - will provide definition + Georgia context")
            elif is_domain_specific:
                # Domain-specific question - use domain knowledge directly
                conceptual_hint = (
                    "NOTE: This is a domain-specific conceptual question about the Georgian electricity market. "
                    "No database query was executed. "
                    "Answer using domain knowledge about Georgia's energy sector."
                )
                log.info("🇬🇪 Domain-specific conceptual question - will use Georgia domain knowledge")
            else:
                # Unknown/other conceptual question
                conceptual_hint = (
                    "NOTE: This is a conceptual/definitional question. "
                    "No database query was executed. "
                    "\n\n"
                    "If this topic is covered in domain knowledge, provide a clear explanation. "
                    "If NOT covered, acknowledge the limitation: "
                    "'This specific topic is not currently in my domain knowledge base. "
                    "For accurate information, I recommend consulting official sources.' "
                    "Then provide what general context you can."
                )
                log.info("❓ Conceptual question - topic may be outside domain scope")

            summary = llm_summarize(
                q.query,
                data_preview="",
                stats_hint=conceptual_hint,
                lang_instruction=lang_instruction,
                conversation_history=q.conversation_history
            )
            summary = scrub_schema_mentions(summary)

            exec_time = time.time() - t0
            metrics.log_request(exec_time)
            log.info(f"✅ Conceptual answer generated in {exec_time:.2f}s (1 LLM call saved)")

            return APIResponse(
                answer=summary,
                chart_data=None,
                chart_type=None,
                chart_metadata=None,
                execution_time=exec_time,
            )
        except Exception as e:
            log.exception("Conceptual answer generation failed")
            raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    # 1) Generate PLAN and SQL in ONE LLM call (now includes internal domain reasoning)
    try:
        combined_output = llm_generate_plan_and_sql(
            user_query=q.query,
            analysis_mode=mode,
            lang_instruction=lang_instruction
            # domain_reasoning parameter removed - now handled internally
        )

        # Validate LLM output format
        separator = "---SQL---"
        if separator not in combined_output:
            log.error(f"❌ LLM output missing separator. Output: {combined_output[:200]}")
            raise ValueError("LLM output malformed: missing '---SQL---' separator")

        plan_text, raw_sql = combined_output.split(separator, 1)
        raw_sql = raw_sql.strip()
        if not raw_sql:
            log.error("❌ LLM returned empty SQL after separator")
            raise ValueError("LLM output malformed: SQL part is empty")
            raw_sql = "SELECT 1"
        try:
            plan = json.loads(plan_text.strip())
        except json.JSONDecodeError:
            log.warning("Plan JSON decoding failed, defaulting to general plan.")
            plan = {"intent": "general", "target": "", "period": ""}
    except Exception as e:
        log.exception("Combined Plan/SQL generation failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate Plan/SQL: {e}")

    log.info(f"📝 Plan: {plan}")

    # ========================================================================
    # Option 3: Validate SQL Relevance
    # Check if query is conceptual or if SQL matches the question
    # ========================================================================
    skip_sql, skip_reason = should_skip_sql_execution(q.query, plan)
    sql_is_relevant = True
    skip_chart_due_to_relevance = False

    if skip_sql:
        log.info(f"⏭️ Skipping SQL execution: {skip_reason}")
        # For conceptual questions, answer using domain knowledge only
        # We'll handle this after the SQL block
    else:
        # Validate SQL relevance (will check after SQL is generated)
        pass

    # ========================================================================
    # CRITICAL: Aggregation Intent Detection & SQL Validation
    # Ensures LLM-generated SQL matches what user actually wants
    # ========================================================================
    aggregation_intent = detect_aggregation_intent(q.query)
    log.info(f"📊 Aggregation intent: {aggregation_intent}")

    # --- Period aggregation detection (optional user-defined range) ---
    period_pattern = re.search(
        r"(?P<start>(?:19|20)\d{2}[-/]?\d{0,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"[\s–\-to]+"
        r"(?P<end>(?:19|20)\d{2}[-/]?\d{0,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        q.query.lower()
    )

    safe_sql_final = None
    try:
        sanitized = raw_sql.strip()
        simple_table_whitelist_check(sanitized)
        safe_sql = plan_validate_repair(sanitized)

        # ========================================================================
        # CRITICAL: Validate SQL matches aggregation intent
        # ========================================================================
        is_valid_aggregation, validation_reason = validate_aggregation_logic(safe_sql, aggregation_intent)
        if not is_valid_aggregation:
            log.warning(f"⚠️ SQL doesn't match aggregation intent: {validation_reason}")
            log.warning(f"⚠️ User query: {q.query}")
            log.warning(f"⚠️ Generated SQL: {safe_sql[:200]}...")
            # Don't fail hard - log warning and continue
            # In future, could regenerate SQL with better guidance
        else:
            log.info(f"✅ SQL validation passed: {validation_reason}")

        # 🎯 CRITICAL: Force pivot injection for balancing share queries
        # LLM often generates SQL querying trade_derived_entities directly instead of using share columns
        # This proactively converts entity rows into share_* columns
        if should_inject_balancing_pivot(q.query, safe_sql):
            log.info("🔄 Force-injecting balancing share pivot based on query intent")
            safe_sql = build_trade_share_cte(safe_sql)

        if period_pattern:
            log.info("🧮 Detected user-defined period range → applying aggregation logic.")

            # detect whether query already includes GROUP BY or aggregation
            lower_sql = safe_sql.lower()
            has_agg = any(x in lower_sql for x in ["avg(", "sum(", "count(", "group by"])

            if has_agg:
                log.info("🧮 Query already aggregated → skipping outer AVG/SUM wrapper.")
                safe_sql_final = safe_sql
            else:
                # DISABLED: Period aggregation wrapper has a bug - assumes column named 'value'
                # TODO: Fix to detect actual column names from inner query
                log.warning("⚠️ Period aggregation requested but wrapper is disabled (column name detection needed)")
                safe_sql_final = safe_sql
        else:
            safe_sql_final = safe_sql

        # Validate SQL relevance after SQL is generated
        if not skip_sql:
            sql_is_relevant, relevance_reason, skip_chart_due_to_relevance = validate_sql_relevance(
                q.query, safe_sql_final, plan
            )

            if not sql_is_relevant:
                log.warning(f"⚠️ SQL relevance issue: {relevance_reason}")

            if skip_chart_due_to_relevance:
                log.info(f"📊 Chart will be skipped due to: {relevance_reason}")


    except Exception as e:
        log.warning(f"SQL validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Unsafe or invalid SQL: {e}")

    # 3) Execute SQL (or skip if conceptual question)
    if skip_sql:
        # For conceptual questions, skip SQL and use domain knowledge
        df = pd.DataFrame()
        rows = []
        cols = []
        log.info(f"⏭️ SQL execution skipped, will answer from domain knowledge")
    else:
        # Execute SQL query
        df = pd.DataFrame()
        rows = []
        cols = []
        try:
            sql_start = time.time()

            # DEBUG: Log the actual SQL being executed
            log.info(f"🔍 Executing SQL:\n{safe_sql_final}")

            # Phase 1D Security: Use secure execution wrapper
            df, cols, rows, elapsed = execute_sql_safely(safe_sql_final)
            metrics.log_sql_query(elapsed)

            try:
                from context import SUPPLY_TECH_TYPES, DEMAND_TECH_TYPES, TRANSIT_TECH_TYPES
            except ImportError:
                # Fallback values matching database schema
                SUPPLY_TECH_TYPES = ["hydro", "thermal", "wind", "solar", "import", "self-cons"]
                DEMAND_TECH_TYPES = ["abkhazeti", "supply-distribution", "direct customers", "losses", "export"]
                TRANSIT_TECH_TYPES = ["transit"]
                log.warning("Using fallback tech type classifications")

            if "type_tech" in df.columns:
                supply_df = df[df["type_tech"].isin(SUPPLY_TECH_TYPES)]
                demand_df = df[df["type_tech"].isin(DEMAND_TECH_TYPES)]
                transit_df = df[df["type_tech"].isin(TRANSIT_TECH_TYPES)]

                user_query_lower = q.query.lower()

                if any(w in user_query_lower for w in ["demand", "consumption", "loss", "export"]):
                    if not demand_df.empty:
                        df = demand_df.copy()
                        log.info(f"⚙️ Showing DEMAND side only: {DEMAND_TECH_TYPES}")
                    else:
                        log.info("⚠️ No DEMAND-side data found, using full dataset.")
                elif "transit" in user_query_lower:
                    if not transit_df.empty:
                        df = transit_df.copy()
                        log.info("⚙️ Showing TRANSIT data only.")
                    else:
                        log.info("⚠️ No TRANSIT data found, using full dataset.")
                else:
                    if not supply_df.empty:
                        df = supply_df.copy()
                        log.info(f"⚙️ Showing SUPPLY side only: {SUPPLY_TECH_TYPES}")
                    else:
                        log.info("⚠️ No SUPPLY-side data found, using full dataset.")



        except OperationalError as e:
            # Handle database connection and timeout errors
            metrics.log_error()
            log.error(f"⚠️ Database operational error: {e}")
            raise HTTPException(status_code=503, detail="Database temporarily unavailable. Please try again later.")

        except DatabaseError as e:
            # Handle SQL-specific errors (syntax, undefined columns, etc.)
            msg = str(e)
            metrics.log_error()

            # --- 🩹 Auto-pivot fix for hallucinated trade_derived_entities columns ---
            if "UndefinedColumn" in msg and "trade_derived_entities" in safe_sql_final:
                log.warning("🩹 Auto-pivoting trade_derived_entities: converting entity rows into share_* columns.")
                log.info("CRITICAL: Using segment='balancing' for share calculation")
                safe_sql_final = build_trade_share_cte(safe_sql_final)
                # Phase 1D Security: Use secure execution wrapper
                df, cols, rows, _ = execute_sql_safely(safe_sql_final)

            elif "UndefinedColumn" in msg:
                # Fallback synonym auto-fix (existing behavior)
                for bad, good in COLUMN_SYNONYMS.items():
                    if re.search(rf"\b{bad}\b", safe_sql_final, flags=re.IGNORECASE):
                        safe_sql_final = re.sub(rf"\b{bad}\b", good, safe_sql_final, flags=re.IGNORECASE)
                        log.warning(f"🔁 Auto-corrected column '{bad}' → '{good}' (retry)")
                        # Phase 1D Security: Use secure execution wrapper
                        df, cols, rows, _ = execute_sql_safely(safe_sql_final)
                        break
                else:
                    log.exception("SQL execution failed (UndefinedColumn)")
                    raise HTTPException(status_code=500, detail=f"Query failed: {e}")

            else:
                log.exception("SQL execution failed (DatabaseError)")
                raise HTTPException(status_code=500, detail=f"Query failed: {e}")

        except SQLAlchemyError as e:
            # Catch any other SQLAlchemy-related errors
            metrics.log_error()
            log.exception("SQLAlchemy error occurred")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

        except Exception as e:
            # Catch-all for unexpected errors
            metrics.log_error()
            log.exception("Unexpected error during SQL execution")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    # 4) Summarize and analyze
    share_intent = str(plan.get("intent", "")).lower()
    share_query_detected = share_intent in {"calculate_share", "share"} or "share" in q.query.lower()
    share_df_for_summary = df
    if share_query_detected:
        try:
            with ENGINE.connect() as conn:
                # Phase 1D Security: Enforce read-only mode
                conn.execute(text("SET TRANSACTION READ ONLY"))
                resolved_df, used_fallback = ensure_share_dataframe(df, conn)
            if used_fallback:
                log.warning("🔄 Share query lacked usable rows — using deterministic balancing share pivot.")
                df = resolved_df
                share_df_for_summary = resolved_df
                cols = list(resolved_df.columns)
                rows = [tuple(r) for r in resolved_df.itertuples(index=False, name=None)]
            else:
                share_df_for_summary = resolved_df
        except Exception as fallback_err:
            log.warning(f"Share pivot resolution failed: {fallback_err}")

    preview = rows_to_preview(rows, cols)
    stats_hint = quick_stats(rows, cols)

    # Option 3: Calculate seasonal-adjusted statistics for time series
    timeseries_info = detect_monthly_timeseries(df)
    if timeseries_info:
        time_col, value_col = timeseries_info
        try:
            seasonal_stats = calculate_seasonal_stats(df, time_col, value_col)
            seasonal_text = format_seasonal_stats(seasonal_stats)
            stats_hint += f"\n\n{seasonal_text}"
            log.info(f"✅ Added seasonal-adjusted statistics to stats_hint")
        except Exception as e:
            log.warning(f"⚠️ Seasonal stats calculation failed: {e}")

    share_summary_override = None
    if share_query_detected:
        try:
            share_summary_override = generate_share_summary(share_df_for_summary, plan, q.query)
            if share_summary_override:
                log.info("✅ Generated deterministic share summary override.")
        except Exception as share_err:
            log.warning(f"Share summary override failed: {share_err}")
    correlation_results = {}

    # --- Semantic correlation intent detection (v18.6 semantic mode) ---
    user_text = q.query.lower().strip()
    intent_text = str(plan.get("intent", "")).lower()
    combined_text = f"{intent_text} {user_text}"

    # --- Broader semantic triggers for cause/effect intent ---
    driver_keywords = [
        "driver", "cause", "effect", "factor", "reason", "impact", "influence",
        "relationship", "correlation", "depend", "why", "behind", "due to",
        "explain", "determinant", "driven by", "lead to", "affect", "because",
        "based on", "results in", "responsible for"
    ]

    # Semantic pattern detection (cause-effect phrases)
    causal_patterns = [
        r"what.*cause", r"what.*affect", r"why.*change", r"why.*increase",
        r"factors?.*behind", r"factors?.*influenc", r"reason.*for",
        r"cause.*of", r"impact.*on", r"driv.*price", r"lead.*to"
    ]

    text_hit = any(k in combined_text for k in driver_keywords)
    pattern_hit = any(re.search(p, combined_text) for p in causal_patterns)

    if text_hit or pattern_hit:
        log.info("🧮 Semantic intent → correlation (detected cause/effect phrasing).")
        plan["intent"] = "correlation"



    # --- Consolidated correlation analysis (overall + seasonal) ---
    if plan.get("intent") == "correlation":
        log.info("🔍 Building comprehensive balancing-price correlation analysis (overall + seasonal)")
        correlation_results = {}

        try:
            with ENGINE.connect() as conn:
                # Phase 1D Security: Enforce read-only mode
                conn.execute(text("SET TRANSACTION READ ONLY"))
                corr_df = build_balancing_correlation_df(conn)

            allowed_targets = ["p_bal_gel", "p_bal_usd"]
            allowed_drivers = [
                "xrate", "share_import", "share_deregulated_hydro",
                "share_regulated_hpp", "share_renewable_ppa",
                "enguri_tariff_gel", "gardabani_tpp_tariff_gel",
                "grouped_old_tpp_tariff_gel"
            ]

            # Filter to allowed columns
            corr_df = corr_df[[c for c in corr_df.columns if c in (["date"] + allowed_targets + allowed_drivers)]]

            # Overall correlations
            numeric_df = corr_df.drop(columns=["date"], errors="ignore").apply(pd.to_numeric, errors="coerce")
            for target in allowed_targets:
                if target not in numeric_df.columns:
                    continue
                series = numeric_df.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
                if series.notna().any():
                    correlation_results[target] = series.sort_values(ascending=False).round(3).to_dict()

            # Seasonal correlations (Summer vs Winter)
            if 'date' in corr_df.columns:
                corr_df['date'] = pd.to_datetime(corr_df['date'], errors='coerce')
                corr_df['month'] = corr_df['date'].dt.month
                summer_df = corr_df[corr_df['month'].isin([4, 5, 6, 7])].drop(columns=['date', 'month'], errors='ignore')
                winter_df = corr_df[~corr_df['month'].isin([4, 5, 6, 7])].drop(columns=['date', 'month'], errors='ignore')

                for label, seasonal_df in {'summer': summer_df, 'winter': winter_df}.items():
                    seasonal_numeric = seasonal_df.apply(pd.to_numeric, errors="coerce")
                    for target in allowed_targets:
                        if target in seasonal_numeric.columns and len(seasonal_numeric) > 2:
                            seasonal_corr = seasonal_numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
                            if seasonal_corr.notna().any():
                                correlation_results[f"{target}_{label}"] = seasonal_corr.sort_values(ascending=False).round(3).to_dict()

            if correlation_results:
                stats_hint = stats_hint + "\n\n--- CORRELATION MATRIX (vs Balancing Price) ---\n" + json.dumps(correlation_results, indent=2)
                log.info(f"✅ Consolidated correlations computed: {list(correlation_results.keys())}")
            else:
                log.info("⚠️ No valid correlations found")

        except Exception as e:
            log.warning(f"⚠️ Correlation analysis failed: {e}")

    # --- v18.8b: Forecasting (CAGR) + "Why?" reasoning (full combined block) ---

    def _detect_forecast_mode(text: str) -> bool:
        keys = ["forecast", "predict", "projection", "project", "future", "next year", "estimate", "estimation", "outlook"]
        t = text.lower()
        return any(k in t for k in keys)

    def _detect_why_mode(text: str) -> bool:
        keys = ["why", "reason", "cause", "factor", "explain", "due to", "behind", "what caused", "what influenced"]
        t = text.lower()
        return any(k in t for k in keys)

    def _month_from_text(s: str) -> int | None:
        months = {
            "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
            "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
        }
        for k,v in months.items():
            if k in s:
                return v
        return None

    def _choose_target_for_forecast(df_in: pd.DataFrame) -> tuple[str, str]:
        """Return (time_col, value_col) for forecasting."""
        cols = [c.lower() for c in df_in.columns]
        time_candidates = [c for c in df_in.columns if any(k in c.lower() for k in ["date", "year", "month"])]
        time_col = time_candidates[0] if time_candidates else None
        for c in df_in.columns:
            if c.lower() in ["p_bal_usd", "p_bal_gel"]:
                return time_col, c
        for c in df_in.columns:
            if any(k in c.lower() for k in ["price", "tariff", "p_bal"]):
                return time_col, c
        for c in df_in.columns:
            if any(k in c.lower() for k in ["quantity_tech", "quantity", "volume_tj", "generation", "demand"]):
                return time_col, c
        for c in df_in.columns:
            if pd.api.types.is_numeric_dtype(df_in[c]):
                return time_col, c
        return time_col, None

    def _detect_data_type(value_col: str) -> str:
        """Classify column into 'price', 'quantity', or 'other'."""
        c = value_col.lower()
        if any(k in c for k in ["p_bal", "price", "tariff"]):
            return "price"
        if any(k in c for k in ["quantity", "volume_tj", "demand", "generation"]):
            return "quantity"
        return "other"

    def _generate_cagr_forecast(df_in: pd.DataFrame, user_query: str) -> tuple[pd.DataFrame, str]:
        """
        Generate forecast:
        - For quantity/demand/generation: yearly totals → CAGR → extend years.
        - For balancing electricity price: yearly + seasonal (summer/winter) forecasts.
        """
        df = df_in.copy()
        time_col, value_col = _choose_target_for_forecast(df)
        if not time_col or not value_col:
            return df_in, "Forecast skipped: no clear time/value columns."

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col, value_col])
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col])
        if df.empty:
            return df_in, "Forecast skipped: no numeric data."

        data_type = _detect_data_type(value_col)
        note_parts = []

        if data_type == "quantity":
            df["year"] = df[time_col].dt.year
            df_y = df.groupby("year")[value_col].sum().reset_index()
            if len(df_y) < 2:
                return df_in, "Forecast skipped: insufficient yearly data."
            first, last = df_y.iloc[0], df_y.iloc[-1]
            span = last["year"] - first["year"]
            if span <= 0 or first[value_col] <= 0:
                return df_in, "Invalid data for CAGR."
            cagr = (last[value_col] / first[value_col]) ** (1 / span) - 1
            note_parts.append(f"Yearly CAGR={cagr*100:.2f}% ({int(first['year'])}→{int(last['year'])}).")
            yrs_in_q = re.findall(r"(20\d{2})", user_query)
            target_years = sorted({int(y) for y in yrs_in_q if int(y) > last["year"]}) or [last["year"] + i for i in range(1, 4)]
            f_rows = []
            for y in target_years:
                val = last[value_col] * ((1 + cagr) ** (y - last["year"]))
                f_rows.append({time_col: pd.to_datetime(f"{y}-01-01"), value_col: val, "is_forecast": True})
            if "is_forecast" not in df.columns:
                df["is_forecast"] = False
            df_f = pd.concat([df, pd.DataFrame(f_rows)], ignore_index=True)
            note_parts.append(f"Forecast years: {', '.join(map(str, target_years))}.")
            return df_f, " ".join(note_parts)

        elif data_type == "price":
            df["year"] = df[time_col].dt.year
            df["month"] = df[time_col].dt.month
            df["season"] = np.where(df["month"].isin([4,5,6,7]), "summer", "winter")

            df_y = df.groupby("year")[value_col].mean().reset_index()
            first, last = df_y.iloc[0], df_y.iloc[-1]
            span = last["year"] - first["year"]

            # Calculate yearly CAGR with validation
            if span > 0 and first[value_col] > 0 and last[value_col] > 0:
                cagr_y = (last[value_col]/first[value_col])**(1/span)-1
            else:
                cagr_y = 0
                if span > 0:
                    log.warning(f"Invalid CAGR calculation: first={first[value_col]}, last={last[value_col]}")

            df_s = df.groupby(["year","season"])[value_col].mean().reset_index()
            summer = df_s[df_s["season"]=="summer"]
            winter = df_s[df_s["season"]=="winter"]

            # Calculate seasonal CAGR with validation
            if len(summer) >= 2:
                s_first, s_last = summer[value_col].iloc[0], summer[value_col].iloc[-1]
                s_year_span = summer["year"].iloc[-1] - summer["year"].iloc[0]
                if s_year_span > 0 and s_first > 0 and s_last > 0:
                    cagr_s = (s_last / s_first)**(1/s_year_span) - 1
                else:
                    cagr_s = np.nan
            else:
                cagr_s = np.nan

            if len(winter) >= 2:
                w_first, w_last = winter[value_col].iloc[0], winter[value_col].iloc[-1]
                w_year_span = winter["year"].iloc[-1] - winter["year"].iloc[0]
                if w_year_span > 0 and w_first > 0 and w_last > 0:
                    cagr_w = (w_last / w_first)**(1/w_year_span) - 1
                else:
                    cagr_w = np.nan
            else:
                cagr_w = np.nan

            # Format CAGR values for display
            def format_cagr(cagr_val):
                return f"{cagr_val*100:.2f}" if not np.isnan(cagr_val) else "N/A"

            note_parts.append(f"Yearly CAGR={format_cagr(cagr_y)}%, Summer={format_cagr(cagr_s)}%, Winter={format_cagr(cagr_w)}%.")

            yrs_in_q = re.findall(r"(20\d{2})", user_query)
            target_years = sorted({int(y) for y in yrs_in_q if int(y) > last["year"]}) or [last["year"] + i for i in range(1, 4)]

            f_rows = []
            for y in target_years:
                val_y = last[value_col] * ((1 + cagr_y) ** (y - last["year"]))
                val_s = last[value_col] * ((1 + cagr_s) ** (y - last["year"])) if not np.isnan(cagr_s) else val_y
                val_w = last[value_col] * ((1 + cagr_w) ** (y - last["year"])) if not np.isnan(cagr_w) else val_y
                f_rows.append({time_col: pd.to_datetime(f"{y}-04-01"), "season": "summer", value_col: val_s, "is_forecast": True})
                f_rows.append({time_col: pd.to_datetime(f"{y}-12-01"), "season": "winter", value_col: val_w, "is_forecast": True})

            if "is_forecast" not in df.columns:
                df["is_forecast"] = False
            df_f = pd.concat([df, pd.DataFrame(f_rows)], ignore_index=True)
            note_parts.append(f"Forecast years: {', '.join(map(str, target_years))}.")
            return df_f, " ".join(note_parts)

        else:
            return df_in, "Forecast skipped: unrecognized data type."

    # 1️⃣ FORECAST MODE -------------------------------------------------
    if _detect_forecast_mode(q.query) and not df.empty:
        try:
            df, _forecast_note = _generate_cagr_forecast(df, q.query)
            stats_hint += f"\n\n--- FORECAST NOTE ---\n{_forecast_note}"
            log.info(_forecast_note)
        except Exception as _e:
            log.warning(f"Forecast generation failed: {_e}")

    # 2️⃣ WHY MODE ------------------------------------------------------
    if _detect_why_mode(q.query) and not df.empty:
        try:
            ctx = {"notes": [], "signals": {}}
            t_series_col = next((c for c in df.columns if any(k in c.lower() for k in ["date","year","month"])), None)
            if t_series_col:
                df[t_series_col] = pd.to_datetime(df[t_series_col], errors="coerce")
                df = df.dropna(subset=[t_series_col]).sort_values(t_series_col)

                years = [int(y) for y in re.findall(r"(20\d{2})", q.query)]
                mon = _month_from_text(q.query.lower())
                target_period = pd.Timestamp(years[0], mon or 1, 1) if years else df[t_series_col].iloc[-1]

                cur_row = df.loc[df[t_series_col] == target_period]
                if cur_row.empty:
                    cur_row = df[df[t_series_col] <= target_period].tail(1)

                # Check if we have valid current row data
                if cur_row.empty:
                    log.warning("No data found for target period in 'why' analysis")
                    # Skip 'why' analysis if no data available
                else:
                    prev_row = df[df[t_series_col] < cur_row[t_series_col].iloc[0]].tail(1)

                    def _get_val(row, cols):
                        """Extract first available numeric value from row columns."""
                        if row.empty:
                            return None
                        for c in cols:
                            if c in row.columns:
                                val = row[c].iloc[0] if len(row) > 0 else None
                                if val is not None and pd.notna(val):
                                    try:
                                        return float(val)
                                    except (ValueError, TypeError) as e:
                                        log.debug(f"Could not convert {c} to float: {e}")
                                        continue
                        return None

                    cur_gel = _get_val(cur_row, ["p_bal_gel"])
                    prev_gel = _get_val(prev_row, ["p_bal_gel"]) if not prev_row.empty else None
                    cur_usd = _get_val(cur_row, ["p_bal_usd"])
                    prev_usd = _get_val(prev_row, ["p_bal_usd"]) if not prev_row.empty else None
                    cur_xrate = _get_val(cur_row, ["xrate"])
                    prev_xrate = _get_val(prev_row, ["xrate"]) if not prev_row.empty else None

                    # Extract share columns safely; fall back to deterministic panel if missing
                    share_cols = [c for c in df.columns if c.startswith("share_")]
                    cur_shares: dict[str, float] = {}
                    prev_shares: dict[str, float] = {}

                    target_ts = None
                    if not cur_row.empty and t_series_col in cur_row.columns:
                        target_ts = pd.to_datetime(cur_row[t_series_col].iloc[0], errors="coerce")

                    def _populate_from_frame(frame: pd.DataFrame, dest: dict[str, float]) -> None:
                        if frame is None or frame.empty:
                            return
                        for col in share_cols:
                            if col in frame.columns and not frame[col].empty:
                                val = frame[col].iloc[0]
                                if pd.notna(val):
                                    try:
                                        dest[col] = float(val)
                                    except (ValueError, TypeError):
                                        continue

                    if share_cols:
                        _populate_from_frame(cur_row, cur_shares)
                        if not prev_row.empty:
                            _populate_from_frame(prev_row, prev_shares)
                    else:
                        share_panel = pd.DataFrame()
                        try:
                            with ENGINE.connect() as conn:
                                # Phase 1D Security: Enforce read-only mode
                                conn.execute(text("SET TRANSACTION READ ONLY"))
                                share_panel = fetch_balancing_share_panel(conn)
                        except Exception as share_err:
                            log.warning(f"Share panel lookup failed: {share_err}")
                        else:
                            if not share_panel.empty:
                                share_panel = share_panel.copy()
                                if "segment" in share_panel.columns:
                                    share_panel = share_panel[share_panel["segment"] == "balancing_electricity"]
                                share_panel["date"] = pd.to_datetime(share_panel["date"], errors="coerce")
                                share_panel = share_panel.dropna(subset=["date"]).sort_values("date")
                                share_cols = [c for c in share_panel.columns if c.startswith("share_")]

                                def _match_share_row(ts: Optional[pd.Timestamp]) -> pd.DataFrame:
                                    if ts is None or pd.isna(ts):
                                        return pd.DataFrame()
                                    ts = pd.to_datetime(ts)
                                    exact = share_panel[share_panel["date"] == ts]
                                    if not exact.empty:
                                        return exact.tail(1)
                                    monthly = share_panel[share_panel["date"].dt.to_period("M") == ts.to_period("M")]
                                    if not monthly.empty:
                                        return monthly.tail(1)
                                    earlier = share_panel[share_panel["date"] <= ts]
                                    if not earlier.empty:
                                        return earlier.tail(1)
                                    return pd.DataFrame()

                                share_cur = _match_share_row(target_ts)
                                if share_cur.empty and not share_panel.empty:
                                    share_cur = share_panel.tail(1)
                                if not share_cur.empty:
                                    for col in share_cols:
                                        val = share_cur[col].iloc[0]
                                        if pd.notna(val):
                                            try:
                                                cur_shares[col] = float(val)
                                            except (ValueError, TypeError):
                                                continue

                                    prev_cutoff = share_cur["date"].iloc[0]
                                    share_prev = share_panel[share_panel["date"] < prev_cutoff].tail(1)
                                    if share_prev.empty and target_ts is not None:
                                        share_prev = share_panel[share_panel["date"] < target_ts].tail(1)
                                    if not share_prev.empty:
                                        for col in share_cols:
                                            val = share_prev[col].iloc[0]
                                            if pd.notna(val):
                                                try:
                                                    prev_shares[col] = float(val)
                                                except (ValueError, TypeError):
                                                    continue

                    deltas = {k: round(cur_shares.get(k, 0) - prev_shares.get(k, 0), 4) for k in cur_shares}

                    ctx["signals"] = {
                        "period": str(cur_row[t_series_col].iloc[0]) if not cur_row.empty else None,
                        "p_bal_gel": {"cur": cur_gel, "prev": prev_gel},
                        "p_bal_usd": {"cur": cur_usd, "prev": prev_usd},
                        "xrate": {"cur": cur_xrate, "prev": prev_xrate},
                        "share_deltas": deltas,
                    }

                    if cur_shares:
                        ctx["signals"]["share_snapshot"] = {
                            k: round(v, 4) for k, v in cur_shares.items()
                        }
                    if prev_shares:
                        ctx["signals"]["share_prev_snapshot"] = {
                            k: round(v, 4) for k, v in prev_shares.items()
                        }

                    if cur_shares:
                        sorted_mix = sorted(cur_shares.items(), key=lambda kv: kv[1], reverse=True)
                        mix_parts = []
                        for key, value in sorted_mix[:5]:
                            label = BALANCING_SHARE_METADATA.get(key, {}).get(
                                "label", key.replace("_", " ")
                            )
                            mix_parts.append(f"{label} {value * 100:.1f}%")
                        if mix_parts:
                            ctx["notes"].append(
                                "Current balancing mix composition: " + ", ".join(mix_parts) + "."
                            )

                    share_notes = build_share_shift_notes(cur_shares, prev_shares)
                    for note in share_notes:
                        ctx["notes"].append(note)

            dk = DOMAIN_KNOWLEDGE
            ctx["notes"].append("Balancing price is a weighted average of electricity sold as balancing energy.")
            ctx["notes"].extend(dk.get("price_with_usd", {}).get("dependencies", []))
            ctx["notes"].append(dk.get("CurrencyInfluence", {}).get("GEL_USD_Effect", ""))
            ctx["notes"].extend(dk.get("CurrencyInfluence", {}).get("USD_Denominated_Costs", []))
            ctx["notes"].append("If GEL depreciates, GEL-denominated balancing price rises due to USD-linked gas/import costs.")
            ctx["notes"].append("Composition shift toward thermal or import increases price; more hydro or renewable lowers it.")
            stats_hint += "\n\n--- CAUSAL CONTEXT ---\n" + json.dumps(ctx, default=str, indent=2)
            log.info("Why-context attached to stats_hint.")
        except Exception as _e:
            log.warning(f"'Why' reasoning context build failed: {_e}")

    # --- 📈 Detect trend/forecast requests and pre-calculate trendlines for answer generation ---
    trend_keywords = [
        "trend", "ტრენდი", "тренд", "trending", "ტრენდინგი",
        "forecast", "პროგნოზი", "прогноз", "projection", "პროექცია", "проекция",
        "predict", "პროგნოზირება", "предсказать", "future", "მომავალი", "будущее",
        "continue", "გაგრძელება", "продолжить", "extrapolate", "ექსტრაპოლაცია"
    ]
    add_trendlines = any(keyword in q.query.lower() for keyword in trend_keywords)

    # Extract future year from query (e.g., "2030", "2035")
    trendline_extend_to = None
    if add_trendlines:
        # Look for 4-digit years in the query
        year_matches = re.findall(r'\b(20[2-9][0-9])\b', q.query)
        if year_matches:
            # Get the latest year mentioned
            future_year = max(int(year) for year in year_matches)
            # Extend to December of that year
            trendline_extend_to = f"{future_year}-12-01"
            log.info(f"📈 Trendline requested: extending to {trendline_extend_to}")
        else:
            # Default: extend 2 years into future
            from datetime import datetime
            current_year = datetime.now().year
            trendline_extend_to = f"{current_year + 2}-12-01"
            log.info(f"📈 Trendline requested: extending to {trendline_extend_to} (default 2 years)")

        # Calculate trendlines NOW for forecast answer generation
        if trendline_extend_to:
            try:
                from visualization.chart_builder import calculate_trendline

                # Detect time column
                time_key = next((c for c in cols if any(k in c.lower() for k in ["date", "year", "month", "თვე", "წელი", "თარიღი"])), None)

                # Detect if there's a season column for seasonal forecasts
                season_col = next((c for c in cols if c.lower() in ["season", "სეზონი"]), None)

                # Fix year-only columns: Convert integer/decimal years to datetime
                if time_key and time_key in df.columns:
                    try:
                        # Check if the time column contains year-only values (integers or decimals like 2014, 2015)
                        first_val = df[time_key].iloc[0]
                        # Check if it's a numeric type (int, Decimal) and looks like a year (1900-2100)
                        if isinstance(first_val, (int, float)) or str(type(first_val).__name__) == 'Decimal':
                            first_val_num = float(first_val)
                            if 1900 <= first_val_num <= 2100:
                                # Convert year integers to datetime (YYYY -> YYYY-01-01)
                                df[time_key] = pd.to_datetime(df[time_key].astype(int), format='%Y')
                                log.info(f"📅 Converted year-only column '{time_key}' to datetime format")
                    except Exception as e:
                        log.warning(f"Year column conversion check failed: {e}")

                # Get numeric columns (exclude time and season columns)
                num_cols = [c for c in cols if c != time_key and c != season_col]

                # Ensure columns are numeric for trendline calculation
                df_calc = df.copy()
                for c in num_cols:
                    try:
                        df_calc[c] = pd.to_numeric(df_calc[c], errors='coerce')
                    except Exception:
                        pass

                if time_key and time_key in df_calc.columns and num_cols:
                    log.info(f"📈 Pre-calculating trendlines for forecast answer generation")

                    trendline_forecasts = {}

                    # Check if this is a seasonal forecast query
                    if season_col and season_col in df_calc.columns:
                        log.info(f"📈 Seasonal forecast detected - calculating separate trendlines for each season")
                        seasons = df_calc[season_col].dropna().unique()

                        for season in seasons:
                            # Filter data for this season
                            season_df = df_calc[df_calc[season_col] == season].copy()

                            for col in num_cols:
                                trendline_data = calculate_trendline(
                                    season_df, time_key, col, extend_to_date=trendline_extend_to
                                )
                                if trendline_data:
                                    forecast_dates = trendline_data["dates"]
                                    forecast_values = trendline_data["values"]

                                    if forecast_dates and forecast_values:
                                        forecast_value = forecast_values[-1]
                                        # Use season-specific key
                                        forecast_key = f"{col}_{season}"
                                        trendline_forecasts[forecast_key] = {
                                            "target_date": forecast_dates[-1],
                                            "forecast_value": round(forecast_value, 2),
                                            "equation": trendline_data["equation"],
                                            "r_squared": round(trendline_data["r_squared"], 3),
                                            "season": season
                                        }
                                        log.info(f"  ✅ {col} ({season}): Forecast for {forecast_dates[-1]} = {forecast_value:.2f}, R²={trendline_data['r_squared']:.3f}")
                    else:
                        # Standard (non-seasonal) forecast
                        for col in num_cols:
                            trendline_data = calculate_trendline(
                                df_calc, time_key, col, extend_to_date=trendline_extend_to
                            )
                            if trendline_data:
                                # Extract forecast value for the target year
                                forecast_dates = trendline_data["dates"]
                                forecast_values = trendline_data["values"]

                                # Get the last forecast value (for the target year)
                                if forecast_dates and forecast_values:
                                    forecast_value = forecast_values[-1]
                                    trendline_forecasts[col] = {
                                        "target_date": forecast_dates[-1],
                                        "forecast_value": round(forecast_value, 2),
                                        "equation": trendline_data["equation"],
                                        "r_squared": round(trendline_data["r_squared"], 3)
                                    }
                                    log.info(f"  ✅ {col}: Forecast for {forecast_dates[-1]} = {forecast_value:.2f}, R²={trendline_data['r_squared']:.3f}")

                    # Add forecast information to stats_hint
                    if trendline_forecasts:
                        forecast_summary = f"\n\n--- TRENDLINE FORECASTS (Linear Regression) ---\n"
                        forecast_summary += f"Target date: {trendline_extend_to}\n"
                        for col, forecast_info in trendline_forecasts.items():
                            if "season" in forecast_info:
                                # Seasonal forecast
                                forecast_summary += f"\n{col.replace('_' + forecast_info['season'], '')} ({forecast_info['season']}):\n"
                            else:
                                # Standard forecast
                                forecast_summary += f"\n{col}:\n"
                            forecast_summary += f"  - Forecast value: {forecast_info['forecast_value']}\n"
                            forecast_summary += f"  - Equation: {forecast_info['equation']}\n"
                            forecast_summary += f"  - R² (goodness of fit): {forecast_info['r_squared']}\n"

                        stats_hint = stats_hint + forecast_summary
                        log.info(f"📊 Added {len(trendline_forecasts)} forecast values to stats_hint")
            except Exception as e:
                log.warning(f"Trendline pre-calculation failed: {e}")
    else:
        log.info("📈 No trendline keywords detected")

    if share_summary_override:
        summary = share_summary_override
    else:
        try:
            summary = llm_summarize(q.query, preview, stats_hint, lang_instruction, conversation_history=q.conversation_history)
        except Exception as e:
            log.warning(f"Summarization failed: {e}")
            summary = preview
    summary = scrub_schema_mentions(summary)
    #if mode == "analyst" and plan.get("intent") != "general":
        #summary = f"**Analysis type: {plan.get('intent')}**\n\n" + summary

    # 5) Chart builder (FINAL: labels from context + unit-only axis + robust numeric coercion)
    chart_data = chart_type = chart_meta = None
    if rows and cols:
        df = df.copy()

        # --- Coerce numeric for all non-time columns so JSON values are numbers, not strings ---
        # --- Detect time column ---
        # Support Georgian column names: თვე (month), წელი (year), თარიღი (date)
        time_key = next((c for c in cols if any(k in c.lower() for k in ["date", "year", "month", "თვე", "წელი", "თარიღი"])), None)

        # --- Fix year-only columns: Convert integer/decimal years to datetime ---
        if time_key and time_key in df.columns:
            try:
                # Check if the time column contains year-only values (integers or decimals like 2014, 2015)
                first_val = df[time_key].iloc[0]
                # Check if it's a numeric type (int, Decimal) and looks like a year (1900-2100)
                if isinstance(first_val, (int, float)) or str(type(first_val).__name__) == 'Decimal':
                    first_val_num = float(first_val)
                    if 1900 <= first_val_num <= 2100:
                        # Convert year integers to datetime (YYYY -> YYYY-01-01)
                        df[time_key] = pd.to_datetime(df[time_key].astype(int), format='%Y')
                        log.info(f"📅 Converted year-only column '{time_key}' to datetime format for chart building")
            except Exception as e:
                log.warning(f"Year column conversion check failed in chart builder: {e}")

        # --- Detect and preserve categorical columns ---
        # Support Georgian column names: ტიპი (type), სექტორი (sector), etc.
        categorical_hints = [
            "type", "tech", "entity", "sector", "source", "segment",
            "region", "category", "ownership", "market", "trade", "fuel",
            "ტიპი", "სექტორი", "წყარო"
        ]
        for c in cols:
            if c != time_key:
                if any(h in c.lower() for h in categorical_hints):
                    df[c] = df[c].astype(str).replace("nan", None)
                else:
                    try:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    except Exception:
                        pass

        # --- Auto-detect all categorical columns (non-numeric, non-time) ---
        categorical_cols = [
            c for c in df.columns
            if c != time_key and not pd.api.types.is_numeric_dtype(df[c])
        ]

        # --- Apply human-readable labels from context ---
        try:
            from context import COLUMN_LABELS
        except ImportError:
            COLUMN_LABELS = {}

        label_map_all = {c: COLUMN_LABELS.get(c, c.replace("_", " ").title()) for c in cols if c != time_key}

        # --- Automatically rename categorical columns to readable names ---
        for c in categorical_cols:
            new_name = label_map_all.get(c, c.replace("_", " ").title())
            if new_name != c:
                df.rename(columns={c: new_name}, inplace=True)

        # --- Optional: reorder columns: [time, categories..., values...] ---
        ordered_cols = []
        if time_key:
            ordered_cols.append(time_key)
        ordered_cols += categorical_cols
        for c in df.columns:
            if c not in ordered_cols:
                ordered_cols.append(c)
        ordered_cols = [c for c in ordered_cols if c in df.columns]
        df = df[ordered_cols]



        
        # --- Numeric columns after coercion ---
        num_cols = [
            c for c in df.columns
            if c != time_key
            and pd.api.types.is_numeric_dtype(df[c])
            and not re.search(r"\b(month|year)\b", c.lower())
        ]

        # --- 🧭 Decide whether to generate chart at all (context-aware) ---
        # Use improved chart necessity detection
        generate_chart = should_generate_chart(q.query, len(df))

        # Additional refinements based on query type and intent
        intent = str(plan.get("intent", "")).lower()
        query_text = q.query.lower()

        # Check if plan includes chart strategy (Option 1)
        chart_strategy = plan.get("chart_strategy", "single")
        chart_groups = plan.get("chart_groups", [])
        if chart_strategy or chart_groups:
            log.info(f"📋 Plan includes chart strategy: {chart_strategy}")
            if chart_groups:
                log.info(f"📋 Chart groups specified: {len(chart_groups)} group(s)")
                for i, group in enumerate(chart_groups, 1):
                    log.info(f"   Group {i}: {group.get('type', 'unknown')} - {group.get('metrics', [])} - {group.get('title', 'Untitled')}")
            if chart_strategy == "multiple" and len(chart_groups) > 1:
                log.warning(f"⚠️ LLM requested multiple charts ({len(chart_groups)}), but user wants only 1 chart")
                log.info(f"✅ Using first chart group: {chart_groups[0].get('title', 'Chart 1')}")

        # Option B: Filter num_cols to first chart group metrics (prevent messy multi-dimension charts)
        if chart_groups and len(chart_groups) > 0:
            first_group = chart_groups[0]
            chart_metrics = first_group.get('metrics', [])

            if chart_metrics:
                # Filter num_cols to only include metrics specified by LLM for first chart
                original_num_cols = num_cols.copy()
                num_cols = [col for col in num_cols if col in chart_metrics]

                if num_cols:
                    log.info(f"📊 Filtered chart metrics (Option B): {len(original_num_cols)} → {len(num_cols)} columns")
                    log.info(f"📊 Showing only: {num_cols}")
                    log.info(f"📊 Chart type: {first_group.get('type', 'auto')} | Title: {first_group.get('title', 'Untitled')}")

                    # CRITICAL: Also filter the DataFrame to only include filtered columns + time_key
                    # This ensures chart_data sent to frontend only has the filtered metrics
                    cols_to_keep = [time_key] + num_cols if time_key and time_key in df.columns else num_cols
                    cols_to_keep = [c for c in cols_to_keep if c in df.columns]  # Safety check
                    df = df[cols_to_keep]
                    log.info(f"📊 DataFrame filtered to {len(df.columns)} columns: {list(df.columns)}")
                else:
                    # Fallback: if no matching columns, keep original (safety net)
                    log.warning(f"⚠️ No matching columns found for chart metrics {chart_metrics}, using all numeric columns")
                    num_cols = original_num_cols
            else:
                log.info(f"📊 No metrics specified in chart group, using all numeric columns ({len(num_cols)})")
        else:
            log.info(f"📊 No chart groups in plan, using all numeric columns ({len(num_cols)})")

        # Override: Disable chart for purely explanatory questions
        if any(word in query_text for word in ["why", "how", "reason", "explain", "because", "cause", "რატომ", "როგორ", "почему"]):
            if len(df) < 5:  # Only skip if result is small
                generate_chart = False

        # Override: Disable chart for definition queries
        if any(word in query_text for word in ["define", "meaning of", "განმარტება"]):
            generate_chart = False

        # Override: Disable chart if SQL was deemed irrelevant to query (Option 3)
        if skip_chart_due_to_relevance:
            generate_chart = False
            log.info(f"🧭 Skipping chart generation: SQL query not relevant to user question")

        # NOTE: Removed forced chart generation for analyst mode
        # Let should_generate_chart decide based on query type classification
        # Analyst mode influences answer depth, not chart generation

        if not generate_chart:
            log.info(f"🧭 Skipping chart generation (query type or data not suitable for visualization, rows={len(df)}).")
            chart_data = chart_type = chart_meta = None
            # Jump to Final response (bypass chart drawing)
            exec_time = time.time() - t0
            metrics.log_request(exec_time)
            log.info(f"Finished request in {exec_time:.2f}s")
            return APIResponse(
                answer=summary,
                chart_data=None,
                chart_type=None,
                chart_metadata=None,
                execution_time=exec_time,
            )
        else:
            log.info("🎨 Proceeding with chart generation.")

        # Note: add_trendlines and trendline_extend_to are already defined earlier (before llm_summarize)

        # --- 🧠 Semantic-aware chart-type selection ---
        # STEP 1: Detect structural features (time, categories, values)
        # Support Georgian column names in addition to English
        cols_lower = [c.lower() for c in df.columns]
        time_cols = [c for c in df.columns if re.search(r"(year|month|date|წელი|თვე|თარიღი)", c.lower())]
        category_cols = [c for c in df.columns if re.search(r"(type|sector|entity|source|segment|ownership|technology|region|area|category|ტიპი|სექტორი)", c.lower())]
        value_cols = [c for c in df.columns if re.search(r"(quantity|volume|value|amount|price|tariff|cpi|index|mwh|tj|usd|gel|რაოდენობა|მოცულობა|ფასი|ტარიფი|საშუალო|სულ)", c.lower())]

        # STEP 2: Infer dimensions from numeric columns (semantic meaning)
        # IMPORTANT: Check in order of specificity (xrate before price, share before other)
        def infer_dimension(col: str) -> str:
            col_l = col.lower()
            # Exchange rate - check FIRST before price (has _gel/_usd but is not a price)
            if any(x in col_l for x in ["xrate", "exchange", "rate", "კურსი"]):
                return "xrate"
            # Shares/proportions - check BEFORE other
            if any(x in col_l for x in ["share_", "წილი_", "proportion", "percent", "პროცენტ"]):
                return "share"
            # Index
            if any(x in col_l for x in ["cpi", "index", "inflation", "ინდექსი"]):
                return "index"
            # Quantity
            if any(x in col_l for x in ["quantity", "generation", "volume_tj", "volume", "mw", "tj", "რაოდენობა", "მოცულობა", "გენერაცია"]):
                return "energy_qty"
            # Price/Tariff - check AFTER xrate
            if any(x in col_l for x in ["price", "tariff", "_gel", "_usd", "p_bal", "p_dereg", "p_gcap", "ფასი", "ტარიფი"]):
                return "price_tariff"
            return "other"

        dim_map = {c: infer_dimension(c) for c in num_cols}
        dims = set(dim_map.values())
        log.info(f"📐 Detected dimensions: {dim_map} → {dims}")

        # STEP 3: Choose chart type based on STRUCTURE + SEMANTICS
        # Philosophy: LLM = "what to show" (intent), Rules = "how to show it" (viz encoding)
        chart_type = "line"  # default fallback

        has_time = len(time_cols) >= 1
        has_categories = len(category_cols) >= 1
        primary_dimension = list(dims)[0] if len(dims) == 1 else "mixed"

        # DECISION MATRIX: Structure + Dimension Semantics → Chart Type

        if has_time and has_categories:
            # Time series with categories: decision depends on dimension
            if "share" in dims:
                # Shares over time → stacked bar (part-to-whole composition)
                chart_type = "stackedbar"
                log.info(f"📊 Chart type: stackedbar (time + categories + share = composition over time)")
            elif any(d in dims for d in ["price_tariff", "energy_qty", "index", "xrate"]):
                # Prices, quantities, indices, exchange rate over time → line (trend comparison)
                chart_type = "line"
                log.info(f"📊 Chart type: line (time + categories + {dims} = trend comparison)")
            else:
                # Mixed or unknown dimensions → default to line for time series
                chart_type = "line"
                log.info(f"📊 Chart type: line (time + categories + mixed/unknown dimensions)")

        elif has_time and not has_categories:
            # Single time series → always line
            chart_type = "line"
            log.info(f"📊 Chart type: line (time series without categories)")

        elif not has_time and has_categories:
            # Categorical comparison (no time): decision depends on dimension
            if "share" in dims and len(category_cols) == 1:
                # Single-period composition: pie if few categories, bar if many
                unique_cats = df[category_cols[0]].nunique()
                if unique_cats <= 8:
                    chart_type = "pie"
                    log.info(f"📊 Chart type: pie (composition snapshot with {unique_cats} categories)")
                else:
                    chart_type = "bar"
                    log.info(f"📊 Chart type: bar (composition snapshot with {unique_cats} categories, too many for pie)")
            else:
                # Categorical comparison (prices, quantities, etc.) → bar
                chart_type = "bar"
                log.info(f"📊 Chart type: bar (categorical comparison, no time)")

        else:
            # Fallback: no clear structure
            chart_type = "line"
            log.info(f"📊 Chart type: line (fallback)")

        log.info(f"🧠 Chart selection complete → {chart_type} | Time={len(time_cols)} | Categories={len(category_cols)} | Dimensions={dims}")

        # --- UNIT inference for axis title (unit only) ---
        def unit_for_price(cols_: list[str]) -> str:
            has_gel = any("_gel" in c.lower() for c in cols_)
            has_usd = any("_usd" in c.lower() for c in cols_)
            # Mixed currencies share the same physical unit; per your rule, keep unit only:
            if has_gel and has_usd:
                return "per MWh"
            if has_gel:
                return "GEL/MWh"
            if has_usd:
                return "USD/MWh"
            # fallback for generic price columns
            return "per MWh"

        def unit_for_qty(cols_: list[str]) -> str:
            has_tj = any("tj" in c.lower() for c in cols_) or any("volume_tj" in c.lower() for c in cols_)
            # your data uses thousand MWh in quantity_tech/trade volume
            has_thousand_mwh = any("quantity" in c.lower() or "quantity_tech" in c.lower() for c in cols_)
            if has_tj and not has_thousand_mwh:
                return "TJ"
            if has_thousand_mwh and not has_tj:
                return "thousand MWh"
            # mixed TJ & thousand MWh → still a single axis by your rule; show generic quantity unit
            return "Energy Quantity"

        def unit_for_index(_: list[str]) -> str:
            return "Index (2015=100)"

        def unit_for_xrate(_: list[str]) -> str:
            return "GEL per USD"

        def unit_for_share(_: list[str]) -> str:
            return "Share (0-1)"

        # --- LIMIT to max 2-3 series for readability ---
        # If too many series, select most relevant based on query keywords
        MAX_SERIES = 3
        if len(num_cols) > MAX_SERIES:
            log.info(f"⚠️ Too many series ({len(num_cols)}), limiting to {MAX_SERIES} most relevant")

            query_lower = q.query.lower()

            # Score each column by keyword relevance
            def relevance_score(col: str) -> int:
                score = 0
                col_lower = col.lower()

                # High priority keywords
                if any(k in query_lower for k in ["price", "ფასი", "цена"]) and any(k in col_lower for k in ["price", "p_bal", "ფასი"]):
                    score += 10
                if any(k in query_lower for k in ["xrate", "exchange", "კურსი", "курс"]) and "xrate" in col_lower:
                    score += 10
                if any(k in query_lower for k in ["share", "წილი", "доля", "composition"]) and "share" in col_lower:
                    score += 5
                if any(k in query_lower for k in ["tariff", "ტარიფი", "тариф"]) and "tariff" in col_lower:
                    score += 5

                # Prefer primary metrics
                if "p_bal" in col_lower:
                    score += 3
                if "xrate" in col_lower:
                    score += 2

                return score

            # Sort by relevance and take top MAX_SERIES
            scored_cols = [(col, relevance_score(col)) for col in num_cols]
            scored_cols.sort(key=lambda x: x[1], reverse=True)
            num_cols = [col for col, _ in scored_cols[:MAX_SERIES]]

            log.info(f"📊 Selected series: {num_cols}")

            # Recalculate dimensions for filtered columns
            dim_map = {c: infer_dimension(c) for c in num_cols}
            dims = set(dim_map.values())
            log.info(f"📐 Filtered dimensions: {dim_map} → {dims}")

        # --- Labels from context.py for EVERY series (not only numeric) ---
        try:
            from context import COLUMN_LABELS
        except ImportError:
            COLUMN_LABELS = {}

        # Build a label map for ALL columns except the time axis
        label_map_all = {c: COLUMN_LABELS.get(c, c.replace("_", " ").title()) for c in cols if c != time_key}

        # Apply renaming for output (legend/tooltip keys)
        df_labeled = df.rename(columns=label_map_all)

        # Recompute the labeled series list in the same order as num_cols
        chart_labels = [label_map_all.get(c, c) for c in num_cols]

        # --- Determine axis mode & titles ---
        if "index" in dims and len(dims) > 1:
            # CPI mixed with any other → dual axes (index is always right)
            log.info("📊 Mixed index + other dimension → dual-axis chart.")
            chart_type = "dualaxis"
            chart_data = df_labeled.to_dict("records")
            # Left axis unit: prefer price or quantity depending on what is present
            if "price_tariff" in dims:
                left_unit = unit_for_price(num_cols)
            else:
                left_unit = unit_for_qty(num_cols)
            chart_meta = {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": left_unit,                 # unit only
                "yAxisRight": unit_for_index(num_cols), # unit only
                "title": "Index vs Other Indicator",
                "axisMode": "dual",
                "labels": chart_labels,
            }

        elif "price_tariff" in dims and "xrate" in dims:
            # Price + Exchange Rate → dual axes (different units!)
            log.info("📊 Mixed price and xrate → dual-axis chart.")
            chart_type = "dualaxis"
            chart_data = df_labeled.to_dict("records")
            chart_meta = {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": unit_for_price(num_cols),   # GEL/MWh or USD/MWh
                "yAxisRight": unit_for_xrate(num_cols),  # GEL per USD
                "title": "Price vs Exchange Rate",
                "axisMode": "dual",
                "labels": chart_labels,
            }

        elif "price_tariff" in dims and "share" in dims:
            # Price + Share → dual axes (different scales: 0-200 vs 0-1)
            log.info("📊 Mixed price and share → dual-axis chart.")
            chart_type = "dualaxis"
            chart_data = df_labeled.to_dict("records")
            chart_meta = {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": unit_for_price(num_cols),  # GEL/MWh or USD/MWh
                "yAxisRight": unit_for_share(num_cols), # Share (0-1)
                "title": "Price vs Composition Shares",
                "axisMode": "dual",
                "labels": chart_labels,
            }

        elif "price_tariff" in dims and "energy_qty" in dims:
            # Price/Tariff + Quantity → dual axes
            log.info("📊 Mixed price/tariff and quantity → dual-axis chart.")
            chart_type = "dualaxis"
            chart_data = df_labeled.to_dict("records")
            chart_meta = {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": unit_for_qty(num_cols),     # unit only (TJ / thousand MWh)
                "yAxisRight": unit_for_price(num_cols),  # unit only (GEL/MWh, USD/MWh, or per MWh)
                "title": "Quantity vs Price/Tariff",
                "axisMode": "dual",
                "labels": chart_labels,
            }

        elif "xrate" in dims and "share" in dims:
            # Exchange Rate + Share → dual axes
            log.info("📊 Mixed xrate and share → dual-axis chart.")
            chart_type = "dualaxis"
            chart_data = df_labeled.to_dict("records")
            chart_meta = {
                "xAxisTitle": time_key or "time",
                "yAxisLeft": unit_for_xrate(num_cols),  # GEL per USD
                "yAxisRight": unit_for_share(num_cols), # Share (0-1)
                "title": "Exchange Rate vs Composition",
                "axisMode": "dual",
                "labels": chart_labels,
            }

        else:
            log.info("📊 Uniform dimension → single-axis chart (respecting earlier chart type).")

            # Respect the earlier classification (stackedbar, bar, etc.)
            if chart_type not in ["stackedbar", "bar", "pie", "dualaxis"]:
                chart_type = "line"
            chart_data = df_labeled.to_dict("records")


            # Decide unit by the only dimension present
            if dims == {"price_tariff"}:
                y_unit = unit_for_price(num_cols)
            elif dims == {"energy_qty"}:
                y_unit = unit_for_qty(num_cols)
            elif dims == {"index"}:
                y_unit = unit_for_index(num_cols)
            elif dims == {"xrate"}:
                y_unit = unit_for_xrate(num_cols)
            elif dims == {"share"}:
                y_unit = unit_for_share(num_cols)
            else:
                y_unit = "Value"

            chart_meta = {
                "xAxisTitle": time_key or "time",
                "yAxisTitle": y_unit,              # unit only
                "title": "Indicator Comparison (same dimension)",
                "axisMode": "single",
                "labels": chart_labels,
            }

        log.info(f"✅ Chart built | type={chart_type} | axisMode={chart_meta.get('axisMode')} | labels={chart_labels}")

        # --- 📈 Calculate trendlines if requested ---
        if add_trendlines and time_key and time_key in df.columns:
            from visualization.chart_builder import calculate_trendline

            log.info(f"📈 Calculating trendlines for {len(num_cols)} series")
            trendlines = []

            for col in num_cols:
                trendline_data = calculate_trendline(
                    df, time_key, col, extend_to_date=trendline_extend_to
                )
                if trendline_data:
                    # Get the label for this series from chart_labels
                    label_idx = num_cols.index(col)
                    label = chart_labels[label_idx] if label_idx < len(chart_labels) else col

                    trendlines.append({
                        "column": col,
                        "label": f"{label} (Trend)",
                        "data": trendline_data,
                        "original_label": label
                    })
                    log.info(f"  ✅ {label}: R²={trendline_data['r_squared']:.3f}, equation={trendline_data['equation']}")

            # Add trendline info to chart metadata
            if trendlines:
                chart_meta["trendlines"] = trendlines
                chart_meta["has_projection"] = bool(trendline_extend_to)
                chart_meta["projection_to"] = trendline_extend_to if trendline_extend_to else None
                log.info(f"📊 Added {len(trendlines)} trendlines to chart metadata")
        elif add_trendlines:
            log.warning(f"⚠️ Trendlines requested but no time column found (time_key={time_key})")

    # 6) Final response
    exec_time = time.time() - t0
    metrics.log_request(exec_time)
    log.info(f"Finished request in {exec_time:.2f}s")

    response = APIResponse(
        answer=summary,
        chart_data=chart_data,
        chart_type=chart_type,
        chart_metadata=chart_meta,
        execution_time=exec_time,
    )
    return response


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
