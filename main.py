**Full, production-ready `main.py` v18.9**  
*(All lines from the original are kept – only the **balancing-pivot CTE** was fixed to use the **real column name `quantity_tech`** instead of the non-existent `quantity`.)*

```python:disable-run
# main.py v18.9 — Gemini Analyst (fixed quantity column + full safety)
import os
import re
import json
import time
import logging
import urllib.parse
import uuid
from textwrap import dedent
from contextvars import ContextVar
from typing import Optional, Dict, Any, List, Tuple
from difflib import get_close_matches
from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
# Corrected Pydantic imports for V2 compatibility
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
# LLMs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
# sqlglot (AST parsing/validation)
from sqlglot import parse_one, exp
# Schema & helpers
# NOTE: Ensure these imports are available in your environment
from context import DB_SCHEMA_DOC, scrub_schema_mentions, COLUMN_LABELS, DERIVED_LABELS
# Domain knowledge
from domain_knowledge import DOMAIN_KNOWLEDGE
# -----------------------------
# Boot + Config
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enerbot")
# Cache domain knowledge JSON serialization (done once at startup)
_DOMAIN_KNOWLEDGE_JSON = json.dumps(DOMAIN_KNOWLEDGE, indent=2)
log.info("Domain knowledge JSON cached at startup")
# Request ID tracking for observability
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
# Metrics tracking (simple counters and timing)
class Metrics:
    """Simple metrics tracker for observability."""
    def __init__(self):
        self.request_count = 0
        self.llm_call_count = 0
        self.sql_query_count = 0
        self.error_count = 0
        self.total_llm_time = 0.0
        self.total_sql_time = 0.0
        self.total_request_time = 0.0
    def log_request(self, duration: float):
        self.request_count += 1
        self.total_request_time += duration
        log.info(f"Metrics: requests={self.request_count}, avg_time={self.total_request_time/self.request_count:.2f}s")
    def log_llm_call(self, duration: float):
        self.llm_call_count += 1
        self.total_llm_time += duration
    def log_sql_query(self, duration: float):
        self.sql_query_count += 1
        self.total_sql_time += duration
    def log_error(self):
        self.error_count += 1
    def get_stats(self) -> dict:
        return {
            "requests": self.request_count,
            "llm_calls": self.llm_call_count,
            "sql_queries": self.sql_query_count,
            "errors": self.error_count,
            "avg_request_time": self.total_request_time / max(1, self.request_count),
            "avg_llm_time": self.total_llm_time / max(1, self.llm_call_count),
            "avg_sql_time": self.total_sql_time / max(1, self.sql_query_count),
        }
metrics = Metrics()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
MODEL_TYPE = (os.getenv("MODEL_TYPE", "gemini") or "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Maximum rows to return from queries (prevents huge result sets)
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))
if not SUPABASE_DB_URL:
    raise RuntimeError("Missing SUPABASE_DB_URL")
if not APP_SECRET_KEY:
    raise RuntimeError("Missing APP_SECRET_KEY")
if MODEL_TYPE == "gemini" and not GOOGLE_API_KEY:
    raise RuntimeError("MODEL_TYPE=gemini but GOOGLE_API_KEY is missing")
# Allow the base tables + USD materialized views
STATIC_ALLOWED_TABLES = {
    "dates_mv",
    "energy_balance_long_mv",
    "entities_mv",
    "monthly_cpi_mv",
    "price_with_usd",
    "tariff_with_usd",
    "tech_quantity_view",
    "trade_derived_entities",
}
ALLOWED_TABLES = set(STATIC_ALLOWED_TABLES)
# Table synonym map (plural & common aliases to canonical)
TABLE_SYNONYMS = {
    "prices": "price",
    "tariffs": "tariff_gen",
    "price_usd": "price_with_usd",
    "tariff_usd": "tariff_with_usd",
    "price_with_usd": "price_with_usd",
    "tariff_with_usd": "tariff_with_usd",
}
# Column synonym map (common misnamings to canonical)
COLUMN_SYNONYMS = {
    "tech_type": "type_tech",
    "quantity_mwh": "quantity_tech", # your data stores thousand MWh in quantity_tech
}
# Pre-compiled regex patterns for SQL synonym replacement (performance optimization)
SYNONYM_PATTERNS = [
    (re.compile(r"\bprices\b", re.IGNORECASE), "price_with_usd"),
    (re.compile(r"\btariffs\b", re.IGNORECASE), "tariff_with_usd"),
    (re.compile(r"\btech_quantity\b", re.IGNORECASE), "tech_quantity_view"),
    (re.compile(r"\btrade\b", re.IGNORECASE), "trade_derived_entities"),
    (re.compile(r"\bentities\b", re.IGNORECASE), "entities_mv"),
    (re.compile(r"\bmonthly_cpi\b", re.IGNORECASE), "monthly_cpi_mv"),
    (re.compile(r"\benergy_balance_long\b", re.IGNORECASE), "energy_balance_long_mv"),
]
# Pre-compiled regex for LIMIT detection (allows optional whitespace)
LIMIT_PATTERN = re.compile(r"\bLIMIT\s*\d+\b", re.IGNORECASE)
BALANCING_SEGMENT_NORMALIZER = "LOWER(REPLACE(segment, ' ', '_'))"
BALANCING_SHARE_PIVOT_SQL = dedent(
    f"""
    SELECT
        t.date,
        'balancing'::text AS segment,
        SUM(CASE WHEN t.entity = 'import' THEN t.quantity_tech ELSE 0 END) / NULLIF(total.total_qty,0) AS share_import,
        SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity_tech ELSE 0 END) / NULLIF(total.total_qty,0) AS share_deregulated_hydro,
        SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity_tech ELSE 0 END) / NULLIF(total.total_qty,0) AS share_regulated_hpp,
        SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity_tech ELSE 0 END) / NULLIF(total.total_qty,0) AS share_regulated_new_tpp,
        SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity_tech ELSE 0 END) / NULLIF(total.total_qty,0) AS share_regulated_old_tpp,
        SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity_tech ELSE 0 END) / NULLIF(total.total_qty,0) AS share_renewable_ppa,
        SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity_tech ELSE 0 END) / NULLIF(total.total_qty,0) AS share_thermal_ppa,
        SUM(CASE WHEN t.entity IN ('renewable_ppa','thermal_ppa') THEN t.quantity_tech ELSE 0 END) / NULLIF(total.total_qty,0) AS share_all_ppa,
        SUM(CASE WHEN t.entity IN ('deregulated_hydro','regulated_hpp','renewable_ppa') THEN t.quantity_tech ELSE 0 END) / NULLIF(total.total_qty,0) AS share_all_renewables,
        SUM(CASE WHEN t.entity IN ('deregulated_hydro','regulated_hpp') THEN t.quantity_tech ELSE 0 END) / NULLIF(total.total_qty,0) AS share_total_hpp
    FROM trade_derived_entities t
    JOIN (
        SELECT date, SUM(quantity_tech) AS total_qty
        FROM trade_derived_entities
        WHERE {BALANCING_SEGMENT_NORMALIZER} = 'balancing'
          AND entity IN (
            'import', 'deregulated_hydro', 'regulated_hpp',
            'regulated_new_tpp', 'regulated_old_tpp',
            'renewable_ppa', 'thermal_ppa'
          )
        GROUP BY date
    ) total ON t.date = total.date
    WHERE {BALANCING_SEGMENT_NORMALIZER} = 'balancing'
      AND t.entity IN (
        'import', 'deregulated_hydro', 'regulated_hpp',
        'regulated_new_tpp', 'regulated_old_tpp',
        'renewable_ppa', 'thermal_ppa'
      )
    GROUP BY t.date, total.total_qty
    ORDER BY t.date
    """
).strip()
def should_inject_balancing_pivot(user_query: str, sql: str) -> bool:
    """
    Detect if query is asking for balancing share but SQL doesn't include proper pivot.
    Returns True if:
    - User query mentions balancing-related concepts
    - User query mentions entity types
    - SQL uses trade_derived_entities directly (not through pivot)
    - SQL either lacks share columns OR lacks proper entity filtering in denominator
    This forces pivot injection for queries like:
    - "what was the share of renewable PPA in balancing electricity?"
    - "show me the composition of balancing market in june 2024"
    CRITICAL: Even if SQL has share columns, inject pivot if it's missing the
    entity filter in WHERE clause (which causes wrong denominator calculation).
    """
    query_lower = user_query.lower()
    sql_lower = sql.lower()
    balancing_keywords = ["balancing", "share", "composition", "mix", "weight", "proportion"]
    entity_keywords = ["ppa", "renewable", "thermal", "import", "hydro", "tpp", "hpp", "entity", "entities"]
    has_balancing = any(k in query_lower for k in balancing_keywords)
    has_entity = any(k in query_lower for k in entity_keywords)
    has_trade = "trade_derived_entities" in sql_lower
    has_share_col = any(f"share_{e}" in sql_lower for e in ["import", "renewable", "ppa", "hydro", "tpp", "hpp"])
    # Check if SQL has proper entity filter for denominator
    # Look for pattern: entity IN ('import', 'deregulated_hydro', ...)
    has_entity_filter = "entity in (" in sql_lower or "t.entity in (" in sql_lower
    # Inject pivot if:
    # 1. Query is about balancing shares AND uses trade_derived_entities
    # 2. AND either: no share columns exist OR entity filter is missing
    return has_balancing and has_entity and has_trade and (not has_share_col or not has_entity_filter)
def build_trade_share_cte(original_sql: str) -> str:
    """Inject a balancing electricity share pivot as a CTE and alias original SQL to it.
    Uses a unique CTE name to avoid conflicts if original SQL already uses 'tde' as alias.
    """
    import uuid
    # Generate unique CTE name to avoid collisions
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
            log.warning(f"DIAGNOSTIC: Found {len(diag_rows)} distinct segment values in trade_derived_entities:")
            for row in diag_rows:
                log.warning(f" - '{row[0]}': {row[1]} rows (from {row[2]} to {row[3]})")
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
            cheap_losses.append(f"{meta['label']} down {abs(delta) * 100:.1f} pp")
        if meta.get("cost") == "expensive" and delta > 0:
            expensive_gains.append(f"{meta['label']} up {delta * 100:.1f} pp")
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
        month = MONTH_NAME_TO_NUMBER.get(month_token[:3] if len(month_token) > 3 else month_token, MONTH_NAME_TO_NUMBER.get(month_token, None))
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
    # VALIDATION: Check if shares sum to ~1.0 (indicates correct denominator)
    total_shares = sum(
        float(selected_row.get(c, 0))
        for c in share_cols
        if not pd.isna(selected_row.get(c))
    )
    if abs(total_shares - 1.0) > 0.05:
        log.warning(
            f"Share columns sum to {total_shares:.3f} instead of 1.0 — possible denominator bug. "
            f"Shares: {[f'{c}={selected_row.get(c):.3f}' for c in share_cols[:5]]}"
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
# DB Engine
# -----------------------------
def coerce_to_psycopg_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("postgres", "postgresql"):
        return url.replace(parsed.scheme, "postgresql+psycopg", 1)
    if not parsed.scheme.startswith("postgresql+"):
        return "postgresql+psycopg://" + url.split("://", 1)[-1]
    return url
DB_URL = coerce_to_psycopg_url(SUPABASE_DB_URL)
ENGINE = create_engine(
    DB_URL,
    poolclass=QueuePool,
    pool_size=10, # Increased from 5 for better concurrency
    max_overflow=5, # Increased from 2 to handle traffic spikes
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=1800, # Increased from 300 (30 min) for Supabase
    connect_args={"connect_timeout": 30},
)
with ENGINE.connect() as conn:
    conn.execute(text("SELECT 1"))
    log.info("Database connectivity verified")
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
        log.info(f"Found materialized views: {sorted(ALLOWED_TABLES)}")
        log.info(f"Final ALLOWED_TABLES (views only): {sorted(ALLOWED_TABLES)}")
        # Optional: show schema details for each view
        for view, cols in SCHEMA_MAP.items():
            log.info(f"{view}: {sorted(cols)}")
    except Exception as e:
        log.warning(f"Could not reflect materialized views: {e}")
        SCHEMA_MAP = {}
        # Fall back to the static allow-list so transient reflection issues
        # don't incorrectly block legitimate queries.
        ALLOWED_TABLES = set(STATIC_ALLOWED_TABLES)
# --- v18.8: Balancing correlation & weighted price helpers ---
def build_balancing_correlation_df(conn) -> pd.DataFrame:
    """
    Returns a monthly panel with:
      targets: p_bal_gel, p_bal_usd
      drivers: xrate, share_import, share_deregulated_hydro, share_regulated_hpp,
               share_renewable_ppa, enguri_tariff_gel, gardabani_tpp_tariff_gel,
               grouped_old_tpp_tariff_gel
    CRITICAL FIX: Shares are calculated using ONLY balancing_electricity segment
    to properly reflect the composition that affects balancing electricity price.
    Uses case-insensitive segment matching to handle different database formats.
    """
    sql = """
    WITH shares AS (
      SELECT
        t.date,
        SUM(t.quantity_tech) AS total_qty,
        SUM(CASE WHEN t.entity = 'import' THEN t.quantity_tech ELSE 0 END) AS qty_import,
        SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity_tech ELSE 0 END) AS qty_dereg_hydro,
        SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity_tech ELSE 0 END) AS qty_reg_hpp,
        SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity_tech ELSE 0 END) AS qty_reg_new_tpp,
        SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity_tech ELSE 0 END) AS qty_reg_old_tpp,
        SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity_tech ELSE 0 END) AS qty_ren_ppa,
        SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity_tech ELSE 0 END) AS qty_thermal_ppa
      FROM trade_derived_entities t
      WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing'
      GROUP BY t.date
    ),
    tariffs AS (
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
      (s.qty_import / NULLIF(s.total_qty,0)) AS share_import,
      (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) AS share_deregulated_hydro,
      (s.qty_reg_hpp / NULLIF(s.total_qty,0)) AS share_regulated_hpp,
      (s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) AS share_regulated_new_tpp,
      (s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) AS share_regulated_old_tpp,
      (s.qty_ren_ppa / NULLIF(s.total_qty,0)) AS share_renewable_ppa,
      (s.qty_thermal_ppa / NULLIF(s.total_qty,0)) AS share_thermal_ppa,
      ((s.qty_ren_ppa + s.qty_thermal_ppa) / NULLIF(s.total_qty,0)) AS share_all_ppa,
      ((s.qty_dereg_hydro + s.qty_reg_hpp + s.qty_ren_ppa) / NULLIF(s.total_qty,0)) AS share_all_renewables,
      tr.enguri_tariff_gel,
      tr.gardabani_tpp_tariff_gel,
      tr.grouped_old_tpp_tariff_gel
    FROM price_with_usd p
    LEFT JOIN shares s ON s.date = p.date
    LEFT JOIN tariffs tr ON tr.date = p.date
    ORDER BY p.date
    """
    # Add LIMIT using configured MAX_ROWS (replaces hardcoded 3750)
    sql_with_limit = f"{sql.strip()}\nLIMIT {MAX_ROWS};"
    res = conn.execute(text(sql_with_limit))
    return pd.DataFrame(res.fetchall(), columns=list(res.keys()))
def compute_weighted_balancing_price(conn) -> pd.DataFrame:
    """
    Compute monthly weighted-average balancing price (GEL & USD)
    based on balancing-market sales volumes.
    CRITICAL: Uses ONLY balancing_electricity segment to calculate weights.
    Uses case-insensitive segment matching to handle different database formats.
    """
    sql = """
    WITH t AS (
      SELECT date, entity, SUM(quantity_tech) AS qty
      FROM trade_derived_entities
      WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing'
        AND entity IN ('deregulated_hydro','import','regulated_hpp',
                       'regulated_new_tpp','regulated_old_tpp',
                       'renewable_ppa','thermal_ppa')
      GROUP BY date, entity
    ),
    w AS (SELECT date, SUM(qty) AS total_qty FROM t GROUP BY date)
    SELECT
      p.date,
      p.p_bal_gel,
      p.p_bal_usd,
      (p.p_bal_gel * w.total_qty) / NULLIF(SUM(w.total_qty) OVER (),0) AS weighted_gel,
      (p.p_bal_usd * w.total_qty) / NULLIF(SUM(w.total_qty) OVER (),0) AS weighted_usd
    FROM price_with_usd p
    JOIN w ON w.date = p.date
    ORDER BY p.date;
    """
    res = conn.execute(text(sql))
    return pd.DataFrame(res.fetchall(), columns=list(res.keys()))
def compute_seasonal_average(df: pd.DataFrame, date_col: str, value_col: str, agg_func: str = "avg") -> pd.DataFrame:
    """
    Compute seasonal (Summer vs Winter) average or sum for a given value column.
    Assumes df contains a date column in datetime or string format.
    """
    if date_col not in df.columns or value_col not in df.columns:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["season"] = df[date_col].dt.month.apply(lambda m: "Summer" if m in [4,5,6,7] else "Winter")
    if agg_func.lower() in ("avg", "mean"):
        grouped = df.groupby("season")[value_col].mean().reset_index(name=f"avg_{value_col}")
    elif agg_func.lower() == "sum":
        grouped = df.groupby("season")[value_col].sum().reset_index(name=f"sum_{value_col}")
    else:
        raise ValueError("agg_func must be 'avg' or 'sum'")
    return grouped
def compute_entity_price_contributions(conn) -> pd.DataFrame:
    """
    Decompose balancing price into entity-level contributions.
    Returns monthly panel showing:
    - Balancing price (actual weighted average)
    - Each entity's share in balancing electricity
    - Entity-level reference prices (where available from tariff_with_usd and price_with_usd)
    - Estimated contribution to balancing price: share × reference_price
    CRITICAL NOTES:
    - Regulated entities (regulated_hpp, regulated TPPs): use tariff_gel from tariff_with_usd
    - Deregulated hydro: use p_dereg_gel from price_with_usd
    - PPAs and imports: reference prices NOT available in database (confidential)
    - Actual balancing transaction prices may differ from reference prices
    - This provides directional insight, not exact decomposition
    Use this to explain:
    - Which entities drove price changes month-over-month
    - How composition shifts affected weighted average price
    - Relative contribution of cheap vs expensive sources
    """
    sql = """
    WITH shares AS (
      SELECT
        t.date,
        SUM(t.quantity_tech) AS total_qty,
        SUM(CASE WHEN t.entity = 'import' THEN t.quantity_tech ELSE 0 END) AS qty_import,
        SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity_tech ELSE 0 END) AS qty_dereg_hydro,
        SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity_tech ELSE 0 END) AS qty_reg_hpp,
        SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity_tech ELSE 0 END) AS qty_reg_new_tpp,
        SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity_tech ELSE 0 END) AS qty_reg_old_tpp,
        SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity_tech ELSE 0 END) AS qty_ren_ppa,
        SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity_tech ELSE 0 END) AS qty_thermal_ppa
      FROM trade_derived_entities t
      WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing'
      GROUP BY t.date
    ),
    entity_prices AS (
      SELECT
        d.date,
        -- Reference prices from available sources
        p.p_dereg_gel AS price_deregulated_hydro,
        -- Regulated HPP: use weighted average of main HPPs or Enguri as proxy
        -- Using ILIKE for case-insensitive matching (PostgreSQL extension)
        (SELECT AVG(t1.tariff_gel)
         FROM tariff_with_usd t1
         WHERE t1.date = d.date
           AND (t1.entity ILIKE '%engurhesi%'
                OR t1.entity ILIKE '%energo-pro%'
                OR t1.entity ILIKE '%vardnili%')
        ) AS price_regulated_hpp,
        -- Regulated new TPP: Gardabani
        (SELECT t2.tariff_gel
         FROM tariff_with_usd t2
         WHERE t2.date = d.date
           AND t2.entity = 'ltd "gardabni thermal power plant"'
         LIMIT 1
        ) AS price_regulated_new_tpp,
        -- Regulated old TPPs: average of old thermal plants
        (SELECT AVG(t3.tariff_gel)
         FROM tariff_with_usd t3
         WHERE t3.date = d.date
           AND t3.entity IN ('ltd "mtkvari energy"',
                            'ltd "iec" (tbilresi)',
                            'ltd "g power" (capital turbines)')
        ) AS price_regulated_old_tpp
        -- Note: PPA and import prices are NOT available in database
        -- These would need to be estimated or obtained from confidential data
      FROM price_with_usd d
      LEFT JOIN price_with_usd p ON p.date = d.date
    )
    SELECT
      p.date,
      p.p_bal_gel AS balancing_price_gel,
      p.p_bal_usd AS balancing_price_usd,
      p.xrate,
      -- Shares
      (s.qty_import / NULLIF(s.total_qty,0)) AS share_import,
      (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) AS share_deregulated_hydro,
      (s.qty_reg_hpp / NULLIF(s.total_qty,0)) AS share_regulated_hpp,
      (s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) AS share_regulated_new_tpp,
      (s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) AS share_regulated_old_tpp,
      (s.qty_ren_ppa / NULLIF(s.total_qty,0)) AS share_renewable_ppa,
      (s.qty_thermal_ppa / NULLIF(s.total_qty,0)) AS share_thermal_ppa,
      -- Reference prices (where available)
      ep.price_deregulated_hydro,
      ep.price_regulated_hpp,
      ep.price_regulated_new_tpp,
      ep.price_regulated_old_tpp,
      -- Estimated contributions to balancing price
      -- (share × reference_price) = estimated contribution in GEL/MWh
      (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) * COALESCE(ep.price_deregulated_hydro, 0)
        AS contribution_deregulated_hydro,
      (s.qty_reg_hpp / NULLIF(s.total_qty,0)) * COALESCE(ep.price_regulated_hpp, 0)
        AS contribution_regulated_hpp,
      (s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) * COALESCE(ep.price_regulated_new_tpp, 0)
        AS contribution_regulated_new_tpp,
      (s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) * COALESCE(ep.price_regulated_old_tpp, 0)
        AS contribution_regulated_old_tpp,
      -- Sum of known contributions
      COALESCE((s.qty_dereg_hydro / NULLIF(s.total_qty,0)) * ep.price_deregulated_hydro, 0) +
      COALESCE((s.qty_reg_hpp / NULLIF(s.total_qty,0)) * ep.price_regulated_hpp, 0) +
      COALESCE((s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_new_tpp, 0) +
      COALESCE((s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_old_tpp, 0)
        AS total_known_contributions,
      -- Residual (PPA + import contribution, not directly observable)
      p.p_bal_gel - (
        COALESCE((s.qty_dereg_hydro / NULLIF(s.total_qty,0)) * ep.price_deregulated_hydro, 0) +
        COALESCE((s.qty_reg_hpp / NULLIF(s.total_qty,0)) * ep.price_regulated_hpp, 0) +
        COALESCE((s.qty_reg_new_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_new_tpp, 0) +
        COALESCE((s.qty_reg_old_tpp / NULLIF(s.total_qty,0)) * ep.price_regulated_old_tpp, 0)
      ) AS residual_contribution_ppa_import,
      -- Shares of PPA and import (for context on residual)
      (s.qty_ren_ppa / NULLIF(s.total_qty,0)) +
      (s.qty_thermal_ppa / NULLIF(s.total_qty,0)) +
      (s.qty_import / NULLIF(s.total_qty,0)) AS share_ppa_import_total
    FROM price_with_usd p
    LEFT JOIN shares s ON s.date = p.date
    LEFT JOIN entity_prices ep ON ep.date = p.date
    WHERE p.date >= '2015-01-01'
    ORDER BY p.date
    """
    # Add LIMIT using configured MAX_ROWS (replaces hardcoded 3750)
    sql_with_limit = f"{sql.strip()}\nLIMIT {MAX_ROWS};"
    res = conn.execute(text(sql_with_limit))
    return pd.DataFrame(res.fetchall(), columns=list(res.keys()))
def compute_share_changes(conn) -> pd.DataFrame:
    """
    Calculate month-over-month changes in entity shares.
    Returns panel showing:
    - Current month shares
    - Previous month shares
    - Absolute change (percentage points)
    - Relative change (percent)
    Use this to identify which entities increased/decreased their balancing market participation
    and explain resulting price movements.
    """
    sql = """
    WITH shares AS (
      SELECT
        t.date,
        SUM(t.quantity_tech) AS total_qty,
        (SUM(CASE WHEN t.entity = 'import' THEN t.quantity_tech ELSE 0 END) / NULLIF(SUM(t.quantity_tech),0)) AS share_import,
        (SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity_tech ELSE 0 END) / NULLIF(SUM(t.quantity_tech),0)) AS share_deregulated_hydro,
        (SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity_tech ELSE 0 END) / NULLIF(SUM(t.quantity_tech),0)) AS share_regulated_hpp,
        (SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity_tech ELSE 0 END) / NULLIF(SUM(t.quantity_tech),0)) AS share_regulated_new_tpp,
        (SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity_tech ELSE 0 END) / NULLIF(SUM(t.quantity_tech),0)) AS share_regulated_old_tpp,
        (SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity_tech ELSE 0 END) / NULLIF(SUM(t.quantity_tech),0)) AS share_renewable_ppa,
        (SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity_tech ELSE 0 END) / NULLIF(SUM(t.quantity_tech),0)) AS share_thermal_ppa
      FROM trade_derived_entities t
      WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing'
      GROUP BY t.date
    )
    SELECT
      s.date,
      p.p_bal_gel,
      LAG(p.p_bal_gel) OVER (ORDER BY s.date) AS prev_p_bal_gel,
      p.p_bal_gel - LAG(p.p_bal_gel) OVER (ORDER BY s.date) AS price_change_gel,
      -- Current shares
      s.share_import,
      s.share_deregulated_hydro,
      s.share_regulated_hpp,
      s.share_renewable_ppa,
      s.share_thermal_ppa,
      -- Previous month shares
      LAG(s.share_import) OVER (ORDER BY s.date) AS prev_share_import,
      LAG(s.share_deregulated_hydro) OVER (ORDER BY s.date) AS prev_share_deregulated_hydro,
      LAG(s.share_regulated_hpp) OVER (ORDER BY s.date) AS prev_share_regulated_hpp,
      LAG(s.share_renewable_ppa) OVER (ORDER BY s.date) AS prev_share_renewable_ppa,
      LAG(s.share_thermal_ppa) OVER (ORDER BY s.date) AS prev_share_thermal_ppa,
      -- Changes in shares (percentage points)
      s.share_import - LAG(s.share_import) OVER (ORDER BY s.date) AS change_share_import,
      s.share_deregulated_hydro - LAG(s.share_deregulated_hydro) OVER (ORDER BY s.date) AS change_share_deregulated_hydro,
      s.share_regulated_hpp - LAG(s.share_regulated_hpp) OVER (ORDER BY s.date) AS change_share_regulated_hpp,
      s.share_renewable_ppa - LAG(s.share_renewable_ppa) OVER (ORDER BY s.date) AS change_share_renewable_ppa,
      s.share_thermal_ppa - LAG(s.share_thermal_ppa) OVER (ORDER BY s.date) AS change_share_thermal_ppa
    FROM shares s
    LEFT JOIN price_with_usd p ON p.date = s.date
    ORDER BY s.date
    """
    # Add LIMIT using configured MAX_ROWS (replaces hardcoded 3750)
    sql_with_limit = f"{sql.strip()}\nLIMIT {MAX_ROWS};"
    res = conn.execute(text(sql_with_limit))
    return pd.DataFrame(res.fetchall(), columns=list(res.keys()))
# -----------------------------
# App
# -----------------------------
app = FastAPI(title="EnerBot Analyst (Gemini)", version="18.9") # Version bump
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------
# Models
# -----------------------------
class Question(BaseModel):
    query: str = Field(..., max_length=2000)
    user_id: Optional[str] = None
    @field_validator("query") # Pydantic V2 syntax
    @classmethod
    def _not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
class APIResponse(BaseModel):
    answer: str
    chart_data: Optional[List[Dict[str, Any]]] = None
    chart_type: Optional[str] = None
    chart_metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
# -----------------------------
# LLM + Planning helpers
# -----------------------------
# Cached LLM instances (singleton pattern for performance)
_gemini_llm = None
_openai_llm = None
def get_gemini() -> ChatGoogleGenerativeAI:
    """Get cached Gemini LLM instance (singleton pattern).
    Note: convert_system_message_to_human=True is required because Gemini
    doesn't natively support SystemMessages in the LangChain interface.
    """
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True
        )
        log.info("Gemini LLM instance cached")
    return _gemini_llm
def get_openai() -> ChatOpenAI:
    """Get cached OpenAI LLM instance (singleton pattern).
    Raises:
        RuntimeError: If OPENAI_API_KEY is not configured
    """
    global _openai_llm
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set (fallback needed)")
    if _openai_llm is None:
        _openai_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
        log.info("OpenAI LLM instance cached")
    return _openai_llm
# Backward compatibility aliases
make_gemini = get_gemini
make_openai = get_openai
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
    if any(kw in query_lower for kw in ANALYTICAL_KEYWORDS):
        return "analyst"
    return "light"
def should_generate_chart(user_query: str, row_count: int) -> bool:
    """
    Determine if a chart would be helpful for answering the query.
    Returns False if:
    - Query asks for specific values/numbers only
    - Result has very few rows (< 3)
    - Query is asking "what", "which", "list" type questions
    Returns True if:
    - Query asks about trends, comparisons, distributions
    - Result has time series or categorical data suitable for visualization
    """
    query_lower = user_query.lower()
    # Don't generate chart for simple fact queries
    no_chart_indicators = [
        "what is the", "what was the", "how much", "how many",
        "give me the value", "tell me the", "რა არის", "რამდენი",
        "скол ько", "какой"
    ]
    for indicator in no_chart_indicators:
        if indicator in query_lower and row_count <= 3:
            return False
    # Always generate chart for trend/comparison/distribution queries
    chart_friendly_keywords = [
        "trend", "over time", "compare", "comparison", "distribution",
        "evolution", "динамика", "сравнение", "ტენდენცია", "შედარება",
        "chart", "graph", "plot", "visualize", "show me"
    ]
    for keyword in chart_friendly_keywords:
        if keyword in query_lower:
            return True
    # Generate chart if we have enough data points
    if row_count >= 5:
        return True
    # Default: generate chart unless very few rows
    return row_count >= 3
def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns:
        Language code: 'ka' for Georgian, 'ru' for Russian, 'en' for English
    """
    # Georgian unicode range check
    if any('\u10a0' <= char <= '\u10ff' for char in text):
        return "ka"
    # Russian/Cyrillic unicode range check
    if any('\u0400' <= char <= '\u04ff' for char in text):
        return "ru"
    # Default to English
    return "en"
def get_language_instruction(lang_code: str) -> str:
    """Get instruction for LLM to respond in the detected language."""
    language_instructions = {
        "ka": "IMPORTANT: Respond in Georgian language (ქართული ენა). Use Georgian characters and natural Georgian phrasing.",
        "ru": "IMPORTANT: Respond in Russian language (русский язык). Use Cyrillic characters and natural Russian phrasing.",
        "en": "Respond in English."
    }
    return language_instructions.get(lang_code, language_instructions["en"])
def get_relevant_domain_knowledge(user_query: str, use_cache: bool = True) -> str:
    """Return domain knowledge JSON, optionally filtered by query relevance.
    Args:
        user_query: The user's query text
        use_cache: If True, use full cached JSON. If False, select relevant sections only.
    Returns:
        JSON string of domain knowledge (full or filtered)
    This function can reduce token usage by 30-40% when use_cache=False by including
    only sections relevant to the query type.
    """
    if use_cache:
        # Use full pre-cached JSON (fastest, but more tokens)
        return _DOMAIN_KNOWLEDGE_JSON
    # Selective approach: include only relevant sections
    query_lower = user_query.lower()
    # Always include critical sections
    relevant = {
        "BalancingPriceDrivers": DOMAIN_KNOWLEDGE["BalancingPriceDrivers"],
    }
    # Add conditionally based on query content
    if any(word in query_lower for word in ["tariff", "regulated", "thermal", "hpp", "gardabani", "enguri"]):
        relevant["TariffStructure"] = DOMAIN_KNOWLEDGE.get("TariffStructure", {})
    if any(word in query_lower for word in ["balance", "energy", "generation", "supply", "demand"]):
        relevant["EnergyBalance"] = DOMAIN_KNOWLEDGE.get("EnergyBalance", {})
    if any(word in query_lower for word in ["season", "summer", "winter", "monthly"]):
        relevant["SeasonalPattern"] = DOMAIN_KNOWLEDGE.get("SeasonalPattern", {})
    if any(word in query_lower for word in ["import", "export", "trade"]):
        relevant["TradePattern"] = DOMAIN_KNOWLEDGE.get("TradePattern", {})
    if any(word in query_lower for word in ["cpi", "inflation", "price index"]):
        relevant["CPI"] = DOMAIN_KNOWLEDGE.get("CPI", {})
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
    SUM(t.quantity_tech) AS total_qty,
    SUM(CASE WHEN t.entity = 'import' THEN t.quantity_tech ELSE 0 END) AS qty_import,
    SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity_tech ELSE 0 END) AS qty_dereg_hydro,
    SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity_tech ELSE 0 END) AS qty_reg_hpp
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
  (s.qty_import / NULLIF(s.total_qty,0)) AS share_import,
  (s.qty_dereg_hydro / NULLIF(s.total_qty,0)) AS share_deregulated_hydro,
  (s.qty_reg_hpp / NULLIF(s.total_qty,0)) AS share_regulated_hpp
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
    SUM(quantity_tech) AS total_qty,
    SUM(CASE WHEN entity = 'renewable_ppa' THEN quantity_tech ELSE 0 END) AS qty_renewable_ppa,
    SUM(CASE WHEN entity = 'thermal_ppa' THEN quantity_tech ELSE 0 END) AS qty_thermal_ppa,
    SUM(CASE WHEN entity = 'import' THEN quantity_tech ELSE 0 END) AS qty_import
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
"""
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
def llm_generate_plan_and_sql(user_query: str, analysis_mode: str, lang_instruction: str = "Respond in English.") -> str:
    # New combined function with language support
    system = (
        "You are an analytical PostgreSQL generator. Your task is to perform two steps: "
        "1. **Plan:** Extract the analysis intent, target variables, and period for the user's question. "
        "2. **Generate SQL:** Write a single, correct PostgreSQL SELECT query to fulfill the plan. "
        "Rules: no INSERT/UPDATE/DELETE; no DDL; NO comments; NO markdown fences. "
        "Use only documented tables and columns. Prefer monthly aggregation. "
        "If USD prices are requested, prefer price_with_usd / tariff_with_usd views. "
        f"{lang_instruction}"
    )
    # Use selective domain knowledge to reduce tokens (30-40% savings)
    domain_json = get_relevant_domain_knowledge(user_query, use_cache=False)
    plan_format = {
        "intent": "trend_analysis" if analysis_mode == "analyst" else "general",
        "target": "<metric name>",
        "period": "YYYY-YYYY or YYYY-MM to YYYY-MM"
    }
    prompt = f"""
User question:
{user_query}
Schema:
{DB_SCHEMA_DOC}
Domain knowledge:
{domain_json}
Guidance:
- Use ONLY documented materialized views.
- For balancing-price analyses, differentiate Summer (Apr–Jul) vs Winter (Aug–Mar).
- Weighted-average balancing price = weighted by total balancing-market quantities (entities: deregulated_hydro, import, regulated_hpp, regulated_new_tpp, regulated_old_tpp, renewable_ppa, thermal_ppa).
CRITICAL - PRIMARY DRIVERS for balancing price analysis:
- Correlation policy - PRIORITY ORDER:
  * Targets: p_bal_gel, p_bal_usd
  * PRIMARY DRIVER #1: xrate (exchange rate) - MOST IMPORTANT for GEL/MWh price
    - Use xrate from price_with_usd view
    - Critical because gas and imports are USD-priced
  * PRIMARY DRIVER #2: Composition (shares) - CRITICAL for both GEL and USD prices
    - Calculate shares from trade_derived_entities
    - IMPORTANT: Use LOWER(REPLACE(segment, ' ', '_')) = 'balancing' for segment filter
    - Database may have 'Balancing Electricity', 'balancing', or other variants
    - Use share CTE pattern, no raw quantities
    - Higher cheap source shares (regulated HPP, deregulated hydro) → lower prices
    - Higher expensive source shares (import, thermal PPA, renewable PPA) → higher prices
  * Secondary drivers: tariffs in GEL only from tariff_with_usd for:
    - Enguri ('ltd "engurhesi"1')
    - Gardabani TPP ('ltd "gardabni thermal power plant"')
    - Old TPP group ('ltd "mtkvari energy"', 'ltd "iec" (tbilresi)', 'ltd "g power" (capital turbines)')
- Tariffs follow cost-plus methodology; thermal tariffs depend on gas price (USD) → correlated with xrate.
- When USD values appear, *_usd = *_gel / xrate.
- Aggregation default = monthly. for energy_balance_long_mv= yearly.
- Season is a derived dimension (not a column): use CASE WHEN EXTRACT(MONTH FROM date) IN (4,5,6,7) THEN 'Summer' ELSE 'Winter' END AS season to group data seasonally when user mentions 'season', 'summer', or 'winter'.
- Use these examples:
{FEW_SHOT_SQL}
Output Format:
Return a single string containing two parts, separated by '---SQL---'. The first part is a JSON object (the plan), and the second part is the raw SELECT statement.
Example Output:
{json.dumps(plan_format)}
---SQL---
SELECT ...
"""
    llm_start = time.time()
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        combined_output = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        metrics.log_llm_call(time.time() - llm_start)
    except Exception as e:
        log.warning(f"Combined generation failed: {e}")
        # Fallback to OpenAI if Gemini fails
        try:
            llm = make_openai()
            combined_output = llm.invoke([("system", system), ("user", prompt)]).content.strip()
            metrics.log_llm_call(time.time() - llm_start)
        except Exception as e_f:
             log.warning(f"Combined generation failed with fallback: {e_f}")
             metrics.log_error()
             raise e_f # Re-raise final exception
    return combined_output
# -----------------------------
# Data helpers (modified quick_stats)
# -----------------------------
def rows_to_preview(rows: List[Tuple], cols: List[str], max_rows: int = 200) -> str:
    if not rows:
        return "No rows returned."
    df = pd.DataFrame(rows[:max_rows], columns=cols)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float).round(3)
    return df.to_string(index=False)
def quick_stats(rows: List[Tuple], cols: List[str]) -> str:
    """Generate quick statistics for query results.
    Args:
        rows: List of tuples containing query results
        cols: List of column names
    Returns:
        String summary of statistics and trends
    """
    if not rows:
        return "0 rows."
    df = pd.DataFrame(rows, columns=cols).copy() # Protect original data
    numeric = df.select_dtypes(include=[np.number])
    out = [f"Rows: {len(df)}"]
   
    # 1. Detect date/year column
    date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower() or "month" in c.lower()]
    if not date_cols or numeric.empty:
        # Fallback to simple stats if no date or numeric data
        # ... (original logic for non-time series can stay here) ...
        return "\n".join(out)
    time_col = date_cols[0]
    # --- NEW TREND CALCULATION: Compare First Full Year vs Last Full Year ---
    try:
        # Ensure the time column is datetime, then extract year/month
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            # Attempt to coerce strings/objects to datetime
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        # Verify conversion worked before using .dt accessors
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            # Conversion failed, still object dtype - skip trend calculation
            log.warning(f"Column {time_col} could not be converted to datetime, skipping trend")
            return "\n".join(out)
        df['__year'] = df[time_col].dt.year
        # --- Detect and exclude incomplete final year ---
        # Applies to all time columns, even if already datetime
        try:
            df['_year_month'] = df[time_col].dt.to_period('M')
            # Count how many records (months) exist for each year
            months_per_year = df.groupby(df['_year_month'].dt.year).size().sort_index()
            # Mark as incomplete if less than 10 months (not a full year of data)
            incomplete_years = months_per_year[months_per_year < 10].index.tolist()
            if incomplete_years:
                log.info(f"Excluding incomplete years from trend calculation: {incomplete_years}")
                df = df[~df['__year'].isin(incomplete_years)]
        except Exception as e:
            log.warning(f"Failed to filter incomplete years: {e}")
       
        valid_years = df['__year'].dropna().unique()
        if len(valid_years) >= 2:
            first_full_year = int(valid_years.min())
            last_full_year = int(valid_years.max())
            # Ensure we are comparing two different years
            if first_full_year != last_full_year:
               
                # Filter data for the first and last full years
                df_first = df[df['__year'] == first_full_year]
                df_last = df[df['__year'] == last_full_year]
                # Get the mean of all numeric values for these years
                # Using .values.mean() to get single average across all values
                mean_first_year = df_first[numeric.columns].values.mean()
                mean_last_year = df_last[numeric.columns].values.mean()
               
                change = ((mean_last_year - mean_first_year) / mean_first_year * 100) if mean_first_year != 0 else 0
                trend = "increasing" if mean_last_year > mean_first_year else "decreasing"
                out.append(f"Trend (Yearly Avg, {first_full_year}→{last_full_year}): {trend} ({change:.1f}%)")
                # --- NEW: Seasonal split (Summer vs Winter) with CAGR ---
                try:
                    df['month'] = df[time_col].dt.month
                    summer_mask = df['month'].isin([4, 5, 6, 7])
                    winter_mask = ~summer_mask
                    def seasonal_avg(df_season, col, year):
                        return df_season.loc[df_season['__year'] == year, col].mean()
                    def seasonal_cagr(df_season, col):
                        """Compute CAGR (Compound Annual Growth Rate) for a column across years within a seasonal subset."""
                        df_y = df_season.groupby('__year')[col].mean().dropna()
                        if len(df_y) >= 2:
                            first, last = df_y.iloc[0], df_y.iloc[-1]
                            n = len(df_y) - 1
                            return ((last / first) ** (1 / n) - 1) * 100 if first > 0 else np.nan
                        return np.nan
                    for col in numeric.columns:
                        if 'p_bal' in col.lower() or 'price' in col.lower():
                            summer_first = seasonal_avg(df.loc[summer_mask], col, first_full_year)
                            summer_last = seasonal_avg(df.loc[summer_mask], col, last_full_year)
                            winter_first = seasonal_avg(df.loc[winter_mask], col, first_full_year)
                            winter_last = seasonal_avg(df.loc[winter_mask], col, last_full_year)
                            cagr_summer = seasonal_cagr(df.loc[summer_mask], col)
                            cagr_winter = seasonal_cagr(df.loc[winter_mask], col)
                            out.append(
                                f"Seasonal Trend ({col}): "
                                f"Summer {first_full_year}→{last_full_year}: "
                                f"{(summer_last - summer_first):.1f} Δ, CAGR {cagr_summer:.2f}%; "
                                f"Winter {first_full_year}→{last_full_year}: "
                                f"{(winter_last - winter_first):.1f} Δ, CAGR {cagr_winter:.2f}%."
                            )
                except Exception as e:
                    log.warning(f'Seasonal trend calculation failed: {e}')
           
            else:
                out.append("Trend: Less than one full year of data for comparison.")
        else:
            out.append("Trend: Insufficient data for yearly comparison.")
    except Exception as e:
        log.warning(f"Yearly trend calculation failed: {e}")
        # Fallback to original logic or just skip trend calculation
    # ... (Keep the date range display) ...
    first = df[time_col].min()
    last = df[time_col].max()
    out.append(f"Period: {first} → {last}")
 
    # ... (Keep the original stats: min, max, avg for numeric columns)
    for col in numeric.columns:
        try:
            mn = numeric[col].min()
            mx = numeric[col].max()
            avg = numeric[col].mean()
            out.append(f"{col}: min={mn:.3f}, max={mx:.3f}, avg={avg:.3f}")
        except Exception:
            pass
    return "\n".join(out)
# -----------------------------
# SQL Safety & Validation
# -----------------------------
def normalize_sql(sql: str) -> str:
    """Normalize SQL for synonym replacement and safety checks."""
    # Remove comments and excess whitespace
    sql = re.sub(r"--.*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql
def replace_synonyms(sql: str) -> str:
    """Replace table/column synonyms using pre-compiled regex patterns."""
    for pattern, replacement in SYNONYM_PATTERNS:
        sql = pattern.sub(replacement, sql)
    # Column-level synonym replacement (case-insensitive)
    for old, new in COLUMN_SYNONYMS.items():
        sql = re.sub(rf"\b{old}\b", new, sql, flags=re.IGNORECASE)
    return sql
def is_safe_sql(sql: str) -> bool:
    """Basic safety check: allow only SELECT, WITH, CTEs, and safe functions."""
    sql_upper = sql.upper()
    banned = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "GRANT", "REVOKE"]
    if any(kw in sql_upper for kw in banned):
        return False
    # Allow only SELECT starting queries
    if not sql_upper.lstrip().startswith("WITH") and not sql_upper.lstrip().startswith("SELECT"):
        return False
    return True
def validate_sql_ast(sql: str) -> bool:
    """Use sqlglot to parse and validate SQL structure."""
    try:
        parsed = parse_one(sql, dialect="postgres")
        # Ensure only SELECT or WITH (CTE) at top level
        if not isinstance(parsed, (exp.Select, exp.With)):
            return False
        # Traverse and ensure no DDL/DML
        for node in parsed.walk():
            if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop)):
                return False
        return True
    except Exception as e:
        log.warning(f"SQL parsing failed: {e}")
        return False
def enforce_limit(sql: str, force: bool = False) -> str:
    """Add LIMIT if missing and either force=True or result might be large."""
    if LIMIT_PATTERN.search(sql):
        return sql
    if force or "trade_derived_entities" in sql.lower() or "tech_quantity_view" in sql.lower():
        return f"{sql.strip()} LIMIT {MAX_ROWS};"
    return sql
# -----------------------------
# Main Execution Path
# -----------------------------
def execute_safe_sql(conn, sql: str) -> Tuple[pd.DataFrame, str]:
    """Execute SQL with full safety pipeline and return DataFrame + preview."""
    start = time.time()
    try:
        # 1. Normalize
        sql = normalize_sql(sql)
        # 2. Replace synonyms
        sql = replace_synonyms(sql)
        # 3. Inject balancing pivot if needed
        if should_inject_balancing_pivot(user_query="", sql=sql):
            sql = build_trade_share_cte(sql)
        # 4. Enforce LIMIT
        sql = enforce_limit(sql, force=True)
        # 5. Final safety
        if not is_safe_sql(sql) or not validate_sql_ast(sql):
            raise ValueError("Unsafe SQL detected")
        log.info(f"Executing SQL:\n{sql}")
        res = conn.execute(text(sql))
        rows = res.fetchall()
        cols = list(res.keys())
        df = pd.DataFrame(rows, columns=[str(c) for c in cols])
        duration = time.time() - start
        metrics.log_sql_query(duration)
        preview = rows_to_preview(rows, cols)
        return df, preview
    except Exception as e:
        metrics.log_error()
        log.error(f"SQL execution failed: {e}\nSQL: {sql}")
        raise
# -----------------------------
# LLM Answer Generation
# -----------------------------
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
def llm_generate_answer(
    user_query: str,
    df: pd.DataFrame,
    sql_preview: str,
    plan: dict,
    lang_instruction: str
) -> str:
    """Generate final answer using LLM with data context."""
    system = (
        "You are an energy analyst. Answer the user's question using the provided data and SQL results. "
        "Be concise, factual, and professional. Use the detected language. "
        "If share data is available, prefer deterministic summary over LLM hallucination. "
        f"{lang_instruction}"
    )
    # Try deterministic share summary first
    share_summary = generate_share_summary(df, plan, user_query)
    if share_summary:
        return share_summary
    # Fallback to LLM with data
    stats = quick_stats(df.values.tolist(), df.columns.tolist()) if not df.empty else "No data."
    domain_json = get_relevant_domain_knowledge(user_query)
    prompt = f"""
User question: {user_query}
Plan: {json.dumps(plan)}
SQL Preview (first 5 rows):
{sql_preview}
Quick Stats:
{stats}
Domain Knowledge:
{domain_json}
Answer directly and concisely.
"""
    llm_start = time.time()
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        answer = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        metrics.log_llm_call(time.time() - llm_start)
        return answer
    except Exception as e:
        log.warning(f"Answer generation failed: {e}")
        metrics.log_error()
        return f"Error generating answer: {str(e)}"
# -----------------------------
# Chart Generation
# -----------------------------
def generate_chart_data(df: pd.DataFrame, user_query: str) -> Optional[dict]:
    """Generate chart configuration if appropriate."""
    if df.empty or len(df) < 3:
        return None
    # Detect time series
    date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower() or "month" in c.lower()]
    if not date_cols:
        return None
    date_col = date_cols[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None
    # Simple line chart
    chart_type = "line"
    chart_data = {
        "labels": df[date_col].astype(str).tolist(),
        "datasets": [
            {"label": col, "data": df[col].round(3).tolist()}
            for col in numeric_cols[:5]  # Limit to 5 lines
        ]
    }
    return {
        "chart_type": chart_type,
        "chart_data": chart_data,
        "chart_metadata": {"title": user_query[:100]}
    }
# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/ask", response_model=APIResponse)
async def ask_post(question: Question, request: Request, authorization: Optional[str] = Header(None)):
    start_time = time.time()
    request_id = request_id_var.get()
    log.info(f"[{request_id}] Processing query: {question.query}")
    if authorization != f"Bearer {APP_SECRET_KEY}":
        metrics.log_error()
        raise HTTPException(status_code=401, detail="Invalid API key")
    # Language detection
    lang_code = detect_language(question.query)
    lang_instruction = get_language_instruction(lang_code)
    # Mode detection
    analysis_mode = detect_analysis_mode(question.query)
    try:
        with ENGINE.connect() as conn:
            # Step 1: Generate plan + SQL
            combined_output = llm_generate_plan_and_sql(
                user_query=question.query,
                analysis_mode=analysis_mode,
                lang_instruction=lang_instruction
            )
            plan_str, sql = combined_output.split("---SQL---", 1)
            plan = json.loads(plan_str.strip())
            sql = sql.strip().rstrip(";")
            # Step 2: Execute SQL
            df, sql_preview = execute_safe_sql(conn, sql)
            # Step 3: Ensure share data if needed
            df, pivot_used = ensure_share_dataframe(df, conn)
            if pivot_used:
                log.info(f"[{request_id}] Injected deterministic balancing pivot")
            # Step 4: Generate answer
            answer = llm_generate_answer(
                user_query=question.query,
                df=df,
                sql_preview=sql_preview,
                plan=plan,
                lang_instruction=lang_instruction
            )
            # Step 5: Chart?
            chart = None
            if should_generate_chart(question.query, len(df)):
                chart = generate_chart_data(df, question.query)
            total_time = time.time() - start_time
            metrics.log_request(total_time)
            log.info(f"[{request_id}] Completed in {total_time:.2f}s")
            return APIResponse(
                answer=answer,
                chart_data=chart["chart_data"] if chart else None,
                chart_type=chart["chart_type"] if chart else None,
                chart_metadata=chart["chart_metadata"] if chart else None,
                execution_time=round(total_time, 2)
            )
    except Exception as e:
        metrics.log_error()
        log.error(f"[{request_id}] Fatal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# -----------------------------
# Health & Metrics
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_TYPE}
@app.get("/metrics")
async def get_metrics():
    return metrics.get_stats()
# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
async def startup_event():
    log.info("EnerBot Analyst API started")
    log.info(f"Model: {MODEL_TYPE.upper()} ({GEMINI_MODEL if MODEL_TYPE == 'gemini' else OPENAI_MODEL})")
    log.info(f"Database: {DB_URL.split('@')[1] if '@' in DB_URL else 'connected'}")
    log.info(f"Allowed tables: {sorted(ALLOWED_TABLES)}")
