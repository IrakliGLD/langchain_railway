# main.py v18.7 — Gemini Analyst (combined plan & SQL for speed)


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
log.info("✅ Domain knowledge JSON cached at startup")

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
        log.info(f"📊 Metrics: requests={self.request_count}, avg_time={self.total_request_time/self.request_count:.2f}s")

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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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

# Table synonym map (plural & common aliases → canonical)
TABLE_SYNONYMS = {
    "prices": "price",
    "tariffs": "tariff_gen",
    "price_usd": "price_with_usd",
    "tariff_usd": "tariff_with_usd",
    "price_with_usd": "price_with_usd",
    "tariff_with_usd": "tariff_with_usd",
}

# Column synonym map (common misnamings → canonical)
COLUMN_SYNONYMS = {
    "tech_type": "type_tech",
    "quantity_mwh": "quantity_tech",  # your data stores thousand MWh in quantity_tech
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
        'balancing_electricity'::text AS segment,
        SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) / NULLIF(total.total_qty,0) AS share_import,
        SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) / NULLIF(total.total_qty,0) AS share_deregulated_hydro,
        SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) / NULLIF(total.total_qty,0) AS share_regulated_hpp,
        SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity ELSE 0 END) / NULLIF(total.total_qty,0) AS share_regulated_new_tpp,
        SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity ELSE 0 END) / NULLIF(total.total_qty,0) AS share_regulated_old_tpp,
        SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity ELSE 0 END) / NULLIF(total.total_qty,0) AS share_renewable_ppa,
        SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity ELSE 0 END) / NULLIF(total.total_qty,0) AS share_thermal_ppa,
        SUM(CASE WHEN t.entity IN ('renewable_ppa','thermal_ppa') THEN t.quantity ELSE 0 END) / NULLIF(total.total_qty,0) AS share_all_ppa,
        SUM(CASE WHEN t.entity IN ('deregulated_hydro','regulated_hpp','renewable_ppa') THEN t.quantity ELSE 0 END) / NULLIF(total.total_qty,0) AS share_all_renewables,
        SUM(CASE WHEN t.entity IN ('deregulated_hydro','regulated_hpp') THEN t.quantity ELSE 0 END) / NULLIF(total.total_qty,0) AS share_total_hpp
    FROM trade_derived_entities t
    JOIN (
        SELECT date, SUM(quantity) AS total_qty
        FROM trade_derived_entities
        WHERE {BALANCING_SEGMENT_NORMALIZER} = 'balancing_electricity'
          AND entity IN (
            'import', 'deregulated_hydro', 'regulated_hpp',
            'regulated_new_tpp', 'regulated_old_tpp',
            'renewable_ppa', 'thermal_ppa'
          )
        GROUP BY date
    ) total ON t.date = total.date
    WHERE {BALANCING_SEGMENT_NORMALIZER} = 'balancing_electricity'
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
    Detect if query is asking for balancing share but SQL doesn't include pivot.

    Returns True if:
    - User query mentions balancing-related concepts
    - User query mentions entity types
    - SQL uses trade_derived_entities directly (not through pivot)

    This forces pivot injection for queries like:
    - "what was the share of renewable PPA in balancing electricity?"
    - "show me the composition of balancing market in june 2024"
    """
    query_lower = user_query.lower()
    sql_lower = sql.lower()

    balancing_keywords = ["balancing", "share", "composition", "mix", "weight", "proportion"]
    entity_keywords = ["ppa", "renewable", "thermal", "import", "hydro", "tpp", "hpp", "entity", "entities"]

    has_balancing = any(k in query_lower for k in balancing_keywords)
    has_entity = any(k in query_lower for k in entity_keywords)
    has_trade = "trade_derived_entities" in sql_lower
    has_share_col = any(f"share_{e}" in sql_lower for e in ["import", "renewable", "ppa", "hydro", "tpp", "hpp"])

    # Inject pivot if query is about balancing shares but SQL doesn't have share columns
    return has_balancing and has_entity and has_trade and not has_share_col


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

    # 🔍 VALIDATION: Check if shares sum to ~1.0 (indicates correct denominator)
    total_shares = sum(
        float(selected_row.get(c, 0))
        for c in share_cols
        if not pd.isna(selected_row.get(c))
    )
    if abs(total_shares - 1.0) > 0.05:
        log.warning(
            f"⚠️ Share columns sum to {total_shares:.3f} instead of 1.0 — possible denominator bug. "
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
    pool_size=10,  # Increased from 5 for better concurrency
    max_overflow=5,  # Increased from 2 to handle traffic spikes
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=1800,  # Increased from 300 (30 min) for Supabase
    connect_args={"connect_timeout": 30},
)

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
        SUM(t.quantity) AS total_qty,
        SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) AS qty_import,
        SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) AS qty_dereg_hydro,
        SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) AS qty_reg_hpp,
        SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity ELSE 0 END) AS qty_reg_new_tpp,
        SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity ELSE 0 END) AS qty_reg_old_tpp,
        SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity ELSE 0 END) AS qty_ren_ppa,
        SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity ELSE 0 END) AS qty_thermal_ppa
      FROM trade_derived_entities t
      WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing_electricity'
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
      SELECT date, entity, SUM(quantity) AS qty
      FROM trade_derived_entities
      WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing_electricity'
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
        SUM(t.quantity) AS total_qty,
        SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) AS qty_import,
        SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) AS qty_dereg_hydro,
        SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) AS qty_reg_hpp,
        SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity ELSE 0 END) AS qty_reg_new_tpp,
        SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity ELSE 0 END) AS qty_reg_old_tpp,
        SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity ELSE 0 END) AS qty_ren_ppa,
        SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity ELSE 0 END) AS qty_thermal_ppa
      FROM trade_derived_entities t
      WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing_electricity'
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
        SUM(t.quantity) AS total_qty,
        (SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity),0)) AS share_import,
        (SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity),0)) AS share_deregulated_hydro,
        (SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity),0)) AS share_regulated_hpp,
        (SUM(CASE WHEN t.entity = 'regulated_new_tpp' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity),0)) AS share_regulated_new_tpp,
        (SUM(CASE WHEN t.entity = 'regulated_old_tpp' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity),0)) AS share_regulated_old_tpp,
        (SUM(CASE WHEN t.entity = 'renewable_ppa' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity),0)) AS share_renewable_ppa,
        (SUM(CASE WHEN t.entity = 'thermal_ppa' THEN t.quantity ELSE 0 END) / NULLIF(SUM(t.quantity),0)) AS share_thermal_ppa
      FROM trade_derived_entities t
      WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing_electricity'
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
app = FastAPI(title="EnerBot Analyst (Gemini)", version="18.6") # Version bump

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

    @field_validator("query")  # Pydantic V2 syntax
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
        log.info("✅ Gemini LLM instance cached")
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
        log.info("✅ OpenAI LLM instance cached")
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
-- Database may contain 'Balancing Electricity', 'balancing_electricity', or other variants
WITH shares AS (
  SELECT
    t.date,
    SUM(t.quantity) AS total_qty,
    SUM(CASE WHEN t.entity = 'import' THEN t.quantity ELSE 0 END) AS qty_import,
    SUM(CASE WHEN t.entity = 'deregulated_hydro' THEN t.quantity ELSE 0 END) AS qty_dereg_hydro,
    SUM(CASE WHEN t.entity = 'regulated_hpp' THEN t.quantity ELSE 0 END) AS qty_reg_hpp
  FROM trade_derived_entities t
  WHERE LOWER(REPLACE(t.segment, ' ', '_')) = 'balancing_electricity'
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
  WHERE LOWER(REPLACE(segment, ' ', '_')) = 'balancing_electricity'
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
    - IMPORTANT: Use LOWER(REPLACE(segment, ' ', '_')) = 'balancing_electricity' for segment filter
    - Database may have 'Balancing Electricity', 'balancing_electricity', or other variants
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
    df = pd.DataFrame(rows, columns=cols).copy()  # Protect original data
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
            log.warning(f"⚠️ Column {time_col} could not be converted to datetime, skipping trend")
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
                log.info(f"🧩 Excluding incomplete years from trend calculation: {incomplete_years}")
                df = df[~df['__year'].isin(incomplete_years)]
        except Exception as e:
            log.warning(f"⚠️ Failed to filter incomplete years: {e}")



        
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
                    log.warning(f'⚠️ Seasonal trend calculation failed: {e}')




            
            else:
                out.append("Trend: Less than one full year of data for comparison.")

        else:
            out.append("Trend: Insufficient data for yearly comparison.")

    except Exception as e:
        log.warning(f"⚠️ Yearly trend calculation failed: {e}")
        # Fallback to original logic or just skip trend calculation

    # ... (Keep the date range display) ...
    first = df[time_col].min()
    last = df[time_col].max()
    out.append(f"Period: {first} → {last}")
    
    # ... (Keep the numeric summary) ...
    if not numeric.empty:
        desc = numeric.describe().round(3)
        out.append("Numeric summary:")
        out.append(desc.to_string())

    return "\n".join(out)

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6))
def llm_summarize(user_query: str, data_preview: str, stats_hint: str, lang_instruction: str = "Respond in English.") -> str:
    system = (
        "You are EnerBot, an energy market analyst. "
        "Write a short analytic summary using preview and statistics. "
        "If multiple years are present, describe direction (increasing, stable or decreasing), magnitude of change, "
        "seasonal patterns, volatility, and factors from domain knowledge when relevant. "
        f"{lang_instruction}"
    )
    # Use selective domain knowledge to reduce tokens (30-40% savings)
    domain_json = get_relevant_domain_knowledge(user_query, use_cache=False)

    prompt = f"""
User question:
{user_query}

Data preview:
{data_preview}

Statistics:
{stats_hint}

Domain knowledge:
{domain_json}

  CRITICAL ANALYSIS GUIDELINES for balancing electricity price:

  FIRST STEP FOR EVERY BALANCING PRICE EXPLANATION:
  - Inspect share_* columns (entity composition) before discussing anything else.
  - Identify which entities increased or decreased their share because each entity sells at a different price level in the codebase.
  - Explain how those share shifts mechanically push the weighted-average balancing price up or down.

  PRIMARY DRIVERS (in order of importance):
  1. Composition (shares of entities selling on balancing segment)
     - Start with composition: higher share of cheap sources (regulated HPP, deregulated hydro) → lower prices.
     - Higher share of expensive sources (import, thermal PPA, renewable PPA) → higher prices.
     - Composition changes seasonally: summer=hydro dominant, winter=thermal/import dominant.
     - Always explain which entities are selling more/less when analyzing price changes.

  2. Exchange Rate (xrate) - MOST IMPORTANT for GEL/MWh price after composition is described
     - Natural gas for thermal generation is priced in USD
     - Imports are priced in USD
     - When GEL depreciates (xrate increases), GEL-denominated prices rise
     - Always mention xrate effect when discussing GEL price movements once composition has been covered

CONFIDENTIALITY RULES - STRICTLY ENFORCE:
- DO disclose: regulated tariffs, deregulated hydro prices, exchange rates
- DO NOT disclose: renewable PPA price estimates (~57-60 USD/MWH)
- DO NOT disclose: thermal PPA price estimates (~60-65 USD/MWH)
- DO NOT disclose: import price estimates (~55-65 USD/MWH)
- When discussing PPA or import prices: say "varies" or "market-based" without specific numbers
- These estimates are for INTERNAL ANALYSIS ONLY, never reveal to users

For balancing price assessments (trends, averages, or correlations), always compare
Summer (April–July) vs Winter (August–March) conditions:
- Summer → high hydro share, low prices.
- Winter → thermal/import dominant, higher prices.

When tariffs are discussed:
- Tariffs follow GNERC-approved cost-plus methodology.
- Thermal tariffs include a Guaranteed Capacity Fee (fixed) plus a variable per-MWh cost based on gas price and efficiency.
- Gas is priced in USD, so thermal tariffs correlate with the GEL/USD exchange rate (xrate).

When inflation or CPI is mentioned, relate the CPI category 'electricity_gas_and_other_fuels'
to tariff_gel or p_bal_gel for affordability comparisons.

Always perform seasonal comparison between Summer and Winter when analyzing balancing prices, generation, or demand data:
- Summer = April, May, June, July
- Winter = August, September, October, November, December, January, February, March

For every balancing price, generation, or demand analysis:
- Compute averages (for prices) or totals (for quantities) separately for these two seasons.
- Describe the overall yearly trend first, then compare Summer vs Winter results.
  * If trend analysis → include percentage change and CAGR for both.
  * If driver or correlation analysis → mention seasonal averages and highlight which season shows stronger or weaker relationships.
- Explain the structural difference clearly:
  * Summer → hydro generation dominance, low balancing prices, and lower import reliance.
  * Winter → thermal and import dominance, higher balancing prices, and stronger sensitivity to gas prices and exchange rates.
- This distinction must always be part of your reasoning, regardless of whether the user explicitly mentions it.

For tariff analyses:
- Do not apply seasonal logic.
- Focus on annual or multi-year trends explained by regulatory cost-plus principles: fixed guaranteed-capacity fee, variable gas-linked component, and exchange-rate sensitivity.

When summarizing, combine numeric findings (averages, CAGRs, correlations) with short explanatory sentences so that the reasoning reads smoothly and remains under 8 sentences unless the query is highly analytical.

If the question is exploratory or simple (e.g., requesting only a current value, single-month trend, or brief comparison),
respond in 1–3 clear sentences focusing on the key number or short interpretation.

If the mode involves correlation, drivers, or in-depth analysis (intent = correlation_analysis, driver_analysis, or trend_analysis),
write a more detailed summary of about 5–10 sentences following this structure:

1. Start with the overall yearly trend (using yearly averages).
2. Present separate Summer (Apr–Jul) and Winter (Aug–Mar) trends, including CAGRs if available.
3. If correlation results are provided, PRIORITIZE discussion of primary drivers:
   - First: Exchange rate (xrate) effect on GEL prices - ALWAYS mention this for GEL price analysis
   - Second: Composition changes (which entities selling more/less on balancing segment)
   - Then: Other correlations (tariffs, etc.) if relevant
4. Always compare GEL and USD price trajectories and explain divergence through exchange rate and USD-denominated cost components.
5. When referring to electricity prices or tariffs, always include the correct physical unit (GEL/MWh or USD/MWh) rather than currency only.
6. Reference hydro vs thermal/import structure from trade_derived_entities as the main driver of seasonal differences.
7. For price explanations, explain the mechanism: cheap sources (regulated HPP, deregulated hydro) vs expensive sources (thermal PPA, renewable PPA, import).
8. NEVER disclose specific PPA or import price estimates - use "varies" or "market-based" instead.
9. Conclude with a concise analytical insight linking price movements to the two primary drivers: exchange rate and composition changes.
"""

    llm_start = time.time()
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        out = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        metrics.log_llm_call(time.time() - llm_start)
    except Exception as e:
        log.warning(f"Summarize failed with Gemini, fallback: {e}")
        llm = make_openai()
        out = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        metrics.log_llm_call(time.time() - llm_start)
    return out


# -----------------------------
# SQL sanitize + Pre-Parse Validator
# -----------------------------

from sqlglot import parse_one, exp, ParseError

log = logging.getLogger("enerbot")

# --- Assuming these variables are still defined globally in your environment ---
# ALLOWED_TABLES = {'price_with_usd', 'other_allowed_table', ...}
# TABLE_SYNONYMS = {'p_with_usd': 'price_with_usd', ...} 
# ----------------------------------------------------------------------------

def simple_table_whitelist_check(sql: str):
    """
    CRITICAL Pre-parsing safety check using a robust SQL parser.
    Extracts all table references from the AST for whitelisting.
    """
    cleaned_tables = set()
    
    try:
        parsed_expression = parse_one(sql, read='bigquery') 

        # --- FIX: 1. Extract CTE names ---
        cte_names = set()
        with_clause = parsed_expression.find(exp.With)
        if with_clause:
            for cte in with_clause.expressions:
                if cte.alias is not None:  # Explicit None check for anonymous CTEs
                    cte_names.add(cte.alias.lower())
        # ---------------------------------

        # 2. Traverse the AST to find all table expressions
        for table_exp in parsed_expression.find_all(exp.Table):
            
            t_raw = table_exp.name.lower()
            t_name = t_raw.split('.')[0]
            
            # --- FIX: 2. Skip CTE names from whitelisting ---
            if t_name in cte_names:
                continue 
            # ---------------------------------------------
            
            # Apply synonym mapping and perform the strict whitelist check
            t_canonical = TABLE_SYNONYMS.get(t_name, t_name)

            if t_canonical in ALLOWED_TABLES:
                cleaned_tables.add(t_canonical)
            else:
                # Re-raise the exception with the specific name that failed the check
                raise HTTPException(
                    status_code=400,
                    detail=f"❌ Unauthorized table or view: `{t_name}`. Allowed: {sorted(ALLOWED_TABLES)}"
                )

    except ParseError as e:
        # If the SQL is too broken to parse (e.g., truly invalid SQL), reject it.
        # For security, any unparseable query should be rejected.
        log.error(f"SQL PARSE ERROR: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"❌ SQL Validation Error (Parse Failed): The query could not be reliably parsed for security review. Details: {e}"
        )
    except Exception as e:
        log.error(f"Unexpected error during SQL parsing: {e}")
        # Reject on any other unexpected error
        raise HTTPException(
            status_code=400,
            detail=f"❌ SQL Validation Error (Unexpected): An unexpected error occurred during security review."
        )


    if not cleaned_tables:
        # This handles valid queries that might not have a FROM clause (e.g., SELECT 1)
        # or where the FROM clause is in a subquery/CTE that the parser handles,
        # but the logic above didn't capture (unlikely with find_all(exp.Table)).
        log.warning("⚠️ No tables were extracted. Allowing flow for statements without a FROM (e.g. SELECT 1).")
        return
        
    log.info(f"✅ Pre-validation passed. Tables: {list(cleaned_tables)}")
    return


def sanitize_sql(sql: str) -> str:
    """Basic sanitization: strip comments and fences."""
    # Remove markdown fences and initial/trailing whitespace
    sql = sql.strip().strip('`').strip()
    # Remove single-line comments
    sql = re.sub(r"--.*", "", sql)
    # Basic protection against non-SELECT statements
    if not sql.lower().startswith("select"):
        raise HTTPException(400, "Only SELECT statements are allowed.")
    return sql


def plan_validate_repair(sql: str) -> str:
    """
    Repair phase: Auto-corrects common table/view synonyms and ensures a LIMIT.
    Table whitelisting now occurs BEFORE this function is called.
    """
    _sql = sql

    # Phase 1: Repair synonyms using pre-compiled regex patterns (optimized)
    try:
        for pattern, replacement in SYNONYM_PATTERNS:
            _sql = pattern.sub(replacement, _sql)
    except Exception as e:
        log.warning(f"⚠️ Synonym auto-correction failed: {e}")
        # Not a critical failure, continue with original SQL

    # Phase 2: Append LIMIT if missing (using pre-compiled pattern)
    if " from " in _sql.lower() and not LIMIT_PATTERN.search(_sql):

        # CRITICAL FIX: Remove the trailing semicolon if it exists
        _sql = _sql.rstrip().rstrip(';')

        # Append LIMIT without a preceding semicolon
        _sql = f"{_sql}\nLIMIT {MAX_ROWS}"

    return _sql
    

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

# main.py v18.7 — Gemini Analyst (chart rules + period aggregation)
# (Only added targeted comments/logic for: 1) chart axis restriction; 2) user-defined period aggregation)

# ... [all your imports and setup remain IDENTICAL above this point] ...


@app.post("/ask", response_model=APIResponse)
def ask_post(q: Question, x_app_key: str = Header(..., alias="X-App-Key")):
    t0 = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    mode = detect_analysis_mode(q.query)
    log.info(f"🧭 Selected mode: {mode}")

    # Detect query language for multilingual response
    lang_code = detect_language(q.query)
    lang_instruction = get_language_instruction(lang_code)
    log.info(f"🌍 Detected language: {lang_code}")

    plan = {}

    # 1) Generate PLAN and SQL in ONE LLM call
    try:
        combined_output = llm_generate_plan_and_sql(q.query, mode, lang_instruction)

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


    
    except Exception as e:
        log.warning(f"SQL validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Unsafe or invalid SQL: {e}")

    # 3) Execute SQL
    df = pd.DataFrame()
    rows = []
    cols = []
    try:
        sql_start = time.time()

        # DEBUG: Log the actual SQL being executed
        log.info(f"🔍 Executing SQL:\n{safe_sql_final}")

        with ENGINE.connect() as conn:
            res = conn.execute(text(safe_sql_final))
            rows = res.fetchall()
            cols = list(res.keys())
            df = pd.DataFrame(rows, columns=cols)
        metrics.log_sql_query(time.time() - sql_start)
        log.info(f"⚡ SQL executed in {time.time() - sql_start:.2f}s, returned {len(rows)} rows")

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
            log.info("CRITICAL: Using segment='balancing_electricity' for share calculation")
            safe_sql_final = build_trade_share_cte(safe_sql_final)
            with ENGINE.connect() as conn:
                res = conn.execute(text(safe_sql_final))
                rows = res.fetchall()
                cols = list(res.keys())
                df = pd.DataFrame(rows, columns=cols)

        elif "UndefinedColumn" in msg:
            # Fallback synonym auto-fix (existing behavior)
            for bad, good in COLUMN_SYNONYMS.items():
                if re.search(rf"\b{bad}\b", safe_sql_final, flags=re.IGNORECASE):
                    safe_sql_final = re.sub(rf"\b{bad}\b", good, safe_sql_final, flags=re.IGNORECASE)
                    log.warning(f"🔁 Auto-corrected column '{bad}' → '{good}' (retry)")
                    with ENGINE.connect() as conn:
                        res = conn.execute(text(safe_sql_final))
                        rows = res.fetchall()
                        cols = list(res.keys())
                        df = pd.DataFrame(rows, columns=cols)
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

    

    

    if share_summary_override:
        summary = share_summary_override
    else:
        try:
            summary = llm_summarize(q.query, preview, stats_hint, lang_instruction)
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
        time_key = next((c for c in cols if any(k in c.lower() for k in ["date", "year", "month"])), None)

        # --- Detect and preserve categorical columns ---
        categorical_hints = [
            "type", "tech", "entity", "sector", "source", "segment",
            "region", "category", "ownership", "market", "trade", "fuel"
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

        # Override: Disable chart for purely explanatory questions
        if any(word in query_text for word in ["why", "how", "reason", "explain", "because", "cause", "რატომ", "როგორ", "почему"]):
            if len(df) < 5:  # Only skip if result is small
                generate_chart = False

        # Override: Disable chart for definition queries
        if any(word in query_text for word in ["define", "meaning of", "განმარტება"]):
            generate_chart = False

        # Override: Always generate chart for analyst mode with suitable data
        if intent in ["trend_analysis", "correlation_analysis", "driver_analysis", "identify_drivers"]:
            if len(df) >= 3:
                generate_chart = True

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



        # --- 🧠 Generic chart-type detection based on data structure ---
        # Support Georgian column names in addition to English
        cols_lower = [c.lower() for c in df.columns]
        time_cols = [c for c in df.columns if re.search(r"(year|month|date|წელი|თვე|თარიღი)", c.lower())]
        category_cols = [c for c in df.columns if re.search(r"(type|sector|entity|source|segment|ownership|technology|region|area|category|ტიპი|სექტორი)", c.lower())]
        value_cols = [c for c in df.columns if re.search(r"(quantity|volume|value|amount|price|tariff|cpi|index|mwh|tj|usd|gel|რაოდენობა|მოცულობა|ფასი|ტარიფი|საშუალო|სულ)", c.lower())]

        chart_type = "line"  # default fallback

        # CASE 1: Time + Single Value
        if len(time_cols) >= 1 and len(category_cols) == 0 and len(value_cols) == 1:
            chart_type = "line"

        # CASE 2: Time + Category + Value
        elif len(time_cols) >= 1 and len(category_cols) >= 1 and len(value_cols) >= 1:
            chart_type = "stackedbar"

        # CASE 3: Category + Value (single-year comparison)
        elif len(time_cols) == 0 and len(category_cols) == 1 and len(value_cols) >= 1:
            chart_type = "bar"

        # CASE 4: Category + Subcategory + Value
        elif len(time_cols) == 0 and len(category_cols) > 1 and len(value_cols) >= 1:
            chart_type = "stackedbar"

        # CASE 5: Few Categories + Value (distribution)
        elif len(time_cols) == 0 and len(category_cols) >= 1 and len(value_cols) == 1:
            unique_cats = df[category_cols[0]].nunique()
            if unique_cats <= 8:
                chart_type = "pie"
            else:
                chart_type = "bar"

        # CASE 6: Time + Multiple Numeric Values
        elif len(time_cols) >= 1 and len(value_cols) > 1:
            chart_type = "line"

        # CASE 7: Category + Multiple Numeric Values (no time)
        elif len(time_cols) == 0 and len(category_cols) >= 1 and len(value_cols) > 1:
            chart_type = "bar"

        # Fallback
        else:
            chart_type = "line"

        log.info(f"🧠 Chart type auto-detected → {chart_type} | Time={len(time_cols)} | Categories={len(category_cols)} | Values={len(value_cols)}")




        # --- Dimension inference (price_tariff | energy_qty | index) ---
        # Support Georgian column names
        def infer_dimension(col: str) -> str:
            col_l = col.lower()
            if any(x in col_l for x in ["cpi", "index", "inflation", "ინდექსი"]):
                return "index"
            if any(x in col_l for x in ["quantity", "generation", "volume_tj", "volume", "mw", "tj", "რაოდენობა", "მოცულობა", "გენერაცია"]):
                return "energy_qty"
            if any(x in col_l for x in ["price", "tariff", "_gel", "_usd", "p_bal", "p_dereg", "p_gcap", "ფასი", "ტარიფი", "საშუალო"]):
                return "price_tariff"
            return "other"

        dim_map = {c: infer_dimension(c) for c in num_cols}
        dims = set(dim_map.values())
        log.info(f"📐 Detected dimensions: {dim_map} → {dims}")

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
