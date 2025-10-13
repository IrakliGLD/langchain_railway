# main.py v18.6 — Gemini Analyst (combined plan & SQL for speed)

import os
import re
import json
import time
import logging
import urllib.parse
from typing import Optional, Dict, Any, List, Tuple
from difflib import get_close_matches

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
# Corrected Pydantic imports for V2 compatibility
from pydantic import BaseModel, Field, field_validator 

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

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
from context import DB_SCHEMA_DOC, scrub_schema_mentions, COLUMN_LABELS
# Domain knowledge
from domain_knowledge import DOMAIN_KNOWLEDGE

# -----------------------------
# Boot + Config
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enerbot")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

MODEL_TYPE = (os.getenv("MODEL_TYPE", "gemini") or "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not SUPABASE_DB_URL:
    raise RuntimeError("Missing SUPABASE_DB_URL")
if not APP_SECRET_KEY:
    raise RuntimeError("Missing APP_SECRET_KEY")
if MODEL_TYPE == "gemini" and not GOOGLE_API_KEY:
    raise RuntimeError("MODEL_TYPE=gemini but GOOGLE_API_KEY is missing")

# Allow the base tables + USD materialized views
ALLOWED_TABLES = {
    "dates_mv",
    "energy_balance_long_mv",
    "entities_mv",
    "monthly_cpi_mv",
    "price_with_usd",
    "tariff_with_usd",
    "tech_quantity_view",
    "trade_derived_entities",
}

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
    pool_size=5,
    max_overflow=2,
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=300,
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
        ALLOWED_TABLES = set()


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="EnerBot Analyst (Gemini)", version="18.6") # Version bump
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
def make_gemini() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)

def make_openai() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set (fallback needed)")
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

def detect_analysis_mode(user_query: str) -> str:
    analytical_keywords = [
        "trend", "change", "growth", "increase", "decrease", "compare", "impact",
        "volatility", "pattern", "season", "relationship", "correlation", "evolution"
    ]
    for kw in analytical_keywords:
        if kw in user_query.lower():
            return "analyst"
    return "light"

# ------------------------------------------------------------------
# REMOVED: llm_plan_analysis - Combined into llm_generate_plan_and_sql
# ------------------------------------------------------------------

FEW_SHOT_SQL = """
-- Example 1: Monthly average balancing price in USD for 2023 (use materialized view)
SELECT
  EXTRACT(YEAR FROM date) AS year,
  EXTRACT(MONTH FROM date) AS month,
  AVG(p_bal_usd) AS avg_balancing_usd
FROM price_with_usd
WHERE EXTRACT(YEAR FROM date) = 2023
GROUP BY 1,2
ORDER BY 1,2
LIMIT 750;

-- Example 2: Single month balancing price (USD) for May 2024
SELECT p_bal_usd
FROM price_with_usd
WHERE date = '2024-05-01'
LIMIT 500;

-- Example 3: Generation (thousand MWh) by technology per month
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  type_tech,
  SUM(quantity_tech) AS qty_thousand_mwh
FROM tech_quantity_view
GROUP BY 1,2
ORDER BY 1,2
LIMIT 500;

-- Example 4: Average regulated tariffs (USD) by entity for 2024
SELECT
  entity,
  AVG(tariff_usd) AS avg_tariff_usd_2024
FROM tariff_with_usd
WHERE EXTRACT(YEAR FROM date) = 2024
GROUP BY entity
ORDER BY entity
LIMIT 500;

-- Example 5: CPI monthly values for electricity fuels category
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  cpi
FROM monthly_cpi_mv
WHERE cpi_type = 'electricity_gas_and_other_fuels'
ORDER BY date
LIMIT 500;

-- Example 6: Monthly data for Balancing Price (GEL) and Shares of key sources (Hydro, Import) for correlation analysis
SELECT
  TO_CHAR(t1.date, 'YYYY-MM') AS month,
  t1.p_bal_gel AS balancing_price_gel,
  t2.share_import,
  t2.share_deregulated_hydro,
  t2.share_regulated_hpp
FROM price_with_usd t1
JOIN trade_derived_entities t2 ON t1.date = t2.date -- Assuming trade_derived_entities contains monthly share data
ORDER BY 1
LIMIT 500;
"""

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
def llm_generate_plan_and_sql(user_query: str, analysis_mode: str) -> str:
    # New combined function
    
    system = (
        "You are an analytical PostgreSQL generator. Your task is to perform two steps: "
        "1. **Plan:** Extract the analysis intent, target variables, and period for the user's question. "
        "2. **Generate SQL:** Write a single, correct PostgreSQL SELECT query to fulfill the plan. "
        "Rules: no INSERT/UPDATE/DELETE; no DDL; NO comments; NO markdown fences. "
        "Use only documented tables and columns. Prefer monthly aggregation. "
        "If USD prices are requested, prefer price_with_usd / tariff_with_usd views. "
    )
    domain_json = json.dumps(DOMAIN_KNOWLEDGE, indent=2)
    
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
- Use price_with_usd / tariff_with_usd when USD is involved.
- Mind that balancing price is influenced by trade volume and price in the trade table.
- Tariffs depend on regulatory principles, inflation, etc.
- Use these examples:
{FEW_SHOT_SQL}

Output Format:
Return a single string containing two parts, separated by '---SQL---'. The first part is a JSON object (the plan), and the second part is the raw SELECT statement.

Example Output:
{json.dumps(plan_format)}
---SQL---
SELECT ...
"""
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        combined_output = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    except Exception as e:
        log.warning(f"Combined generation failed: {e}")
        # Fallback to OpenAI if Gemini fails
        try:
            llm = make_openai()
            combined_output = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        except Exception as e_f:
             log.warning(f"Combined generation failed with fallback: {e_f}")
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
    if not rows:
        return "0 rows."
    df = pd.DataFrame(rows, columns=cols)
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
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df['__year'] = df[time_col].dt.year
        else:
            # Attempt to coerce strings/objects to datetime
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df['__year'] = df[time_col].dt.year

        valid_years = df['__year'].dropna().unique()
        if len(valid_years) >= 2:
            first_full_year = int(valid_years.min())
            last_full_year = int(valid_years.max())

            # Ensure we are comparing two different years
            if first_full_year != last_full_year:
                
                # Filter data for the first and last full years
                df_first = df[df['__year'] == first_full_year]
                df_last = df[df['__year'] == last_full_year]
                
                # Get the mean of all numeric columns for these years
                mean_first_year = df_first[numeric.columns].mean().mean()
                mean_last_year = df_last[numeric.columns].mean().mean()
                
                change = ((mean_last_year - mean_first_year) / mean_first_year * 100) if mean_first_year != 0 else 0
                trend = "increasing" if mean_last_year > mean_first_year else "decreasing"
                
                out.append(f"Trend (Yearly Avg, {first_full_year}→{last_full_year}): {trend} ({change:.1f}%)")
                
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
def llm_summarize(user_query: str, data_preview: str, stats_hint: str) -> str:
    system = (
        "You are EnerBot, an energy market analyst. "
        "Write a short analytic summary using preview and statistics. "
        "If multiple years are present, describe direction (increasing, stable or decreasing), magnitude of change, "
        "seasonal patterns, volatility, and factors from domain knowledge when relevant."
    )
    domain_json = json.dumps(DOMAIN_KNOWLEDGE, indent=2)
    prompt = f"""
User question:
{user_query}

Data preview:
{data_preview}

Statistics:
{stats_hint}

Domain knowledge:
{domain_json}
# NOTE: If a comparison between GEL and USD is present, use the 'CurrencyInfluence' knowledge to explain the divergence based on the exchange rate and USD-denominated costs (gas, imports).

Write 4–7 sentences:
1. State the overall long-term trend (using Yearly Avg).
2. If dual-currency, explain the **divergence** by citing the **GEL/USD exchange rate trend** (depreciation/appreciation).
3. Mention the specific USD-denominated cost factors (e.g., thermal gas, imports) that are affected.
4. Analyze and mention seasonal patterns or volatility.
"""

    
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        out = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    except Exception as e:
        log.warning(f"Summarize failed with Gemini, fallback: {e}")
        llm = make_openai()
        out = llm.invoke([("system", system), ("user", prompt)]).content.strip()
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
    
    # Phase 1: Repair synonyms (non-sqlglot based)
    try:
        repaired = re.sub(r"\bprices\b", "price_with_usd", _sql, flags=re.IGNORECASE)
        repaired = re.sub(r"\btariffs\b", "tariff_with_usd", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\btech_quantity\b", "tech_quantity_view", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\btrade\b", "trade_derived_entities", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bentities\b", "entities_mv", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bmonthly_cpi\b", "monthly_cpi_mv", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\benergy_balance_long\b", "energy_balance_long_mv", repaired, flags=re.IGNORECASE)
        _sql = repaired
    except Exception as e:
        log.warning(f"⚠️ Synonym auto-correction failed: {e}")
        # Not a critical failure, continue with original SQL

    # Phase 2: Append LIMIT 500 if missing
    if " from " in _sql.lower() and not re.search(r"\blimit\s+\d+\b", _sql, flags=re.IGNORECASE):
        
        # CRITICAL FIX: Remove the trailing semicolon if it exists
        _sql = _sql.rstrip().rstrip(';') 
        
        # Append LIMIT 500 without a preceding semicolon
        _sql = f"{_sql}\nLIMIT 500"

    return _sql
    

@app.get("/ask")
def ask_get():
    return {
        "message": "✅ /ask is active. Send POST with JSON: {'query': 'What was the average balancing price in 2023?'} and header X-App-Key."
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

    plan = {}

    # 1) Generate PLAN and SQL in ONE LLM call
    try:
        combined_output = llm_generate_plan_and_sql(q.query, mode)
        if "---SQL---" in combined_output:
            plan_text, raw_sql = combined_output.split("---SQL---", 1)
            raw_sql = raw_sql.strip()
        else:
            plan_text = combined_output
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

        if period_pattern:
            log.info("🧮 Detected user-defined period range → applying aggregation logic.")
            if any(x in q.query.lower() for x in ["generation", "quantity", "volume", "demand", "supply"]):
                agg_func = "SUM"
            else:
                agg_func = "AVG"
            safe_sql_final = f"SELECT {agg_func}(x.*) FROM ({safe_sql}) AS x"
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
        with ENGINE.connect() as conn:
            res = conn.execute(text(safe_sql_final))
            rows = res.fetchall()
            cols = list(res.keys())
            df = pd.DataFrame(rows, columns=cols)
    except Exception as e:
        msg = str(e)
        if "UndefinedColumn" in msg:
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
            log.exception("SQL execution failed")
            raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    # 4) Summarize and analyze
    preview = rows_to_preview(rows, cols)
    stats_hint = quick_stats(rows, cols)
    correlation_results = {}

    if mode == "analyst" and plan.get("intent") == "correlation" and not df.empty:
        log.info("🔍 Calculating correlation matrix for LLM analysis.")
        target_cols = [c for c in df.columns if 'price' in c.lower() or 'bal' in c.lower()]
        explanatory_cols = [c for c in df.columns if 'share' in c.lower() or 'import' in c.lower() or 'hydro' in c.lower() or 'tpp' in c.lower()]
        if target_cols and explanatory_cols:
            corr_df = df[target_cols + explanatory_cols].apply(pd.to_numeric, errors='coerce').dropna()
            for target in target_cols:
                if target in corr_df.columns:
                    corr_series = corr_df.corr()[target].sort_values(ascending=False).round(3)
                    correlation_results[target] = corr_series.drop(index=target, errors='ignore').to_dict()
        if correlation_results:
            stats_hint += "\n\n--- CORRELATION MATRIX (vs Price) ---\n"
            stats_hint += json.dumps(correlation_results, indent=2)
            log.info(f"Generated correlations: {correlation_results}")

    try:
        summary = llm_summarize(q.query, preview, stats_hint)
    except Exception as e:
        log.warning(f"Summarization failed: {e}")
        summary = preview
    summary = scrub_schema_mentions(summary)
    if mode == "analyst" and plan.get("intent") != "general":
        summary = f"**Analysis type: {plan.get('intent')}**\n\n" + summary

    # 5) Chart builder (FINAL: labels from context + unit-only axis + robust numeric coercion)
    chart_data = chart_type = chart_meta = None
    if rows and cols:
        df = df.copy()

        # --- Coerce numeric for all non-time columns so JSON values are numbers, not strings ---
        time_key = next((c for c in cols if any(k in c.lower() for k in ["date", "year", "month"])), None)
        for c in cols:
            if c != time_key:
                try:
                    # coerce to numeric where possible; non-numerics become NaN and remain fine
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                except Exception:
                    pass

        # --- Numeric columns after coercion ---
        num_cols = [c for c in df.columns if c != time_key and pd.api.types.is_numeric_dtype(df[c])]

        # --- Dimension inference (price_tariff | energy_qty | index) ---
        def infer_dimension(col: str) -> str:
            col_l = col.lower()
            if any(x in col_l for x in ["cpi", "index", "inflation"]):
                return "index"
            if any(x in col_l for x in ["quantity", "generation", "volume_tj", "volume", "mw", "tj"]):
                return "energy_qty"
            if any(x in col_l for x in ["price", "tariff", "_gel", "_usd", "p_bal", "p_dereg", "p_gcap"]):
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
            # Uniform dimension → single axis
            log.info("📊 Uniform dimension → single-axis chart.")
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
