
# main.py v18.9 — NaN fix for correlations (all thermal tariffs), time reduction, summer/winter balancing price, 502 mitigation

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
    "quantity_mwh": "quantity_tech",
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
app = FastAPI(title="EnerBot Analyst (Gemini)", version="18.9")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint to prevent cold starts
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# -----------------------------
# Models
# -----------------------------
class Question(BaseModel):
    query: str = Field(..., max_length=2000)
    user_id: Optional[str] = None

    @field_validator("query")
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
        "volatility", "pattern", "season", "relationship", "correlation", "evolution",
        "driver", "cause", "effect", "factor", "reason", "influence", "depend", "why", "behind"
    ]
    for kw in analytical_keywords:
        if kw in user_query.lower():
            return "analyst"
    return "light"

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
LIMIT 3750;

-- Example 2: Single month balancing price (USD) for May 2024
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

-- Example 4: Average regulated tariffs (USD) by entity for 2024
SELECT
  entity,
  AVG(tariff_usd) AS avg_tariff_usd_2024
FROM tariff_with_usd
WHERE EXTRACT(YEAR FROM date) = 2024
GROUP BY entity
ORDER BY entity
LIMIT 3750;

-- Example 5: CPI monthly values for electricity fuels category
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  cpi
FROM monthly_cpi_mv
WHERE cpi_type = 'electricity_gas_and_other_fuels'
ORDER BY date
LIMIT 3750;

-- Example 6: Monthly data for Balancing Price (GEL) and Shares of key sources (Hydro, Import) for correlation analysis
SELECT
  TO_CHAR(t1.date, 'YYYY-MM') AS month,
  t1.p_bal_gel AS balancing_price_gel,
  t2.share_import,
  t2.share_deregulated_hydro,
  t2.share_regulated_hpp
FROM price_with_usd t1
JOIN trade_derived_entities t2 ON t1.date = t2.date
ORDER BY 1
LIMIT 3750;

-- Example 7: Monthly data for Balancing Price (GEL, USD) and specific tariffs (Engurhesi, Gardabani, Old TPPs) for correlation
SELECT
  TO_CHAR(t1.date, 'YYYY-MM') AS month,
  t1.p_bal_gel,
  t1.p_bal_usd,
  t1.xrate,
  (SELECT tariff_gel FROM tariff_with_usd t2 WHERE t2.date = t1.date AND t2.entity = 'ltd "engurhesi"1') AS enguri_tariff_gel,
  (SELECT tariff_gel FROM tariff_with_usd t2 WHERE t2.date = t1.date AND t2.entity = 'ltd "gardabni thermal power plant"') AS gardabani_tpp_tariff_gel,
  (SELECT AVG(tariff_gel) FROM tariff_with_usd t2 WHERE t2.date = t1.date AND t2.entity IN ('ltd "mtkvari energy"', 'ltd "iec" (tbilresi)', 'ltd "g power" (capital turbines)')) AS grouped_old_tpp_tariff_gel
FROM price_with_usd t1
ORDER BY t1.date
LIMIT 3750;

-- Example 8: Monthly tariff trend for Engurhesi in GEL
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  tariff_gel AS enguri_tariff_gel
FROM tariff_with_usd
WHERE entity = 'ltd "engurhesi"1'
ORDER BY date
LIMIT 3750;

-- Example 9: Monthly hydro vs thermal generation volumes
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  SUM(CASE WHEN type_tech = 'hydro' THEN quantity_tech ELSE 0 END) AS hydro_qty_thousand_mwh,
  SUM(CASE WHEN type_tech = 'thermal' THEN quantity_tech ELSE 0 END) AS thermal_qty_thousand_mwh
FROM tech_quantity_view
GROUP BY 1
ORDER BY 1
LIMIT 3750;
"""

@retry(stop=stop_after_attempt(1))
def llm_generate_plan_and_sql(user_query: str, analysis_mode: str) -> str:
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

    if analysis_mode == "light":
        guidance = (
            "Use appropriate views from schema (e.g., price_with_usd for prices, tariff_with_usd for tariffs, "
            "tech_quantity_view for generation volumes, monthly_cpi_mv for CPI). "
            "Prefer monthly aggregation unless yearly specified. Use examples."
        )
    else:
        guidance = (
            "Use appropriate views from schema (e.g., price_with_usd for prices, tariff_with_usd for tariffs, "
            "tech_quantity_view for generation volumes, monthly_cpi_mv for CPI). "
            "For balancing prices (p_bal_gel, p_bal_usd) and deregulated prices (p_dereg_gel), use price_with_usd. "
            "For generation or trade volumes, use tech_quantity_view or trade_derived_entities, respecting units (thousand MWh). "
            "For CPI, use monthly_cpi_mv with cpi_type filter (e.g., 'electricity_gas_and_other_fuels'). "
            "For correlation or driver analysis (when intent is 'correlation' or 'driver_analysis'): "
            "- Fetch tariff_gel from tariff_with_usd using subqueries joined on date, filtering for entities in tariff_entities "
            "(e.g., 'ltd \"engurhesi\"1' for Enguri, 'ltd \"gardabni thermal power plant\"' for Gardabani, "
            "average for old TPPs: 'ltd \"mtkvari energy\"', 'ltd \"iec\" (tbilresi)', 'ltd \"g power\" (capital turbines)'). "
            "- Example subqueries: (SELECT tariff_gel FROM tariff_with_usd WHERE date = main.date AND entity = 'ltd \"engurhesi\"1') AS enguri_tariff_gel, "
            "(SELECT tariff_gel FROM tariff_with_usd WHERE date = main.date AND entity = 'ltd \"gardabni thermal power plant\"') AS gardabani_tpp_tariff_gel, "
            "(SELECT AVG(tariff_gel) FROM tariff_with_usd WHERE date = main.date AND entity IN "
            "('ltd \"mtkvari energy\"', 'ltd \"iec\" (tbilresi)', 'ltd \"g power\" (capital turbines)')) AS grouped_old_tpp_tariff_gel. "
            "- Include PriceDrivers columns (xrate, shares from trade_derived_entities, p_dereg_gel, tariffs; exclude quantities). "
            "Use CurrencyInfluence for GEL/USD divergence and SeasonalityHint for trends. "
            "Prefer monthly aggregation unless yearly specified."
        )

    prompt = f"""
User question:
{user_query}

Schema:
{DB_SCHEMA_DOC}

Domain knowledge:
{domain_json}

Guidance:
{guidance}
- Use these examples:
{FEW_SHOT_SQL}

Output Format:
Return a single string containing two parts, separated by '---SQL---'. The first part is a JSON object (the plan), and the second part is the raw SELECT statement.

Example Output:
{json.dumps(plan_format)}
---SQL---
SELECT ...
"""
    t_llm_start = time.time()
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        combined_output = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    except Exception as e:
        log.warning(f"Combined generation failed: {e}")
        raise e
    log.info(f"LLM took {time.time() - t_llm_start:.2f}s")
    return combined_output

# -----------------------------
# Data helpers
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
    
    # Detect date/year column
    date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower() or "month" in c.lower()]
    if not date_cols or numeric.empty:
        return "\n".join(out)

    time_col = date_cols[0]

    # --- NEW TREND CALCULATION: Compare First Full Year vs Last Full Year ---
    try:
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df['__year'] = df[time_col].dt.year
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df['__year'] = df[time_col].dt.year

        valid_years = df['__year'].dropna().unique()
        if len(valid_years) >= 2:
            first_full_year = int(valid_years.min())
            last_full_year = int(valid_years.max())

            if first_full_year != last_full_year:
                df_first = df[df['__year'] == first_full_year]
                df_last = df[df['__year'] == last_full_year]
                
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

    first = df[time_col].min()
    last = df[time_col].max()
    out.append(f"Period: {first} → {last}")
    
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

Write 4–7 sentences:
1. State the overall long-term trend (using Yearly Avg).
2. If dual-currency, explain the **divergence** by citing the **GEL/USD exchange rate trend** (depreciation/appreciation).
3. Mention the specific USD-denominated cost factors (e.g., thermal gas, imports) that are affected.
4. For balancing price (trends or averages), differentiate summer (Apr-May-Jun-Jul: low prices, high hydro generation) vs winter (other months: high prices, thermal/import dominant) due to structural differences.
5. Analyze and mention seasonal patterns or volatility.
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
def simple_table_whitelist_check(sql: str):
    cleaned_tables = set()
    try:
        parsed_expression = parse_one(sql, read='bigquery') 
        cte_names = set()
        with_clause = parsed_expression.find(exp.With)
        if with_clause:
            for cte in with_clause.expressions:
                cte_names.add(cte.alias.lower()) 
        for table_exp in parsed_expression.find_all(exp.Table):
            t_raw = table_exp.name.lower()
            t_name = t_raw.split('.')[0]
            if t_name in cte_names:
                continue 
            cleaned_tables.add(t_name)
        final_tables = set()
        for t in cleaned_tables:
            canonical = TABLE_SYNONYMS.get(t, t)
            if canonical in ALLOWED_TABLES:
                final_tables.add(canonical)
            else:
                close = get_close_matches(t, list(ALLOWED_TABLES) + list(TABLE_SYNONYMS.keys()), n=1, cutoff=0.6)
                if close and TABLE_SYNONYMS.get(close[0], close[0]) in ALLOWED_TABLES:
                    final_tables.add(TABLE_SYNONYMS.get(close[0], close[0]))
                else:
                    raise ValueError(f"Table '{t}' not allowed or unknown. Closest match: {close[0] if close else 'none'}")
        if SCHEMA_MAP:
            for table in final_tables:
                if table not in SCHEMA_MAP:
                    raise ValueError(f"Table '{table}' not found in schema map")
                for col_exp in parsed_expression.find_all(exp.Column):
                    col_name = col_exp.name.lower()
                    canonical_col = COLUMN_SYNONYMS.get(col_name, col_name)
                    if col_name not in SCHEMA_MAP[table] and canonical_col not in SCHEMA_MAP[table]:
                        close = get_close_matches(canonical_col, SCHEMA_MAP[table], n=1, cutoff=0.6)
                        if not close:
                            raise ValueError(f"Column '{col_name}' not allowed in table '{table}'")
                        log.warning(f"Column '{col_name}' mapped to closest match '{close[0]}' in {table}")
        log.info(f"✅ Pre-validation passed. Tables: {sorted(final_tables)}")
        return True
    except ParseError as e:
        log.error(f"SQL parse error: {e}")
        raise ValueError(f"Invalid SQL syntax: {e}")
    except ValueError as e:
        log.error(f"SQL validation error: {e}")
        raise
    except Exception as e:
        log.error(f"Unexpected SQL validation error: {e}")
        raise ValueError(f"SQL validation failed: {e}")

# ... [Rest of the code for SQL execution and chart building unchanged] ...

# Updated correlation section
if plan.get("intent") == "correlation" and not df.empty:
    log.info("🔍 Calculating correlation matrix for LLM analysis.")
    target_cols = [c for c in df.columns if c in ['p_bal_gel', 'p_bal_usd']]
    explanatory_cols = [
        c for c in df.columns 
        if c in [
            'xrate',
            'enguri_tariff_gel',
            'gardabani_tpp_tariff_gel',
            'grouped_old_tpp_tariff_gel',
            'share_deregulated_hydro',
            'share_import',
            'share_renewable_ppa'
        ]
    ]
    if target_cols and explanatory_cols:
        def flatten_nested(df_in: pd.DataFrame) -> pd.DataFrame:
            for c in df_in.columns:
                if df_in[c].apply(lambda x: isinstance(x, (pd.DataFrame, list, dict))).any():
                    try:
                        df_in[c] = df_in[c].apply(
                            lambda x: x.iloc[0, 0] if isinstance(x, pd.DataFrame) and not x.empty else (
                                x[0] if isinstance(x, list) and len(x) > 0 else np.nan
                            )
                        )
                        log.info(f"🩹 Flattened nested structure in column '{c}'")
                    except Exception as e:
                        log.warning(f"⚠️ Could not flatten column '{c}': {e}")
            return df_in
        subset = df[target_cols + explanatory_cols].copy()
        log.info(f"🧩 Subset before flatten: cols={list(subset.columns)} shape={subset.shape}")
        def collapse_to_scalar(val):
            try:
                return float(str(val).replace(",", ".").replace("%", "").strip())
            except Exception:
                pass
            if isinstance(val, pd.DataFrame):
                flat = pd.to_numeric(val.stack(), errors="coerce")
                return flat.mean() if not flat.empty else np.nan
            if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                s = pd.to_numeric(pd.Series(val), errors="coerce")
                return s.mean() if s.notna().any() else np.nan
            return np.nan
        subset = subset.loc[:, ~subset.columns.duplicated()]
        log.info(f"🧮 After deduplication: cols={list(subset.columns)} shape={subset.shape}")
        for c in subset.columns:
            if isinstance(subset[c], pd.DataFrame):
                subset[c] = subset[c].applymap(collapse_to_scalar).stack().groupby(level=0).mean()
            elif not pd.api.types.is_numeric_dtype(subset[c]):
                subset[c] = subset[c].map(collapse_to_scalar)
        subset = subset.apply(pd.to_numeric, errors="coerce")
        log.info(f"🔍 Sample values before dropna:\n{subset.head(5)}")
        log.info(f"🔍 Non-null counts:\n{subset.notna().sum()}")
        log.info(f"✅ After flatten + coercion: shape={subset.shape}, dtypes={subset.dtypes.to_dict()}")
        corr_df = subset.select_dtypes(include=[np.number])
        if corr_df.shape[1] < 2:
            log.warning("⚠️ Not enough numeric columns for correlation.")
        else:
            for target in target_cols:
                if target in corr_df.columns:
                    corr_matrix = corr_df.corr(numeric_only=True)
                    if target not in corr_matrix.index:
                        log.warning(f"⚠️ Target '{target}' not in corr_matrix.index {list(corr_matrix.index)}")
                        continue
                    val = corr_matrix.loc[target, :]
                    if isinstance(val, pd.DataFrame):
                        val = val.iloc[:, 0]
                    corr_series = val.sort_values(ascending=False).round(3)
                    correlation_results[target] = corr_series.drop(index=target, errors='ignore').to_dict()
    if correlation_results:
        stats_hint += "\n\n--- CORRELATION MATRIX (vs Price) ---\n"
        stats_hint += json.dumps(correlation_results, indent=2)
        log.info(f"Generated correlations: {correlation_results}")
    else:
        log.info("⚠️ No numeric overlap found for correlation calculation.")

try:
    summary = llm_summarize(q.query, preview, stats_hint)
except Exception as e:
    log.warning(f"Summarization failed: {e}")
    summary = preview
summary = scrub_schema_mentions(summary)
if mode == "analyst" and plan.get("intent") != "general":
    summary = f"**Analysis type: {plan.get('intent')}**\n\n" + summary

# ... [Chart builder unchanged] ...

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

# -----------------------------
# Server Startup
# -----------------------------
if __name__ == "__main__":
    try:
        import uvicorn
        port = int(os.getenv("PORT", 8000)) 
        log.info(f"🚀 Starting Uvicorn server on 0.0.0.0:{port} with 1 worker")
        uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", workers=1)
    except ImportError:
        log.error("Uvicorn is not installed. Please install it with 'pip install uvicorn'.")
    except Exception as e:
        log.error(f"FATAL: Uvicorn server failed to start: {e}")
