# main.py v18.5 — Gemini Analyst (hybrid agent, domain knowledge, better summarization)

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
from pydantic import BaseModel, Field, validator

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
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
    "dates",
    "energy_balance_long",
    "entities",
    "monthly_cpi",
    "price",
    "tariff_gen",
    "tech_quantity",
    "trade",
    # USD materialized views
    "price_with_usd",
    "tariff_with_usd",
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
app = FastAPI(title="EnerBot Analyst (Gemini)", version="18.5")
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

    @validator("query")
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

def llm_plan_analysis(user_query: str) -> dict:
    system = (
        "You are an analytical planner. Given a user question about energy data, "
        "use domain knowledge and schema awareness to extract the analysis intent, "
        "target variables, and period. Return JSON with keys: intent, target, period."
    )
    # include domain knowledge
    domain_json = json.dumps(DOMAIN_KNOWLEDGE, indent=2)
    prompt = f"""
User question:
{user_query}

Domain knowledge:
{domain_json}

Output:
A JSON object like:
{{
  "intent": "trend_analysis" | "comparison" | "volatility" | "correlation",
  "target": "<metric name>",
  "period": "YYYY-YYYY or YYYY-MM to YYYY-MM"
}}
"""
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        plan_text = llm.invoke([("system", system), ("user", prompt)]).content.strip()
        plan = json.loads(plan_text) if plan_text.startswith("{") else {"intent": "general", "target": "", "period": ""}
        return plan
    except Exception as e:
        log.warning(f"Plan generation failed: {e}")
        return {"intent": "general", "target": "", "period": ""}

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
LIMIT 500;

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
FROM tech_quantity
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
FROM monthly_cpi
WHERE cpi_type = 'electricity_gas_and_other_fuels'
ORDER BY date
LIMIT 500;
"""

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
def llm_generate_sql(user_query: str) -> str:
    system = (
        "You write a SINGLE PostgreSQL SELECT query to answer the user's question. "
        "Rules: no INSERT/UPDATE/DELETE; no DDL; NO comments; NO markdown fences. "
        "Use only documented tables and columns. Prefer monthly aggregation. "
        "If USD prices are requested, prefer price_with_usd / tariff_with_usd views. "
        "If domain dependencies apply (balancing price from trade volumes, tariff regulation principles), consider them when selecting columns."
    )
    domain_json = json.dumps(DOMAIN_KNOWLEDGE, indent=2)
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

Output: one raw SELECT statement.
"""
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        sql = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    except Exception as e:
        log.warning(f"Gemini failed to generate SQL, fallback to OpenAI: {e}")
        llm = make_openai()
        sql = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    return sql

# -----------------------------
# Data helpers (modified quick_stats)
# -----------------------------
def rows_to_preview(rows: List[Tuple], cols: List[str], max_rows: int = 8) -> str:
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
    date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower()]
    if date_cols:
        first = df[date_cols[0]].min()
        last = df[date_cols[0]].max()
        out.append(f"Period: {first} → {last}")
    if not numeric.empty:
        desc = numeric.describe().round(3)
        out.append("Numeric summary:")
        out.append(desc.to_string())
        # approximate trend
        third = max(1, len(df) // 3)
        mean_first = numeric.head(third).mean().mean()
        mean_last = numeric.tail(third).mean().mean()
        change = ((mean_last - mean_first) / mean_first * 100) if mean_first != 0 else 0
        trend = "increasing" if mean_last > mean_first else "decreasing"
        out.append(f"Approximate trend: {trend} ({change:.1f}% over period)")
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

Write 3–6 sentences:
1. State the overall trend across the full period.
2. Estimate how much the change is (in % or absolute terms).
3. Mention seasonal or volatility insights.
4. Link to domain factors (tariff policy, trade volumes) if applicable.
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
# SQL sanitize + sqlglot validate/repair
# -----------------------------

# =========================================
# 🧠 Universal Reflection-Based Validator
# Handles aliases, derived fields, and aggregates robustly
# =========================================
from sqlglot import parse_one, exp
from fastapi import HTTPException

def is_alias_or_derived(col: exp.Column) -> bool:
    """
    Returns True if the column is part of an alias, aggregate, or expression.
    Prevents false rejections of derived / aliased columns.
    """
    parent = col.parent
    if not parent:
        return False
    # skip aliases, functions, aggregates, or expressions
    if isinstance(parent, (exp.Alias, exp.Func, exp.Aggregate, exp.Binary, exp.Cast, exp.Extract)):
        return True
    return False


def _validate_allowed(ast: exp.Expression):
    """
    Validate that tables are allowed (materialized views only)
    and column references exist within their schema definitions.
    Derived columns, aliases, and aggregates are allowed automatically.
    """
    # Alias map: alias → real table name
    alias_map = {}
    for tbl in ast.find_all(exp.Table):
        real_name = tbl.name.lower()
        alias = (tbl.alias or real_name).lower()
        alias_map[alias] = real_name

    log.info(f"🔎 Table/Alias map: {alias_map}")

    # Validate tables
    for alias, real in alias_map.items():
        if real not in ALLOWED_TABLES:
            raise HTTPException(
                status_code=400,
                detail=f"Table `{real}` (alias `{alias}`) is not allowed. Only materialized views may be queried."
            )

    # Build schema map: table → list of known columns
    SCHEMA_MAP = {t: [c.lower() for c in cols] for t, cols in REFLECTED_COLUMNS.items()}

    # Validate column references
    for col in ast.find_all(exp.Column):
        name = col.name.lower()
        tbl_alias = (col.table or "").lower()

        if is_alias_or_derived(col):
            continue  # skip aliases/aggregates/derived fields

        # Allow computed columns like *_usd or avg_*, sum_*, etc.
        if name.endswith("_usd") or name.startswith(("avg_", "sum_", "count_", "min_", "max_")):
            continue

        # Determine which table(s) this column could come from
        candidate_tables = []
        if tbl_alias and tbl_alias in alias_map:
            candidate_tables = [alias_map[tbl_alias]]
        else:
            candidate_tables = list(alias_map.values())

        # Validate column presence
        found = False
        for t in candidate_tables:
            if t in SCHEMA_MAP and name in SCHEMA_MAP[t]:
                found = True
                break

        if not found:
            log.warning(f"⚠️ Column `{name}` not found in allowed schema ({candidate_tables}) — treating as derived/alias.")
            # soft warning, not hard rejection


def plan_validate_repair(sql: str) -> str:
    """
    Validate SQL using sqlglot AST.
    - Automatically repairs plural → singular table names
    - Normalizes schema references
    - Allows derived columns and aliases
    """
    def transform(_sql: str) -> str:
        ast = parse_one(_sql, read="postgres")
        _validate_allowed(ast)
        return ast.sql(dialect="postgres")

    try:
        return transform(sql)
    except Exception as e:
        log.warning(f"First pass validation failed: {e}. Attempting auto-repair...")

        repaired = sql
        repaired = re.sub(r"\bprices\b", "price_with_usd", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\btariffs\b", "tariff_with_usd", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\btech_quantity\b", "tech_quantity_view", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\btrade\b", "trade_derived_entities", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bentities\b", "entities_mv", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\bmonthly_cpi\b", "monthly_cpi_mv", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\benergy_balance_long\b", "energy_balance_long_mv", repaired, flags=re.IGNORECASE)

        try:
            return transform(repaired)
        except Exception as e2:
            log.exception("❌ Second pass validation failed:")
            raise HTTPException(status_code=400, detail=f"Validation failed: {e2}")



@app.get("/ask")
def ask_get():
    return {
        "message": "✅ /ask is active. Send POST with JSON: {'query': 'What was the average balancing price in 2023?'} and header X-App-Key."
    }

@app.post("/ask", response_model=APIResponse)
def ask_post(q: Question, x_app_key: str = Header(..., alias="X-App-Key")):
    t0 = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    mode = detect_analysis_mode(q.query)
    log.info(f"🧭 Selected mode: {mode}")

    plan = {}
    if mode == "analyst":
        plan = llm_plan_analysis(q.query)
        log.info(f"📝 Plan: {plan}")

    # 1) Generate SQL
    try:
        raw_sql = llm_generate_sql(q.query)
    except Exception as e:
        log.exception("SQL generation failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {e}")

    # 2) Sanitize, validate, repair
    try:
        sanitized = sanitize_sql(raw_sql)
        safe_sql = plan_validate_repair(sanitized)
        if " from " in safe_sql.lower() and re.search(r"\blimit\s+\d+\b", safe_sql, flags=re.IGNORECASE) is None:
            safe_sql = f"{safe_sql}\nLIMIT 500"
        log.info(f"✅ SQL after validation/repair:\n{safe_sql}")
    except Exception as e:
        log.warning(f"Rejected SQL: {raw_sql}")
        raise HTTPException(status_code=400, detail=f"Unsafe or invalid SQL: {e}")

    # 3) Execute
    try:
        with ENGINE.connect() as conn:
            res = conn.execute(text(safe_sql))
            rows = res.fetchall()
            cols = list(res.keys())
    except Exception as e:
        # existing fallback logic
        msg = str(e)
        if "UndefinedColumn" in msg:
            for bad, good in COLUMN_SYNONYMS.items():
                if re.search(rf"\b{bad}\b", safe_sql, flags=re.IGNORECASE):
                    safe_sql = re.sub(rf"\b{bad}\b", good, safe_sql, flags=re.IGNORECASE)
                    log.warning(f"🔁 Auto-corrected column '{bad}' → '{good}' (retry)")
                    with ENGINE.connect() as conn:
                        res = conn.execute(text(safe_sql))
                        rows = res.fetchall()
                        cols = list(res.keys())
                    break
            else:
                log.exception("SQL execution failed (UndefinedColumn)")
                raise HTTPException(status_code=500, detail=f"Query failed: {e}")
        else:
            log.exception("SQL execution failed")
            raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    # 4) Summarize / Reason
    preview = rows_to_preview(rows, cols)
    stats_hint = quick_stats(rows, cols)
    try:
        summary = llm_summarize(q.query, preview, stats_hint)
    except Exception as e:
        log.warning(f"Summarization failed: {e}")
        summary = preview

    summary = scrub_schema_mentions(summary)

    if mode == "analyst" and plan.get("intent") != "general":
        summary = f"**Analysis type: {plan.get('intent')}**\n\n" + summary

    # 5) Chart builder (unchanged)
    chart_data = chart_type = chart_meta = None
    if rows and len(cols) >= 2:
        df = pd.DataFrame(rows, columns=cols)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            val_col = num_cols[0]
            label_col = None
            for c in df.columns:
                if c == val_col:
                    continue
                if df[c].dtype == "O" or "date" in c.lower() or "year" in c.lower():
                    label_col = c
                    break
            if label_col:
                sample = df[[label_col, val_col]].head(50)
                if "date" in label_col.lower() or "year" in label_col.lower():
                    chart_type = "line"
                    chart_data = [{"date": str(r[label_col]), "value": float(r[val_col])} for _, r in sample.iterrows()]
                    chart_meta = {"xAxisTitle": label_col, "yAxisTitle": val_col, "title": "Trend"}
                else:
                    chart_type = "bar"
                    chart_data = [{"label": str(r[label_col]), "value": float(r[val_col])} for _, r in sample.iterrows()]
                    chart_meta = {"xAxisTitle": label_col, "yAxisTitle": val_col, "title": "Comparison"}

    return APIResponse(
        answer=summary,
        chart_data=chart_data,
        chart_type=chart_type,
        chart_metadata=chart_meta,
        execution_time=round(time.time() - t0, 2),
    )

# -----------------------------
# Local dev runner
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
