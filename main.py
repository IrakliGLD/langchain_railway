# main.py v18.4 — Gemini Analyst (sqlglot validation + repair + usd views + few-shot)
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

# Allow the base tables + usd materialized views
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
    # common USD typos will be handled by switching to *_with_usd views
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

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="EnerBot Analyst (Gemini)", version="18.4")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
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
# LLM helpers
# -----------------------------
def make_gemini() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)

def make_openai() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set (fallback needed)")
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

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

-- Example 3: Generation (thousand MWh) by technology per month (note: quantity_tech is thousand MWh)
SELECT
  TO_CHAR(date, 'YYYY-MM') AS month,
  type_tech,
  SUM(quantity_tech) AS qty_thousand_mwh
FROM tech_quantity
GROUP BY 1,2
ORDER BY 1,2
LIMIT 500;

-- Example 4: Average regulated tariffs (USD) by entity for 2024 (use tariff_with_usd view)
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
        "Use only the documented tables and columns. Prefer monthly aggregation. "
        "If prices in USD are requested, prefer the materialized views price_with_usd / tariff_with_usd. "
        "If quantities are requested, remember tech_quantity.quantity_tech is thousand MWh."
    )
    prompt = f"""
User question:
{user_query}

Schema (reference only):
{DB_SCHEMA_DOC}

Guidance:
- If USD prices/tariffs are needed, read from price_with_usd or tariff_with_usd (they expose *_usd columns).
- For tech_quantity, the column is quantity_tech (thousand MWh).
- For simple single-month answers, filter by the month's first day (YYYY-MM-01).
- Add LIMIT 500 for safety.

Use these examples as style/structure (do not echo them back, just follow the pattern):
{FEW_SHOT_SQL}

Output:
- Return ONLY raw SQL (no fences, no prose), one SELECT statement.
"""

    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        sql = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    except Exception as e:
        log.warning(f"Gemini failed to generate SQL, trying OpenAI: {e}")
        llm = make_openai()
        sql = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    return sql

# -----------------------------
# Data helpers (unchanged)
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
    if not numeric.empty:
        describe = numeric.describe().round(3)
        out.append("Numeric summary:")
        out.append(describe.to_string())
    return "\n".join(out)

# -----------------------------
# SQL sanitize + sqlglot validate/repair
# -----------------------------
def sanitize_sql(sql: str) -> str:
    """Trim fences and comments; enforce SELECT-only."""
    if not sql:
        raise HTTPException(status_code=400, detail="Empty SQL")
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE)
    sql = sql.replace("```", "")
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = sql.strip().rstrip(";").strip()
    if not re.match(r"^\s*select\b", sql, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Only SELECT statements are allowed")
    forbidden = ["insert", "update", "delete", "drop", "create", "alter", "grant", "revoke"]
    if any(re.search(rf"\b{f}\b", sql, re.IGNORECASE) for f in forbidden):
        raise HTTPException(status_code=400, detail="Forbidden SQL operation detected")
    return sql

def _collect_tables(ast: exp.Expression) -> List[str]:
    tables = []
    for t in ast.find_all(exp.Table):
        name = (t.name or "").lower()
        if name:
            tables.append(name)
    return list(dict.fromkeys(tables))  # unique preserve order

def _collect_columns(ast: exp.Expression) -> List[str]:
    cols = []
    for c in ast.find_all(exp.Column):
        # c.this: Identifier (column name), c.table: Identifier or None
        name = (c.name or "").lower()
        if name:
            cols.append(name)
    return list(dict.fromkeys(cols))

def _apply_table_synonyms(name: str) -> str:
    n = name.lower()
    # plural to singular if present
    if n.endswith("s") and n[:-1] in ALLOWED_TABLES:
        return n[:-1]
    if n in TABLE_SYNONYMS:
        return TABLE_SYNONYMS[n]
    return n

def _apply_column_synonyms(name: str) -> str:
    n = name.lower()
    return COLUMN_SYNONYMS.get(n, n)

def _rewrite_usd_to_views(ast: exp.Expression) -> exp.Expression:
    """
    If any *_usd columns are referenced from price or tariff tables,
    switch those table refs to price_with_usd / tariff_with_usd.
    """
    usd_cols = {c for c in _collect_columns(ast) if c.endswith("_usd")}
    if not usd_cols:
        return ast

    # Map table names
    for tbl in ast.find_all(exp.Table):
        name = (tbl.name or "").lower()
        if name in {"price", "prices", "price_with_usd"}:
            tbl.set("this", exp.to_identifier("price_with_usd"))
        if name in {"tariff_gen", "tariffs", "tariff_with_usd"}:
            tbl.set("this", exp.to_identifier("tariff_with_usd"))
    return ast

def _normalize_tables(ast: exp.Expression) -> exp.Expression:
    """Apply table synonym/plural normalization on AST."""
    for tbl in ast.find_all(exp.Table):
        name = (tbl.name or "").lower()
        new_name = _apply_table_synonyms(name)
        if new_name != name:
            tbl.set("this", exp.to_identifier(new_name))
    return ast

def _normalize_columns(ast: exp.Expression) -> exp.Expression:
    """Apply column synonym normalization on AST (rename columns in identifiers)."""
    for col in ast.find_all(exp.Column):
        name = (col.name or "").lower()
        new_name = _apply_column_synonyms(name)
        if new_name != name:
            col.set("this", exp.to_identifier(new_name))
    return ast

def _validate_allowed(ast: exp.Expression):
    # tables
    for t in _collect_tables(ast):
        t2 = _apply_table_synonyms(t)
        if t2 not in ALLOWED_TABLES:
            raise HTTPException(status_code=400, detail=f"Unknown or disallowed table: {t}")
    # columns (lightweight, name-based against known labels)
    known_cols = {c.lower() for c in COLUMN_LABELS.keys()} | {
        # allow *_usd that exist in your materialized views
        "p_dereg_usd", "p_bal_usd", "p_gcap_usd", "tariff_usd"
    }
    for c in _collect_columns(ast):
        c2 = _apply_column_synonyms(c)
        if c2.endswith("_usd") and c2 not in known_cols:
            # allowed; they'll come from *_with_usd views
            continue
        if c2 not in known_cols:
            # let DB hint try to fix later; raise a soft warning instead of hard fail
            log.warning(f"⚠️ Unknown column name encountered: {c} (normalized: {c2})")

def plan_validate_repair(sql: str) -> str:
    """
    Parse with sqlglot → normalize tables/columns → switch to USD views if needed →
    validate → return SQL string. Attempt a second repair pass if first fails.
    """
    def transform(_sql: str) -> str:
        ast = parse_one(_sql, read="postgres")
        ast = _normalize_tables(ast)
        ast = _normalize_columns(ast)
        ast = _rewrite_usd_to_views(ast)
        _validate_allowed(ast)
        return ast.sql(dialect="postgres")

    try:
        return transform(sql)
    except Exception as e:
        log.warning(f"First pass validation/repair failed: {e}. Attempting second pass...")
        # Second pass: try more aggressive synonym application in raw SQL
        repaired = sql
        # plural tables
        for plural, canon in TABLE_SYNONYMS.items():
            repaired = re.sub(rf"\b{plural}\b", canon, repaired, flags=re.IGNORECASE)
        # columns
        for bad, good in COLUMN_SYNONYMS.items():
            repaired = re.sub(rf"\b{bad}\b", good, repaired, flags=re.IGNORECASE)
        # explicit usd view preference if *_usd seen
        if re.search(r"\b(p_.*_usd|tariff_usd)\b", repaired, flags=re.IGNORECASE):
            repaired = re.sub(r"\bprice\b", "price_with_usd", repaired, flags=re.IGNORECASE)
            repaired = re.sub(r"\btariff_gen\b", "tariff_with_usd", repaired, flags=re.IGNORECASE)
        # try again
        return transform(repaired)

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "EnerBot Analyst v18.4 is running 🚀"}

@app.get("/healthz")
def healthz(check_db: Optional[bool] = Query(False)):
    if not check_db:
        return {"status": "ok"}
    try:
        with ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "db_status": "connected"}
    except Exception as e:
        log.exception("Health DB check failed")
        return {"status": "ok", "db_status": f"failed: {e}"}

@app.get("/ask")
def ask_get():
    return {
        "message": "✅ /ask is active. Send POST with JSON: {'query': 'What was the average balancing price in 2023?'} and header X-App-Key."
    }

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6))
def llm_summarize(user_query: str, data_preview: str, stats_hint: str) -> str:
    system = (
        "You are EnerBot, an energy market analyst. "
        "Write a concise explanation grounded ONLY in the provided data preview. "
        "Do NOT mention SQL, tables, columns, schema, or database internals."
    )
    prompt = f"""
User asked: {user_query}

Data preview (first rows / computed stats):
{data_preview}

Optional hints:
{stats_hint}

Write 3–6 sentences: trend, notable highs/lows, any obvious seasonal pattern, and a short takeaway.
Avoid jargon. Keep it factual and cautious.
"""
    try:
        llm = make_gemini() if MODEL_TYPE == "gemini" else make_openai()
        out = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    except Exception as e:
        log.warning(f"Gemini failed to summarize, trying OpenAI: {e}")
        llm = make_openai()
        out = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    return out

@app.post("/ask", response_model=APIResponse)
def ask_post(q: Question, x_app_key: str = Header(..., alias="X-App-Key")):
    t0 = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1) Generate SQL
    try:
        raw_sql = llm_generate_sql(q.query)
    except Exception as e:
        log.exception("SQL generation failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {e}")

    # 2) Sanitize, parse, validate, repair (sqlglot-driven)
    try:
        sanitized = sanitize_sql(raw_sql)
        safe_sql = plan_validate_repair(sanitized)
        # Ensure LIMIT if there is a FROM clause and none already
        if " from " in safe_sql.lower() and re.search(r"\blimit\s+\d+\b", safe_sql, flags=re.IGNORECASE) is None:
            safe_sql = f"{safe_sql}\nLIMIT 500"
        log.info(f"✅ SQL after validation/repair:\n{safe_sql}")
    except Exception as e:
        log.warning(f"Rejected SQL: {raw_sql}")
        raise HTTPException(status_code=400, detail=f"Unsafe or invalid SQL: {e}")

    # 3) Execute with minimal auto-repair (use DB hint once)
    try:
        with ENGINE.connect() as conn:
            res = conn.execute(text(safe_sql))
            rows = res.fetchall()
            cols = list(res.keys())
    except Exception as e:
        # Try one shot: map common column typos
        msg = str(e)
        if "UndefinedColumn" in msg:
            for bad, good in COLUMN_SYNONYMS.items():
                if re.search(rf"\b{bad}\b", safe_sql, flags=re.IGNORECASE):
                    safe_sql = re.sub(rf"\b{bad}\b", good, safe_sql, flags=re.IGNORECASE)
                    log.warning(f"🔁 Auto-corrected column name '{bad}' → '{good}' (execution retry)")
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

    # 4) Summarize
    preview = rows_to_preview(rows, cols)
    stats_hint = quick_stats(rows, cols)
    try:
        summary = llm_summarize(q.query, preview, stats_hint)
    except Exception as e:
        log.warning(f"Summary failed: {e}")
        summary = preview

    summary = scrub_schema_mentions(summary)

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
