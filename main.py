# main.py v18.2 — Gemini Analyst (no RAG, SQL + summary)
import os
import re
import json
import time
import logging
import urllib.parse
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# LLMs (Gemini primary, OpenAI fallback)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Your schema doc + output scrubber
from context import DB_SCHEMA_DOC, scrub_schema_mentions

# -----------------------------
# Boot + Config
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
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

# Only these tables may be queried
ALLOWED_TABLES = {
    "dates",
    "energy_balance_long",
    "entities",
    "monthly_cpi",
    "price",
    "tariff_gen",
    "tech_quantity",
    "trade",
}

# -----------------------------
# DB Engine (psycopg v3)
# -----------------------------
def coerce_to_psycopg_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    # If scheme isn't already the psycopg dialect, coerce it
    if parsed.scheme in ("postgres", "postgresql"):
        return url.replace(parsed.scheme, "postgresql+psycopg", 1)
    if not parsed.scheme.startswith("postgresql+"):
        # last resort
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

# Eager ping (fail fast on bad creds/URL)
with ENGINE.connect() as conn:
    conn.execute(text("SELECT 1"))
log.info("✅ Database connectivity verified")

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="EnerBot Analyst (Gemini)", version="18.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
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
# LLM helpers
# -----------------------------
def make_gemini() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

def make_openai() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set (fallback needed)")
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
def llm_generate_sql(user_query: str) -> str:
    """
    Ask LLM for a single SELECT query (no markdown), limited to allowed tables/columns per schema doc.
    """
    system = (
        "You write a SINGLE PostgreSQL SELECT query to answer the user's question. "
        "Rules: no INSERT/UPDATE/DELETE; no DDL; NO comments; NO markdown fences. "
        "Use only the documented tables and columns. Prefer monthly aggregation. "
        "If unsure, produce a minimal safe SELECT that still helps answer."
    )
    prompt = f"""
User question:
{user_query}

Schema (for your reference only):
{DB_SCHEMA_DOC}

Output rules:
- Return ONLY raw SQL (no ``` fences, no prose)
- SELECT queries only
- Use at most the necessary tables
- If large, add LIMIT 500
"""

    # Prefer Gemini; fallback to OpenAI
    try:
        if MODEL_TYPE == "gemini":
            llm = make_gemini()
        else:
            llm = make_openai()
        sql = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    except Exception as e:
        log.warning(f"Gemini failed to generate SQL, trying OpenAI: {e}")
        llm = make_openai()
        sql = llm.invoke([("system", system), ("user", prompt)]).content.strip()

    return sql

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
def llm_summarize(user_query: str, data_preview: str, stats_hint: str) -> str:
    """
    Ask LLM for a concise, non-technical summary based ONLY on the previewed data and hints.
    """
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
        if MODEL_TYPE == "gemini":
            llm = make_gemini()
        else:
            llm = make_openai()
        out = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    except Exception as e:
        log.warning(f"Gemini failed to summarize, trying OpenAI: {e}")
        llm = make_openai()
        out = llm.invoke([("system", system), ("user", prompt)]).content.strip()

    return out

# -----------------------------
# SQL sanitization
# -----------------------------
_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)

def _extract_tables(sql: str) -> List[str]:
    # naive table grab from FROM/JOIN words
    tokens = re.findall(r"\b(from|join)\s+([a-zA-Z0-9_\.]+)", sql, flags=re.IGNORECASE)
    tables = []
    for _, name in tokens:
        # strip schema prefix like public.price
        tbl = name.split(".")[-1]
        tables.append(tbl)
    return tables

def sanitize_sql(sql: str) -> str:
    if not sql:
        raise ValueError("Empty SQL")
    # remove fences and comments
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE)
    sql = sql.replace("```", "")
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = sql.strip().rstrip(";").strip()

    if not _SELECT_RE.match(sql):
        raise ValueError("Only SELECT statements are allowed")

    tables = _extract_tables(sql)
    if not tables:
        # allow SELECT 1 or aggregates with no FROM
        pass
    else:
        for t in tables:
            if t not in ALLOWED_TABLES:
                raise ValueError(f"Table '{t}' is not allowed")

    # Add LIMIT if missing and not an aggregate-only query
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.IGNORECASE) is None:
        # Skip LIMIT if the query is a single aggregate without FROM
        if " from " in sql.lower():
            sql = f"{sql} LIMIT 500"

    return sql

# -----------------------------
# Data helpers
# -----------------------------
def rows_to_preview(rows: List[Tuple], cols: List[str], max_rows: int = 8) -> str:
    if not rows:
        return "No rows returned."
    df = pd.DataFrame(rows[:max_rows], columns=cols)
    # small floats formatting
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
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "EnerBot Analyst v18.2 is running 🚀"}

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

@app.post("/ask", response_model=APIResponse)
def ask_post(
    q: Question,
    x_app_key: str = Header(..., alias="X-App-Key")
):
    t0 = time.time()

    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1) Ask LLM for SQL
    try:
        raw_sql = llm_generate_sql(q.query)
    except Exception as e:
        log.exception("SQL generation failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {e}")

    # 2) Sanitize SQL
    try:
        safe_sql = sanitize_sql(raw_sql)
    except Exception as e:
        log.warning(f"Rejected SQL: {raw_sql}")
        raise HTTPException(status_code=400, detail=f"Unsafe SQL: {e}")

    # 3) Execute SQL
    try:
        with ENGINE.connect() as conn:
            res = conn.execute(text(safe_sql))
            rows = res.fetchall()
            cols = list(res.keys())
    except Exception as e:
        log.exception("SQL execution failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    # 4) Summarize results
    preview = rows_to_preview(rows, cols)
    stats_hint = quick_stats(rows, cols)
    try:
        summary = llm_summarize(q.query, preview, stats_hint)
    except Exception as e:
        log.warning(f"Summary failed, using preview only: {e}")
        summary = preview

    summary = scrub_schema_mentions(summary)

    # (Optional) simple chart suggestion (very lightweight heuristic)
    chart_data = None
    chart_type = None
    chart_meta = None
    if rows and len(cols) >= 2:
        # try to form (label,value) or (date,value)
        df = pd.DataFrame(rows, columns=cols)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            val_col = num_cols[0]
            # pick a likely label axis
            label_col = None
            for c in df.columns:
                if c == val_col:
                    continue
                if df[c].dtype == "O" or "date" in c.lower() or "year" in c.lower():
                    label_col = c
                    break
            if label_col:
                # small sample for client plotting
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
# Local dev runner (Railway ignores this)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3000)
