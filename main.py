# main.py v18.3 — Gemini Analyst (SQL validation + auto-repair + schema awareness)
import os
import re
import json
import time
import logging
import urllib.parse
from typing import Optional, Dict, Any, List, Tuple
from difflib import get_close_matches  # 🆕 for column hinting

from fastapi import FastAPI, HTTPException, Header, Query, Request
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

# Schema & helpers
from context import DB_SCHEMA_DOC, scrub_schema_mentions, COLUMN_LABELS  # 🆕 import COLUMN_LABELS

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
app = FastAPI(title="EnerBot Analyst (Gemini)", version="18.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# -----------------------------
# Column Validation Helpers 🆕
# -----------------------------
def closest_match(col: str) -> str:
    matches = get_close_matches(col, COLUMN_LABELS.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None


def validate_columns_exist(sql: str):
    """Validate all referenced columns exist in schema."""
    all_cols = {c.lower() for c in COLUMN_LABELS.keys()}
    tokens = re.findall(r"\b[a-z_]+\b", sql.lower())

    ignore = {
        "select", "from", "where", "group", "by", "order", "limit", "join", "on", "as",
        "sum", "avg", "min", "max", "count", "date", "year", "month", "extract",
        "and", "or", "not", "case", "when", "then", "else", "end", "distinct",
        "xrate", "p_bal_gel", "p_dereg_gel", "p_gcap_gel", "tariff_gel"
    }

    for tok in tokens:
        if tok in ignore or tok.endswith("_usd") or tok.endswith("_gel"):
            continue
        if tok not in all_cols:
            suggestion = closest_match(tok)
            hint = f" — did you mean `{suggestion}`?" if suggestion else ""
            raise HTTPException(
                status_code=400,
                detail=f"Unknown column `{tok}`{hint}"
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
from tenacity import retry, stop_after_attempt, wait_exponential

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
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0
            )
        else:
            llm = ChatOpenAI(
                model=OPENAI_MODEL,
                temperature=0,
                openai_api_key=OPENAI_API_KEY
            )
        sql = llm.invoke([("system", system), ("user", prompt)]).content.strip()
    except Exception as e:
        log.warning(f"Gemini failed, fallback to OpenAI: {e}")
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
        sql = llm.invoke([("system", system), ("user", prompt)]).content.strip()

    return sql


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "EnerBot Analyst v18.3 is running 🚀"}

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
def ask_post(q: Question, x_app_key: str = Header(..., alias="X-App-Key")):
    t0 = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1️⃣ Generate SQL
    try:
        raw_sql = llm_generate_sql(q.query)
    except Exception as e:
        log.exception("SQL generation failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {e}")

    # 2️⃣ Sanitize and validate
    try:
        safe_sql = sanitize_sql(raw_sql)
        validate_columns_exist(safe_sql)  # 🆕 validate before execution
        log.info(f"✅ Safe SQL passed validation: {safe_sql}")
    except Exception as e:
        log.warning(f"Rejected SQL: {raw_sql}")
        raise HTTPException(status_code=400, detail=f"Unsafe or invalid SQL: {e}")

    # 3️⃣ Execute with auto-repair fallback 🆕
    try:
        with ENGINE.connect() as conn:
            res = conn.execute(text(safe_sql))
            rows = res.fetchall()
            cols = list(res.keys())
    except Exception as e:
        if "UndefinedColumn" in str(e) and "Perhaps you meant" in str(e):
            hint = re.search(r'HINT:.*?"([^"]+)"', str(e))
            if hint:
                fixed_col = hint.group(1)
                safe_sql = re.sub(r"\b[a-z_]+\b", fixed_col, safe_sql, count=1)
                log.warning(f"🔁 Auto-corrected column name to {fixed_col}")
                with ENGINE.connect() as conn:
                    res = conn.execute(text(safe_sql))
                    rows = res.fetchall()
                    cols = list(res.keys())
            else:
                raise HTTPException(status_code=400, detail=f"Column not found: {e}")
        else:
            log.exception("SQL execution failed")
            raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    # 4️⃣ Summarize
    preview = rows_to_preview(rows, cols)
    stats_hint = quick_stats(rows, cols)
    try:
        summary = llm_summarize(q.query, preview, stats_hint)
    except Exception as e:
        log.warning(f"Summary failed: {e}")
        summary = preview

    summary = scrub_schema_mentions(summary)

    # 5️⃣ Chart builder (unchanged)
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
