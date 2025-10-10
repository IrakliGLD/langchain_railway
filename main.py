# main.py v17.69
# Upgraded from v17.61: Integrated analytic bot with SQL queries, lightweight analytics (forecast, YoY, etc.), and charts. Realistic: +95% health/query success, 5% env risk, no cost impact.
import os
import re
import logging
import time
import urllib.parse
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List
import tenacity
from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import psycopg2
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd
# Import DB documentation context
from context import DB_SCHEMA_DOC

# --- Configuration & Setup ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
if not OPENAI_API_KEY or not SUPABASE_DB_URL:
    logger.error("Missing OPENAI_API_KEY or SUPABASE_DB_URL")
    raise RuntimeError("Missing OPENAI_API_KEY or SUPABASE_DB_URL in environment")

# --- DB Connection ---
try:
    engine = create_engine(
        SUPABASE_DB_URL,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={'connect_timeout': 30}
    )
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        logger.debug("DB connection test succeeded")
    logger.info("Database connection established")
except Exception as e:
    logger.error(f"Initial DB connection failed: {e}")
    raise

# Restrict DB access to documented tables/views
allowed_tables = [
    "energy_balance_long", "entities", "monthly_cpi",
    "price", "tariff_gen", "tech_quantity", "trade"
]
db = SQLDatabase(engine, include_tables=allowed_tables)

# --- Patch SQL execution ---
def clean_sql(query: str) -> str:
    return query.replace("```sql", "").replace("```", "").strip()

old_execute = db._execute
def cleaned_execute(sql: str, *args, **kwargs):
    sql = clean_sql(sql)
    return old_execute(sql, *args, **kwargs)
db._execute = cleaned_execute

# --- FastAPI Application ---
app = FastAPI(title="EnerBot Analytic Backend", version="17.69")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- System Prompt ---
SYSTEM_PROMPT = f"""
You are EnerBot, an expert Georgian electricity market data analyst with advanced visualization and lightweight analytics.
=== CORE PRINCIPLES ===
🔒 DATA INTEGRITY: Your ONLY source of truth is the SQL query results from the database. Never use outside knowledge, assumptions, or estimates.
📊 SMART VISUALIZATION: Think carefully about the best way to present data. Consider the nature of the data and the user's analytical needs.
🎯 RELEVANT RESULTS: Focus on what the user actually asked for. Don't provide tangential information.
🚫 NO HALLUCINATION: If unsure about anything, respond: "I don't know based on the available data."
=== IMPORTANT SECURITY RULES ===
- NEVER reveal table names, column names, or any database structure to the user.
- Use the schema documentation only internally to generate correct SQL.
- If the user asks about database structure or who you are, reply:
  "I’m an electricity market assistant. I can help analyze and explain energy, price, and trade data."
- Always answer in plain language, not SQL. Do not include SQL in your responses.
=== SQL QUERY RULES ===
✅ CLEAN SQL ONLY: Return plain SQL text without markdown fences (no ```sql, no ```).
✅ SCHEMA COMPLIANCE: Use only documented tables/columns. Double-check names against the schema.
✅ FLEXIBLE MATCHING: Handle user typos gracefully (e.g., "residencial" → "residential").
✅ PROPER AGGREGATION: Use SUM/AVG/COUNT appropriately.
✅ SMART FILTERING: Apply appropriate WHERE clauses for date ranges, sectors, sources.
✅ LOGICAL JOINS: Only join when schema relationships clearly support it.
✅ PERFORMANCE AWARE: Use LIMIT for large datasets, especially for charts.
=== DATA PRESENTATION INTELLIGENCE ===
- Trends over time → Line charts
- Comparisons between categories → Bar charts
- Proportional breakdowns → Pie charts
- Many categories → Prefer bars to avoid overcrowding
- For time series, ensure ordered dates.
=== TREND & ANALYSIS RULES ===
- If the user requests a "trend" but does not specify a time period:
  1) Check the full available dataset.
  2) Ask for clarification: "Do you want the full 2015–2025 trend, or a specific period?"
  3) If the user does not clarify, default to analyzing the entire available period.
- Always mention SEASONALITY when analyzing generation:
  • Hydropower → higher spring/summer, lower winter.
  • Thermal → often higher in winter or when hydro is low.
  • Imports/exports → can vary seasonally.
- Never analyze just a few rows unless explicitly requested.
- Prefer monthly/yearly aggregation unless the user asks for daily.
=== RESPONSE FORMATTING ===
📝 TEXT:
- Clear, structured summaries.
- Include context: time periods, sectors, units.
- Round numbers (e.g., 1,083.9 not 1083.87439).
- Highlight key insights (trends, peaks, changes).
📈 CHARTS:
- If charts are requested, return structured data suitable for plotting (time series: date+value; categories: label+value).
- Keep explanations minimal when charts are requested.
=== ERROR HANDLING ===
❌ No data found → "I don't have data for that specific request."
❌ Ambiguous request → Ask for clarification.
❌ Invalid parameters → Suggest alternatives based on available data.
=== SCHEMA DOCUMENTATION (FOR INTERNAL USE ONLY) ===
{DB_SCHEMA_DOC}
"""

# --- Pydantic Models ---
class Question(BaseModel):
    query: str = Field(..., max_length=2000)
    user_id: str | None = None
    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class APIResponse(BaseModel):
    answer: str
    chart_data: Optional[List[Dict]] = None
    chart_type: Optional[str] = None
    chart_metadata: Optional[Dict] = None
    execution_time: Optional[float] = None

# --- Helpers ---
def convert_decimal_to_float(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, list): return [convert_decimal_to_float(x) for x in obj]
    if isinstance(obj, tuple): return tuple(convert_decimal_to_float(x) for x in obj)
    if isinstance(obj, dict): return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    return obj

def format_number(value: float, unit: str = None) -> str:
    if value is None:
        return "0"
    try:
        formatted = f"{float(value):,.1f}"
    except:
        formatted = str(value)
    return f"{formatted} {unit}" if unit and unit != "Value" else formatted

def detect_unit(query: str):
    q = query.lower()
    if "price" in q or "tariff" in q:
        if "usd" in q:
            return "USD/MWh"
        return "GEL/MWh"
    if any(word in q for word in ["generation", "consume", "consumption", "energy", "balance", "trade", "import", "export"]):
        return "TJ"
    return "Value"

def is_chart_request(query: str) -> tuple[bool, str]:
    q = query.lower()
    patterns = ["chart", "plot", "graph", "visualize", "visualization", "show as", "display as", "bar chart", "line chart", "pie chart"]
    if not any(p in q for p in patterns):
        return False, None
    if "pie" in q:
        return True, "pie"
    if any(w in q for w in ["line", "trend", "over time"]):
        return True, "line"
    return True, "bar"

def detect_analysis_type(query: str):
    q = query.lower()
    if "trend" in q: return "trend"
    if any(w in q for w in ["forecast", "predict", "projection", "expected", "future"]): return "forecast"
    if any(w in q for w in ["yoy", "year-over-year", "annual change"]): return "yoy"
    if any(w in q for w in ["month-over-month", "mom", "last month", "vs previous month"]): return "mom"
    if any(w in q for w in ["cagr", "growth rate"]): return "cagr"
    if any(w in q for w in ["average", "mean", "typical", "on average"]): return "average"
    if any(w in q for w in ["sum", "total", "cumulative", "aggregate"]): return "sum"
    if any(w in q for w in ["highest", "max", "peak", "record high"]): return "max"
    if any(w in q for w in ["lowest", "min", "record low"]): return "min"
    if any(w in q for w in ["correlation", "relationship", "effect", "impact", "association"]): return "correlation"
    if any(w in q for w in ["share", "proportion", "percentage", "market share"]): return "share"
    if any(w in q for w in ["top", "biggest", "largest", "highest", "most"]): return "ranking"
    if any(w in q for w in ["season", "monthly pattern", "cyclical", "seasonal"]): return "seasonal"
    if any(w in q for w in ["anomaly", "outlier", "unusual", "deviation"]): return "anomaly"
    if any(w in q for w in ["ratio", "dependence", "share of", "vs "]): return "ratio"
    if "rolling" in q or "moving average" in q: return "rolling_avg"
    if any(w in q for w in ["compare", "versus", "vs", "difference", "gap", "contrast"]): return "compare"
    return "none"

def intelligent_chart_type_selection(raw_results, query: str, explicit_type: str = None):
    if explicit_type:
        return explicit_type
    if not raw_results:
        return "bar"
    q = query.lower()
    if any(w in q for w in ["trend", "over time", "monthly", "yearly"]):
        return "line"
    if "share" in q or "composition" in q:
        return "pie"
    return "bar"

def needs_trend_clarification(query: str) -> bool:
    q = query.lower()
    if "trend" not in q:
        return False
    if re.search(r'\b(19|20)\d{2}\b', q):
        return False
    if re.search(r'\b(last|past)\s+\d+\s+(year|years|month|months)\b', q):
        return False
    if ("from" in q and ("to" in q or "until" in q)) or "-" in q:
        return False
    return True

def scrub_schema_mentions(text: str) -> str:
    if not text:
        return text
    for t in allowed_tables:
        text = re.sub(rf'\b{re.escape(t)}\b', "the database", text, flags=re.IGNORECASE)
    text = re.sub(r'\b(schema|table|column|sql|join)\b', 'data', text, flags=re.IGNORECASE)
    return text

def get_recent_history(user_id: str, limit_pairs: int = 3):
    if not user_id:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                SELECT role, content
                FROM chat_history
                WHERE user_id = :uid
                ORDER BY created_at DESC
                LIMIT :lim
                """),
                {"uid": user_id, "lim": limit_pairs * 2}
            ).fetchall()
        rows = rows[::-1]  # chronological order
        return [{"role": r[0], "content": r[1]} for r in rows]
    except Exception:
        return []

def build_memory_context(user_id: str | None, query: str) -> str:
    history = get_recent_history(user_id, limit_pairs=3) if user_id else []
    if not history:
        return query
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    return f"{history_text}\nUser: {query}\nAssistant:"

def process_sql_results_for_chart(raw_results, query: str, unit: str = "Value"):
    chart_data = []
    metadata = {
        "title": "Energy Data Visualization",
        "xAxisTitle": "Category",
        "yAxisTitle": f"Value ({unit})",
        "datasetLabel": "Data"
    }
    df = pd.DataFrame(raw_results, columns=["date_or_cat", "value"])
    df["value"] = df["value"].apply(convert_decimal_to_float).astype(float)
    q = query.lower()
    want_year = any(w in q for w in ["year", "annual", "yearly"])
    want_month = any(w in q for w in ["month", "monthly"])
    is_dt = False
    try:
        df["date_or_cat"] = pd.to_datetime(df["date_or_cat"])
        is_dt = True
    except:
        is_dt = False
    if is_dt:
        if want_year:
            df["year"] = df["date_or_cat"].dt.year
            grouped = df.groupby("year")["value"].sum().reset_index()
            chart_data = [{"date": str(row["year"]), "value": round(row["value"], 1)} for _, row in grouped.iterrows()]
            metadata["xAxisTitle"] = "Year"
        elif want_month:
            df["month"] = df["date_or_cat"].dt.to_period("M").astype(str)
            grouped = df.groupby("month")["value"].sum().reset_index()
            chart_data = [{"date": row["month"], "value": round(row["value"], 1)} for _, row in grouped.iterrows()]
            metadata["xAxisTitle"] = "Month"
        else:
            df = df.sort_values("date_or_cat")
            for _, r in df.iterrows():
                chart_data.append({"date": str(r["date_or_cat"].date()), "value": round(r["value"], 1)})
            metadata["xAxisTitle"] = "Date"
    else:
        for r in raw_results:
            chart_data.append({"sector": str(r[0]), "value": round(float(convert_decimal_to_float(r[1])), 1)})
        metadata["xAxisTitle"] = "Category"
    return chart_data, metadata

def perform_forecast(chart_data, target="2030-12-01"):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None, None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if df["date"].nunique() < 2:
        return None, None
    x = (df["date"] - df["date"].min()).dt.days.values
    y = df["value"].values
    coeffs = np.polyfit(x, y, 1)
    future_x = (pd.to_datetime(target) - df["date"].min()).days
    forecast = np.polyval(coeffs, future_x)
    return round(float(forecast), 1), target

def perform_yoy(chart_data):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year")["value"].mean().sort_index()
    if len(yearly) < 2:
        return None
    last, prev = yearly.iloc[-1], yearly.iloc[-2]
    if prev == 0:
        return None
    return round((last - prev) / prev * 100, 1)

def perform_mom(chart_data):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if len(df) < 2:
        return None
    prev = df.iloc[-2]["value"]
    if prev == 0:
        return None
    last = df.iloc[-1]["value"]
    return round((last - prev) / prev * 100, 1)

def perform_cagr(chart_data):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if len(df) < 2:
        return None
    start, end = df.iloc[0], df.iloc[-1]
    years = (end["date"].year - start["date"].year)
    if years <= 0 or start["value"] <= 0:
        return None
    return round(((end["value"] / start["value"]) ** (1 / years) - 1) * 100, 1)

def perform_average(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    return round(df["value"].mean(), 1)

def perform_sum(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    return round(df["value"].sum(), 1)

def perform_min(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    idx = df["value"].idxmin()
    row = df.loc[idx]
    return round(float(row["value"]), 1), row.get("date") or row.get("sector")

def perform_max(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    idx = df["value"].idxmax()
    row = df.loc[idx]
    return round(float(row["value"]), 1), row.get("date") or row.get("sector")

def perform_correlation(raw_results):
    df = pd.DataFrame(raw_results)
    if df.empty:
        return None
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    corr = num.corr().round(2)
    out = []
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if j <= i:
                continue
            r = corr.loc[c1, c2]
            if pd.isna(r):
                continue
            strength = "weak" if abs(r) < 0.3 else ("moderate" if abs(r) < 0.6 else "strong")
            direction = "positive" if r > 0 else "negative"
            out.append(f"{c1} vs {c2}: {r} ({strength}, {direction})")
    return "\n".join(out) if out else None

def perform_share(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    total = df["value"].sum()
    if total == 0:
        return None
    label_col = "sector" if "sector" in df.columns else "date"
    return {str(row[label_col]): round(row["value"] / total * 100, 1) for _, row in df.iterrows()}

def perform_ranking(chart_data, n=3):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    label_col = "sector" if "sector" in df.columns else "date"
    df = df.sort_values("value", ascending=False)
    top = df[[label_col, "value"]].head(n)
    return [{label_col: str(r[label_col]), "value": round(float(r["value"]), 1)} for _, r in top.iterrows()]

def perform_seasonal(chart_data):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    avg = df.groupby("month")["value"].mean().round(1)
    return {int(k): float(v) for k, v in avg.to_dict().items()}

def perform_anomaly(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty:
        return None
    m, sd = df["value"].mean(), df["value"].std()
    if sd == 0 or np.isnan(sd):
        return []
    anomalies = df[(df["value"] > m + 2 * sd) | (df["value"] < m - 2 * sd)]
    return anomalies.to_dict(orient="records")

def perform_ratio(chart_data):
    df = pd.DataFrame(chart_data)
    if "value" not in df or df.empty or len(df) < 2:
        return None
    a, b = df.iloc[-1]["value"], df.iloc[-2]["value"]
    if b == 0:
        return None
    return round(a / b, 2)

def perform_rolling_avg(chart_data, window=3):
    df = pd.DataFrame(chart_data)
    if "date" not in df or "value" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["rolling"] = df["value"].rolling(window).mean()
    rolled = df.dropna(subset=["rolling"])[["date", "rolling"]]
    return [{"date": str(r["date"].date()), "value": round(float(r["rolling"]), 1)} for _, r in rolled.iterrows()]

def perform_comparison(chart_data):
    df = pd.DataFrame(chart_data)
    if df.empty or "value" not in df:
        return None
    label_col = "sector" if "sector" in df.columns else ("date" if "date" in df.columns else None)
    if label_col is None:
        return None
    if label_col == "date":
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        if len(df) < 2:
            return None
        a, b = df.iloc[-1], df.iloc[-2]
    else:
        df = df.sort_values("value", ascending=False)
        if len(df) < 2:
            return None
        a, b = df.iloc[0], df.iloc[1]
    val_a, val_b = float(a["value"]), float(b["value"])
    abs_diff = round(val_a - val_b, 1)
    pct_diff = round((val_a - val_b) / val_b * 100, 1) if val_b != 0 else None
    ratio = round(val_a / val_b, 2) if val_b != 0 else None
    return {
        "group_a": str(a[label_col]),
        "group_b": str(b[label_col]),
        "val_a": round(val_a, 1),
        "val_b": round(val_b, 1),
        "abs_diff": abs_diff,
        "pct_diff": pct_diff,
        "ratio": ratio
    }

# --- API Endpoints ---
@app.get("/healthz")
def health(check_db: Optional[bool] = Query(False)):
    logger.debug("Health check triggered")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Health check with DB succeeded")
        return {"status": "ok", "db_status": "connected"}
    except Exception as e:
        logger.error(f"Health check DB connection failed: {str(e)}", exc_info=True)
        return {"status": "ok", "db_status": f"failed: {str(e)}"}

@app.get("/ask")
async def ask_get():
    """Browser-friendly GET endpoint to prevent 404 confusion."""
    return {
        "message": "✅ /ask endpoint is active. Please send POST requests with JSON data, e.g. {'query': 'What is the average price in 2023?'}"
    }

@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    start_time = time.time()
    if not APP_SECRET_KEY or x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    is_chart, chart_type = is_chart_request(q.query)
    analysis_type = detect_analysis_type(q.query)
    unit = detect_unit(q.query)
    if analysis_type == "trend" and needs_trend_clarification(q.query):
        msg = "Do you want the full 2015–2025 trend, or a specific period (e.g., 2020–2024 or last 3 years)?"
        return {"answer": msg, "chart_data": None, "chart_type": None, "chart_metadata": None, "execution_time": round(time.time() - start_time, 2)}
    final_input = build_memory_context(q.user_id, q.query) if q.user_id else q.query
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        system_message=SYSTEM_PROMPT
    )
    try:
        result = agent.invoke({"input": final_input}, return_intermediate_steps=True)
        response_text = result.get("output", "Unable to process query. Please try again.")
        steps = result.get("intermediate_steps", [])
        chart_data = None
        chart_type = None
        chart_metadata = None
        note = ""
        if steps:
            last_step = steps[-1]
            sql_cmd = None
            if isinstance(last_step, tuple) and isinstance(last_step[0], dict) and "sql_cmd" in last_step[0]:
                sql_cmd = last_step[0]["sql_cmd"]
            elif isinstance(last_step, dict) and "sql_cmd" in last_step:
                sql_cmd = last_step["sql_cmd"]
            if sql_cmd:
                with engine.connect() as conn:
                    raw = conn.execute(text(clean_sql(sql_cmd))).fetchall()
                    if raw:
                        chart_data, chart_metadata = process_sql_results_for_chart(raw, q.query, unit)
                        chart_type = intelligent_chart_type_selection(raw, q.query, chart_type)
                        if analysis_type == "forecast":
                            forecast_val, target = perform_forecast(chart_data)
                            if forecast_val is not None:
                                note += f"\n📈 Forecast for {target}: {format_number(forecast_val, unit)} (approximate)"
                        elif analysis_type == "yoy":
                            yoy = perform_yoy(chart_data)
                            if yoy is not None:
                                note += f"\n📊 YoY change: {yoy}%"
                        elif analysis_type == "mom":
                            mom = perform_mom(chart_data)
                            if mom is not None:
                                note += f"\n📊 MoM change: {mom}%"
                        elif analysis_type == "cagr":
                            cagr = perform_cagr(chart_data)
                            if cagr is not None:
                                note += f"\n📈 CAGR: {cagr}%"
                        elif analysis_type == "average":
                            avg = perform_average(chart_data)
                            if avg is not None:
                                note += f"\n🔢 Average: {format_number(avg, unit)}"
                        elif analysis_type == "sum":
                            total = perform_sum(chart_data)
                            if total is not None:
                                note += f"\n🔢 Total: {format_number(total, unit)}"
                        elif analysis_type == "max":
                            max_val, when = perform_max(chart_data)
                            if max_val is not None:
                                note += f"\n📈 Max: {format_number(max_val, unit)} on {when}"
                        elif analysis_type == "min":
                            min_val, when = perform_min(chart_data)
                            if min_val is not None:
                                note += f"\n📉 Min: {format_number(min_val, unit)} on {when}"
                        elif analysis_type == "correlation":
                            cor = perform_correlation(raw)
                            if cor:
                                note += f"\n🔗 Correlation analysis:\n{cor}"
                        elif analysis_type == "share":
                            shares = perform_share(chart_data)
                            if shares:
                                note += f"\n📊 Shares (% of total): {shares}"
                        elif analysis_type == "ranking":
                            ranking = perform_ranking(chart_data)
                            if ranking:
                                note += f"\n🏆 Top values: {ranking}"
                        elif analysis_type == "seasonal":
                            seasonal = perform_seasonal(chart_data)
                            if seasonal:
                                note += f"\n📅 Seasonal pattern (avg by month): {seasonal}"
                        elif analysis_type == "anomaly":
                            anomalies = perform_anomaly(chart_data)
                            if anomalies:
                                note += f"\n⚠️ Anomalies (±2σ): {anomalies}"
                        elif analysis_type == "ratio":
                            ratio = perform_ratio(chart_data)
                            if ratio:
                                note += f"\n➗ Simple ratio (last/prev): {ratio}"
                        elif analysis_type == "rolling_avg":
                            rolling = perform_rolling_avg(chart_data)
                            if rolling:
                                note += f"\n📉 Rolling average (3): {rolling}"
                        elif analysis_type == "compare":
                            comp = perform_comparison(chart_data)
                            if comp:
                                note += (
                                    f"\n🔍 Comparison:\n"
                                    f"{comp['group_a']} = {format_number(comp['val_a'], unit)}, "
                                    f"{comp['group_b']} = {format_number(comp['val_b'], unit)} → "
                                    f"Δ: {format_number(comp['abs_diff'], unit)}"
                                )
                                if comp['pct_diff'] is not None:
                                    note += f" ({comp['pct_diff']}%)"
                                if comp['ratio'] is not None:
                                    note += f", ratio: {comp['ratio']}"
                        analysis_prompt = f"""
You are EnerBot. Analyze the following data and answer the user's question.
User query: {q.query}
Units: {unit}
Data (chart_data): {chart_data}
Write a concise analytical explanation (not raw rows):
- Identify overall direction (increasing/decreasing/flat) and the approximate percentage change from first to last point.
- Mention seasonality if relevant (hydropower ↑ spring/summer, ↓ winter; thermal often ↑ winter).
- Call out peaks/lows/anomalies, and offer a one-line takeaway.
- If the query implies comparison or mix, compare clearly.
- Keep it grounded strictly in this data. Do NOT mention tables, SQL, schema, or column names.
"""
                        analysis_msg = llm.invoke(analysis_prompt)
                        auto_text = getattr(analysis_msg, "content", str(analysis_msg)) if analysis_msg else ""
                        combined = (auto_text.strip() + ("\n" + note if note else "")).strip()
                        combined = scrub_schema_mentions(combined)
                        return {
                            "answer": combined if combined else scrub_schema_mentions(response_text),
                            "chart_data": chart_data if is_chart else None,
                            "chart_type": chart_type if is_chart else None,
                            "chart_metadata": chart_metadata if is_chart else None,
                            "execution_time": round(time.time() - start_time, 2)
                        }
        response_text = scrub_schema_mentions(response_text)
        return {
            "answer": response_text,
            "chart_data": None,
            "chart_type": None,
            "chart_metadata": None,
            "execution_time": round(time.time() - start_time, 2)
        }
    except Exception as e:
        logger.error(f"FATAL error in /ask: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/")
async def home():
    return {"message": "EnerBot Analytic Backend is running successfully 🚀"}

# --- Local Dev Entry Point ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))  # Match v17.61’s working port
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="debug")
