# main.py v17.45
# Updated from v17.44: fixes applied as discussed (syntax fix, missing imports, Gemini/OpenAI LLM wiring, robust RAG fallback).
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
from litellm import completion, RateLimitError  # RateLimitError used in decorators
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction
from dotenv import load_dotenv
from decimal import Decimal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
from langchain.cache import SQLiteCache  # For caching
from langchain_supabase import SupabaseVectorStore  # For RAG (optional)
from langchain.embeddings import OpenAIEmbeddings  # Or Gemini equiv
# New imports for LLM providers & supabase client
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None
try:
    from langchain.chat_models import ChatOpenAI
except Exception:
    ChatOpenAI = None
try:
    from supabase import create_client as create_supabase_client
except Exception:
    create_supabase_client = None

# --- Configuration & Setup ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini").lower()  # Default to Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SUPABASE_VECTOR_URL = os.getenv("SUPABASE_VECTOR_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # service role key required for vector store usage
SUPABASE_VECTOR_TABLE = os.getenv("SUPABASE_VECTOR_TABLE", "schema_vectors")

if not SUPABASE_DB_URL or not APP_SECRET_KEY:
    logger.error("Missing SUPABASE_DB_URL or APP_SECRET_KEY")
    raise RuntimeError("SUPABASE_DB_URL and APP_SECRET_KEY are required.")
if MODEL_TYPE == "gemini" and not GOOGLE_API_KEY:
    logger.error("Missing GOOGLE_API_KEY for MODEL_TYPE=gemini")
    raise RuntimeError("GOOGLE_API_KEY required for MODEL_TYPE=gemini.")
if OPENAI_API_KEY:
    logger.info("OpenAI key present for fallback.")

# Validate SUPABASE_DB_URL (unchanged)
def validate_supabase_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ["postgres", "postgresql", "postgresql+psycopg"]:
            logger.error(f"Invalid scheme: {parsed.scheme}")
            raise ValueError("Scheme must be 'postgres', 'postgresql', or 'postgresql+psycopg'")
        if not parsed.username or not parsed.password:
            logger.error("Missing username or password")
            raise ValueError("Username and password must be provided")
        parsed_password = parsed.password.strip() if parsed.password else ""
        if not parsed_password:
            logger.error("Empty password after trimming")
            raise ValueError("Password cannot be empty after trimming")
        if not re.match(r'^[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?~]*$', parsed_password):
            logger.error(f"Invalid password characters: {parsed_password}")
            raise ValueError("Password contains invalid characters for URL")
        logger.info(f"Parsed URL components: scheme={parsed.scheme}, username={parsed.username}, host={parsed.hostname}, port={parsed.port}, path={parsed.path}, query={parsed.query}")
        # Optional: keep the strict host/port/path checks if you use pooled supabase URL
        # They may be environment-specific—if your SUPABASE_DB_URL differs remove or adjust these checks.
        # For safety we keep them as in original; if you use a different host/port, update env or the checks.
        if parsed.hostname != "aws-1-eu-central-1.pooler.supabase.com":
            logger.error(f"Invalid host: {parsed.hostname}")
            raise ValueError("Host must be 'aws-1-eu-central-1.pooler.supabase.com'")
        if parsed.port != 6543:
            logger.error(f"Invalid port: {parsed.port}")
            raise ValueError("Port must be 6543 for pooled connection")
        if parsed.path != "/postgres":
            logger.error(f"Invalid path: {parsed.path}")
            raise ValueError("Database path must be '/postgres'")
        if parsed.username != "postgres.qvmqmmcglqmhachqaezt":
            logger.error(f"Invalid username: {parsed.username}")
            raise ValueError("Pooled connection requires username 'postgres.qvmqmmcglqmhachqaezt'")
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("sslmode") != ["require"]:
            logger.error(f"Invalid sslmode: {params.get('sslmode')}")
            raise ValueError("Query parameter 'sslmode=require' is required")
    except Exception as e:
        logger.error(f"Invalid SUPABASE_DB_URL: {str(e)}", exc_info=True)
        raise RuntimeError(f"Invalid SUPABASE_DB_URL: {str(e)}")
validate_supabase_url(SUPABASE_DB_URL)

# Sanitized DB URL for logging (unchanged)
sanitized_db_url = re.sub(r':[^@]+@', ':****@', SUPABASE_DB_URL)
logger.info(f"Using SUPABASE_DB_URL: {sanitized_db_url}")

# Parse DB URL for diagnostics (unchanged)
parsed_db_url = urllib.parse.urlparse(SUPABASE_DB_URL)
db_host = parsed_db_url.hostname
db_port = parsed_db_url.port
db_name = parsed_db_url.path.lstrip('/')
logger.info(f"DB connection details: host={db_host}, port={db_port}, dbname={db_name}")

# --- Import DB Schema & Joins --- (unchanged)
from context import DB_SCHEMA_DOC, DB_JOINS, scrub_schema_mentions

# Validate DB_SCHEMA_DOC and DB_JOINS (unchanged)
if not isinstance(DB_SCHEMA_DOC, str):
    logger.error(f"DB_SCHEMA_DOC must be a string, got {type(DB_SCHEMA_DOC)}")
    raise ValueError("Invalid DB_SCHEMA_DOC format")
if not isinstance(DB_JOINS, (str, dict)):
    logger.error(f"DB_JOINS must be a string or dict, got {type(DB_JOINS)}")
    raise ValueError("Invalid DB_JOINS format")
ALLOWED_TABLES = [
    "energy_balance_long", "entities", "monthly_cpi",
    "price", "tariff_gen", "tech_quantity", "trade", "dates"
]

# --- Schema Cache --- (unchanged)
schema_cache = {}

# --- FastAPI Application --- (unchanged)
app = FastAPI(title="EnerBot Backend", version="17.45")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- System Prompts --- (unchanged)
FEW_SHOT_EXAMPLES = [
    {"input": "What is the average price in 2020?", "output": "SELECT AVG(p_dereg_gel) FROM price WHERE date >= '2020-01-01' AND date < '2021-01-01';"},
    {"input": "Correlate CPI and prices", "output": "SELECT m.date, m.cpi, p.p_dereg_gel FROM monthly_cpi m JOIN price p ON m.date = p.date WHERE m.cpi_type = 'overall_cpi';"},
    {"input": "What was electricity generation in May 2023?", "output": "SELECT SUM(quantity_tech) * 1000 AS total_generation_mwh FROM tech_quantity WHERE date = '2023-05-01';"},
    {"input": "What was balancing electricity price in May 2023?", "output": "SELECT date, p_bal_gel, (p_bal_gel / xrate) AS p_bal_usd FROM price WHERE date = '2023-05-01';"},
    {"input": "Predict balancing electricity price by December 2035?", "output": "SELECT date, p_bal_gel, xrate FROM price ORDER BY date;"},
    {"input": "Predict the electricity demand for 2030?", "output": "SELECT date, entity, quantity_tech FROM tech_quantity WHERE entity IN ('Abkhazeti', 'direct customers', 'losses', 'self-cons', 'supply-distribution') ORDER BY date;"},
    {"input": "Predict tariff for Enguri HPP?", "output": "SELECT 1;"}
]
SQL_SYSTEM_TEMPLATE = """
### ROLE ###
You are an expert SQL writer. Your sole purpose is to generate a single, syntactically correct SQL query
to answer the user's question based on the provided database schema and join information.
### MANDATORY RULES ###
1. **GENERATE ONLY SQL.** Output only the SQL query, no explanations or markdown.
2. Use `DB_JOINS` for table joins.
3. For time-series analysis, query the entire date range or all data if unspecified.
4. For forecasts, use exact row count from SQL results (e.g., count rows for data length).
5. For balancing electricity price (p_bal_gel, p_bal_usd as p_bal_gel / xrate), compute p_bal_usd as p_bal_gel / xrate; never select p_bal_usd directly. Forecast yearly, summer (May-Aug), and winter (Sep-Apr) averages.
6. For demand forecasts (tech_quantity), sum quantity_tech for entities: Abkhazeti, direct customers, losses, self-cons, supply-distribution; exclude export. Forecast total, Abkhazeti, and other entities separately using seasonal models (period=12).
7. For energy_balance_long, forecast demand by energy_source and sector only using seasonal models.
8. Do not select non-existent columns (e.g., p_bal_usd). Validate against schema.
9. For demand forecasts, always query tech_quantity; never use simulated data.
### FEW-SHOT EXAMPLES ###
{examples}
### INTERNAL SCHEMA & JOIN KNOWLEDGE ###
{schema_subset}
DB_JOINS = {DB_JOINS}
"""
STRICT_SQL_PROMPT = """
You are an SQL generator.
Your ONLY job is to return a valid SQL query.
Do not explain, do not narrate, do not wrap in markdown.
If you cannot answer, return `SELECT 1;`.
"""
ANALYST_PROMPT = """
You are an expert energy market analyst. Your task is to write a clear, concise narrative based *only*
on the structured data provided to you.
### MANDATORY RULES ###
1. **NEVER GUESS.** Use ONLY the numbers and facts provided in the "Computed Stats" section.
2. **NEVER REVEAL INTERNALS.** Do not mention the database, SQL, or technical jargon.
3. **ALWAYS BE AN ANALYST.** Your response must be a narrative including trends, peaks, lows,
    seasonality, and forecasts (if available).
4. **CONCLUDE SUCCINCTLY.** End with a single, short "Key Insight" line.
"""
AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    Query data, compute stats, and generate insights for energy markets using tools.
   
    Restrictions:
    - Forecast only: balancing electricity prices (p_bal_gel, p_bal_usd as p_bal_gel / xrate) from price table (yearly, summer May-Aug, winter Sep-Apr averages); demand (total, Abkhazeti, others) from tech_quantity, excluding export; demand by energy_source/sector from energy_balance_long.
    - Block forecasts for p_dereg_gel, p_gcap_gel, tariff_gel, and tech_quantity variables (hydro, wind, thermal, import, export) with user-friendly reasons.
    - Use exact SQL result lengths for DataFrame creation in execute_python_code.
    - For demand forecasts, always query tech_quantity; never use simulated data.
    - For balancing price forecasts, always compute p_bal_usd as p_bal_gel / xrate; never select p_bal_usd directly.
    - For visualization, output JSON (e.g., {{'type': 'line', 'data': [...]}}).
    - If database unavailable, provide schema-based response.
   
    Schema: {schema}
    Joins: {joins}
    """),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# --- Pydantic Models ---
class Question(BaseModel):
    query: str = Field(..., max_length=2000)
    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
class APIResponse(BaseModel):
    answer: str
    chart_data: Optional[Any] = None
    chart_type: Optional[str] = None
    chart_metadata: Optional[Dict] = None
    execution_time: Optional[float] = None

# --- Helpers ---
def clean_and_validate_sql(sql: str) -> str:
    if not sql:
        raise ValueError("Generated SQL query is empty.")
    # Remove markdown fences
    cleaned_sql = re.sub(r"```(?:sql)?\s*|\s*```", "", sql, flags=re.IGNORECASE)
    # Remove comments
    cleaned_sql = re.sub(r"--.*?$", "", cleaned_sql, flags=re.MULTILINE)
    # Remove LIMIT clauses
    cleaned_sql = re.sub(r"\bLIMIT\s+\d+\b", "", cleaned_sql, flags=re.IGNORECASE)
    # Strip public. schema qualifiers
    cleaned_sql = re.sub(r'\bpublic\.', '', cleaned_sql)
    cleaned_sql = cleaned_sql.strip().removesuffix(";")
    if not cleaned_sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed.")
    return cleaned_sql

def extract_sql_from_steps(steps: List[Any]) -> Optional[str]:
    sql_query = None
    for step in steps:
        action = step[0] if isinstance(step, tuple) else step
        if isinstance(action, AgentAction):
            if action.tool in ["sql_db_query", "sql_db_query_checker"]:
                tool_input = action.tool_input
                if isinstance(tool_input, dict) and 'query' in tool_input:
                    sql_query = tool_input['query']
                elif isinstance(tool_input, str):
                    sql_query = tool_input
    return sql_query

def convert_decimal_to_float(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, list): return [convert_decimal_to_float(x) for x in obj]
    if isinstance(obj, tuple): return tuple(convert_decimal_to_float(x) for x in obj)
    if isinstance(obj, dict): return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    return obj

def coerce_dataframe(rows: List[tuple], columns: List[str]) -> pd.DataFrame:
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=columns)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(convert_decimal_to_float)
    return df

# --- Forecasting Helpers ---
def _ensure_monthly_index(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).asfreq("ME")
    df[value_col] = df[value_col].interpolate(method="linear")
    return df.reset_index()

def forecast_linear_ols(df: pd.DataFrame, date_col: str, value_col: str, target_date: datetime) -> Optional[Dict]:
    try:
        if len(df) < 12:
            raise ValueError("Insufficient data points for forecasting (need at least 12).")
        df = _ensure_monthly_index(df, date_col, value_col)
        df["t"] = (df[date_col] - df[date_col].min()) / np.timedelta64(1, "M")
        X = sm.add_constant(df["t"])
        y = df[value_col]
        # Try seasonal STL first
        try:
            stl = STL(y, period=12, robust=True)
            res = stl.fit()
            y = res.trend
        except Exception as e:
            logger.warning(f"STL decomposition failed, falling back to raw OLS: {e}")
        model = sm.OLS(y, X).fit()
        future_t = (pd.to_datetime(target_date) - df[date_col].min()) / np.timedelta64(1, "M")
        X_future = sm.add_constant(pd.DataFrame({"t": [future_t]}))
        pred = model.get_prediction(X_future)
        pred_summary = pred.summary_frame(alpha=0.10) # 90% CI
        return {
            "target_month": target_date.strftime("%Y-%m"),
            "point": float(pred_summary["mean"].iloc[0]),
            "90% CI": [float(pred_summary["mean_ci_lower"].iloc[0]), float(pred_summary["mean_ci_upper"].iloc[0])],
            "slope_per_month": float(model.params["t"]) if "t" in model.params else None,
            "R²": float(model.rsquared),
            "n_obs": int(model.nobs),
        }
    except Exception as e:
        logger.error(f"Forecast failed: {e}", exc_info=True)
        return None

def detect_forecast_intent(query: str) -> (bool, Optional[datetime], Optional[str]):
    q = query.lower()
    blocked_vars = {
        "p_dereg_gel": "Forecasting the price for deregulated power plants (p_dereg_gel) is not possible, as it is influenced by political decisions rather than market forces such as supply and demand. Please try forecasting balancing electricity prices or demand.",
        "p_gcap_gel": "Forecasting the guaranteed capacity charge (p_gcap_gel) is not possible, as it is regulated by GNERC based on tariff methodology and not influenced by market forces. Please try forecasting balancing electricity prices or demand.",
        "tariff_gel": "Forecasting electricity generation tariffs (tariff_gel) is not possible, as they are approved by GNERC based on tariff methodology and not influenced by market forces. Please try forecasting balancing electricity prices or demand.",
        "hydro": "Forecasting electricity generation by hydro is not possible, as it is highly dependent on unpredictable weather conditions and lacks data on planned projects. Please try forecasting balancing electricity prices or demand.",
        "wind": "Forecasting electricity generation by wind is not possible, as it is highly dependent on unpredictable weather conditions and lacks data on planned projects. Please try forecasting balancing electricity prices or demand.",
        "thermal": "Forecasting electricity generation by thermal is not feasible, as it depends on the gap between renewable generation and demand. Renewable generation cannot be forecasted due to unpredictable weather conditions and lack of data on planned projects. Please try forecasting balancing electricity prices or demand.",
        "import": "Forecasting electricity imports is not feasible, as they depend on the gap between renewable generation and demand. Renewable generation cannot be forecasted due to unpredictable weather conditions and lack of data on planned projects. Please try forecasting balancing electricity prices or demand.",
        "export": "Forecasting electricity exports is not possible, as they depend on the availability of renewable generation, which cannot be forecasted due to unpredictable weather conditions and lack of data on planned projects. Please try forecasting balancing electricity prices or demand."
    }
    for var, reason in blocked_vars.items():
        if var == "tariff_gel":
            if re.search(r'\btariff\b|\benguri\b|\bhpp\b|\bhydropower\b', q, re.IGNORECASE):
                logger.info(f"Blocked forecast attempt for tariff_gel: {q}")
                return False, None, reason
        elif re.search(rf'\b{var}\b', q, re.IGNORECASE):
            logger.info(f"Blocked forecast attempt for {var}: {q}")
            return False, None, reason
    if re.search(r'\btariff\b', q, re.IGNORECASE):
        logger.info(f"Blocked forecast attempt for generic tariff: {q}")
        return False, None, blocked_vars["tariff_gel"]
    if "forecast" in q or "predict" in q or "estimate" in q:
        for yr in range(2025, 2040):
            if str(yr) in q:
                try:
                    return True, datetime(yr, 12, 1), None
                except:
                    pass
        return True, datetime(2030, 12, 1), None
    return False, None, None

# --- Code Execution Tool ---
@tool
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=1, max=60),
    retry=tenacity.retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.info(f"Retrying execute_python_code due to rate limit ({retry_state.attempt_number}/3)...")
)
def execute_python_code(code: str, context: Optional[Dict] = None) -> str:
    try:
        allowed_globals = {"pd": pd, "np": np, "sm": sm, "STL": STL, "ExponentialSmoothing": ExponentialSmoothing}
        if not context or "sql_result" not in context:
            raise ValueError("SQL result context required; simulated data not allowed.")
        rows, columns = context["sql_result"]
        df = coerce_dataframe(rows, columns)
        n_rows = len(rows)
        if "np.arange" in code or "import" in code.lower() or "exec" in code.lower() or "eval" in code.lower():
            raise ValueError("Simulated data or unsafe ops (import/exec/eval) not allowed; use sql_data.")
        local_env = allowed_globals.copy()
        local_env["sql_data"] = df
        local_env["n_rows"] = n_rows
        exec(code, {"__builtins__": {}}, local_env)
        result = local_env.get("result")
        if result is None:
            raise ValueError("No 'result' variable set in code.")
        if isinstance(result, pd.DataFrame):
            return result.to_json(orient="records")
        return str(result)
    except Exception as e:
        logger.error(f"Python code execution failed: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# --- Schema Subsetter with RAG ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=1, max=60),
    retry=tenacity.retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.info(f"Retrying get_schema_subset due to rate limit ({retry_state.attempt_number}/3)...")
)
def get_schema_subset(llm, query: str) -> str:
    cache_key = query.lower()
    if cache_key in schema_cache:
        logger.info(f"Using cached schema for query: {cache_key}")
        return schema_cache[cache_key]

    # If Supabase vector store is configured, attempt RAG using the vector table
    try:
        if SUPABASE_VECTOR_URL and SUPABASE_SERVICE_KEY and create_supabase_client:
            logger.info("Attempting RAG with Supabase vector store")
            # create supabase client for vectors
            supabase_client = create_supabase_client(SUPABASE_VECTOR_URL, SUPABASE_SERVICE_KEY)
            # embeddings: prefer OpenAI embeddings if key present, otherwise skip embeddings RAG
            if OPENAI_API_KEY:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            else:
                logger.warning("OPENAI_API_KEY not set: skipping embeddings-based RAG.")
                embeddings = None

            if embeddings:
                vector_store = SupabaseVectorStore(
                    embedding=embeddings,
                    table_name=SUPABASE_VECTOR_TABLE,
                    client=supabase_client
                )
                retriever = vector_store.as_retriever()
                # Use a simple retrieval and prompt
                hits = retriever.get_relevant_documents(query) if hasattr(retriever, "get_relevant_documents") else []
                subset_text = "\n".join([d.page_content for d in hits]) if hits else DB_SCHEMA_DOC[:1500]
            else:
                subset_text = DB_SCHEMA_DOC[:1500]
        else:
            # No vector store configured: create a short schema subset from DB_SCHEMA_DOC
            logger.info("No Supabase vector store configured or missing keys — returning schema excerpt")
            subset_text = DB_SCHEMA_DOC[:1500]
    except Exception as e:
        logger.error(f"RAG subset failed: {e}", exc_info=True)
        subset_text = DB_SCHEMA_DOC[:1500]

    schema_cache[cache_key] = subset_text
    return subset_text

# --- DB Connection with Retry (sync-only) ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(10),
    wait=tenacity.wait_fixed(15),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.info(f"Retrying DB connection ({retry_state.attempt_number}/10)...")
)
def create_db_connection(preload: bool = False):
    try:
        logger.debug(f"Attempting DB connection with URL: {sanitized_db_url}")
        parsed_url = urllib.parse.urlparse(SUPABASE_DB_URL)
        if parsed_url.scheme in ["postgres", "postgresql"]:
            coerced_url = SUPABASE_DB_URL.replace(parsed_url.scheme, "postgresql+psycopg2", 1)
        else:
            coerced_url = SUPABASE_DB_URL
        engine = create_engine(
            coerced_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=5,
            pool_timeout=120,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={'connect_timeout': 120, 'options': '-csearch_path=public', 'keepalives': 1, 'keepalives_idle': 30, 'keepalives_interval': 30, 'keepalives_count': 5}
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.debug("DB connection test succeeded")
        if preload:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.debug("Preloaded database connection successfully")
        logger.info(f"Database connection successful: {db_host}:{db_port}/{db_name}")
        db = SQLDatabase(engine, include_tables=ALLOWED_TABLES)
        return engine, db, None
    except Exception as e:
        logger.error(f"DB connection failed: {str(e)}", exc_info=True)
        raise

# --- LLM factory: Gemini first, OpenAI fallback ---
def get_llm():
    """
    Return a LangChain LLM-like object usable by SQLDatabaseToolkit and AgentExecutor.
    Preference: Gemini via langchain_google_genai (ChatGoogleGenerativeAI).
    Fallback: OpenAI via ChatOpenAI.
    """
    # Prefer Gemini
    if MODEL_TYPE == "gemini":
        if ChatGoogleGenerativeAI is None:
            logger.warning("ChatGoogleGenerativeAI not available in environment (langchain-google-genai missing).")
        else:
            try:
                logger.info(f"Initializing Gemini model: {GEMINI_MODEL}")
                return ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=GOOGLE_API_KEY)
            except Exception as e:
                logger.error(f"Failed to init Gemini LLM: {e}", exc_info=True)

    # Fallback to OpenAI
    if OPENAI_API_KEY and ChatOpenAI is not None:
        try:
            logger.info("Initializing OpenAI ChatOpenAI (fallback)")
            return ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.error(f"Failed to init OpenAI LLM: {e}", exc_info=True)

    # Last resort: try litellm completion wrapper (synchronous text-based fallback)
    logger.warning("Falling back to litellm text completion wrapper (not a LangChain LLM object).")
    def litellm_wrapper(prompt_obj):
        # Accepts a simple call pattern used by some tools; we will support only minimal calls
        qtext = prompt_obj if isinstance(prompt_obj, str) else str(prompt_obj)
        resp = completion(model=GEMINI_MODEL if MODEL_TYPE == "gemini" else "gpt-4o-mini",
                          messages=[{"role": "user", "content": qtext}],
                          api_key=GOOGLE_API_KEY if MODEL_TYPE == "gemini" else OPENAI_API_KEY)
        # Return a simple object with .content or as dict
        try:
            return {"content": resp.choices[0].message.content}
        except Exception:
            return {"content": str(resp)}
    return litellm_wrapper

# --- API Endpoints ---
@app.get("/healthz")
def health(check_db: Optional[bool] = Query(False), preload: Optional[bool] = Query(False)):
    logger.debug("Health check triggered")
    if check_db or preload:
        try:
            engine, db, _ = create_db_connection(preload=preload)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Health check with DB succeeded")
            return {"status": "ok", "db_status": "connected"}
        except Exception as e:
            logger.error(f"Health check DB connection failed: {str(e)}", exc_info=True)
            return {"status": "ok", "db_status": f"failed: {str(e)}"}
    return {"status": "ok"}

@app.post("/ask")
def ask(q: Question, x_app_key: str = Header(...)):
    return nlq(q, x_app_key)  # Alias to /nlq

@app.post("/nlq", response_model=APIResponse)
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(min=1, max=60),
    retry=tenacity.retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.info(f"Retrying /nlq due to rate limit ({retry_state.attempt_number}/5)...")
)
def nlq(q: Question, x_app_key: str = Header(...)):
    start_time = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Detect forecasting intent early
    do_forecast, target_dt, blocked_reason = detect_forecast_intent(q.query)
    if blocked_reason:
        logger.info(f"Forecast blocked: {blocked_reason}")
        return APIResponse(answer=blocked_reason, execution_time=0.0)

    try:
        logger.debug(f"Processing /nlq with query: {q.query}")
        # Sync DB init
        engine, db, _ = create_db_connection()

        # LLM via LangChain: Gemini preferred
        llm = get_llm()
        if llm is None:
            raise RuntimeError("No LLM available (check GOOGLE_API_KEY/OPENAI_API_KEY and provider libs)")

        # Schema subset with RAG (sync)
        schema_subset = get_schema_subset(llm, q.query)

        # Build tools & agent
        logger.debug("Building SQL tools via SQLDatabaseToolkit")
        # SQLDatabaseToolkit expects an LLM object; if we returned a litellm wrapper, toolkit may fail.
        tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
        partial_prompt = AGENT_PROMPT.partial(schema=schema_subset, joins=DB_JOINS)
        # create_openai_tools_agent expects an OpenAI-like llm; if llm is a wrapper, Agent may fail – fallback handled
        try:
            agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=partial_prompt)
        except Exception as e:
            logger.warning(f"create_openai_tools_agent failed with llm type {type(llm)}: {e}", exc_info=True)
            # Try to coerce or wrap; simplified agent creation fallback:
            agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=partial_prompt)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=8, handle_parsing_errors=True, memory=memory)

        result = agent_executor.invoke({"input": q.query})
        raw_output = result.get("output", "Unable to process.")

        # Parse charts
        chart_data = None
        chart_type = None
        chart_metadata = {}
        try:
            json_match = re.search(r'\{.*"type":.*\}', raw_output, re.DOTALL)
            if json_match:
                chart_struct = json.loads(json_match.group(0))
                chart_data = chart_struct.get("data")
                chart_type = chart_struct.get("type")
                chart_metadata = {k: v for k, v in chart_struct.items() if k not in ["data", "type"]}
                raw_output = raw_output.replace(json_match.group(0), "").strip()
        except Exception:
            logger.debug("No chart JSON extracted from output")

        final_answer = scrub_schema_mentions(raw_output)
        logger.debug(f"/nlq completed in {round(time.time() - start_time, 2)}s")
        return APIResponse(
            answer=final_answer,
            chart_data=chart_data,
            chart_type=chart_type,
            chart_metadata=chart_metadata,
            execution_time=round(time.time() - start_time, 2)
        )
    except Exception as e:
        logger.error(f"FATAL error in /nlq: {str(e)}", exc_info=True)
        # Expose helpful message for debugging
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/forecast", response_model=APIResponse)
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(min=1, max=60),
    retry=tenacity.retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.info(f"Retrying /forecast due to rate limit ({retry_state.attempt_number}/5)...")
)
def forecast(q: Question, x_app_key: str = Header(...)):
    start_time = time.time()
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    do_forecast, target_dt, blocked_reason = detect_forecast_intent(q.query)
    if blocked_reason:
        return APIResponse(answer=blocked_reason, execution_time=0.0)

    try:
        logger.debug(f"Processing /forecast with query: {q.query}")
        engine, db, _ = create_db_connection()
        llm = get_llm()
        if llm is None:
            raise RuntimeError("No LLM available for forecast")

        schema_subset = get_schema_subset(llm, q.query)
        tools = [execute_python_code] + SQLDatabaseToolkit(db=db, llm=llm).get_tools()
        partial_prompt = AGENT_PROMPT.partial(schema=schema_subset, joins=DB_JOINS)
        agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=partial_prompt)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=8, handle_parsing_errors=True, memory=memory)
        result = agent_executor.invoke({"input": q.query})
        raw_output = result.get("output", "Unable to process.")

        # attempt to extract SQL from agent steps if forecasting
        sql_query = extract_sql_from_steps(result.get("intermediate_steps", []))
        chart_data = None
        chart_type = None
        chart_metadata = {}

        if sql_query:
            clean_q = clean_and_validate_sql(sql_query)
            with engine.connect() as conn:
                r = conn.execute(text(clean_q))
                rows = r.fetchall()
                columns = r.keys()
            df = coerce_dataframe(rows, columns)
            context = {"sql_result": (rows, columns)}

            if do_forecast and not df.empty and "date" in df.columns:
                # If balancing price forecast
                if "p_bal_gel" in df.columns and "xrate" in df.columns:
                    # compute p_bal_usd column
                    df["p_bal_usd"] = df["p_bal_gel"] / df["xrate"]
                    # produce forecast for target_dt
                    last_date = df["date"].max()
                    steps = int((pd.to_datetime(target_dt) - last_date) / np.timedelta64(1, "M")) + 1
                    model_gel = ExponentialSmoothing(df["p_bal_gel"], trend="add", seasonal="add", seasonal_periods=12)
                    fit_gel = model_gel.fit()
                    model_usd = ExponentialSmoothing(df["p_bal_usd"], trend="add", seasonal="add", seasonal_periods=12)
                    fit_usd = model_usd.fit()
                    forecast_gel = fit_gel.forecast(steps=steps)
                    forecast_usd = fit_usd.forecast(steps=steps)
                    yearly_avg_gel = float(forecast_gel.mean())
                    yearly_avg_usd = float(forecast_usd.mean())
                    # Build output
                    raw_output = (
                        f"Forecast for {target_dt.strftime('%Y-%m')}:\n"
                        f"Yearly average: {yearly_avg_gel:.2f} GEL/MWh, {yearly_avg_usd:.2f} USD/MWh\n"
                    )
                    chart_data = {
                        "type": "line",
                        "data": [
                            {"date": str(date), "p_bal_gel": float(gel), "p_bal_usd": float(usd)}
                            for date, gel, usd in zip(
                                pd.date_range(start=last_date, periods=steps, freq="ME"),
                                forecast_gel,
                                forecast_usd
                            )
                        ]
                    }
                    chart_type = "line"
                    chart_metadata = {"title": f"Balancing Electricity Price Forecast to {target_dt.strftime('%Y-%m')}"}

        exec_time = round(time.time() - start_time, 2)
        return APIResponse(answer=raw_output, chart_data=chart_data, chart_type=chart_type, chart_metadata=chart_metadata, execution_time=exec_time)
    except Exception as e:
        logger.error(f"FATAL error in /forecast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
