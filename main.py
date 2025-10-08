# main.py v17.54
# Final version: production-stable. Keeps full logic from v17.53 but fixes missing imports, adds stability in /ask & /nlq, and ensures valid healthz behavior.

import os
import re
import logging
import time
import urllib.parse
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
import tenacity
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from litellm import completion, RateLimitError
from dotenv import load_dotenv  # ✅ Added missing import

# --- Configuration & Setup ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")

load_dotenv()  # ✅ load environment variables

logger.debug(f"Loaded env vars: SUPABASE_DB_URL={os.getenv('SUPABASE_DB_URL')[:10]}..., APP_SECRET_KEY={os.getenv('APP_SECRET_KEY')[:5]}..., MODEL_TYPE={os.getenv('MODEL_TYPE')}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not SUPABASE_DB_URL or not APP_SECRET_KEY:
    logger.error("Missing SUPABASE_DB_URL or APP_SECRET_KEY")
    raise RuntimeError("SUPABASE_DB_URL and APP_SECRET_KEY are required.")

if MODEL_TYPE == "gemini" and not GOOGLE_API_KEY:
    logger.error("Missing GOOGLE_API_KEY for MODEL_TYPE=gemini")
    raise RuntimeError("GOOGLE_API_KEY required for MODEL_TYPE=gemini.")

if OPENAI_API_KEY:
    logger.info("OpenAI key present for fallback.")


# --- Validate SUPABASE_DB_URL ---
def validate_supabase_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ["postgres", "postgresql", "postgresql+psycopg"]:
            raise ValueError("Scheme must be 'postgres', 'postgresql', or 'postgresql+psycopg'")
        if not parsed.username or not parsed.password:
            raise ValueError("Username and password must be provided")
        parsed_password = parsed.password.strip()
        if not parsed_password:
            raise ValueError("Password cannot be empty after trimming")
        if not re.match(r'^[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?~]*$', parsed_password):
            raise ValueError("Password contains invalid characters for URL")
        if parsed.hostname != "aws-1-eu-central-1.pooler.supabase.com":
            raise ValueError("Host must be 'aws-1-eu-central-1.pooler.supabase.com'")
        if parsed.port != 6543:
            raise ValueError("Port must be 6543 for pooled connection")
        if parsed.path != "/postgres":
            raise ValueError("Database path must be '/postgres'")
        if parsed.username != "postgres.qvmqmmcglqmhachqaezt":
            raise ValueError("Username must be 'postgres.qvmqmmcglqmhachqaezt'")
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("sslmode") != ["require"]:
            raise ValueError("Query parameter 'sslmode=require' is required")
    except Exception as e:
        logger.error(f"Invalid SUPABASE_DB_URL: {str(e)}", exc_info=True)
        raise RuntimeError(f"Invalid SUPABASE_DB_URL: {str(e)}")

validate_supabase_url(SUPABASE_DB_URL)

sanitized_db_url = re.sub(r':[^@]+@', ':****@', SUPABASE_DB_URL)
logger.info(f"Using SUPABASE_DB_URL: {sanitized_db_url}")

parsed_db_url = urllib.parse.urlparse(SUPABASE_DB_URL)
db_host = parsed_db_url.hostname
db_port = parsed_db_url.port
db_name = parsed_db_url.path.lstrip('/')
logger.info(f"DB connection details: host={db_host}, port={db_port}, dbname={db_name}")

# --- FastAPI App ---
app = FastAPI(title="EnerBot Backend", version="17.54")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Models ---
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


# --- DB Connection ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(10),
    wait=tenacity.wait_fixed(15),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda rs: logger.info(f"Retrying DB connection ({rs.attempt_number}/10)...")
)
def create_db_connection(preload: bool = False):
    try:
        parsed_url = urllib.parse.urlparse(SUPABASE_DB_URL)
        coerced_url = SUPABASE_DB_URL.replace(parsed_url.scheme, "postgresql+psycopg2", 1)
        engine = create_engine(
            coerced_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=2,
            pool_timeout=120,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={
                'connect_timeout': 120,
                'options': '-csearch_path=public',
                'keepalives': 1,
                'keepalives_idle': 30,
                'keepalives_interval': 30,
                'keepalives_count': 5
            }
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        if preload:
            time.sleep(15)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        logger.info(f"Database connection successful: {db_host}:{db_port}/{db_name}")
        return engine, None, None
    except Exception as e:
        logger.error(f"DB connection failed: {str(e)}", exc_info=True)
        raise


# --- Routes ---
@app.get("/healthz")
def health(check_db: str = Query("false"), preload: str = Query("false")):
    logger.debug(f"Health check with check_db={check_db}, preload={preload}")
    db_status = "not checked"
    if check_db.lower() == "true" or preload.lower() == "true":
        try:
            engine, _, _ = create_db_connection(preload=preload.lower() == "true")
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception as e:
            logger.error(f"Health check DB failed: {str(e)}", exc_info=True)
            db_status = f"failed: {str(e)}"
    return {"status": "ok", "db_status": db_status}


# --- NLQ endpoint ---
@app.post("/nlq", response_model=APIResponse)
async def nlq(q: Question, x_app_key: str = Header(...)):
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    start = time.time()
    model_name = None
    api_key = None
    # Try Gemini first
    try:
        model_name = GEMINI_MODEL
        api_key = GOOGLE_API_KEY
        logger.info(f"Attempting Gemini: model={model_name}, api_key set={bool(api_key)}")
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": q.query}],
            api_key=api_key
        )
        answer = response.choices[0].message.content
        source = "gemini"
    except Exception as e_gemini:
        logger.error(f"Gemini call failed: {e_gemini}", exc_info=True)
        # fallback to OpenAI
        try:
            model_name = "gpt-4o-mini"
            api_key = OPENAI_API_KEY
            logger.info(f"Falling back to OpenAI: model={model_name}, api_key set={bool(api_key)}")
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": q.query}],
                api_key=api_key
            )
            answer = response.choices[0].message.content
            source = "openai"
        except Exception as e_openai:
            logger.error(f"OpenAI fallback failed: {e_openai}", exc_info=True)
            # Expose Gemini error or OpenAI error
            raise HTTPException(status_code=500, detail=f"Gemini error: {str(e_gemini)}; OpenAI error: {str(e_openai)}")
    exec_time = round(time.time() - start, 2)
    return APIResponse(answer=answer, execution_time=exec_time)



# --- ASK endpoint (wrapper for /nlq) ---
@app.post("/ask", response_model=APIResponse)
async def ask(q: Question, x_app_key: str = Header(...)):
    """Wrapper for /nlq."""
    try:
        return await nlq(q, x_app_key)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"/ask endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
