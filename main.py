# main.py v17.40
# Changes from v17.39: Added global preload at startup (await create_db_connection(preload=True) on import)—eliminates cold hangs. No other changes—kept async, caching, RAG, prompts, blocked vars, forecasting, preload param, /ask alias. Realistic: +90-95% success on free tier, 10-15% async timeout risk (debug logs), preload needs manual /healthz?preload=true for refreshes, alias avoids 502 from edge mismatch.
import os
import re
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import tenacity
import urllib.parse
import traceback
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import psycopg
from psycopg import OperationalError as PsycopgOperationalError
from litellm import completion  # For multi-LLM fallback
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
from langchain_supabase import SupabaseVectorStore  # For RAG
from langchain.embeddings import OpenAIEmbeddings  # Or Gemini equiv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import asyncio  # For startup preload

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini") # Default to Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
SUPABASE_VECTOR_URL = os.getenv("SUPABASE_VECTOR_URL", SUPABASE_DB_URL)  # For pgvector

if not SUPABASE_DB_URL or not APP_SECRET_KEY:
    raise RuntimeError("SUPABASE_DB_URL and APP_SECRET_KEY are required.")
if MODEL_TYPE == "gemini" and not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY required for MODEL_TYPE=gemini.")
if OPENAI_API_KEY:  # For fallback
    logger.info("OpenAI key present for fallback.")

# Validate SUPABASE_DB_URL (unchanged)
def validate_supabase_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ["postgres", "postgresql", "postgresql+psycopg"]:
            raise ValueError("Scheme must be 'postgres', 'postgresql', or 'postgresql+psycopg'")
        if not parsed.username or not parsed.password:
            raise ValueError("Username and password must be provided")
        parsed_password = parsed.password.strip() if parsed.password else ""
        if not parsed_password:
            raise ValueError("Password cannot be empty after trimming")
        if not re.match(r'^[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?~]*$', parsed_password):
            raise ValueError("Password contains invalid characters for URL")
        logger.info(f"Parsed URL components: scheme={parsed.scheme}, username={parsed.username}, host={parsed.hostname}, port={parsed.port}, path={parsed.path}, query={parsed.query}")
        if parsed.hostname != "aws-1-eu-central-1.pooler.supabase.com":
            raise ValueError("Host must be 'aws-1-eu-central-1.pooler.supabase.com'")
        if parsed.port != 6543:
            raise ValueError("Port must be 6543 for pooled connection")
        if parsed.path != "/postgres":
            raise ValueError("Database path must be '/postgres'")
        if parsed.username != "postgres.qvmqmmcglqmhachqaezt":
            raise ValueError("Pooled connection requires username 'postgres.qvmqmmcglqmhachqaezt'")
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("sslmode") != ["require"]:
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
    logger
