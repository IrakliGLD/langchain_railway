# main.py — Railway DB Connectivity Test
import os
import urllib.parse
from fastapi import FastAPI
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

app = FastAPI()

@app.get("/healthz")
async def healthz():
    """Basic healthcheck"""
    return {"status": "ok"}

@app.get("/testdb")
async def testdb():
    """Try to connect to Supabase and run SELECT 1"""
    try:
        SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
        if not SUPABASE_DB_URL:
            return {"error": "SUPABASE_DB_URL not found in env"}

        parsed_url = urllib.parse.urlparse(SUPABASE_DB_URL)
        if parsed_url.scheme in ["postgres", "postgresql"]:
            coerced_url = SUPABASE_DB_URL.replace(parsed_url.scheme, "postgresql+psycopg2", 1)
        else:
            coerced_url = SUPABASE_DB_URL

        engine = create_engine(
            coerced_url,
            poolclass=QueuePool,
            pool_size=2,
            max_overflow=1,
            pool_timeout=10,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return {"result": "✅ Database connection successful!"}
    except Exception as e:
        return {"error": f"❌ DB connection failed: {str(e)}"}
