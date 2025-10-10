# main.py v17.61
# Same as previous working version + GET-friendly /ask route

import os
import json
import traceback
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import urllib.parse

# -----------------------------
# FastAPI Initialization
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Database Connection
# -----------------------------
try:
    SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
    if not SUPABASE_DB_URL:
        raise ValueError("SUPABASE_DB_URL not set")

    parsed_url = urllib.parse.urlparse(SUPABASE_DB_URL)
    if parsed_url.scheme in ["postgres", "postgresql"]:
        coerced_url = SUPABASE_DB_URL.replace(parsed_url.scheme, "postgresql+psycopg2", 1)
    else:
        coerced_url = SUPABASE_DB_URL

    engine = create_engine(
        coerced_url,
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
        print("✅ Initial DB connection established")
except Exception as e:
    print(f"❌ Initial DB connection failed: {e}")
    raise

# -----------------------------
# Models and Schema
# -----------------------------
class ChatRequest(BaseModel):
    query: str
    service_tier: str = "user"

# -----------------------------
# Routes
# -----------------------------
@app.get("/healthz")
async def healthz(check_db: bool = False):
    """Health check route (with optional DB check)."""
    if check_db:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "ok", "db": "connected"}
        except Exception as e:
            return {"status": "ok", "db_error": str(e)}
    return {"status": "ok"}

@app.get("/ask")
async def ask_get():
    """Browser-friendly GET endpoint to prevent 404 confusion."""
    return {
        "message": "✅ /ask endpoint is active. Please send POST requests with JSON data, e.g. {'query': 'Hello Enerbot!'}"
    }

@app.post("/ask")
async def ask(request: Request, x_app_key: str = Header(default=None)):
    """Main POST endpoint for chat queries."""
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        service_tier = data.get("service_tier", "user")

        if not query:
            return {"error": "Missing 'query' in request body"}

        print(f"📩 Received query: {query}")

        # --- Example SQL select (for demonstration)
        sql_query = f"SELECT * FROM price LIMIT 1" if "price" in query.lower() else "SELECT 1"
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            if rows:
                result_text = str(rows[0])
            else:
                result_text = "No data found."

        return {"answer": f"Echoing query: '{query}' | DB says: {result_text}"}

    except Exception as e:
        print("❌ Internal error:", e)
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/")
async def home():
    return {"message": "EnerBot backend is running successfully 🚀"}

# -----------------------------
# Local Dev Entry Point
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
