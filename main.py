# main.py v17.60
# Lazy-loaded model + DB for Railway stability (fixes Healthcheck timeout)
# Otherwise identical to v17.59

import os
import json
import traceback
import urllib.parse
from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# -----------------------------
# FASTAPI SETUP
# -----------------------------
app = FastAPI()

# Allow all origins (for frontend/Supabase Edge)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# GLOBAL VARIABLES (lazy)
# -----------------------------
engine = None
model = None

MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -----------------------------
# Request Schema
# -----------------------------
class ChatRequest(BaseModel):
    query: str
    service_tier: str = "user"

# -----------------------------
# Lazy Model Initialization
# -----------------------------
def get_model():
    global model
    if model is not None:
        return model

    try:
        if MODEL_TYPE == "gemini":
            print(f"🟢 Using Gemini model (langchain-google-genai): {GEMINI_MODEL}")
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
        else:
            print(f"🟢 Using OpenAI model: {OPENAI_MODEL}")
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model=OPENAI_MODEL)
        return model
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        raise

# -----------------------------
# Lazy DB Initialization
# -----------------------------
def get_engine():
    global engine
    if engine is not None:
        return engine

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

        # Test connection once
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ DB connection established successfully")
        return engine
    except Exception as e:
        print(f"❌ DB connection failed: {e}")
        raise

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/healthz")
async def healthz():
    """Instant response for Railway healthcheck"""
    return {"status": "ok"}

@app.post("/ask")
async def ask(request: Request, x_app_key: str = Header(...)):
    """Main query handler with DB + LLM logic"""
    try:
        data = await request.json()
        print("📩 Incoming data:", data)
        query = data.get("query", "").strip()
        service_tier = data.get("service_tier", "user")
        if not query:
            return {"error": "Missing 'query' in request body"}
        if x_app_key != os.getenv("APP_SECRET_KEY"):
            return {"error": "Unauthorized"}

        # Initialize dependencies lazily
        db = get_engine()
        model = get_model()

        # Define system prompt
        system_prompt = (
            "You are EnerBot, an assistant specialized in the Georgian electricity and gas market."
            if service_tier == "admin"
            else "You are EnerBot, an informative chatbot about the Georgian energy sector."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # SQL select based on query (basic pattern)
        sql_query = (
            f"SELECT * FROM price WHERE date LIKE '%{query.split()[-1]}%'" 
            if "price" in query.lower() 
            else "SELECT 1"
        )

        with db.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            columns = result.keys()
            if rows:
                # Take first row only for brevity
                result_text = json.dumps(
                    {columns[i]: row[i] for i, _ in enumerate(columns) for row in rows[:1]}
                )
            else:
                result_text = "No data found for your query."

        # Combine SQL result + model reasoning
        response = model.invoke(messages + [{"role": "assistant", "content": result_text}])

        if hasattr(response, "content"):
            result_text = response.content
        elif isinstance(response, dict) and "content" in response:
            result_text = response["content"]
        else:
            result_text = str(response)

        print("✅ Model output:", result_text[:300])
        return {"answer": result_text}

    except Exception as e:
        print("❌ Internal server error:", str(e))
        traceback.print_exc()
        return {"error": f"Internal server error: {str(e)}"}

@app.get("/")
async def home():
    return {"message": "EnerBot backend is running successfully 🚀"}
