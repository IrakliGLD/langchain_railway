# main.py — Database connection test version

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

app = FastAPI()

# Allow CORS (for external testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold DB connection status
DB_STATUS = "Not checked yet"


@app.on_event("startup")
async def startup_event():
    """Try connecting to Supabase DB at startup"""
    global DB_STATUS
    try:
        db_url = os.getenv("SUPABASE_DB_URL")
        if not db_url:
            DB_STATUS = "❌ SUPABASE_DB_URL not set"
            print(DB_STATUS)
            return

        # Create SQLAlchemy engine
        engine = create_engine(
            db_url.replace("postgres://", "postgresql+psycopg2://"),
            poolclass=QueuePool,
            pool_size=3,
            max_overflow=1,
            connect_args={"connect_timeout": 10},
        )

        # Test query
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            print(f"✅ DB connection OK, test result: {result}")
            DB_STATUS = "✅ DB connection successful"
    except Exception as e:
        DB_STATUS = f"❌ DB connection failed: {e}"
        print(DB_STATUS)


@app.get("/healthz")
async def healthz():
    """Health check for Railway"""
    return {"status": "ok", "db": DB_STATUS}


@app.get("/")
async def home():
    """Simple root endpoint"""
    return {"message": "DB connection test service running", "db_status": DB_STATUS}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
