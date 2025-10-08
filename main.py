import os
import socket
import time
import traceback
from fastapi import FastAPI
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import urlparse

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/testdb")
def test_db():
    """Diagnose Supabase / Railway DB connection."""
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")

    if not db_url:
        return {"error": "No SUPABASE_DB_URL or DATABASE_URL found in environment variables"}

    result = {"raw_url": db_url, "steps": []}
    try:
        parsed = urlparse(db_url)
        host, port, user = parsed.hostname, parsed.port, parsed.username
        result["parsed"] = {"host": host, "port": port, "user": user}

        # Step 1. DNS resolution
        try:
            ip = socket.gethostbyname(host)
            result["steps"].append(f"✅ DNS resolution success: {host} -> {ip}")
        except Exception as e:
            result["steps"].append(f"❌ DNS resolution failed: {repr(e)}")
            return result

        # Step 2. Connection attempt with retries
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5, "sslmode": "require"},
        )

        connected = False
        for attempt in range(5):
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT NOW()"))
                connected = True
                result["steps"].append(f"✅ Connection success on attempt {attempt + 1}")
                break
            except SQLAlchemyError as e:
                err_msg = str(e.__cause__) if e.__cause__ else str(e)
                result["steps"].append(f"Attempt {attempt + 1} failed: {err_msg}")
                time.sleep(2)

        if not connected:
            result["steps"].append("❌ Could not connect after 5 attempts")

        return result

    except Exception as e:
        tb = traceback.format_exc()
        result["error"] = str(e)
        result["traceback"] = tb
        return result
