from fastapi import FastAPI
from sqlalchemy import create_engine, text
import os

app = FastAPI()

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/dbtest")
async def dbtest():
    try:
        db_url = os.getenv("SUPABASE_DB_URL")
        if not db_url:
            return {"error": "SUPABASE_DB_URL not set"}

        if "sslmode" not in db_url:
            db_url += "?sslmode=require"

        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT NOW()"))
            time = result.scalar()
        return {"db_connection": "ok", "time": str(time)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
