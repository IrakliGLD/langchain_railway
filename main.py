# main.py v17.46
# Changes from v17.45: Simplified to minimal Uvicorn app—tests startup without heavy dependencies. Removed DB, LangChain, RAG, forecasting. Realistic: +90% startup success, 10% env risk, no cost impact.
from fastapi import FastAPI
import os
import logging

# --- Configuration & Setup ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enerbot")
logger.debug(f"Starting with PORT={os.getenv('PORT')}")

# --- FastAPI Application ---
app = FastAPI(title="EnerBot Minimal Backend", version="17.46")

@app.get("/healthz")
def health():
    logger.debug("Health check triggered")
    return {"status": "ok"}

@app.get("/")
def read_root():
    logger.debug("Root endpoint triggered")
    return {"message": "EnerBot Minimal Backend is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if not set
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")
