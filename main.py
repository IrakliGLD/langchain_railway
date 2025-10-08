import os
import time
import logging
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from litellm import completion

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI app ---
app = FastAPI()

# --- Models ---
class Question(BaseModel):
    query: str

class APIResponse(BaseModel):
    answer: str
    execution_time: float

# --- Environment variables ---
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Routes ---
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/ask", response_model=APIResponse)
async def ask(q: Question, x_app_key: str = Header(...)):
    if x_app_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    start = time.time()
    answer = ""
    source = ""

    try:
        # --- Gemini first ---
        if MODEL_TYPE == "gemini" and GOOGLE_API_KEY:
            model = GEMINI_MODEL
            logger.info(f"Trying Gemini model: {model}")
            response = completion(
                model=model,
                messages=[{"role": "user", "content": q.query}],
                api_key=GOOGLE_API_KEY,
            )
            answer = response.choices[0].message.content
            source = "gemini"
        elif OPENAI_API_KEY:
            # --- Fallback to OpenAI ---
            logger.info("Gemini not available. Falling back to OpenAI (gpt-4o-mini)")
            response = completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": q.query}],
                api_key=OPENAI_API_KEY,
            )
            answer = response.choices[0].message.content
            source = "openai"
        else:
            raise HTTPException(status_code=500, detail="No valid API key configured")

    except Exception as e:
        logger.exception(f"Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    exec_time = round(time.time() - start, 2)
    logger.info(f"Response from {source} in {exec_time}s")
    return APIResponse(answer=answer, execution_time=exec_time)
