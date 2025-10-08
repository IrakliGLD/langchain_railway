import os
import json
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# -----------------------------
# MODEL INITIALIZATION
# -----------------------------
from langchain.chat_models import init_chat_model

# --- Try Gemini first (default) ---
MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Initialize FastAPI
app = FastAPI()

# Allow all origins (for frontend use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Define request schema
# -----------------------------
class ChatRequest(BaseModel):
    query: str
    service_tier: str = "user"


# -----------------------------
# Initialize model
# -----------------------------
def get_model():
    try:
        if MODEL_TYPE == "gemini":
            print(f"🟢 Using Gemini model: {GEMINI_MODEL}")
            model = init_chat_model(GEMINI_MODEL, model_type="gemini")
        else:
            print(f"🟢 Using OpenAI model: {OPENAI_MODEL}")
            model = init_chat_model(OPENAI_MODEL, model_type="openai")
        return model
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        raise


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/healthz")
async def healthz():
    """Health check route for Railway"""
    return {"status": "ok"}


@app.post("/ask")
async def ask(request: Request):
    """Handle incoming queries from Supabase Edge Function"""
    try:
        data = await request.json()
        print("📩 Incoming data:", data)

        query = data.get("query", "").strip()
        service_tier = data.get("service_tier", "user")

        if not query:
            return {"error": "Missing 'query' in request body"}

        model = get_model()

        # Construct the system message depending on the user tier
        system_prompt = (
            "You are EnerBot, an assistant specialized in the Georgian electricity and gas market."
            if service_tier == "admin"
            else "You are EnerBot, an informative chatbot about the Georgian energy sector."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = model.invoke(messages)

        # Ensure compatibility with LangChain’s response structure
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
