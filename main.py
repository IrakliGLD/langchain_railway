from fastapi import FastAPI

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting app on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
