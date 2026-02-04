from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from .matcher import Matcher


app = FastAPI(title="Medical Cost Estimator API")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
matcher = Matcher(PROJECT_ROOT)


FRONTEND_DIR = PROJECT_ROOT / "backend" / "frontend"
app.mount("/home", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

@app.get("/health")
def health():
    return {"status": "ok"}

class MatchRequest(BaseModel):
    text: str
    top_k: int = 10

@app.post("/match")
def match(req: MatchRequest):
    return {"input": req.text, "candidates": matcher.search(req.text, req.top_k)}
